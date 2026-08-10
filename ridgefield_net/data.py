from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch


def voxelize_events(coords_xyz, polarities, grid_size_xyz):
    """将 [0,1)^3 事件聚合为 [t,y,x] 稀疏体素及局部特征。"""
    x_size, y_size, t_size = [int(x) for x in grid_size_xyz]
    coords = np.clip(coords_xyz, 0.0, 1.0 - 1e-7)
    ix = np.minimum((coords[:, 0] * x_size).astype(np.int64), x_size - 1)
    iy = np.minimum((coords[:, 1] * y_size).astype(np.int64), y_size - 1)
    it = np.minimum((coords[:, 2] * t_size).astype(np.int64), t_size - 1)
    tyx = np.stack([it, iy, ix], axis=1)
    unique, inverse = np.unique(tyx, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(np.float32)
    pos = np.bincount(inverse, weights=(polarities > 0.5).astype(np.float32), minlength=len(count))
    neg = count - pos
    sum_t = np.bincount(inverse, weights=coords[:, 2], minlength=len(count)).astype(np.float32)
    sum_t2 = np.bincount(inverse, weights=coords[:, 2] ** 2, minlength=len(count)).astype(np.float32)
    latest = np.zeros(len(count), dtype=np.float32)
    np.maximum.at(latest, inverse, coords[:, 2].astype(np.float32))
    mean_t = sum_t / np.maximum(count, 1.0)
    spread = np.sqrt(np.maximum(sum_t2 / np.maximum(count, 1.0) - mean_t ** 2, 0.0))
    frac_pos = pos / np.maximum(count, 1.0)
    balance = (pos - neg) / np.maximum(count, 1.0)
    # 特征包含强度、极性和时间统计，强度仅作温和的 log 压缩。
    feats = np.stack([
        np.log1p(count), frac_pos, balance, mean_t, latest, spread,
        np.clip(count / 16.0, 0.0, 4.0),
    ], axis=1).astype(np.float32)
    return unique.astype(np.int32), feats


def _subsample(coords, raw, polarity, labels, inst, max_events, rng):
    if max_events <= 0 or len(coords) <= max_events:
        return coords, raw, polarity, labels, inst
    pos = np.flatnonzero(labels > 0.5)
    neg = np.flatnonzero(labels <= 0.5)
    if len(pos) >= max_events:
        keep = rng.choice(pos, size=max_events, replace=False)
    else:
        n_neg = min(len(neg), max_events - len(pos))
        keep = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
    keep.sort()
    return coords[keep], raw[keep], polarity[keep], labels[keep], inst[keep]


def _event_queries(labels, num_queries, pos_fraction, rng):
    n = int(num_queries)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    pos = np.flatnonzero(labels > 0.5)
    neg = np.flatnonzero(labels <= 0.5)
    np_pos = int(round(n * pos_fraction))
    np_neg = n - np_pos
    parts = []
    if len(pos):
        parts.append(rng.choice(pos, size=np_pos, replace=len(pos) < np_pos))
    if len(neg):
        parts.append(rng.choice(neg, size=np_neg, replace=len(neg) < np_neg))
    selected = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    if len(selected) < n:
        all_idx = np.arange(len(labels))
        selected = np.concatenate([selected, rng.choice(all_idx, size=n - len(selected), replace=True)])
    rng.shuffle(selected)
    return selected[:n].astype(np.int64)


def _sample_anchors(pos_coords, n, rng):
    if n <= 0 or len(pos_coords) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return pos_coords[rng.choice(len(pos_coords), size=int(n), replace=len(pos_coords) < n)].astype(np.float32)


def _sample_background(pos_coords, n, grid_xyz, exclusion_xyz, rng):
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    x_size, y_size, t_size = [int(x) for x in grid_xyz]
    if len(pos_coords) == 0:
        return rng.random((n, 3), dtype=np.float32)
    coords = np.clip(pos_coords, 0.0, 1.0 - 1e-7)
    ix = np.minimum((coords[:, 0] * x_size).astype(np.int64), x_size - 1)
    iy = np.minimum((coords[:, 1] * y_size).astype(np.int64), y_size - 1)
    it = np.minimum((coords[:, 2] * t_size).astype(np.int64), t_size - 1)
    occupied = np.zeros((t_size, y_size, x_size), dtype=bool)
    occupied[it, iy, ix] = True
    rx = max(1, int(np.ceil(float(exclusion_xyz[0]) * x_size)))
    ry = max(1, int(np.ceil(float(exclusion_xyz[1]) * y_size)))
    rt = max(1, int(np.ceil(float(exclusion_xyz[2]) * t_size)))
    # 用切片 OR 实现固定的 l_inf 邻域排除，避免 scipy 对大结构元素的
    # brute-force dilation 在 DDP 首批数据上占满 CPU。
    forbidden = np.zeros_like(occupied)
    for dt in range(-rt, rt + 1):
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                src_t0, src_t1 = max(0, -dt), min(t_size, t_size - dt)
                src_y0, src_y1 = max(0, -dy), min(y_size, y_size - dy)
                src_x0, src_x1 = max(0, -dx), min(x_size, x_size - dx)
                dst_t0, dst_t1 = max(0, dt), min(t_size, t_size + dt)
                dst_y0, dst_y1 = max(0, dy), min(y_size, y_size + dy)
                dst_x0, dst_x1 = max(0, dx), min(x_size, x_size + dx)
                forbidden[dst_t0:dst_t1, dst_y0:dst_y1, dst_x0:dst_x1] |= occupied[src_t0:src_t1, src_y0:src_y1, src_x0:src_x1]
    free = np.argwhere(~forbidden)
    if len(free) == 0:
        return rng.random((n, 3), dtype=np.float32)
    chosen = free[rng.choice(len(free), size=n, replace=len(free) < n)]
    jitter = rng.random((n, 3), dtype=np.float32)
    return np.stack([
        (chosen[:, 2] + jitter[:, 0]) / x_size,
        (chosen[:, 1] + jitter[:, 1]) / y_size,
        (chosen[:, 0] + jitter[:, 2]) / t_size,
    ], axis=1).astype(np.float32)


class EVUAVDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.data.root) / cfg.data.splits[split]
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"没有找到数据：{self.root}")
        self.grid_xyz = tuple(int(x) for x in cfg.data.voxel_resolution)
        self.sensor_w, self.sensor_h = [int(x) for x in cfg.data.sensor_size]
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            evs = data["evs_norm"].astype(np.float32)
            raw = data["ev_loc"].astype(np.float32)
        coords = np.clip(evs[:, :3], 0.0, 1.0 - 1e-7)
        polarity = evs[:, 3].astype(np.float32)
        labels = evs[:, 4].astype(np.float32)
        inst = evs[:, 5].astype(np.int64)
        seed = int(self.cfg.train.seed) + index * 1009 + self.epoch * 100003
        rng = np.random.default_rng(seed)
        if self.split == "train":
            coords, raw, polarity, labels, inst = _subsample(
                coords, raw, polarity, labels, inst, int(self.cfg.data.max_input_events), rng
            )
        voxel_coords, voxel_feats = voxelize_events(coords, polarity, self.grid_xyz)
        item = {
            "sample_name": path.stem,
            "coords_xyz": coords.astype(np.float32),
            "coords_raw": raw.astype(np.float32),
            "polarity": polarity.astype(np.float32),
            "labels": labels.astype(np.float32),
            "inst_ids": inst.astype(np.int64),
            "voxel_coords": voxel_coords,
            "voxel_feats": voxel_feats,
        }
        if self.split == "train":
            pos_coords = coords[labels > 0.5]
            qidx = _event_queries(labels, int(self.cfg.train.event_queries), float(self.cfg.train.event_pos_fraction), rng)
            item["event_query_coords"] = coords[qidx]
            item["event_query_labels"] = labels[qidx]
            item["anchors"] = _sample_anchors(pos_coords, int(self.cfg.train.psc_patches), rng)
            item["bg_queries"] = _sample_background(
                pos_coords, int(self.cfg.train.background_queries),
                tuple(int(x) for x in self.cfg.train.background_sampling_grid),
                tuple(float(x) for x in self.cfg.train.background_exclusion_radius), rng,
            )
        return item


def evuav_collate(batch):
    out = {"sample_names": [x["sample_name"] for x in batch]}
    voxel_coords, voxel_feats = [], []
    arrays = {key: [] for key in ["coords_xyz", "coords_raw", "labels", "inst_ids"]}
    event_batch, query_coords, query_labels, query_batch = [], [], [], []
    anchors, anchor_batch, bg_queries, bg_batch = [], [], [], []
    for batch_id, item in enumerate(batch):
        vc = torch.from_numpy(item["voxel_coords"]).long()
        voxel_coords.append(torch.cat([torch.full((len(vc), 1), batch_id, dtype=torch.long), vc], dim=1))
        voxel_feats.append(torch.from_numpy(item["voxel_feats"]).float())
        for key in arrays:
            arrays[key].append(torch.from_numpy(item[key]))
        event_batch.append(torch.full((len(item["coords_xyz"]),), batch_id, dtype=torch.long))
        if "event_query_coords" in item:
            query_coords.append(torch.from_numpy(item["event_query_coords"]).float())
            query_labels.append(torch.from_numpy(item["event_query_labels"]).float())
            query_batch.append(torch.full((len(item["event_query_coords"]),), batch_id, dtype=torch.long))
            anchors.append(torch.from_numpy(item["anchors"]).float())
            anchor_batch.append(torch.full((len(item["anchors"]),), batch_id, dtype=torch.long))
            bg_queries.append(torch.from_numpy(item["bg_queries"]).float())
            bg_batch.append(torch.full((len(item["bg_queries"]),), batch_id, dtype=torch.long))
    for key, values in arrays.items():
        out[key] = torch.cat(values, dim=0)
    out["voxel_coords"] = torch.cat(voxel_coords, dim=0)
    out["voxel_feats"] = torch.cat(voxel_feats, dim=0)
    out["event_batch_ids"] = torch.cat(event_batch, dim=0)
    for key, values, shape, dtype in [
        ("event_query_coords", query_coords, (0, 3), torch.float32),
        ("event_query_labels", query_labels, (0,), torch.float32),
        ("event_query_batch_ids", query_batch, (0,), torch.long),
        ("anchors", anchors, (0, 3), torch.float32),
        ("anchor_batch_ids", anchor_batch, (0,), torch.long),
        ("bg_queries", bg_queries, (0, 3), torch.float32),
        ("bg_batch_ids", bg_batch, (0,), torch.long),
    ]:
        out[key] = torch.cat(values, dim=0) if values else torch.zeros(shape, dtype=dtype)
    return out
