import math

import numpy as np
import scipy.ndimage as ndi
import torch

from .losses import soft_skeletonize_3d


def rasterize(coords_xyz, grid_xyz, dilate_steps=0):
    x_size, y_size, t_size = [int(x) for x in grid_xyz]
    grid = np.zeros((t_size, y_size, x_size), dtype=np.uint8)
    if len(coords_xyz):
        c = np.clip(coords_xyz, 0.0, 1.0 - 1e-7)
        ix = np.minimum((c[:, 0] * x_size).astype(np.int64), x_size - 1)
        iy = np.minimum((c[:, 1] * y_size).astype(np.int64), y_size - 1)
        it = np.minimum((c[:, 2] * t_size).astype(np.int64), t_size - 1)
        grid[it, iy, ix] = 1
    if dilate_steps:
        grid = ndi.binary_dilation(grid, structure=np.ones((3, 3, 3), bool), iterations=int(dilate_steps)).astype(np.uint8)
    return grid


def structure_metrics(coords_xyz, pred, target, cfg):
    """按补充材料：rasterize → 一次膨胀 → 10 次 soft skeleton。"""
    grid_xyz = tuple(int(x) for x in cfg.eval.metric_grid)
    pred_grid = rasterize(coords_xyz[pred], grid_xyz, int(cfg.eval.metric_dilation_steps))
    gt_grid = rasterize(coords_xyz[target], grid_xyz, int(cfg.eval.metric_dilation_steps))
    pt = torch.from_numpy(pred_grid[None, None].astype(np.float32))
    gt = torch.from_numpy(gt_grid[None, None].astype(np.float32))
    with torch.no_grad():
        sp = soft_skeletonize_3d(pt, int(cfg.eval.cldice_skeleton_iterations))
        sg = soft_skeletonize_3d(gt, int(cfg.eval.cldice_skeleton_iterations))
        tprec = float((sp * gt).sum() / (sp.sum() + 1e-6))
        tsens = float((sg * pt).sum() / (sg.sum() + 1e-6))
        cldice = 2.0 * tprec * tsens / (tprec + tsens + 1e-6)
        pred_skel = (sp[0, 0].numpy() >= float(cfg.eval.binary_skeleton_threshold)).astype(np.uint8)
        gt_skel = (sg[0, 0].numpy() >= float(cfg.eval.binary_skeleton_threshold)).astype(np.uint8)
    conn = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity，固定记录在 README
    pred_labels, pred_count = ndi.label(pred_skel, structure=conn)
    gt_labels, gt_count = ndi.label(gt_skel, structure=conn)
    if gt_count == 0:
        frag = 0.0
    else:
        values = []
        for component in range(1, gt_count + 1):
            dilated = ndi.binary_dilation(gt_labels == component, structure=conn, iterations=1)
            ids = np.unique(pred_labels[dilated])
            n_parts = int((ids > 0).sum())
            values.append(max(n_parts - 1, 0))
        frag = float(np.mean(values))
    return float(cldice), float(frag)


def detection_counts(pred, labels, inst_ids, raw_coords, sensor_size, window_ms, slice_ms, eta):
    """按 50 ms 不重叠切片计算 Pd/Fa 的分子和分母。"""
    width, height = [int(x) for x in sensor_size]
    ts = raw_coords[:, 2].astype(np.float64)
    if len(ts) == 0:
        return np.zeros(5, dtype=np.float64)
    slice_ms = float(slice_ms)
    n_frames = int(math.ceil(float(window_ms) / slice_ms))
    obj = correct = false = 0
    for frame in range(n_frames):
        mask = (ts >= frame * slice_ms) & (ts < (frame + 1) * slice_ms)
        if not mask.any():
            continue
        frame_pred, frame_gt = pred[mask], labels[mask] > 0.5
        frame_inst = inst_ids[mask]
        for instance in np.unique(frame_inst):
            if instance == 0:
                continue
            instance_mask = frame_inst == instance
            gt_pos = frame_gt[instance_mask]
            if not gt_pos.any():
                continue
            obj += 1
            covered = np.logical_and(frame_pred[instance_mask], gt_pos).sum() / max(int(gt_pos.sum()), 1)
            correct += int(covered >= float(eta))
        false_map = np.zeros((height, width), dtype=np.uint8)
        fp = np.logical_and(frame_pred, ~frame_gt)
        xs = np.clip(raw_coords[mask, 0].astype(np.int64)[fp], 0, width - 1)
        ys = np.clip(raw_coords[mask, 1].astype(np.int64)[fp], 0, height - 1)
        false_map[ys, xs] = 1
        _, components = ndi.label(false_map, structure=np.ones((3, 3), dtype=np.uint8))
        false += int(components)
    return np.array([correct, obj, false, n_frames, 0.0], dtype=np.float64)


class MetricAccumulator:
    def __init__(self, cfg):
        self.cfg = cfg
        # TP FP FN TN, clDice sum, Frag sum, sample count, Pd correct, Pd all, Fa components, frame count
        self.values = np.zeros(11, dtype=np.float64)

    def update(self, probs, labels, coords_xyz, raw_coords, inst_ids):
        pred = probs >= float(self.cfg.eval.threshold)
        gt = labels > 0.5
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()
        tn = np.logical_and(~pred, ~gt).sum()
        cldice, frag = structure_metrics(coords_xyz, pred, gt, self.cfg)
        det = detection_counts(
            pred, labels, inst_ids, raw_coords, self.cfg.data.sensor_size,
            self.cfg.data.window_ms, self.cfg.eval.pd_slice_ms, self.cfg.eval.pd_eta,
        )
        self.values += np.array([tp, fp, fn, tn, cldice, frag, 1, det[0], det[1], det[2], det[3]], dtype=np.float64)

    def merge_distributed(self, device):
        tensor = torch.as_tensor(self.values, dtype=torch.float64, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        self.values = tensor.cpu().numpy()

    def summary(self):
        tp, fp, fn, tn, cl_sum, frag_sum, samples, pd_correct, pd_all, false_num, frames = self.values
        iou = tp / max(tp + fp + fn, 1.0)
        dice = 2 * tp / max(2 * tp + fp + fn, 1.0)
        return {
            "IoU": 100.0 * float(iou),
            "Dice": 100.0 * float(dice),
            "clDice": 100.0 * float(cl_sum / max(samples, 1.0)),
            "Pd": 100.0 * float(pd_correct / max(pd_all, 1.0)),
            "Fa_1e-4": 1e4 * float(false_num / max(frames * self.cfg.data.sensor_size[0] * self.cfg.data.sensor_size[1], 1.0)),
            "Frag": float(frag_sum / max(samples, 1.0)),
            "threshold": float(self.cfg.eval.threshold),
            "samples": int(samples),
        }
