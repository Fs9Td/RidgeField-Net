import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode_3d(x):
    a = -F.max_pool3d(-x, (3, 1, 1), 1, (1, 0, 0))
    b = -F.max_pool3d(-x, (1, 3, 1), 1, (0, 1, 0))
    c = -F.max_pool3d(-x, (1, 1, 3), 1, (0, 0, 1))
    return torch.minimum(torch.minimum(a, b), c)


def soft_skeletonize_3d(x, iterations=10):
    """clDice 风格的可微 3D soft skeleton。"""
    x = x.float().clamp(0.0, 1.0)
    opened = F.max_pool3d(soft_erode_3d(x), 3, 1, 1)
    skeleton = F.relu(x - opened)
    for _ in range(int(iterations)):
        x = soft_erode_3d(x)
        opened = F.max_pool3d(soft_erode_3d(x), 3, 1, 1)
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def soft_cldice(pred, target, iterations=10, eps=1e-6):
    pred = pred.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    sp = soft_skeletonize_3d(pred, iterations)
    st = soft_skeletonize_3d(target, iterations)
    dims = tuple(range(1, pred.ndim))
    precision = (sp * target).sum(dims) / (sp.sum(dims) + eps)
    sensitivity = (st * pred).sum(dims) / (st.sum(dims) + eps)
    return (2.0 * precision * sensitivity / (precision + sensitivity + eps)).clamp(0.0, 1.0)


def segmentation_loss(logits, targets, alpha=0.75, gamma=2.0, dice_weight=1.0):
    targets = targets.float()
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = (alpha_t * (1.0 - pt).pow(gamma) * ce).mean()
    intersection = (probs * targets).sum()
    dice = 1.0 - (2.0 * intersection + 1e-6) / (probs.sum() + targets.sum() + 1e-6)
    return focal + dice_weight * dice, focal, dice


def patch_offsets(cfg, device):
    nx, ny, nt = [int(x) for x in cfg.train.psc_patch_size]
    rx, ry, rt = [float(x) for x in cfg.train.psc_patch_radius]
    xs = torch.linspace(-rx + rx / nx, rx - rx / nx, nx, device=device)
    ys = torch.linspace(-ry + ry / ny, ry - ry / ny, ny, device=device)
    ts = torch.linspace(-rt + rt / nt, rt - rt / nt, nt, device=device)
    tt, yy, xx = torch.meshgrid(ts, ys, xs)
    return torch.stack([xx, yy, tt], dim=-1).reshape(-1, 3)


def patch_queries(anchors, anchor_batch, offsets):
    k, p = anchors.shape[0], offsets.shape[0]
    coords = (anchors[:, None] + offsets[None]).clamp(0.0, 1.0 - 1e-6).reshape(k * p, 3)
    bids = anchor_batch[:, None].expand(k, p).reshape(-1)
    return coords, bids


def patch_targets(pos_coords, pos_batch, anchors, anchor_batch, cfg, device):
    nx, ny, nt = [int(x) for x in cfg.train.psc_patch_size]
    rx, ry, rt = [float(x) for x in cfg.train.psc_patch_radius]
    result = anchors.new_zeros((len(anchors), 1, nt, ny, nx), device=device)
    for i in range(len(anchors)):
        points = pos_coords[pos_batch == anchor_batch[i]]
        if points.numel() == 0:
            continue
        rel = points - anchors[i]
        keep = (
            (rel[:, 0] >= -rx) & (rel[:, 0] < rx) &
            (rel[:, 1] >= -ry) & (rel[:, 1] < ry) &
            (rel[:, 2] >= -rt) & (rel[:, 2] < rt)
        )
        rel = rel[keep]
        if len(rel) == 0:
            continue
        ix = ((rel[:, 0] + rx) / (2 * rx) * nx).floor().long().clamp(0, nx - 1)
        iy = ((rel[:, 1] + ry) / (2 * ry) * ny).floor().long().clamp(0, ny - 1)
        it = ((rel[:, 2] + rt) / (2 * rt) * nt).floor().long().clamp(0, nt - 1)
        result[i, 0, it, iy, ix] = 1.0
    for _ in range(int(cfg.train.psc_gt_dilation_steps)):
        result = F.max_pool3d(result, 3, 1, 1).clamp(0.0, 1.0)
    return result


class RidgeFieldLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._offsets = None

    def forward(self, model, dense_grid, batch):
        device = dense_grid.device
        event_coords = batch["event_query_coords"].to(device)
        event_labels = batch["event_query_labels"].to(device)
        event_batch = batch["event_query_batch_ids"].to(device)
        event_logits = model.query_logits(dense_grid, event_coords, event_batch)
        seg, focal, dice = segmentation_loss(
            event_logits, event_labels, float(self.cfg.train.focal_alpha),
            float(self.cfg.train.focal_gamma), float(self.cfg.train.lambda_dice),
        )

        bg = batch["bg_queries"].to(device)
        bg_batch = batch["bg_batch_ids"].to(device)
        if len(bg):
            vol = torch.sigmoid(model.query_logits(dense_grid, bg, bg_batch)).mean()
        else:
            vol = dense_grid.new_zeros(())

        anchors = batch["anchors"].to(device)
        anchor_batch = batch["anchor_batch_ids"].to(device)
        pos_coords = batch["coords_xyz"].to(device)
        pos_batch = batch["event_batch_ids"].to(device)
        if len(anchors):
            if self._offsets is None or self._offsets.device != device:
                self._offsets = patch_offsets(self.cfg, device)
            patch_coords, patch_batch = patch_queries(anchors, anchor_batch, self._offsets)
            patch_logits = model.query_logits(dense_grid, patch_coords, patch_batch)
            nx, ny, nt = [int(x) for x in self.cfg.train.psc_patch_size]
            pred = torch.sigmoid(patch_logits).view(len(anchors), 1, nt, ny, nx)
            target = patch_targets(pos_coords, pos_batch, anchors, anchor_batch, self.cfg, device)
            skel = 1.0 - soft_cldice(pred, target, int(self.cfg.train.psc_skeleton_iterations)).mean()
        else:
            skel = dense_grid.new_zeros(())

        total = (
            float(self.cfg.train.lambda_seg) * seg +
            float(self.cfg.train.lambda_skel) * skel +
            float(self.cfg.train.lambda_vol) * vol
        )
        return {
            "loss": total,
            "seg": seg.detach(), "focal": focal.detach(), "dice": dice.detach(),
            "skel": skel.detach(), "vol": vol.detach(),
        }
