import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


class SparseResidualBlock(spconv.SparseModule):
    def __init__(self, channels, key):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False, indice_key=key)
        self.bn1 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.conv2 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False, indice_key=key)
        self.bn2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.features
        y = self.conv1(x)
        y = y.replace_feature(self.act(self.bn1(y.features)))
        y = self.conv2(y)
        y = y.replace_feature(self.bn2(y.features))
        return y.replace_feature(self.act(y.features + identity))


class Sparse3DEncoder(nn.Module):
    """紧凑的稀疏 U 形编码器，输出输入体素分辨率一半的查询网格。"""
    def __init__(self, in_channels=7, base=16, out_channels=32):
        super().__init__()
        norm = lambda c: nn.BatchNorm1d(c, eps=1e-3, momentum=0.01)
        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base, 3, padding=1, bias=False, indice_key="stem"),
            norm(base), nn.ReLU(inplace=True),
        )
        self.enc1 = spconv.SparseSequential(SparseResidualBlock(base, "enc1"))
        self.down1 = spconv.SparseSequential(
            spconv.SparseConv3d(base, base * 2, 3, stride=2, padding=1, bias=False, indice_key="down1"),
            norm(base * 2), nn.ReLU(inplace=True),
        )
        self.enc2 = spconv.SparseSequential(SparseResidualBlock(base * 2, "enc2"))
        self.down2 = spconv.SparseSequential(
            spconv.SparseConv3d(base * 2, base * 3, 3, stride=2, padding=1, bias=False, indice_key="down2"),
            norm(base * 3), nn.ReLU(inplace=True),
        )
        self.bottleneck = spconv.SparseSequential(SparseResidualBlock(base * 3, "bottleneck"))
        self.up2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(base * 3, base * 2, 3, indice_key="down2", bias=False),
            norm(base * 2), nn.ReLU(inplace=True),
        )
        self.fuse = spconv.SparseSequential(
            spconv.SubMConv3d(base * 4, base * 2, 1, bias=False, indice_key="fuse"),
            norm(base * 2), nn.ReLU(inplace=True),
        )
        self.proj = spconv.SparseSequential(
            spconv.SubMConv3d(base * 2, out_channels, 1, bias=False, indice_key="proj"),
            norm(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.enc1(self.stem(x))
        x2 = self.enc2(self.down1(x1))
        xb = self.bottleneck(self.down2(x2))
        up = self.up2(xb)
        up = up.replace_feature(torch.cat([up.features, x2.features], dim=1))
        return self.proj(self.fuse(up))


class FourierPositionalEncoding(nn.Module):
    def __init__(self, bands=6):
        super().__init__()
        self.bands = int(bands)
        self.register_buffer("freq", 2.0 ** torch.arange(self.bands, dtype=torch.float32), persistent=False)

    @property
    def out_dim(self):
        return 3 + 6 * self.bands

    def forward(self, coords):
        parts = [coords]
        for frequency in self.freq:
            angle = coords * (2.0 * math.pi * frequency)
            parts.extend([torch.sin(angle), torch.cos(angle)])
        return torch.cat(parts, dim=-1)


class ImplicitDecoder(nn.Module):
    """论文规定的 5 层 MLP（4 个隐层 + 1 个输出层）。"""
    def __init__(self, in_dim, hidden=128, layers=5):
        super().__init__()
        blocks = []
        dim = in_dim
        for _ in range(int(layers) - 1):
            blocks.extend([nn.Linear(dim, hidden), nn.ReLU(inplace=True)])
            dim = hidden
        blocks.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_sparse_input(batch, cfg, device):
    spatial = [int(cfg.data.voxel_resolution[2]), int(cfg.data.voxel_resolution[1]), int(cfg.data.voxel_resolution[0])]
    return spconv.SparseConvTensor(
        features=batch["voxel_feats"].to(device, non_blocking=True),
        indices=batch["voxel_coords"].to(device, non_blocking=True).int(),
        spatial_shape=spatial,
        batch_size=len(batch["sample_names"]),
    )


def trilinear_query(feature_grid, coords_xyz, batch_ids):
    """对 [B,C,T,Y,X] 网格做真正的局部 2×2×2 三线性插值。"""
    if coords_xyz.numel() == 0:
        return feature_grid.new_zeros((0, feature_grid.shape[1]))
    coords_xyz = coords_xyz.clamp(0.0, 1.0 - 1e-6)
    output = feature_grid.new_zeros((coords_xyz.shape[0], feature_grid.shape[1]))
    for bid in batch_ids.unique(sorted=True).tolist():
        mask = batch_ids == bid
        grid = (coords_xyz[mask] * 2.0 - 1.0).view(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            feature_grid[int(bid):int(bid) + 1], grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        )
        output[mask] = sampled[0, :, :, 0, 0].transpose(0, 1).to(output.dtype)
    return output


class RidgeFieldNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Sparse3DEncoder(
            in_channels=int(cfg.model.input_channels),
            base=int(cfg.model.encoder_base_channels),
            out_channels=int(cfg.model.latent_channels),
        )
        self.pe = FourierPositionalEncoding(int(cfg.model.positional_encoding_bands))
        self.decoder = ImplicitDecoder(
            self.pe.out_dim + int(cfg.model.latent_channels),
            hidden=int(cfg.model.implicit_hidden),
            layers=int(cfg.model.implicit_layers),
        )

    def forward(self, sparse_input):
        # 调用该 forward 是 DDP 的同步入口；dense 仅是查询尺度的 latent grid。
        return self.encoder(sparse_input).dense().contiguous()

    def query_logits(self, dense_grid, coords_xyz, batch_ids):
        local = trilinear_query(dense_grid, coords_xyz, batch_ids)
        return self.decoder(torch.cat([local, self.pe(coords_xyz)], dim=-1))

    def query_logits_chunked(self, dense_grid, coords_xyz, batch_ids, chunk_size):
        outputs = []
        for start in range(0, len(coords_xyz), int(chunk_size)):
            end = min(start + int(chunk_size), len(coords_xyz))
            outputs.append(self.query_logits(dense_grid, coords_xyz[start:end], batch_ids[start:end]))
        return torch.cat(outputs, dim=0) if outputs else dense_grid.new_zeros((0,))

    def query_probs_chunked(self, dense_grid, coords_xyz, batch_ids, chunk_size):
        return torch.sigmoid(self.query_logits_chunked(dense_grid, coords_xyz, batch_ids, chunk_size))
