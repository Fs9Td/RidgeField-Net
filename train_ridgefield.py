import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import spconv.pytorch as spconv
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ridgefield_net.config import load_config
from ridgefield_net.data import EVUAVDataset, evuav_collate
from ridgefield_net.losses import RidgeFieldLoss
from ridgefield_net.metrics import MetricAccumulator
from ridgefield_net.models import RidgeFieldNet, build_sparse_input


def parse_args():
    parser = argparse.ArgumentParser(description="训练 RidgeField-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--reset-lr", type=float, default=0.0, help="resume 后重置为固定微调学习率")
    parser.add_argument("--sanity-iters", type=int, default=0, help="只运行若干训练 iteration 后退出")
    parser.add_argument("--data-root", default="", help="EV-UAV 数据集根目录；覆盖 YAML 中的 data.root")
    parser.add_argument("--opts", nargs="*", default=None)
    return parser.parse_args()


def setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    if distributed:
        # 本机两张 3090 的 NCCL P2P 通道在参数 broadcast 阶段会卡住；
        # 关闭 P2P 后仍使用 NCCL 进行双卡梯度同步，实测可正常运行。
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        local_rank = 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return distributed, local_rank, device


def seed_all(seed, rank):
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def is_main(rank):
    return rank == 0


def save_state(path, model, optimizer, scheduler, scaler, epoch, cfg):
    state = {
        "model": (model.module if isinstance(model, DDP) else model).state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "epoch": int(epoch), "config": cfg.to_dict(),
    }
    torch.save(state, path)


def load_state(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    state = torch.load(path, map_location=device)
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("epoch", -1)) + 1


@torch.no_grad()
def validate(net, model_ref, loader, cfg, device, distributed):
    model_ref.eval()
    accum = MetricAccumulator(cfg)
    query_chunk = int(cfg.eval.event_query_chunk)
    for batch in loader:
        sparse = build_sparse_input(batch, cfg, device)
        dense = model_ref(sparse)
        coords = batch["coords_xyz"].to(device)
        bids = batch["event_batch_ids"].to(device)
        probs = model_ref.query_probs_chunked(dense, coords, bids, query_chunk).cpu().numpy()
        batch_ids = batch["event_batch_ids"].numpy()
        for sample_id in range(len(batch["sample_names"])):
            mask = batch_ids == sample_id
            accum.update(
                probs[mask], batch["labels"].numpy()[mask], batch["coords_xyz"].numpy()[mask],
                batch["coords_raw"].numpy()[mask], batch["inst_ids"].numpy()[mask],
            )
    accum.merge_distributed(device)
    return accum.summary()


def main():
    args = parse_args()
    distributed, rank, device = setup_distributed()
    overrides = list(args.opts or [])
    if args.data_root:
        overrides.append(f"data.root={args.data_root}")
    cfg = load_config(args.config, overrides)
    seed_all(cfg.train.seed, rank)
    output = Path(cfg.experiment.output_dir)
    if is_main(rank):
        output.mkdir(parents=True, exist_ok=True)
        (output / "config_used.json").write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.sanity_iters:
        print(f"rank={rank} device={device} distributed={distributed} initialized", flush=True)

    model = RidgeFieldNet(cfg).to(device)
    if args.sanity_iters:
        print(f"rank={rank} model built", flush=True)
    model_ref = model
    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
        model_ref = model.module
        if args.sanity_iters:
            print(f"rank={rank} DDP wrapped", flush=True)
    criterion = RidgeFieldLoss(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg.train.epochs), eta_min=float(cfg.train.min_lr)
    )
    # spconv + 3D soft-skeleton 的早期梯度范围较宽，低一些的初始 scale 可避免
    # torch 1.9 的 GradScaler 在第一步就产生溢出。
    scaler = GradScaler(
        enabled=bool(cfg.train.amp) and device.type == "cuda",
        init_scale=1024.0, growth_interval=2000,
    )
    start_epoch = load_state(args.resume, model, optimizer, scheduler, scaler, device) if args.resume else 0
    if args.resume and args.reset_lr > 0:
        for group in optimizer.param_groups:
            group["lr"] = float(args.reset_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(cfg.train.epochs) - start_epoch, 1),
            eta_min=float(args.reset_lr) * 0.1,
        )
    if args.sanity_iters:
        print(f"rank={rank} optimizer ready", flush=True)

    train_set = EVUAVDataset(cfg, "train")
    val_set = EVUAVDataset(cfg, "val")
    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_set, batch_size=int(cfg.train.per_gpu_batch_size), shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=int(cfg.data.num_workers), pin_memory=True,
        collate_fn=evuav_collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, sampler=val_sampler,
        num_workers=int(cfg.data.num_workers), pin_memory=True, collate_fn=evuav_collate,
    )
    if args.sanity_iters:
        print(f"rank={rank} loaders ready", flush=True)
    if is_main(rank):
        print(f"train={len(train_set)} val={len(val_set)} params={sum(p.numel() for p in model_ref.parameters()) / 1e6:.3f}M")
    best_iou = -1.0
    best_score = -1.0
    sanity_seen = 0

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        train_set.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        for iteration, batch in enumerate(train_loader):
            if args.sanity_iters:
                print(f"rank={rank} batch={iteration + 1} loaded", flush=True)
            sparse = build_sparse_input(batch, cfg, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg.train.amp) and device.type == "cuda"):
                dense = model(sparse)
                losses = criterion(model_ref, dense, batch)
                loss = losses["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite_grads = all(
                p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
            )
            if finite_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
            sanity_seen += 1
            if is_main(rank) and ((iteration + 1) % 10 == 0 or iteration + 1 == len(train_loader)):
                print(f"epoch {epoch + 1:03d} iter {iteration + 1:03d}/{len(train_loader)} "
                      f"loss={loss.item():.5f} seg={losses['seg'].item():.5f} "
                      f"psc={losses['skel'].item():.5f} vol={losses['vol'].item():.5f}", flush=True)
            if args.sanity_iters and sanity_seen >= args.sanity_iters:
                if is_main(rank):
                    torch.save({"model": model_ref.state_dict(), "config": cfg.to_dict()}, output / "sanity.pth")
                if distributed:
                    dist.barrier()
                if distributed:
                    dist.destroy_process_group()
                return
        scheduler.step()
        if (epoch + 1) % int(cfg.train.eval_interval) == 0:
            summary = validate(model, model_ref, val_loader, cfg, device, distributed)
            if is_main(rank):
                print(f"epoch {epoch + 1:03d} val={json.dumps(summary, ensure_ascii=False)}", flush=True)
                save_state(output / "latest.pth", model, optimizer, scheduler, scaler, epoch, cfg)
                score = summary["IoU"] + summary["Dice"] + summary["clDice"] - summary["Frag"]
                if summary["IoU"] > best_iou:
                    best_iou = summary["IoU"]
                    save_state(output / "best.pth", model, optimizer, scheduler, scaler, epoch, cfg)
                if score > best_score:
                    best_score = score
                    save_state(output / "best_tradeoff.pth", model, optimizer, scheduler, scaler, epoch, cfg)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
