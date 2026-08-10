import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ridgefield_net.config import load_config
from ridgefield_net.data import EVUAVDataset, evuav_collate
from ridgefield_net.metrics import MetricAccumulator
from ridgefield_net.models import RidgeFieldNet, build_sparse_input


def main():
    parser = argparse.ArgumentParser(description="评估 RidgeField-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="")
    parser.add_argument("--data-root", default="", help="EV-UAV 数据集根目录；覆盖 YAML 中的 data.root")
    parser.add_argument("--no-save", action="store_true", help="仅打印结果，不写入 JSON 文件")
    args = parser.parse_args()
    overrides = [f"data.root={args.data_root}"] if args.data_root else None
    cfg = load_config(args.config, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RidgeFieldNet(cfg).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    dataset = EVUAVDataset(cfg, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(cfg.data.num_workers), pin_memory=True, collate_fn=evuav_collate)
    accum = MetricAccumulator(cfg)
    with torch.no_grad():
        for number, batch in enumerate(loader, 1):
            dense = model(build_sparse_input(batch, cfg, device))
            probs = model.query_probs_chunked(
                dense, batch["coords_xyz"].to(device), batch["event_batch_ids"].to(device), int(cfg.eval.event_query_chunk)
            ).cpu().numpy()
            accum.update(
                probs, batch["labels"].numpy(), batch["coords_xyz"].numpy(),
                batch["coords_raw"].numpy(), batch["inst_ids"].numpy(),
            )
            print(f"{number}/{len(loader)} {batch['sample_names'][0]}", flush=True)
    result = accum.summary()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.no_save:
        output = args.output or str(Path(cfg.experiment.output_dir) / f"{args.split}_metrics.json")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
