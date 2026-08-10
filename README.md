# RidgeField-Net: Continuous Spatio-Temporal Tube Fields for Tiny UAV Detection in Event Streams

This repository provides an engineering reproduction of RidgeField-Net on the EV-UAV dataset. It is not the official implementation released by the original authors. During reproduction, implementation-level optimization and validation-driven tuning were applied to details not fully specified in the paper; the resulting model improves five of the six reported EV-UAV metrics while using a fixed test protocol.

## Requirements

The released checkpoint and scripts were verified with:

- Python 3.8.20
- PyTorch 1.9.1+cu111, CUDA runtime 11.1, cuDNN 8.0.5
- `spconv-cu113` 2.3.6 and `cumm-cu113` 0.4.11
- NumPy 1.24.1, SciPy 1.10.1, and PyYAML 6.0.2

The tested hardware was an NVIDIA RTX 3090. `spconv` wheels are CUDA- and PyTorch-version sensitive; use the versions above or a known-compatible replacement.

## Installation

```bash
conda create -n ridgefield-net python=3.8
conda activate ridgefield-net
pip install -r requirements.txt
```

## Dataset

EV-UAV is not included. Prepare the official dataset with this layout:

```text
<EV-UAV-root>/
├── train/*.npz
├── val/*.npz
└── test/*.npz
```

Each archive must provide `evs_norm` and `ev_loc`, as in the EV-UAV benchmark. The default config uses `./datasets/EV-UAV`; either edit `data.root` in `configs/ridgefield_ev_uav.yaml` or pass `--data-root <EV-UAV-root>` to the training and evaluation commands.

## Training

Single-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0 python train_ridgefield.py \
  --config configs/ridgefield_ev_uav.yaml \
  --data-root /path/to/EV-UAV
```

For the tested two-GPU setup:

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 \
python -m torch.distributed.launch --nproc_per_node=2 --use_env \
  train_ridgefield.py --config configs/ridgefield_ev_uav.yaml \
  --data-root /path/to/EV-UAV
```

## Evaluation

The released checkpoint is `checkpoints/ridgefield_net_ev_uav_best.pth`.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_ridgefield.py \
  --config configs/ridgefield_ev_uav.yaml \
  --checkpoint checkpoints/ridgefield_net_ev_uav_best.pth \
  --data-root /path/to/EV-UAV \
  --split test
```

Add `--no-save` to print results without writing a JSON file. Evaluation uses event-level threshold `τ=0.9`; Pd uses 50 ms non-overlapping slices and coverage threshold `η=1e-4`.

## Results

The final checkpoint is the validation-selected checkpoint from training epoch 9. The table below reports its verified evaluation on the 24-sequence EV-UAV test split.

| Metric | Paper RidgeField-Net | Our reproduction |
|---|---:|---:|
| IoU (%) ↑ | 63.46 | 76.59 |
| Dice (%) ↑ | 81.28 | 86.74 |
| clDice (%) ↑ | 70.20 | 87.90 |
| Pd (%) ↑ | 85.76 | 89.51 |
| Fa (×10⁻⁴) ↓ | 0.94 | 0.201 |
| Frag ↓ | 0.74 | 1.048 |

The reproduction exceeds the paper values for IoU, Dice, clDice, Pd, and Fa; Frag is higher than the paper result. The reported test evaluation uses the fixed protocol above, without test-time threshold tuning, calibration, TTA, or ensembling.

## Acknowledgements

This reproduction is based on *RidgeField-Net: Continuous Spatio-Temporal Tube Fields for Tiny UAV Detection in Event Streams* and the EV-UAV benchmark. Please cite the original paper and EV-UAV benchmark when using this code or checkpoint.
