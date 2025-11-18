# DarkIR Finetuning Guide

> **Complete finetuning pipeline for DarkIR low-light image enhancement**

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision numpy tqdm lpips pytorch-msssim pillow ptflops pyyaml

# 2. Prepare dataset (low/high image pairs)
python prepare_dataset.py \
    --source_low /path/to/dark/images \
    --source_high /path/to/bright/images \
    --output ./data/datasets/custom

# 3. Validate setup
python validate_setup.py

# 4. Start training
python finetune.py -p options/finetune/finetune_default.yml
```

Or use interactive mode: `./quick_start_finetune.sh`

---

## Dataset Structure

```
data/datasets/your_dataset/
├── train/
│   ├── low/        # 10101_03_0.1s.png, 10101_05_0.2s.png (multiple variants)
│   └── high/       # 10101.png (shared ground truth)
└── test/
    ├── low/
    └── high/
```

**Many-to-One Mapping**: Multiple low images can share one high image. Matched by prefix (e.g., `10101_*` → `10101.png`).

---

## Configuration

Edit `options/finetune/finetune_default.yml`:

```yaml
# Key settings to adjust
datasets:
  train:
    train_path: ./data/datasets/custom/train
    batch_size: 8              # Reduce if OOM (2-16)
    cropsize: 256              # 128/256/384

network:
  pretrained_path: ./models/DarkIR_384.pt

train:
  epochs: 100
  lr_initial: 0.0001          # Lower (0.00005) for finetuning
  use_side_loss: True

losses:
  main_loss: {type: CharbonnierLoss, weight: 1.0}
  ssim_loss: {type: SSIMloss, weight: 0.2}

save:
  path: ./models/finetuned
```

**Available Configs**:
- `finetune_default.yml` - General purpose (batch=8, crop=256, epochs=100)
- `finetune_lolblur.yml` - LOLBlur optimized (lower LR)
- `finetune_minimal.yml` - Quick testing (batch=2, crop=128, epochs=10)

---

## Training Commands

```bash
# Basic training
python finetune.py -p options/finetune/finetune_default.yml

# Resume from checkpoint
python finetune.py --resume models/finetuned/epoch_10.pt

# Multi-GPU (automatic)
python finetune.py  # Uses all available GPUs

# Specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python finetune.py
```

**Output**: Models saved to `{save.path}/epoch_N.pt` and `best_model.pt`

---

## Loss Functions

Combine multiple losses for better results:

| Loss | Type | Weight | Use Case |
|------|------|--------|----------|
| `CharbonnierLoss` | Reconstruction | 1.0 | **Required** - main loss |
| `SSIMloss` | Structural | 0.1-0.2 | **Recommended** - structure |
| `VGGLoss` | Perceptual | 0.01-0.05 | Visual quality (slower) |
| `EdgeLoss` | Sharpness | 0.05-0.2 | Deblurring tasks |
| `FrequencyLoss` | Frequency | 0.01 | Detail preservation |
| `EnhanceLoss` | Multi-scale | 0.3-0.7 | Requires `use_side_loss: True` |

**Recommended Combinations**:
```yaml
# Basic (fast)
losses:
  main_loss: {type: CharbonnierLoss, weight: 1.0}

# Standard (good)
losses:
  main_loss: {type: CharbonnierLoss, weight: 1.0}
  ssim_loss: {type: SSIMloss, weight: 0.2}

# High quality (best)
losses:
  main_loss: {type: CharbonnierLoss, weight: 1.0}
  ssim_loss: {type: SSIMloss, weight: 0.2}
  perceptual_loss: {type: VGGLoss, weight: 0.01}
  enhance_loss: {type: EnhanceLoss, weight: 0.5}
```

---

## Monitoring

**Console Output**:
```
Epoch [1/100]
Training: 100%|████| 125/125 [00:32<00:00, 3.85it/s, loss=0.0234]
Training Loss: 0.0234
  Validation --- PSNR: 23.45, SSIM: 0.8234, LPIPS: 0.1234
Model saved to models/finetuned/epoch_1.pt
```

**Metrics**:
- **PSNR**: >30 is excellent, 25-30 is good, <20 is poor
- **SSIM**: >0.9 is excellent, 0.8-0.9 is good, <0.7 is poor
- **LPIPS**: <0.1 is excellent, 0.1-0.2 is good, >0.3 is poor

**W&B Logging** (optional): Set `wandb.init: True` in config

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of Memory** | Reduce `batch_size: 2`, `cropsize: 128`, or `n_workers: 0` |
| **Slow training** | Increase `n_workers: 8`, use SSD storage, disable VGGLoss |
| **Poor results** | Lower LR to `0.00005`, add perceptual loss, train longer |
| **Import errors** | `pip install torch lpips pytorch-msssim` |
| **No matching images** | Check prefix matching: `10101_*.png` → `10101.png` |
| **NaN loss** | Lower learning rate, check data quality |

**Quick Fix for OOM**:
```yaml
train:
  batch_size: 2
  cropsize: 128
```

---

## Helper Scripts

```bash
# Organize your dataset
python prepare_dataset.py \
    --source_low /raw/dark \
    --source_high /raw/bright \
    --output ./data/custom \
    --train_ratio 0.9

# Validate everything is set up correctly
python validate_setup.py

# Demo how prefix matching works
python demo_prefix_matching.py

# Interactive guided setup
./quick_start_finetune.sh
```

---

## Advanced Tips

**Hyperparameter Guidelines**:
- **batch_size**: 8 for 12GB GPU, 4 for 8GB, 2 for 6GB
- **cropsize**: 256 standard, 384 for high quality, 128 for speed
- **lr_initial**: 0.0001 standard, 0.00005 for careful finetuning
- **epochs**: 50-100 typical, monitor validation metrics

**Performance** (RTX 3090, 1000 images):
- ~5-10 min/epoch training
- 50 epochs ≈ 5-8 hours total

**Data Requirements**:
- Minimum: 100 pairs
- Good: 500+ pairs
- Best: 1000+ pairs

---

## Usage After Training

```python
import torch
from archs.DarkIR import DarkIR

# Load finetuned model
model = DarkIR(img_channel=3, width=32, ...)
checkpoint = torch.load('models/finetuned/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    enhanced = model(low_light_image)
```

---

## FAQ

**Q: GPU required?** Yes, CPU is too slow.

**Q: Multiple exposures per scene?** Yes! Use `10101_03.png`, `10101_05.png` → `10101.png` naming.

**Q: Training time?** Small dataset: 1-2 hours. Large: 8-12 hours.

**Q: Stop and resume?** Yes: `python finetune.py --resume models/finetuned/epoch_N.pt`

**Q: Pretrained weights needed?** Highly recommended for best results.

---

## Files

- `finetune.py` - Main training script
- `prepare_dataset.py` - Dataset organization
- `validate_setup.py` - Setup validation  
- `demo_prefix_matching.py` - Matching demo
- `quick_start_finetune.sh` - Interactive helper
- `options/finetune/*.yml` - Configuration templates

---

**That's it!** Start with `./quick_start_finetune.sh` or `python finetune.py`
