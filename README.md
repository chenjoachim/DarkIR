# DarkIR: Robust Low-Light Image Restoration

This repository contains the implementation of DarkIR, a robust low-light image restoration model.

## Data Preparation

### 1. Synthesize Motion Blur
Synthesize motion blur on RAW images while preserving EXIF metadata.
```bash
python synthesize.py --input_dir <input_dir> --gt_dir <gt_dir> --output_dir <output_dir> [--blur_gt] [--gt_output_dir <dir>]
```

### 2. Extract EXIF Data
Extract EXIF metadata from images to JSON files.
```bash
python extract_exif.py -i <input_dir> -o <output_dir>
```

### 3. Convert RAW to sRGB
Convert RAW/TIFF images to 16-bit sRGB PNG format.
```bash
python raw_to_srgb.py -i <input_dir> -o <output_dir>
```

### 4. Prepare Dataset Structure
Organize images into the required directory structure for training/testing.
```bash
python prepare_dataset.py --source_low <low_dir> --source_high <high_dir> --output <output_dir>
```

**Note:** The synthesized data (with blurred tiff and exif data) can be found at [LINK](https://drive.google.com/drive/folders/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC?usp=sharing).

## Training (Finetuning)

To finetune the model on your dataset:
```bash
python finetune.py --config options/finetune/finetune_default.yml [--resume <checkpoint_path>]
```

## Inference

To run inference on images:
```bash
python inference.py --config options/inference/default.yml --inp_path <input_dir> --out_path <output_dir> [--use_exif --exif_path <exif_dir>]
```

## Evaluation

Calculate PSNR, SSIM, and LPIPS metrics between predicted and ground truth images.
```bash
python scoring.py --pred_dir <pred_dir> --gt_dir <gt_dir> --output metrics.csv
```

## Configuration

The model and training parameters are defined in YAML configuration files located in the `options/` directory.

### Finetuning Configuration (`options/finetune/*.yml`)
Key parameters to modify:
- `datasets.train.train_path`: Path to your training dataset (containing `low` and `high` folders).
- `datasets.val.test_path`: Path to your validation dataset.
- `network.pretrained_path`: Path to the pretrained model weights to start from.
- `train.lr_initial`: Initial learning rate.
- `train.epochs`: Number of training epochs.

### Inference Configuration (`options/inference/*.yml`)
Key parameters to modify:
- `save.path`: Path to the model weights file to use for inference.
- `network`: Architecture settings (must match the trained model).

### Using EXIF Metadata

**Finetuning:**
To use EXIF metadata during training, modify your YAML config (e.g., `options/finetune/finetune_exif.yml`) to include:
```yaml
datasets:
  train:
    use_exif: True
    exif_path: /path/to/train/exif_json_folder
  val:
    use_exif: True
    exif_path: /path/to/test/exif_json_folder

train:
  use_exif: True
```

**Inference:**
To use EXIF metadata during inference, add the arguments to the command:
```bash
python inference.py ... --use_exif --exif_path /path/to/exif_json_folder
```

