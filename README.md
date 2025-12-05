# DarkIR: Robust Low-Light Image Restoration

This repository contains a project based on DarkIR, a robust low-light image restoration model.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chenjoachim/DarkIR.git
   cd DarkIR
   git checkout clean
   ```

2. **Create a Python environment (recommended):**
   ```bash
   conda create -n darkir python=3.10
   conda activate darkir
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install `exiftool` (required for EXIF extraction):**
   - **Ubuntu/Debian:** `sudo apt-get install libimage-exiftool-perl`
   - **MacOS:** `brew install exiftool`
   - **Windows:** Download from [exiftool.org](https://exiftool.org/) and add to PATH.

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
python prepare_dataset.py --source_low <low_dir> --source_high <high_dir> --output <output_dir> [--source_exif <exif_dir>]
```
If `--source_exif` is provided, the script will also organize the EXIF JSON files into the corresponding train/test directories.

**Note:** The synthesized data (with blurred tiff and exif data) can be found [HERE](https://drive.google.com/drive/folders/1pLOJqrVbo0zowHCbHOYTHiClrBLV40vt?usp=sharing).

## Optional: Two-Stage Pipeline Preprocessing

If you are using a two-stage pipeline involving NAFNet, follow these steps:

1.  **Convert to sRGB:** First, convert your RAW images to sRGB using the command from **Data Preparation Step 3**.
2.  **NAFNet Prediction:** Run `predict.py` from the NAFNet repository on the converted sRGB images.

**Note:** This preprocessing is **optional** and specific to the two-stage pipeline. The standard DarkIR fine-tuning and inference (described below) do **not** require sRGB conversion and operate directly on the data prepared in Step 4.

## Training (Finetuning)

To finetune the DarkIR model on your own dataset, use the `finetune.py` script.

**Usage:**
```bash
python finetune.py --config options/finetune/finetune_default.yml [--resume <checkpoint_path>]
```

- `--config`: Path to the YAML configuration file (e.g., `options/finetune/finetune_default.yml`). This file defines dataset paths, hyperparameters, and model settings.
- `--resume`: (Optional) Path to a checkpoint file (`.pt`) to resume training from.

**Example:**
```bash
python finetune.py --config options/finetune/finetune_exif.yml
```

## Inference

To restore low-light images using a trained model, use the `inference.py` script.

**Usage:**
```bash
python inference.py --config options/inference/default.yml --inp_path <input_dir> --out_path <output_dir> [--use_exif --exif_path <exif_dir>]
```

- `--config`: Path to the inference configuration file. Ensure the network settings match your trained model.
- `--inp_path`: Directory containing the input low-light images.
- `--out_path`: Directory where the restored images will be saved.
- `--use_exif`: (Optional) Flag to enable using EXIF metadata for restoration. Only use if the model was trained with EXIF data.
- `--exif_path`: (Optional) Directory containing the corresponding EXIF JSON files (required if `--use_exif` is set).

**Example:**
```bash
python inference.py --config options/inference/default.yml --inp_path ./data/test/low --out_path ./results --use_exif --exif_path ./data/test/exif
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


## Citation

This project is based on DarkIR. Please cite the original paper:

```bibtex
@InProceedings{Feijoo_2025_CVPR,
    author    = {Feijoo, Daniel and Benito, Juan C. and Garcia, Alvaro and Conde, Marcos V.},
    title     = {DarkIR: Robust Low-Light Image Restoration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {10879-10889}
}
```
