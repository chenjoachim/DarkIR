#!/bin/bash

# Quick start script for finetuning DarkIR
# This script helps you get started with finetuning

echo "=========================================="
echo "DarkIR Finetuning Quick Start"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Check if required packages are installed
echo "Checking required packages..."
python -c "import torch; import torchvision; import numpy; import tqdm; import lpips; import pytorch_msssim" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Some required packages are missing."
    echo "Would you like to install them? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        pip install torch torchvision numpy tqdm lpips pytorch-msssim pillow ptflops
    else
        echo "Please install required packages before running."
        exit 1
    fi
fi

echo ""
echo "✓ All required packages are installed"
echo ""

# Check if dataset exists
echo "Checking dataset..."
DEFAULT_CONFIG="./options/finetune/finetune_default.yml"

if [ -f "$DEFAULT_CONFIG" ]; then
    TRAIN_PATH=$(python -c "import yaml; print(yaml.safe_load(open('$DEFAULT_CONFIG'))['datasets']['train']['train_path'])")
    
    if [ ! -d "$TRAIN_PATH/low" ] || [ ! -d "$TRAIN_PATH/high" ]; then
        echo "Warning: Training dataset not found at $TRAIN_PATH"
        echo "Please ensure your dataset is organized as:"
        echo "  $TRAIN_PATH/"
        echo "  ├── low/    # Low-light images"
        echo "  └── high/   # Normal-light images"
        echo ""
    else
        LOW_COUNT=$(ls -1 "$TRAIN_PATH/low" | wc -l)
        HIGH_COUNT=$(ls -1 "$TRAIN_PATH/high" | wc -l)
        echo "✓ Dataset found: $LOW_COUNT low-light images, $HIGH_COUNT normal-light images"
        echo ""
    fi
fi

# Check if pretrained model exists
echo "Checking pretrained model..."
if [ -f "$DEFAULT_CONFIG" ]; then
    PRETRAINED_PATH=$(python -c "import yaml; print(yaml.safe_load(open('$DEFAULT_CONFIG'))['network']['pretrained_path'])")
    
    if [ ! -f "$PRETRAINED_PATH" ]; then
        echo "Warning: Pretrained model not found at $PRETRAINED_PATH"
        echo "Please download the pretrained DarkIR model first."
        echo ""
    else
        echo "✓ Pretrained model found at $PRETRAINED_PATH"
        echo ""
    fi
fi

# Ask user which config to use
echo "Which configuration would you like to use?"
echo "1) Default configuration (options/finetune/finetune_default.yml)"
echo "2) LOLBlur configuration (options/finetune/finetune_lolblur.yml)"
echo "3) Custom configuration (specify path)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        CONFIG="./options/finetune/finetune_default.yml"
        ;;
    2)
        CONFIG="./options/finetune/finetune_lolblur.yml"
        ;;
    3)
        read -p "Enter config file path: " CONFIG
        if [ ! -f "$CONFIG" ]; then
            echo "Error: Config file not found: $CONFIG"
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice. Using default configuration."
        CONFIG="./options/finetune/finetune_default.yml"
        ;;
esac

echo ""
echo "Using configuration: $CONFIG"
echo ""

# Ask if user wants to resume from checkpoint
read -p "Do you want to resume from a checkpoint? (y/n): " resume_choice
if [[ "$resume_choice" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    read -p "Enter checkpoint path: " CHECKPOINT
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Warning: Checkpoint not found: $CHECKPOINT"
        echo "Starting training from scratch..."
        RESUME_FLAG=""
    else
        RESUME_FLAG="--resume $CHECKPOINT"
    fi
else
    RESUME_FLAG=""
fi

echo ""
echo "=========================================="
echo "Starting Finetuning"
echo "=========================================="
echo "Config: $CONFIG"
if [ -n "$RESUME_FLAG" ]; then
    echo "Resuming from: $CHECKPOINT"
fi
echo ""
echo "Press Ctrl+C to stop training"
echo ""

# Start training
python finetune.py -p "$CONFIG" $RESUME_FLAG

echo ""
echo "=========================================="
echo "Training finished!"
echo "=========================================="
