"""
Utility script to prepare a custom dataset for DarkIR finetuning

This script helps you organize your images into the required directory structure:
    dataset/
    ├── train/
    │   ├── low/     # Low-light images
    │   └── high/    # Normal-light images  
    └── test/
        ├── low/
        └── high/

Usage:
    python prepare_dataset.py --source_low <low_dir> --source_high <high_dir> --output <output_dir>
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def get_image_files(directory):
    """Get all image files from directory"""
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = []
    
    for ext in extensions:
        image_files.extend(Path(directory).glob(f'*{ext}'))
        image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def create_dataset_structure(output_dir):
    """Create the required directory structure"""
    dirs = [
        os.path.join(output_dir, 'train', 'low'),
        os.path.join(output_dir, 'train', 'high'),
        os.path.join(output_dir, 'test', 'low'),
        os.path.join(output_dir, 'test', 'high'),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return dirs


def copy_files(source_files, dest_dir, desc="Copying", use_symlinks=False):
    """Copy or symlink files with progress bar"""
    for src in tqdm(source_files, desc=desc):
        dest_path = os.path.join(dest_dir, src.name)
        if use_symlinks:
            # Create symbolic link to source file
            if os.path.exists(dest_path):
                os.unlink(dest_path)
            os.symlink(os.path.abspath(src), dest_path)
        else:
            shutil.copy2(src, dest_dir)


def verify_pairs(low_dir, high_dir):
    """Verify that low and high directories have matching filenames based on prefix
    
    Handles many-to-one mapping where multiple low images (e.g., 10101_03_0.1s.png, 10101_05_0.2s.png)
    share the same high image (e.g., 10101.png) by matching based on prefix numbers.
    """
    low_files = get_image_files(low_dir)
    high_files = get_image_files(high_dir)
    
    # Create mapping of high images by prefix
    high_dict = {}
    for high_file in high_files:
        prefix = high_file.stem.split('_')[0] if '_' in high_file.stem else high_file.stem
        high_dict[prefix] = high_file
    
    # Match low files to high files by prefix
    matched_low = []
    unmatched_low = []
    
    for low_file in low_files:
        prefix = low_file.stem.split('_')[0] if '_' in low_file.stem else low_file.stem
        if prefix in high_dict:
            matched_low.append(low_file)
        else:
            unmatched_low.append(low_file)
    
    if unmatched_low:
        print(f"Warning: {len(unmatched_low)} low images have no matching high image")
        print("First few examples:", [f.name for f in unmatched_low[:5]])
    
    # Check for unused high images
    used_prefixes = set()
    for low_file in low_files:
        prefix = low_file.stem.split('_')[0] if '_' in low_file.stem else low_file.stem
        if prefix in high_dict:
            used_prefixes.add(prefix)
    
    unused_high = len(high_dict) - len(used_prefixes)
    if unused_high > 0:
        print(f"Info: {unused_high} high images have no corresponding low images")
    
    print(f"✓ {len(matched_low)} low images matched to {len(used_prefixes)} unique high images")
    
    return len(matched_low) > 0


def split_dataset(low_files, high_files, train_ratio=0.9):
    """Split dataset into train and test sets
    
    Handles many-to-one mapping where multiple low images share the same high image.
    Splits by unique prefixes to ensure all variants of the same scene go together.
    """
    # Create mapping of high images by prefix
    high_dict = {}
    for high_file in high_files:
        prefix = high_file.stem.split('_')[0] if '_' in high_file.stem else high_file.stem
        high_dict[prefix] = high_file
    
    # Group low files by prefix
    low_by_prefix = {}
    for low_file in low_files:
        prefix = low_file.stem.split('_')[0] if '_' in low_file.stem else low_file.stem
        if prefix in high_dict:
            if prefix not in low_by_prefix:
                low_by_prefix[prefix] = []
            low_by_prefix[prefix].append(low_file)
    
    # Sort prefixes for reproducibility
    all_prefixes = sorted(low_by_prefix.keys())
    
    # Split by prefix (keeps all variants of same scene together)
    n_train = int(len(all_prefixes) * train_ratio)
    train_prefixes = set(all_prefixes[:n_train])
    test_prefixes = set(all_prefixes[n_train:])
    
    # Collect train and test files
    train_low = []
    train_high = []
    test_low = []
    test_high = []
    
    for prefix in all_prefixes:
        high_file = high_dict[prefix]
        low_files_for_prefix = low_by_prefix[prefix]
        
        if prefix in train_prefixes:
            for low_file in low_files_for_prefix:
                train_low.append(low_file)
                train_high.append(high_file)
        else:
            for low_file in low_files_for_prefix:
                test_low.append(low_file)
                test_high.append(high_file)
    
    return train_low, train_high, test_low, test_high


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for DarkIR finetuning')
    parser.add_argument('--source_low', type=str, required=True,
                        help='Directory containing low-light images')
    parser.add_argument('--source_high', type=str, required=True,
                        help='Directory containing normal-light (ground truth) images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for organized dataset')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of training data (default: 0.9)')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of creating symlinks')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DarkIR Dataset Preparation")
    print("="*60)
    print()
    
    # Check source directories
    if not os.path.exists(args.source_low):
        print(f"Error: Source low directory not found: {args.source_low}")
        return
    
    if not os.path.exists(args.source_high):
        print(f"Error: Source high directory not found: {args.source_high}")
        return
    
    # Get image files
    print("Scanning source directories...")
    low_files = get_image_files(args.source_low)
    high_files = get_image_files(args.source_high)
    
    print(f"Found {len(low_files)} low-light images")
    print(f"Found {len(high_files)} normal-light images")
    print()
    
    # Verify matching pairs
    print("Verifying image pairs...")
    if not verify_pairs(args.source_low, args.source_high):
        print("Error: No matching image pairs found!")
        print("Make sure filenames match between low and high directories.")
        return
    print()
    
    # Create output structure
    print(f"Creating dataset structure in {args.output}...")
    create_dataset_structure(args.output)
    print()
    
    # Split dataset
    print(f"Splitting dataset (train ratio: {args.train_ratio})...")
    train_low, train_high, test_low, test_high = split_dataset(
        low_files, high_files, args.train_ratio
    )
    
    print(f"Train set: {len(train_low)} pairs")
    print(f"Test set: {len(test_low)} pairs")
    print()
    
    # Copy or symlink files
    action = "Copying" if args.copy else "Linking"
    use_symlinks = not args.copy
    
    copy_files(train_low, 
               os.path.join(args.output, 'train', 'low'),
               f"{action} train low images",
               use_symlinks=use_symlinks)
    
    copy_files(train_high,
               os.path.join(args.output, 'train', 'high'),
               f"{action} train high images",
               use_symlinks=use_symlinks)
    
    copy_files(test_low,
               os.path.join(args.output, 'test', 'low'),
               f"{action} test low images",
               use_symlinks=use_symlinks)
    
    copy_files(test_high,
               os.path.join(args.output, 'test', 'high'),
               f"{action} test high images",
               use_symlinks=use_symlinks)
    
    print()
    print("="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print()
    print("Dataset structure:")
    print(f"  {args.output}/")
    print(f"  ├── train/")
    print(f"  │   ├── low/     ({len(train_low)} images)")
    print(f"  │   └── high/    ({len(train_high)} images)")
    print(f"  └── test/")
    print(f"      ├── low/     ({len(test_low)} images)")
    print(f"      └── high/    ({len(test_high)} images)")
    print()
    print("Next steps:")
    print("1. Update the config file with your dataset path:")
    print(f"   train_path: {args.output}/train")
    print(f"   test_path: {args.output}/test")
    print()
    print("2. Start finetuning:")
    print("   python finetune.py -p options/finetune/finetune_default.yml")
    print()


if __name__ == '__main__':
    main()
