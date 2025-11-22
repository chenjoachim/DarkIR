import imageio
import rawpy
import numpy as np
import tifffile
import subprocess
import os
from tqdm import tqdm
from skimage.filters import gaussian
from utils.motion_blur import Kernel
from utils.demosaic import demosaic
import random

def apply_motion_blur(image, kernel_size=(70, 70), intensity=0.5):
    """
    Apply motion blur to an image using a specified kernel size and intensity.
    """
    kernel = Kernel(size=kernel_size, intensity=intensity)
    blurred_image = kernel.applyTo(image, keep_image_dim=True)
    return blurred_image
    

def save_modified_raw_with_exif(input_arw, gt_path, output_tiff, kernel_size=(70, 70), intensity=0.5, blur_gt=False, gt_output_dir=None):
    """
    Save modified Bayer with metadata preserved (excluding structural tags)
    """
    # 0. Calculate the ratio of input vs gt brightness
    input_exposure = float(input_arw.split('_')[-1].replace('s.ARW', ''))
    gt_exposure = float(gt_path.split('_')[-1].replace('s.ARW', ''))
    exposure_ratio = input_exposure / gt_exposure if gt_exposure != 0 else 1

    # 1. Read and modify Bayer
    with rawpy.imread(input_arw) as raw:
        original_bayer = raw.raw_image.copy()
        white_level = raw.white_level

    # Read the ground truth image
    with rawpy.imread(gt_path) as gt_raw:
        gt_bayer = gt_raw.raw_image.copy()

    # Modify and apply motion blur
    modified_bayer = original_bayer.astype(np.float32)
    
    gt_bayer = gt_bayer.astype(np.float32)
    
    # Demosaic, add blur
    demosaiced = demosaic(modified_bayer)
    demosaiced_gt = demosaic(gt_bayer)

    signal = np.clip(demosaiced_gt * exposure_ratio, 0, white_level)
    noise = demosaiced - signal
    
    kernel = Kernel(size=kernel_size, intensity=intensity)
    demosaiced = kernel.applyTo(signal, keep_image_dim=True) + noise

    # Convert back to Bayer
    modified_bayer[0::2, 0::2] = demosaiced[0::2, 0::2, 0]  # R
    modified_bayer[0::2, 1::2] = demosaiced[0::2, 1::2, 1]  # G
    modified_bayer[1::2, 0::2] = demosaiced[1::2, 0::2, 1]  # G
    modified_bayer[1::2, 1::2] = demosaiced[1::2, 1::2, 2]  # B
    
    modified_bayer = np.clip(modified_bayer, 0, white_level).astype(np.uint16)

    # 2. Save modified Bayer as TIFF
    tifffile.imwrite(
        output_tiff,
        modified_bayer,
        photometric='CFA',
        compression='none',
        planarconfig='contig'
    )
    
    # 3. Copy metadata EXCLUDING structural tags
    result = subprocess.run([
        'exiftool',
        '-TagsFromFile', input_arw,
        '-all:all',
        '-overwrite_original',
        output_tiff
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: {result.stderr}")
    
    # If blur_gt is set, also save blurred GT
    if blur_gt and gt_output_dir is not None:
        gt_output_path = os.path.join(gt_output_dir, os.path.basename(gt_path).replace('.ARW', '.tiff'))
        # Skip if already saved
        if os.path.exists(gt_output_path):
            return
        demosaiced_gt_blurred = kernel.applyTo(demosaiced_gt, keep_image_dim=True)
        gt_bayer[0::2, 0::2] = demosaiced_gt_blurred[0::2, 0::2, 0]  # R
        gt_bayer[0::2, 1::2] = demosaiced_gt_blurred[0::2, 1::2, 1]  # G
        gt_bayer[1::2, 0::2] = demosaiced_gt_blurred[1::2, 0::2, 1]  # G
        gt_bayer[1::2, 1::2] = demosaiced_gt_blurred[1::2, 1::2, 2]  # B
        gt_bayer = np.clip(gt_bayer, 0, white_level).astype(np.uint16)

        tifffile.imwrite(
            gt_output_path,
            gt_bayer,
            photometric='CFA',
            compression='none',
            planarconfig='contig'
        )
        
        # Copy EXIF for GT as well
        subprocess.run([
            'exiftool',
            '-TagsFromFile', gt_path,
            '-all:all',
            '-overwrite_original',
            gt_output_path
        ], capture_output=True, text=True)
    
    # 4. Verify
    # try:
    #     with rawpy.imread(output_tiff) as raw:
    #         return raw.raw_image
    # except Exception as e:
    #     print(f"Error reading output: {e}")
    #     return None

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Synthesize modified RAW with motion blur and preserve EXIF metadata.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input .ARW files')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing ground truth .ARW files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output TIFF files')
    parser.add_argument('--kernel_size_min', type=int, default=30, help='Minimum kernel size for motion blur')
    parser.add_argument('--kernel_size_max', type=int, default=80, help='Maximum kernel size for motion blur')
    parser.add_argument('--intensity', type=float, default=0.5, help='Intensity of motion blur')
    parser.add_argument('--blur_gt', action='store_true', help='Whether to apply motion blur to ground truth as well')
    parser.add_argument('--gt_output_dir', type=str, default=None, help='Directory to save blurred ground truth TIFF files (if --blur_gt is set)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    input_dir = args.input_dir
    gt_dir = args.gt_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.arw')]
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith('.arw')]
    gt_dict = {f.split('_')[0] if '_' in f else os.path.splitext(f)[0]: f for f in gt_files}

    if args.blur_gt and args.gt_output_dir is not None:
        os.makedirs(args.gt_output_dir, exist_ok=True)

    for filename in tqdm(files, desc="Processing files"):
        input_path = os.path.join(input_dir, filename)
        # Find corresponding GT file - it is the only one that has the same prefix before underscore
        prefix = filename.split('_')[0] if '_' in filename else os.path.splitext(filename)[0]
        if prefix not in gt_dict:
            print(f"Warning: No matching GT file for {filename}, skipping.")
            continue
        gt_path = os.path.join(gt_dir, gt_dict[prefix])
        output_path = os.path.join(output_dir, filename.replace('.ARW', '.tiff'))
        
        # Randomly sample kernel size for each image
        k_size = random.randint(args.kernel_size_min, args.kernel_size_max)
        kernel_size = (k_size, k_size)

        save_modified_raw_with_exif(input_path, gt_path, output_path, kernel_size=kernel_size, intensity=args.intensity, blur_gt=args.blur_gt, gt_output_dir=args.gt_output_dir)

