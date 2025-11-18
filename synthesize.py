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
    lpf = gaussian(image, sigma=1)
    low_freq_component = lpf
    high_freq_component = image - lpf
    blurred_image = kernel.applyTo(low_freq_component, keep_image_dim=True) + high_freq_component
    return blurred_image
    

def save_modified_raw_with_exif(input_arw, output_tiff, kernel_size=(70, 70), intensity=0.5):
    """
    Save modified Bayer with metadata preserved (excluding structural tags)
    """
    
    # 1. Read and modify Bayer
    with rawpy.imread(input_arw) as raw:
        original_bayer = raw.raw_image.copy()
        white_level = raw.white_level
        
        # Modify and apply motion blur
        modified_bayer = original_bayer.copy()
        modified_bayer = modified_bayer.astype(np.float32)
        
        # Demosaic, add blur
        demosaiced = demosaic(modified_bayer)
        demosaiced = apply_motion_blur(demosaiced, kernel_size=kernel_size, intensity=intensity)

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
    
    # 4. Verify
    try:
        with rawpy.imread(output_tiff) as raw:
            return raw.raw_image
    except Exception as e:
        print(f"Error reading output: {e}")
        return None

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Synthesize modified RAW with motion blur and preserve EXIF metadata.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input .ARW files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output TIFF files')
    parser.add_argument('--kernel_size', type=int, nargs=2, default=(120, 120), help='Maximum kernel size for motion blur')
    parser.add_argument('--intensity', type=float, default=0.5, help='Intensity of motion blur')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.arw')]
    for filename in tqdm(files, desc="Processing files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.ARW', '.tiff'))
        save_modified_raw_with_exif(input_path, output_path, kernel_size=args.kernel_size, intensity=args.intensity)

