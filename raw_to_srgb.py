import rawpy
import imageio
import argparse
import os
import glob
from tqdm import tqdm

def raw_to_srgb(raw_image_path, output_image_path):
    """
    Convert a RAW image to sRGB format and save it.

    Parameters:
    - raw_image_path: str, path to the input RAW image file.
    - output_image_path: str, path to save the output sRGB image file.
    """
    # Read the RAW image using rawpy
    with rawpy.imread(raw_image_path) as raw:
        # Post-process the RAW image to get an sRGB image
        rgb_image = raw.postprocess()

    # Save the sRGB image using imageio
    imageio.imsave(output_image_path, rgb_image)

def parser_args():
    parser = argparse.ArgumentParser(description="Convert RAW/TIFF images to sRGB format")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Path to the input directory containing RAW/TIFF images')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to save the output sRGB images')
    return parser.parse_args()

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all .ARW and .TIFF files in the input directory
    arw_files = glob.glob(os.path.join(input_dir, "*.ARW")) + glob.glob(os.path.join(input_dir, "*.arw"))
    tiff_files = glob.glob(os.path.join(input_dir, "*.TIFF")) + glob.glob(os.path.join(input_dir, "*.tiff"))
    raw_files = arw_files + tiff_files

    if not raw_files:
        print(f"No .ARW or .TIFF files found in {input_dir}")
        return

    print(f"Found {len(raw_files)} files to process")

    # Process each file
    for raw_file_path in tqdm(raw_files, desc="Converting images"):
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(raw_file_path))[0]
        
        # Create output path with .png extension
        output_path = os.path.join(output_dir, f"{filename}.png")
        
        try:
            raw_to_srgb(raw_file_path, output_path)
            # print(f"Successfully converted: {os.path.basename(raw_file_path)} -> {os.path.basename(output_path)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(raw_file_path)}: {str(e)}")

    print(f"Conversion complete! Output files saved to: {output_dir}")

if __name__ == "__main__":
    args = parser_args()
    main(args)