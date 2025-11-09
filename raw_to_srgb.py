import rawpy
import imageio
import argparse
import os
from tqdm import tqdm
import csv

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
    parser = argparse.ArgumentParser(description="Convert RAW image to sRGB format")
    parser.add_argument('-r', '--raw_image_dir', type=str, required=True, help='Path to the input RAW image directory')
    parser.add_argument('-o', '--output_image_dir', type=str, required=True, help='Path to save the output sRGB image directory')
    parser.add_argument('--list_file', type=str, required=True, help='Path to the text file containing list of RAW image filenames')
    return parser.parse_args()

def main(args):
    raw_image_dir = args.raw_image_dir
    output_image_dir = args.output_image_dir
    list_file = args.list_file

    os.makedirs(os.path.join(output_image_dir, "short"), exist_ok=True)
    os.makedirs(os.path.join(output_image_dir, "long"), exist_ok=True)

    raw_image_files = []
    with open(list_file, 'r') as f:
        for line in f:
            raw_image_files += line.strip().split()[:2]

    image_dict = {}
    image_dict['pred_image'] = []
    image_dict['target_image'] = []

    for raw_image_file in tqdm(raw_image_files):
        raw_image_path = os.path.join(raw_image_dir, raw_image_file)
        output_image_path = os.path.join(output_image_dir, raw_image_path.split('Sony/')[-1].replace('.ARW', '.png'))
        raw_to_srgb(raw_image_path, output_image_path)

        # Add to image dictionary
        if 'short' in raw_image_file:
            image_dict['pred_image'].append(output_image_path.split('/')[-1])
        else:
            image_dict['target_image'].append(output_image_path.split('/')[-1])

    # Write image pairs to CSV
    with open(os.path.join(output_image_dir, 'file_list.csv'), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['pred_image', 'target_image'])
        for pred, target in zip(image_dict['pred_image'], image_dict['target_image']):
            csvwriter.writerow([pred, target])

if __name__ == "__main__":
    args = parser_args()
    main(args)