import subprocess
import json
import argparse
import os
from tqdm import tqdm

# For each file in directory, do exiftool -json -a --b 
def extract_exif_to_json(input_file, output_json):
    """
    Extract EXIF metadata from an image file and save it as a JSON file.
    """
    with open(input_file, 'rb') as f:
        result = subprocess.run(['exiftool', '-json', '-a', '--b', input_file], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error extracting EXIF data: {result.stderr}")
            return

        exif_data = json.loads(result.stdout)[0]

    with open(output_json, 'w') as json_file:
        json.dump(exif_data, json_file, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract EXIF metadata from an image file and save it as a JSON file.")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Input directory containing image files.")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Output directory to save JSON files.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)

    for filename in tqdm(files, desc="Extracting EXIF metadata"):
        if filename.lower().endswith(('.arw', '.jpg', '.jpeg', '.tiff', '.png')):
            input_file = os.path.join(input_dir, filename)
            output_json = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_exif.json")
            extract_exif_to_json(input_file, output_json)