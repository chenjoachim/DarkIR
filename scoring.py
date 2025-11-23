import torch
import argparse
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from losses import SSIM
from lpips import LPIPS
from tqdm import tqdm
import numpy as np
import re
from data.dataset_reader.datapipeline import load_image as load_image_pipeline

def calculate_metrics(pred_tensor, gt_tensor, calc_SSIM, calc_LPIPS):
    """Calculate PSNR, SSIM, LPIPS using pre-initialized models"""
    with torch.no_grad():
        # PSNR calculation (same as in test_utils.py)
        mse = torch.mean((gt_tensor - pred_tensor)**2)
        psnr = 20 * torch.log10(1. / torch.sqrt(mse))
        
        # SSIM calculation  
        ssim = calc_SSIM(pred_tensor, gt_tensor)
        
        # LPIPS calculation
        # Downsample 2x if larger than 1024 in any dimension
        if max(pred_tensor.shape[2], pred_tensor.shape[3]) > 1024:
            pred_tensor = torch.nn.functional.interpolate(pred_tensor, scale_factor=0.5, mode='bilinear', align_corners=False)
            gt_tensor = torch.nn.functional.interpolate(gt_tensor, scale_factor=0.5, mode='bilinear', align_corners=False)
        lpips = calc_LPIPS(pred_tensor, gt_tensor)
    
    return psnr.item(), ssim.item(), torch.mean(lpips).item()

def load_image(image_path, device):
    """Load and convert image to tensor"""
    tensor = load_image_pipeline(image_path)
    return tensor.unsqueeze(0).to(device)

def find_gt_by_prefix(pred_file, gt_files):
    """Find GT file with the same prefix number as pred_file"""
    # Extract prefix number from pred_file (e.g., "10101" from "10101_03_0.1s.png")
    pred_match = re.match(r'^(\d+)', pred_file)
    if not pred_match:
        return None
    
    pred_prefix = pred_match.group(1)
    
    # Find GT file with same prefix
    for gt_file in gt_files:
        gt_match = re.match(r'^(\d+)', gt_file)
        if gt_match and gt_match.group(1) == pred_prefix:
            return gt_file
    
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate PSNR, SSIM, LPIPS metrics")
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory with predicted images')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with ground truth images')
    parser.add_argument('--output', type=str, default='metrics.csv', help='Output CSV file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models ONCE outside the loop
    calc_SSIM = SSIM(data_range=1.)
    calc_LPIPS = LPIPS(net='vgg', verbose=False).to(device)
    calc_LPIPS.eval()
    
    # Get image pairs using prefix matching
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.arw', '.tiff', '.tif'))])
    gt_files = [f for f in os.listdir(args.gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.arw', '.tiff', '.tif'))]
    
    pairs = []
    for pred_file in pred_files:
        gt_file = find_gt_by_prefix(pred_file, gt_files)
        if gt_file:
            pairs.append((pred_file, gt_file))
        else:
            print(f"Warning: No GT file found for {pred_file}")
    
    print(f"Found {len(pairs)} image pairs")
    
    # Calculate metrics
    results = []
    for pred_name, gt_name in tqdm(pairs, desc="Computing metrics"):
        try:
            pred_path = os.path.join(args.pred_dir, pred_name)
            gt_path = os.path.join(args.gt_dir, gt_name)
            
            pred_img = load_image(pred_path, device)
            gt_img = load_image(gt_path, device)
            
            # Resize if needed
            if pred_img.shape != gt_img.shape:
                pred_img = torch.nn.functional.interpolate(pred_img, size=gt_img.shape[2:], mode='bilinear', align_corners=False)
            
            psnr, ssim, lpips = calculate_metrics(pred_img, gt_img, calc_SSIM, calc_LPIPS)
            
            results.append({
                'pred_image': pred_name,
                'target_image': gt_name,
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips
            })
            
            # Clear GPU cache after each image to prevent accumulation
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {pred_name}: {e}")
            raise e
    
    # Save and display results
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    
    # Summary
    valid = df_results.dropna()
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Processed: {len(valid)}/{len(pairs)} pairs")
    print(f"PSNR: {valid['psnr'].mean():.4f} ± {valid['psnr'].std():.4f}")
    print(f"SSIM: {valid['ssim'].mean():.4f} ± {valid['ssim'].std():.4f}") 
    print(f"LPIPS: {valid['lpips'].mean():.4f} ± {valid['lpips'].std():.4f}")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
