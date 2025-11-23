import numpy as np 
import os
import json
import rawpy
from typing import Optional

from PIL import Image

import wandb
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.nn as nn

def crop_center(img, cropx=224, cropy=256):
    """
    Given an image, it returns a center cropped version of size [cropx,cropy]
    """
    y,x,c  = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def load_image(path: str, exif_raw_data: Optional[dict] = None) -> torch.Tensor:
    if path.lower().endswith(('.arw', '.tiff', '.tif')):
        with rawpy.imread(path) as raw:
            # output_bps=16 returns uint16 array, RGB
            if exif_raw_data is not None:
                wb_level = exif_raw_data.get('WB_RGGBLevels', '1024 1024 1024 1024')
                r_number = float(wb_level.split()[0])
                g_number = float(wb_level.split()[1])
                b_number = float(wb_level.split()[3])
                white_balance = [r_number / g_number, 1.0, b_number / g_number, 1.0]
                img_np = raw.postprocess(output_bps=16, user_wb=white_balance)
            else: 
                img_np = raw.postprocess(output_bps=16, use_camera_wb=True)
        # Normalize 16-bit to 0-1 float
        img_tensor = torch.from_numpy(img_np.astype(np.float32) / 65535.0)
        # HWC to CHW
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor
    else:
        # Standard 8-bit image loading
        img = Image.open(path).convert('RGB')
        return transforms.ToTensor()(img)

class CropTo4(nn.Module):
    '''
    A function that crops an image into 4 patches (top-left, top-right, bottom-left, bottom-right), adn returns a list of the patches.
    '''
    def __init__(self):
        super(CropTo4, self).__init__()
        

    def forward(self, img1, img2):
        
        #pad the images with zeros if their size is lower than the cropsize
        img1 = self.pad(img1)
        # print(img1.shape)
        img2 = self.pad(img2)
        _, _, h, w = img1.shape
        crops1 = [TF.crop(img1, 0, 0, h//2, w//2), TF.crop(img1, 0, w//2, h//2, w//2),
                  TF.crop(img1, h//2, 0, h//2, w//2),TF.crop(img1, h//2, w//2, h//2, w//2)]
        crops2 = [TF.crop(img2, 0, 0, h//2, w//2), TF.crop(img2, 0, w//2, h//2, w//2),
                  TF.crop(img2, h//2, 0, h//2, w//2),TF.crop(img2, h//2, w//2, h//2, w//2)]
            
        return crops1, crops2
        # return torch.cat(crops1), torch.cat(crops2)

    def pad(self, img):
        _, _, h, w = img.shape
        mod_pad_h = (h - h//2) % 2
        mod_pad_w = (w - w//2) % 2
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), mode = 'constant', value = 0)
        
        return img

class RandomCropSame:
    '''
    A function that random crops a pair of images with a fixed size (and the same crop).
    '''
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img1, img2):
        
        #pad the images with zeros if their size is lower than the cropsize
        if img1.shape[1] <= self.size[0] or img1.shape[2] <= self.size[1]:
            img1 = self.pad(img1)
            img2 = self.pad(img2)
        
        i, j, th, tw = self.get_params(img1, self.size)
            
        return TF.crop(img1, i, j, th, tw), TF.crop(img2, i, j, th, tw)  # Use th and tw here

    def get_params(self, img, output_size):
        h, w = img.shape[1], img.shape[2]
        th, tw = output_size
        
        if w <= tw or h <= th:
            return 0, 0, h, w
        
        # Calculate the starting top-left corner (i, j) such that the entire crop is within the image.
        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        
        return i, j, th, tw

    def pad(self, img):
        _, h, w = img.shape
        mod_pad_h = self.size[0] - h
        mod_pad_w = self.size[1] - w
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h))
        
        return img

class MyDataset_Crop(Dataset):
    """
    A Dataset of the low and high light images with data values in each channel in the range 0-1 (normalized).
    """
    
    def __init__(self, images_low, images_high, cropsize = None, tensor_transform = None, flips=None, test=False, crop_type = 'Random'):
        """
        - images_high: list of RGB images of normal-light used for training or testing the model
        - images_low: list of RGB images of low-light used for training or testing the model
        - test: indicates if the dataset is for training (False) or testing (True)
        - image_size: contains the dimension of the final image (H, W, C). This is important
                     to do the propper crop of the image.
        """
        self.imgs_low   = sorted(images_low)
        self.imgs_high  = sorted(images_high)
        self.test       = test
        self.cropsize   = cropsize
        self.to_tensor  = tensor_transform
        self.flips      = flips
        
        if self.cropsize:
            if crop_type   == 'Random':
                
                self.random_crop = RandomCropSame(self.cropsize)
                self.center_crop = None
            elif crop_type == 'Center':
                
                self.center_crop = transforms.CenterCrop(cropsize)  
                self.random_crop = None

    def __len__(self):
        return len(self.imgs_low)

    def __getitem__(self, idx):
        """
        Given a (random) index. The dataloader selects the corresponding image path, and loads the image.
        Then it returns the image, after applying any required transformation.
        """
        
        img_low_path  = self.imgs_low[idx]
        img_high_path = self.imgs_high[idx]

        rgb_low = load_image(img_low_path)
        rgb_high = load_image(img_high_path)

        # stack high and low to do the exact same flip on the two images
        high_and_low = torch.stack((rgb_high, rgb_low))
        if self.flips:
            high_and_low      = self.flips(high_and_low)
            rgb_high, rgb_low = high_and_low #separate again the images
        
        # print(rgb_high.shape, rgb_low.shape)
        if self.cropsize: # do random crops of the image
            if self.random_crop:   
                rgb_high, rgb_low = self.random_crop(rgb_high, rgb_low)
            elif self.center_crop:
                rgb_high, rgb_low = self.center_crop(rgb_high), self.center_crop(rgb_low)
    
        return rgb_high, rgb_low
    
def exif_transform(exif_raw_data):
    exposure_time_val = exif_raw_data.get('ExposureTime', '1/10')
    exposure_time_str = str(exposure_time_val)
    if '/' in exposure_time_str:
        numerator, denominator = exposure_time_str.split('/')
        exposure_time = float(numerator) / float(denominator)
    else:
        try:
            exposure_time = float(exposure_time_str)
        except ValueError:
            exposure_time = 0.1
    
    f_number = float(exif_raw_data.get('FNumber', 4.0))       
    iso = float(exif_raw_data.get('ISO', 100))
    
    focal_length_val = exif_raw_data.get('FocalLength', '0')
    focal_length_str = str(focal_length_val)
    if 'mm' in focal_length_str:
        focal_length = float(focal_length_str.replace('mm', '').strip())
    else:
        try:
            focal_length = float(focal_length_str)
        except ValueError:
            focal_length = 0.0
    
    wb_level_val = exif_raw_data.get('WB_RGGBLevels', '1024 1024 1024 1024')
    if isinstance(wb_level_val, str):
        wb_levels = [float(x) for x in wb_level_val.split()]
    elif isinstance(wb_level_val, list):
        wb_levels = [float(x) for x in wb_level_val]
    else:
        wb_levels = [1024.0, 1024.0, 1024.0, 1024.0]
        
    if len(wb_levels) < 4:
            wb_levels = [1024.0, 1024.0, 1024.0, 1024.0]

    r_g_ratio = wb_levels[0] / wb_levels[1] if wb_levels[1] != 0 else 1.0
    b_g_ratio = wb_levels[3] / wb_levels[1] if wb_levels[1] != 0 else 1.0
    
    # Normalize
    # exposure_time: log scale, mapped roughly to [0, 1] for 1/4000s to 30s
    norm_exposure_time = (np.log10(exposure_time + 1e-5) + 4) / 6 
    
    # f_number: linear scale, mapped [0, 1] for f/1.0 to f/32.0
    norm_f_number = f_number / 32.0
    
    # iso: log scale, mapped [0, 1] for ISO 50 to 102400
    norm_iso = np.log10(iso + 1e-5) / 6.0
    
    # focal_length: linear scale, mapped [0, 1] for 0mm to 500mm
    norm_focal_length = focal_length / 500.0
    
    # r_g_ratio, b_g_ratio: linear scale, mapped [0, 1] for 0 to 5
    norm_r_g_ratio = r_g_ratio / 5.0
    norm_b_g_ratio = b_g_ratio / 5.0

    exif_data = torch.tensor([
        norm_exposure_time,
        norm_f_number,
        norm_iso,
        norm_focal_length,
        norm_r_g_ratio,
        norm_b_g_ratio
    ], dtype=torch.float32)
    
    return exif_data

class MyDataset_Crop_EXIF(MyDataset_Crop):
    def __init__(self, images_low, images_high, exif_path, cropsize=None, tensor_transform=None, flips=None, test=False, crop_type='Random'):
        super().__init__(images_low, images_high, cropsize, tensor_transform, flips, test, crop_type)
        self.exif_path = sorted(exif_path)
        
    def __getitem__(self, idx):
        img_low_path  = self.imgs_low[idx]
        img_high_path = self.imgs_high[idx]
        meta = self.exif_path[idx]
        with open(meta, 'r') as f:
            exif_raw_data = json.load(f)

        rgb_low = load_image(img_low_path, exif_raw_data)
        rgb_high = load_image(img_high_path)
        
        # stack high and low to do the exact same flip on the two images
        high_and_low = torch.stack((rgb_high, rgb_low))
        if self.flips:
            high_and_low      = self.flips(high_and_low)
            rgb_high, rgb_low = high_and_low #separate again the images
        
        # print(rgb_high.shape, rgb_low.shape)
        if self.cropsize: # do random crops of the image
            if self.random_crop:   
                rgb_high, rgb_low = self.random_crop(rgb_high, rgb_low)
            elif self.center_crop:
                rgb_high, rgb_low = self.center_crop(rgb_high), self.center_crop(rgb_low)
        
        exif_data = exif_transform(exif_raw_data)
        return rgb_high, rgb_low, exif_data
if __name__== '__main__':
    tensor = torch.rand([1, 3, 1000, 1000])
    
    crop_to_4 = CropTo4()
    crops1, crops2 = crop_to_4(tensor, tensor)
    
    for crop in crops1:
        print(crop.shape)