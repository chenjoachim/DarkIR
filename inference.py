import os
from PIL import Image
import cv2 as cv
from options.options import parse
import argparse
from archs.retinexformer import RetinexFormer
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description="Script for prediction")
parser.add_argument('-p', '--config', type=str, default='./options/inference/default.yml', help = 'Config file of prediction')
parser.add_argument('-i', '--inp_path', type=str, default='./images/inputs', 
                help="Folder path")
parser.add_argument('-o', '--out_path', type=str, default='./images/results', 
                help="Folder path")
parser.add_argument('--use_exif', action='store_true', help='Use EXIF data for inference')
parser.add_argument('--exif_path', type=str, default=None, help='Path to EXIF json files')
args = parser.parse_args()


path_options = args.config
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# PyTorch library
import torch
import torch.optim
import torch.multiprocessing as mp
from tqdm import tqdm
from torchvision.transforms import Resize

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.test_utils import *
from ptflops import get_model_complexity_info

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

#define some auxiliary functions
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

def path_to_tensor(path, exif_raw_data=None):
    img = load_image(path, exif_raw_data)
    return img.unsqueeze(0)

def load_exif(path):
    with open(path, 'r') as f:
        exif_raw_data = json.load(f)
    
    exif_data = exif_transform(exif_raw_data)
    return exif_data.unsqueeze(0) # Add batch dimension

def normalize_tensor(tensor):
    
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value)/(max_value)
    return output

def save_tensor(tensor, path):
    
    tensor = tensor.squeeze(0)
    print(tensor.shape, tensor.dtype, torch.max(tensor), torch.min(tensor))
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple = 8):
    '''pad the tensor to be multiple of some number'''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

def load_model(model, path_weights):
    map_location = 'cpu'
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)

    if 'params' in checkpoints:
        weights = checkpoints['params']
        weights = {'module.' + key: value for key, value in weights.items()}
    else:
        weights = checkpoints['model_state_dict']
    
    # Filter out keys that don't match (e.g. if loading non-vec model into vec model)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in weights.items() if k in model_dict}
    
    # Check if we are missing keys related to vec_proj
    missing_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys missing in checkpoint. This is expected if initializing new layers (e.g. vec_proj).")
        # print(f"Missing keys: {missing_keys}")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
    print(macs, params)
    print('Loaded weights correctly')
    
    return model

#parameters for saving model
PATH_MODEL = opt['save']['path']
resize = opt['Resize']

def predict_folder(rank, world_size):
    
    setup(rank, world_size=world_size, Master_port='12354')
    
    if args.use_exif:
        opt['network']['vec_dim'] = 6

    # DEFINE NETWORK, SCHEDULER AND OPTIMIZER
    model, _, _ = create_model(opt['network'], rank=rank)

    model = load_model(model, path_weights = opt['save']['path'])
    # create data
    PATH_IMAGES= args.inp_path
    PATH_RESULTS = args.out_path

    #create folder if it doen't exist
    os.makedirs(PATH_RESULTS, exist_ok=True)

    path_images = [os.path.join(PATH_IMAGES, path) for path in os.listdir(PATH_IMAGES) if path.lower().endswith(('.png', '.jpg', '.jpeg', '.arw', '.tiff', '.tif'))]
    path_images = [file for file in path_images if not file.endswith('.csv') and not file.endswith('.txt')]
   
    model.eval()
    if rank==0:
        pbar = tqdm(total = len(path_images))
        
    for path_img in path_images:
        exif_data = None
        exif_raw_data = None
        if args.use_exif and args.exif_path:
            filename_no_ext = os.path.splitext(os.path.basename(path_img))[0]
            # Handle potential prefix issues if needed, similar to training
            # For now assume direct mapping or simple prefix
            
            # Try direct match first
            exif_file = os.path.join(args.exif_path, f"{filename_no_ext}_exif.json")
            if not os.path.exists(exif_file):
                 # Try splitting by underscore
                 prefix = filename_no_ext.split('_')[0]
                 exif_file = os.path.join(args.exif_path, f"{prefix}_exif.json")
            
            if os.path.exists(exif_file):
                with open(exif_file, 'r') as f:
                    exif_raw_data = json.load(f)
                exif_data = exif_transform(exif_raw_data).unsqueeze(0).to(device)
            else:
                print(f"Warning: EXIF file not found for {path_img}")

        # Only pass exif_raw_data to image loader if it contains WB info
        # Otherwise fall back to camera WB to avoid default Unit WB (green tint)
        loader_exif = exif_raw_data
        if exif_raw_data and 'WB_RGGBLevels' not in exif_raw_data:
            loader_exif = None

        tensor = path_to_tensor(path_img, exif_raw_data=loader_exif).to(device)
        _, _, H, W = tensor.shape
        
        if resize and (H >=1500 or W>=1500):
            new_size = [int(dim//2) for dim in (H, W)]
            downsample = Resize(new_size)
        else:
            downsample = torch.nn.Identity()
        tensor = downsample(tensor)
        
        tensor = pad_tensor(tensor)

        with torch.no_grad():
            output = model(tensor, side_loss=False, vec=exif_data)
        if resize:
            upsample = Resize((H, W))
        else: upsample = torch.nn.Identity()
        output = upsample(output)
        output = torch.clamp(output, 0., 1.)
        output = output[:,:, :H, :W]
        
        # Save as PNG
        filename = os.path.splitext(os.path.basename(path_img))[0]
        save_tensor(output, os.path.join(PATH_RESULTS, filename + '.png'))


        pbar.update(1)
        pass

    print('Finished inference!')
    if rank == 0:
        pbar.close()   
    cleanup()

def main():
    world_size = 1
    print('Used GPUS:', world_size)
    mp.spawn(predict_folder, args =(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()










