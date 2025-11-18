import numpy as np
import os, sys
from tqdm import tqdm
from options.options import parse
import argparse

parser = argparse.ArgumentParser(description="Script for finetuning DarkIR")
parser.add_argument('-p', '--config', type=str, default='./options/finetune/finetune_default.yml', help='Config file for finetuning')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
args = parser.parse_args()

# Read options file
path_options = args.config
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.get('gpu', "0")

# PyTorch library
import torch
import torch.optim
import torch.multiprocessing as mp
import torch.distributed as dist

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.utils import create_path_models, init_wandb, logging_dict
from utils.test_utils import *
from ptflops import get_model_complexity_info

# Parameters for saving model
PATH_MODEL = create_path_models(opt['save'])


def load_pretrained_model(model, path_weights, rank):
    """
    Load pretrained weights from checkpoint
    """
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
    
    # Handle different checkpoint formats
    if 'params' in checkpoints:
        weights = checkpoints['params']
    elif 'model_state_dict' in checkpoints:
        weights = checkpoints['model_state_dict']
    else:
        weights = checkpoints
    
    # Remove 'module.' prefix if present
    new_weights = {}
    for k, v in weights.items():
        if k.startswith('module.'):
            new_weights[k[7:]] = v
        else:
            new_weights[k] = v
    
    # Load weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_weights.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    if rank == 0:
        print(f'Loaded pretrained weights from {path_weights}')
        print(f'Loaded {len(pretrained_dict)}/{len(model_dict)} parameters')
    
    return model


def create_train_data(rank, world_size, opt):
    """
    Create training data loader
    Handles many-to-one mapping where multiple low images share the same high image
    by matching based on prefix numbers (e.g., 10101_03_0.1s.png -> 10101.png)
    """
    name = opt['name']
    train_path = opt['train']['train_path']
    batch_size = opt['train']['batch_size']
    cropsize = opt['train']['cropsize']
    verbose = opt['train']['verbose']
    num_workers = opt['train']['n_workers']
    
    if rank != 0:
        verbose = False
    
    # Get image paths
    low_dir = os.path.join(train_path, 'low')
    high_dir = os.path.join(train_path, 'high')
    
    low_files = sorted([f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    high_files = sorted([f for f in os.listdir(high_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    # Create mapping of high images by their prefix
    high_dict = {}
    for high_file in high_files:
        # Extract prefix (everything before first underscore, or full name without extension)
        prefix = high_file.split('_')[0] if '_' in high_file else os.path.splitext(high_file)[0]
        high_dict[prefix] = os.path.join(high_dir, high_file)
    
    # Match low images to high images based on prefix
    paths_low_train = []
    paths_high_train = []
    unmatched_count = 0
    
    for low_file in low_files:
        # Extract prefix from low image
        prefix = low_file.split('_')[0] if '_' in low_file else os.path.splitext(low_file)[0]
        
        if prefix in high_dict:
            paths_low_train.append(os.path.join(low_dir, low_file))
            paths_high_train.append(high_dict[prefix])
        else:
            unmatched_count += 1
            if verbose and unmatched_count <= 5:
                print(f'Warning: No matching high image found for {low_file} (prefix: {prefix})')
    
    if verbose:
        print(f'Training images found: {len(paths_low_train)} low-light, {len(high_files)} unique normal-light')
        print(f'Successfully paired: {len(paths_low_train)} pairs')
        if unmatched_count > 0:
            print(f'Unmatched low images: {unmatched_count}')
    
    # Create dataset with augmentation
    tensor_transform = transforms.ToTensor()
    flips = transforms.RandomHorizontalFlip(p=0.5)
    
    train_dataset = MyDataset_Crop(
        paths_low_train, 
        paths_high_train, 
        cropsize=cropsize,
        tensor_transform=tensor_transform, 
        flips=flips,
        test=False,
        crop_type='Random'
    )
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle=True, rank=rank)
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=True, 
            sampler=train_sampler
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=True
        )
    
    if rank == 0:
        print(f'Using {name} Dataset for training')
    
    return train_loader, train_sampler


def create_losses(opt, rank):
    """
    Create loss functions based on config
    """
    losses_dict = {}
    
    for loss_name, loss_config in opt.items():
        loss_type = loss_config['type']
        loss_weight = loss_config.get('weight', 1.0)
        
        if loss_type == 'L1Loss':
            losses_dict[loss_name] = L1Loss(loss_weight=loss_weight)
        elif loss_type == 'MSELoss':
            losses_dict[loss_name] = MSELoss(loss_weight=loss_weight)
        elif loss_type == 'CharbonnierLoss':
            losses_dict[loss_name] = CharbonnierLoss(loss_weight=loss_weight)
        elif loss_type == 'SSIMloss':
            losses_dict[loss_name] = SSIMloss(loss_weight=loss_weight)
        elif loss_type == 'VGGLoss':
            losses_dict[loss_name] = VGGLoss(loss_weight=loss_weight)
        elif loss_type == 'FrequencyLoss':
            losses_dict[loss_name] = FrequencyLoss(loss_weight=loss_weight)
        elif loss_type == 'EdgeLoss':
            losses_dict[loss_name] = EdgeLoss(rank=rank, loss_weight=loss_weight)
        elif loss_type == 'EnhanceLoss':
            losses_dict[loss_name] = EnhanceLoss(loss_weight=loss_weight)
        else:
            raise NotImplementedError(f'Loss {loss_type} not implemented')
    
    return losses_dict


def train_one_epoch(model, train_loader, optimizer, losses_dict, rank, world_size, use_side_loss=False):
    """
    Train for one epoch
    """
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    
    if rank == 0:
        pbar = tqdm(total=num_batches, desc='Training')
    
    for i, (high_batch, low_batch) in enumerate(train_loader):
        high_batch = high_batch.to(rank)
        low_batch = low_batch.to(rank)
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_side_loss:
            out_side, output = model(low_batch, side_loss=True)
        else:
            output = model(low_batch)
        
        # Calculate losses
        total_loss = 0
        for loss_name, loss_fn in losses_dict.items():
            if 'enhance' in loss_name.lower() and use_side_loss:
                # EnhanceLoss uses the side output
                loss = loss_fn(high_batch, out_side)
            else:
                loss = loss_fn(output, high_batch)
            total_loss += loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Record loss
        epoch_loss += total_loss.item()
        
        if rank == 0:
            pbar.update(1)
            pbar.set_postfix({'loss': total_loss.item()})
    
    if rank == 0:
        pbar.close()
    
    # Average loss across all GPUs
    avg_loss = epoch_loss / num_batches
    avg_loss_tensor = reduce_tensor(torch.tensor(avg_loss).to(rank), world_size)
    
    return avg_loss_tensor.item()


def run_finetuning(rank, world_size):
    """
    Main finetuning function
    """
    setup(rank, world_size=world_size)
    
    # Initialize wandb
    init_wandb(rank, opt)
    
    # Load dataloaders
    train_loader, train_sampler = create_train_data(rank, world_size=world_size, opt=opt['datasets'])
    test_loader, test_sampler = create_test_data(rank, world_size=world_size, opt=opt['datasets'])
    
    # Define network
    model, macs, params = create_model(opt['network'], rank=rank)
    
    # Load pretrained weights
    if opt['network'].get('pretrained_path'):
        model = load_pretrained_model(model.module, opt['network']['pretrained_path'], rank)
        # Re-wrap with DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(rank), 
            device_ids=[rank], 
            find_unused_parameters=False
        )
    
    # Define optimizer and scheduler
    optimizer, scheduler = create_optim_scheduler(opt['train'], model)
    
    # Resume from checkpoint if specified
    if args.resume:
        model, optimizer, scheduler, start_epoch = resume_model(
            model, optimizer, scheduler, args.resume, rank, resume=True
        )
    else:
        start_epoch = 0
    
    # Define losses
    losses_dict = create_losses(opt['losses'], rank)
    
    # Move losses to device
    for loss_fn in losses_dict.values():
        loss_fn.to(rank)
    
    dist.barrier()
    
    # Training metrics
    best_psnr = 0.0
    metrics_eval = {}
    use_side_loss = opt['train'].get('use_side_loss', False)
    
    if rank == 0:
        print(f'\n{"="*50}')
        print(f'Starting finetuning from epoch {start_epoch}')
        print(f'Total epochs: {opt["train"]["epochs"]}')
        print(f'Use side loss: {use_side_loss}')
        print(f'{"="*50}\n')
    
    # Training loop
    for epoch in range(start_epoch, opt['train']['epochs']):
        if rank == 0:
            print(f'\nEpoch [{epoch+1}/{opt["train"]["epochs"]}]')
        
        # Shuffle data
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if test_sampler:
            shuffle_sampler([test_sampler], epoch)
        
        # Train one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, losses_dict, 
            rank, world_size, use_side_loss
        )
        
        if rank == 0:
            print(f'Training Loss: {train_loss:.4f}')
        
        # Validation
        model.eval()
        metrics_eval, imgs_dict = eval_model(
            model, test_loader, metrics_eval, 
            rank=rank, world_size=world_size, eta=(rank==0)
        )
        
        dist.barrier()
        
        # Print validation results
        if rank == 0:
            if type(next(iter(metrics_eval.values()))) == dict:
                for key, metric_eval in metrics_eval.items():
                    print(f'  {key} --- PSNR: {metric_eval["valid_psnr"]:.2f}, '
                          f'SSIM: {metric_eval["valid_ssim"]:.4f}, '
                          f'LPIPS: {metric_eval["valid_lpips"]:.4f}')
            else:
                print(f'  Validation --- PSNR: {metrics_eval["valid_psnr"]:.2f}, '
                      f'SSIM: {metrics_eval["valid_ssim"]:.4f}, '
                      f'LPIPS: {metrics_eval["valid_lpips"]:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if rank == 0:
            metrics_train = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'best_psnr': best_psnr
            }
            
            # Prepare save paths
            paths = {
                'new': os.path.join(PATH_MODEL, f'epoch_{epoch+1}.pt'),
                'best': os.path.join(PATH_MODEL, 'best_model.pt')
            }
            
            # Create directory if needed
            os.makedirs(PATH_MODEL, exist_ok=True)
            
            # Save checkpoint
            best_psnr = save_checkpoint(
                model, optimizer, scheduler, 
                metrics_eval, metrics_train, paths, 
                adapter=False, rank=rank
            )
            
            print(f'Model saved to {paths["new"]}')
            
            # Log to wandb
            if opt.get('wandb', {}).get('init', False):
                import wandb
                logger = logging_dict(metrics_train, metrics_eval, imgs_dict)
                wandb.log(logger)
    
    if rank == 0:
        print(f'\n{"="*50}')
        print('Finetuning completed!')
        print(f'Best PSNR: {best_psnr:.2f}')
        print(f'{"="*50}\n')
    
    cleanup()


def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f'Using {world_size} GPUs for distributed training')
        mp.spawn(run_finetuning, args=(world_size,), nprocs=world_size, join=True)
    else:
        print('Using single GPU training')
        run_finetuning(0, 1)


if __name__ == '__main__':
    main()
