from .dataset_reader.dataset_LOLBlur import main_dataset_lolblur
from .dataset_reader.dataset_all_LOL import main_dataset_all_lol
from .dataset_reader.dataset_real_LSRW import main_dataset_real_LSRW
from .dataset_reader.dataset_realblur_night import main_dataset_realblur_night
from .dataset_reader.dataset_dicm import main_dataset_dicm
from .dataset_reader.dataset_lime import main_dataset_lime
from .dataset_reader.dataset_mef import main_dataset_mef
from .dataset_reader.dataset_npe import main_dataset_npe
from .dataset_reader.dataset_vv import main_dataset_vv
from .dataset_reader.dataset_exdark import main_dataset_exdark
import os

def create_test_data(rank, world_size, opt):
    '''
    opt: a dictionary from the yaml config key datasets 
    '''
    name = opt['name']
    test_path = opt['val']['test_path']
    batch_size_test=opt['val']['batch_size_test']
    verbose=opt['train']['verbose']
    num_workers=opt['train']['n_workers']
    
    if rank != 0:
        verbose = False
    samplers = None # TEmporal change!!
    if name == 'LOLBlur':
        test_loader, samplers = main_dataset_lolblur(rank = rank,
                                                test_path = test_path,
                                                batch_size_test=batch_size_test,
                                                verbose=verbose,
                                                num_workers=num_workers,
                                                world_size = world_size) 
    elif name == 'All_LOL':
        test_loader, samplers = main_dataset_all_lol(rank=rank, 
                                                test_path = test_path,
                                                batch_size_test=batch_size_test,
                                                verbose=verbose,
                                                num_workers=num_workers,
                                                world_size=world_size)   
    elif name == 'real_LSRW':
        test_loader, samplers = main_dataset_real_LSRW(rank=rank, 
                                                test_path = test_path,
                                                batch_size_test=batch_size_test,
                                                verbose=verbose,
                                                num_workers=num_workers,
                                                world_size=world_size)  
    elif name == 'RealBlur_Night':
        test_loader, samplers = main_dataset_realblur_night(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'DICM':
        test_loader, samplers = main_dataset_dicm(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'MEF':
        test_loader, samplers = main_dataset_mef(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'NPE':
        test_loader, samplers = main_dataset_npe(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'VV':
        test_loader, samplers = main_dataset_vv(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'LIME':
        test_loader, samplers = main_dataset_lime(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)

    elif name == 'ExDark':
        test_loader, samplers = main_dataset_exdark(rank = 1,
                                                test_path=test_path,
                                                batch_size_test=1, 
                                                verbose=False, 
                                                num_workers=1, 
                                                world_size = 1)
    elif name == 'Custom':
        # Handle custom dataset similar to training data
        from .dataset_reader.datapipeline import MyDataset_Crop, MyDataset_Crop_EXIF
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader, DistributedSampler
        
        # Get image paths
        low_dir = os.path.join(test_path, 'low')
        high_dir = os.path.join(test_path, 'high')
        
        use_exif = opt['val'].get('use_exif', False)
        exif_dir = opt['val'].get('exif_path', None)
        target_8bit = opt['val'].get('target_8bit', True)
        
        if not os.path.exists(low_dir) or not os.path.exists(high_dir):
            raise ValueError(f"Custom dataset requires 'low' and 'high' subdirectories in {test_path}")
        
        low_files = sorted([f for f in os.listdir(low_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.arw', '.tiff', '.tif'))])
        high_files = sorted([f for f in os.listdir(high_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.arw', '.tiff', '.tif'))])
        
        # Create mapping of high images by their prefix
        high_dict = {}
        for high_file in high_files:
            prefix = high_file.split('_')[0] if '_' in high_file else os.path.splitext(high_file)[0]
            high_dict[prefix] = os.path.join(high_dir, high_file)
        
        # Match low images to high images based on prefix
        paths_low_test = []
        paths_high_test = []
        paths_exif_test = []
        
        for low_file in low_files:
            prefix = low_file.split('_')[0] if '_' in low_file else os.path.splitext(low_file)[0]
            if prefix in high_dict:
                paths_low_test.append(os.path.join(low_dir, low_file))
                paths_high_test.append(high_dict[prefix])
                
                if use_exif and exif_dir:
                    low_filename_no_ext = os.path.splitext(low_file)[0]
                    exif_filename = f"{low_filename_no_ext}_exif.json"
                    paths_exif_test.append(os.path.join(exif_dir, exif_filename))
        
        if verbose:
            print(f'Test images found: {len(paths_low_test)} low-light, {len(high_files)} unique normal-light')
            print(f'Successfully paired: {len(paths_low_test)} test pairs')
        
        # Create dataset
        tensor_transform = transforms.ToTensor()
        
        if use_exif and exif_dir:
            test_dataset = MyDataset_Crop_EXIF(
                paths_low_test, 
                paths_high_test, 
                paths_exif_test,
                cropsize=None,
                tensor_transform=tensor_transform, 
                test=True,
                crop_type='Center',
                target_8bit=target_8bit
            )
        else:
            test_dataset = MyDataset_Crop(
                paths_low_test, 
                paths_high_test, 
                cropsize=None,
                tensor_transform=tensor_transform, 
                test=True,
                crop_type='Center',
                target_8bit=target_8bit
            )
        
        if world_size > 1:
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle=False, rank=rank)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=batch_size_test, 
                shuffle=False,
                num_workers=num_workers, 
                pin_memory=False, 
                drop_last=False,
                sampler=test_sampler
            )
            samplers = [test_sampler]
        else:
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=batch_size_test, 
                shuffle=False,
                num_workers=num_workers, 
                pin_memory=False, 
                drop_last=False
            )
            samplers = None

    else:
        raise NotImplementedError(f'{name} is not implemented')        
    if rank ==0: print(f'Using {name} Dataset')
    
    return test_loader, samplers


__all__ = ['create_test_data']