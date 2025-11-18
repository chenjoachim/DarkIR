"""
Validation script to check if finetuning setup is ready

This script validates:
1. Required Python packages
2. Configuration files
3. Dataset structure
4. Pretrained model
5. Script files

Run this before starting finetuning to catch issues early.
"""

import os
import sys
from pathlib import Path


def check_imports():
    """Check if all required packages are installed"""
    print("Checking Python packages...")
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'lpips': 'LPIPS',
        'pytorch_msssim': 'pytorch-msssim',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install torch torchvision numpy tqdm lpips pytorch-msssim pillow pyyaml")
        return False
    
    print("✓ All required packages installed\n")
    return True


def check_config_files():
    """Check if configuration files exist"""
    print("Checking configuration files...")
    
    configs = [
        'options/finetune/finetune_default.yml',
        'options/finetune/finetune_lolblur.yml',
        'options/finetune/finetune_minimal.yml',
    ]
    
    all_exist = True
    for config in configs:
        if os.path.exists(config):
            print(f"  ✓ {config}")
        else:
            print(f"  ✗ {config} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("✓ All configuration files present\n")
    else:
        print("⚠ Some configuration files missing\n")
    
    return all_exist


def check_scripts():
    """Check if main scripts exist"""
    print("Checking scripts...")
    
    scripts = [
        'finetune.py',
        'prepare_dataset.py',
        'quick_start_finetune.sh',
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("✓ All scripts present\n")
    else:
        print("⚠ Some scripts missing\n")
    
    return all_exist


def check_dataset(config_path):
    """Check if dataset exists and is properly structured"""
    print(f"Checking dataset (from {config_path})...")
    
    if not os.path.exists(config_path):
        print(f"  ✗ Config file not found: {config_path}\n")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        train_path = config['datasets']['train']['train_path']
        test_path = config['datasets']['val']['test_path']
        
        # Check train data
        train_low = os.path.join(train_path, 'low')
        train_high = os.path.join(train_path, 'high')
        
        if not os.path.exists(train_low):
            print(f"  ✗ Training low-light directory not found: {train_low}")
            print(f"    Create it or update config file")
            return False
        
        if not os.path.exists(train_high):
            print(f"  ✗ Training normal-light directory not found: {train_high}")
            print(f"    Create it or update config file")
            return False
        
        # Count images
        train_low_count = len([f for f in os.listdir(train_low) if f.endswith(('.png', '.jpg', '.jpeg'))])
        train_high_count = len([f for f in os.listdir(train_high) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"  ✓ Train dataset found:")
        print(f"    - {train_low_count} low-light images")
        print(f"    - {train_high_count} normal-light images")
        
        if train_low_count == 0 or train_high_count == 0:
            print(f"  ⚠ Warning: Dataset appears empty!")
            return False
        
        if abs(train_low_count - train_high_count) > 0:
            print(f"  ⚠ Warning: Number of low and high images don't match!")
        
        # Check test data
        if os.path.exists(test_path):
            test_low = os.path.join(test_path, 'low')
            test_high = os.path.join(test_path, 'high')
            
            if os.path.exists(test_low) and os.path.exists(test_high):
                test_low_count = len([f for f in os.listdir(test_low) if f.endswith(('.png', '.jpg', '.jpeg'))])
                test_high_count = len([f for f in os.listdir(test_high) if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  ✓ Test dataset found:")
                print(f"    - {test_low_count} low-light images")
                print(f"    - {test_high_count} normal-light images")
        else:
            print(f"  ⚠ Test dataset not found (optional): {test_path}")
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking dataset: {e}\n")
        return False


def check_pretrained_model(config_path):
    """Check if pretrained model exists"""
    print(f"Checking pretrained model (from {config_path})...")
    
    if not os.path.exists(config_path):
        print(f"  ✗ Config file not found: {config_path}\n")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        pretrained_path = config['network']['pretrained_path']
        
        if os.path.exists(pretrained_path):
            size_mb = os.path.getsize(pretrained_path) / (1024 * 1024)
            print(f"  ✓ Pretrained model found: {pretrained_path}")
            print(f"    Size: {size_mb:.1f} MB")
            print()
            return True
        else:
            print(f"  ✗ Pretrained model not found: {pretrained_path}")
            print(f"    Download the pretrained DarkIR model first")
            print()
            return False
            
    except Exception as e:
        print(f"  ✗ Error checking pretrained model: {e}\n")
        return False


def check_gpu():
    """Check GPU availability"""
    print("Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  ✓ {gpu_count} GPU(s) available")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"    GPU {i}: {gpu_name}")
            print()
            return True
        else:
            print("  ⚠ No GPU detected - training will be very slow")
            print("    Consider using a machine with GPU")
            print()
            return False
    except:
        print("  ✗ Cannot check GPU (torch not installed)")
        print()
        return False


def main():
    print("="*70)
    print("DarkIR Finetuning Setup Validation")
    print("="*70)
    print()
    
    # Check working directory
    if not os.path.exists('finetune.py'):
        print("⚠ Warning: Please run this script from the DarkIR root directory")
        print()
    
    # Run all checks
    checks = {
        'Python Packages': check_imports(),
        'Configuration Files': check_config_files(),
        'Scripts': check_scripts(),
        'GPU': check_gpu(),
    }
    
    # Check dataset and pretrained model for default config
    default_config = 'options/finetune/finetune_default.yml'
    if os.path.exists(default_config):
        checks['Dataset'] = check_dataset(default_config)
        checks['Pretrained Model'] = check_pretrained_model(default_config)
    
    # Summary
    print("="*70)
    print("Validation Summary")
    print("="*70)
    
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:25} {status}")
    
    print()
    
    if all(checks.values()):
        print("✓ All checks passed! You're ready to start finetuning.")
        print()
        print("Quick start:")
        print("  ./quick_start_finetune.sh")
        print()
        print("Or directly:")
        print("  python finetune.py -p options/finetune/finetune_default.yml")
        return 0
    else:
        print("⚠ Some checks failed. Please fix the issues above before finetuning.")
        print()
        print("Common fixes:")
        print("  - Install missing packages: pip install <package_name>")
        print("  - Prepare dataset: python prepare_dataset.py --help")
        print("  - Download pretrained model to models/ directory")
        return 1


if __name__ == '__main__':
    sys.exit(main())
