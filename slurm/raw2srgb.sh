#!/bin/bash
#SBATCH --partition=cpunodes
#SBATCH --time=4:00:00          # 4 hours buffer
#SBATCH --job-name=raw2srgb
#SBATCH --output=/u/chenjoachim/log/raw2srgb_%j.log
#SBATCH --error=/u/chenjoachim/log/raw2srgb_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joachim.chen@mail.utoronto.ca

source .venv/bin/activate
python3 raw_to_srgb.py -i ../dataset/SID/Sony/test/blurred/ -o ../dataset/SID/Sony/test/blur_rgb/