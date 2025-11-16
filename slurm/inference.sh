#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --time=4:00:00          # 4 hours buffer
#SBATCH --job-name=darkir_64
#SBATCH --output=/u/chenjoachim/log/darkir64_%j.log
#SBATCH --error=/u/chenjoachim/log/darkir64_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joachim.chen@mail.utoronto.ca

source .venv/bin/activate
python3 inference.py -i ../dataset/SID/LID/short -o ../dataset/SID/LID/predicted_resize -p options/inference/default.yml