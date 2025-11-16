#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --time=4:00:00          # 4 hours buffer
#SBATCH --job-name=darkir_64
#SBATCH --output=/u/chenjoachim/log/eval64_%j.log
#SBATCH --error=/u/chenjoachim/log/eval64_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joachim.chen@mail.utoronto.ca

source .venv/bin/activate
python3 scoring.py --pred_dir ../dataset/SID/LID/predicted_64 --gt_dir ../dataset/SID/LID/long --csv ../dataset/SID/LID/file_list.csv --output metrics_64.csv
