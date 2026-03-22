#!/bin/bash
#SBATCH --job-name=dr_eval
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/s2759545/logs/eval_%j.out
#SBATCH --error=/home/s2759545/logs/eval_%j.err

mkdir -p /home/s2759545/logs

echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlp

cd /home/s2759545

python evaluate.py

echo "Job finished: $(date)"
