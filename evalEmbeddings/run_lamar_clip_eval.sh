#!/bin/bash
#SBATCH --job-name=lamar_clip_eval
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/logs/lamar_clip_eval_%j.out
#SBATCH --error=/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/logs/lamar_clip_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

# Load env
source ~/.bashrc
conda activate lamar_finetune

echo "Starting LAMAR CLIP Evaluation"
date

python /home/fr/fr_fr/fr_ml642/Thesis/LAMAR/evalEmbeddings/LAMAR_CNN_clip_data.py

echo "Job finished"
date
