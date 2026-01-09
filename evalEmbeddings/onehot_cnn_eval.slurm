#!/bin/bash
#SBATCH --job-name=onehot_cnn_eval
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/onehot_cnn_eval_%j.out
#SBATCH --error=logs/onehot_cnn_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

# Info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
source ~/.bashrc

# Activate environment
conda activate lamar_fixed
echo "Environment: $CONDA_DEFAULT_ENV"

# Run script
python OneHot_CNN.py