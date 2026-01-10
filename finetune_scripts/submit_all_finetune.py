import os
import glob
import subprocess

# Config
base_dir = "/home/fr/fr_fr/fr_ml642/Thesis"
data_dir = os.path.join(base_dir, "LAMAR/data/finetune_data")
script_path = os.path.join(base_dir, "LAMAR/finetune_scripts/finetune_rbp.py")
output_dir = os.path.join(base_dir, "LAMAR/models/finetuned")
logs_dir = os.path.join(base_dir, "LAMAR/finetune_scripts/logs")

# Model variants
variants = {
    "Pretrained": "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/weights/model.safetensors",
    "TAPT": "/home/fr/fr_fr/fr_ml642/Thesis/pretrain/saving_model/tapt_lamar/checkpoint-100000/model.safetensors",
    "Random": ""
}

os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Find RBPs
rbps = [os.path.basename(d) for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]
print(f"Found {len(rbps)} RBPs to finetune: {rbps}")

for rbp in rbps:
    for variant_name, pretrain_path in variants.items():
        job_name = f"ft_{rbp}_{variant_name}"
        slurm_file = os.path.join(logs_dir, f"{job_name}.slurm")
        
        # Output dir for this specific variant
        variant_output_dir = os.path.join(output_dir, variant_name)
        os.makedirs(variant_output_dir, exist_ok=True)
        
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maximilian.lewinfr@gmail.com

# Load env
source ~/.bashrc
conda activate lamar_finetune

echo "Job started for {rbp} - {variant_name}"
date

python {script_path} \\
    --rbp_name {rbp} \\
    --data_path {os.path.join(data_dir, rbp)} \\
    --output_dir {variant_output_dir} \\
    --pretrain_path "{pretrain_path}" \\
    --epochs 10 \\
    --batch_size 16 \\
    --lr 5e-5

echo "Job finished"
date
"""
        
        with open(slurm_file, "w") as f:
            f.write(slurm_content)
            
        # Submit
        print(f"Submitting job for {rbp} ({variant_name})...")
        subprocess.run(["sbatch", slurm_file])

print("All jobs submitted.")
