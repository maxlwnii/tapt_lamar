import os
import glob
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import sys

# Paths
base_dir = "/home/fr/fr_fr/fr_ml642/Thesis"
data_dir = os.path.join(base_dir, "data/clip_training_data")
lamar_dir = os.path.join(base_dir, "LAMAR")
tokenizer_path = os.path.join(lamar_dir, "tokenizer/single_nucleotide")
output_base_dir = os.path.join(lamar_dir, "data/finetune_data")

os.makedirs(output_base_dir, exist_ok=True)

# Helper to read fasta
def read_fasta(file_path):
    seqs = []
    with open(file_path, 'r') as f:
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    seqs.append(seq)
                seq = ""
            else:
                seq += line
        if seq:
            seqs.append(seq)
    return seqs

# Identify RBPs
files = glob.glob(os.path.join(data_dir, "*.positives.fa"))
rbps = [os.path.basename(f).replace(".positives.fa", "") for f in files]

print(f"Found {len(rbps)} RBPs: {rbps}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=200) # Short clips usually

def group_texts(examples):
    # Tokenize. Note: LAMAR tokenizer expects DNA, so replace U with T
    seqs = [s.upper().replace("U", "T") for s in examples["seq"]]
    tokenized_inputs = tokenizer(seqs, truncation=True, max_length=tokenizer.model_max_length, padding="max_length")
    tokenized_inputs["labels"] = examples["label"]
    return tokenized_inputs

for rbp in rbps:
    print(f"Processing {rbp}...")
    pos_file = os.path.join(data_dir, f"{rbp}.positives.fa")
    neg_file = os.path.join(data_dir, f"{rbp}.negatives.fa")
    
    if not os.path.exists(neg_file):
        print(f"Warning: Negative file for {rbp} not found. Skipping.")
        continue
        
    pos_seqs = read_fasta(pos_file)
    neg_seqs = read_fasta(neg_file)
    
    # Create DataFrame
    df_pos = pd.DataFrame({"seq": pos_seqs, "label": [1] * len(pos_seqs)})
    df_neg = pd.DataFrame({"seq": neg_seqs, "label": [0] * len(neg_seqs)})
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split Train/Test (Evaluation/Validation)
    # Using 90/10 split
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    
    # Tokenize
    tokenized_datasets = dataset_dict.map(group_texts, batched=True)
    
    # Save
    save_path = os.path.join(output_base_dir, rbp)
    tokenized_datasets.save_to_disk(save_path)
    print(f"Saved tokenized data for {rbp} to {save_path}")

print("Data preparation complete.")
