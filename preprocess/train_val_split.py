import numpy as np
from sklearn.model_selection import train_test_split
import os


RANDOM_SEED = 42

def train_val_split(input_file):
    
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Set validation size to 2%
    train_lines, val_lines = train_test_split(lines, test_size=0.02, random_state=RANDOM_SEED)

    print(f"Total lines: {len(lines)}")
    print(f"Training lines: {len(train_lines)}")
    print(f"Validation lines: {len(val_lines)}")
    print(f"Fraction train/val: {len(train_lines)/len(lines):.4f} / {len(val_lines)/len(lines):.4f}")
    print(f"Train/Val split done. Saving to files...")
    train_file = os.path.splitext(input_file)[0] + "_train.txt"
    val_file = os.path.splitext(input_file)[0] + "_val.txt"

    with open(train_file, 'w') as f:
        f.writelines(train_lines)

    with open(val_file, 'w') as f:
        f.writelines(val_lines)

if __name__ == "__main__":
    input_path = "/home/fr/fr_fr/fr_ml642/Thesis/preprocess/preprocessed_data_sequences.txt"
    train_val_split(input_path)