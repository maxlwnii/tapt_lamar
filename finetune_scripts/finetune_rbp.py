import os
import sys
import argparse
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from safetensors.torch import load_file, load_model

# Add parent directory to path to allow imports from LAMAR package
sys.path.append("/home/fr/fr_fr/fr_ml642/Thesis/LAMAR")
try:
    from LAMAR.sequence_classification_patch import EsmForSequenceClassification
except ImportError:
    # Fallback if running from within LAMAR directory structure differently
    sys.path.append("/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/LAMAR")
    from sequence_classification_patch import EsmForSequenceClassification
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rbp_name", type=str, required=True, help="Name of the RBP")
    parser.add_argument("--data_path", type=str, required=True, help="Path to tokenized data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save models")
    parser.add_argument("--pretrain_path", type=str, default="/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/weights/model.safetensors", help="Path to pretrained weights")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Assuming single GPU job
    
    # Configs
    tokenizer_path = "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/tokenizer/single_nucleotide/"
    config_path = "/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/config/config_150M.json"
    
    # Create output dir specific to RBP
    run_output_dir = os.path.join(args.output_dir, args.rbp_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    print(f"Starting finetuning for {args.rbp_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=512, padding_side='left')
    
    # Config
    config = AutoConfig.from_pretrained(
        config_path, 
        vocab_size=len(tokenizer), 
        pad_token_id=tokenizer.pad_token_id, 
        mask_token_id=tokenizer.mask_token_id, 
        num_labels=2, # Binary classification
        token_dropout=False, 
        positional_embedding_type='rotary',
        hidden_size=768, 
        intermediate_size=3072, 
        num_attention_heads=12, 
        num_hidden_layers=12
    )
    
    # Load Data
    print(f"Loading data from {args.data_path}")
    dataset = load_from_disk(args.data_path)
    print(dataset)
    
    # Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    # Model
    model = EsmForSequenceClassification(config, head_type="Linear", freeze=False, kernel_sizes=[3, 5, 7], ocs=256)
    
    # Load pretrained weights
    if args.pretrain_path and os.path.exists(args.pretrain_path):
        print(f"Loading pretrained weights from {args.pretrain_path}")
        if args.pretrain_path.endswith('.bin'):
            state_dict = torch.load(args.pretrain_path, map_location="cpu")
        elif args.pretrain_path.endswith('.safetensors'):
            state_dict = load_file(args.pretrain_path)
            
        # Filter keys if necessary (adapted from notebook logic if complex mapping needed, 
        # but usually load_state_dict handles it with strict=False for classification head mismatch)
        # The notebook used load_model/load_state_dict with strict=False.
        
        # We need to map 'esm.' appropriately if the pretrain weights are from MLM model
        # The ESMForSequenceClassification wraps 'esm' model. 
        # Check if keys match.
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Unmatched keys: {len(missing)} missing, {len(unexpected)} unexpected")
    else:
        print(f"Pretrained weights not found at {args.pretrain_path}, initializing from scratch/config")

    # Training Args
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_dir=os.path.join(run_output_dir, "logs"),
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=args.seed,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        dataloader_num_workers=4
    )
    
    # Metrics
    import evaluate
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final
    final_path = os.path.join(run_output_dir, "final_model")
    trainer.save_model(final_path)
    print(f"Model saved to {final_path}")

if __name__ == "__main__":
    main()
