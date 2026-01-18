import argparse
import os
import torch
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from safetensors.torch import load_file
from transformers import (
    EsmConfig,
    EsmForSequenceClassification, 
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    EarlyStoppingCallback,  # ADD THIS IMPORT
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

TOKENIZER_PATH = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/pretrain/saving_model/tapt_lamar/checkpoint-100000/"

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    probabilities = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()
    pred_labels = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels, average='binary', zero_division=0)
    acc = accuracy_score(labels, pred_labels)
    try:
        auc = roc_auc_score(labels, probabilities)
    except Exception:
        auc = 0.5
    
    try:
        auprc = average_precision_score(labels, probabilities)
    except Exception:
        auprc = 0.5
        
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'auprc': auprc
    }

def init_weights(module):
    """Custom random initialization for LAMAR model."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


def load_encoder_weights(model, weights_path):
    """Load ONLY encoder weights from safetensors, leaving classifier random."""
    print(f"\n{'='*60}")
    print(f"Loading encoder weights from: {weights_path}")
    print(f"{'='*60}")
    
    state_dict = load_file(weights_path)
    
    # Filter to keep ONLY encoder weights (esm.*), exclude lm_head and classifier
    encoder_weights = {}
    for k, v in state_dict.items():
        # Skip language model head and classifier
        if 'lm_head' in k or 'classifier' in k:
            continue
            
        # Ensure proper esm. prefix for encoder
        if k.startswith("esm."):
            encoder_weights[k] = v
        else:
            # Add esm. prefix if missing
            encoder_weights["esm." + k] = v
    
    print(f"Found {len(encoder_weights)} encoder weight tensors")
    
    # Load encoder weights only (strict=False for classifier mismatch)
    missing_keys, unexpected_keys = model.load_state_dict(encoder_weights, strict=False)
    
    # Verify loading
    encoder_loaded = [k for k in missing_keys if k.startswith('esm.')]
    classifier_missing = [k for k in missing_keys if 'classifier' in k]
    
    print(f"\n✓ Encoder weights loaded: {len(encoder_weights)} tensors")
    print(f"✓ Classifier randomly initialized: {len(classifier_missing)} tensors")
    
    if encoder_loaded:
        print(f"⚠ WARNING: {len(encoder_loaded)} encoder weights missing!")
        print(f"  Sample missing: {encoder_loaded[:3]}")
    
    if unexpected_keys:
        print(f"⚠ WARNING: {len(unexpected_keys)} unexpected keys")
        print(f"  Sample unexpected: {unexpected_keys[:3]}")
    
    print(f"{'='*60}\n")
    
    return model


def freeze_encoder(model, freeze=True):
    """Freeze encoder layers, keep classifier trainable."""
    for name, param in model.named_parameters():
        if name.startswith('esm.'):
            param.requires_grad = not freeze
        else:  # classifier layers
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--rbp_name", type=str, required=True)
        parser.add_argument("--data_path", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--pretrain_path", type=str, default="")
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--freeze_encoder", action="store_true", 
                            help="Freeze encoder, train only classifier (recommended for small datasets)")
        parser.add_argument("--warmup_epochs", type=int, default=0,
                            help="Number of epochs to train only classifier before unfreezing encoder")
        parser.add_argument("--early_stopping_patience", type=int, default=None,
                            help="Patience for early stopping (None to disable)")
        parser.add_argument("--subsample_pos", type=int, default=None,
                            help="Number of positive samples to subsample (for limited data)")
        parser.add_argument("--subsample_neg", type=int, default=None,
                            help="Number of negative samples to subsample (for limited data)")
        
        args = parser.parse_args()
        
        print(f"\n{'='*60}", flush=True)
        print(f"Finetuning Configuration", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"RBP: {args.rbp_name}", flush=True)
        print(f"Data: {args.data_path}", flush=True)
        print(f"Output: {args.output_dir}", flush=True)
        print(f"Pretrain: {args.pretrain_path if args.pretrain_path else 'None (Random Init)'}", flush=True)
        print(f"Freeze encoder: {args.freeze_encoder}", flush=True)
        print(f"Warmup epochs: {args.warmup_epochs}", flush=True)
        print(f"Early stopping patience: {args.early_stopping_patience if args.early_stopping_patience else 'Disabled'}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Load tokenizer
        print(f"Loading tokenizer from {TOKENIZER_PATH}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})\n", flush=True)
        
        # Load and preprocess dataset
        print("Loading dataset...", flush=True)
        dataset = load_from_disk(args.data_path)
        print(f"✓ Dataset loaded: {list(dataset.keys())}\n", flush=True)
        
        # Subsample if requested
        if args.subsample_pos is not None and args.subsample_neg is not None:
            print(f"Subsampling to {args.subsample_pos} pos and {args.subsample_neg} neg per split...", flush=True)
            try:
                subsampled_data = {}
                for split in ['train', 'test']:
                    ds = dataset[split]
                    print(f"  Processing {split} split ({len(ds)} samples)...", flush=True)
                    
                    pos_ds = ds.filter(lambda x: x['label'] == 1)
                    neg_ds = ds.filter(lambda x: x['label'] == 0)
                    
                    print(f"    Found {len(pos_ds)} positives, {len(neg_ds)} negatives", flush=True)
                    
                    pos_count = min(args.subsample_pos, len(pos_ds))
                    neg_count = min(args.subsample_neg, len(neg_ds))
                    
                    pos_sample = pos_ds.shuffle(seed=42).select(range(pos_count))
                    neg_sample = neg_ds.shuffle(seed=42).select(range(neg_count))
                    
                    subsampled_data[split] = concatenate_datasets([pos_sample, neg_sample]).shuffle(seed=42)
                    print(f"    ✓ {split}: {len(subsampled_data[split])} samples after subsampling", flush=True)
                
                # Replace dataset with subsampled version
                from datasets import DatasetDict
                dataset = DatasetDict(subsampled_data)
                print("✓ Subsampling complete\n", flush=True)
            except Exception as e:
                print(f"❌ ERROR during subsampling: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
        
        # Convert RNA to DNA and tokenize
        def preprocess_function(examples):
            seqs = [seq.replace('U', 'T').replace('u', 't') for seq in examples['seq']]
            return tokenizer(seqs, truncation=True, padding='max_length', max_length=512)
        
        print("Tokenizing sequences...", flush=True)
        try:
            encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['seq'])
            print(f"✓ Tokenization complete\n", flush=True)
        except Exception as e:
            print(f"❌ ERROR during tokenization: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Create config
        num_labels = 2
        config = EsmConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            token_dropout=False,
            positional_embedding_type="rotary",
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            num_hidden_layers=12,
            num_labels=num_labels
        )
        
        # Initialize model
        print("Initializing model...", flush=True)
        try:
            model = EsmForSequenceClassification(config)
            model.apply(init_weights)  # Always init random first
            print("✓ Model initialized with random weights\n", flush=True)
        except Exception as e:
            print(f"❌ ERROR during model initialization: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Load pretrained encoder if provided
        if args.pretrain_path and os.path.exists(args.pretrain_path):
            try:
                model = load_encoder_weights(model, args.pretrain_path)
            except Exception as e:
                print(f"❌ ERROR loading pretrained weights: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
        else:
            print("No pretrained weights - using random initialization for entire model\n", flush=True)
        
        # Freeze encoder if requested
        if args.freeze_encoder:
            print("Freezing encoder layers...", flush=True)
            freeze_encoder(model, freeze=True)
            print()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=2,  # Smaller eval batch to prevent OOM
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            save_total_limit=2,
            logging_dir=f"{args.output_dir}/logs",
            logging_steps=50,
            dataloader_num_workers=4,  # No workers to reduce memory overhead
            report_to="none",
            fp16=True,  # Disable mixed precision - can cause crashes
        )
        
        # Create callbacks list
        callbacks = []
        if args.early_stopping_patience is not None:
            print(f"Early stopping enabled with patience={args.early_stopping_patience}\n", flush=True)
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_threshold=0.001  # Require 0.1% improvement to count as improvement
                )
            )
        
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=encoded_dataset["train"],
                eval_dataset=encoded_dataset["validation"],  # Use validation for early stopping
                compute_metrics=compute_metrics,
                callbacks=callbacks,  # ADD CALLBACKS HERE
            )
        except Exception as e:
            print(f"❌ ERROR creating trainer: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Warmup training (classifier only)
        if args.warmup_epochs > 0 and not args.freeze_encoder:
            print(f"\n{'='*60}", flush=True)
            print(f"WARMUP PHASE: Training classifier only for {args.warmup_epochs} epochs", flush=True)
            print(f"{'='*60}\n", flush=True)
            
            freeze_encoder(model, freeze=True)
            
            # Create separate trainer for warmup (without early stopping to ensure full warmup)
            warmup_args = TrainingArguments(
                output_dir=f"{args.output_dir}/warmup",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=args.lr,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=2,  # Smaller eval batch
                num_train_epochs=args.warmup_epochs,
                weight_decay=0.01,
                load_best_model_at_end=False,
                save_total_limit=1,
                logging_dir=f"{args.output_dir}/warmup/logs",
                logging_steps=50,
                dataloader_num_workers=4,  # No workers to reduce memory
                report_to="none",
                fp16=False,  # Disable mixed precision
            )
            
            try:
                warmup_trainer = Trainer(
                    model=model,
                    args=warmup_args,
                    train_dataset=encoded_dataset["train"],
                    eval_dataset=encoded_dataset["validation"],  # Use validation
                    compute_metrics=compute_metrics,
                )
                warmup_trainer.train()
            except Exception as e:
                print(f"❌ ERROR during warmup training: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            print(f"\n{'='*60}", flush=True)
            print(f"MAIN PHASE: Unfreezing encoder for full fine-tuning", flush=True)
            print(f"{'='*60}\n", flush=True)
            
            freeze_encoder(model, freeze=False)
            
            # Recreate main trainer with early stopping for main phase
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=encoded_dataset["train"],
                eval_dataset=encoded_dataset["validation"],  # Use validation
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
        
        # Main training
        print("Starting training...", flush=True)
        try:
            trainer.train()
        except Exception as e:
            print(f"❌ ERROR during training: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Final evaluation
        print("\nFinal evaluation...", flush=True)
        try:
            eval_results = trainer.evaluate(encoded_dataset["test"])  # Evaluate on test set
            print(f"\n{'='*60}", flush=True)
            print("Results:", flush=True)
            for k, v in eval_results.items():
                print(f"  {k}: {v:.4f}", flush=True)
            print(f"{'='*60}\n", flush=True)
        except Exception as e:
            print(f"❌ ERROR during evaluation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Save model
        try:
            trainer.save_model(args.output_dir)
            print(f"✓ Model saved to {args.output_dir}\n", flush=True)
        except Exception as e:
            print(f"❌ ERROR saving model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    except Exception as e:
        print(f"\n\n❌ MAIN ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user", flush=True)
        import sys
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
