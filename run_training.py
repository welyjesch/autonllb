#!/usr/bin/env python3
"""
Training and Inference Script for NLLB Fine-tuning
Loads pre-prepared training_data.jsonl, trains in batches, and validates with inference.

Usage: python run_training.py
       python run_training.py --data training_data.jsonl --output ./results
       python run_training.py --batch-size 500 --num-epochs 2
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


# ==============================================================================
# SECTION 1: Progress Tracker
# ==============================================================================

class ProgressTracker:
    """Track training progress with batch indices and timestamps."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "progress.json"
        
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from checkpoint."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "total_samples": 0,
            "current_batch": 0,
            "batches_trained": [],
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
        }
    
    def _save_progress(self):
        """Save progress to checkpoint."""
        self.progress["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def initialize(self, total_samples: int):
        """Initialize with total sample count."""
        self.progress["total_samples"] = total_samples
        self._save_progress()
    
    def start_batch(self, batch_idx: int, num_samples: int):
        """Mark start of training batch."""
        self.progress["current_batch"] = batch_idx
        self._save_progress()
        print(f"\n{'='*70}")
        print(f"TRAINING BATCH {batch_idx} ({num_samples} samples)")
        print(f"{'='*70}")
    
    def complete_batch(self, batch_idx: int, model_dir: str):
        """Mark completion of training batch."""
        self.progress["batches_trained"].append({
            "batch_index": batch_idx,
            "model_dir": str(model_dir),
            "timestamp": datetime.now().isoformat(),
        })
        self._save_progress()
    
    def get_status(self) -> str:
        """Get formatted status string."""
        batches_completed = len(self.progress["batches_trained"])
        return (
            f"Total Samples: {self.progress['total_samples']} | "
            f"Batches Trained: {batches_completed} | "
            f"Current Batch: {self.progress['current_batch']}"
        )


# ==============================================================================
# SECTION 2: Data Loading
# ==============================================================================

def load_training_data(jsonl_file: Path) -> List[Dict[str, Any]]:
    """Load training data from JSONL file."""
    print(f"Loading training data from {jsonl_file}...")
    
    if not jsonl_file.exists():
        raise FileNotFoundError(f"Training data file not found: {jsonl_file}")
    
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✓ Loaded {len(data)} samples")
    return data


# ==============================================================================
# SECTION 3: Model Setup
# ==============================================================================

def setup_model(model_name: str = "facebook/nllb-200-distilled-600M"):
    """Load the NLLB model and tokenizer."""
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("✓ Model and tokenizer loaded")
    return model, tokenizer


# ==============================================================================
# SECTION 4: Data Preparation
# ==============================================================================

def prepare_batch_dataset(
    tokenizer,
    batch_data: List[Dict[str, str]],
    max_length: int = 128,
):
    """Prepare a batch for training."""
    english_texts = [d["english"] for d in batch_data]
    hiligaynon_texts = [d["hiligaynon"] for d in batch_data]
    
    # Tokenize
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "hil_Latn"
    
    model_inputs = tokenizer(
        english_texts,
        text_target=hiligaynon_texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    
    return model_inputs


# ==============================================================================
# SECTION 5: Training
# ==============================================================================

class SimpleDataset:
    """Simple dataset wrapper for Trainer."""
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


def train_batch(
    model,
    tokenizer,
    batch_data: List[Dict[str, str]],
    batch_idx: int,
    output_dir: Path,
    num_epochs: int = 1,
    learning_rate: float = 2e-5,
) -> str:
    """Train model on a single batch."""
    
    # Prepare data
    tokenized = prepare_batch_dataset(tokenizer, batch_data)
    train_dataset = SimpleDataset(tokenized)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Setup output directory for this batch
    batch_output_dir = output_dir / f"batch_{batch_idx}" / "model"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = output_dir / f"batch_{batch_idx}" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(batch_output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        logging_steps=50,
        logging_dir=str(logs_dir),
        log_level="info",
        report_to=[],  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print(f"Training on batch {batch_idx}...")
    trainer.train()
    
    # Save final model
    model.save_pretrained(batch_output_dir)
    tokenizer.save_pretrained(batch_output_dir)
    print(f"✓ Batch {batch_idx} complete. Model saved to {batch_output_dir}")
    
    return str(batch_output_dir)


# ==============================================================================
# SECTION 6: Inference Testing
# ==============================================================================

def run_inference_test(
    model_dir: str,
    test_data: List[Dict[str, str]],
    num_samples: int = 10,
) -> List[Dict[str, Any]]:
    """Test the fine-tuned model with random samples."""
    print(f"\n{'='*70}")
    print(f"INFERENCE TEST ({num_samples} random samples)")
    print(f"{'='*70}")
    
    from transformers import pipeline
    
    print(f"Loading model from {model_dir}...")
    translator = pipeline(
        "translation",
        model=model_dir,
        tokenizer=model_dir,
        src_lang="eng_Latn",
        tgt_lang="hil_Latn",
        max_length=400,
    )
    
    # Select random samples
    if len(test_data) < num_samples:
        samples = test_data
    else:
        samples = random.sample(test_data, num_samples)
    
    print(f"\nTesting on {len(samples)} random samples:\n")
    
    results = []
    for i, sample in enumerate(samples, 1):
        english_input = sample["english"]
        hiligaynon_reference = sample["hiligaynon"]
        
        # Translate
        output = translator(english_input)
        hiligaynon_predicted = output[0]["translation_text"]
        
        result = {
            "sample_id": sample.get("seq_id"),
            "english_input": english_input,
            "hiligaynon_reference": hiligaynon_reference,
            "hiligaynon_predicted": hiligaynon_predicted,
        }
        results.append(result)
        
        print(f"[Sample {i}/{len(samples)}]")
        print(f"  English:           {english_input}")
        print(f"  Reference:         {hiligaynon_reference}")
        print(f"  Predicted:         {hiligaynon_predicted}")
        print()
    
    return results


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train NLLB model on prepared data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py
  python run_training.py --data training_data.jsonl --output ./results
  python run_training.py --batch-size 500 --num-epochs 2
  python run_training.py --inference-samples 20
        """
    )
    
    parser.add_argument(
        '--data',
        type=Path,
        default=Path("training_data.jsonl"),
        help='Path to training data JSONL file (default: training_data.jsonl)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("outputs"),
        help='Output directory for models and logs (default: outputs)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of samples per training batch (default: 1000)'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=1,
        help='Training epochs per batch (default: 1)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    
    parser.add_argument(
        '--inference-samples',
        type=int,
        default=10,
        help='Number of random samples for inference test (default: 10)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help='Model name (default: facebook/nllb-200-distilled-600M)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print("=" * 70)
    print("NLLB Training Pipeline")
    print("=" * 70)
    
    try:
        training_data = load_training_data(args.data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"First run: python prepare_data.py")
        return 1
    
    if not training_data:
        print("No training data loaded!")
        return 1
    
    # Setup progress tracker
    output_dir = Path(args.output)
    progress_tracker = ProgressTracker(output_dir)
    progress_tracker.initialize(len(training_data))
    
    print(f"Data file: {args.data}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs per batch: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 70)
    
    # Setup model
    model, tokenizer = setup_model(args.model)
    
    # Calculate batches
    num_batches = (len(training_data) + args.batch_size - 1) // args.batch_size
    print(f"\nTraining Configuration:")
    print(f"  Total samples: {len(training_data)}")
    print(f"  Number of batches: {num_batches}")
    print(f"  {progress_tracker.get_status()}")
    
    # Training loop
    trained_model_dir = None
    
    for batch_idx in range(num_batches):
        progress_tracker.start_batch(batch_idx, args.batch_size)
        
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(training_data))
        batch_data = training_data[batch_start:batch_end]
        
        trained_model_dir = train_batch(
            model,
            tokenizer,
            batch_data,
            batch_idx,
            output_dir,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )
        
        progress_tracker.complete_batch(batch_idx, trained_model_dir)
        print(f"Progress: {progress_tracker.get_status()}")
        
        # Reload model for next batch
        if batch_idx < num_batches - 1:
            print(f"Reloading model for next batch...")
            model, tokenizer = setup_model(args.model)
    
    # Inference testing
    if trained_model_dir:
        inference_results = run_inference_test(
            trained_model_dir,
            training_data,
            num_samples=args.inference_samples,
        )
        
        # Save results
        results_file = output_dir / "inference_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(inference_results, f, ensure_ascii=False, indent=2)
        print(f"✓ Inference results saved to {results_file}")
    
    print(f"\n{'='*70}")
    print("✓ Training Pipeline Complete!")
    print(f"{'='*70}")
    print(f"Final Progress: {progress_tracker.get_status()}")
    print(f"Output directory: {output_dir.absolute()}")
    print("\nOutput structure:")
    print("  outputs/")
    print("  ├── progress.json (training state)")
    print("  ├── inference_results.json (test results)")
    print("  ├── batch_0/")
    print("  │   ├── model/ (trained model checkpoint)")
    print("  │   └── logs/ (training logs)")
    print("  ├── batch_1/")
    print("  │   ├── model/")
    print("  │   └── logs/")
    print("  └── ...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
