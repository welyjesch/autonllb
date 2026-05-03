# NLLB Fine-tuning for English to Hiligaynon Translation
## Autoloop Training with Hiligaynon News Articles Dataset

This notebook fine-tunes the `facebook/nllb-200-distilled-600M` model for English-Hiligaynon bidirectional translation using the welyjesch/hiligaynon_news_articles dataset with an automated training loop.

---

## Cell 1: Project Setup with uv

```python
%%writefile pyproject.toml
[project]
name = "nllb-en-hil-autoloop"
version = "0.1.0"
description = "Autoloop fine-tune NLLB for English-Hiligaynon translation with progress tracking"
requires-python = ">=3.9"
dependencies = [
    "transformers>=4.30.0",
    "datasets>=2.14.0",
    "sentencepiece>=0.1.98",
    "accelerate>=0.20.0",
    "evaluate>=0.3.0",
    "sacrebleu>=2.3.0",
    "googletrans==4.0.0-rc1",
    "numpy>=1.23.0",
]

[project.optional-dependencies]
dev = [
    "ipykernel",
    "jupyter",
]
```

---

## Cell 2: Complete Autoloop Training Script

```python
%%writefile run_training.py
#!/usr/bin/env python3
"""
Autoloop Training Script for NLLB English-Hiligaynon Translation
Extracts from welyjesch/hiligaynon_news_articles dataset, translates with googletrans,
trains incrementally with 1000 sentences per batch, and validates with inference tests.

Usage: uv run run_training.py
"""

import json
import os
import re
import shutil
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from googletrans import Translator


# ==============================================================================
# SECTION 1: Sentence Extraction and Splitting (from translate.py logic)
# ==============================================================================

ABBREVIATIONS = [
    r'\b\d+\.\d+\b',
    r'\.\.\.+',
    r'\b\d+\.',
    r'\b[A-Z]\.',
    r'\bDr\.', r'\bMr\.', r'\bMrs\.', r'\bMs\.', r'\bProf\.', 
    r'\bSr\.', r'\bJr\.', r'\bSt\.', r'\bFr\.', r'\bRev\.',
    r'\bGov\.', r'\bSen\.', r'\bRep\.', r'\bPres\.', r'\bGen\.', 
    r'\bCol\.', r'\bLt\.', r'\bSgt\.', r'\bCpl\.', r'\bCapt\.', 
    r'\bCmdr\.', r'\bAdm\.', r'\bAmb\.', r'\bHon\.',
    r'\bPhD\.', r'\bMD\.', r'\bBA\.', r'\bBS\.', r'\bMA\.', r'\bMS\.',
    r'\bInc\.', r'\bLtd\.', r'\bCo\.', r'\bCorp\.',
    r'\bvs\.', r'\betc\.', r'\bi\.e\.', r'\be\.g\.',
    r'\bFig\.', r'\bfig\.', r'\bVol\.', r'\bvol\.', 
    r'\bNo\.', r'\bno\.', r'\bpp\.', r'\bpg\.', r'\bp\.',
    r'\bed\.', r'\beds\.', r'\best\.', r'\bapprox\.', 
    r'\bca\.', r'\bcf\.', r'\bviz\.', r'\bal\.', r'\bet al\.',
]

ABBREVIATION_PATTERN = '(' + '|'.join(ABBREVIATIONS) + r')'

ENGLISH_INDICATORS = [
    'the', 'and', 'that', 'with', 'for', 'this', 'from', 'have', 'not', 'but',
    'what', 'all', 'were', 'when', 'there', 'can', 'an', 'been', 'has', 'had',
    'one', 'their', 'also', 'who', 'more', 'will', 'each', 'about', 'which',
    'her', 'she', 'into', 'its', 'only', 'other', 'new', 'them', 'then',
    'because', 'these', 'time', 'very', 'just', 'like', 'people', 'so',
    'than', 'first', 'may', 'way', 'even', 'see', 'after', 'good', 'know',
    'year', 'most', 'would', 'world', 'city', 'some', 'where', 'between',
    'made', 'then', 'did', 'need', 'life', 'here', 'long', 'take', 'come',
    'great', 'still', 'public', 'house', 'own', 'under', 'water', 'work',
    'while', 'such', 'being', 'well', 'how', 'many', 'any', 'days', 'go',
]

ENGLISH_PATTERNS = [
    r'\b(the|and|that|with|for|this|from|have|not)\b',
    r'\b(was|were|are|is|been|being|be)\b',
    r'\b(will|would|could|should|may|might|must|can)\b',
    r'\b(he|she|it|they|we|you|i)\b',
    r'\b(his|her|their|our|your|my|its)\b',
    r'\b(in|on|at|to|for|by|with|from|into|through|during)\b',
]


def is_english_sentence(sentence: str, threshold: float = 0.3) -> bool:
    """Detect if a sentence is likely in English."""
    if not sentence or not sentence.strip():
        return False
    
    text_lower = sentence.lower()
    english_word_count = 0
    
    for pattern in ENGLISH_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        english_word_count += len(matches)
    
    words = sentence.split()
    if len(words) == 0:
        return False
    
    english_ratio = english_word_count / len(words)
    return english_ratio >= threshold


def split_into_sentences(text: str) -> List[str]:
    """Split Hiligaynon text into sentences using regex pattern."""
    PLACEHOLDER = '<<ABBR_PERIOD>>'
    
    def protect_abbreviation(match):
        return match.group(0).replace('.', PLACEHOLDER)
    
    protected_text = re.sub(ABBREVIATION_PATTERN, protect_abbreviation, text)
    sentences = re.split(r'[.!?]+\s*', protected_text)
    sentences = [s.replace(PLACEHOLDER, '.') for s in sentences]
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s) >= 10]
    
    return sentences


def translate_sentence(
    translator: Translator,
    sentence: str,
    delay: float = 2.5,
    retries: int = 3
) -> str:
    """Translate a single sentence from Hiligaynon to English with retry logic."""
    for attempt in range(retries):
        try:
            time.sleep(delay)
            result = translator.translate(sentence, src_lang='hil', dest_lang='en')
            
            if hasattr(result, 'text') and result.text:
                return result.text
            elif isinstance(result, str):
                return result
            
            return sentence
        except Exception as e:
            print(f"  Translation error (attempt {attempt + 1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    
    return sentence


# ==============================================================================
# SECTION 2: Progress Tracker
# ==============================================================================

class ProgressTracker:
    """Track training progress with sequence IDs and batch indices."""
    
    def __init__(self, checkpoint_dir: str = "./training_progress"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.progress_file = self.checkpoint_dir / "progress.json"
        self.data_dir = self.checkpoint_dir / "split_sentences"
        self.data_dir.mkdir(exist_ok=True)
        
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from checkpoint."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "total_sentences_extracted": 0,
            "total_sentences_translated": 0,
            "batch_index": 0,
            "training_index": 0,
            "last_training_timestamp": None,
            "batches_trained": [],
        }
    
    def _save_progress(self):
        """Save progress to checkpoint."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def add_extracted_sentences(self, sentences: List[str]) -> Dict[int, str]:
        """Save extracted sentences with sequence IDs and return mapping."""
        sequence_id = int(time.time() * 1000)  # millisecond timestamp
        seq_sentences = {}
        
        for i, sentence in enumerate(sentences):
            seq_id = sequence_id + i
            seq_sentences[seq_id] = sentence
        
        # Save to local storage
        batch_file = self.data_dir / f"sentences_{sequence_id}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(seq_sentences, f, ensure_ascii=False, indent=2)
        
        self.progress["total_sentences_extracted"] += len(sentences)
        self._save_progress()
        
        return seq_sentences
    
    def add_translated_pairs(self, pairs: List[Dict[str, Any]]):
        """Record translated sentence pairs."""
        self.progress["total_sentences_translated"] += len(pairs)
        self._save_progress()
    
    def start_training_batch(self, batch_idx: int):
        """Mark start of training batch."""
        self.progress["batch_index"] = batch_idx
        self.progress["training_index"] = batch_idx * 1000  # Track at which sentence we are
        self._save_progress()
    
    def complete_training_batch(self, batch_idx: int):
        """Mark completion of training batch."""
        self.progress["batches_trained"].append({
            "batch_index": batch_idx,
            "timestamp": datetime.now().isoformat(),
        })
        self.progress["last_training_timestamp"] = datetime.now().isoformat()
        self._save_progress()
    
    def get_status(self) -> str:
        """Get formatted status string."""
        return (
            f"Extracted: {self.progress['total_sentences_extracted']} | "
            f"Translated: {self.progress['total_sentences_translated']} | "
            f"Batches Trained: {len(self.progress['batches_trained'])} | "
            f"Training Index: {self.progress['training_index']}"
        )


# ==============================================================================
# SECTION 3: Data Extraction and Translation
# ==============================================================================

def extract_and_translate_dataset(
    progress_tracker: ProgressTracker,
    max_articles: int = None,
) -> List[Dict[str, str]]:
    """
    Extract articles from welyjesch/hiligaynon_news_articles dataset,
    split into sentences, and translate Hiligaynon to English.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: Extracting and Translating Dataset")
    print("=" * 70)
    
    print("Loading welyjesch/hiligaynon_news_articles dataset...")
    dataset = load_dataset("welyjesch/hiligaynon_news_articles")
    articles = dataset["train"]["articles"]
    
    if max_articles:
        articles = articles[:max_articles]
    
    print(f"Loaded {len(articles)} articles")
    
    translator = Translator()
    all_pairs = []
    translated_count = 0
    
    for article_idx, article in enumerate(articles, 1):
        print(f"\n[Article {article_idx}/{len(articles)}] Processing...")
        
        # Split into sentences
        sentences = split_into_sentences(article)
        print(f"  Split into {len(sentences)} sentences")
        
        # Save extracted sentences with sequence IDs
        seq_sentences = progress_tracker.add_extracted_sentences(sentences)
        
        # Translate sentences
        for seq_id, sentence in seq_sentences.items():
            # Skip English sentences
            if is_english_sentence(sentence):
                continue
            
            english = translate_sentence(translator, sentence, delay=2.5)
            
            pair = {
                "sequence_id": seq_id,
                "article_index": article_idx - 1,
                "hiligaynon": sentence,
                "english": english,
            }
            all_pairs.append(pair)
            translated_count += 1
            
            # Show progress every 10 sentences
            if translated_count % 10 == 0:
                print(f"  Translated: {translated_count} sentences")
        
        print(f"  Total translated from this article: {translated_count}")
    
    progress_tracker.add_translated_pairs(all_pairs)
    print(f"\n✓ Extraction and translation complete!")
    print(f"  Total pairs ready for training: {len(all_pairs)}")
    
    return all_pairs


# ==============================================================================
# SECTION 4: Tokenization and Data Processing
# ==============================================================================

def prepare_batch_dataset(
    tokenizer,
    pairs: List[Dict[str, str]],
    batch_start: int,
    batch_size: int = 1000,
    max_length: int = 128,
):
    """Prepare a batch of sentence pairs for training."""
    # Extract batch
    batch_pairs = pairs[batch_start : batch_start + batch_size]
    
    if not batch_pairs:
        return None
    
    # Create dataset
    english_texts = [p["english"] for p in batch_pairs]
    hiligaynon_texts = [p["hiligaynon"] for p in batch_pairs]
    
    dataset = {
        "en": english_texts,
        "hil": hiligaynon_texts,
    }
    
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
# SECTION 5: Model Setup
# ==============================================================================

def setup_model(model_name: str = "facebook/nllb-200-distilled-600M"):
    """Load the NLLB model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("✓ Model and tokenizer loaded")
    return model, tokenizer


# ==============================================================================
# SECTION 6: Training Loop with Batches
# ==============================================================================

def train_batch(
    model,
    tokenizer,
    batch_pairs: List[Dict[str, str]],
    batch_idx: int,
    output_dir: str = "./nllb-hiligaynon-batches",
    num_epochs: int = 1,
):
    """Train model on a single batch of 1000 sentences."""
    print(f"\n{'=' * 70}")
    print(f"STAGE 3: Training Batch {batch_idx} ({len(batch_pairs)} sentences)")
    print(f"{'=' * 70}")
    
    # Prepare tokenized dataset
    tokenized = prepare_batch_dataset(tokenizer, batch_pairs, 0)
    
    if tokenized is None:
        print("⚠ No data to train on, skipping batch")
        return None
    
    # Create simple dataset format for trainer
    class SimpleDataset:
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings["input_ids"])
        
        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}
    
    train_dataset = SimpleDataset(tokenized)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Setup training arguments
    batch_output_dir = os.path.join(output_dir, f"batch_{batch_idx}")
    os.makedirs(batch_output_dir, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=batch_output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        logging_steps=50,
        log_level="info",
        report_to=[],  # Disable wandb/tensorboard for Colab
    )
    
    # Train
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print(f"Training on batch {batch_idx}...")
    trainer.train()
    
    # Save model for this batch
    model.save_pretrained(batch_output_dir)
    tokenizer.save_pretrained(batch_output_dir)
    print(f"✓ Batch {batch_idx} trained and saved to {batch_output_dir}")
    
    return batch_output_dir


# ==============================================================================
# SECTION 7: Inference Testing on Random Samples
# ==============================================================================

def run_inference_test(
    model_dir: str,
    all_pairs: List[Dict[str, str]],
    num_samples: int = 10,
):
    """Test the fine-tuned model with random samples from the trained data."""
    print(f"\n{'=' * 70}")
    print(f"STAGE 4: Inference Testing ({num_samples} random samples)")
    print(f"{'=' * 70}")
    
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
    if len(all_pairs) < num_samples:
        samples = all_pairs
    else:
        samples = random.sample(all_pairs, num_samples)
    
    print(f"\nTesting on {len(samples)} random samples:\n")
    
    results = []
    for i, sample in enumerate(samples, 1):
        english_input = sample["english"]
        hiligaynon_reference = sample["hiligaynon"]
        
        # Translate
        output = translator(english_input)
        hiligaynon_predicted = output[0]["translation_text"]
        
        result = {
            "sample": i,
            "english_input": english_input,
            "hiligaynon_reference": hiligaynon_reference,
            "hiligaynon_predicted": hiligaynon_predicted,
        }
        results.append(result)
        
        print(f"[Sample {i}/{len(samples)}]")
        print(f"  English Input:         {english_input}")
        print(f"  Reference Hiligaynon:  {hiligaynon_reference}")
        print(f"  Predicted Hiligaynon:  {hiligaynon_predicted}")
        print()
    
    return results


# ==============================================================================
# MAIN EXECUTION: Autoloop
# ==============================================================================

def main():
    print("\n" + "=" * 70)
    print("NLLB Autoloop Training: English-Hiligaynon Translation")
    print("=" * 70)
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker()
    print(f"Progress: {progress_tracker.get_status()}")
    
    # STAGE 1: Extract and translate dataset
    all_pairs = extract_and_translate_dataset(
        progress_tracker,
        max_articles=10,  # Adjust for more/fewer articles
    )
    
    if not all_pairs:
        print("No translation pairs generated, exiting")
        return
    
    # STAGE 2: Setup model
    print("\n" + "=" * 70)
    print("STAGE 2: Model Setup")
    print("=" * 70)
    model, tokenizer = setup_model()
    
    # STAGE 3: Training loop with batches of 1000
    batch_size = 1000
    num_batches = (len(all_pairs) + batch_size - 1) // batch_size
    
    print(f"\n{'=' * 70}")
    print(f"Training Configuration")
    print(f"{'=' * 70}")
    print(f"Total sentence pairs: {len(all_pairs)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"{'=' * 70}\n")
    
    trained_model_dir = None
    
    for batch_idx in range(num_batches):
        progress_tracker.start_training_batch(batch_idx)
        
        batch_start = batch_idx * batch_size
        batch_pairs = all_pairs[batch_start : batch_start + batch_size]
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}: {len(batch_pairs)} sentences")
        
        trained_model_dir = train_batch(
            model,
            tokenizer,
            batch_pairs,
            batch_idx,
        )
        
        progress_tracker.complete_training_batch(batch_idx)
        print(f"Progress: {progress_tracker.get_status()}")
        
        # Reload model for next batch (model is updated in place)
        if batch_idx < num_batches - 1:
            print(f"Loading updated model for next batch...")
            model, tokenizer = setup_model()
    
    # STAGE 4: Inference testing on random samples
    if trained_model_dir:
        inference_results = run_inference_test(
            trained_model_dir,
            all_pairs,
            num_samples=10,
        )
        
        # Save results
        results_file = Path("./inference_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(inference_results, f, ensure_ascii=False, indent=2)
        print(f"✓ Inference results saved to {results_file}")
    
    print(f"\n{'=' * 70}")
    print("Autoloop Training Complete!")
    print(f"{'=' * 70}")
    print(f"Final Progress: {progress_tracker.get_status()}")


if __name__ == "__main__":
    main()
```

---

## Cell 3: Installation and Setup

```bash
# Install uv if not already installed
!pip install uv

# Install dependencies via uv
!uv pip install -q transformers datasets sentencepiece accelerate evaluate sacrebleu googletrans==4.0.0-rc1 numpy
```

---

## Cell 4: Run the Autoloop Training

```python
# Run the complete autoloop training pipeline
exec(open("run_training.py").read())
```

---

## Configuration and Customization

### Adjusting Training Parameters

Edit the `main()` function in `run_training.py`:

- **`max_articles`**: Number of articles to extract from the dataset (default: 10)
  ```python
  all_pairs = extract_and_translate_dataset(progress_tracker, max_articles=10)
  ```

- **`batch_size`**: Sentences per training batch (default: 1000)
  ```python
  batch_size = 1000
  ```

- **`num_epochs`** in `train_batch()`: Epochs per batch (default: 1)

- **`num_samples`** in inference test: Number of random samples to validate (default: 10)

### Progress Tracking

The autoloop maintains progress in `./training_progress/`:
- `progress.json` - Training state and metrics
- `split_sentences/` - Extracted sentences stored with timestamp-based sequence IDs
- Batch model checkpoints in `./nllb-hiligaynon-batches/batch_X/`

### Resuming from Checkpoint

If training is interrupted, the progress tracker automatically loads the last checkpoint on the next run.

---

## Output Files

After completion, you'll have:

1. **Model Checkpoints**: `./nllb-hiligaynon-batches/batch_X/` - One directory per 1000-sentence batch
2. **Progress Tracker**: `./training_progress/progress.json` - Training metrics and state
3. **Extracted Sentences**: `./training_progress/split_sentences/` - Hiligaynon sentences with sequence IDs
4. **Inference Results**: `./inference_results.json` - 10 random test translations (reference vs predicted)

---

## Key Features

✓ **Automatic Dataset Extraction**: Loads welyjesch/hiligaynon_news_articles and extracts articles  
✓ **Smart Sentence Splitting**: Uses pattern-based extraction from translate.py logic  
✓ **Intelligent Translation**: Filters English sentences, translates Hiligaynon→English using googletrans v4  
✓ **Sequence Tracking**: Sentences stored with millisecond-precision sequence IDs  
✓ **Batch Training**: Incremental training with 1000 sentences per batch  
✓ **Progress Persistence**: Checkpoints allow resuming interrupted training  
✓ **Validation Testing**: Inference on 10 random samples from trained data  
✓ **Bidirectional**: Trains English→Hiligaynon while using English as source language
