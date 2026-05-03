# NLLB Fine-tuning Workflow

Complete workflow for training NLLB models on Hiligaynon-English translation.

## File Organization

```
prepare_data.py    → Extracts, splits, translates → training_data.jsonl
                                                         ↓
run_training.py    → Trains model on batches    → outputs/ (models, logs, progress)
```

## Step-by-Step Usage

### 1. Install Dependencies (One-time)

```bash
python install.py
```

Or with bash (Linux/macOS):
```bash
chmod +x install.sh
./install.sh
```

### 2. Prepare Training Data

Extract articles, split into sentences, and translate to English:

```bash
python prepare_data.py
```

**Output:** `training_data.jsonl` with fields:
- `seq_id` - Unique sequence ID (millisecond timestamp)
- `english` - Translated English text
- `hiligaynon` - Original Hiligaynon text

**Options:**
```bash
# Process only 50 articles
python prepare_data.py --max-articles 50

# Custom output file and longer delays
python prepare_data.py --output data/my_data.jsonl --delay 3.0
```

### 3. Train Model on Prepared Data

Train NLLB model in batches with progress tracking:

```bash
python run_training.py
```

**Output Structure:**
```
outputs/
├── progress.json              # Training state (resumable)
├── inference_results.json     # Test translations
├── batch_0/
│   ├── model/                 # Model checkpoint
│   └── logs/                  # Training logs
├── batch_1/
│   ├── model/
│   └── logs/
└── ...
```

**Options:**
```bash
# Custom batch size and epochs
python run_training.py --batch-size 500 --num-epochs 2

# Higher learning rate
python run_training.py --learning-rate 5e-5

# More inference test samples
python run_training.py --inference-samples 20

# Custom output directory
python run_training.py --output ./my_results

# All together
python run_training.py \
    --data training_data.jsonl \
    --output ./results \
    --batch-size 500 \
    --num-epochs 2 \
    --learning-rate 5e-5 \
    --inference-samples 20
```

## Key Points

### Data Preparation
- Filters out English sentences automatically
- Handles abbreviations (Dr., Mr., Dr., etc.)
- Minimum sentence length: 10 characters
- Sequence IDs use millisecond timestamps for uniqueness

### Training
- Configurable batch size (default: 1000 samples)
- Each batch saved independently
- Progress tracked for resumability
- Automatic model reloading between batches

### Output Files
- **Models:** Located in `outputs/batch_X/model/`
- **Logs:** Located in `outputs/batch_X/logs/`
- **Progress:** `outputs/progress.json` (resumable)
- **Test Results:** `outputs/inference_results.json`

## Resume Interrupted Training

Training saves progress after each batch. To resume:

```bash
python run_training.py
```

The script automatically loads the last checkpoint from `outputs/progress.json`.

To start fresh, delete `outputs/progress.json` and rerun.

## Example: Multi-Batch Training

```bash
# Prepare data from multiple sources
python prepare_data.py --max-articles 50 --output batch1.jsonl
python prepare_data.py --max-articles 50 --output batch2.jsonl

# Combine datasets
cat batch1.jsonl batch2.jsonl > combined.jsonl

# Train on combined data
python run_training.py --data combined.jsonl --output results
```

## Troubleshooting

### Translation Delays
If hitting rate limits, increase delay:
```bash
python prepare_data.py --delay 5.0
```

### Out of VRAM
Reduce batch size during training:
```bash
python run_training.py --batch-size 300
```

### Dataset Download Issues
Ensure HuggingFace access:
```bash
python -c "from datasets import load_dataset; load_dataset('welyjesch/hiligaynon_news_articles')"
```

## Expected Runtime

- **Data Preparation (1000 articles):** ~8-12 hours (depends on internet)
- **Training (1000 samples, 1 epoch):** ~15-30 minutes per batch (on GPU)
- **Inference:** ~1-2 minutes per 10 samples

## Performance Tips

1. **Batch Size:** Larger batches (500-1000) = faster, more GPU memory needed
2. **Learning Rate:** Start with 2e-5, increase to 5e-5 for aggressive training
3. **Epochs:** 1 epoch usually sufficient, 2+ for smaller datasets
4. **GPU Memory:** Monitor with `nvidia-smi` during training

## Next Steps

After training:
- Use trained model for inference with `transformers` pipeline
- Fine-tune further with more data
- Evaluate with BLEU, METEOR, or other metrics
- Deploy model via Hugging Face Model Hub or Inference API
