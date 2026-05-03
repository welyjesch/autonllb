# NLLB Autoloop Training: English-Hiligaynon Translation

A modular, production-ready pipeline for fine-tuning the `facebook/nllb-200-distilled-600M` model on English-Hiligaynon bidirectional translation using the `welyjesch/hiligaynon_news_articles` dataset.

## Features

✓ **Modular Design** - Separated data prep and training pipelines  
✓ **Automatic Dataset Extraction** - Loads articles from HuggingFace dataset  
✓ **Smart Sentence Splitting** - Pattern-based extraction with abbreviation handling  
✓ **Intelligent Translation** - Filters English sentences, translates Hiligaynon→English via googletrans v4  
✓ **Sequence Tracking** - Sentences stored with millisecond-precision IDs  
✓ **Batch Training** - Incremental training with configurable batch sizes  
✓ **Progress Persistence** - Automatic checkpointing for resumable training  
✓ **Organized Output** - Model checkpoints, logs, and results in structured directories  
✓ **Validation Testing** - Inference tests on random samples from trained data  

## Project Structure

```
.
├── pyproject.toml           # Project dependencies (uv/pip)
├── prepare_data.py          # Data extraction and translation
├── run_training.py          # Model training and inference
├── install.py               # Python installation script
├── install.sh               # Bash installation script
├── translate.py             # Standalone translation utility (optional)
├── README.md                # This file
└── .gitignore               # Git ignore file
```

## Quick Start

### Step 1: Install Dependencies

**Option A: Using Python installer**
```bash
python install.py
```

**Option B: Using Bash installer (Linux/macOS)**
```bash
chmod +x install.sh
./install.sh
```

**Option C: Manual with uv**
```bash
pip install uv
uv pip install transformers datasets sentencepiece accelerate googletrans==4.0.0-rc1
```

### Step 2: Prepare Data

Extract, split, and translate articles:

```bash
python prepare_data.py
```

This creates `training_data.jsonl` with fields: `seq_id`, `english`, `hiligaynon`

**Custom options:**
```bash
python prepare_data.py --max-articles 100                    # Limit articles
python prepare_data.py --output data/custom.jsonl            # Custom output file
python prepare_data.py --delay 3.0                           # Translation delay (seconds)
```

### Step 3: Train Model

Train on prepared data with automatic batching:

```bash
python run_training.py
```

**Custom options:**
```bash
python run_training.py --batch-size 500              # Batch size (default: 1000)
python run_training.py --num-epochs 2                # Epochs per batch (default: 1)
python run_training.py --learning-rate 1e-5          # Learning rate (default: 2e-5)
python run_training.py --inference-samples 20        # Inference test samples (default: 10)
python run_training.py --output ./my_results         # Custom output directory
```

## Workflow

### Stage 1: Data Preparation (`prepare_data.py`)
1. Loads `welyjesch/hiligaynon_news_articles` from HuggingFace
2. Splits articles into sentences using regex patterns with abbreviation handling
3. Filters out English sentences
4. Translates Hiligaynon→English using googletrans
5. Outputs `training_data.jsonl` with `seq_id`, `english`, `hiligaynon` fields

### Stage 2: Model Training (`run_training.py`)
1. Loads prepared `training_data.jsonl`
2. Sets up NLLB-200-distilled-600M model and tokenizer
3. Trains in configurable batches (default: 1000 samples per batch)
4. Saves model checkpoints, logs, and progress after each batch
5. Tracks training state with progress tracker for resumability

### Stage 3: Inference Validation
1. Tests trained model on random samples from training data
2. Compares reference vs. predicted translations
3. Saves results to JSON for analysis

## Output Structure

After running both scripts:

```
outputs/
├── progress.json                    # Training state (resumable)
├── inference_results.json           # Test translations (ref vs predicted)
├── batch_0/
│   ├── model/                       # Trained model checkpoint
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── sentencepiece.bpe.model
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   └── ...
│   └── logs/                        # Training logs
│       └── events.out.tfevents.*
├── batch_1/
│   ├── model/
│   └── logs/
└── ...

training_data.jsonl                 # Prepared training data
```

## Progress Tracking

Training state is saved to `outputs/progress.json`:

```json
{
  "total_samples": 5000,
  "current_batch": 3,
  "batches_trained": [
    {
      "batch_index": 0,
      "model_dir": "outputs/batch_0/model",
      "timestamp": "2026-05-02T12:00:00.000000"
    },
    {
      "batch_index": 1,
      "model_dir": "outputs/batch_1/model",
      "timestamp": "2026-05-02T13:15:30.000000"
    }
  ],
  "start_time": "2026-05-02T11:45:00.000000",
  "last_update": "2026-05-02T13:45:00.000000"
}
```

**Resuming Training:** Simply run `run_training.py` again; it will resume from the last completed batch.

## Configuration

### prepare_data.py

```python
python prepare_data.py \
    --max-articles 50           # Number of articles (None = all)
    --output my_data.jsonl      # Output JSONL file
    --delay 2.5                 # Delay between translations (seconds)
```

### run_training.py

```python
python run_training.py \
    --data training_data.jsonl              # Input data file
    --output outputs                        # Output directory
    --batch-size 1000                       # Samples per batch
    --num-epochs 1                          # Epochs per batch
    --learning-rate 2e-5                    # Learning rate
    --inference-samples 10                  # Validation samples
    --model facebook/nllb-200-distilled-600M  # Model to fine-tune
```

## Standalone Tools

### Translate Utility

Translate individual JSONL files:

```bash
python translate.py dataset.json -o translated.jsonl --delay 2.5
```

## Requirements

- Python 3.9+
- ~8GB VRAM (for NLLB-200-distilled-600M)
- Internet connection (for dataset and model downloads)

## Key Dependencies

- `transformers` - Model loading and training
- `datasets` - HuggingFace dataset utilities
- `sentencepiece` - Tokenization
- `accelerate` - Distributed training support
- `googletrans` - Translation API (v4.0.0-rc1)

## Troubleshooting

### Translation Rate Limits
If hitting googletrans rate limits:
```bash
python prepare_data.py --delay 5.0  # Increase delay
```

### CUDA Memory Issues
If running out of VRAM, modify `train_batch()` in `run_training.py`:
```python
per_device_train_batch_size=4,  # Instead of 8
```

### Dataset Download Issues
- Check internet connection
- For authentication: `huggingface-cli login`

### Resume Interrupted Training
Delete `outputs/progress.json` to start fresh, or keep it to resume.

## Advanced Usage

### Custom Model
Change model in `run_training.py`:
```bash
python run_training.py --model facebook/nllb-200-1.3B
```

### Different Batch Configuration
```bash
python run_training.py --batch-size 500 --num-epochs 2
```

### Prepare Multiple Datasets
```bash
python prepare_data.py --max-articles 50 --output data_batch_1.jsonl
python prepare_data.py --max-articles 100 --output data_batch_2.jsonl

# Then train on combined data
cat data_batch_1.jsonl data_batch_2.jsonl > combined_data.jsonl
python run_training.py --data combined_data.jsonl
```

## Notes

- Sequence IDs are based on millisecond timestamps for uniqueness
- Sentences shorter than 10 characters are filtered during preparation
- English sentences are automatically skipped during translation
- Models are saved after each batch, allowing partial training results
- Progress is automatically saved after each batch for resumability

## License

This project uses open-source models and datasets. See individual projects for license information:
- [NLLB Model](https://huggingface.co/facebook/nllb-200-distilled-600M) - CC-BY-NC-4.0
- [Hiligaynon News Articles](https://huggingface.co/datasets/welyjesch/hiligaynon_news_articles)
- [Transformers](https://github.com/huggingface/transformers) - Apache 2.0

## References

- [NLLB Model Card](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [Hiligaynon Dataset](https://huggingface.co/datasets/welyjesch/hiligaynon_news_articles)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [googletrans GitHub](https://github.com/ssut/py-googletrans)
