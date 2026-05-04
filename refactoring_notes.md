# Data Preparation Refactoring: Decoupled 3-Phase Workflow

## Overview

The data preparation pipeline has been refactored to **decouple chunking from translation**, protecting against data loss and simplifying progress tracking.

### Problem Solved
- **Previous Issue**: Translation errors/bugs could cause loss of progress on sentence chunking
- **Progress Tracking**: Previous row-based tracking was fragile; now uses chunk-based filepath tracking
- **Context Preservation**: Each chunk respects article boundaries so context is maintained

---

## New Architecture

### Phase 1: Chunking
**Command**: `python prepare_data.py chunk [options]`

- Extracts articles from Hiligaynon News dataset
- Splits each article into sentences (with smart abbreviation handling)
- Filters out English sentences
- Saves sentence batches to disk as `translated_data_chunk_<N>.txt` files
- **Creates**: `chunks/manifest.json` to track all chunks

**Key Files Created**:
```
chunks/
├── manifest.json                       # Progress tracking
├── translated_data_chunk_1.txt         # ~100 sentences (default)
├── translated_data_chunk_2.txt
└── ...
```

**Options**:
```
--max-articles N          Process only first N articles (default: all)
--chunks-dir PATH         Directory for chunk files (default: chunks/)
--chunk-size N            Sentences per chunk file (default: 100)
```

**Example**:
```bash
python prepare_data.py chunk --max-articles 50 --chunk-size 200
```

---

### Phase 2: Translation
**Command**: `python prepare_data.py translate [options]`

- Reads all untranslated chunks from `chunks/` directory
- Translates each sentence using Google Translate (Hiligaynon → English)
- Saves translated sentences to `translated_chunk_<N>.txt`
- Updates manifest with translation status and file paths

**Key Files Created**:
```
chunks/
├── translated_data_chunk_1.txt         # Original sentences
├── translated_chunk_1.txt              # Translated sentences
├── translated_data_chunk_2.txt
├── translated_chunk_2.txt
└── manifest.json                       # Updated with is_translated=true
```

**Options**:
```
--chunks-dir PATH         Directory containing chunks (default: chunks/)
--delay N                 Seconds between translation requests (default: 2.5)
```

**Example**:
```bash
python prepare_data.py translate --delay 3.0
```

**Resume**: Automatically skips chunks marked as translated in manifest.json

---

### Phase 3: Assembly
**Command**: `python prepare_data.py assemble [options]`

- Pairs original + translated sentences
- Creates JSONL output with structure:
  ```json
  {
    "seq_id": 1234567890001,
    "english": "Hello world",
    "hiligaynon": "Halo mundo",
    "chunk_id": 1
  }
  ```
- Reads metadata from `chunks/manifest.json`

**Key Files Created**:
```
training_data.jsonl        # Final output ready for training
```

**Options**:
```
--chunks-dir PATH         Directory containing chunks (default: chunks/)
--output FILE             Output JSONL file (default: training_data.jsonl)
```

**Example**:
```bash
python prepare_data.py assemble --output my_training_data.jsonl
```

---

## Run All Phases

**Command**: `python prepare_data.py all [options]`

Runs phases 1 → 2 → 3 sequentially with a single command.

**Options**:
```
--max-articles N          Articles to process (Phase 1)
--chunks-dir PATH         Chunk directory
--chunk-size N            Sentences per chunk
--delay N                 Translation delay (Phase 2)
--output FILE             Output file (Phase 3)
```

**Example**:
```bash
python prepare_data.py all --max-articles 100 --delay 3.0
```

---

## Progress Tracking

### Manifest Structure (`chunks/manifest.json`)

```json
{
  "chunks": {
    "1": {
      "file_path": "chunks/translated_data_chunk_1.txt",
      "sentence_count": 100,
      "is_translated": true,
      "created_at": "2026-05-04T10:30:00.000000",
      "translated_at": "2026-05-04T11:45:00.000000",
      "translated_file": "chunks/translated_chunk_1.txt"
    },
    "2": {
      "file_path": "chunks/translated_data_chunk_2.txt",
      "sentence_count": 87,
      "is_translated": false,
      "created_at": "2026-05-04T10:35:00.000000",
      "translated_at": null,
      "translated_file": null
    }
  },
  "metadata": {
    "created_at": "2026-05-04T10:30:00.000000",
    "last_updated": "2026-05-04T11:45:00.000000"
  }
}
```

**Benefits**:
- ✅ Simple boolean `is_translated` flag per chunk
- ✅ Tracks file paths, not fragile row indices
- ✅ Timestamps for debugging and resumption
- ✅ Human-readable JSON format

---

## Recovery & Error Handling

### If Translation Fails Mid-Phase

1. Check `chunks/manifest.json` to see which chunks are incomplete
2. Fix the issue (e.g., network, API limits)
3. Re-run: `python prepare_data.py translate`
   - Automatically skips translated chunks
   - Retries untranslated ones

### If Assembly Fails

1. Verify all chunks in manifest show `is_translated: true`
2. Check file paths exist
3. Re-run: `python prepare_data.py assemble`
   - Skips any chunks without `is_translated: true`

### Manual Cleanup

To restart from scratch:
```bash
rm -rf chunks/
python prepare_data.py all --max-articles 50
```

---

## Example Workflow

```bash
# 1. Create chunks (can stop here if only doing prep work)
python prepare_data.py chunk --max-articles 100

# Check progress
cat chunks/manifest.json | jq '.metadata'

# 2. Translate chunks (can retry this step multiple times)
python prepare_data.py translate --delay 3.0

# 3. Assemble into final JSONL
python prepare_data.py assemble --output training_data.jsonl

# Ready to train!
python run_training.py --data training_data.jsonl
```

---

## Code Changes

### New Classes
- **`ChunkManager`**: Manages chunk files and manifest.json
  - `save_chunk()`: Save sentences to file + track in manifest
  - `mark_translated()`: Update completion status
  - `get_untranslated_chunks()`: Find incomplete work
  - `get_status()`: Human-readable progress

### New Functions
- **`phase_chunk()`**: Phase 1 implementation
- **`phase_translate()`**: Phase 2 implementation  
- **`phase_assemble()`**: Phase 3 implementation

### Removed
- Old `prepare_training_data()` monolithic function
- Fragile `skipped_rows` list tracking
- Row-based progress estimation

---

## Migration from Old Script

If you have existing `training_data.jsonl` files:

1. **Keep them**: They're not affected by this refactor
2. **Or regenerate** using new workflow:
   ```bash
   # Start fresh
   rm -rf chunks/
   rm training_data.jsonl
   python prepare_data.py all --max-articles ALL
   ```

---

## Next Steps

- Phase 2 can now be run independently without losing Phase 1 work
- Easy to parallelize Phase 2 if needed (multiple translation workers)
- Chunk-based approach scales better with large datasets
- Progress is visible and recoverable at each phase
