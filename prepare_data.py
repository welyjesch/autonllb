#!/usr/bin/env python3
"""
[DEPRECATED] Data Preparation Script for NLLB Training

⚠️  This module is DEPRECATED and no longer maintained.
    Use the new decoupled workflow instead:
    
    1. Sentence chunking:    python prepare_sentences.py chunk --max-articles 100
    2. Batch division:       python batch_sentence_chunks.py --num-batches 4
    3. Parallel translation: python prepare_translations.py translate --chunk-batch-manifest chunks/batch_0_manifest.json
    4. Assembly:             python prepare_translations.py assemble

The new workflow provides:
  ✓ Separation of concerns (sentences vs. translation)
  ✓ Resilience to translation failures (chunking progress preserved)
  ✓ Parallel processing capability for faster translation
  ✓ Better progress tracking (file-based, not row-indexed)

Legacy documentation (archived):
  Extracts articles from welyjesch/hiligaynon_news_articles dataset,
  splits into sentences, translates Hiligaynon→English, and outputs training_data.jsonl
"""

import sys

# ==============================================================================
# DEPRECATED - ARCHIVED CODE
# ==============================================================================
# 
# All the original functions and classes have been archived.
# This file now only displays a deprecation notice.
# 
# The implementation has been split into:
#   - prepare_sentences.py (sentence extraction and chunking)
#   - prepare_translations.py (translation and assembly)
#   - batch_sentence_chunks.py (batch division for parallel processing)
#
# ==============================================================================


def main():
    """Entry point - shows deprecation notice only."""
    print("\n" + "=" * 70)
    print("⚠️  DEPRECATION NOTICE")
    print("=" * 70)
    print("\nThis module (prepare_data.py) is DEPRECATED and no longer maintained.")
    print("\nUse the new decoupled workflow instead:\n")
    print("  Step 1 - Sentence chunking:")
    print("    python prepare_sentences.py chunk --max-articles 100\n")
    print("  Step 2 - Batch division (for parallel processing):")
    print("    python batch_sentence_chunks.py --num-batches 4\n")
    print("  Step 3 - Parallel translation (run in separate terminals):")
    print("    python prepare_translations.py translate --chunk-batch-manifest chunks/batch_0_manifest.json")
    print("    python prepare_translations.py translate --chunk-batch-manifest chunks/batch_1_manifest.json")
    print("    ... (repeat for each batch)\n")
    print("  Step 4 - Assembly (after all workers finish):")
    print("    python prepare_translations.py assemble --output training_data.jsonl\n")
    print("Benefits of the new workflow:")
    print("  ✓ Separation of concerns (sentences vs. translation)")
    print("  ✓ Resilience to translation failures (chunking progress preserved)")
    print("  ✓ Parallel processing capability for faster translation")
    print("  ✓ Better progress tracking (file-based, not row-indexed)")
    print("\n" + "=" * 70 + "\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
