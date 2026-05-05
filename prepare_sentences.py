#!/usr/bin/env python3
"""
Sentence Preparation Module for NLLB Training
Extracts articles from welyjesch/hiligaynon_news_articles dataset,
splits into sentences, and saves chunks to disk.

This module is decoupled from translation to preserve progress and enable
parallel translation via batch_sentence_chunks.py

Usage:
  python prepare_sentences.py chunk --max-articles 100
  python prepare_sentences.py chunk --max-articles 50 --chunk-size 200
"""

import argparse
import json
import re
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import requests


# ==============================================================================
# SECTION 0: Dataset Loading
# ==============================================================================

def load_dataset_manually() -> List[str]:
    """Manually download and load the dataset to avoid httpx compatibility issues."""
    # Download the parquet file using the direct download link
    url = "https://huggingface.co/datasets/welyjesch/hiligaynon_news_articles/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    
    # Download the file
    import requests
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download dataset: {response.status_code}")
    
    # Save the file temporarily
    temp_file = "temp_dataset.parquet"
    with open(temp_file, "wb") as f:
        f.write(response.content)
    
    # Load the dataset using pandas
    import pandas as pd
    df = pd.read_parquet(temp_file)
    
    # Extract the articles
    articles = df["text"].tolist()
    
    # Clean up
    import os
    os.remove(temp_file)
    
    return articles


# ==============================================================================
# SECTION 1: Sentence Extraction and Splitting
# ==============================================================================

ABBREVIATIONS = [
    # Version numbers - MORE SPECIFIC patterns first!
    r'\b\d+\.\d+\.\d+\b',  # 3-part version (2.5.1) BEFORE 2-part (2.5)
    r'\b\d+\.\d+\b',       # 2-part version (2.5)
    r'\.\.\.+',
    r'\b\d+\.',
    r'\b[A-Z]\.',
    r'\bDr\.', r'\bMr\.', r'\bMrs\.', r'\bMs\.', r'\bProf\.', 
    r'\bSr\.', r'\bJr\.', r'\bSt\.', r'\bFr\.', r'\bRev\.',
    r'\bGov\.', r'\bSen\.', r'\bRep\.', r'\bPres\.', r'\bGen\.', 
    r'\bCol\.', r'\bLt\.', r'\bSgt\.', r'\bCpl\.', r'\bCapt\.', 
    r'\bCmdr\.', r'\bAdm\.', r'\bAmb\.', r'\bHon\.',
    r'\bPhD\.', r'\bMD\.', r'\bBA\.', r'\bBS\.', r'\bMA\.', r'\bMS\.',
    r'\bInc\.', r'\bLtd\.', r'\bCo\.', r'\bCorp\.', r'\bLLC\.',
    r'\bvs\.', r'\betc\.', r'\bi\.e\.', r'\be\.g\.',
    r'\bFig\.', r'\bfig\.', r'\bVol\.', r'\bvol\.', 
    r'\bNo\.', r'\bno\.', r'\bpp\.', r'\bpg\.', r'\bp\.',
    r'\bed\.', r'\beds\.', r'\best\.', r'\bapprox\.', 
    r'\bca\.', r'\bcf\.', r'\bviz\.', r'\bal\.', r'\bet al\.',
    # NOTE: Don't add comma-prefixed patterns like r',\s*etc\.' - they break sentence splitting!
    # The \betc\. pattern already matches "etc." after commas via word boundary
]

ABBREVIATION_PATTERN = '(' + '|'.join(ABBREVIATIONS) + r')'

def is_english_sentence(sentence: str) -> bool:
    """Detect if a sentence is likely in English using langdetect."""
    if not sentence or not sentence.strip():
        return False
    try:
        return detect(sentence) == 'en'
    except Exception:
        return False

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex pattern."""
    PLACEHOLDER = '<<ABBR_PERIOD>>'
    
    def protect_abbreviation(match):
        abbrev = match.group(0)
        
        # NEVER replace "etc." - it should never be a sentence boundary
        if abbrev == 'etc.':
            return abbrev
        
        # PROTECT ELLIPSIS - do not replace dots in "..."
        if '...' in abbrev:
            return abbrev
        
        end_pos = match.end()
        
        # Don't replace when followed by comma
        if end_pos < len(text) and text[end_pos] == ',':
            return abbrev
        
        # Don't replace when at end of string
        if end_pos >= len(text):
            return abbrev
        
        return abbrev.replace('.', PLACEHOLDER)
    
    protected_text = re.sub(ABBREVIATION_PATTERN, protect_abbreviation, text)
    
    # Split on period followed by space + capital letter
    # The negative lookbehind ensures we don't split after a letter (like in abbreviations)
    sentences = re.split(r'(?<=[.!?])\s+', protected_text)
    sentences = [s.replace(PLACEHOLDER, '.') for s in sentences]
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s) >= 10]
    
    return sentences


# ==============================================================================
# SECTION 2: Chunking Management (Decoupled from Translation)
# ==============================================================================

class ChunkManager:
    """Manage sentence chunks and translation progress tracking."""
    
    def __init__(self, chunks_dir: Path = Path("chunks")):
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.chunks_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load or initialize the translation progress manifest."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {
            "chunks": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            }
        }
    
    def _save_manifest(self):
        """Save manifest to disk."""
        self.manifest["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.manifest_file, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)
    
    def save_chunk(self, chunk_number: int, sentences: List[str]) -> Path:
        """
        Save a chunk of sentences to disk.
        
        Args:
            chunk_number: Sequential chunk ID
            sentences: List of sentences (one per line)
        
        Returns:
            Path to saved chunk file
        """
        chunk_file = self.chunks_dir / f"translated_data_chunk_{chunk_number}.txt"
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        
        # Track in manifest
        self.manifest["chunks"][str(chunk_number)] = {
            "file_path": str(chunk_file),
            "sentence_count": len(sentences),
            "is_translated": False,
            "created_at": datetime.now().isoformat(),
            "translated_at": None,
        }
        self._save_manifest()
        
        return chunk_file
    
    def get_untranslated_chunks(self) -> List[int]:
        """Get list of chunk numbers that haven't been translated yet."""
        untranslated = []
        for chunk_id, chunk_info in self.manifest["chunks"].items():
            if not chunk_info.get("is_translated", False):
                untranslated.append(int(chunk_id))
        return sorted(untranslated)
    
    def get_chunk_file(self, chunk_number: int) -> Path:
        """Get the file path for a chunk."""
        return self.chunks_dir / f"translated_data_chunk_{chunk_number}.txt"
    
    def get_status(self) -> str:
        """Get formatted status of all chunks."""
        total = len(self.manifest["chunks"])
        translated = sum(1 for c in self.manifest["chunks"].values() if c.get("is_translated", False))
        return f"Chunks: {translated}/{total} translated"


# ==============================================================================
# SECTION 3: Phase 1 - Chunking (Decouple from Translation)
# ==============================================================================

async def phase_chunk(
    max_articles: int = None,
    chunks_dir: Path = Path("chunks"),
    chunk_size: int = 100,
) -> int:
    """
    Phase 1: Split articles into sentences and save as chunks.
    
    Args:
        max_articles: Limit articles to process (None = all)
        chunks_dir: Directory to store chunk files
        chunk_size: Approximate sentences per chunk file
    
    Returns:
        Total chunks created
    """
    print("\n" + "=" * 70)
    print("PHASE 1: CHUNKING - Extract and split sentences")
    print("=" * 70)
    
    print("Loading welyjesch/hiligaynon_news_articles dataset...")
    articles = load_dataset_manually()
    
    if max_articles:
        articles = articles[:max_articles]
    
    print(f"✓ Loaded {len(articles)} articles\n")
    
    chunk_manager = ChunkManager(chunks_dir=chunks_dir)
    chunk_number = len(chunk_manager.manifest["chunks"])  # Resume from last chunk
    current_chunk_sentences = []
    skipped_english = 0
    total_sentences = 0
    
    for article_idx, article in enumerate(articles, 1):
        print(f"[Article {article_idx}/{len(articles)}] Splitting into sentences...")
        
        # Split article into sentences
        sentences = split_into_sentences(article)
        print(f"  Found {len(sentences)} sentences")
        
        # Filter out English sentences
        filtered_sentences = []
        for sent in sentences:
            if is_english_sentence(sent):
                skipped_english += 1
            else:
                filtered_sentences.append(sent)
        
        print(f"  Kept {len(filtered_sentences)} (skipped {skipped_english - sum(1 for s in sentences if is_english_sentence(s))} English)")
        
        total_sentences += len(filtered_sentences)
        
        # Add to current chunk
        for sentence in filtered_sentences:
            current_chunk_sentences.append(sentence)
            
            # Save chunk if it reaches desired size
            if len(current_chunk_sentences) >= chunk_size:
                chunk_number += 1
                chunk_file = chunk_manager.save_chunk(chunk_number, current_chunk_sentences)
                print(f"  Saved chunk {chunk_number} ({len(current_chunk_sentences)} sentences) → {chunk_file.name}")
                current_chunk_sentences = []
    
    # Save remaining sentences as final chunk
    if current_chunk_sentences:
        chunk_number += 1
        chunk_file = chunk_manager.save_chunk(chunk_number, current_chunk_sentences)
        print(f"  Saved chunk {chunk_number} ({len(current_chunk_sentences)} sentences) → {chunk_file.name}")
    
    print(f"\n✓ Chunking complete!")
    print(f"  Total sentences processed: {total_sentences}")
    print(f"  English sentences skipped: {skipped_english}")
    print(f"  Chunks created: {chunk_number}")
    print(f"  Chunks directory: {chunks_dir.absolute()}")
    print(f"  {chunk_manager.get_status()}")
    
    return chunk_number


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare sentence chunks for NLLB translation (Phase 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extract articles and split into sentence chunks saved to disk.
Decoupled from translation to preserve progress.

Examples:
  python prepare_sentences.py chunk --max-articles 100
  python prepare_sentences.py chunk --max-articles 50 --chunk-size 200
  python prepare_sentences.py chunk  # Process all articles
        """
    )
    
    subparsers = parser.add_subparsers(dest='phase', help='Preparation phase')
    
    # Chunk
    chunk_parser = subparsers.add_parser('chunk', help='Phase 1: Create sentence chunks')
    chunk_parser.add_argument(
        '--max-articles',
        type=int,
        default=None,
        help='Maximum number of articles to process (default: all)'
    )
    chunk_parser.add_argument(
        '--chunks-dir',
        type=Path,
        default=Path("chunks"),
        help='Directory to store chunks (default: chunks/)'
    )
    chunk_parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Approximate sentences per chunk file (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Show phase banner
    print("=" * 70)
    print("NLLB Sentence Preparation (Phase 1: Chunking)")
    print("=" * 70)
    
    try:
        if args.phase == 'chunk':
            print(f"Phase: CHUNKING")
            print(f"Max articles: {args.max_articles if args.max_articles else 'All'}")
            print(f"Chunk size: {args.chunk_size} sentences")
            print(f"Chunks directory: {args.chunks_dir}")
            print("=" * 70)
            asyncio.run(phase_chunk(
                max_articles=args.max_articles,
                chunks_dir=args.chunks_dir,
                chunk_size=args.chunk_size,
            ))
            return 0
        
        else:
            print("No phase specified. Use: chunk")
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\n\nSentence preparation interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during sentence preparation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
