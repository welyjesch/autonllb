#!/usr/bin/env python3
"""
Data Preparation Script for NLLB Training
Extracts articles from welyjesch/hiligaynon_news_articles dataset,
splits into sentences, translates Hiligaynon→English, and outputs training_data.jsonl

Usage: python prepare_data.py
       or: python prepare_data.py --max-articles 50 --output custom_data.jsonl
"""

import argparse
import json
import re
import sys
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any

import requests
import googletrans
from googletrans import Translator


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


async def translate_sentence(
    translator: Translator,
    sentence: str,
    delay: float = 2.5,
    retries: int = 3
) -> str:
    """Translate a single sentence from Hiligaynon to English with retry logic."""
    for attempt in range(retries):
        try:
            await asyncio.sleep(delay)
            result = await translator.translate(sentence, src='hil', dest='en')
            
            if hasattr(result, 'text') and result.text:
                return result.text
            elif isinstance(result, str):
                return result
            
            return sentence
        except Exception as e:
            print(f"  Translation error (attempt {attempt + 1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
    
    return sentence




# ==============================================================================
# SECTION 2: Data Preparation
# ==============================================================================

async def prepare_training_data(
    output_file: Path,
    max_articles: int = None,
    translation_delay: float = 2.5,
) -> int:
    """
    Extract articles, split into sentences, translate, and save to JSONL.
    
    Returns:
        Total number of sentence pairs written
    """
    print("\n" + "=" * 70)
    print("DATA PREPARATION: Extract, Split, and Translate")
    print("=" * 70)
    
    print("Loading welyjesch/hiligaynon_news_articles dataset...")
    # Manually download the dataset files to avoid httpx compatibility issues
    articles = load_dataset_manually()
    
    if max_articles:
        articles = articles[:max_articles]
    
    print(f"✓ Loaded {len(articles)} articles\n")
    
    translator = Translator()
    total_pairs = 0
    skipped_english = 0
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for article_idx, article in enumerate(articles, 1):
            print(f"[Article {article_idx}/{len(articles)}] Processing...")
            
            # Split into sentences
            sentences = split_into_sentences(article)
            print(f"  Split into {len(sentences)} sentences")
            
            # Translate each sentence
            for sent_idx, sentence in enumerate(sentences, 1):
                # Skip English sentences
                if is_english_sentence(sentence):
                    skipped_english += 1
                    continue
                
                # Generate sequence ID (millisecond timestamp + index)
                seq_id = int(time.time() * 1000) + sent_idx
                
                # Translate
                english = await translate_sentence(translator, sentence, delay=translation_delay)
                
                # Write to JSONL
                pair = {
                    "seq_id": seq_id,
                    "english": english,
                    "hiligaynon": sentence,
                }
                json.dump(pair, outfile, ensure_ascii=False)
                outfile.write('\n')
                total_pairs += 1
                
                # Show progress
                if total_pairs % 50 == 0:
                    print(f"    Progress: {total_pairs} pairs written")
            
            print(f"  Processed {len(sentences)} sentences from this article")
        
        print(f"\n✓ Data preparation complete!")
        print(f"  Total pairs written: {total_pairs}")
        print(f"  English sentences skipped: {skipped_english}")
        print(f"  Output file: {output_file.absolute()}")
    
    return total_pairs


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data for NLLB fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py
  python prepare_data.py --max-articles 100
  python prepare_data.py --output data/custom_training.jsonl --delay 3.0
        """
    )
    
    parser.add_argument(
        '--max-articles',
        type=int,
        default=None,
        help='Maximum number of articles to process (default: all)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("training_data.jsonl"),
        help='Output JSONL file path (default: training_data.jsonl)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=2.5,
        help='Delay between translation requests in seconds (default: 2.5)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NLLB Data Preparation")
    print("=" * 70)
    print(f"Max articles: {args.max_articles if args.max_articles else 'All'}")
    print(f"Output file: {args.output}")
    print(f"Translation delay: {args.delay}s")
    print("=" * 70)
    
    try:
        total_pairs = asyncio.run(prepare_training_data(
            output_file=args.output,
            max_articles=args.max_articles,
            translation_delay=args.delay,
        ))
        
        print("\n" + "=" * 70)
        print(f"✓ Data preparation successful!")
        print(f"  Ready to train with: python run_training.py --data {args.output}")
        print("=" * 70)
        
        return 0
    except KeyboardInterrupt:
        print("\n\nData preparation interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during data preparation: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
