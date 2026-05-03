#!/usr/bin/env python3
"""
Translate a split Hiligaynon dataset part using Google Translate.

This script reads a JSONL file containing Hiligaynon text, splits each article
into sentences, translates them using googletrans, and outputs a new JSONL file
with {"hiligaynon": <text>, "english": <text>} format for merging.

Usage:
    python translate.py dataset_part_1.jsonl --output translated_part_1.jsonl
    python translate.py dataset_part_1.jsonl -o translated_part_1.jsonl --delay 3
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    from googletrans import Translator
except ImportError:
    print("Error: googletrans library not installed.")
    print("Install it with: pip install googletrans==4.0.0-rc1")
    sys.exit(1)


# Common abbreviations and titles that shouldn't trigger sentence splits
# This includes honorifics, academic titles, business suffixes, Latin abbreviations,
# decimal numbers, ellipsis, numbered lists, and single initials
ABBREVIATIONS = [
    # Decimal numbers (e.g., 3.14, 0.5, 100.00) - must come first
    r'\b\d+\.\d+\b',
    # Ellipsis (three or more dots)
    r'\.\.\.+',
    # Numbered lists (e.g., 1., 2., 9., 10., etc.)
    r'\b\d+\.',
    # Single letter initials (e.g., D., C., J., M., etc.)
    r'\b[A-Z]\.',
    # Honorifics and titles
    r'\bDr\.', r'\bMr\.', r'\bMrs\.', r'\bMs\.', r'\bProf\.', 
    r'\bSr\.', r'\bJr\.', r'\bSt\.', r'\bFr\.', r'\bRev\.',
    # Government and military titles
    r'\bGov\.', r'\bSen\.', r'\bRep\.', r'\bPres\.', r'\bGen\.', 
    r'\bCol\.', r'\bLt\.', r'\bSgt\.', r'\bCpl\.', r'\bCapt\.', 
    r'\bCmdr\.', r'\bAdm\.', r'\bAmb\.', r'\bHon\.',
    # Academic and professional degrees
    r'\bPhD\.', r'\bMD\.', r'\bBA\.', r'\bBS\.', r'\bMA\.', r'\bMS\.',
    # Business suffixes
    r'\bInc\.', r'\bLtd\.', r'\bCo\.', r'\bCorp\.',
    # Common abbreviations
    r'\bvs\.', r'\betc\.', r'\bi\.e\.', r'\be\.g\.',
    # Reference abbreviations
    r'\bFig\.', r'\bfig\.', r'\bVol\.', r'\bvol\.', 
    r'\bNo\.', r'\bno\.', r'\bpp\.', r'\bpg\.', r'\bp\.',
    # Editorial and other abbreviations
    r'\bed\.', r'\beds\.', r'\best\.', r'\bapprox\.', 
    r'\bca\.', r'\bcf\.', r'\bviz\.', r'\bal\.', r'\bet al\.',
]

# Build a regex pattern to protect these abbreviations
ABBREVIATION_PATTERN = '(' + '|'.join(ABBREVIATIONS) + r')'

# Common English words that indicate a sentence is in English
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
    'came', 'made', 'may', 'should', 'us', 'now', 'over', 'most', 'into',
]

# English-specific patterns
ENGLISH_PATTERNS = [
    r'\b(the|and|that|with|for|this|from|have|not)\b',
    r'\b(was|were|are|is|been|being|be)\b',
    r'\b(will|would|could|should|may|might|must|can)\b',
    r'\b(he|she|it|they|we|you|i)\b',
    r'\b(his|her|their|our|your|my|its)\b',
    r'\b(in|on|at|to|for|by|with|from|into|through|during)\b',
]


def is_english_sentence(sentence: str, threshold: float = 0.3) -> bool:
    """
    Detect if a sentence is likely in English based on common English words.
    
    Args:
        sentence: The sentence to analyze.
        threshold: Minimum ratio of English indicators to classify as English.
    
    Returns:
        True if the sentence appears to be in English.
    """
    if not sentence or not sentence.strip():
        return False
    
    text_lower = sentence.lower()
    english_word_count = 0
    
    for pattern in ENGLISH_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        english_word_count += len(matches)
    
    for indicator in ENGLISH_INDICATORS:
        if indicator in text_lower:
            english_word_count += 1
    
    words = sentence.split()
    if len(words) == 0:
        return False
    
    english_ratio = english_word_count / len(words)
    return english_ratio >= threshold


def split_into_sentences(text: str) -> List[str]:
    """
    Split Hiligaynon text into sentences using regex pattern.
    
    Uses a two-pass approach:
    1. Temporarily replace periods in common abbreviations with a placeholder
    2. Split on sentence boundaries
    3. Restore the periods in abbreviations
    
    Args:
        text: Hiligaynon text to split
        
    Returns:
        List of sentences
    """
    # Placeholder that won't appear in normal text
    PLACEHOLDER = '<<ABBR_PERIOD>>'
    
    # Step 1: Protect abbreviations by replacing their periods
    def protect_abbreviation(match):
        return match.group(0).replace('.', PLACEHOLDER)
    
    protected_text = re.sub(ABBREVIATION_PATTERN, protect_abbreviation, text)
    
    # Split on sentence boundaries (period, exclamation, question mark)
    sentences = re.split(r'[.!?]+\s*', protected_text)
    
    # Step 3: Restore the periods in abbreviations
    sentences = [s.replace(PLACEHOLDER, '.') for s in sentences]
    
    # Filter out empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Filter out very short sentences (less than 10 characters) - likely artifacts or mistakes
    sentences = [s for s in sentences if len(s) >= 10]
    
    return sentences


def translate_sentence(translator: Translator, sentence: str, delay: float = 2.5, retries: int = 3) -> str:
    """
    Translate a single sentence from Hiligaynon to English with retry logic.
    
    Args:
        translator: googletrans Translator instance
        sentence: Hiligaynon sentence to translate
        delay: Delay between translation requests in seconds
        retries: Number of retry attempts on failure
        
    Returns:
        Translated English sentence
    """
    for attempt in range(retries):
        try:
            # Add delay to avoid rate limiting
            time.sleep(delay)
            
            # Use the synchronous translate method
            result = translator.translate(sentence, src='hil', dest='en')
            
            # Handle both sync and async result types
            if hasattr(result, 'text'):
                if result.text:
                    return result.text
            elif isinstance(result, str):
                return result
            
            print(f"  Warning: Empty translation for sentence: {sentence[:50]}...")
            return sentence  # Return original if translation fails
            
        except Exception as e:
            print(f"  Error translating (attempt {attempt + 1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                # Exponential backoff
                time.sleep(delay * (attempt + 1))
            else:
                print(f"  Failed after {retries} attempts, keeping original")
                return sentence  # Return original on final failure
    
    return sentence


def translate_article(translator: Translator, article: str, delay: float = 2.5) -> List[Dict[str, str]]:
    """
    Translate a full Hiligaynon article by splitting into sentences.
    
    Args:
        translator: googletrans Translator instance
        article: Full Hiligaynon article text
        delay: Delay between sentence translations
        
    Returns:
        List of {"hiligaynon": <text>, "english": <text>} dictionaries
    """
    sentences = split_into_sentences(article)
    translated_pairs = []
    english_skipped = 0
    
    print(f"  Translating {len(sentences)} sentences...")
    
    for i, sentence in enumerate(sentences, 1):
        if not sentence.strip():
            continue
        
        # Skip English sentences
        if is_english_sentence(sentence):
            english_skipped += 1
            continue
            
        # Show progress
        if i % 10 == 0 or i == len(sentences):
            print(f"    Progress: {i}/{len(sentences)} sentences")
        
        print(f"  Translating: {sentence[:60]}...")
        # Translate the sentence
        english = translate_sentence(translator, sentence, delay)
        
        # Create the pair
        translated_pairs.append({
            "hiligaynon": sentence,
            "english": english
        })
    
    if english_skipped > 0:
        print(f"  Skipped {english_skipped} English sentence(s)")
    
    return translated_pairs


def process_json_file(input_path: Path, output_path: Path, delay: float = 2.5) -> None:
    """
    Process a JSON file containing Hiligaynon sentences and translate them.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output translated JSONL file
        delay: Delay between translation requests
    """
    translator = Translator()
    total_pairs = 0
    
    print(f"Reading input file: {input_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read the entire JSON file
    with open(input_path, 'r', encoding='utf-8') as infile:
        try:
            data = json.load(infile)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            return
    
    # Process each sentence
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item_num, item in enumerate(data, 1):
            if not isinstance(item, dict):
                print(f"Warning: Item {item_num} is not a dictionary, skipping")
                continue
            
            # Extract Hiligaynon text
            hiligaynon_text = item.get('hiligaynon')
            if not hiligaynon_text:
                print(f"Warning: No Hiligaynon text found in item {item_num}, skipping")
                continue
            
            # Get article and sentence indices if available
            article_index = item.get('article_index', 0)
            sentence_index = item.get('sentence_index', 0)
            
            print(f"\nProcessing sentence {item_num} (article {article_index}, sentence {sentence_index}): {len(hiligaynon_text)} characters")
            
            # Skip English sentences
            if is_english_sentence(hiligaynon_text):
                print(f"  Skipped English sentence")
                continue
            
            # Translate the sentence
            english = translate_sentence(translator, hiligaynon_text, delay)
            
            # Write the pair
            pair = {
                "article_index": article_index,
                "sentence_index": sentence_index,
                "hiligaynon": hiligaynon_text,
                "english": english
            }
            json.dump(pair, outfile, ensure_ascii=False)
            outfile.write('\n')
            total_pairs += 1
    
    print(f"\n✓ Translation complete!")
    print(f"  Total sentence pairs written: {total_pairs}")
    print(f"  Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Translate a split Hiligaynon dataset part using Google Translate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python translate.py dataset_part_1.json
  python translate.py dataset_part_1.json -o translated_part_1.jsonl
  python translate.py dataset_part_1.json -o translated_part_1.jsonl --delay 3
  python translate.py hiligaynon_sentences_1.json -o translated_1.jsonl --delay 2
        """
    )
    
    parser.add_argument(
        'input',
        type=Path,
        help='Input JSON file containing Hiligaynon text (e.g., dataset_part_1.json)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output JSONL file for translated pairs (default: translated_<input_name>.jsonl)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=2.5,
        help='Delay between translation requests in seconds (default: 2.5, increase to avoid rate limits)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        output_name = f"translated_{args.input.stem}.jsonl"
        args.output = args.input.parent / output_name
    
    print("=" * 70)
    print("Hiligaynon Dataset Translation Script")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Delay:  {args.delay}s between requests")
    print("=" * 70)
    
    try:
        process_json_file(args.input, args.output, args.delay)
    except KeyboardInterrupt:
        print("\n\nTranslation interrupted by user")
        print(f"Partial results saved to: {args.output}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during translation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
