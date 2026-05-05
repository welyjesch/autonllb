#!/usr/bin/env python3
"""
Translation Module for NLLB Training
Translates Hiligaynon sentence chunks to English and assembles into JSONL.

This module depends on prepare_sentences.py output and can be run in parallel
via batch_sentence_chunks.py for multiple workers.

Usage:
  python prepare_translations.py translate --chunks-dir chunks --delay 2.5
  python prepare_translations.py assemble --chunks-dir chunks --output training_data.jsonl
  python prepare_translations.py translate --chunk-batch-manifest chunks/batch_0_manifest.json
"""

import argparse
import json
import sys
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import googletrans
from googletrans import Translator


# ==============================================================================
# SECTION 0: Shared ChunkManager (Import from prepare_sentences)
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
    
    def mark_translated(self, chunk_number: int, output_file: Path = None):
        """Mark a chunk as translated."""
        if str(chunk_number) in self.manifest["chunks"]:
            self.manifest["chunks"][str(chunk_number)]["is_translated"] = True
            self.manifest["chunks"][str(chunk_number)]["translated_at"] = datetime.now().isoformat()
            if output_file:
                self.manifest["chunks"][str(chunk_number)]["translated_file"] = str(output_file)
            self._save_manifest()
    
    def get_untranslated_chunks(self) -> List[int]:
        """Get list of chunk numbers that haven't been translated yet."""
        untranslated = []
        for chunk_id, chunk_info in self.manifest["chunks"].items():
            if not chunk_info.get("is_translated", False):
                untranslated.append(int(chunk_id))
        return sorted(untranslated)
    
    def get_chunk_file(self, chunk_number: int) -> Path:
        """Get the file path for a chunk (contains ORIGINAL Hiligaynon sentences)."""
        return self.chunks_dir / f"hiligaynon_chunk_{chunk_number}.txt"
    
    def get_status(self) -> str:
        """Get formatted status of all chunks."""
        total = len(self.manifest["chunks"])
        translated = sum(1 for c in self.manifest["chunks"].values() if c.get("is_translated", False))
        return f"Chunks: {translated}/{total} translated"


# ==============================================================================
# SECTION 1: Translation Logic
# ==============================================================================

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
            # googletrans.Translator is synchronous, so wrap in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: translator.translate(sentence, src='hil', dest='en')
            )
            
            if hasattr(result, 'text') and result.text:
                translated = result.text
                # Verify translation is different from source (contamination check)
                if translated.strip() == sentence.strip():
                    raise ValueError("Translation returned unchanged sentence (possible fallback)")
                return translated
            elif isinstance(result, str):
                if result.strip() == sentence.strip():
                    raise ValueError("Translation returned unchanged sentence (possible fallback)")
                return result
            
            # If result is invalid, raise exception to trigger retry logic
            raise ValueError("Translation returned an invalid or empty result")
        except Exception as e:
            print(f"  Translation error (attempt {attempt + 1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
    
    # CRITICAL: NEVER return the original sentence as a fallback. 
    # Returning the source text when translation fails sabotages the entire dataset.
    # If all retries fail, the function must raise an exception to stop the pipeline.
    raise RuntimeError(f"Permanent translation failure for sentence: {sentence[:50]}...")


# ==============================================================================
# SECTION 2: Phase 2 - Translation
# ==============================================================================

async def phase_translate(
    chunks_dir: Path = Path("chunks"),
    translation_delay: float = 2.5,
    chunk_batch_manifest: Path = None,
) -> int:
    """
    Phase 2: Translate all untranslated chunks.
    
    Can operate on:
    - All chunks in chunks_dir (default)
    - Specific chunk IDs via chunk_batch_manifest JSON file
    
    Args:
        chunks_dir: Directory containing chunk files
        translation_delay: Delay between translation requests
        chunk_batch_manifest: Optional path to batch manifest with specific chunk IDs
    
    Returns:
        Number of chunks translated
    """
    print("\n" + "=" * 70)
    print("PHASE 2: TRANSLATION - Translate sentence chunks")
    print("=" * 70)
    
    chunk_manager = ChunkManager(chunks_dir=chunks_dir)
    
    # Load specific chunk IDs if batch manifest provided
    if chunk_batch_manifest:
        print(f"Loading batch manifest: {chunk_batch_manifest}")
        with open(chunk_batch_manifest, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        untranslated = batch_data.get("chunk_ids", [])
        print(f"  Batch contains {len(untranslated)} chunks\n")
    else:
        untranslated = chunk_manager.get_untranslated_chunks()
    
    if not untranslated:
        print("✓ All chunks already translated!")
        return 0
    
    print(f"Found {len(untranslated)} untranslated chunks\n")
    
    translator = Translator()
    chunks_translated = 0
    
    for chunk_id in untranslated:
        chunk_file = chunk_manager.get_chunk_file(chunk_id)
        output_file = chunk_file.parent / f"translated_chunk_{chunk_id}.txt"
        
        print(f"[Chunk {chunk_id}] Translating {chunk_file.name}...")
        
        # Read sentences from chunk
        with open(chunk_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        print(f"  Found {len(sentences)} sentences")
        
        # Translate sentences
        translated_sentences = []
        failed_sentences = 0
        for idx, sentence in enumerate(sentences, 1):
            try:
                english = await translate_sentence(translator, sentence, delay=translation_delay)
                translated_sentences.append(english)
            except RuntimeError as e:
                print(f"    CRITICAL: Translation failed for sentence {idx}: {str(e)}")
                failed_sentences += 1
                # Do NOT write this chunk - contamination prevention
                break
            
            if idx % 20 == 0:
                print(f"    Progress: {idx}/{len(sentences)} translated")
        
        # Only save if all translations succeeded
        if failed_sentences > 0:
            print(f"  ✗ Chunk {chunk_id} translation aborted ({failed_sentences} failures) - file NOT created to prevent contamination")
            continue
        
        # Save translated chunk
        with open(output_file, 'w', encoding='utf-8') as f:
            for translated in translated_sentences:
                f.write(translated + '\n')
        
        # Mark as translated in manifest
        chunk_manager.mark_translated(chunk_id, output_file=output_file)
        chunks_translated += 1
        
        print(f"  ✓ Translated chunk saved → {output_file.name}")
    
    print(f"\n✓ Translation complete!")
    print(f"  Chunks translated: {chunks_translated}")
    print(f"  {chunk_manager.get_status()}")
    
    return chunks_translated


# ==============================================================================
# SECTION 3: Phase 3 - Assembly
# ==============================================================================

def phase_assemble(
    chunks_dir: Path = Path("chunks"),
    output_file: Path = Path("training_data.jsonl"),
) -> int:
    """
    Phase 3: Assemble translated chunks into final JSONL file.
    
    Args:
        chunks_dir: Directory containing chunk files
        output_file: Output JSONL file path
    
    Returns:
        Total sentence pairs written
    """
    print("\n" + "=" * 70)
    print("PHASE 3: ASSEMBLY - Assemble chunks into JSONL")
    print("=" * 70)
    
    chunk_manager = ChunkManager(chunks_dir=chunks_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_pairs = 0
    
    # Sort chunks by ID
    chunk_ids = sorted([int(cid) for cid in chunk_manager.manifest["chunks"].keys()])
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for chunk_id in chunk_ids:
            chunk_info = chunk_manager.manifest["chunks"][str(chunk_id)]
            
            if not chunk_info.get("is_translated", False):
                print(f"⚠ Chunk {chunk_id} not translated, skipping...")
                continue
            
            translated_file = Path(chunk_info.get("translated_file", ""))
            
            if not translated_file.exists():
                print(f"⚠ Translated file not found: {translated_file}")
                continue
            
            print(f"[Chunk {chunk_id}] Processing {translated_file.name}...")
            
            # Read translated and original sentences
            original_file = chunk_manager.get_chunk_file(chunk_id)
            
            if not original_file.exists():
                print(f"  ✗ Original file not found: {original_file}")
                continue
            
            with open(original_file, 'r', encoding='utf-8') as f:
                original_sentences = [line.strip() for line in f if line.strip()]
            
            with open(translated_file, 'r', encoding='utf-8') as f:
                translated_sentences = [line.strip() for line in f if line.strip()]
            
            # Contamination check: verify translations are actually different from originals
            identical_count = 0
            for orig, trans in zip(original_sentences, translated_sentences):
                if orig.strip() == trans.strip():
                    identical_count += 1
            
            if identical_count > 0:
                print(f"  ✗ CONTAMINATION DETECTED: {identical_count}/{len(original_sentences)} sentences identical (not translated)")
                print(f"    This chunk is skipped to prevent poisoning the dataset")
                continue
            
            # Create pairs
            for idx, (hiligaynon, english) in enumerate(zip(original_sentences, translated_sentences), 1):
                seq_id = int(time.time() * 1000) + idx
                
                pair = {
                    "seq_id": seq_id,
                    "english": english,
                    "hiligaynon": hiligaynon,
                    "chunk_id": chunk_id,
                }
                
                json.dump(pair, outfile, ensure_ascii=False)
                outfile.write('\n')
                total_pairs += 1
            
            print(f"  ✓ Written {len(original_sentences)} pairs")
    
    print(f"\n✓ Assembly complete!")
    print(f"  Total pairs written: {total_pairs}")
    print(f"  Output file: {output_file.absolute()}")
    
    return total_pairs


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Translate chunks and assemble JSONL for NLLB training (Phases 2-3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Translate sentence chunks and assemble into JSONL format.
Can translate specific batches for parallel processing.

Examples:
  python prepare_translations.py translate --delay 2.5
  python prepare_translations.py translate --chunk-batch-manifest chunks/batch_0_manifest.json
  python prepare_translations.py assemble --output training_data.jsonl
        """
    )
    
    subparsers = parser.add_subparsers(dest='phase', help='Preparation phase')
    
    # Phase 2: Translate
    translate_parser = subparsers.add_parser('translate', help='Phase 2: Translate chunks')
    translate_parser.add_argument(
        '--chunks-dir',
        type=Path,
        default=Path("chunks"),
        help='Directory containing chunks (default: chunks/)'
    )
    translate_parser.add_argument(
        '--delay',
        type=float,
        default=2.5,
        help='Delay between translation requests in seconds (default: 2.5)'
    )
    translate_parser.add_argument(
        '--chunk-batch-manifest',
        type=Path,
        default=None,
        help='Optional: path to batch manifest JSON for parallel processing'
    )
    
    # Phase 3: Assemble
    assemble_parser = subparsers.add_parser('assemble', help='Phase 3: Assemble into JSONL')
    assemble_parser.add_argument(
        '--chunks-dir',
        type=Path,
        default=Path("chunks"),
        help='Directory containing chunks (default: chunks/)'
    )
    assemble_parser.add_argument(
        '--output',
        type=Path,
        default=Path("training_data.jsonl"),
        help='Output JSONL file (default: training_data.jsonl)'
    )
    
    args = parser.parse_args()
    
    # Show phase banner
    print("=" * 70)
    print("NLLB Translation & Assembly (Phases 2-3)")
    print("=" * 70)
    
    try:
        if args.phase == 'translate':
            print(f"Phase: TRANSLATION")
            print(f"Chunks directory: {args.chunks_dir}")
            print(f"Translation delay: {args.delay}s")
            if args.chunk_batch_manifest:
                print(f"Batch manifest: {args.chunk_batch_manifest}")
            print("=" * 70)
            asyncio.run(phase_translate(
                chunks_dir=args.chunks_dir,
                translation_delay=args.delay,
                chunk_batch_manifest=args.chunk_batch_manifest,
            ))
            return 0
        
        elif args.phase == 'assemble':
            print(f"Phase: ASSEMBLY")
            print(f"Chunks directory: {args.chunks_dir}")
            print(f"Output file: {args.output}")
            print("=" * 70)
            phase_assemble(
                chunks_dir=args.chunks_dir,
                output_file=args.output,
            )
            return 0
        
        else:
            print("No phase specified. Use: translate | assemble")
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\n\nTranslation interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during translation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
