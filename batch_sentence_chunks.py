#!/usr/bin/env python3
"""
Batch Chunk Division Tool for Parallel Translation
Divides sentence chunks into evenly-sized batches for parallel translation.

Each batch gets its own manifest JSON file with chunk IDs, enabling multiple
workers to translate chunks in parallel without conflicts.

Usage:
  python batch_sentence_chunks.py --num-batches 4 --chunks-dir chunks
  python batch_sentence_chunks.py --num-batches 8  # Creates batch_0_manifest.json ... batch_7_manifest.json

Workflow for parallel translation:
  1. Run: python batch_sentence_chunks.py --num-batches 4
  2. Start workers in parallel:
     - Worker 1: uv run python prepare_translations.py translate --chunk-batch-manifest chunks/batch_0_manifest.json
     - Worker 2: uv run python prepare_translations.py translate --chunk-batch-manifest chunks/batch_1_manifest.json
     - Worker 3: uv run python prepare_translations.py translate --chunk-batch-manifest chunks/batch_2_manifest.json
     - Worker 4: uv run python prepare_translations.py translate --chunk-batch-manifest chunks/batch_3_manifest.json
  3. After all workers complete:
     - python prepare_translations.py assemble
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def get_chunk_ids(chunks_dir: Path) -> List[int]:
    """Get sorted list of all chunk IDs from manifest."""
    manifest_file = chunks_dir / "manifest.json"
    
    if not manifest_file.exists():
        print(f"✗ Manifest file not found: {manifest_file}")
        print(f"  Run 'python prepare_sentences.py chunk' first to create chunks")
        return []
    
    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    chunk_ids = sorted([int(cid) for cid in manifest["chunks"].keys()])
    return chunk_ids


def divide_into_batches(chunk_ids: List[int], num_batches: int) -> List[List[int]]:
    """Divide chunk IDs evenly into specified number of batches."""
    if num_batches <= 0:
        raise ValueError("num_batches must be >= 1")
    
    if num_batches > len(chunk_ids):
        print(f"⚠ Warning: num_batches ({num_batches}) > total chunks ({len(chunk_ids)})")
        print(f"  Some batches will be empty")
    
    batches = [[] for _ in range(num_batches)]
    
    # Distribute chunks evenly in round-robin fashion
    for idx, chunk_id in enumerate(chunk_ids):
        batch_idx = idx % num_batches
        batches[batch_idx].append(chunk_id)
    
    return batches


def save_batch_manifests(
    chunks_dir: Path,
    batches: List[List[int]],
    num_batches: int
) -> List[Path]:
    """Save batch manifests and return their paths."""
    saved_files = []
    
    for batch_idx, chunk_ids in enumerate(batches):
        batch_manifest = {
            "batch_number": batch_idx,
            "total_batches": num_batches,
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunk_ids),
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "purpose": "parallel_translation",
                "instructions": f"Translate chunks {chunk_ids[0]} to {chunk_ids[-1]}" if chunk_ids else "Empty batch"
            }
        }
        
        manifest_file = chunks_dir / f"batch_{batch_idx}_manifest.json"
        
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(batch_manifest, f, indent=2, ensure_ascii=False)
        
        saved_files.append(manifest_file)
        print(f"  ✓ Batch {batch_idx}: {len(chunk_ids):3d} chunks → {manifest_file.name}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Divide sentence chunks into batches for parallel translation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Divides all sentence chunks into N evenly-distributed batches.
Each batch gets a manifest JSON file for independent parallel processing.

Examples:
  python batch_sentence_chunks.py --num-batches 4
  python batch_sentence_chunks.py --num-batches 8 --chunks-dir chunks
  python batch_sentence_chunks.py --num-batches 2 --chunks-dir /data/chunks

After creating batches, run workers in parallel:
  Worker 0: python prepare_translations.py translate --chunk-batch-manifest chunks/batch_0_manifest.json
  Worker 1: python prepare_translations.py translate --chunk-batch-manifest chunks/batch_1_manifest.json
  ... (repeat for each batch)

Then assemble results:
  python prepare_translations.py assemble --output training_data.jsonl
        """
    )
    
    parser.add_argument(
        '--num-batches',
        type=int,
        required=True,
        help='Number of parallel batches to create'
    )
    parser.add_argument(
        '--chunks-dir',
        type=Path,
        default=Path("chunks"),
        help='Directory containing chunks and manifest (default: chunks/)'
    )
    
    args = parser.parse_args()
    
    # Show banner
    print("=" * 70)
    print("Batch Chunk Division for Parallel Translation")
    print("=" * 70)
    print(f"Chunks directory: {args.chunks_dir.absolute()}")
    print(f"Target batches: {args.num_batches}")
    print("=" * 70)
    
    try:
        # Get all chunk IDs
        print("\nLoading chunks...")
        chunk_ids = get_chunk_ids(args.chunks_dir)
        
        if not chunk_ids:
            return 1
        
        print(f"✓ Found {len(chunk_ids)} chunks\n")
        
        # Divide into batches
        print(f"Dividing into {args.num_batches} batches...\n")
        batches = divide_into_batches(chunk_ids, args.num_batches)
        
        # Save batch manifests
        print("Saving batch manifests:")
        saved_files = save_batch_manifests(args.chunks_dir, batches, args.num_batches)
        
        # Show summary
        print("\n" + "=" * 70)
        print("✓ Batch manifests created successfully!")
        print("=" * 70)
        print(f"\nBatch distribution:")
        for batch_idx, batch_chunk_ids in enumerate(batches):
            if batch_chunk_ids:
                print(f"  Batch {batch_idx}: Chunks [{batch_chunk_ids[0]} ... {batch_chunk_ids[-1]}] ({len(batch_chunk_ids)} total)")
            else:
                print(f"  Batch {batch_idx}: (empty)")
        
        print(f"\nParallel translation commands:")
        for batch_idx in range(args.num_batches):
            print(f"  Worker {batch_idx}: python prepare_translations.py translate --chunk-batch-manifest {args.chunks_dir}/batch_{batch_idx}_manifest.json --delay 1.5")
        
        print(f"\nAfter all workers complete, assemble results:")
        print(f"  python prepare_translations.py assemble --output training_data.jsonl")
        print()
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nBatch division interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error during batch division: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
