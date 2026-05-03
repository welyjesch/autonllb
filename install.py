#!/usr/bin/env python3
"""
Installation script for NLLB Fine-tuning
Installs all dependencies for data preparation and model training

Usage: python install.py
"""

import subprocess
import sys
import shutil


def run_command(cmd: list, description: str = "") -> bool:
    """Run a shell command and report status."""
    if description:
        print(f"  {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if description:
            print(f"  ✓ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e}")
        if e.stderr:
            print(f"  {e.stderr}")
        return False


def main():
    print("=" * 50)
    print("NLLB Fine-tuning - Setup")
    print("=" * 50)
    print()
    
    # Check if uv is installed
    print("Checking package manager...")
    if shutil.which("uv"):
        print("  ✓ uv is already installed")
        use_uv = True
    else:
        print("  Installing uv...")
        if run_command([sys.executable, "-m", "pip", "install", "uv"]):
            use_uv = True
        else:
            print("  Warning: Could not install uv, using pip instead")
            use_uv = False
    
    print()
    print("Installing project dependencies...")
    
    dependencies = [
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "sentencepiece>=0.1.98",
        "accelerate>=0.20.0",
        "googletrans==4.0.0-rc1",
    ]
    
    if use_uv:
        cmd = ["uv", "pip", "install", "--quiet"] + dependencies
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + dependencies
    
    if run_command(cmd, "Installing dependencies"):
        success = True
    else:
        success = False
    
    print()
    print("=" * 50)
    if success:
        print("✓ Installation complete!")
        print("=" * 50)
        print()
        print("Next steps:")
        print("  1. Prepare data: python prepare_data.py")
        print("  2. Train model: python run_training.py")
        return 0
    else:
        print("✗ Installation failed!")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
