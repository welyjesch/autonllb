#!/bin/bash
# Installation script for NLLB Fine-tuning
# Installs all dependencies for data preparation and model training

set -e  # Exit on error

echo "================================"
echo "NLLB Fine-tuning - Setup"
echo "================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    pip install uv
else
    echo "✓ uv is already installed"
fi

# Install dependencies
echo ""
echo "Installing project dependencies..."
uv pip install --quiet transformers datasets sentencepiece accelerate googletrans==4.0.0-rc1

echo ""
echo "================================"
echo "✓ Installation complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Prepare data: python prepare_data.py"
echo "  2. Train model: python run_training.py"

