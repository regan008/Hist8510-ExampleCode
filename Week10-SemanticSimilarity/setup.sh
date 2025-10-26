#!/bin/bash
# Setup script for Semantic Similarity Analysis

echo "Setting up virtual environment for Semantic Similarity Analysis..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages for semantic similarity analysis..."
echo "This includes Sentence Transformers and neural language models (~200MB download)"
pip install -r requirements.txt

echo ""
echo "Setup complete! Ready for semantic similarity analysis."
echo ""
echo "Usage:"
echo "  source venv/bin/activate"
echo "  python semantic_similarity_analysis.py"
echo "  deactivate"
echo ""
echo "Note: First run will download the neural language model (~90MB)"
echo "Output: semantic_similarity_results.csv"



