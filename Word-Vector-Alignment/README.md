# Word Vector Alignment Project

## Project Overview

### Data and Output Locations

- **Text Files**: Historical documents are stored in `txt/1900-20/` and `txt/1940-60/` directories
- **Trained Models**: Word2Vec models are saved as `.model` and `.pkl` files in the root directory
- **Visualizations**: Generated plots are saved to `visualizations/` directory
- **Results**: Analysis results are stored in `alignment_results/` directory

### Procrustes Alignment Algorithm

When you train word2vec models on different time periods, the word vectors end up in different coordinate systems - like having two maps of the same city but rotated differently. The Procrustes alignment algorithm finds the best way to rotate one map to match the other.

Think of it like this: if you have two word clouds where "king" and "queen" are close together in both, but the clouds are rotated differently, Procrustes finds the rotation that makes the clouds overlap as much as possible. This allows us to meaningfully compare how word meanings changed between time periods by ensuring we're looking at them from the same perspective.

## Setup Instructions

### 1. Check Python Version

```bash
# Check current Python version
python3 --version

# If you don't have Python 3.11, install it via Homebrew
brew install python@3.11

# Verify installation
python3.11 --version
```

### 2. Create Virtual Environment

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv word_vector_env

# Activate virtual environment
source word_vector_env/bin/activate  # On macOS/Linux
# OR
word_vector_env\Scripts\activate     # On Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Train Word2Vec Models

**Default usage** (uses `txt/1900-20` and `txt/1940-60`):
```bash
python train_word2vec_models.py
```

**Custom document paths:**
```bash
# Train on a single directory
python train_word2vec_models.py --paths /path/to/your/documents --names my_model

# Train on multiple directories
python train_word2vec_models.py --paths /path/to/period1 /path/to/period2 --names model1 model2

# Auto-generate model names from directory names
python train_word2vec_models.py --paths /path/to/period1 /path/to/period2

# Custom output directory and parameters
python train_word2vec_models.py --paths ./my_docs --names my_model --output-dir ./models --vector-size 200 --epochs 20
```

**Available options:**
- `--paths`: One or more paths to directories containing `.txt` files
- `--names`: Model name(s) (optional, auto-generated from directory names if not provided)
- `--output-dir`: Directory to save models (default: current directory)
- `--vector-size`: Size of word vectors (default: 100)
- `--window`: Context window size (default: 5)
- `--min-count`: Minimum word frequency (default: 5)
- `--workers`: Number of worker threads (default: 4)
- `--epochs`: Number of training epochs (default: 10)
- `--test-word`: Word to test after training (default: "education")

**Example with new text corpus:**
```bash
python train_word2vec_models.py --paths ./my_historical_texts --names 19th_century --output-dir ./trained_models
```

### 2. Align Vector Spaces

**Default usage** (uses `word2vec_1900_20.model` and `word2vec_1940_60.model`):
```bash
python align_word_vectors.py
```

**Custom model names/paths:**
```bash
# Use custom model names
python align_word_vectors.py --model1 my_model1.model --model2 my_model2.model

# Use models from different directories
python align_word_vectors.py --model1 ./models/period1.model --model2 ./models/period2.model

# Custom output directory
python align_word_vectors.py --output-dir ./my_alignment_results
```

**Available options:**
- `--model1`: Path to first Word2Vec model (default: `word2vec_1900_20.model`)
- `--model2`: Path to second Word2Vec model (default: `word2vec_1940_60.model`)
- `--output-dir`: Directory to save alignment results (default: `alignment_results`)
- `--min-freq`: Minimum word frequency for common vocabulary (default: 5)

**Complete workflow example with custom models:**
```bash
# Train custom models
python train_word2vec_models.py --paths ./period1 ./period2 --names model1 model2 --output-dir ./models

# Align the custom models
python align_word_vectors.py --model1 ./models/model1.model --model2 ./models/model2.model --output-dir ./my_results
```

### 3. Analyze Results
```bash
python demo_results.py
```

### 4. Create Visualizations

**Basic semantic shifts** (uses default models):
```bash
python visualize_semantic_shifts.py academy absolute abnormal
```

**With custom models:**
```bash
python visualize_semantic_shifts.py academy absolute abnormal --model-1900 ./models/model1.model --model-1940 ./models/model2.model
```

**Aligned semantic shifts** (uses Procrustes-aligned models):
```bash
python visualize_aligned_shifts.py academy absolute abnormal
```

**With custom models:**
```bash
python visualize_aligned_shifts.py academy absolute abnormal --model-1900 ./models/model1.model --model-1940 ./models/model2.model
```

**Available options for visualization scripts:**
- `--model-1900`: Path to first period model (default: `word2vec_1900_20.model`)
- `--model-1940`: Path to second period model (default: `word2vec_1940_60.model`)
- `--output-dir`: Output directory for plots (defaults vary by script)
- `--alignment-results`: Path to alignment results CSV (for aligned visualizations)
- `--context-words`: Number of context words to show (default: 15)

### 5. Compare Aligned vs Unaligned Results

**Default models:**
```bash
python compare_aligned_unaligned.py academy absolute abnormal
```

**Custom models:**
```bash
python compare_aligned_unaligned.py academy absolute abnormal --model-1900 ./models/model1.model --model-1940 ./models/model2.model
```

### 6. Reset Project (if needed)
```bash
python reset_project.py
```