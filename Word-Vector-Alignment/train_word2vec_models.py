#!/usr/bin/env python3
"""
Word2Vec Model Training Script for Historical Text Analysis

This script trains word2vec models on text documents from specified directories.
Can train models for multiple time periods or datasets.

Usage examples:
    # Train on default directories (1900-20 and 1940-60)
    python train_word2vec_models.py
    
    # Train on a single custom directory
    python train_word2vec_models.py --paths /path/to/documents --names my_model
    
    # Train on multiple custom directories
    python train_word2vec_models.py --paths /path/to/period1 /path/to/period2 --names model1 model2
    
    # Custom output directory and parameters
    python train_word2vec_models.py --paths /path/to/docs --names my_model --output-dir ./models --vector-size 200

Author: Lesson Plan for Digital Humanities Course
"""

import argparse
import os
import re
import pickle
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.utils import simple_preprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_text(text):
    """
    Preprocess text by cleaning and tokenizing using gensim's simple_preprocess.
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        list: List of tokens
    """
    # Use gensim's simple_preprocess for consistent tokenization
    tokens = simple_preprocess(text, min_len=2, max_len=50)
    
    return tokens

def load_text_files(directory):
    """
    Load and preprocess all text files from a directory.
    
    Args:
        directory (str): Path to directory containing text files
        
    Returns:
        list: List of tokenized sentences
    """
    sentences = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.error(f"Directory {directory} does not exist")
        return sentences
    
    # Get all .txt files in the directory
    txt_files = list(directory_path.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files in {directory}")
    
    for file_path in txt_files:
        logger.info(f"Processing {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Split content into sentences (simple approach)
            # Split on periods, exclamation marks, and question marks
            raw_sentences = re.split(r'[.!?]+', content)
            
            for sentence in raw_sentences:
                if sentence.strip():  # Skip empty sentences
                    tokens = preprocess_text(sentence)
                    if len(tokens) > 2:  # Only keep sentences with more than 2 words
                        sentences.append(tokens)
                        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Total sentences processed: {len(sentences)}")
    return sentences

def train_word2vec_model(sentences, model_name, output_dir=None, vector_size=100, window=5, min_count=5, workers=4, epochs=10):
    """
    Train a Word2Vec model on the provided sentences using gensim.
    
    Args:
        sentences (list): List of tokenized sentences
        model_name (str): Name for the model file
        output_dir (Path, optional): Directory to save models. If None, saves to current directory.
        vector_size (int): Size of word vectors
        window (int): Maximum distance between current and predicted word
        min_count (int): Minimum count of words to be included in vocabulary
        workers (int): Number of worker threads
        epochs (int): Number of training epochs
        
    Returns:
        tuple: (Word2Vec model, model_path, pickle_path)
    """
    logger.info(f"Training Word2Vec model: {model_name}")
    logger.info(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}, epochs={epochs}")
    
    # Train the model
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,  # Skip-gram model (1) vs CBOW (0)
        negative=5,  # Number of negative samples
        hs=0,  # Hierarchical softmax (0 = disabled)
        sample=1e-3,  # Downsampling threshold
        alpha=0.025,  # Initial learning rate
        min_alpha=0.0001,  # Final learning rate
        seed=42  # Random seed for reproducibility
    )
    
    # Determine output paths
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_name}.model"
        pickle_path = output_dir / f"{model_name}.pkl"
    else:
        model_path = Path(f"{model_name}.model")
        pickle_path = Path(f"{model_name}.pkl")
    
    # Save the model
    model.save(str(model_path))
    logger.info(f"Model saved to: {model_path}")
    
    # Also save as pickle for easier loading
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model also saved as pickle: {pickle_path}")
    
    # Print some statistics
    logger.info(f"Vocabulary size: {len(model.wv)}")
    logger.info(f"Total words in corpus: {model.corpus_total_words}")
    
    return model, model_path, pickle_path

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Word2Vec models on text documents from specified directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directories (txt/1900-20 and txt/1940-60)
  python train_word2vec_models.py
  
  # Train on a single custom directory
  python train_word2vec_models.py --paths /path/to/documents --names my_model
  
  # Train on multiple directories
  python train_word2vec_models.py --paths /path/to/period1 /path/to/period2 --names model1 model2
  
  # Custom output and parameters
  python train_word2vec_models.py --paths ./my_docs --names my_model --output-dir ./models --vector-size 200
        """
    )
    
    parser.add_argument(
        "--paths",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to directory(ies) containing text files. If not provided, uses default txt/1900-20 and txt/1940-60"
    )
    
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Name(s) for the model file(s). Must match number of paths. If not provided, auto-generated from directory names"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save models. If not provided, saves to current directory"
    )
    
    parser.add_argument(
        "--vector-size",
        type=int,
        default=100,
        help="Size of word vectors (default: 100)"
    )
    
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Maximum distance between current and predicted word (default: 5)"
    )
    
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum count of words to be included in vocabulary (default: 5)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    
    parser.add_argument(
        "--test-word",
        type=str,
        default="education",
        help="Word to test for similar words after training (default: 'education')"
    )
    
    return parser.parse_args()


def get_default_paths():
    """Get default paths for backward compatibility."""
    base_dir = Path(__file__).parent
    txt_dir = base_dir / "txt"
    period_1900_20 = txt_dir / "1900-20"
    period_1940_60 = txt_dir / "1940-60"
    return [period_1900_20, period_1940_60]


def generate_model_name(path):
    """Generate a model name from a directory path."""
    path = Path(path)
    # Use directory name, or parent if it's a file
    if path.is_file():
        name = path.stem
    else:
        name = path.name
    
    # Clean up the name for use as filename
    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"word2vec_{name}"


def main():
    """Main function to train word2vec models."""
    args = parse_arguments()
    
    # Determine paths and names
    if args.paths is None:
        # Use default paths for backward compatibility
        paths = get_default_paths()
        names = ["word2vec_1900_20", "word2vec_1940_60"]
        logger.info("Using default paths: txt/1900-20 and txt/1940-60")
    else:
        paths = [Path(p) for p in args.paths]
        if args.names is None:
            # Generate names from paths
            names = [generate_model_name(p) for p in paths]
            logger.info(f"Auto-generated model names: {names}")
        else:
            names = args.names
            if len(names) != len(paths):
                logger.error(f"Number of names ({len(names)}) must match number of paths ({len(paths)})")
                return
    
    # Validate paths
    for path in paths:
        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            return
        if not path.is_dir():
            logger.error(f"Path is not a directory: {path}")
            return
    
    # Model parameters
    model_params = {
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "workers": args.workers,
        "epochs": args.epochs,
        "output_dir": Path(args.output_dir) if args.output_dir else None
    }
    
    logger.info("=" * 70)
    logger.info("Starting Word2Vec model training")
    logger.info("=" * 70)
    logger.info(f"Training {len(paths)} model(s)")
    logger.info(f"Parameters: vector_size={args.vector_size}, window={args.window}, "
                f"min_count={args.min_count}, epochs={args.epochs}, workers={args.workers}")
    if args.output_dir:
        logger.info(f"Output directory: {args.output_dir}")
    
    # Train models
    trained_models = []
    saved_files = []
    
    for i, (path, name) in enumerate(zip(paths, names), 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"TRAINING MODEL {i}/{len(paths)}: {name}")
        logger.info(f"Source directory: {path}")
        logger.info("=" * 70)
        
        sentences = load_text_files(path)
        if sentences:
            model, model_path, pickle_path = train_word2vec_model(
                sentences,
                name,
                **model_params
            )
            trained_models.append((model, name))
            saved_files.extend([model_path, pickle_path])
            
            # Show example similar words
            logger.info(f"\nExample similar words for '{args.test_word}':")
            try:
                similar_words = model.wv.most_similar(args.test_word, topn=5)
                for word, score in similar_words:
                    logger.info(f"  {word}: {score:.3f}")
            except KeyError:
                logger.info(f"  '{args.test_word}' not found in vocabulary")
        else:
            logger.error(f"No sentences found for {path}")
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info("Models saved:")
    for file_path in saved_files:
        logger.info(f"  - {file_path}")
    
    if len(trained_models) == 2:
        logger.info("\nNext step: Run alignment script to align the models")
        logger.info("  python align_word_vectors.py")


if __name__ == "__main__":
    main()
