#!/usr/bin/env python3
"""
Semantic Similarity Analysis using Sentence Transformers

This script analyzes location titles from data_philly.csv to find semantically
similar locations using neural language models. It demonstrates how modern
AI can capture meaning and context that traditional n-gram approaches miss.

Author: Created for History 8510 at Clemson University
"""

import csv
import re
from typing import List, Set, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def clean_text(text: str) -> str:
    """Clean and normalize text for comparison."""
    if not text or text == 'NA':
        return ""
    
    text = str(text).lower().strip()
    
    # Remove punctuation that doesn't affect meaning
    text = re.sub(r'[.,!?;:]', '', text)
    
    # Normalize apostrophes/quotes
    text = text.replace("'", "").replace('"', '')
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text

def load_locations(filename: str) -> List[Dict]:
    """Load location data from CSV file."""
    locations = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['title'] and row['title'] != 'NA':
                locations.append({
                    'title': row['title'],
                    'address': row.get('streetaddress', ''),
                    'city': row.get('city', ''),
                    'year': row.get('Year', '')
                })
    
    return locations

def create_embeddings(locations: List[Dict], model_name: str = 'all-MiniLM-L6-v2') -> Tuple[List[str], np.ndarray]:
    """Create embeddings for all location titles using Sentence Transformers."""
    model = SentenceTransformer(model_name)
    
    # Clean titles and create embeddings
    titles = [clean_text(loc['title']) for loc in locations]
    embeddings = model.encode(titles, show_progress_bar=False)
    
    return titles, embeddings

def calculate_semantic_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between all pairs of embeddings."""
    similarity_matrix = cosine_similarity(embeddings)
    
    # Set diagonal to 0 to avoid self-similarity
    np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix

def find_semantic_similar_pairs(locations: List[Dict], titles: List[str], 
                               similarity_matrix: np.ndarray, threshold: float = 0.3) -> List[Dict]:
    """Find pairs of semantically similar locations."""
    similar_pairs = []
    
    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            similarity = similarity_matrix[i, j]
            
            if similarity >= threshold:
                # Skip if titles are exactly the same (exact duplicates)
                if titles[i].lower().strip() == titles[j].lower().strip():
                    continue
                
                similar_pairs.append({
                    'title1': locations[i]['title'],
                    'title2': locations[j]['title'],
                    'address1': locations[i]['address'],
                    'address2': locations[j]['address'],
                    'semantic_similarity': similarity
                })
    
    # Sort by semantic similarity
    similar_pairs.sort(key=lambda x: x['semantic_similarity'], reverse=True)
    
    return similar_pairs

def export_results_to_csv(similar_pairs: List[Dict], filename: str = "semantic_similarity_results.csv"):
    """Export semantic similarity results to CSV for further analysis."""
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['title1', 'title2', 'address1', 'address2', 'semantic_similarity']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for pair in similar_pairs:
            writer.writerow({
                'title1': pair['title1'],
                'title2': pair['title2'],
                'address1': pair['address1'],
                'address2': pair['address2'],
                'semantic_similarity': pair['semantic_similarity']
            })

def main():
    """Main function to run the semantic similarity analysis."""
    try:
        # Load the data
        filename = '/Users/amandaregan/Library/CloudStorage/Dropbox/*Teaching/8510-CodeForClass/Week9-TextReuse/data_philly.csv'
        locations = load_locations(filename)
        
        # Create embeddings
        titles, embeddings = create_embeddings(locations)
        
        # Calculate semantic similarities
        similarity_matrix = calculate_semantic_similarity(embeddings)
        
        # Find similar pairs
        similar_pairs = find_semantic_similar_pairs(locations, titles, similarity_matrix, threshold=0.3)
        
        # Export to CSV
        if similar_pairs:
            export_results_to_csv(similar_pairs)
            print(f"Analysis complete. Found {len(similar_pairs)} similar pairs.")
            print(f"Results saved to semantic_similarity_results.csv")
        else:
            print("No similar pairs found with the current threshold.")
    
    except FileNotFoundError:
        print("Error: Could not find the data file.")
        print("Please make sure data_philly.csv is in the correct location.")
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required packages: pip install sentence-transformers scikit-learn")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
