#!/usr/bin/env python3
"""
Semantic Similarity Analysis of Presidential Inaugural Addresses

This script analyzes presidential inaugural addresses using sentence transformers
to find semantic similarities, cluster them by theme, and visualize the results.

Author: Created for History 8510 at Clemson University
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_inaugural_addresses(address_dir: str = "txt") -> List[Dict]:
    """Load all inaugural addresses from text files."""
    addresses = []
    
    # Get all .txt files from the txt subdirectory
    txt_files = glob.glob(os.path.join(address_dir, "*.txt"))
    
    print(f"Found {len(txt_files)} inaugural addresses")
    
    for filepath in sorted(txt_files):
        # Extract president and year from filename
        filename = os.path.basename(filepath)
        match = re.match(r'(\d{4})_(.+)\.txt', filename)
        
        if match:
            year = match.group(1)
            president = match.group(2).replace('_', ' ')
        
        # Read the text
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter very short sentences
        
        addresses.append({
            'filename': filename,
            'year': year,
            'president': president,
            'text': text,
            'sentences': sentences,
            'word_count': len(text.split())
        })
    
    return addresses

def create_embeddings(addresses: List[Dict], 
                     model_name: str = 'all-MiniLM-L6-v2') -> Tuple[List[np.ndarray], np.ndarray]:
    """Create embeddings for each inaugural address.
    
    Returns:
        - List of address-level embeddings (average of sentence embeddings)
        - Full embedding matrix (address, embedding_dim)
    """
    print(f"\nLoading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Creating embeddings for {len(addresses)} inaugural addresses...")
    
    all_embeddings = []
    
    for i, address in enumerate(addresses, 1):
        print(f"  [{i}/{len(addresses)}] {address['year']} - {address['president']}", end="")
        
        # Create embeddings for each sentence
        sentence_embeddings = model.encode(address['sentences'], show_progress_bar=False)
        
        # Average sentence embeddings to get document-level embedding
        doc_embedding = np.mean(sentence_embeddings, axis=0)
        all_embeddings.append(doc_embedding)
        
        print(f" ({address['word_count']} words)")
    
    # Convert to numpy array
    embedding_matrix = np.array(all_embeddings)
    
    return all_embeddings, embedding_matrix

def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity matrix between all addresses."""
    print("\nCalculating similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    return similarity_matrix

def find_most_similar_pairs(addresses: List[Dict], similarity_matrix: np.ndarray, 
                           top_n: int = 10) -> pd.DataFrame:
    """Find the most semantically similar pairs of inaugural addresses."""
    print(f"\nFinding top {top_n} most similar address pairs...")
    
    pairs = []
    
    for i in range(len(addresses)):
        for j in range(i + 1, len(addresses)):
            similarity = similarity_matrix[i, j]
            pairs.append({
                'president1': addresses[i]['president'],
                'year1': addresses[i]['year'],
                'president2': addresses[j]['president'],
                'year2': addresses[j]['year'],
                'similarity': similarity,
                'combined': f"{addresses[i]['year']} vs {addresses[j]['year']}"
            })
    
    # Sort by similarity
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('similarity', ascending=False).head(top_n)
    
    return pairs_df

def cluster_addresses(embeddings: np.ndarray, n_clusters: int = 5, 
                     method: str = 'kmeans') -> np.ndarray:
    """Cluster inaugural addresses by semantic similarity."""
    print(f"\nClustering addresses into {n_clusters} groups using {method}...")
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:  # agglomerative
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    
    labels = clusterer.fit_predict(embeddings)
    
    print(f"Clusters created: {len(np.unique(labels))} unique clusters")
    
    return labels

def create_similarity_heatmap(addresses: List[Dict], similarity_matrix: np.ndarray, 
                            output_file: str = "inaugural_similarity_heatmap.png"):
    """Create a heatmap showing similarity between all inaugural addresses."""
    print(f"\nCreating similarity heatmap: {output_file}")
    
    # Prepare labels
    labels = [f"{addr['year']} {addr['president'].split()[-1]}" 
              for addr in addresses]
    
    # Calculate actual range of values (excluding diagonal)
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    vmin = similarity_matrix[mask].min()
    vmax = similarity_matrix[mask].max()
    
    print(f"  Similarity range: {vmin:.3f} to {vmax:.3f}")
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Create heatmap with better colormap and centered on actual data range
    sns.heatmap(similarity_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                annot=False,
                cmap='viridis',  # Better color graduation
                vmin=vmin, vmax=vmax,  # Use actual data range
                center=None,  # No centering, use full range
                square=True,
                cbar_kws={"shrink": 0.8, "label": "Similarity"})
    
    plt.title('Semantic Similarity Between Inaugural Addresses', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Inaugural Address', fontsize=12)
    plt.ylabel('Inaugural Address', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

def create_tsne_visualization(addresses: List[Dict], embeddings: np.ndarray,
                            labels: np.ndarray,
                            output_file: str = "inaugural_tsne_clusters.png"):
    """Create a t-SNE visualization of addresses in 2D space."""
    print(f"\nCreating t-SNE visualization: {output_file}")
    
    print("  Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot clusters with different colors
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for label in unique_labels:
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[label]], s=100, alpha=0.6, 
                   label=f'Cluster {label+1}')
    
    # Add labels for each address
    for i, address in enumerate(addresses):
        plt.annotate(f"{address['year']}\n{address['president'].split()[-1]}",
                    xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=8,
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('Semantic Clustering of Inaugural Addresses (t-SNE)', 
              fontsize=16, fontweight='bold')
    plt.legend(title='Cluster', loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

def analyze_clusters(addresses: List[Dict], labels: np.ndarray) -> pd.DataFrame:
    """Analyze which addresses are in each cluster."""
    cluster_info = []
    
    for i, address in enumerate(addresses):
        cluster_info.append({
            'year': address['year'],
            'president': address['president'],
            'cluster': labels[i] + 1,  # Make 1-indexed
            'word_count': address['word_count']
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS")
    print("="*70)
    
    for cluster_id in sorted(cluster_df['cluster'].unique()):
        cluster_addresses = cluster_df[cluster_df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_addresses)} addresses):")
        for _, addr in cluster_addresses.iterrows():
            print(f"  • {addr['year']} - {addr['president']}")
    
    return cluster_df

def export_results(addresses: List[Dict], similarity_matrix: np.ndarray,
                  pairs_df: pd.DataFrame, cluster_df: pd.DataFrame,
                  output_file: str = "inaugural_analysis_results.csv"):
    """Export detailed results to CSV."""
    print(f"\nExporting results to: {output_file}")
    
    # Create comprehensive results
    results = []
    for i, address in enumerate(addresses):
        # Find most similar address (not itself)
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Ignore self
        most_similar_idx = np.argmax(similarities)
        
        results.append({
            'year': address['year'],
            'president': address['president'],
            'cluster': cluster_df.iloc[i]['cluster'],
            'word_count': address['word_count'],
            'most_similar_to': addresses[most_similar_idx]['president'],
            'most_similar_year': addresses[most_similar_idx]['year'],
            'most_similar_score': similarities[most_similar_idx],
            'mean_similarity': np.mean(similarities[similarities >= 0])
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"Saved: {output_file}")
    
    # Also save the top pairs
    pairs_file = "inaugural_top_similar_pairs.csv"
    pairs_df.to_csv(pairs_file, index=False)
    print(f"Saved: {pairs_file}")

def print_summary_statistics(addresses: List[Dict], similarity_matrix: np.ndarray,
                            labels: np.ndarray):
    """Print summary statistics about the analysis."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Mean similarity
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    mean_sim = similarity_matrix[mask].mean()
    print(f"\nMean pairwise similarity: {mean_sim:.3f}")
    print(f"Max pairwise similarity: {similarity_matrix[mask].max():.3f}")
    print(f"Min pairwise similarity: {similarity_matrix[mask].min():.3f}")
    
    # Address statistics
    years = [int(addr['year']) for addr in addresses]
    word_counts = [addr['word_count'] for addr in addresses]
    
    print(f"\nTime span: {min(years)} - {max(years)} ({max(years) - min(years)} years)")
    print(f"Average speech length: {np.mean(word_counts):.0f} words")
    print(f"Shortest speech: {min(word_counts)} words ({addresses[np.argmin(word_counts)]['president']} {min(years)})")
    print(f"Longest speech: {max(word_counts)} words ({addresses[np.argmax(word_counts)]['president']} {max(years)})")
    
    print(f"\nNumber of clusters: {len(np.unique(labels))}")
    for cluster_id in sorted(np.unique(labels)):
        count = np.sum(labels == cluster_id)
        print(f"  Cluster {cluster_id + 1}: {count} addresses")

def main():
    """Main analysis function."""
    print("="*70)
    print("SEMANTIC ANALYSIS OF PRESIDENTIAL INAUGURAL ADDRESSES")
    print("Using Sentence Transformers for Thematic Similarity")
    print("="*70)
    
    # Load addresses
    addresses = load_inaugural_addresses()
    
    if not addresses:
        print("Error: No inaugural addresses found!")
        print("Make sure you've run the scraper first:")
        print("  python scrape_inaugural_addresses.py")
        return
    
    # Create embeddings
    embeddings_list, embedding_matrix = create_embeddings(addresses)
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(embedding_matrix)
    
    # Find most similar pairs
    similar_pairs = find_most_similar_pairs(addresses, similarity_matrix, top_n=15)
    
    # Cluster addresses
    cluster_labels = cluster_addresses(embedding_matrix, n_clusters=5, method='kmeans')
    
    # Print summary statistics
    print_summary_statistics(addresses, similarity_matrix, cluster_labels)
    
    # Analyze clusters
    cluster_df = analyze_clusters(addresses, cluster_labels)
    
    # Create visualizations
    create_similarity_heatmap(addresses, similarity_matrix)
    create_tsne_visualization(addresses, embedding_matrix, cluster_labels)
    
    # Export results
    export_results(addresses, similarity_matrix, similar_pairs, cluster_df)
    
    # Print top similar pairs
    print("\n" + "="*70)
    print("TOP 15 MOST SIMILAR INAUGURAL ADDRESSES")
    print("="*70)
    for idx, row in enumerate(similar_pairs.iterrows(), 1):
        print(f"\n{idx}. Similarity: {row[1]['similarity']:.3f}")
        print(f"   {row[1]['year1']} - {row[1]['president1']}")
        print(f"   {row[1]['year2']} - {row[1]['president2']}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  • inaugural_similarity_heatmap.png - Similarity matrix heatmap")
    print("  • inaugural_tsne_clusters.png - 2D clustering visualization")
    print("  • inaugural_analysis_results.csv - Detailed results")
    print("  • inaugural_top_similar_pairs.csv - Top similar pairs")
    print("\nThese visualizations show:")
    print("  - Which speeches are thematically similar")
    print("  - How addresses cluster by topic/theme")
    print("  - Historical evolution of inaugural rhetoric")

if __name__ == "__main__":
    main()

