#!/usr/bin/env python3
"""
Embedding-based entity matching using sentence-transformers.
For each messy record, finds the best matching clean record based on cosine similarity of embeddings.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Configurable threshold
COSINE_THRESHOLD = 0.75

def create_match_string(record):
    """Create a concatenated string for embedding."""
    return f"{record['full_name']} {record['address']}"

def find_best_matches(clean_df, messy_df, threshold=COSINE_THRESHOLD):
    """Find best embedding similarity matches for each messy record."""
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create strings for embedding
    clean_strings = [create_match_string(record) for _, record in clean_df.iterrows()]
    messy_strings = [create_match_string(record) for _, record in messy_df.iterrows()]
    
    print("Computing embeddings for clean records...")
    clean_embeddings = model.encode(clean_strings)
    
    print("Computing embeddings for messy records...")
    messy_embeddings = model.encode(messy_strings)
    
    print(f"Finding matches using cosine similarity (threshold: {threshold})...")
    
    results = []
    
    for idx, messy_embedding in enumerate(messy_embeddings):
        # Compute cosine similarity with all clean records
        similarities = cosine_similarity([messy_embedding], clean_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Only predict match if above threshold
        predicted_id = clean_df.iloc[best_idx]['id'] if best_score >= threshold else None
        
        results.append({
            'messy_id': messy_df.iloc[idx]['messy_id'],
            'predicted_clean_id': predicted_id,
            'similarity_score': best_score
        })
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(messy_df)} records...")
    
    return pd.DataFrame(results)

def main():
    print("Loading data...")
    try:
        clean_df = pd.read_csv('clean_records.csv')
        messy_df = pd.read_csv('messy_records.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run generate_data.py first to create the input files.")
        sys.exit(1)
    
    print(f"Loaded {len(clean_df)} clean records and {len(messy_df)} messy records")
    
    # Run embedding-based matching
    predictions_df = find_best_matches(clean_df, messy_df)
    
    # Save results
    predictions_df.to_csv('predictions_embeddings.csv', index=False)
    print(f"Embedding similarity predictions saved to predictions_embeddings.csv")
    
    # Print summary statistics
    total_predictions = len(predictions_df)
    positive_predictions = predictions_df['predicted_clean_id'].notna().sum()
    negative_predictions = predictions_df['predicted_clean_id'].isna().sum()
    
    print(f"\nSummary:")
    print(f"- Total predictions: {total_predictions}")
    print(f"- Predicted matches: {positive_predictions}")
    print(f"- Predicted non-matches: {negative_predictions}")
    print(f"- Average similarity score: {predictions_df['similarity_score'].mean():.3f}")

if __name__ == '__main__':
    main()