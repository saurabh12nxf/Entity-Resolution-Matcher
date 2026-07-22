#!/usr/bin/env python3
"""
String similarity-based entity matching using rapidfuzz.
For each messy record, finds the best matching clean record based on token_sort_ratio.
"""

import pandas as pd
from rapidfuzz import fuzz
import sys

# Configurable threshold
SIMILARITY_THRESHOLD = 80

def create_match_string(record):
    """Create a concatenated string for matching."""
    return f"{record['full_name']} {record['address']}"

def find_best_matches(clean_df, messy_df, threshold=SIMILARITY_THRESHOLD):
    """Find best string similarity matches for each messy record."""
    results = []
    
    # Precompute clean record strings
    clean_strings = {}
    for _, clean_record in clean_df.iterrows():
        clean_strings[clean_record['id']] = create_match_string(clean_record)
    
    print(f"Finding matches using string similarity (threshold: {threshold})...")
    
    for idx, messy_record in messy_df.iterrows():
        messy_string = create_match_string(messy_record)
        
        best_score = 0
        best_match_id = None
        
        # Compare against all clean records
        for clean_id, clean_string in clean_strings.items():
            score = fuzz.token_sort_ratio(messy_string, clean_string)
            
            if score > best_score:
                best_score = score
                best_match_id = clean_id
        
        # Only predict match if above threshold
        predicted_id = best_match_id if best_score >= threshold else None
        
        results.append({
            'messy_id': messy_record['messy_id'],
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
    
    # Run string similarity matching
    predictions_df = find_best_matches(clean_df, messy_df)
    
    # Save results
    predictions_df.to_csv('predictions_string.csv', index=False)
    print(f"String similarity predictions saved to predictions_string.csv")
    
    # Print summary statistics
    total_predictions = len(predictions_df)
    positive_predictions = predictions_df['predicted_clean_id'].notna().sum()
    negative_predictions = predictions_df['predicted_clean_id'].isna().sum()
    
    print(f"\nSummary:")
    print(f"- Total predictions: {total_predictions}")
    print(f"- Predicted matches: {positive_predictions}")
    print(f"- Predicted non-matches: {negative_predictions}")
    print(f"- Average similarity score: {predictions_df['similarity_score'].mean():.1f}")

if __name__ == '__main__':
    main()