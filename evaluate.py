#!/usr/bin/env python3
"""
Evaluate the performance of string similarity and embedding-based entity matching methods.
Computes precision, recall, and F1 scores, and shows disagreement examples.
"""

import pandas as pd
import sys

def compute_metrics(y_true, y_pred, method_name="", debug_true_negatives=False):
    """Compute precision, recall, and F1 score using correct confusion matrix logic."""
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    true_negative_debug = []
    
    # Loop over each record exactly once
    for i, (true_match_id, predicted_id) in enumerate(zip(y_true, y_pred)):
        # Convert pandas null values to None for consistent comparison
        if pd.isna(true_match_id):
            true_match_id = None
        if pd.isna(predicted_id):
            predicted_id = None
            
        is_true_positive_case = (true_match_id is not None)
        predicted_a_match = (predicted_id is not None)
        
        classification = None
        
        if is_true_positive_case and predicted_a_match and predicted_id == true_match_id:
            TP += 1
            classification = "TP"
        elif is_true_positive_case and (not predicted_a_match or predicted_id != true_match_id):
            FN += 1
            classification = "FN"
        elif not is_true_positive_case and predicted_a_match:
            FP += 1
            classification = "FP"
        elif not is_true_positive_case and not predicted_a_match:
            TN += 1
            classification = "TN"
        
        # Debug true negatives
        if not is_true_positive_case and debug_true_negatives:
            true_negative_debug.append({
                'record_index': i,
                'true_match_id': true_match_id,
                'predicted_id': predicted_id,
                'classification': classification
            })
    
    # Print debug info for true negatives if requested
    if debug_true_negatives and true_negative_debug:
        print(f"\n--- DEBUG: True Negative Records for {method_name} ---")
        for debug_info in true_negative_debug:
            print(f"Record {debug_info['record_index']}: true_match_id={debug_info['true_match_id']}, "
                  f"predicted_id={debug_info['predicted_id']} → {debug_info['classification']}")
        print(f"--- End Debug for {method_name} ---\n")
    
    # Verify counts sum to total records
    total = TP + FP + FN + TN
    assert total == len(y_true), f"Confusion matrix total ({total}) doesn't match dataset size ({len(y_true)})"
    
    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': TP,
        'false_positives': FP,
        'false_negatives': FN,
        'true_negatives': TN,
        'total': total
    }

def find_disagreements(messy_df, string_pred_df, embedding_pred_df, limit=5):
    """Find cases where the two methods disagree."""
    disagreements = []
    
    # Rename columns to avoid conflicts during merge
    string_renamed = string_pred_df.rename(columns={
        'predicted_clean_id': 'predicted_clean_id_string',
        'similarity_score': 'similarity_score_string'
    })
    
    embedding_renamed = embedding_pred_df.rename(columns={
        'predicted_clean_id': 'predicted_clean_id_embedding', 
        'similarity_score': 'similarity_score_embedding'
    })
    
    # Merge all data
    merged = messy_df.merge(string_renamed, on='messy_id')
    merged = merged.merge(embedding_renamed, on='messy_id')
    
    # Find disagreements
    disagreement_mask = merged['predicted_clean_id_string'] != merged['predicted_clean_id_embedding']
    disagreement_cases = merged[disagreement_mask]
    
    for _, row in disagreement_cases.head(limit).iterrows():
        string_pred = row['predicted_clean_id_string']
        embedding_pred = row['predicted_clean_id_embedding']
        true_match = row['true_match_id']
        
        # Determine which is correct (FIX BUG 2: Handle true negatives properly)
        # For true negatives (true_match is None), correctly predicting None is CORRECT
        if true_match is None:
            # True negative case
            string_correct = (string_pred is None)
            embedding_correct = (embedding_pred is None)
        else:
            # True positive case
            string_correct = (string_pred == true_match)
            embedding_correct = (embedding_pred == true_match)
        
        if string_correct and not embedding_correct:
            winner = "String method"
        elif embedding_correct and not string_correct:
            winner = "Embedding method"
        elif string_correct and embedding_correct:
            winner = "Both correct (different valid matches)"
        else:
            winner = "Both incorrect"
        
        disagreements.append({
            'messy_record': f"{row['full_name']} | {row['address']}",
            'string_prediction': string_pred or "NO MATCH",
            'embedding_prediction': embedding_pred or "NO MATCH", 
            'ground_truth': true_match or "NO MATCH",
            'correct_method': winner,
            'string_score': row['similarity_score_string'],
            'embedding_score': row['similarity_score_embedding']
        })
    
    return disagreements

def main():
    print("Loading evaluation data...")
    try:
        messy_df = pd.read_csv('messy_records.csv')
        string_pred_df = pd.read_csv('predictions_string.csv')
        embedding_pred_df = pd.read_csv('predictions_embeddings.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run generate_data.py and both matching scripts first.")
        sys.exit(1)
    
    print(f"Loaded {len(messy_df)} ground truth records")
    print(f"Loaded {len(string_pred_df)} string predictions")
    print(f"Loaded {len(embedding_pred_df)} embedding predictions")
    
    # Extract ground truth and predictions
    ground_truth = messy_df.set_index('messy_id')['true_match_id'].to_dict()
    
    string_predictions = string_pred_df.set_index('messy_id')['predicted_clean_id'].to_dict()
    embedding_predictions = embedding_pred_df.set_index('messy_id')['predicted_clean_id'].to_dict()
    
    # Ensure all records are present
    messy_ids = set(ground_truth.keys())
    assert messy_ids == set(string_predictions.keys()), "String predictions missing records"
    assert messy_ids == set(embedding_predictions.keys()), "Embedding predictions missing records"
    
    # Compute metrics for both methods
    y_true = [ground_truth[mid] for mid in messy_ids]
    y_pred_string = [string_predictions[mid] for mid in messy_ids]
    y_pred_embedding = [embedding_predictions[mid] for mid in messy_ids]
    
    # Compute metrics for both methods with debug info
    string_metrics = compute_metrics(y_true, y_pred_string, "String Similarity", debug_true_negatives=True)
    embedding_metrics = compute_metrics(y_true, y_pred_embedding, "Embeddings", debug_true_negatives=True)
    
    # Print comparison table
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"{'Method':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 60)
    print(f"{'String Similarity':<20} {string_metrics['precision']:<12.3f} {string_metrics['recall']:<12.3f} {string_metrics['f1']:<12.3f}")
    print(f"{'Embeddings':<20} {embedding_metrics['precision']:<12.3f} {embedding_metrics['recall']:<12.3f} {embedding_metrics['f1']:<12.3f}")
    
    print(f"\nConfusion Matrix (all {string_metrics['total']} records):")
    print(f"String Similarity:")
    print(f"  TP: {string_metrics['true_positives']:3d} | FP: {string_metrics['false_positives']:3d}")
    print(f"  FN: {string_metrics['false_negatives']:3d} | TN: {string_metrics['true_negatives']:3d}")
    print(f"Embeddings:")
    print(f"  TP: {embedding_metrics['true_positives']:3d} | FP: {embedding_metrics['false_positives']:3d}")
    print(f"  FN: {embedding_metrics['false_negatives']:3d} | TN: {embedding_metrics['true_negatives']:3d}")
    
    # Find and display disagreements
    print(f"\n" + "="*60)
    print("METHOD DISAGREEMENT ANALYSIS")
    print("="*60)
    
    disagreements = find_disagreements(messy_df, string_pred_df, embedding_pred_df)
    
    if disagreements:
        for i, case in enumerate(disagreements, 1):
            print(f"\nDisagreement {i}:")
            print(f"  Messy Record: {case['messy_record']}")
            print(f"  String Method: {case['string_prediction']} (score: {case['string_score']:.2f})")
            print(f"  Embedding Method: {case['embedding_prediction']} (score: {case['embedding_score']:.3f})")
            print(f"  Ground Truth: {case['ground_truth']}")
            print(f"  Winner: {case['correct_method']}")
    else:
        print("No disagreements found between the two methods.")
    
    print(f"\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()