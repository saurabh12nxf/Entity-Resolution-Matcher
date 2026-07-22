# Entity Resolution Matcher

This project demonstrates two different approaches to fuzzy matching of messy name/address records: string similarity using rapidfuzz and semantic embeddings using sentence-transformers. It generates synthetic messy data with realistic corruptions, applies both matching methods, and evaluates their performance with precision, recall, and F1 metrics.

The project creates 150 clean records and 180 messy records (150 corrupted versions + 30 true negatives that should not match anything), then tests how well each method can identify the correct matches.

## How to Run

Execute these commands in order:

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic clean and messy data (150 + 180 records)
python generate_data.py

# Run string similarity matching (rapidfuzz token_sort_ratio, threshold=80)
python match_string_similarity.py

# Run embedding-based matching (sentence-transformers cosine similarity, threshold=0.75)
python match_embeddings.py

# Evaluate both methods and compare results
python evaluate.py
```

## Results

Based on the evaluation of both methods:

| Method | Precision | Recall | F1 Score | TP | FP | FN | TN |
|--------|-----------|--------|----------|----|----|----|----|
| String Similarity | 1.000 | 0.993 | 0.997 | 149 | 0 | 1 | 30 |
| Embeddings | 0.943 | 0.987 | 0.964 | 148 | 9 | 2 | 21 |

## Error Analysis

**String Similarity** achieves near-perfect performance with perfect precision (no false positives) and only 1 missed match. It correctly rejects all 30 true negative records, showing excellent specificity. However, it missed 1 legitimate match due to its strict threshold.

**Embeddings** shows strong recall but lower precision, making 9 false positive predictions on records that shouldn't match anything. The semantic approach captures similarity well but is more prone to over-matching, particularly on the true negative cases where it incorrectly identified matches for records with no ground truth correspondence.

The disagreement analysis shows that string similarity tends to be more conservative (higher threshold leads to fewer false matches), while embeddings are more liberal in finding potential matches (semantic similarity can identify connections that lexical similarity misses, but sometimes incorrectly).

## Project Structure

- `generate_data.py`: Creates 150 clean records and 180 messy records (150 corrupted + 30 true negatives)
- `match_string_similarity.py`: Implements fuzzy matching using rapidfuzz token_sort_ratio (threshold: 80)
- `match_embeddings.py`: Implements semantic matching using sentence-transformers embeddings (threshold: 0.75)
- `evaluate.py`: Computes precision/recall/F1 and analyzes method disagreements
- `requirements.txt`: Project dependencies

## Data Corruptions

The synthetic messy data includes these realistic corruptions:
- **Nickname substitution**: Robert → Bob, Elizabeth → Liz, etc. (28 mappings)
- **Character typos**: Adjacent character swaps or single character deletions
- **Address abbreviations**: Street → St, Avenue → Ave, Apartment → Apt
- **Word reordering**: Last name first, address component swaps
- **Missing fields**: Dropped zip codes or apartment numbers

Each messy record has exactly one corruption applied, plus 30 true negative records with no corresponding clean match to test specificity.

## Key Findings

1. **String similarity excels at precision**: Zero false positives make it ideal when avoiding incorrect matches is critical
2. **Embeddings better at fuzzy recall**: Captures semantic similarity that string matching might miss, but at the cost of some false positives
3. **Threshold tuning matters**: Both methods' performance heavily depends on their similarity thresholds
4. **Corruption type impact**: Different corruptions affect each method differently (nickname substitution, character typos, etc.)