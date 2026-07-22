# Entity Resolution Matcher

This project demonstrates two different approaches to fuzzy matching of messy name/address records: string similarity using rapidfuzz and semantic embeddings using sentence-transformers. It generates synthetic messy data with realistic corruptions, applies both matching methods, and evaluates their performance with precision, recall, and F1 metrics.

## How to Run

Execute these commands in order:

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic clean and messy data
python generate_data.py

# Run string similarity matching
python match_string_similarity.py

# Run embedding-based matching
python match_embeddings.py

# Evaluate both methods and compare results
python evaluate.py
```

## Results

After running `evaluate.py`, the actual performance metrics will be:

| Method | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| String Similarity | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Embeddings | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |

## Error Analysis

[TO BE FILLED: 2-3 sentences about which method failed on which type of corruption based on the disagreement examples from evaluate.py output]

## Project Structure

- `generate_data.py`: Creates 150 clean records and 180 messy records (150 corrupted + 30 true negatives)
- `match_string_similarity.py`: Implements fuzzy matching using rapidfuzz token_sort_ratio
- `match_embeddings.py`: Implements semantic matching using sentence-transformers embeddings
- `evaluate.py`: Computes precision/recall/F1 and analyzes method disagreements
- `requirements.txt`: Project dependencies

## Data Corruptions

The synthetic messy data includes these realistic corruptions:
- Nickname substitution (Robert → Bob, Elizabeth → Liz, etc.)
- Character typos (swaps, deletions)
- Address abbreviations (Street → St, Avenue → Ave)
- Word reordering (last name first, address component swaps)
- Missing fields (dropped zip codes)

Each messy record has exactly one corruption applied, plus 30 true negative records with no corresponding clean match.