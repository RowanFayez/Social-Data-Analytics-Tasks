# Task 2 - Text Preprocessing

This task preprocesses Reddit political posts produced by Task 1.

## Input
- Default input: `..\final_data\reddit_posts_enriched.csv`

## Run
```bash
python main.py --top_k 200
```

## Outputs (`task2/final_data/processed`)
- `preprocessed_posts.csv`
- `unigram_freq.csv`
- `bigram_freq.csv`
- `tfidf_features.csv`
- `top_terms_per_post.csv`
- `preprocessing_summary.json`

## Notes
- The pipeline is dependency-light (uses only `pandas`) and does not require online downloads.
- It includes: text cleaning, tokenization, stopword removal, lightweight stemming, and feature extraction.
