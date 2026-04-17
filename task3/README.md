# Task 3 - Full Sentiment Modelling Pipeline

Implements the requested workflow:

1. API-based labeling (3 prompts) + Fleiss Kappa
2. Text representation:
- Bag of Words
- GloVe (average embeddings)
3. Modelling:
- Lexical:
  - SentiWordNet-style classifier
  - Bing Liu dictionary model (with negation handling)
- Machine learning:
  - Naive Bayes
  - Decision Tree
4. Evaluation and artifact export

## Default Input
- `..\task2\final_data\processed\preprocessed_posts.csv`

## Run (home requirement 200 records)
```bash
python main.py --sample_size 200 --gemini_api_key YOUR_KEY
```

## Run (section requirement 100 records)
```bash
python main.py --sample_size 100 --gemini_api_key YOUR_KEY
```

## Outputs
Inside `task3/final_data/`:
- `labels/labeled_dataset.csv`
- `labels/fleiss_kappa.json`
- `representations/*` (BoW + GloVe for each preprocessing variant)
- `models/lexical_predictions.csv`
- `models/ml_results.csv`
- `models/model_metrics_summary.csv`
- `reports/task3_summary.json`
