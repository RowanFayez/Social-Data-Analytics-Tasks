# Political Public Opinion Analyzer (Task 1)

Task 1 pipeline for collecting political opinion signals from:
- Google Trends (topic seeds)
- Google News via NewsAPI (news context per trend)
- Reddit political subreddits (public opinion posts)

The output is now richer and organized for downstream tasks.
Each run also updates cumulative merged datasets in `final_data/`.

## What Is New
- Multi-sort Reddit search (`relevance`, `new`, `top`) for broader coverage
- Extra post features (engagement, text lengths, time features, flags)
- Sentiment label (`positive`, `neutral`, `negative`) in addition to VADER scores
- Ready-made summaries by trend term and by subreddit
- Simple output files directly in `data/` per run

## Quickstart
1. Create `.env` from `.env.example` and fill your API keys.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run:

```bash
python main.py --topn 10 --reddit_limit 120 --news_per_term 5 --geo GLOBAL
```

## Output Structure
Each run creates timestamped CSV files directly in `data/`:

- `data/trends_<run_id>.csv`
- `data/news_<run_id>.csv`
- `data/reddit_posts_raw_<run_id>.csv`
- `data/reddit_posts_enriched_<run_id>.csv`
- `data/term_summary_<run_id>.csv`
- `data/subreddit_summary_<run_id>.csv`

A backward-compatible file is still saved at:
- `data/task1_results_<run_id>.csv`

Persistent merged files for future tasks:
- `final_data/trends.csv`
- `final_data/news.csv`
- `final_data/reddit_posts_raw.csv`
- `final_data/reddit_posts_enriched.csv`
- `final_data/term_summary.csv`
- `final_data/subreddit_summary.csv`

These files are appended every run (no cross-run dedup), with safe handling when columns differ.

## Main Features in `reddit_posts_enriched`
- `engagement = score + num_comments`
- `sentiment_label` from VADER `compound`
- `term_in_title`, `term_in_selftext`
- `title_char_len`, `selftext_char_len`, `text_char_len`
- `created_day_of_week_utc`, `created_hour_utc`

## Config (Optional)
Set in `.env`:
- `REDDIT_SEARCH_SORTS=relevance,new,top`
- `REDDIT_TOP_TIME_FILTER=month`
- `SENTIMENT_POSITIVE_THRESHOLD=0.05`
- `SENTIMENT_NEGATIVE_THRESHOLD=-0.05`
