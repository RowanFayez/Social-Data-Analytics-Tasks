import pandas as pd
from datetime import datetime, timezone

from data_collectors.trends import get_top_trends
from data_collectors.news import fetch_news_for_query
from data_collectors.reddit_search import search_reddit
from analysis.sentiment import analyze_text
from utils.config import SENTIMENT_POSITIVE_THRESHOLD, SENTIMENT_NEGATIVE_THRESHOLD


def _label_sentiment(compound):
    if compound >= SENTIMENT_POSITIVE_THRESHOLD:
        return "positive"
    if compound <= SENTIMENT_NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def _as_datetime_utc(ts):
    if ts is None:
        return None
    try:
        if isinstance(ts, float) and pd.isna(ts):
            return None
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        return None


def _clean_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _to_int(value, default=0):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(value)
    except Exception:
        return default


def run_pipeline(top_n=10, reddit_limit=50, news_per_term=3, geo='GLOBAL'):
    """Run trends -> news -> reddit and return rich task1 datasets."""
    trends = get_top_trends(limit=top_n, geo=geo)

    trend_rows = []
    news_rows = []
    post_rows = []
    enriched_rows = []

    for rank, term in enumerate(trends, start=1):
        trend_rows.append({
            "trend_term": term,
            "trend_rank": rank,
            "geo": geo,
        })

        articles = fetch_news_for_query(term, page_size=news_per_term)
        news_titles = []
        for idx, article in enumerate(articles, start=1):
            title = (article.get('title') or '').strip()
            desc = (article.get('description') or '').strip()
            content = " ".join([title, desc]).strip()
            sent = analyze_text(content)
            news_titles.append(title)
            news_rows.append({
                'term': term,
                'trend_rank': rank,
                'article_rank': idx,
                'source': article.get('source', {}).get('name'),
                'title': title,
                'description': desc,
                'url': article.get('url'),
                'published_at': article.get('publishedAt'),
                'news_compound': sent.get('compound'),
                'news_neg': sent.get('neg'),
                'news_neu': sent.get('neu'),
                'news_pos': sent.get('pos'),
            })

        reddit_posts = search_reddit(term, limit=reddit_limit)
        for post in reddit_posts:
            title = _clean_text(post.get('title'))
            selftext = _clean_text(post.get('selftext'))
            full_text = " ".join([title, selftext]).strip()
            sent = analyze_text(full_text)
            created_dt = _as_datetime_utc(post.get('created_utc'))
            created_iso = created_dt.isoformat() if created_dt else None
            score = _to_int(post.get('score'))
            num_comments = _to_int(post.get('num_comments'))

            post_rows.append({
                'term': term,
                'trend_rank': rank,
                'post_id': post.get('id'),
                'subreddit': post.get('subreddit'),
                'title': title,
                'selftext': selftext,
                'score': score,
                'num_comments': num_comments,
                'upvote_ratio': post.get('upvote_ratio'),
                'is_self': post.get('is_self'),
                'over_18': post.get('over_18'),
                'locked': post.get('locked'),
                'stickied': post.get('stickied'),
                'created_utc': post.get('created_utc'),
                'created_datetime_utc': created_iso,
                'url': post.get('url'),
                'permalink': post.get('permalink'),
                'search_sort': post.get('search_sort'),
            })

            lowered_term = term.lower()
            title_l = title.lower()
            body_l = selftext.lower()
            engagement = score + num_comments
            enriched_rows.append({
                'term': term,
                'trend_rank': rank,
                'post_id': post.get('id'),
                'subreddit': post.get('subreddit'),
                'title': title,
                'selftext': selftext,
                'score': score,
                'num_comments': num_comments,
                'engagement': engagement,
                'upvote_ratio': post.get('upvote_ratio'),
                'is_self': post.get('is_self'),
                'over_18': post.get('over_18'),
                'locked': post.get('locked'),
                'stickied': post.get('stickied'),
                'created_utc': post.get('created_utc'),
                'created_datetime_utc': created_iso,
                'created_day_of_week_utc': created_dt.strftime("%A") if created_dt else None,
                'created_hour_utc': created_dt.hour if created_dt else None,
                'url': post.get('url'),
                'permalink': post.get('permalink'),
                'search_sort': post.get('search_sort'),
                'title_char_len': len(title),
                'selftext_char_len': len(selftext),
                'text_char_len': len(full_text),
                'has_selftext': bool(selftext),
                'term_in_title': lowered_term in title_l if lowered_term else False,
                'term_in_selftext': lowered_term in body_l if lowered_term else False,
                'news_count_for_term': len(articles),
                'related_news_titles': news_titles,
                'compound': sent.get('compound'),
                'neg': sent.get('neg'),
                'neu': sent.get('neu'),
                'pos': sent.get('pos'),
                'sentiment_label': _label_sentiment(sent.get('compound', 0.0)),
            })

    trends_df = pd.DataFrame(trend_rows)
    news_df = pd.DataFrame(news_rows)
    posts_df = pd.DataFrame(post_rows)
    enriched_df = pd.DataFrame(enriched_rows)

    # Deduplicate repeated matches from different sort modes.
    if not enriched_df.empty:
        enriched_df = (
            enriched_df
            .sort_values(by=['engagement', 'created_utc'], ascending=[False, False], na_position='last')
            .drop_duplicates(subset=['term', 'subreddit', 'post_id'], keep='first')
            .reset_index(drop=True)
        )

    if not posts_df.empty:
        posts_df = (
            posts_df
            .sort_values(by=['score', 'created_utc'], ascending=[False, False], na_position='last')
            .drop_duplicates(subset=['term', 'subreddit', 'post_id'], keep='first')
            .reset_index(drop=True)
        )

    if enriched_df.empty:
        term_summary_df = pd.DataFrame()
        subreddit_summary_df = pd.DataFrame()
    else:
        term_summary_df = (
            enriched_df
            .groupby('term', dropna=False)
            .agg(
                trend_rank=('trend_rank', 'min'),
                posts=('post_id', 'count'),
                unique_subreddits=('subreddit', 'nunique'),
                avg_compound=('compound', 'mean'),
                avg_engagement=('engagement', 'mean'),
                median_engagement=('engagement', 'median'),
                positive_posts=('sentiment_label', lambda s: int((s == 'positive').sum())),
                neutral_posts=('sentiment_label', lambda s: int((s == 'neutral').sum())),
                negative_posts=('sentiment_label', lambda s: int((s == 'negative').sum())),
            )
            .reset_index()
            .sort_values(by=['trend_rank', 'posts'], ascending=[True, False])
            .reset_index(drop=True)
        )

        subreddit_summary_df = (
            enriched_df
            .groupby('subreddit', dropna=False)
            .agg(
                posts=('post_id', 'count'),
                unique_terms=('term', 'nunique'),
                avg_compound=('compound', 'mean'),
                avg_engagement=('engagement', 'mean'),
                median_engagement=('engagement', 'median'),
            )
            .reset_index()
            .sort_values(by='posts', ascending=False)
            .reset_index(drop=True)
        )

    datasets = {
        "trends": trends_df,
        "news": news_df,
        "reddit_posts_raw": posts_df,
        "reddit_posts_enriched": enriched_df,
        "term_summary": term_summary_df,
        "subreddit_summary": subreddit_summary_df,
    }
    return datasets, trends
