from pytrends.request import TrendReq
from newsapi import NewsApiClient
import os
from utils.config import NEWS_API_KEY


def _strict_news_api() -> bool:
    return os.getenv("STRICT_NEWS_API", "").strip().lower() in {"1", "true", "yes", "on"}


def get_top_trends(limit=10, geo='GLOBAL'):
    """Return a list of top trending search terms using pytrends.

    Tries several regions and falls back to NewsAPI top headlines or a
    small default list when Google Trends access fails.
    """
    pn_candidates = []
    if geo and geo.upper() == 'GLOBAL':
        pn_candidates = ['global', 'united_states', 'united_kingdom']
    else:
        pn_candidates = [geo]

    # Try pytrends for each candidate region
    for pn in pn_candidates:
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            df = pytrends.trending_searches(pn=pn)
            if df is None:
                continue
            if hasattr(df, 'iloc'):
                items = df.iloc[:, 0].astype(str).tolist()
            else:
                items = list(df)
            if items:
                return items[:limit]
        except Exception as e:
            print('pytrends error for', pn, ':', e)

    # Fallback: try NewsAPI top headlines as proxy for trending topics
    if NEWS_API_KEY:
        try:
            client = NewsApiClient(api_key=NEWS_API_KEY)
            res = client.get_top_headlines(language='en', page_size=limit)
            articles = res.get('articles', []) if res else []
            titles = [a.get('title') for a in articles if a.get('title')]
            if titles:
                return titles[:limit]
        except Exception as e:
            print('NewsAPI fallback error:', e)

            if _strict_news_api():
                raise RuntimeError(f"NewsAPI fallback error: {e}") from e

    # Final fallback: sensible defaults
    if _strict_news_api():
        raise RuntimeError(
            "Could not fetch trends via pytrends, and NewsAPI fallback did not return results. "
            "STRICT_NEWS_API is enabled, so no hardcoded default trends will be used."
        )
    defaults = [
        'election',
        'economy',
        'immigration',
        'climate change',
        'healthcare',
    ]
    return defaults[:limit]
