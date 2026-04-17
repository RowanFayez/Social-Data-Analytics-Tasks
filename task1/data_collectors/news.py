from newsapi import NewsApiClient
import os

from utils.config import NEWS_API_KEY


def _strict_news_api() -> bool:
    return os.getenv("STRICT_NEWS_API", "").strip().lower() in {"1", "true", "yes", "on"}


def fetch_news_for_query(query, page_size=5):
    """Fetch recent news articles for a query using News API."""
    if not NEWS_API_KEY:
        msg = 'NEWS_API_KEY not set'
        if _strict_news_api():
            raise RuntimeError(msg)
        print(msg)
        return []
    client = NewsApiClient(api_key=NEWS_API_KEY)
    try:
        res = client.get_everything(q=query, language='en', page_size=page_size, sort_by='relevancy')
        return res.get('articles', [])
    except Exception as e:
        if _strict_news_api():
            raise RuntimeError(f"NewsAPI error: {e}") from e
        print('NewsAPI error:', e)
        return []
