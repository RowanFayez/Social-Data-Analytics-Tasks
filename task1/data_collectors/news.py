from newsapi import NewsApiClient

from utils.config import NEWS_API_KEY


def fetch_news_for_query(query, page_size=5):
    """Fetch recent news articles for a query using News API."""
    if not NEWS_API_KEY:
        print('NEWS_API_KEY not set')
        return []
    client = NewsApiClient(api_key=NEWS_API_KEY)
    try:
        res = client.get_everything(q=query, language='en', page_size=page_size, sort_by='relevancy')
        return res.get('articles', [])
    except Exception as e:
        print('NewsAPI error:', e)
        return []
