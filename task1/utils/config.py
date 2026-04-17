from dotenv import load_dotenv
import os

load_dotenv()


def _sanitize_ssl_env():
    """
    Some environments set REQUESTS_CA_BUNDLE/SSL_CERT_FILE to invalid paths.
    That breaks all API calls (NewsAPI/pytrends/PRAW update checks).
    """
    for var in ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"]:
        val = os.getenv(var)
        if val and not os.path.exists(val):
            print(f"Warning: {var} points to missing file -> {val}. Unsetting it for this run.")
            os.environ.pop(var, None)


_sanitize_ssl_env()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "social_analytics_project/1.0")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
POLITICAL_SUBREDDITS = [s.strip() for s in os.getenv("POLITICAL_SUBREDDITS", "politics,PoliticalDiscussion,Conservative,Liberal,democrats,Republican,worldnews,news").split(",") if s.strip()]
PYTRENDS_REGION = os.getenv("PYTRENDS_REGION", "global")
REDDIT_SEARCH_SORTS = [s.strip().lower() for s in os.getenv("REDDIT_SEARCH_SORTS", "relevance,new,top").split(",") if s.strip()]
REDDIT_TOP_TIME_FILTER = os.getenv("REDDIT_TOP_TIME_FILTER", "month")
REDDIT_FALLBACK_CSV_PATH = os.getenv("REDDIT_FALLBACK_CSV_PATH", "")
SENTIMENT_POSITIVE_THRESHOLD = float(os.getenv("SENTIMENT_POSITIVE_THRESHOLD", "0.05"))
SENTIMENT_NEGATIVE_THRESHOLD = float(os.getenv("SENTIMENT_NEGATIVE_THRESHOLD", "-0.05"))


def check_config():
    missing = []
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        missing.append('REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET')
    if not NEWS_API_KEY:
        missing.append('NEWS_API_KEY')
    return missing
