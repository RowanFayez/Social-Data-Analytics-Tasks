import praw
import pandas as pd

from utils.config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    POLITICAL_SUBREDDITS,
    REDDIT_SEARCH_SORTS,
    REDDIT_TOP_TIME_FILTER,
    REDDIT_FALLBACK_CSV_PATH,
)


def create_reddit_instance():
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET]):
        print('Reddit credentials missing.')
        return None
    try:
        return praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            check_for_updates=False,
        )
    except Exception as e:
        print("Could not initialize Reddit client. Falling back to CSV:", e)
        return None


def _load_fallback_reddit_from_csv(query, fallback_csv_path, limit=100, subreddits=None):
    if not fallback_csv_path:
        return []
    try:
        df = pd.read_csv(fallback_csv_path)
    except Exception as e:
        print(f"Could not read fallback CSV ({fallback_csv_path}):", e)
        return []

    if df.empty:
        return []

    query_l = (query or "").lower()
    for col in ["title", "selftext", "search_query", "subreddit"]:
        if col not in df.columns:
            df[col] = ""

    mask = (
        df["title"].fillna("").str.lower().str.contains(query_l, regex=False)
        | df["selftext"].fillna("").str.lower().str.contains(query_l, regex=False)
        | df["search_query"].fillna("").str.lower().str.contains(query_l, regex=False)
    )
    filtered = df[mask].copy()
    if subreddits:
        allowed = {s.lower() for s in subreddits}
        filtered = filtered[filtered["subreddit"].fillna("").str.lower().isin(allowed)]

    if filtered.empty:
        return []

    if "score" in filtered.columns:
        filtered["score"] = pd.to_numeric(filtered["score"], errors="coerce").fillna(0)
        filtered = filtered.sort_values(by="score", ascending=False)
    filtered = filtered.head(limit)

    rows = []
    for _, row in filtered.iterrows():
        rows.append({
            'id': row.get('post_id'),
            'subreddit': row.get('subreddit'),
            'title': row.get('title') or '',
            'selftext': row.get('selftext') or '',
            'score': row.get('score', 0),
            'upvote_ratio': row.get('upvote_ratio'),
            'is_self': bool(row.get('selftext')),
            'over_18': None,
            'locked': None,
            'stickied': None,
            'created_utc': row.get('created_utc'),
            'url': row.get('external_url') or row.get('post_url') or '',
            'num_comments': row.get('num_comments', 0),
            'permalink': row.get('post_url', ''),
            'search_sort': 'fallback_csv',
        })
    return rows


def search_reddit(query, subreddits=None, limit=100, sorts=None, top_time_filter=None):
    """Search political subreddits for a query with multiple sort modes."""
    reddit = create_reddit_instance()
    if subreddits is None:
        subreddits = POLITICAL_SUBREDDITS
    if sorts is None:
        sorts = REDDIT_SEARCH_SORTS or ["relevance", "new", "top"]
    if top_time_filter is None:
        top_time_filter = REDDIT_TOP_TIME_FILTER

    if reddit is None:
        return _load_fallback_reddit_from_csv(
            query=query,
            fallback_csv_path=REDDIT_FALLBACK_CSV_PATH,
            limit=limit,
            subreddits=subreddits,
        )

    sort_count = max(1, len(sorts))
    per_sort_limit = max(1, limit // sort_count)
    results = []
    seen_keys = set()

    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            for sort_mode in sorts:
                search_kwargs = {
                    "query": query,
                    "limit": per_sort_limit,
                    "sort": sort_mode,
                }
                if sort_mode == "top":
                    search_kwargs["time_filter"] = top_time_filter

                for post in subreddit.search(**search_kwargs):
                    post_id = getattr(post, "id", None)
                    key = (sub, post_id)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    results.append({
                        'id': post_id,
                        'subreddit': sub,
                        'title': getattr(post, 'title', ''),
                        'selftext': getattr(post, 'selftext', ''),
                        'score': getattr(post, 'score', 0),
                        'upvote_ratio': getattr(post, 'upvote_ratio', None),
                        'is_self': getattr(post, 'is_self', False),
                        'over_18': getattr(post, 'over_18', False),
                        'locked': getattr(post, 'locked', False),
                        'stickied': getattr(post, 'stickied', False),
                        'created_utc': getattr(post, 'created_utc', None),
                        'url': getattr(post, 'url', ''),
                        'num_comments': getattr(post, 'num_comments', 0),
                        'permalink': getattr(post, 'permalink', ''),
                        'search_sort': sort_mode,
                    })
        except Exception as e:
            print(f"Reddit search error for /r/{sub}:", e)
    if results:
        return results

    return _load_fallback_reddit_from_csv(
        query=query,
        fallback_csv_path=REDDIT_FALLBACK_CSV_PATH,
        limit=limit,
        subreddits=subreddits,
    )
