import re
from typing import List


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from", "has", "have",
    "had", "he", "her", "hers", "him", "his", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "my", "myself", "no", "not", "of", "on", "or", "our", "ours", "ourselves", "she", "so",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "to", "too", "under", "up", "us", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves",
    "will", "can", "could", "should", "would", "just", "than", "about", "after", "before", "also",
    "do", "does", "did", "done", "doing", "more", "most", "much", "many", "such", "only", "own",
    "over", "out", "off", "once", "because", "during", "again", "further",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
NON_WORD_RE = re.compile(r"[^a-z0-9'\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def basic_clean(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_RE.findall(text)


def heuristic_stem(token: str) -> str:
    if len(token) <= 3:
        return token
    suffixes = ["ization", "ational", "fulness", "ousness", "iveness", "tional", "ing", "edly", "ed", "ly", "ies", "sses", "ment", "tion", "s"]
    for suf in suffixes:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            base = token[: -len(suf)]
            if suf == "ies":
                return base + "y"
            if suf == "sses":
                return base + "ss"
            return base
    return token


def preprocess_variant(text: str, variant: str) -> List[str]:
    cleaned = basic_clean(text)
    tokens = tokenize(cleaned)

    if variant == "v1_basic":
        return [t for t in tokens if len(t) > 1]

    if variant == "v2_no_stop":
        return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]

    if variant == "v3_stem":
        tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
        return [heuristic_stem(t) for t in tokens]

    raise ValueError(f"Unknown preprocessing variant: {variant}")
