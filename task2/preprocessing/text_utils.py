import re
import unicodedata

try:
    import emoji  # type: ignore
except Exception:
    emoji = None


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from", "has", "have",
    "had", "he", "her", "hers", "him", "his", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "my", "myself", "no", "not", "of", "on", "or", "our", "ours", "ourselves", "she", "so",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "to", "too", "under", "up", "us", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves",
    "will", "can", "could", "should", "would", "just", "than", "about", "after", "before", "also",
    "do", "does", "did", "done", "doing", "more", "most", "much", "many", "such", "only", "own",
    "over", "out", "off", "once", "because", "during", "again", "further", "s", "t", "d", "ll", "m",
    "re", "ve", "y",
}

NOISE_TOKENS = {
    "http", "https", "www", "com", "org", "net", "amp", "nbsp", "gt", "lt", "rt",
}

BOILERPLATE_PHRASES = [
    "ask me anything",
    "ask us anything",
    "proof",
    "learn more",
    "look up the options",
    "all wrapped up",
    "here s who will be answering",
    "we are the team behind",
    "questions today between",
]

MAX_ANALYSIS_TOKENS = 350

RAW_BOILERPLATE_LINE_KEYWORDS = [
    "proof",
    "ask me anything",
    "ask us anything",
    "learn more",
    "look up the options",
    "all wrapped up",
    "here's who will be answering",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
MENTION_RE = re.compile(r"[@#]\w+")
SLASH_MENTION_RE = re.compile(r"(?:^|\s)[/\\]?(?:u|r)[/\\][a-z0-9_-]+", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#\w+")
NUMBER_RE = re.compile(r"\d+")
NON_WORD_RE = re.compile(r"[^a-z0-9'\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STAR_RE = re.compile(r"\*+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def normalize_common_mojibake(text):
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Â": "",
        "Ã¢â‚¬â„¢": "'",
        "Ã¢â‚¬Ëœ": "'",
        "Ã¢â‚¬Å“": '"',
        "Ã¢â‚¬Â\x9d": '"',
        "Ã¢â‚¬â€œ": "-",
        "Ã¢â‚¬â€\x9d": "-",
        "Ã¢â‚¬Â¦": "...",
        "Ã‚": "",
    }
    out = text
    for bad, good in replacements.items():
        out = out.replace(bad, good)
    return out


def strip_raw_boilerplate_lines(text):
    if not text:
        return ""
    kept = []
    for line in str(text).splitlines():
        low = line.strip().lower()
        if not low:
            continue
        if any(k in low for k in RAW_BOILERPLATE_LINE_KEYWORDS):
            continue
        if low.startswith("**about ") or low.startswith("about "):
            continue
        if low.startswith("**edit") or low.startswith("edit:") or low.startswith("update:"):
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    if emoji is not None:
        try:
            text = emoji.demojize(text)
        except Exception:
            pass
    text = strip_raw_boilerplate_lines(text)
    text = unicodedata.normalize("NFKC", text)
    text = normalize_common_mojibake(text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = STAR_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = SLASH_MENTION_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = NUMBER_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def remove_boilerplate_sentences(text):
    if not text:
        return ""
    sentences = SENTENCE_SPLIT_RE.split(text)
    kept = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("about ") and ":" in low[:120]:
            continue
        if any(phrase in low for phrase in BOILERPLATE_PHRASES):
            continue
        kept.append(s)
    out = " ".join(kept).strip()
    out = MULTI_SPACE_RE.sub(" ", out).strip()
    # Fallback: do not over-prune to empty/near-empty text.
    if len(out.split()) < 12:
        return text
    return out


def tokenize(text):
    if not text:
        return []
    return TOKEN_RE.findall(text)


def heuristic_stem(token):
    if len(token) <= 3:
        return token

    suffixes = [
        "ization", "ational", "fulness", "ousness", "iveness", "tional", "biliti",
        "ing", "edly", "edly", "edly", "ed", "ly", "ies", "sses", "ment", "tion", "s",
    ]
    for suf in suffixes:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            base = token[: -len(suf)]
            if suf == "ies":
                return base + "y"
            if suf == "sses":
                return base + "ss"
            return base
    return token


def preprocess_text(raw_text):
    cleaned = clean_text(raw_text)
    cleaned = remove_boilerplate_sentences(cleaned)
    tokens = tokenize(cleaned)
    tokens_no_stop = [
        t for t in tokens
        if t not in STOPWORDS and t not in NOISE_TOKENS and len(t) > 1
    ]
    stemmed_tokens = [heuristic_stem(t) for t in tokens_no_stop]

    was_truncated = len(stemmed_tokens) > MAX_ANALYSIS_TOKENS
    if was_truncated:
        stemmed_tokens = stemmed_tokens[:MAX_ANALYSIS_TOKENS]

    final_text = " ".join(stemmed_tokens)
    return {
        "clean_text": cleaned,
        "tokens": tokens,
        "tokens_no_stop": tokens_no_stop,
        "stemmed_tokens": stemmed_tokens,
        "final_text": final_text,
        "was_truncated_for_analysis": was_truncated,
    }
