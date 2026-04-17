from typing import List, Tuple

try:
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import wordnet as wn
except Exception:  # nltk is an optional dependency for Task 3
    nltk = None
    swn = None
    wn = None


NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "barely", "without", "n't"}


_NLTK_CORPORA_READY = False


def _ensure_sentiwordnet_ready() -> None:
    """Ensure NLTK corpora needed for SentiWordNet are available.

    This intentionally fails (no fallback) if downloads are blocked.
    """
    global _NLTK_CORPORA_READY
    if _NLTK_CORPORA_READY:
        return

    if nltk is None or swn is None or wn is None:
        raise ImportError(
            "NLTK is required for the real SentiWordNet classifier. "
            "Install it with: pip install nltk"
        )

    required = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/sentiwordnet", "sentiwordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, download_name in required:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            ok = nltk.download(download_name, quiet=True)
            if not ok:
                raise RuntimeError(
                    f"NLTK corpus '{download_name}' is missing and could not be downloaded. "
                    "Run nltk.download() manually or ensure internet access."
                )

    _NLTK_CORPORA_READY = True


def load_wordlist(path: str) -> set:
    words = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if not w or w.startswith(";"):
                continue
            words.add(w)
    return words


def sentiwordnet_style_predict(tokens: List[str], positive_words: set, negative_words: set, neutral_margin=0.05) -> Tuple[float, str]:
    """Real SentiWordNet sentiment classifier.

    Uses NLTK's `sentiwordnet` + WordNet synsets to compute token polarity as the
    average of (pos_score - neg_score) across a few synsets. The overall score
    is the mean across scored tokens.

    NOTE: `positive_words`/`negative_words` are kept only for backward
    compatibility with the pipeline signature (they are not used here).
    """

    _ensure_sentiwordnet_ready()

    total = 0.0
    scored = 0
    neg_scope = 0
    max_synsets = 3

    for raw_t in tokens:
        t = (raw_t or "").lower().strip()
        if not t:
            continue

        if t in NEGATIONS:
            neg_scope = 3
            continue

        synsets = wn.synsets(t)
        if not synsets:
            if neg_scope > 0:
                neg_scope -= 1
            continue

        syn_scores = []
        for syn in synsets[:max_synsets]:
            try:
                ss = swn.senti_synset(syn.name())
                syn_scores.append(float(ss.pos_score()) - float(ss.neg_score()))
            except Exception:
                continue

        if not syn_scores:
            if neg_scope > 0:
                neg_scope -= 1
            continue

        token_score = sum(syn_scores) / len(syn_scores)
        if neg_scope > 0:
            token_score *= -1

        total += token_score
        scored += 1

        if neg_scope > 0:
            neg_scope -= 1

    score = 0.0 if scored == 0 else (total / scored)
    if abs(score) < neutral_margin:
        return score, "neutral"
    return score, ("positive" if score > 0 else "negative")


def bing_liu_predict_with_negation(tokens: List[str], positive_words: set, negative_words: set, neutral_margin=0.05) -> Tuple[float, str]:
    pos = 0
    neg = 0
    neg_scope = 0
    for t in tokens:
        if t in NEGATIONS:
            neg_scope = 3
            continue

        polarity = 0
        if t in positive_words:
            polarity = 1
        elif t in negative_words:
            polarity = -1

        if polarity != 0 and neg_scope > 0:
            polarity *= -1

        if polarity > 0:
            pos += 1
        elif polarity < 0:
            neg += 1

        if neg_scope > 0:
            neg_scope -= 1

    m = pos + neg
    score = 0.0 if m == 0 else (pos - neg) / m
    if abs(score) < neutral_margin:
        return score, "neutral"
    return score, ("positive" if score > 0 else "negative")
