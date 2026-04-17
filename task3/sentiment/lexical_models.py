from typing import Dict, List, Tuple


NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "barely", "without", "n't"}


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
    """
    A lightweight SentiWordNet-style scorer using token polarity scores.
    (Uses +1/-1 scores from available lexicons when explicit SentiWordNet file is not provided.)
    """
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)
    m = pos + neg
    score = 0.0 if m == 0 else (pos - neg) / m

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
