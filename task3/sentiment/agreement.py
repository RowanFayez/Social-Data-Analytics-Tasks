from typing import Dict, List


LABELS = ["positive", "neutral", "negative"]


def normalize_label(label: str) -> str:
    if not label:
        return "neutral"
    s = str(label).strip().lower()
    if "pos" in s:
        return "positive"
    if "neg" in s:
        return "negative"
    return "neutral"


def majority_vote(labels: List[str]) -> str:
    votes = [normalize_label(x) for x in labels]
    counts: Dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
    for v in votes:
        counts[v] += 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if top[0][1] == top[1][1]:
        return "neutral"
    return top[0][0]


def fleiss_kappa_from_ratings(ratings: List[List[str]]) -> float:
    """
    ratings: N subjects x n_raters labels
    """
    if not ratings:
        return 0.0

    n_subjects = len(ratings)
    n_raters = len(ratings[0])
    if n_raters < 2:
        return 0.0

    # n_ij counts matrix
    n_ij = []
    for row in ratings:
        counts = [0, 0, 0]
        for r in row:
            lab = normalize_label(r)
            counts[LABELS.index(lab)] += 1
        n_ij.append(counts)

    # P_i
    p_i_vals = []
    for counts in n_ij:
        numerator = sum(c * (c - 1) for c in counts)
        denom = n_raters * (n_raters - 1)
        p_i_vals.append(numerator / denom if denom else 0.0)
    p_bar = sum(p_i_vals) / n_subjects

    # p_j
    p_j = []
    for j in range(len(LABELS)):
        col_sum = sum(row[j] for row in n_ij)
        p_j.append(col_sum / (n_subjects * n_raters))
    p_e = sum(x * x for x in p_j)

    denom = (1 - p_e)
    if denom == 0:
        return 0.0
    return (p_bar - p_e) / denom
