from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def build_bow_vocabulary(token_lists: List[List[str]], max_features=2000) -> List[str]:
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab = [w for w, _ in counter.most_common(max_features)]
    return vocab


def vectorize_bow(token_lists: List[List[str]], vocab: List[str]) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(token_lists), len(vocab)), dtype=float)
    for r, tokens in enumerate(token_lists):
        for t in tokens:
            c = idx.get(t)
            if c is not None:
                mat[r, c] += 1.0
    return mat


def load_glove_embeddings(glove_path: str, wanted_words: set = None) -> Tuple[Dict[str, np.ndarray], int]:
    emb = {}
    dim = None
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            word = parts[0]
            if wanted_words is not None and word not in wanted_words:
                continue
            vec = np.array([float(x) for x in parts[1:]], dtype=float)
            emb[word] = vec
            if dim is None:
                dim = len(vec)
    if dim is None:
        dim = 50
    return emb, dim


def vectorize_glove_average(token_lists: List[List[str]], embeddings: Dict[str, np.ndarray], dim: int) -> np.ndarray:
    mat = np.zeros((len(token_lists), dim), dtype=float)
    for i, tokens in enumerate(token_lists):
        vecs = [embeddings[t] for t in tokens if t in embeddings]
        if vecs:
            mat[i] = np.mean(np.vstack(vecs), axis=0)
    return mat
