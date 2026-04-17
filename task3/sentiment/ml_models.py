import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


LABEL_TO_INT = {"negative": 0, "neutral": 1, "positive": 2}
INT_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def encode_labels(labels: List[str]) -> np.ndarray:
    return np.array([LABEL_TO_INT.get(str(x).lower(), 1) for x in labels], dtype=int)


def decode_labels(y: np.ndarray) -> List[str]:
    return [INT_TO_LABEL.get(int(v), "neutral") for v in y.tolist()]


def stratified_split_indices(y: np.ndarray, test_size=0.2, seed=42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.array(sorted(train_idx), dtype=int), np.array(sorted(test_idx), dtype=int)


@dataclass
class MultinomialNBModel:
    class_log_prior: np.ndarray
    feature_log_prob: np.ndarray


def train_multinomial_nb(X: np.ndarray, y: np.ndarray, alpha=1.0) -> MultinomialNBModel:
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    class_count = np.zeros(n_classes, dtype=float)
    feature_count = np.zeros((n_classes, n_features), dtype=float)

    for c in range(n_classes):
        Xc = X[y == c]
        class_count[c] = Xc.shape[0]
        feature_count[c] = Xc.sum(axis=0)

    smoothed_fc = feature_count + alpha
    smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)

    class_log_prior = np.log(class_count / class_count.sum())
    feature_log_prob = np.log(smoothed_fc / smoothed_cc)
    return MultinomialNBModel(class_log_prior=class_log_prior, feature_log_prob=feature_log_prob)


def predict_multinomial_nb(model: MultinomialNBModel, X: np.ndarray) -> np.ndarray:
    jll = X @ model.feature_log_prob.T + model.class_log_prior
    return np.argmax(jll, axis=1)


@dataclass
class GaussianNBModel:
    class_prior: np.ndarray
    mean: np.ndarray
    var: np.ndarray


def train_gaussian_nb(X: np.ndarray, y: np.ndarray, var_smoothing=1e-9) -> GaussianNBModel:
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]
    class_prior = np.zeros(n_classes, dtype=float)
    mean = np.zeros((n_classes, n_features), dtype=float)
    var = np.zeros((n_classes, n_features), dtype=float)

    for i, c in enumerate(classes):
        Xc = X[y == c]
        class_prior[i] = Xc.shape[0] / X.shape[0]
        mean[i, :] = Xc.mean(axis=0)
        var[i, :] = Xc.var(axis=0) + var_smoothing

    return GaussianNBModel(class_prior=class_prior, mean=mean, var=var)


def predict_gaussian_nb(model: GaussianNBModel, X: np.ndarray) -> np.ndarray:
    log_probs = []
    for i in range(model.mean.shape[0]):
        prior = math.log(model.class_prior[i] + 1e-12)
        ll = -0.5 * np.sum(np.log(2 * np.pi * model.var[i]))
        ll -= 0.5 * np.sum(((X - model.mean[i]) ** 2) / model.var[i], axis=1)
        log_probs.append(prior + ll)
    log_probs = np.vstack(log_probs).T
    return np.argmax(log_probs, axis=1)


class SimpleDecisionTree:
    def __init__(self, max_depth=6, min_samples_split=6, feature_subset=60, seed=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subset = feature_subset
        self.seed = seed
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        self.tree = self._build(X, y, depth=0, rng=rng)

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        probs = [np.mean(y == c) for c in np.unique(y)]
        return 1 - sum(p * p for p in probs)

    def _best_split(self, X, y, rng):
        n_samples, n_features = X.shape
        base_gini = self._gini(y)
        if base_gini == 0:
            return None

        feat_idx = np.arange(n_features)
        if n_features > self.feature_subset:
            feat_idx = rng.choice(feat_idx, size=self.feature_subset, replace=False)

        best = None
        best_gain = 0
        for f in feat_idx:
            col = X[:, f]
            threshold = 0.5 if np.array_equal(col, col.astype(int)) else np.median(col)
            left = y[col <= threshold]
            right = y[col > threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            g_left = self._gini(left)
            g_right = self._gini(right)
            weighted = (len(left) / n_samples) * g_left + (len(right) / n_samples) * g_right
            gain = base_gini - weighted
            if gain > best_gain:
                best_gain = gain
                best = (f, threshold)
        return best

    def _build(self, X, y, depth, rng):
        node = {"leaf": False, "pred": int(np.bincount(y).argmax())}
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            node["leaf"] = True
            return node

        split = self._best_split(X, y, rng)
        if split is None:
            node["leaf"] = True
            return node

        f, th = split
        left_mask = X[:, f] <= th
        right_mask = ~left_mask
        node["feature"] = int(f)
        node["threshold"] = float(th)
        node["left"] = self._build(X[left_mask], y[left_mask], depth + 1, rng)
        node["right"] = self._build(X[right_mask], y[right_mask], depth + 1, rng)
        return node

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["pred"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X], dtype=int)
