from typing import Dict, List

import numpy as np


LABELS = ["negative", "neutral", "positive"]
LBL_TO_INT = {l: i for i, l in enumerate(LABELS)}


def confusion_matrix(y_true: List[str], y_pred: List[str]) -> np.ndarray:
    m = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        ti = LBL_TO_INT.get(str(t).lower(), 1)
        pi = LBL_TO_INT.get(str(p).lower(), 1)
        m[ti, pi] += 1
    return m


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0

    precisions = []
    recalls = []
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }
