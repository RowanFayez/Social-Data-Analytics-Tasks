import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parent.parent


def _find_latest_task4_run(repo_root: Path) -> Optional[Path]:
    task4_final = repo_root / "task4" / "final_data"
    if not task4_final.exists():
        return None

    run_dirs = sorted([p for p in task4_final.glob("run_*") if p.is_dir()])
    if not run_dirs:
        return None

    return run_dirs[-1]


def _basic_clean_text(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _normalize_label(label: Any) -> str:
    if label is None:
        return "neutral"
    s = str(label).strip().lower()
    if "pos" in s:
        return "positive"
    if "neg" in s:
        return "negative"
    return "neutral"


def _load_manifest(task4_run_dir: Path) -> Dict[str, Any]:
    manifest = task4_run_dir / "task4_manifest.json"
    if not manifest.exists():
        return {}
    with manifest.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_paths(task4_run_dir: Path, manifest: Dict[str, Any]) -> Tuple[Path, Path, Optional[str]]:
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}

    model_path = None
    if isinstance(paths, dict):
        model_path = paths.get("best_model")
    if not model_path:
        model_path = str(task4_run_dir / "optimization" / "best_model.joblib")

    labeled_path = manifest.get("labeled_dataset")
    if not labeled_path:
        labeled_path = ""

    text_col = None
    opt_summary = None
    if isinstance(paths, dict):
        opt_summary = paths.get("optimization_summary")
    if opt_summary:
        try:
            with Path(opt_summary).open("r", encoding="utf-8") as f:
                opt = json.load(f)
            text_col = opt.get("text_column")
        except Exception:
            text_col = None

    return Path(model_path), Path(labeled_path) if labeled_path else Path(), text_col


def _confidence_for_model(model: Any, texts: list[str]) -> list[Optional[float]]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
        return [float(np.max(row)) for row in proba]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        scores = np.array(scores)
        if scores.ndim == 1:
            p = 1 / (1 + np.exp(-scores))
            return [float(max(x, 1 - x)) for x in p]
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        p = exp / exp.sum(axis=1, keepdims=True)
        return [float(np.max(row)) for row in p]

    return [None for _ in texts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export best-model predictions for the Task 3 labeled dataset.")
    parser.add_argument(
        "--task4_run_dir",
        type=str,
        default="",
        help="Path to a Task 4 run directory (task4/final_data/run_<id>). If empty, uses latest.",
    )
    parser.add_argument(
        "--labeled_dataset",
        type=str,
        default="",
        help="Path to labeled_dataset.csv. If empty, uses the path recorded in task4_manifest.json.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Output CSV path. If empty, writes to <task4_run_dir>/error_analysis/predictions_all_labeled.csv.",
    )

    args = parser.parse_args()

    repo_root = _repo_root_from_this_file()

    if args.task4_run_dir:
        task4_run_dir = Path(args.task4_run_dir)
        if not task4_run_dir.is_absolute():
            task4_run_dir = (repo_root / task4_run_dir).resolve()
    else:
        task4_run_dir = _find_latest_task4_run(repo_root) or Path()

    if not task4_run_dir.exists():
        raise FileNotFoundError(
            "Task 4 run directory not found. Provide --task4_run_dir or run task4/main.py first."
        )

    manifest = _load_manifest(task4_run_dir)
    model_path, manifest_labeled_path, text_col = _resolve_paths(task4_run_dir, manifest)

    labeled_path = Path(args.labeled_dataset) if args.labeled_dataset else manifest_labeled_path
    if labeled_path and not labeled_path.is_absolute():
        labeled_path = (repo_root / labeled_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found: {labeled_path}. Provide --labeled_dataset or ensure task4_manifest.json exists."
        )

    # Load artifacts
    model = load(model_path)
    df = pd.read_csv(labeled_path)
    if df.empty:
        raise ValueError("Labeled dataset is empty.")

    # Choose text column (match what optimization used)
    if not text_col:
        text_col = "v1_basic_text"

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in labeled dataset.")

    df["_clean_text"] = df[text_col].map(lambda x: _basic_clean_text(_safe_text(x)))
    texts = df["_clean_text"].tolist()

    preds = model.predict(texts).tolist()
    confs = _confidence_for_model(model, texts)

    true_labels = df["final_label"].map(_normalize_label).tolist() if "final_label" in df.columns else [""] * len(df)
    pred_labels = [_normalize_label(p) for p in preds]
    is_correct = [tl == pl for tl, pl in zip(true_labels, pred_labels)]

    out_df = df.copy()
    out_df["pred_label"] = pred_labels
    out_df["confidence"] = confs
    if "final_label" in out_df.columns:
        out_df["final_label"] = out_df["final_label"].map(_normalize_label)
        out_df["is_correct"] = is_correct

    out_csv = Path(args.out_csv) if args.out_csv else (task4_run_dir / "error_analysis" / "predictions_all_labeled.csv")
    if not out_csv.is_absolute():
        out_csv = (repo_root / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    mis_df = out_df[out_df.get("is_correct", True) == False].copy()  # noqa: E712
    mis_csv = out_csv.with_name("misclassified_all_labeled.csv")
    mis_df.to_csv(mis_csv, index=False)

    print("Export completed")
    print(f"- task4_run_dir: {task4_run_dir}")
    print(f"- model: {model_path}")
    print(f"- labeled_dataset: {labeled_path}")
    print(f"- text_column: {text_col}")
    print(f"- rows: {len(out_df)}")
    if "is_correct" in out_df.columns:
        print(f"- misclassified_rows: {int((~out_df['is_correct']).sum())}")
    print(f"- predictions_csv: {out_csv}")
    print(f"- misclassified_csv: {mis_csv}")


if __name__ == "__main__":
    main()
