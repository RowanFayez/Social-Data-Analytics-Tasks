import os
import re
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load


def _basic_clean_text(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parent.parent


def _find_latest_model(repo_root: Path) -> Optional[Path]:
    task4_final = repo_root / "task4" / "final_data"
    if not task4_final.exists():
        return None

    run_dirs = sorted([p for p in task4_final.glob("run_*") if p.is_dir()])
    if not run_dirs:
        return None

    # Prefer latest run by name
    for run_dir in reversed(run_dirs):
        cand = run_dir / "optimization" / "best_model.joblib"
        if cand.exists():
            return cand

    return None


def _find_latest_task4_run(repo_root: Path) -> Optional[Path]:
    task4_final = repo_root / "task4" / "final_data"
    if not task4_final.exists():
        return None

    run_dirs = sorted([p for p in task4_final.glob("run_*") if p.is_dir()])
    if not run_dirs:
        return None
    return run_dirs[-1]


def _infer_task4_run_from_model_path(model_path: Path) -> Optional[Path]:
    # Expected: .../task4/final_data/run_<id>/optimization/best_model.joblib
    try:
        if model_path.name != "best_model.joblib":
            return None
        if model_path.parent.name != "optimization":
            return None
        run_dir = model_path.parent.parent
        if run_dir.name.startswith("run_"):
            return run_dir
    except Exception:
        return None
    return None


def _resolve_model_path(repo_root: Path) -> Path:
    model_path_env = os.getenv("TASK4_MODEL_PATH", "").strip()
    if model_path_env:
        p = Path(model_path_env)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        return p

    return _find_latest_model(repo_root) or Path()


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


@st.cache_data
def _safe_load_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_resource
def _load_model(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            "Best model not found. Run Task 4 first to generate best_model.joblib, or set TASK4_MODEL_PATH."
        )
    return load(p)


def _predict(model: Any, text: str) -> Dict[str, Any]:
    cleaned = _basic_clean_text(text)
    if not cleaned:
        raise ValueError("Empty text after cleaning.")

    pred = model.predict([cleaned])[0]
    pred = str(pred)

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])
        confidence = float(np.max(proba))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function([cleaned])
        scores = np.array(scores)
        if scores.ndim == 1:
            p = 1 / (1 + np.exp(-scores))
            confidence = float(max(p[0], 1 - p[0]))
        else:
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            p = exp / exp.sum(axis=1, keepdims=True)
            confidence = float(np.max(p[0]))

    return {"sentiment": pred, "confidence": confidence}


def _format_confidence(confidence: Optional[float]) -> str:
    if confidence is None:
        return "n/a"
    # Handle NaN
    if confidence != confidence:
        return "n/a"
    return f"{confidence * 100:.1f}%"


def _render_result(result: Dict[str, Any]) -> None:
    sentiment = str(result.get("sentiment", "")).strip().lower()
    confidence = result.get("confidence")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("Sentiment", sentiment.capitalize() if sentiment else "n/a")
    with col2:
        st.metric("Confidence", _format_confidence(confidence))

    if sentiment == "positive":
        st.success("Overall sentiment looks positive.")
    elif sentiment == "negative":
        st.error("Overall sentiment looks negative.")
    else:
        st.info("Overall sentiment looks neutral.")

    with st.expander("Raw JSON response"):
        st.code(json.dumps({"sentiment": sentiment, "confidence": confidence}, indent=2), language="json")


def _render_confusion_matrix(confusion: list[list[int]], labels: list[str], title: str) -> None:
    st.subheader(title)
    df_cm = pd.DataFrame(confusion, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    st.dataframe(df_cm, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Task 4 Sentiment Analyzer", layout="centered")

    st.title("Task 4 Sentiment Analyzer")
    st.write("Paste a news headline or paragraph and get a sentiment prediction from the best optimized model.")

    repo_root = _repo_root_from_this_file()
    model_path = _resolve_model_path(repo_root)

    task4_run_dir = _infer_task4_run_from_model_path(model_path) or _find_latest_task4_run(repo_root) or Path()

    st.caption(f"Using model: {model_path}")
    if task4_run_dir and task4_run_dir.exists():
        st.caption(f"Task 4 run: {task4_run_dir}")

    try:
        model = _load_model(str(model_path))
    except Exception as e:
        st.error(str(e))
        st.stop()

    text = st.text_area(
        "News text",
        height=180,
        placeholder="Example: 'Company X reports record profits, shares jump 10%...'",
    )

    if st.button("Analyze sentiment"):
        with st.spinner("Running the model..."):
            try:
                result = _predict(model, text)
            except Exception as e:
                st.error(str(e))
            else:
                _render_result(result)

    # Evaluation section
    st.divider()
    st.header("Evaluation")
    st.write("Shows Task 4 benchmark results and optimized-model performance.")

    if not (task4_run_dir and task4_run_dir.exists()):
        st.warning("Task 4 run directory not found; cannot display evaluation artifacts.")
        return

    opt_summary_path = task4_run_dir / "optimization" / "optimization_summary.json"
    bench_path = task4_run_dir / "evaluation" / "benchmark_results.csv"
    conf_path = task4_run_dir / "evaluation" / "confusion_matrices.json"

    opt = _safe_load_json(opt_summary_path)
    bench = _safe_load_csv(str(bench_path))
    conf = _safe_load_json(conf_path)

    if opt:
        st.subheader("Best model metrics")
        test_metrics = opt.get("test_metrics") if isinstance(opt.get("test_metrics"), dict) else {}
        dummy_cv = opt.get("dummy_baseline_cv_macro_f1")
        best_cv = opt.get("best_cv_macro_f1")
        meets_20 = opt.get("meets_20pct_over_random_macro_f1")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Test accuracy", f"{float(test_metrics.get('accuracy', float('nan'))):.3f}")
        with c2:
            st.metric("Test macro-F1", f"{float(test_metrics.get('macro_f1', float('nan'))):.3f}")
        with c3:
            st.metric("CV macro-F1 (best)", f"{float(best_cv) if best_cv is not None else float('nan'):.3f}")

        if dummy_cv is not None and best_cv is not None and dummy_cv == dummy_cv and dummy_cv != 0:
            improvement = (float(best_cv) / float(dummy_cv) - 1.0) * 100.0
            st.write(
                f"Random-chance baseline (CV macro-F1): **{float(dummy_cv):.3f}**  |  Best model (CV macro-F1): **{float(best_cv):.3f}**  |  Improvement: **{improvement:.1f}%**"
            )

        if meets_20 is True:
            st.success("Best model is at least 20% higher macro-F1 than random chance (CV-based check).")
        elif meets_20 is False:
            st.error("Best model is NOT 20% higher macro-F1 than random chance (CV-based check).")
        else:
            st.info("20% over-random check unavailable.")

        test_conf = opt.get("test_confusion") if isinstance(opt.get("test_confusion"), dict) else {}
        labels = test_conf.get("labels") if isinstance(test_conf.get("labels"), list) else ["negative", "neutral", "positive"]
        confusion = test_conf.get("confusion") if isinstance(test_conf.get("confusion"), list) else None
        if confusion:
            _render_confusion_matrix(confusion=confusion, labels=[str(x) for x in labels], title="Optimized model confusion matrix (test split)")

        with st.expander("All optimized-model test metrics"):
            if isinstance(test_metrics, dict) and test_metrics:
                metrics_df = pd.DataFrame(
                    [{"metric": str(k), "value": v} for k, v in sorted(test_metrics.items(), key=lambda kv: str(kv[0]))]
                )
                st.dataframe(metrics_df, width="stretch")
            else:
                st.info("No test_metrics found in optimization summary.")
    else:
        st.warning(f"Missing optimization summary: {opt_summary_path}")

    if bench is not None and not bench.empty:
        st.subheader("Baseline benchmark models (Task 3 artifacts)")
        cols = [c for c in ["name", "macro_f1", "accuracy", "variant", "representation", "model"] if c in bench.columns]
        st.dataframe(bench[cols], width="stretch")

        if conf and isinstance(conf, dict):
            # Best baseline is the first row (benchmark_results is sorted by macro_f1)
            best_name = str(bench.iloc[0]["name"]) if "name" in bench.columns else ""
            cm_obj = conf.get(best_name) if isinstance(conf.get(best_name), dict) else None
            if cm_obj and isinstance(cm_obj.get("confusion"), list) and isinstance(cm_obj.get("labels"), list):
                _render_confusion_matrix(
                    confusion=cm_obj["confusion"],
                    labels=[str(x) for x in cm_obj["labels"]],
                    title=f"Best baseline confusion matrix ({best_name})",
                )
    else:
        st.warning(f"Missing benchmark results: {bench_path}")


if __name__ == "__main__":
    main()
