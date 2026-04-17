import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from sentiment.agreement import fleiss_kappa_from_ratings, majority_vote, normalize_label
from sentiment.evaluation import confusion_matrix, metrics_from_confusion
from sentiment.features import (
    build_bow_vocabulary,
    load_glove_embeddings,
    vectorize_bow,
    vectorize_glove_average,
)
from sentiment.labeling import build_labels
from sentiment.lexical_models import (
    bing_liu_predict_with_negation,
    load_wordlist,
    sentiwordnet_style_predict,
)
from sentiment.ml_models import (
    SimpleDecisionTree,
    decode_labels,
    encode_labels,
    predict_gaussian_nb,
    predict_multinomial_nb,
    stratified_split_indices,
    train_gaussian_nb,
    train_multinomial_nb,
)
from sentiment.preprocessing import preprocess_variant


PREP_VARIANTS = ["v1_basic", "v2_no_stop", "v3_stem"]


def _safe_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def _safe_text(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def run_task3(
    input_csv: str,
    output_dir: str,
    positive_words_path: str,
    negative_words_path: str,
    sample_size: int = 200,
    random_seed: int = 42,
    llm_provider: str = "gemini",
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.5-flash",
    groq_api_key: str = "",
    groq_model: str = "llama-3.1-8b-instant",
    strict_llm: bool = False,
    glove_path: str = "",
    neutral_margin: float = 0.05,
    bow_max_features: int = 1200,
):
    np.random.seed(random_seed)

    labels_dir = os.path.join(output_dir, "labels")
    reps_dir = os.path.join(output_dir, "representations")
    models_dir = os.path.join(output_dir, "models")
    reports_dir = os.path.join(output_dir, "reports")
    cache_dir = os.path.join(output_dir, "cache")
    for d in [labels_dir, reps_dir, models_dir, reports_dir, cache_dir]:
        os.makedirs(d, exist_ok=True)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    # sample
    sample_n = min(sample_size, len(df))
    df = df.sample(n=sample_n, random_state=random_seed).reset_index(drop=True)

    # base text
    if "processed_text" in df.columns:
        df["task3_text"] = df["processed_text"].map(_safe_text)
    else:
        df["task3_text"] = (df.get("title", "").map(_safe_text) + " " + df.get("selftext", "").map(_safe_text)).str.strip()

    # preprocessing variants
    for v in PREP_VARIANTS:
        tok_col = f"{v}_tokens"
        txt_col = f"{v}_text"
        df[tok_col] = df["task3_text"].map(lambda t: preprocess_variant(t, v))
        df[txt_col] = df[tok_col].map(lambda toks: " ".join(toks))

    # labeling (3 prompts)
    fallback = df["sentiment_label"].map(normalize_label).tolist() if "sentiment_label" in df.columns else None

    provider = (llm_provider or "gemini").strip().lower()
    if provider in {"none", "off", "disabled", "no"}:
        provider = "gemini"
        api_key = ""
        model_name = gemini_model
    elif provider == "groq":
        api_key = groq_api_key
        model_name = groq_model
    else:
        provider = "gemini"
        api_key = gemini_api_key
        model_name = gemini_model

    label_tuples = build_labels(
        provider=provider,
        texts=df["task3_text"].tolist(),
        api_key=api_key,
        model_name=model_name,
        cache_path=os.path.join(cache_dir, f"{provider}_label_cache.json"),
        fallback_labels=fallback,
        strict_api=strict_llm,
    )
    df["label_prompt_1"] = [x[0] for x in label_tuples]
    df["label_prompt_2"] = [x[1] for x in label_tuples]
    df["label_prompt_3"] = [x[2] for x in label_tuples]
    df["final_label"] = [majority_vote(list(x)) for x in label_tuples]

    kappa = fleiss_kappa_from_ratings([[a, b, c] for a, b, c in label_tuples])
    kappa_out = {
        "sample_size": int(len(df)),
        "n_raters": 3,
        "labels": ["positive", "neutral", "negative"],
        "fleiss_kappa": float(kappa),
    }
    _safe_write_json(os.path.join(labels_dir, "fleiss_kappa.json"), kappa_out)

    labeled_path = os.path.join(labels_dir, "labeled_dataset.csv")
    df.to_csv(labeled_path, index=False)

    # load lexical resources
    positive_words = load_wordlist(positive_words_path)
    negative_words = load_wordlist(negative_words_path)

    # GloVe setup
    glove_loaded = False
    glove_dim = 50
    glove_embeddings = {}
    if glove_path and os.path.exists(glove_path):
        all_words = set()
        for v in PREP_VARIANTS:
            for toks in df[f"{v}_tokens"].tolist():
                all_words.update(toks)
        glove_embeddings, glove_dim = load_glove_embeddings(glove_path, wanted_words=all_words)
        glove_loaded = True

    lexical_rows = []
    ml_rows = []
    # store confusion matrices for printing
    ml_confusions = []
    lex_confusions = []

    for v in PREP_VARIANTS:
        tok_lists: List[List[str]] = df[f"{v}_tokens"].tolist()
        # BoW representation
        vocab = build_bow_vocabulary(tok_lists, max_features=bow_max_features)
        X_bow = vectorize_bow(tok_lists, vocab=vocab)
        bow_df = pd.DataFrame(X_bow, columns=[f"bow_{w}" for w in vocab])
        if "post_id" in df.columns:
            bow_df.insert(0, "post_id", df["post_id"])
        bow_df.insert(1 if "post_id" in bow_df.columns else 0, "final_label", df["final_label"])
        bow_df.to_csv(os.path.join(reps_dir, f"{v}_bow.csv"), index=False)

        # GloVe representation
        if glove_loaded:
            X_glove = vectorize_glove_average(tok_lists, glove_embeddings, glove_dim)
        else:
            X_glove = np.zeros((len(df), glove_dim), dtype=float)
        glove_df = pd.DataFrame(X_glove, columns=[f"glove_{i}" for i in range(glove_dim)])
        if "post_id" in df.columns:
            glove_df.insert(0, "post_id", df["post_id"])
        glove_df.insert(1 if "post_id" in glove_df.columns else 0, "final_label", df["final_label"])
        glove_df.to_csv(os.path.join(reps_dir, f"{v}_glove.csv"), index=False)

        # lexical models
        senti_preds = []
        bing_preds = []
        senti_scores = []
        bing_scores = []
        for toks in tok_lists:
            s_score, s_pred = sentiwordnet_style_predict(
                toks, positive_words=positive_words, negative_words=negative_words, neutral_margin=neutral_margin
            )
            b_score, b_pred = bing_liu_predict_with_negation(
                toks, positive_words=positive_words, negative_words=negative_words, neutral_margin=neutral_margin
            )
            senti_scores.append(s_score)
            senti_preds.append(s_pred)
            bing_scores.append(b_score)
            bing_preds.append(b_pred)

        for i in range(len(df)):
            lexical_rows.append({
                "variant": v,
                "index": i,
                "post_id": df.loc[i, "post_id"] if "post_id" in df.columns else i,
                "true_label": df.loc[i, "final_label"],
                "sentiwordnet_style_score": senti_scores[i],
                "sentiwordnet_style_pred": senti_preds[i],
                "bing_liu_score": bing_scores[i],
                "bing_liu_pred": bing_preds[i],
            })

        # ML models (NB + DT) on BoW and GloVe
        y_text = df["final_label"].tolist()
        y = encode_labels(y_text)
        tr_idx, te_idx = stratified_split_indices(y, test_size=0.2, seed=random_seed)

        def run_ml_set(X, representation_name):
            X_train, X_test = X[tr_idx], X[te_idx]
            y_train, y_test = y[tr_idx], y[te_idx]
            y_test_text = decode_labels(y_test)

            # Naive Bayes
            if representation_name == "bow":
                nb_model = train_multinomial_nb(X_train, y_train, alpha=1.0)
                pred_nb = predict_multinomial_nb(nb_model, X_test)
            else:
                nb_model = train_gaussian_nb(X_train, y_train, var_smoothing=1e-8)
                pred_nb = predict_gaussian_nb(nb_model, X_test)
            pred_nb_text = decode_labels(pred_nb)
            cm_nb = confusion_matrix(y_test_text, pred_nb_text)
            met_nb = metrics_from_confusion(cm_nb)
            ml_rows.append({
                "variant": v,
                "representation": representation_name,
                "model": "naive_bayes",
                **met_nb,
            })
            # save confusion matrix
            try:
                ml_confusions.append({
                    "variant": v,
                    "representation": representation_name,
                    "model": "naive_bayes",
                    "confusion": cm_nb.tolist(),
                })
            except Exception:
                pass

            # Decision Tree
            dt = SimpleDecisionTree(max_depth=6, min_samples_split=5, feature_subset=60, seed=random_seed)
            dt.fit(X_train, y_train)
            pred_dt = dt.predict(X_test)
            pred_dt_text = decode_labels(pred_dt)
            cm_dt = confusion_matrix(y_test_text, pred_dt_text)
            met_dt = metrics_from_confusion(cm_dt)
            ml_rows.append({
                "variant": v,
                "representation": representation_name,
                "model": "decision_tree",
                **met_dt,
            })
            # save confusion matrix
            try:
                ml_confusions.append({
                    "variant": v,
                    "representation": representation_name,
                    "model": "decision_tree",
                    "confusion": cm_dt.tolist(),
                })
            except Exception:
                pass

        run_ml_set(X_bow, "bow")
        run_ml_set(X_glove, "glove")

    lexical_df = pd.DataFrame(lexical_rows)
    lexical_path = os.path.join(models_dir, "lexical_predictions.csv")
    lexical_df.to_csv(lexical_path, index=False)

    # lexical metrics summary
    lex_metrics = []
    for v in PREP_VARIANTS:
        sub = lexical_df[lexical_df["variant"] == v].copy()
        cm_s = confusion_matrix(sub["true_label"].tolist(), sub["sentiwordnet_style_pred"].tolist())
        cm_b = confusion_matrix(sub["true_label"].tolist(), sub["bing_liu_pred"].tolist())
        ms = metrics_from_confusion(cm_s)
        mb = metrics_from_confusion(cm_b)
        lex_metrics.append({"variant": v, "representation": "lexical", "model": "sentiwordnet_style", **ms})
        lex_metrics.append({"variant": v, "representation": "lexical", "model": "bing_liu_negation", **mb})
        # store lexical confusion matrices for printing
        try:
            lex_confusions.append({"variant": v, "model": "sentiwordnet_style", "confusion": cm_s.tolist()})
        except Exception:
            pass
        try:
            lex_confusions.append({"variant": v, "model": "bing_liu_negation", "confusion": cm_b.tolist()})
        except Exception:
            pass

    ml_df = pd.DataFrame(ml_rows)
    ml_path = os.path.join(models_dir, "ml_results.csv")
    ml_df.to_csv(ml_path, index=False)

    metrics_summary_df = pd.concat([pd.DataFrame(lex_metrics), ml_df], ignore_index=True)
    metrics_summary_path = os.path.join(models_dir, "model_metrics_summary.csv")
    metrics_summary_df.to_csv(metrics_summary_path, index=False)

    summary = {
        "rows_input_sampled": int(len(df)),
        "sample_size_requested": int(sample_size),
        "llm_provider": provider,
        "llm_model": model_name,
        "used_api_labeling": bool(api_key),
        "fleiss_kappa": float(kappa),
        "preprocessing_variants": PREP_VARIANTS,
        "representations": ["bow", "glove"],
        "lexical_models": ["sentiwordnet_style", "bing_liu_negation"],
        "ml_models": ["naive_bayes", "decision_tree"],
        "glove_loaded": glove_loaded,
        "glove_dim": int(glove_dim),
        "paths": {
            "labeled_dataset": labeled_path,
            "fleiss_kappa": os.path.join(labels_dir, "fleiss_kappa.json"),
            "lexical_predictions": lexical_path,
            "ml_results": ml_path,
            "metrics_summary": metrics_summary_path,
        },
    }
    summary_path = _safe_write_json(os.path.join(reports_dir, "task3_summary.json"), summary)

    # Print a concise terminal summary of the most important outputs
    try:
        print("\n=== Task 3 - Terminal Summary ===", flush=True)
        print(f"Input sampled rows: {len(df)} (requested: {sample_size})", flush=True)
        print(f"LLM provider: {provider}", flush=True)
        print(f"Used API labeling: {bool(api_key)}", flush=True)
        print(f"Fleiss Kappa: {kappa:.4f}", flush=True)
        print(f"Labeled dataset: {labeled_path}", flush=True)
        print(f"Fleiss kappa file: {os.path.join(labels_dir, 'fleiss_kappa.json')}", flush=True)
        print(f"Representations dir: {reps_dir}", flush=True)
        print(f"Models dir: {models_dir}", flush=True)

        # Label distribution
        try:
            counts = df['final_label'].value_counts().to_dict()
            print("Label distribution:", counts, flush=True)
        except Exception:
            pass

        # Lexical models performance
        try:
            lex_df = pd.DataFrame(lex_metrics)
            if not lex_df.empty:
                print("\nLexical models performance:", flush=True)
                # show accuracy and macro_f1 if present
                cols = [c for c in ['variant', 'model', 'accuracy', 'macro_f1'] if c in lex_df.columns]
                if cols:
                    print(lex_df[cols].round(3).to_string(index=False), flush=True)
                else:
                    print(lex_df.to_string(index=False), flush=True)
        except Exception:
            pass

        # ML models performance summary
        try:
            ms_df = metrics_summary_df.copy()
            if not ms_df.empty:
                print("\nML models performance (variant / representation / model / accuracy / macro_f1):", flush=True)
                cols = [c for c in ['variant', 'representation', 'model', 'accuracy', 'macro_f1'] if c in ms_df.columns]
                if cols:
                    print(ms_df[cols].round(3).to_string(index=False), flush=True)
                else:
                    print(ms_df.to_string(index=False), flush=True)
        except Exception:
            pass

        # Print confusion matrices for lexical and ML models
        try:
            labels = ["negative", "neutral", "positive"]
            if lex_confusions:
                print("\nLexical model confusion matrices (rows=true, cols=pred):", flush=True)
                for item in lex_confusions:
                    cm = np.array(item['confusion'], dtype=int)
                    print(f"\n{item['variant']} - {item['model']}", flush=True)
                    try:
                        print(pd.DataFrame(cm, index=labels, columns=labels).to_string(), flush=True)
                    except Exception:
                        print(cm, flush=True)

            if ml_confusions:
                print("\nML model confusion matrices (rows=true, cols=pred):", flush=True)
                for item in ml_confusions:
                    cm = np.array(item['confusion'], dtype=int)
                    print(f"\n{item['variant']} - {item.get('representation','?')} - {item['model']}", flush=True)
                    try:
                        print(pd.DataFrame(cm, index=labels, columns=labels).to_string(), flush=True)
                    except Exception:
                        print(cm, flush=True)
        except Exception:
            pass

    except Exception:
        # never fail the pipeline because of printing
        pass

    return {
        "labeled_dataset": labeled_path,
        "fleiss_kappa": os.path.join(labels_dir, "fleiss_kappa.json"),
        "lexical_predictions": lexical_path,
        "ml_results": ml_path,
        "metrics_summary": metrics_summary_path,
        "task3_summary": summary_path,
    }
