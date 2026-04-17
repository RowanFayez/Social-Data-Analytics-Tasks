import json
import math
import os
import datetime
from collections import Counter, defaultdict

import pandas as pd

from preprocessing.text_utils import preprocess_text


def _safe_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _safe_csv_write(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        alt_path = f"{base}_{ts}{ext}"
        df.to_csv(alt_path, index=False)
        return alt_path


def _safe_json_write(obj, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        alt_path = f"{base}_{ts}{ext}"
        with open(alt_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return alt_path


def _build_unigram_freq(doc_tokens, top_k):
    freq = Counter()
    for tokens in doc_tokens:
        freq.update(tokens)
    rows = [{"token": token, "frequency": count} for token, count in freq.most_common(top_k)]
    return pd.DataFrame(rows)


def _build_bigram_freq(doc_tokens, top_k):
    freq = Counter()
    for tokens in doc_tokens:
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]} {tokens[i+1]}"
            freq[bg] += 1
    rows = [{"bigram": token, "frequency": count} for token, count in freq.most_common(top_k)]
    return pd.DataFrame(rows)


def _build_tfidf_features(doc_tokens, top_k):
    n_docs = len(doc_tokens)
    if n_docs == 0:
        return pd.DataFrame(), pd.DataFrame()

    doc_freq = Counter()
    token_tf_by_doc = []

    for tokens in doc_tokens:
        counts = Counter(tokens)
        token_tf_by_doc.append(counts)
        for token in counts.keys():
            doc_freq[token] += 1

    idf = {}
    for token, df in doc_freq.items():
        idf[token] = math.log((1 + n_docs) / (1 + df)) + 1.0

    token_scores = defaultdict(list)
    top_terms_rows = []
    for idx, tf_counts in enumerate(token_tf_by_doc):
        total_terms = sum(tf_counts.values())
        if total_terms == 0:
            top_terms_rows.append({"doc_index": idx, "top_terms": ""})
            continue

        tfidf_local = {}
        for token, cnt in tf_counts.items():
            tf = cnt / total_terms
            score = tf * idf[token]
            tfidf_local[token] = score
            token_scores[token].append(score)

        top_terms = sorted(tfidf_local.items(), key=lambda x: x[1], reverse=True)[:10]
        top_terms_rows.append(
            {
                "doc_index": idx,
                "top_terms": " | ".join([f"{t}:{s:.4f}" for t, s in top_terms]),
            }
        )

    features_rows = []
    for token, scores in token_scores.items():
        features_rows.append(
            {
                "token": token,
                "doc_freq": doc_freq[token],
                "idf": idf[token],
                "mean_tfidf": sum(scores) / len(scores),
                "max_tfidf": max(scores),
            }
        )

    features_df = pd.DataFrame(features_rows).sort_values(
        by=["mean_tfidf", "doc_freq"], ascending=[False, False]
    )
    if top_k and top_k > 0:
        features_df = features_df.head(top_k).reset_index(drop=True)

    top_terms_df = pd.DataFrame(top_terms_rows)
    return features_df, top_terms_df


def run_task2_pipeline(input_csv, output_dir, top_k=200):
    os.makedirs(output_dir, exist_ok=True)
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    # Build raw text from Task 1 fields.
    df["title"] = df["title"].map(_safe_text)
    df["selftext"] = df["selftext"].map(_safe_text)
    df["original_title"] = df["title"]
    df["original_selftext"] = df["selftext"]
    df["raw_text"] = (df["original_title"] + " " + df["original_selftext"]).str.strip()

    prep_rows = []
    title_prep_rows = []
    selftext_prep_rows = []
    for _, row in df.iterrows():
        prep = preprocess_text(row["raw_text"])
        title_prep = preprocess_text(row["original_title"])
        selftext_prep = preprocess_text(row["original_selftext"])
        prep_rows.append(prep)
        title_prep_rows.append(title_prep)
        selftext_prep_rows.append(selftext_prep)

    prep_df = pd.DataFrame(prep_rows)
    title_prep_df = pd.DataFrame(title_prep_rows).add_prefix("title_")
    selftext_prep_df = pd.DataFrame(selftext_prep_rows).add_prefix("selftext_")
    merged = pd.concat([df.reset_index(drop=True), prep_df], axis=1)
    merged = pd.concat([merged, title_prep_df, selftext_prep_df], axis=1)
    merged["raw_char_len"] = merged["raw_text"].str.len()
    merged["clean_char_len"] = merged["clean_text"].str.len()
    merged["token_count"] = merged["tokens"].map(len)
    merged["token_count_no_stop"] = merged["tokens_no_stop"].map(len)
    merged["stemmed_token_count"] = merged["stemmed_tokens"].map(len)
    merged["is_text_empty_after_cleaning"] = merged["stemmed_token_count"] == 0

    # Save list columns as strings for CSV portability.
    merged_out = merged.copy()
    for col in [
        "tokens",
        "tokens_no_stop",
        "stemmed_tokens",
        "title_tokens",
        "title_tokens_no_stop",
        "title_stemmed_tokens",
        "selftext_tokens",
        "selftext_tokens_no_stop",
        "selftext_stemmed_tokens",
    ]:
        merged_out[col] = merged_out[col].map(lambda x: " ".join(x))

    # Make visible title/selftext columns explicitly processed.
    merged_out["processed_title"] = merged_out["title_final_text"]
    merged_out["processed_selftext"] = merged_out["selftext_final_text"]
    merged_out["processed_text"] = merged_out["final_text"]
    merged_out["title"] = merged_out["processed_title"]
    merged_out["selftext"] = merged_out["processed_selftext"]

    preferred_first_cols = [
        "post_id",
        "term",
        "subreddit",
        "title",
        "selftext",
        "processed_text",
        "original_title",
        "original_selftext",
    ]
    existing_preferred = [c for c in preferred_first_cols if c in merged_out.columns]
    remaining_cols = [c for c in merged_out.columns if c not in existing_preferred]
    merged_out = merged_out[existing_preferred + remaining_cols]

    preprocessed_path = os.path.join(processed_dir, "preprocessed_posts.csv")
    preprocessed_path = _safe_csv_write(merged_out, preprocessed_path)

    doc_tokens = merged["stemmed_tokens"].tolist()
    unigram_df = _build_unigram_freq(doc_tokens, top_k=top_k)
    bigram_df = _build_bigram_freq(doc_tokens, top_k=top_k)
    tfidf_df, top_terms_df = _build_tfidf_features(doc_tokens, top_k=top_k)

    unigram_path = os.path.join(processed_dir, "unigram_freq.csv")
    bigram_path = os.path.join(processed_dir, "bigram_freq.csv")
    tfidf_path = os.path.join(processed_dir, "tfidf_features.csv")
    top_terms_path = os.path.join(processed_dir, "top_terms_per_post.csv")

    unigram_path = _safe_csv_write(unigram_df, unigram_path)
    bigram_path = _safe_csv_write(bigram_df, bigram_path)
    tfidf_path = _safe_csv_write(tfidf_df, tfidf_path)

    if "post_id" in merged.columns and not top_terms_df.empty:
        top_terms_df = top_terms_df.copy()
        top_terms_df["post_id"] = merged["post_id"].values
        top_terms_df = top_terms_df[["post_id", "doc_index", "top_terms"]]
    top_terms_path = _safe_csv_write(top_terms_df, top_terms_path)

    summary = {
        "input_csv": input_csv,
        "rows_in": int(len(df)),
        "rows_out": int(len(merged_out)),
        "empty_posts_after_cleaning": int(merged["is_text_empty_after_cleaning"].sum()),
        "avg_token_count": float(merged["token_count"].mean()) if len(merged) > 0 else 0.0,
        "avg_token_count_no_stop": float(merged["token_count_no_stop"].mean()) if len(merged) > 0 else 0.0,
        "unique_unigrams": int(len(unigram_df)),
        "unique_bigrams": int(len(bigram_df)),
        "unique_tfidf_features": int(len(tfidf_df)),
        "output_dir": output_dir,
        "processed_dir": processed_dir,
    }
    summary_path = os.path.join(processed_dir, "preprocessing_summary.json")
    summary_path = _safe_json_write(summary, summary_path)

    return {
        "preprocessed_posts": preprocessed_path,
        "unigram_freq": unigram_path,
        "bigram_freq": bigram_path,
        "tfidf_features": tfidf_path,
        "top_terms_per_post": top_terms_path,
        "summary_json": summary_path,
    }
