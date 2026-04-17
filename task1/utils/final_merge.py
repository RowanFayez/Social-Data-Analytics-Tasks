import os
import datetime
import pandas as pd


def _safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _align_columns(existing_df, new_df):
    existing_cols = list(existing_df.columns)
    new_cols = [c for c in new_df.columns if c not in existing_cols]
    all_cols = existing_cols + new_cols
    if not all_cols:
        return existing_df, new_df
    return existing_df.reindex(columns=all_cols), new_df.reindex(columns=all_cols)


def merge_datasets_into_final_data(datasets, run_id, final_dir="final_data"):
    """Append run datasets into cumulative final_data CSVs with schema-safe alignment."""
    os.makedirs(final_dir, exist_ok=True)
    merged_paths = {}
    summary = {}
    ingested_at_utc = datetime.datetime.now(datetime.UTC).isoformat()

    for name, current_df in datasets.items():
        if current_df is None:
            current_df = pd.DataFrame()
        if not isinstance(current_df, pd.DataFrame):
            current_df = pd.DataFrame(current_df)

        current_df = current_df.copy()
        if "source_run_id" not in current_df.columns:
            current_df["source_run_id"] = run_id
        if "ingested_at_utc" not in current_df.columns:
            current_df["ingested_at_utc"] = ingested_at_utc

        final_path = os.path.join(final_dir, f"{name}.csv")
        existing_df = _safe_read_csv(final_path)
        existing_count = len(existing_df)
        incoming_count = len(current_df)

        existing_df, current_df = _align_columns(existing_df, current_df)
        merged_df = pd.concat([existing_df, current_df], ignore_index=True, sort=False)
        merged_df.to_csv(final_path, index=False)

        merged_paths[name] = final_path
        summary[name] = {
            "existing_rows": existing_count,
            "incoming_rows": incoming_count,
            "final_rows": len(merged_df),
            "columns_count": len(merged_df.columns),
        }

    return merged_paths, summary
