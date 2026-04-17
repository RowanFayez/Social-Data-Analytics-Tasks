import os
import pandas as pd


def ensure_data_dir(path='data'):
    os.makedirs(path, exist_ok=True)
    return path


def save_dataframe(df, filename, path='data'):
    os.makedirs(path, exist_ok=True)
    full = os.path.join(path, filename)
    # if df is None or empty, create an empty csv
    try:
        df.to_csv(full, index=False)
    except Exception:
        # fallback: write headers only
        with open(full, 'w', encoding='utf-8') as f:
            f.write('')
    return full


def save_json(obj, filename, path='data'):
    os.makedirs(path, exist_ok=True)
    full = os.path.join(path, filename)
    import json
    with open(full, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return full
