import hashlib
import json
import os
import time
from typing import Dict, List, Tuple

import requests

from sentiment.agreement import normalize_label


PROMPT_TEMPLATES = [
    "Classify sentiment of this political text as one label only: positive, negative, or neutral. Return only one word.\nText: {text}",
    "You are labeling public opinion sentiment in political discourse. Choose exactly one label: positive / negative / neutral. Return one word only.\nText: {text}",
    "Sentiment task: decide whether tone toward topic is positive, negative, or neutral. Output only one of: positive, negative, neutral.\nText: {text}",
]


def _cache_key(model_name: str, prompt: str) -> str:
    raw = f"{model_name}::{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_cache(cache_path: str) -> Dict[str, str]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_cache(cache_path: str, cache_data: Dict[str, str]):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def _gemini_call(api_key: str, model_name: str, prompt: str, timeout=25) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        text = "neutral"
    return normalize_label(text)


def _label_with_prompt(
    text: str,
    api_key: str,
    model_name: str,
    prompt_template: str,
    cache_data: Dict[str, str],
    retries=2,
) -> str:
    prompt = prompt_template.format(text=text[:2500])
    key = _cache_key(model_name, prompt)
    if key in cache_data:
        return normalize_label(cache_data[key])

    last_error = None
    for _ in range(retries + 1):
        try:
            label = _gemini_call(api_key=api_key, model_name=model_name, prompt=prompt)
            cache_data[key] = label
            return label
        except Exception as e:
            last_error = e
            time.sleep(0.7)
    # fallback
    cache_data[key] = "neutral"
    return "neutral"


def label_text_with_three_prompts(
    text: str,
    api_key: str,
    model_name: str,
    cache_data: Dict[str, str],
) -> Tuple[str, str, str]:
    p1 = _label_with_prompt(text, api_key, model_name, PROMPT_TEMPLATES[0], cache_data)
    p2 = _label_with_prompt(text, api_key, model_name, PROMPT_TEMPLATES[1], cache_data)
    p3 = _label_with_prompt(text, api_key, model_name, PROMPT_TEMPLATES[2], cache_data)
    return p1, p2, p3


def build_labels(
    texts: List[str],
    api_key: str,
    model_name: str,
    cache_path: str,
    fallback_labels: List[str] = None,
) -> List[Tuple[str, str, str]]:
    cache_data = _read_cache(cache_path)
    out = []

    for i, text in enumerate(texts):
        if api_key:
            labels = label_text_with_three_prompts(
                text=text,
                api_key=api_key,
                model_name=model_name,
                cache_data=cache_data,
            )
        else:
            fb = "neutral"
            if fallback_labels and i < len(fallback_labels):
                fb = normalize_label(fallback_labels[i])
            labels = (fb, fb, fb)
        out.append(labels)

    _write_cache(cache_path, cache_data)
    return out
