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


def _cache_key(provider: str, model_name: str, prompt: str) -> str:
    raw = f"{provider}::{model_name}::{prompt}"
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


def _extract_error_message(response: requests.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err.get("message"))
    except Exception:
        pass

    text = response.text or ""
    return text.strip()[:500]


def _gemini_call(api_key: str, model_name: str, prompt: str, timeout=25) -> str:
    model_name = (model_name or "").strip()
    if model_name.startswith("models/"):
        model_name = model_name[len("models/"):]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)
    if response.status_code != 200:
        msg = _extract_error_message(response)
        raise RuntimeError(f"Gemini API error (status={response.status_code}): {msg}")

    try:
        data = response.json()
    except Exception as e:
        raise RuntimeError(f"Gemini API returned non-JSON response: {e.__class__.__name__}") from e

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RuntimeError("Gemini API returned no candidates/content.") from e
    return normalize_label(text)


def _groq_call(api_key: str, model_name: str, prompt: str, timeout=25) -> str:
    # Groq is OpenAI-compatible.
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 5,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        msg = _extract_error_message(response)
        raise RuntimeError(f"Groq API error (status={response.status_code}): {msg}")

    try:
        data = response.json()
    except Exception as e:
        raise RuntimeError(f"Groq API returned non-JSON response: {e.__class__.__name__}") from e

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError("Groq API returned no choices/message content.") from e
    return normalize_label(text)


def _llm_call(provider: str, api_key: str, model_name: str, prompt: str, timeout=25) -> str:
    provider = (provider or "").strip().lower()
    if provider == "groq":
        return _groq_call(api_key=api_key, model_name=model_name, prompt=prompt, timeout=timeout)
    if provider == "gemini":
        return _gemini_call(api_key=api_key, model_name=model_name, prompt=prompt, timeout=timeout)
    raise ValueError(f"Unknown LLM provider: {provider}")


def _label_with_prompt(
    provider: str,
    text: str,
    api_key: str,
    model_name: str,
    prompt_template: str,
    cache_data: Dict[str, str],
    retries=2,
    strict_api: bool = False,
) -> str:
    prompt = prompt_template.format(text=text[:2500])
    key = _cache_key(provider, model_name, prompt)
    if key in cache_data:
        return normalize_label(cache_data[key])

    last_error = None
    for _ in range(retries + 1):
        try:
            label = _llm_call(provider=provider, api_key=api_key, model_name=model_name, prompt=prompt)
            cache_data[key] = label
            return label
        except Exception as e:
            last_error = e
            time.sleep(0.7)

    if strict_api:
        raise RuntimeError(
            f"LLM labeling failed after {retries + 1} attempts ({provider}): {last_error.__class__.__name__}: {str(last_error)[:300]}"
        )

    # fallback
    return "neutral"


def label_text_with_three_prompts(
    provider: str,
    text: str,
    api_key: str,
    model_name: str,
    cache_data: Dict[str, str],
    strict_api: bool = False,
) -> Tuple[str, str, str]:
    p1 = _label_with_prompt(provider, text, api_key, model_name, PROMPT_TEMPLATES[0], cache_data, strict_api=strict_api)
    p2 = _label_with_prompt(provider, text, api_key, model_name, PROMPT_TEMPLATES[1], cache_data, strict_api=strict_api)
    p3 = _label_with_prompt(provider, text, api_key, model_name, PROMPT_TEMPLATES[2], cache_data, strict_api=strict_api)
    return p1, p2, p3


def build_labels(
    provider: str,
    texts: List[str],
    api_key: str,
    model_name: str,
    cache_path: str,
    fallback_labels: List[str] = None,
    strict_api: bool = False,
) -> List[Tuple[str, str, str]]:
    cache_data = _read_cache(cache_path)
    out = []

    if strict_api and not api_key:
        raise ValueError(f"strict_api=True but API key is empty for provider '{provider}'.")

    for i, text in enumerate(texts):
        if api_key:
            labels = label_text_with_three_prompts(
                provider=provider,
                text=text,
                api_key=api_key,
                model_name=model_name,
                cache_data=cache_data,
                strict_api=strict_api,
            )
        else:
            fb = "neutral"
            if fallback_labels and i < len(fallback_labels):
                fb = normalize_label(fallback_labels[i])
            labels = (fb, fb, fb)
        out.append(labels)

    _write_cache(cache_path, cache_data)
    return out
