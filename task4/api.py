import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel


def _basic_clean_text(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _find_latest_model(repo_root: Path) -> Optional[Path]:
    task4_final = repo_root / "task4" / "final_data"
    if not task4_final.exists():
        return None
    run_dirs = sorted([p for p in task4_final.glob("run_*") if p.is_dir()])
    if not run_dirs:
        return None

    # Prefer latest run by name
    run_dir = run_dirs[-1]
    cand = run_dir / "optimization" / "best_model.joblib"
    if cand.exists():
        return cand

    # fallback: search any run dir
    for d in reversed(run_dirs):
        cand = d / "optimization" / "best_model.joblib"
        if cand.exists():
            return cand
    return None


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parent.parent


MODEL_PATH_ENV = os.getenv("TASK4_MODEL_PATH", "").strip()
REPO_ROOT = _repo_root_from_this_file()

if MODEL_PATH_ENV:
    MODEL_PATH = Path(MODEL_PATH_ENV)
    if not MODEL_PATH.is_absolute():
        MODEL_PATH = (REPO_ROOT / MODEL_PATH).resolve()
else:
    MODEL_PATH = _find_latest_model(REPO_ROOT) or Path()


def _load_model(path: Path):
    if not path or not path.exists():
        raise FileNotFoundError(
            "Task 4 model not found. Run task4/main.py first, or set TASK4_MODEL_PATH to a best_model.joblib path."
        )
    return load(path)


try:
    MODEL = _load_model(MODEL_PATH)
except Exception:
    MODEL = None

app = FastAPI(title="Task 4 Sentiment API")


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    global MODEL
    if MODEL is None:
        # Lazy-load to allow container/server startup even if model is produced later.
        try:
            MODEL = _load_model(MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    cleaned = _basic_clean_text(req.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Empty text after cleaning.")

    pred = MODEL.predict([cleaned])[0]
    pred = str(pred)

    confidence = None
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba([cleaned])
        confidence = float(np.max(proba))
    elif hasattr(MODEL, "decision_function"):
        scores = MODEL.decision_function([cleaned])
        scores = np.array(scores)
        if scores.ndim == 1:
            p = 1 / (1 + np.exp(-scores))
            confidence = float(max(p[0], 1 - p[0]))
        else:
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            p = exp / exp.sum(axis=1, keepdims=True)
            confidence = float(np.max(p[0]))

    resp: Dict[str, Any] = {
        "sentiment": pred,
        "confidence": confidence,
    }
    return resp
