# Task 4 — Evaluation, Optimization, Error Analysis, Deployment

This task follows the Task 4 PDF requirements:
- Evaluate/benchmark the **18 baseline models** produced in Task 3
- Optimize selected models (incl. PCA-like dimensionality reduction via `TruncatedSVD`)
- Produce an **error analysis conclusion**
- Provide a deployable API (`/predict`)

## Install
From the repo root (or your active venv):

```bash
pip install -r task4/requirements.txt
```

## Run
### 1) Run the full Task 4 pipeline
Auto-detects the latest Task 3 run folder (prefers `run_*_groq`) and uses its `labeled_dataset.csv`.

```bash
python task4/main.py
```

Optional: explicitly point to a Task 3 run output:

```bash
python task4/main.py --task3_run_dir task3/final_data/run_20260417T231050Z_groq
```

Outputs are written under `task4/final_data/run_<run_id>/`.

### 2) Start the API
After training, start FastAPI:

```bash
cd task4
uvicorn api:app --reload --port 8000
```

Request example:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"I love this product\"}"
```

If you want to force a specific model artifact, set:
- `TASK4_MODEL_PATH=task4/final_data/run_<run_id>/optimization/best_model.joblib`

### 3) Streamlit UI (single text input)
Runs a simple UI that loads the latest `best_model.joblib` and predicts sentiment + confidence.

```bash
cd task4
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

### 4) Export per-row predictions (for manual error analysis)
Writes a CSV containing the best-model prediction for every row in the labeled dataset (and a separate CSV of only the misclassified rows).

```bash
python task4/export_predictions_csv.py --task4_run_dir task4/final_data/run_<run_id>
```
