# Offline Kaggle Setup

This project currently runs a BM25 baseline, which does not require downloading any model at inference time.

If you add embedding/reranker models, pre-download them before running in Kaggle offline mode.

## 1) Prefetch models locally (with internet)

```powershell
.\.venv\Scripts\python.exe scripts\prefetch_offline_models.py
```

By default this downloads:
- `intfloat/multilingual-e5-small`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

Custom model list example:

```powershell
.\.venv\Scripts\python.exe scripts\prefetch_offline_models.py --model intfloat/multilingual-e5-small --model BAAI/bge-reranker-base
```

## 2) Attach model folder as Kaggle Dataset

Upload `models/hf` as a Kaggle Dataset and attach it to your notebook.

## 3) Force offline mode in notebook

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
```

## 4) Load model from local path only

Example paths after dataset mount:
- `/kaggle/input/<your-model-dataset>/hf/intfloat--multilingual-e5-small`
- `/kaggle/input/<your-model-dataset>/hf/cross-encoder--ms-marco-MiniLM-L-6-v2`

Always pass the local path to transformers/sentence-transformers loaders.

## Submission sanity checks

- `submission.csv` must have columns: `query_id,predicted_citations`
- `predicted_citations` values must be semicolon-separated citation strings, not free text paragraphs.
