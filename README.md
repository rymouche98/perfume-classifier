# DistilBERT fine-tune on Fragrantica (Gender classification)

This project fine-tunes `distilbert-base-uncased` to predict the `Gender` of a perfume (men/women/unisex) using the `fragrantica_dataset/fra_cleaned.csv` dataset.

Quickstart

1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train

For a quick CPU-friendly run on a small subset (fast):

```bash
SAMPLE_FRAC=0.05 EPOCHS=1 python train.py
```

For a full run (GPU recommended):

```bash
python train.py
```

3. Run API (after saving model to `models/distilbert-gender`):

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

4. CI: A GitHub Actions workflow is provided in `.github/workflows/ci.yml` to run tests.

Notes
- The dataset is semicolon-delimited. The project uses `Top`, `Middle`, `Base`, and `Perfume` fields concatenated as input text.
- Adjust `train.py` hyperparameters for your environment.
