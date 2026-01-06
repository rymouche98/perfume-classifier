import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_DIR = os.environ.get('MODEL_DIR', 'models/distilbert-gender')
MODEL_NAME = os.environ.get('MODEL_NAME', 'distilbert-base-uncased')

app = FastAPI()

class RequestItem(BaseModel):
    text: str


@app.on_event("startup")
async def load_model():
    global tokenizer, model
    # Try loading tokenizer/model from saved fine-tuned folder; fall back to base model if missing
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()


@app.post('/predict')
async def predict(item: RequestItem):
    inputs = tokenizer(item.text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.cpu().numpy()[0]
    probs = (np.exp(logits) / np.exp(logits).sum()).tolist()
    label_idx = int(np.argmax(logits))
    label_map = {0: 'men', 1: 'women', 2: 'unisex'}
    return {'label': label_map.get(label_idx, str(label_idx)), 'scores': probs}
