import os
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate
import torch
from torch.utils.data import Dataset as TorchDatasetBase
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from dataset_processing import load_and_prepare

MODEL_NAME = os.environ.get('MODEL_NAME', 'distilbert-base-uncased')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'models/distilbert-gender')
SAMPLE_FRAC = float(os.environ.get('SAMPLE_FRAC', '0.05'))


def compute_metrics(eval_pred):
    metric_acc = evaluate.load('accuracy')
    metric_f1 = evaluate.load('f1')
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(references=labels, predictions=preds, average='weighted')
    return {**acc, 'f1_weighted': f1['f1']}


def main(csv_path: str = 'fragrantica_dataset/fra_cleaned.csv'):
    df, label_map = load_and_prepare(csv_path, sample_frac=SAMPLE_FRAC)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize
    train_enc = tokenizer(train_df['text'].tolist(), truncation=True, padding='max_length', max_length=128)
    test_enc = tokenizer(test_df['text'].tolist(), truncation=True, padding='max_length', max_length=128)

    class TorchDataset(TorchDatasetBase):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(int(self.labels[idx]))
            return item

    train_ds = TorchDataset(train_enc, train_df['label'].tolist())
    test_ds = TorchDataset(test_enc, test_df['label'].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(set(df['label'])))

    # Use a minimal set of TrainingArguments to maximize compatibility
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=int(os.environ.get('EPOCHS', '1')),
        logging_steps=50,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(OUTPUT_DIR)


if __name__ == '__main__':
    main()
