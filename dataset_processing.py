import pandas as pd

LABEL_MAP = {"men": 0, "women": 1, "unisex": 2}


def load_and_prepare(path: str, sample_frac: float = 1.0):
    try:
        df = pd.read_csv(path, sep=';', low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=';', low_memory=False, encoding='cp1252')
    df = df.dropna(subset=['Gender'])
    df['text'] = (
        df['Perfume'].fillna('') + ' ' +
        df['Top'].fillna('') + ' ' +
        df['Middle'].fillna('') + ' ' +
        df['Base'].fillna('')
    )
    df['label'] = df['Gender'].str.lower().map(LABEL_MAP)
    df = df[df['label'].notnull()].reset_index(drop=True)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    return df[['text', 'label']].copy(), LABEL_MAP
