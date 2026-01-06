import os
import pandas as pd

def test_dataset_file_exists():
    path = os.path.join('fragrantica_dataset', 'fra_cleaned.csv')
    assert os.path.exists(path)
    df = pd.read_csv(path, sep=';', nrows=5)
    assert not df.empty
