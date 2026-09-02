import pandas as pd
from pathlib import Path

raw_df = pd.read_csv('../data/raw/ptbxl_database.csv', index_col='ecg_id')

splits = ['train_meta.csv', 'val_meta.csv', 'test_meta.csv']
processed_dir = Path('../data/processed')

for split_file in splits:
    file_path = processed_dir / split_file
    if not file_path.exists():
        continue

    df_split = pd.read_csv(file_path, index_col='ecg_id', low_memory=False)
    df_split['age'] = raw_df.loc[df_split.index, 'age']

    # Zastosowanie zasady HIPAA: 300 -> 90
    df_split.loc[df_split['age'] == 300, 'age'] = 90

    df_split.to_csv(file_path)
    print(f"Zaktualizowano wiek w: {split_file}")