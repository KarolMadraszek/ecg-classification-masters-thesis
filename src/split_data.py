import pandas as pd
from pathlib import Path

def split_and_save_metadata(cleaned_csv_path: str):
    base_dir = Path('../data/processed')
    df = pd.read_csv(cleaned_csv_path, index_col='ecg_id')

    # Podział danych zgodnie z zaleceniami autorów PTB-XL
    train_df = df[df['strat_fold'] <= 8].copy()
    val_df = df[df['strat_fold'] == 9].copy()       # do strojenia hiperparametrów i Early Stoppingu
    test_df = df[df['strat_fold'] == 10].copy()     # do końcowych testów

    train_df.to_csv(base_dir / 'train_meta.csv')
    val_df.to_csv(base_dir / 'val_meta.csv')
    test_df.to_csv(base_dir / 'test_meta.csv')

    print("Podsumowanie podziału danych")
    print(f"Zbiór treningowy (folds 1-8): {len(train_df)} badań ({len(train_df) / len(df):.1%})")
    print(f"Zbiór walidacyjny (fold 9):   {len(val_df)} badań ({len(val_df) / len(df):.1%})")
    print(f"Zbiór testowy (fold 10):      {len(test_df)} badań ({len(test_df) / len(df):.1%})")
    print("\nPliki zapisane w data/processed/.")

if __name__ == '__main__':
    split_and_save_metadata('../data/processed/ptbxl_metadata_cleaned.csv')
