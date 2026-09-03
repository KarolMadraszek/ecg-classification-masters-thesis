from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_metadata(train_df, val_df, test_df, continuous_cols=['age']):
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    for col in continuous_cols:
        col_median = train_out[col].median()
        train_out[col] = train_out[col].fillna(col_median)
        val_out[col] = val_out[col].fillna(col_median)
        test_out[col] = test_out[col].fillna(col_median)

    scaler = StandardScaler()
    train_out[continuous_cols] = scaler.fit_transform(train_out[continuous_cols])
    val_out[continuous_cols] = scaler.transform(val_out[continuous_cols])
    test_out[continuous_cols] = scaler.transform(test_out[continuous_cols])

    return train_out, val_out, test_out

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / 'data' / 'processed'

    df_train = pd.read_csv(data_dir / 'train_meta.csv', index_col='ecg_id', low_memory=False)
    df_val = pd.read_csv(data_dir / 'val_meta.csv', index_col='ecg_id', low_memory=False)
    df_test = pd.read_csv(data_dir / 'test_meta.csv', index_col='ecg_id', low_memory=False)

    train_clean, val_clean, test_clean = scale_metadata(df_train, df_val, df_test)
    print(f"Średnia wieku (zbiór treningowy): {train_clean['age'].mean():.4f}, Odchylenie: {train_clean['age'].std():.4f}")