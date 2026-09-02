import pandas as pd


def clean_tabular_data(df_path, is_train=True, age_median_train=None):
    df = pd.read_csv(df_path, index_col='ecg_id', low_memory=False)

    # Usunięcie nieprzydatnych kolumn (zbyt dużo braków lub wyciek danych)
    cols_to_drop = ['height', 'weight', 'nurse', 'site', 'device']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Imputacja braków (wiek)
    if is_train:
        age_median = df['age'].median()
    else:
        age_median = age_median_train

    df['age'] = df['age'].fillna(age_median)
    return df, age_median

if __name__ == '__main__':
    df_train_clean, train_median = clean_tabular_data('../data/processed/train_meta.csv', is_train=True)
    df_val_clean, _ = clean_tabular_data('../data/processed/val_meta.csv', is_train=False, age_median_train=train_median)