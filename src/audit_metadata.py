import pandas as pd
import numpy as np
from pathlib import Path

def audit_and_clean_metadata(raw_csv_path: str | Path, output_csv_path: str | Path = None) -> pd.DataFrame:
    path = Path(raw_csv_path)
    print(f"Audyt i czyszczenie metadanych: {path.name}")
    df = pd.read_csv(path, index_col='ecg_id')

    print("\n1. Skanowanie danych wrażliwych (PII)")
    pii_keywords = ['name', 'surname', 'ssn', 'pesel', 'address', 'phone', 'email', 'zip'] # Lista zakazanych słów
    df_columns_lower = df.columns.str.lower()

    found_pii = [col for col in pii_keywords if col in df_columns_lower]
    if not found_pii:
        print("Brak bezpośrednich identyfikatorów (zgodność z RODO/GDPR)")
    else:
        print(f"UWAGA: Znaleziono potencjalnie wrażliwe kolumny: {found_pii}")

    print("\n2. Weryfikacja wieku (HIPAA Safe Harbor)")
    # HIPAA Safe Harbor wymaga agregacji wieku > 89 lat
    if 'age' in df.columns:
        anomalies_mask = (df['age'] > 120) | (df['age'].isna())
        anomalies_count = anomalies_mask.sum()

        if anomalies_count > 0:
            print(f"Znaleziono {anomalies_count} rekordów z brakiem danych lub wiekiem > 120 lat.")
            print(f"Zamiana wartości strażniczych (np. 300) na systemowe brakujące dane (NaN)")
            df.loc[df['age'] > 120, 'age'] = np.nan
            print(f"Wiek po oczyszczeniu - Max: {df['age'].max()} lat, Min: {df['age'].min()} lat")
        else:
            print("Kolumna wieku nie zawiera widocznych anomalii (>120 lat).")

        max_age = df['age'].max()
        if max_age > 89:
            print("""Zidentyfikowano pacjentów powyżej 89. roku życia. 
W rygorystycznych normach medycznych USA wiek ten bywa maskowany (np. jako '90+').""")

    if output_csv_path:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)
        print(f"\nZapisano oczyszczone metadane do: {out_path}")

    return df


if __name__ == "__main__":
    INPUT_FILE = "../data/raw/ptbxl_database.csv"
    OUTPUT_FILE = "../data/processed/ptbxl_metadata_cleaned.csv"
    cleaned_df = audit_and_clean_metadata(
        raw_csv_path=INPUT_FILE,
        output_csv_path=OUTPUT_FILE
    )