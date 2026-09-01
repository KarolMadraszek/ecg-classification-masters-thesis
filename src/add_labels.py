import pandas as pd
import ast

df = pd.read_csv('../data/processed/ptbxl_metadata_cleaned.csv', index_col='ecg_id')
agg_df = pd.read_csv('../data/raw/scp_statements.csv', index_col=0)
diag_agg_df = agg_df[agg_df.diagnostic == 1]
scp_to_superclass = diag_agg_df['diagnostic_class'].dropna().to_dict()

def get_superclasses(scp_string):
    scp_dict = ast.literal_eval(scp_string)
    classes = set()
    for key in scp_dict.keys():
        if key in scp_to_superclass:
            classes.add(str(scp_to_superclass[key]))
    return list(classes)

df['diagnostic_superclass'] = df['scp_codes'].apply(get_superclasses)

target_classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
for c in target_classes:
    df[c] = df['diagnostic_superclass'].apply(lambda x: 1 if c in x else 0)

df.to_csv('../data/processed/ptbxl_metadata_cleaned.csv')
print("Kolumny NORM, MI, STTC, CD, HYP zostały dodane do bazy.")