import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# Carica i dati
fn_df = pd.read_csv('datasets/reversal/false_negatives.csv')  # Reversal veri mancati
pred_df = pd.read_csv('datasets/reversal/X_test_predicted_filtered.csv')  # Tutti i dati con prediction

# Filtra i True Positive (dove prediction = 1 e target = 1)
tp_df = pred_df[(pred_df['prediction'] == 1) & (pred_df['reversal'] == 1)]

# Feature comuni
feature_cols = list(set(fn_df.columns).intersection(tp_df.columns))
#Calcolo differenze medie
print("\n Differenze medie tra TP (presi) e FN (mancati):")
for col in sorted(feature_cols):
    if pd.api.types.is_numeric_dtype(tp_df[col]):
        tp_mean = tp_df[col].mean()
        fn_mean = fn_df[col].mean()
        delta = abs(tp_mean - fn_mean)
        print(f"{col:<25} | TP: {tp_mean:.4f} | FN: {fn_mean:.4f} | Δ = {delta:.4f}")