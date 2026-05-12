import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# Carica i dati
X_fn = pd.read_csv('datasets/reversal/false_negatives.csv')
X_fp = pd.read_csv('datasets/reversal/false_positives.csv')

print("\n Statistiche Falsi Negativi (reversal veri mancati):")
print(X_fn.describe())

print("\n Statistiche Falsi Positivi (falsi reversal previsti):")
print(X_fp.describe())
