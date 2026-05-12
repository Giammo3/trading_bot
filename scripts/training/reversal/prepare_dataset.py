import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from sklearn.model_selection import train_test_split


#parametri
file_path = 'datasets/forex/forex_labeled_balanced.csv'  # usa sempre quello aggiornato
target_col = 'reversal'

df = pd.read_csv(file_path)

# === RIMUOVIAMO COLONNE INUTILI ===
columns_to_drop = [
    'timestamp', 'open', 'high', 'low', 
    'valid_trade',
    'future_close', 'future_return', 'future_return_pct',
    'target_binary_01', 'target_binary_02', 'target_short',
    'target_3class', 'target_5class',
    'trend_continuation', 'volatility_breakout'  # altri target che parlano del futuro
    #'reversal'  # ATTENZIONE! Lo togliamo da X ma lo teniamo come y
]
X = df.drop(columns=columns_to_drop)
#target
y = df[[target_col]]

# === ORDINA PER TEMPO (sempre sicurezza)
df = df.sort_values('timestamp')

# === DIVISIONE train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# === SALVIAMO ===
X_train.to_csv('datasets/reversal/X_train.csv', index=False)
X_test.to_csv('datasets/reversal/X_test.csv', index=False)
y_train.to_csv('datasets/reversal/y_train.csv', index=False)
y_test.to_csv('datasets/reversal/y_test.csv', index=False)

print(" Time-Based Split per Reversal completato!")
print(f"Train set: {len(X_train)} righe")
print(f"Test set:  {len(X_test)} righe")

# Distribuzione target
print("\n Distribuzione target nel test set:")
print(y_test[target_col].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
