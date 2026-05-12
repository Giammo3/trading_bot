import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/forex/forex_labeled_balanced.csv')

#ordina per timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values(by='timestamp', inplace=True)

#rimuovo colonne non utili
drop_cols = [
    'timestamp', 'future_close', 'future_return', 'future_return_pct',
    'target_binary_01', 'target_3class', 'target_5class',
    'reversal', 'trend_continuation', 'volatility_breakout',
    'target_short'
]
df.drop(columns=drop_cols, inplace=True)


#fetures (X) e target(y)
target = 'target_binary_02'
X = df.drop(columns=[target])
y = df[target]

#=== Split TRAIN/TEST  80% / 20% ====

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

#stampe per info
print(" Time-Based Split completato!")
print(f"Train set: {X_train.shape[0]} righe")
print(f"Test set:  {X_test.shape[0]} righe\n")

print(" Distribuzione target nel test set:")
print(y_test.value_counts(normalize=True).mul(100).round(2).astype(str) + " %")

X_train.to_csv('datasets/binary_02/X_train.csv', index=False)
X_test.to_csv('datasets/binary_02/X_test.csv', index=False)
y_train.to_csv('datasets/binary_02/y_train.csv', index=False)
y_test.to_csv('datasets/binary_02/y_test.csv', index=False)
