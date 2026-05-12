import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# Carica il file
df = pd.read_csv('datasets/forex/forex_labeled.csv')

# Parametri
lookahead = 40
threshold = 0.0015  # 0.4%

# Inizializza target
target = []

for i in range(len(df)):
    if df.loc[i, 'reversal'] == 1:
        current_price = df.loc[i, 'close']
        found_strong_move = False

        for j in range(1, lookahead + 1):
            if i + j >= len(df):
                break

            future_price = df.loc[i + j, 'close']
            pct_change = (future_price - current_price) / current_price

            if abs(pct_change) >= threshold:
                found_strong_move = True
                break

        target.append(1 if found_strong_move else 0)
    else:
        target.append(0)

df['target_binary_03'] = target

# Salva nuovo file
df.to_csv('datasets/binary_03/forex_labeled_with_binary03.csv', index=False)
print(" target_binary_03 aggiunto e salvato.")


df = pd.read_csv("datasets/binary_03/forex_labeled_with_binary03.csv")
print(df['target_binary_03'].value_counts(normalize=True))
