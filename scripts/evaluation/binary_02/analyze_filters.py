import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# === Carica dati ===
df = pd.read_csv('datasets/binary_02/X_test_predicted.csv')

# Teniamo solo i trade con segnale
df = df[df['prediction'] == 1].copy()
print(f" Trade con segnale: {len(df)}\n")

# === Definizione filtri ===
filters = {
    'ADX < 35': df['adx'] < 35,
    'RSI > 35': df['rsi_14'] > 35,
    'Z-Score > 0': df['zscore_ma50_filtered'] > 0,
    'Accel > 0': df['acceleration_5_norm'] > 0,
    'Wick Ratio < 10': df['wick_body_ratio'] < 10,
    'Volatility > 30° percentile': df['volatility_10'] > df['volatility_10'].quantile(0.3)
}

# === Applica filtri singolarmente ===
results = []
for name, cond in filters.items():
    surviving = df[cond]
    results.append((name, len(surviving), len(surviving) / len(df) * 100))

# === Stampa risultati ===
print(" Trade sopravvissuti per filtro:")
for name, count, perc in results:
    print(f"{name:<35} | {count:>2} trade | {perc:5.1f}%")

# === Combinazione di tutti i filtri ===
combined = df.copy()
for cond in filters.values():
    combined = combined[cond]

print(f"\n Trade che passano TUTTI i filtri: {len(combined)} / {len(df)} ({(len(combined)/len(df))*100:.1f}%)")
