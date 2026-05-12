import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from sklearn.utils import resample

# === PARAMETRI ===
file_path = 'datasets/forex/forex_features_optimized.csv'
df = pd.read_csv(file_path)

# === CONFIGURAZIONE ===
n = 5                   # step futuri (25 min)
soglia_01 = 0.0003       # 0.2% (soglia minima realistica)
soglia_02 = 0.0005       # 0.4% (soglia per target ambiziosi)
volatility_threshold = df['volatility_10'].quantile(0.25)  # Soglia per filtri

# === CALCOLO RENDIMENTO FUTURO ===
df['future_close'] = df['close'].shift(-n)
df['future_return'] = (df['future_close'] - df['close']) / df['close']
df['future_return_pct'] = df['future_return'] * 100

# === TARGET BINARI ===
df['target_binary_01'] = (df['future_return'] > soglia_01).astype(int)  # 0.2%
df['target_binary_02'] = (df['future_return'] > soglia_02).astype(int)  # 0.4%
df['target_short'] = (df['future_return'] < -soglia_02).astype(int)     # Short 0.4%

# === TARGET MULTICLASSE ===
# 3 Classi (-1: Sell, 0: Hold, 1: Buy)
df['target_3class'] = 0
df.loc[df['future_return'] > soglia_02, 'target_3class'] = 1
df.loc[df['future_return'] < -soglia_02, 'target_3class'] = -1
# Aggiungi "Hold attivo" se volatilità è significativa
df.loc[(df['future_return'].abs() <= soglia_02) & (df['volatility_10'] > volatility_threshold), 'target_3class'] = 2

# 5 Classi (-2: Strong Sell, -1: Sell, 0: Neutral, 1: Buy, 2: Strong Buy)
def classify_multi_soglia(r):
    if r > 0.003:    # 0.3%
        return 2
    elif r > 0.001:  # 0.1%
        return 1
    elif r < -0.003:
        return -2
    elif r < -0.001:
        return -1
    else:
        return 0
df['target_5class'] = df['future_return'].apply(classify_multi_soglia)

# === TARGET SPECIALI ===
# Reversal con filtro volatilità
df['reversal'] = (
    ((df['close'].shift(-1) < df['close']) & 
     (df['close'].shift(-n) > df['close']) & 
     (df['volatility_10'] > volatility_threshold)) |
    ((df['close'].shift(-1) > df['close']) & 
     (df['close'].shift(-n) < df['close']) & 
     (df['volatility_10'] > volatility_threshold))
).astype(int)

# Trend Continuation (solo in trend ADX forte)
df['trend_continuation'] = (
    ((df['close'].shift(-n) > df['close']) & 
    (df['adx_direction'] == 1)) | 
    ((df['close'].shift(-n) < df['close']) & 
    (df['adx_direction'] == -1))
).astype(int)

# Volatility Breakout (1.5 deviazioni)
df['volatility_breakout'] = (
    (df['close'].shift(-n) > (df['close'] + 1.5 * df['volatility_10'])) | 
    (df['close'].shift(-n) < (df['close'] - 1.5 * df['volatility_10']))
).astype(int)

# === BILANCIAMENTO DATI (per IA) ===
def balance_dataset(df, target_col):
    majority = df[df[target_col] == 0]
    minority = df[df[target_col] == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority))
    return pd.concat([majority, minority_upsampled])

# === PULIZIA E SALVATAGGIO ===
df.dropna(inplace=True)

# Esempio: bilancia target_binary_02

df.to_csv('datasets/forex/forex_labeled.csv', index=False)

df_balanced = balance_dataset(df, 'target_binary_02')
df_balanced.to_csv('datasets/forex/forex_labeled_balanced.csv', index=False)

# === LOG DI CONTROLLO ===
print(" Tutti i target creati e salvati in 'forex_labeled.csv'")
print(f"Totale righe: {len(df)}\n")

target_cols = [
    'target_binary_01', 'target_binary_02', 'target_3class', 
    'target_5class', 'reversal', 'trend_continuation', 'volatility_breakout'
]

for col in target_cols:
    print(f" Distribuzione {col}:")
    print(df[col].value_counts(normalize=True).mul(100).round(1).astype(str) + "%\n")

# Verifica coerenza tra future_return e target
print("\n Controllo soglie target_binary_02:")
print(f"- Soglia attuale: {soglia_02*100}%")
print(f"- Movimenti > soglia: {df['future_return'].gt(soglia_02).mean()*100:.1f}%")
print(f"- Esempio incoerente (future_return < soglia ma target=1):")
# Sostituisci la riga problematica con:
incoerenti = df[(df['target_binary_02'] == 1) & (df['future_return'] < soglia_02)]
print("- Righe incoerenti trovate:", len(incoerenti))
if len(incoerenti) > 0:
    print(incoerenti.sample(min(2, len(incoerenti))))
else:
    print("- Nessuna incoerenza (OK)")

print("\n Esempio di calcolo target_binary_02:")
sample = df.sample(5)[['close', 'future_close', 'future_return', 'target_binary_02']]
print(sample.round(6))