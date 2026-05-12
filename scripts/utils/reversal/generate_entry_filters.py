import pandas as pd

# === Parametri ===
INPUT_TRADE_FILE = 'datasets/reversal/X_test_traded.csv'
INPUT_BASE_FILE = 'datasets/reversal/X_test_filtered_with_flat.csv'
OUTPUT_FILTER_FILE = 'scripts/utils/reversal/entry_filters_auto.py'

# === Carica i dati ===
df_traded = pd.read_csv(INPUT_TRADE_FILE)
df_base = pd.read_csv(INPUT_BASE_FILE)

# === Assicura che ci sia la colonna 'reversal' ===
if 'reversal' not in df_traded.columns and 'reversal' in df_base.columns:
    df_traded['reversal'] = df_base['reversal']

# === Rimuovi NO TRADE ===
df = df_traded[df_traded['prediction'] != 'NO TRADE'].copy()
df = df[df['prediction'] != 'NO TRADE'].copy()
df['prediction'] = df['prediction'].astype(int)

# === Seleziona TP e FP ===
df_tp = df[(df['prediction'] == 1) & (df['reversal'] == 1)]
df_fp = df[(df['prediction'] == 1) & (df['reversal'] == 0)]

print(f"✅ True Positives: {len(df_tp)}")
print(f"❌ False Positives: {len(df_fp)}")

# === Feature candidate ===
candidate_features = [
    'rsi_14', 'adx', 'zscore_ma50_filtered', 'volatility_10',
    'acceleration_5_norm', 'wick_body_ratio', 'liquidity_proxy'
]

for feat in candidate_features:
    tp_mean = df_tp[feat].mean()
    fp_mean = df_fp[feat].mean()
    std = df[feat].std()
    diff = tp_mean - fp_mean
    print(f"{feat}: diff={diff:.4f}, std={std:.4f}, threshold={0.1 * std:.4f}")

# === Genera codice filtri ===
filters_code = []
entry_list = []

max_filters = 3  # scegli tu il limite
threshold_factor = 0.03  # <--- nuovo valore
filter_count = 0

for feat in candidate_features:
    if feat not in df.columns:
        continue
    tp_mean = df_tp[feat].mean()
    fp_mean = df_fp[feat].mean()
    std = df[feat].std()
    diff = tp_mean - fp_mean

    if abs(diff) > threshold_factor * std:
        if diff > 0:
            filters_code.append(f"def filter_{feat}(row): return row.get('{feat}', 0) > {round(fp_mean, 6)}")
        else:
            filters_code.append(f"def filter_{feat}(row): return row.get('{feat}', 0) < {round(fp_mean, 6)}")
        entry_list.append(f"filter_{feat}")
        filter_count += 1
        if filter_count >= max_filters:
            break

# === Salva su file ===
with open(OUTPUT_FILTER_FILE, 'w') as f:
    for line in filters_code:
        f.write(line + '\n')
    f.write('\nentry_filters_reversal = [\n')
    for name in entry_list:
        f.write(f"    {name},\n")
    f.write(']\n')

print(f"✅ Filtri adattivi salvati in '{OUTPUT_FILTER_FILE}'")
