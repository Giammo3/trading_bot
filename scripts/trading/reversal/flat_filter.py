import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# === INPUT/OUTPUT ===
INPUT_FILE = 'datasets/reversal/X_test.csv'
OUTPUT_FILE = 'datasets/reversal/X_test_filtered_with_flat.csv'
TOP_FEATURES_FILE = 'datasets/reversal/top_features.csv'
LABELED_DATA = 'datasets/forex/forex_labeled.csv'  # <--- Aggiunto

# === Carica il dataset reversal X_test
X_test = pd.read_csv(INPUT_FILE)

# === Carica le top features
top_features = pd.read_csv(TOP_FEATURES_FILE, header=None).squeeze().tolist()
top_features = [f for f in top_features if f != '0']

# Aggiungiamo manualmente le feature usate nel filtro flat
for feature in ['volatility_10', 'wick_size']:
    if feature not in top_features:
        top_features.append(feature)

# Sicurezza: se 'close' non è già tra le features, aggiungila
if 'close' not in top_features:
    top_features.append('close')
    
# === Seleziona solo le feature selezionate
X_test = X_test[top_features].copy()

# === Crea colonna flat_market usando le due feature chiave
flat_market = (
    (X_test['volatility_10'] < 0.00055) &
    (X_test['wick_size'] < 0.0007)
)
X_test['flat_market'] = flat_market.astype(int)

# === Aggiungi colonna 'reversal' (allineata al test set)
df_labeled = pd.read_csv(LABELED_DATA)
reversal_col = df_labeled.tail(len(X_test)).reset_index(drop=True)['reversal']
X_test['reversal'] = reversal_col

# === Salvataggio
X_test.to_csv(OUTPUT_FILE, index=False)

print(f" Filtro 'flat market' applicato su {flat_market.sum()} righe.")
print(f" Salvato in '{OUTPUT_FILE}'")
