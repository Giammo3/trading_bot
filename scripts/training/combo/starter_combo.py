import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

# === Percorsi ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.feature_engineering import apply_all_features
from scripts.features.feature_selection import select_features

INPUT_FILE = "datasets/forex/forex_labeled.csv"
OUTPUT_FOLDER = "datasets/combo_reversal_binary03/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Carica dati e applica tutte le feature
df = pd.read_csv(INPUT_FILE)
df = apply_all_features(df)

# === Pulisci e gestisci NaN
df.dropna(inplace=True)

# === Calcolo future_return (per P&L)
if "future_return" not in df.columns:
    df["future_close"] = df["close"].shift(-40)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

# === Aggiungi target_binary_03 se mancante
if 'target_binary_03' not in df.columns:
    df['target_binary_03'] = ((df['return_pct'].shift(-40) > 0.0015) | 
                              (df['return_pct'].shift(-40) < -0.0015)).astype(int)

# === Split X e y
feature_cols = df.columns.difference([
    'reversal', 'target_binary_03', 'target_binary_02',
    'target_reversal', 'reversal_real'
])
X = df[feature_cols]
y = df['target_binary_03']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

# === Salvataggi base
X_train.to_csv(os.path.join(OUTPUT_FOLDER, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_FOLDER, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_FOLDER, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_FOLDER, "y_test.csv"), index=False)

# === Feature selection top_k
X_train_num = X_train.select_dtypes(include=['number'])
top_features = select_features(X_train_num, y_train, top_k=30)

# === Carica top_features reversal
with open("datasets/reversal/top_features.csv") as f:
    top_features_reversal = [line.strip() for line in f]

# === Carica top_features binary_03
with open("datasets/binary_03/top_features.csv") as f:
    top_features_binary03 = [line.strip() for line in f]

# === Carica tutte le feature usate dal modello binary_03
model_path = "models/combo/binary03_combo_model_xgb.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    features_from_model = model.get_booster().feature_names
else:
    features_from_model = []

# === Unisci tutte le feature necessarie
combined_features = list(set(top_features + top_features_reversal + top_features_binary03 + features_from_model))

# Rimuovi eventuali valori strani tipo "0" o stringhe vuote
combined_features = [f for f in combined_features if f in X_train.columns]


# === Filtra X_train e X_test
X_train_filtered = X_train[combined_features].copy()
X_test_filtered = X_test[combined_features].copy()

# === Aggiunte per il test set
df_tail = df.tail(len(X_test)).reset_index(drop=True)
X_test_filtered["reversal"] = df_tail["reversal"].values

# flat_market calcolato con soglie calibrate su train (non sul test set)
train_vol_p25  = float(X_train["volatility_10"].quantile(0.25)) \
                 if "volatility_10" in X_train.columns else 0.00055
train_wick_p25 = float(X_train["wick_size"].quantile(0.25)) \
                 if "wick_size" in X_train.columns else 0.0007
X_test_filtered["flat_market"] = (
    (df_tail["volatility_10"] < train_vol_p25) &
    (df_tail["wick_size"]     < train_wick_p25)
).astype(int)

X_test_filtered["target_binary_03"] = df_tail["target_binary_03"].values

# market_session_code: necessario per SessionFilter nel combo bot
# (richiesto da SignalAggregator._DEFAULT_CONTEXT_COLS)
if "market_session_code" in df_tail.columns:
    X_test_filtered["market_session_code"] = df_tail["market_session_code"].values
    X_train_filtered["market_session_code"] = \
        X_train.reset_index(drop=True)["market_session_code"].values \
        if "market_session_code" in X_train.columns else 0

if "future_return" in df_tail.columns:
    X_test_filtered["future_return"] = df_tail["future_return"].values


#  Aggiungi future_return per analisi P&L
if "future_return" in df_tail.columns:
    X_test_filtered["future_return"] = df_tail["future_return"].values

# Allinea i target con la lunghezza dei set filtrati
y_train_filtered = y_train.reset_index(drop=True).iloc[:len(X_train_filtered)]
y_test_filtered = y_test.reset_index(drop=True).iloc[:len(X_test_filtered)]

# === Salvataggi finali ===
X_train_filtered.to_csv(os.path.join(OUTPUT_FOLDER, "X_train_filtered_combo.csv"), index=False)
X_test_filtered.to_csv(os.path.join(OUTPUT_FOLDER, "X_test_filtered_combo.csv"), index=False)
y_train_filtered.to_csv(os.path.join(OUTPUT_FOLDER, "y_train_filtered_combo.csv"), index=False)
y_test_filtered.to_csv(os.path.join(OUTPUT_FOLDER, "y_test_filtered_combo.csv"), index=False)

print(" Starter combo completato.")
print(f" Dati salvati in: {OUTPUT_FOLDER}")

