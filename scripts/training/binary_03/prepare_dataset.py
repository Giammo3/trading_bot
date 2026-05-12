import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# === CONFIG ===
INPUT_PATH = "datasets/binary_03/forex_labeled_with_binary03.csv"
OUTPUT_DIR = "datasets/binary_03/"
TARGET_COLUMN = "target_binary_03"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# === CREA CARTELLA SE NON ESISTE ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === CARICA I DATI ===
df = pd.read_csv(INPUT_PATH)

# === FILTRA SOLO TRADE VALIDI ===
df = df[df["valid_trade"] == True].copy()

# === FEATURES (tutte tranne target + future + timestamp) ===
drop_cols = [
    "timestamp", "future_close", "future_return", "future_return_pct",
    "valid_trade",  # già filtrato
    "target_binary_01", "target_binary_02", "target_short", "target_3class", "target_5class",
    "reversal", "trend_continuation", "volatility_breakout",
    TARGET_COLUMN  # lo togliamo da X, ma lo usiamo come y
]

X = df.drop(columns=drop_cols, errors="ignore")
y = df[TARGET_COLUMN]

# === TRAIN/TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# === SALVATAGGIO ===
X_train.to_csv(OUTPUT_DIR + "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR + "X_test.csv", index=False)
y_train.to_csv(OUTPUT_DIR + "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR + "y_test.csv", index=False)

# === VERSIONE "FILTERED" (senza target) ===
X_train.to_csv(OUTPUT_DIR + "X_train_filtered.csv", index=False)
X_test.to_csv(OUTPUT_DIR + "X_test_filtered.csv", index=False)

print(" Dataset per binary_03 preparato e salvato.")
