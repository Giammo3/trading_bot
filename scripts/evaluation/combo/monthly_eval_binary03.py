import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score

# === PERCORSI ===
DATASET_PATH = "datasets/binary_03/forex_labeled_with_binary03.csv"
MODEL_PATH = "models/combo/binary03_combo_model_xgb.pkl"
OUTPUT_CSV = "datasets/combo_reversal_binary03/monthly_eval_binary03.csv"

# === CARICA IL DATASET COMPLETO (con target_binary_03 già presente) ===
df_full = pd.read_csv(DATASET_PATH)
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
df_full["month"] = df_full["timestamp"].dt.to_period("M")

# === CARICA IL MODELLO ===
model = joblib.load(MODEL_PATH)
features = model.get_booster().feature_names

#  Se mancano feature richieste dal modello, aggiungile con valore 0
for f in features:
    if f not in df_full.columns:
        df_full[f] = 0

# === PREVISIONI ===
proba = model.predict_proba(df_full[features])[:, 1]
threshold = 0.001   # threshold che hai scelto
df_full["pred"] = (proba >= threshold).astype(int)
df_full["proba"] = proba

# === ANALISI MENSILE ===
rows = []
months = sorted(df_full["month"].unique())

for m in months:
    df_m = df_full[df_full["month"] == m]
    if df_m.empty:
        continue

    y_true = df_m["target_binary_03"]
    y_pred = df_m["pred"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # Simulazione saldo mensile (1% rischio per trade)
    saldo = 10000
    pnl = []
    for yt, yp in zip(y_true, y_pred):
        if yp == 1:
            if yt == 1:
                saldo *= 1.01
                pnl.append(0.01)
            else:
                saldo *= 0.99
                pnl.append(-0.01)
    wins = sum(1 for x in pnl if x > 0)
    losses = sum(1 for x in pnl if x < 0)
    profit_factor = (sum(x for x in pnl if x > 0) /
                     abs(sum(x for x in pnl if x < 0))) if losses > 0 else np.inf

    rows.append({
        "month": str(m),
        "n_trades": len(pnl),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(pnl) if len(pnl) > 0 else 0,
        "saldo_finale": saldo,
        "profit_factor": profit_factor
    })

# === SALVATAGGIO ===
df_result = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_result.to_csv(OUTPUT_CSV, index=False)

print(f" Valutazione mensile completata! Risultati salvati in {OUTPUT_CSV}")
