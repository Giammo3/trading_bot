# scripts/evaluation/combo/monthly_threshold_search.py

import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib
import numpy as np
import os
from collections import defaultdict

# === CONFIG ===
DATASET_PATH = "datasets/binary_03/forex_labeled_with_binary03.csv"
MODEL_PATH = "models/combo/binary03_combo_model_xgb.pkl"
OUTPUT_CSV = "datasets/combo_reversal_binary03/monthly_thresholds.csv"

# === Carica dataset completo ===
df = pd.read_csv(DATASET_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["month"] = df["timestamp"].dt.to_period("M")

# Carica modello
model = joblib.load(MODEL_PATH)
features = model.get_booster().feature_names

# Predizioni probabilistiche
proba = model.predict_proba(df[features])[:, 1]
df["proba_binary03"] = proba

# === Funzione per simulare trading ===
def simulate_trades(preds, y_true, initial_balance=10000, risk=0.01):
    balance = initial_balance
    wins = losses = 0
    pnl_list = []

    for pred, true in zip(preds, y_true):
        if pred == 1:  # eseguiamo un trade
            if true == 1:  # WIN
                profit = balance * risk * 2  # esempio RR 2:1
                balance += profit
                wins += 1
                pnl_list.append(profit / initial_balance * 100)
            else:  # LOSS
                loss = balance * risk
                balance -= loss
                losses += 1
                pnl_list.append(-loss / initial_balance * 100)

    n_trades = wins + losses
    win_rate = wins / n_trades if n_trades > 0 else 0
    profit_factor = (sum([x for x in pnl_list if x > 0]) / abs(sum([x for x in pnl_list if x < 0]))) if losses > 0 else float("inf")

    return {
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "saldo_finale": balance,
        "profit_factor": profit_factor
    }

# === Loop mensile con varie soglie ===
thresholds = [0.0005, 0.001, 0.0015, 0.002, 0.005]
results = []

for month, group in df.groupby("month"):
    y_true = group["target_binary_03"].values
    proba_month = group["proba_binary03"].values

    best_row = None

    for th in thresholds:
        preds = (proba_month >= th).astype(int)
        stats = simulate_trades(preds, y_true)

        row = {
            "month": str(month),
            "threshold": th,
            **stats
        }

        # Scegliamo il threshold migliore in base al profit_factor
        if best_row is None or row["profit_factor"] > best_row["profit_factor"]:
            best_row = row

    results.append(best_row)

# === Salva risultati ===
df_results = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_results.to_csv(OUTPUT_CSV, index=False)

print(f" Threshold mensili calcolati! Salvati in {OUTPUT_CSV}")
print(df_results)
