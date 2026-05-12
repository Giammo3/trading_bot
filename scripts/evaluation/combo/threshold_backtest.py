import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib
import numpy as np
import os

# === Percorsi ===
DATASET_COMBO = "datasets/combo_reversal_binary03/X_test_filtered_combo.csv"
MODEL_BINARY03 = "models/combo/binary03_combo_model_xgb.pkl"
MODEL_REVERSAL = "models/reversal/best_model_xgb.pkl"
OUTPUT_PATH = "datasets/combo_reversal_binary03/threshold_backtest.csv"

os.makedirs("results", exist_ok=True)

# === Carica dataset e modelli ===
df = pd.read_csv(DATASET_COMBO)
model_binary03 = joblib.load(MODEL_BINARY03)
model_reversal = joblib.load(MODEL_REVERSAL)

# Predizioni reversal
X_rev = df[model_reversal.get_booster().feature_names].copy()
pred_reversal = model_reversal.predict(X_rev)

# Probabilità binary_03
X_bin = df[model_binary03.get_booster().feature_names].copy()
proba_binary03 = model_binary03.predict_proba(X_bin)[:, 1]

# === Funzione backtest ===
def backtest(threshold):
    pred_binary03 = (proba_binary03 >= threshold).astype(int)

    # Costruzione combo
    df_tmp = df.copy()
    df_tmp["binary03_signal"] = pred_binary03
    df_tmp["reversal_signal"] = pred_reversal
    df_tmp["trade"] = (df_tmp["binary03_signal"] == 1) & (df_tmp["reversal_signal"] == 1)

    trades = df_tmp[df_tmp["trade"] == True].copy()

    if trades.empty:
        return [threshold, 0, 0, 0, 0, 10000, float("inf")]

    # Simulazione P&L
    balance = 10000
    wins, losses = 0, 0
    for _, row in trades.iterrows():
        pnl = row.get("return_pct", 0) * 100  # % profit
        balance *= (1 + pnl / 100)
        if pnl > 0:
            wins += 1
        else:
            losses += 1

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    profit_factor = (wins / max(losses, 1)) if losses > 0 else float("inf")

    return [threshold, len(trades), wins, losses, win_rate, balance, profit_factor]

# === Testa più soglie ===
thresholds = [0.0005, 0.001, 0.0015, 0.002, 0.005, 0.01]
results = [backtest(th) for th in thresholds]

# === Salva ===
cols = ["threshold", "n_trades", "wins", "losses", "win_rate", "saldo_finale", "profit_factor"]
df_results = pd.DataFrame(results, columns=cols)
df_results.to_csv(OUTPUT_PATH, index=False)

print(f" Backtest completato! Risultati salvati in {OUTPUT_PATH}")
