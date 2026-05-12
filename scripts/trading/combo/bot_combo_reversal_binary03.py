import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import joblib
import numpy as np


# === MODELLO REVERSAL ===
df_reversal = pd.read_csv("datasets/reversal/X_test_filtered_with_flat.csv")
model_reversal = joblib.load("models/reversal/best_model_xgb.pkl")

# Usa solo le feature viste in training
X_reversal = df_reversal[model_reversal.get_booster().feature_names].copy()
pred_reversal = model_reversal.predict(X_reversal)

# Aggiungi segnali reversal
df_reversal = df_reversal.reset_index(drop=True).copy()
df_reversal["reversal_signal"] = pred_reversal

# === MODELLO BINARY_03 ===
X_binary03 = pd.read_csv("datasets/combo_reversal_binary03/X_test_filtered_combo.csv")

# Carica il modello allenato con SMOTE
model_binary03 = joblib.load("models/combo/binary03_combo_model_xgb.pkl")

# Prendi solo le colonne che il modello si aspetta
features_binary03 = model_binary03.get_booster().feature_names
X_binary03 = X_binary03[features_binary03].copy()

# === THRESHOLD HANDLING ===
monthly_path = "datasets/combo_reversal_binary03/monthly_thresholds.csv"
backtest_path = "datasets/combo_reversal_binary03/threshold_backtest.csv"

monthly_df = pd.read_csv(monthly_path)
backtest_df = pd.read_csv(backtest_path)

# Threshold migliore globale (mese corrente)
current_best = backtest_df.sort_values("saldo_finale", ascending=False).iloc[0]["threshold"]

# Threshold storici (solo debug/analisi)
monthly_thresholds = dict(zip(monthly_df["month"], monthly_df["threshold"]))

print(" Threshold storici:", monthly_thresholds)
print(f" Threshold mese corrente (da backtest): {current_best}")

best_threshold = current_best

# Predizioni probabilistiche
proba_binary03 = model_binary03.predict_proba(X_binary03)[:, 1]
pred_binary03 = (proba_binary03 >= best_threshold).astype(int)

# Debug distribuzione
print("Distribuzione Binary_03 con threshold scelto:", np.unique(pred_binary03, return_counts=True))

# Aggiungiamo il segnale dentro un df separato
df_binary03 = pd.DataFrame({"binary03_signal": pred_binary03})

# === COSTRUZIONE COMBO ===
min_len = min(len(df_reversal), len(df_binary03))
df_combo = pd.concat([
    df_reversal.iloc[:min_len].reset_index(drop=True),
    df_binary03[["binary03_signal"]].iloc[:min_len].reset_index(drop=True)
], axis=1)

# Aggiungi future_return per calcolare P&L
if "future_return" in df_reversal.columns:
    df_combo["future_return"] = df_reversal["future_return"].iloc[:len(df_combo)].values
else:
    print("️ Colonna 'future_return' mancante! Aggiungila nello starter_combo.py")


# Logica trade
df_combo["trade"] = (df_combo["binary03_signal"] == 1) & (df_combo["reversal_signal"] == 1)

# === SIMULAZIONE TRADING (P&L) ===
# Usiamo la stessa logica del reversal: pnl = future_return * 100
if "future_return" in df_combo.columns:
    df_combo["pnl_pct"] = df_combo["future_return"] * 100
else:
    print("️ Colonna 'future_return' mancante! Aggiungila in prepare_dataset combo.")
    df_combo["pnl_pct"] = 0.0

# Filtra solo i trade
df_traded = df_combo[df_combo["trade"] == True].copy()
df_traded.to_csv("datasets/combo_reversal_binary03/X_test_traded.csv", index=False)

# === LOG RISULTATI ===
print(f" Threshold scelto: {best_threshold}")
print("Segnali reversal =", df_combo["reversal_signal"].sum())
print("Segnali binary03 =", df_combo["binary03_signal"].sum())
print("Segnali COMBO =", df_combo["trade"].sum())
print(f" Trade eseguiti: {len(df_traded)} su {len(df_combo)}")
print(" Bot combo eseguito con successo.")