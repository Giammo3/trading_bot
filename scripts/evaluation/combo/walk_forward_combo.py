import sys
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import joblib
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

#percorsi
DATA_FILE = "datasets/binary_03/forex_labeled_with_binary03.csv"
OUTPUT_CSV = "datasets/combo_reversal_binary03/walk_forward_results.csv"

#carichiamo i dati
df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
df["month"] = df["timestamp"].dt.to_period("M")

#features da usare(escludiamo target e roba non numerica)
feature_cols = df.select_dtypes(include=["number"]).columns.difference([
    "reversal", "target_binary_02", "target_reversal",
    "reversal_real", "target_short", "target_3class",
    "target_5class", "trend_continuation", "volatility_breakout",
    "target_binary_01", "target_binary_03"
])

results = []

#loop mese per mese
months = sorted(df["month"].unique())
for i in range(1, len(months)):
    train_months = months[:i]   #fino al mese precedente
    test_month = months[i]      #mese corrente

    df_train = df[df["month"].isin(train_months)]
    df_test = df[df["month"] == test_month]

    X_train, y_train = df_train[feature_cols], df_train["target_binary_03"]
    X_test, y_test = df_test[feature_cols], df_test["target_binary_03"]

    #APPLICHIAMO SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    #Alleniamo modello binaty_03
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_res, y_res)

    # Predizioni
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.001).astype(int)  # soglia fissa scelta

    # Metriche base
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)

    results.append({
        "train_until": str(train_months[-1]),
        "test_month": str(test_month),
        "n_trades": pred.sum(),
        "accuracy": acc,
        "precision": prec,
        "recall": rec
    })

# salviamo i risultati
df_results = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_results.to_csv(OUTPUT_CSV, index=False)

print(f" Walk-forward mensile completato! Risultati salvati in {OUTPUT_CSV}")