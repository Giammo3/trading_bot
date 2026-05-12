"""
retrain_binary03_fixed.py — Retraining corretto del modello binary_03 combo.

ROOT CAUSE del bug precedente
------------------------------
Il vecchio training (retrain_combo_binary03_model.py) usava y_train_filtered_combo.csv
come target, ma quel file era stato corrotto da starter_combo.py che ricalcolava
target_binary_03 su un dataframe con shift sbagliato → solo 62 positivi su 8127 (0.76%).

Il target CORRETTO è y_train_binary03.csv (generato da split_target_binary03.py):
  - y_train_binary03: 1526 positivi su 8127 (18.78%)
  - y_test_binary03:   486 positivi su 3484 (13.95%)

Questo script
-------------
1. Carica X_train_filtered_combo + y_train_binary03 (target corretto)
2. Rimuove le colonne forbidden (look-ahead / label leakage)
3. Usa scale_pos_weight per gestire lo sbilanciamento residuo
4. Valuta su X_test_filtered_combo + y_test_binary03
5. Salva il modello in models/combo/binary03_combo_model_xgb.pkl
6. Stampa distribuzioni di probabilità per verificare che il collasso sia risolto

Usage
-----
    python scripts/training/combo/retrain_binary03_fixed.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Colonne che NON devono mai entrare nel modello (look-ahead / label leakage)
FORBIDDEN = frozenset([
    "future_return",
    "future_return_pct",
    "future_close",
    "target_binary_03",
    "reversal",
    "reversal_real",
    "target_reversal",
    "target_binary_02",
    "flat_market",       # flag calcolato sul test set — non disponibile live
])

# ── Percorsi ──────────────────────────────────────────────────────────────────
DATA_DIR         = ROOT / "datasets" / "combo_reversal_binary03"
MODEL_PATH       = ROOT / "models" / "combo" / "binary03_combo_model_xgb.pkl"
THRESHOLDS_PATH  = ROOT / "models" / "combo" / "filter_thresholds.json"


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train_filtered_combo.csv")
    X_test  = pd.read_csv(DATA_DIR / "X_test_filtered_combo.csv")

    # Target CORRETTO (generato da split_target_binary03.py)
    y_train = pd.read_csv(DATA_DIR / "y_train_binary03.csv").squeeze()
    y_test  = pd.read_csv(DATA_DIR / "y_test_binary03.csv").squeeze()

    return X_train, X_test, y_train, y_test


def filter_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Rimuove le colonne forbidden e allinea train/test sulle stesse colonne."""
    feature_cols = [c for c in X_train.columns if c not in FORBIDDEN]
    # Usa solo le colonne presenti in entrambi i set
    feature_cols = [c for c in feature_cols if c in X_test.columns]
    return X_train[feature_cols].copy(), X_test[feature_cols].copy(), feature_cols


def main():
    print("=" * 65)
    print("  RETRAINING binary03_combo_model_xgb  (fixed target)")
    print("=" * 65)

    # ── Carica dati ───────────────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = load_data()
    print(f"\nDati caricati:")
    print(f"  X_train : {X_train_raw.shape}")
    print(f"  X_test  : {X_test_raw.shape}")

    # ── Verifica target ───────────────────────────────────────────────────────
    n_pos_train = int(y_train.sum())
    n_neg_train = int((y_train == 0).sum())
    n_pos_test  = int(y_test.sum())
    n_neg_test  = int((y_test == 0).sum())
    print(f"\nTarget distribution:")
    print(f"  Train: {n_pos_train} positivi / {len(y_train)} totali "
          f"({n_pos_train / len(y_train) * 100:.1f}%)")
    print(f"  Test : {n_pos_test} positivi / {len(y_test)} totali "
          f"({n_pos_test / len(y_test) * 100:.1f}%)")

    if n_pos_train < 100:
        print(f"\n  ATTENZIONE: solo {n_pos_train} positivi in training.")
        print("  Controlla che y_train_binary03.csv sia il file corretto.")
        sys.exit(1)

    # ── Filtra feature ────────────────────────────────────────────────────────
    X_train, X_test, feature_cols = filter_features(X_train_raw, X_test_raw)
    print(f"\nFeature usate per il training: {len(feature_cols)}")
    print(f"  {feature_cols}")
    forbidden_dropped = [c for c in X_train_raw.columns if c in FORBIDDEN]
    if forbidden_dropped:
        print(f"\nColonne forbidden rimosse: {forbidden_dropped}")

    # ── scale_pos_weight ──────────────────────────────────────────────────────
    spw = round(n_neg_train / max(n_pos_train, 1), 2)
    print(f"\nscale_pos_weight = {n_neg_train}/{n_pos_train} = {spw}")

    # ── Addestramento ─────────────────────────────────────────────────────────
    print("\nAddestramento XGBoost ...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print("  Addestramento completato.")

    # ── Valutazione ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  VALUTAZIONE SUL TEST SET")
    print("=" * 65)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print()
    print("  Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("  Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Distribuzione probabilità (verifica collasso) ─────────────────────────
    print("\n" + "=" * 65)
    print("  DISTRIBUZIONE PROBABILITA' (verifica collasso)")
    print("=" * 65)
    pct = np.percentile(y_proba, [25, 50, 75, 90, 95, 99])
    print(f"\n  min={y_proba.min():.4f}  max={y_proba.max():.4f}  "
          f"mean={y_proba.mean():.4f}  median={np.median(y_proba):.4f}")
    print(f"  p25={pct[0]:.4f}  p50={pct[1]:.4f}  p75={pct[2]:.4f}  "
          f"p90={pct[3]:.4f}  p95={pct[4]:.4f}  p99={pct[5]:.4f}")

    for th in [0.35, 0.40, 0.45, 0.50]:
        n = (y_proba >= th).sum()
        print(f"  Barre con proba >= {th}: {n} / {len(y_proba)} ({n/len(y_proba)*100:.1f}%)")

    if y_proba.max() < 0.35:
        print("\n  ATTENZIONE: il massimo di proba e' ancora < 0.35.")
        print("  Il modello potrebbe necessitare di ulteriore tuning.")
    else:
        print(f"\n  OK: il modello produce proba >= 0.35 su alcune barre.")

    # ── Feature importance top-10 ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FEATURE IMPORTANCE (top 10)")
    print("=" * 65)
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = imp.sort_values(ascending=False).head(10)
    for feat, score in top10.items():
        print(f"  {feat:35s}  {score:.4f}")

    # ── Calibrazione soglie FlatMarketFilter su X_train (no leakage) ─────────
    print("\n" + "=" * 65)
    print("  CALIBRAZIONE SOGLIE FlatMarketFilter (su X_train)")
    print("=" * 65)

    import json

    vol_p25  = float(np.percentile(X_train["volatility_10"].dropna(), 25)) \
               if "volatility_10" in X_train.columns else 0.00055
    wick_p25 = float(np.percentile(X_train["wick_size"].dropna(), 25)) \
               if "wick_size" in X_train.columns else 0.0007

    thresholds = {
        "flat_market_filter": {
            "volatility_threshold": vol_p25,
            "wick_threshold":       wick_p25,
            "percentile_used":      25,
            "calibrated_on":        "X_train_filtered_combo",
            "n_train_rows":         len(X_train),
        }
    }

    os.makedirs(THRESHOLDS_PATH.parent, exist_ok=True)
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n  volatility_threshold (p25 train) : {vol_p25:.6f}")
    print(f"  wick_threshold       (p25 train) : {wick_p25:.6f}")
    print(f"  Soglie salvate in               : {THRESHOLDS_PATH}")

    # ── Salvataggio modello ───────────────────────────────────────────────────
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n  Modello salvato in: {MODEL_PATH}")
    print("\n  Per validare il fix, esegui:")
    print("    python scripts/trading/combo/bot_combo_v2.py --threshold static --static-value 0.50")
    print("=" * 65)


if __name__ == "__main__":
    main()
