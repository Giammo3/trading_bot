"""
generate_features.py — Genera i dataset di feature a partire dai dati grezzi.

Sostituisce utils/features.py come script standalone.
Usa apply_all_features() da utils/feature_engineering.py come fonte unica,
garantendo che tutti i modelli (reversal, binary_02, binary_03, combo) ricevano
le stesse feature con la stessa scala.

Output
------
  datasets/forex/forex_features_optimized.csv   ← letto da utils/target.py
  datasets/forex/forex_features.csv             ← versione completa con tutte le colonne

Usage
-----
    python utils/generate_features.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.feature_engineering import apply_all_features

# ── Percorsi ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH     = ROOT / "scripts" / "data_collection" / "forex_data.csv"
FEATURES_FULL     = ROOT / "datasets" / "forex" / "forex_features.csv"
FEATURES_OPT      = ROOT / "datasets" / "forex" / "forex_features_optimized.csv"

# Colonne essenziali salvate in forex_features_optimized.csv
# (le stesse che usava features.py — target.py e tutti gli script downstream
#  si aspettano esattamente queste colonne)
ESSENTIAL_COLS = [
    "timestamp",
    "close", "open", "high", "low",
    "return_pct", "body_size", "wick_size", "candle_type",
    "ma5", "ma10", "ma50", "ma200", "ma50_slope",
    "volatility_10", "rsi_14", "macd", "macd_signal", "macd_diff",
    "bb_upper", "bb_lower", "distance_from_ma50",
    "hour_of_day", "day_of_week", "bullish_engulfing",
    "adx", "adx_direction", "doji_type", "zscore_ma50_filtered",
    "is_lon_ny_overlap", "breakout_up", "breakout_down",
    "wick_body_ratio", "liquidity_proxy", "valid_trade",
    "acceleration_5_norm",
]


def main() -> None:
    print("=" * 60)
    print("  GENERAZIONE FEATURE (utils/feature_engineering.py)")
    print("=" * 60)

    # ── Carica dati grezzi ────────────────────────────────────────────────────
    if not RAW_DATA_PATH.exists():
        print(f"\n  ERRORE: file dati grezzi non trovato: {RAW_DATA_PATH}")
        print("  Esegui prima scripts/data_collection/main.py per scaricare i dati.")
        sys.exit(1)

    print(f"\n  Caricamento dati grezzi: {RAW_DATA_PATH.name}")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    print(f"  Righe grezze: {len(df_raw)}")

    # ── Pulizia dati sporchi ──────────────────────────────────────────────────
    # EUR/USD oscilla tipicamente tra 0.95 e 1.30 — valori fuori range sono errori
    close_min, close_max = 0.90, 2.00
    outliers = (df_raw["close"] < close_min) | (df_raw["close"] > close_max)
    if outliers.sum() > 0:
        print(f"  Rimossi {outliers.sum()} campioni con close fuori range "
              f"[{close_min}, {close_max}] (dati sporchi)")
        df_raw = df_raw[~outliers].reset_index(drop=True)

    # ── Applica feature engineering ───────────────────────────────────────────
    print("\n  Calcolo feature (apply_all_features) ...")
    df = apply_all_features(df_raw, dropna=True)
    print(f"  Righe dopo dropna: {len(df)}")

    # ── Salva versione completa ───────────────────────────────────────────────
    os.makedirs(FEATURES_FULL.parent, exist_ok=True)
    df.to_csv(FEATURES_FULL, index=False)
    print(f"\n  Salvato: {FEATURES_FULL.name}  ({len(df)} righe, {len(df.columns)} colonne)")

    # ── Salva versione ottimizzata (solo colonne essenziali) ──────────────────
    # Includi solo le colonne presenti (alcune potrebbero mancare se timestamp assente)
    cols_to_save = [c for c in ESSENTIAL_COLS if c in df.columns]
    missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
    if missing:
        print(f"\n  ATTENZIONE: colonne mancanti nell'ottimizzata: {missing}")

    df[cols_to_save].to_csv(FEATURES_OPT, index=False)
    print(f"  Salvato: {FEATURES_OPT.name}  ({len(df)} righe, {len(cols_to_save)} colonne)")

    # ── Statistiche rapide ────────────────────────────────────────────────────
    print("\n  Statistiche chiave:")
    for col in ["return_pct", "volatility_10", "acceleration_5_norm"]:
        if col in df.columns:
            s = df[col].dropna()
            print(f"    {col:25s}  mean={s.mean():.6f}  std={s.std():.6f}  "
                  f"min={s.min():.6f}  max={s.max():.6f}")

    print("\n  Prossimo passo: python utils/target.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
