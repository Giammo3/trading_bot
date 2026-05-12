"""
run_combo_v2.py — Pipeline completa per il combo ensemble v2.

Esecuzione step-by-step dall'inizio alla fine:
  1.  Labeling con target_binary_03 (da forex_labeled.csv)
  2.  Preparazione dataset binary_03 (X/y train/test)
  3.  Top features binary_03
  4.  Preparazione dataset combo (X/y train/test filtrati + split target)
  5.  Retraining modello reversal (usa best_model_xgb.pkl gia' presente)
  6.  Retraining modello binary03_combo (con target corretto + scale_pos_weight)
  7.  Esecuzione bot_combo_v2 (WeightedScorePolicy + StaticThreshold 0.50)
  8.  Analisi dei trade prodotti

Uso:
    python starter_scripts/run_combo_v2.py

Opzioni:
    --skip-data      Salta i passi 1-4 (se i dataset esistono gia')
    --skip-reversal  Salta il retraining del modello reversal
    --skip-training  Salta i passi 5-6 (usa i modelli gia' presenti)
    --policy         Politica di decisione: weighted (default) | and_gate
    --threshold      Strategia soglia: static (default) | walk_forward
    --static-value   Valore soglia per la policy (default: 0.50)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Root del progetto = cartella padre di starter_scripts/ ───────────────────
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                          # tutti gli script usano path relative
PYTHON = sys.executable                 # stesso interprete (venv) del launcher


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def run(script: str, extra_args: list[str] | None = None, env_extra: dict | None = None) -> bool:
    """
    Esegue uno script Python con il PYTHONPATH impostato alla root del progetto.
    Ritorna True se lo script termina con successo, False altrimenti.
    """
    cmd = [PYTHON, script] + (extra_args or [])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)       # necessario per import 'scripts.*' e 'utils.*'
    if env_extra:
        env.update(env_extra)

    print(f"\n{'='*65}")
    print(f"  LANCIO: {script}")
    if extra_args:
        print(f"  ARGS  : {' '.join(extra_args)}")
    print(f"{'='*65}")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n  ERRORE nello script: {script}")
        print(f"  Esecuzione interrotta.")
    return result.returncode == 0


def section(title: str) -> None:
    width = 65
    print(f"\n{'#'*width}")
    print(f"#  {title}")
    print(f"{'#'*width}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline combo_v2 completa")
    parser.add_argument("--skip-data",     action="store_true",
                        help="Salta preparazione dataset (passi 1-4)")
    parser.add_argument("--skip-reversal", action="store_true",
                        help="Salta retraining modello reversal (usa best_model_xgb.pkl esistente)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Salta tutto il training (usa i modelli gia' salvati)")
    parser.add_argument("--policy",        choices=["weighted", "and_gate"],
                        default="weighted",
                        help="Politica di decisione (default: weighted)")
    parser.add_argument("--threshold",     choices=["static", "walk_forward"],
                        default="static",
                        help="Strategia soglia (default: static)")
    parser.add_argument("--static-value",  type=float, default=0.50,
                        help="Valore soglia StaticThresholdManager (default: 0.50)")
    args = parser.parse_args()

    skip_training = args.skip_training

    print("\n" + "="*65)
    print("  COMBO ENSEMBLE v2 — PIPELINE COMPLETA")
    print("="*65)
    print(f"  Policy     : {args.policy}")
    print(f"  Threshold  : {args.threshold}" +
          (f" ({args.static_value})" if args.threshold == "static" else ""))
    print(f"  Skip data  : {args.skip_data or skip_training}")
    print(f"  Skip train : {skip_training}")
    print("="*65)

    # ── FASE 1: PREPARAZIONE DATASET ────────────────────────────────────────
    if not args.skip_data and not skip_training:

        section("FASE 1 — Feature engineering (fonte unica: feature_engineering.py)")
        if not run("utils/generate_features.py"):
            sys.exit(1)

        section("FASE 2 — Labeling: calcola tutti i target (forex_labeled.csv)")
        if not run("utils/target.py"):
            sys.exit(1)

        section("FASE 3 — Aggiunge target_binary_03 a forex_labeled_with_binary03.csv")
        if not run("scripts/training/binary_03/forex_with_binary_03.py"):
            sys.exit(1)

        section("FASE 4 — Preparazione dataset binary_03 (X/y train+test)")
        if not run("scripts/training/binary_03/prepare_dataset.py"):
            sys.exit(1)

        section("FASE 5 — Selezione top features binary_03")
        if not run("scripts/training/binary_03/generate_top_features.py"):
            sys.exit(1)

        section("FASE 6 — Preparazione dataset combo (starter_combo + split target)")
        # starter_combo: genera X_train/test_filtered_combo e y_train/test_filtered_combo
        if not run("scripts/training/combo/starter_combo.py"):
            sys.exit(1)
        # split_target: genera y_train_binary03.csv e y_test_binary03.csv (target CORRETTO)
        if not run("scripts/training/combo/split_target_binary03.py"):
            sys.exit(1)

    # ── FASE 2: TRAINING MODELLI ─────────────────────────────────────────────
    if not skip_training:

        if not args.skip_reversal:
            section("FASE 7 — Retraining modello reversal (tune_model_xgb)")
            if not run("scripts/training/reversal/tune_model_xgb.py"):
                sys.exit(1)
        else:
            print("\n  [SKIP] Retraining reversal — usando best_model_xgb.pkl esistente")

        section("FASE 8 — Retraining binary03_combo (target corretto + scale_pos_weight)")
        if not run("scripts/training/combo/retrain_binary03_fixed.py"):
            sys.exit(1)

    # ── FASE 3: BOT COMBO v2 ─────────────────────────────────────────────────
    section("FASE 9 — Esecuzione bot_combo_v2")
    bot_args = [
        "--policy",        args.policy,
        "--threshold",     args.threshold,
        "--static-value",  str(args.static_value),
    ]
    if not run("scripts/trading/combo/bot_combo_v2.py", extra_args=bot_args):
        sys.exit(1)

    # ── FASE 4: ANALISI ──────────────────────────────────────────────────────
    section("FASE 10 — Analisi trade prodotti")
    # analyze_trades.py legge X_test_traded.csv (vecchio bot)
    # bot_combo_v2 scrive X_test_traded_v2.csv — creiamo un alias se necessario
    v2_output  = ROOT / "datasets/combo_reversal_binary03/X_test_traded_v2.csv"
    old_output = ROOT / "datasets/combo_reversal_binary03/X_test_traded.csv"
    if v2_output.exists():
        import pandas as pd
        df = pd.read_csv(v2_output)

        traded = df[df.get("traded", df.get("should_trade", False)) == True] \
            if "traded" in df.columns else df[df["should_trade"] == True]

        print(f"\n  Totale barre analizzate : {len(df)}")
        print(f"  Trade eseguiti          : {len(traded)}")

        if "pnl_pct" in traded.columns and not traded.empty:
            pnl = traded["pnl_pct"].dropna()
            wins   = (pnl > 0).sum()
            losses = (pnl <= 0).sum()
            total  = pnl.sum()
            wr     = wins / max(wins + losses, 1) * 100
            pf     = (pnl[pnl > 0].sum() / abs(pnl[pnl <= 0].sum())) \
                     if (pnl <= 0).any() else float("inf")
            print(f"\n  Win rate        : {wr:.1f}%  ({wins}W / {losses}L)")
            print(f"  PnL totale      : {total:.3f}%")
            print(f"  Profit factor   : {pf:.3f}")
        else:
            print("  (nessun dato PnL — normale se il dataset non ha timestamp)")

        if "meta__rejection_reason" in df.columns:
            print("\n  Top rejection reasons:")
            reasons = df["meta__rejection_reason"].value_counts().head(5)
            for reason, count in reasons.items():
                print(f"    {count:5d}  {reason}")

        # Copia in X_test_traded.csv per compatibilita' con analyze_trades.py
        df_compat = traded[["pnl_pct"]].copy() if "pnl_pct" in traded.columns else traded
        df_compat.to_csv(old_output, index=False)

    if not run("scripts/evaluation/combo/analyze_trades.py"):
        print("  (analyze_trades.py non ha trovato trade — normale se PnL non e' disponibile)")

    # ── FINE ──────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PIPELINE COMBO v2 COMPLETATA")
    print("="*65)
    print(f"  Output principale : datasets/combo_reversal_binary03/X_test_traded_v2.csv")
    print(f"  Modelli aggiornati:")
    print(f"    models/combo/binary03_combo_model_xgb.pkl")
    if not args.skip_reversal and not skip_training:
        print(f"    models/reversal/best_model_xgb.pkl")
    print("="*65)


if __name__ == "__main__":
    main()
