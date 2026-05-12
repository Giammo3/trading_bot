"""
bot_combo_v2.py – New combo bot entry point using the modular ensemble.

This is a drop-in replacement for bot_combo_reversal_binary03.py.
It is intentionally thin: all logic lives in the ensemble package.

What changed vs. the old bot
-----------------------------

OLD bot_combo_reversal_binary03.py problems:
  ❌ Positional pd.concat merge (silent misalignment)
  ❌ Threshold selected by maximising balance on the test set (look-ahead)
  ❌ P&L computed from future_return (no real TP/SL execution)
  ❌ No filter composability
  ❌ Hard AND-gate only (no confidence weighting)

NEW bot_combo_v2.py:
  ✅ Timestamp-keyed merge via SignalAggregator
  ✅ Walk-forward threshold calibration (no look-ahead bias)
  ✅ Real TP/SL simulation via EnsembleOrchestrator._run_backtest_execution
  ✅ Modular FilterChain (each filter independently toggleable)
  ✅ WeightedScorePolicy with hard reversal gate + soft binary_03 weighting

Usage
-----
    python scripts/trading/combo/bot_combo_v2.py

    # Override threshold strategy:
    python scripts/trading/combo/bot_combo_v2.py --threshold static --static-value 0.52

    # Run in live mode (prints decision, no CSV write):
    python scripts/trading/combo/bot_combo_v2.py --live

Configuration
-------------
All paths and parameters can be overridden via CLI args or by editing
the DEFAULT_* constants below.  For production, consider loading from
a YAML/JSON config file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trading.combo.ensemble import (
    Binary03Adapter,
    EnsembleOrchestrator,
    FilterChain,
    FlatMarketFilter,
    LiquidityFilter,
    SessionFilter,
)
from scripts.trading.combo.ensemble.decision import WeightedScorePolicy, ANDGatePolicy
from scripts.trading.combo.ensemble.threshold import (
    StaticThresholdManager,
    WalkForwardThresholdManager,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger("bot_combo_v2")

# ── Default configuration ─────────────────────────────────────────────────
DEFAULT_DATASET     = "datasets/combo_reversal_binary03/X_test_filtered_combo.csv"
DEFAULT_REVERSAL_MODEL  = "models/reversal/best_model_xgb.pkl"
DEFAULT_BINARY03_MODEL  = "models/combo/binary03_combo_model_xgb.pkl"
DEFAULT_OUTPUT_PATH = "datasets/combo_reversal_binary03/X_test_traded_v2.csv"

# TP/SL/look-ahead (matching existing reversal bot defaults)
TAKE_PROFIT_PCT  = 0.006   # 0.6%
STOP_LOSS_PCT    = 0.004   # 0.4% base (dynamic SL capped at 3×)
LOOK_AHEAD_STEPS = 20


# ---------------------------------------------------------------------------
# Threshold factory
# ---------------------------------------------------------------------------

def build_threshold_manager(args, df: pd.DataFrame, binary03_model_path: str):
    """
    Build and optionally fit a threshold manager.

    Walk-forward fitting requires the binary_03 model to generate
    probabilities on the training portion of the data.
    """
    if args.threshold == "static":
        th = getattr(args, "static_value", 0.50)
        logger.info("Using StaticThresholdManager(threshold=%.4f)", th)
        return StaticThresholdManager(threshold=th)

    if args.threshold == "walk_forward":
        logger.info("Fitting WalkForwardThresholdManager …")
        import joblib

        model = joblib.load(binary03_model_path)
        features = model.get_booster().feature_names

        forbidden = {"future_return", "future_return_pct", "future_close",
                     "target_binary_03", "reversal", "reversal_real",
                     "target_reversal", "target_binary_02"}
        features = [f for f in features if f not in forbidden]

        available = [f for f in features if f in df.columns]
        if len(available) < len(features):
            missing = set(features) - set(available)
            logger.warning("Walk-forward fit: %d features missing, skipping: %s",
                           len(missing), missing)

        X = df[available].copy()
        proba = model.predict_proba(X)[:, 1]

        # Labels — use target_binary_03 if present, otherwise infer from proba
        if "target_binary_03" in df.columns:
            labels = df["target_binary_03"].values
        else:
            logger.warning(
                "target_binary_03 not in dataset — walk-forward will use "
                "proba > 0.5 as pseudo-labels (less reliable)."
            )
            labels = (proba > 0.5).astype(int)

        ts_col = "timestamp" if "timestamp" in df.columns else None
        timestamps = (
            pd.to_datetime(df[ts_col]) if ts_col else pd.Series(range(len(df)))
        )

        tm = WalkForwardThresholdManager(
            min_train_bars=2000,
            val_window_bars=500,
        )
        tm.fit(timestamps, proba, labels)
        logger.info("Walk-forward threshold schedule:\n%s", tm.summary().to_string())
        return tm

    raise ValueError(f"Unknown threshold strategy: {args.threshold!r}")


# ---------------------------------------------------------------------------
# Filter chain factory
# ---------------------------------------------------------------------------

# Percorso del JSON soglie — relativo alla root del progetto
# Usa Path(__file__) per essere indipendente dalla CWD
_BOT_ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS_PATH = _BOT_ROOT / "models" / "combo" / "filter_thresholds.json"


def build_filter_chain(args, df: pd.DataFrame) -> FilterChain:
    """
    Costruisce il FilterChain con soglie prive di data leakage.

    Priorità per le soglie di FlatMarketFilter:
      1. --vol-threshold / --wick-threshold  (CLI esplicito)
      2. models/combo/filter_thresholds.json (calibrate su X_train durante il training)
      3. Errore esplicito — NON ricalcola sul dataset di input (sarebbe leakage)

    Il file filter_thresholds.json viene generato da retrain_binary03_fixed.py
    che lo calibra esclusivamente su X_train_filtered_combo.csv.
    """
    import json

    vol_th  = getattr(args, "vol_threshold",  0.0)
    wick_th = getattr(args, "wick_threshold", 0.0)

    if vol_th > 0 and wick_th > 0:
        # Soglie fornite esplicitamente da CLI
        logger.info(
            "FlatMarketFilter: soglie da CLI — vol=%.6f  wick=%.6f", vol_th, wick_th
        )
    else:
        # Carica dal file calibrato su X_train
        if THRESHOLDS_PATH.exists():
            with open(THRESHOLDS_PATH) as f:
                th_data = json.load(f)
            fm = th_data.get("flat_market_filter", {})
            vol_th  = float(fm.get("volatility_threshold", 0.0))
            wick_th = float(fm.get("wick_threshold",       0.0))
            cal_on  = fm.get("calibrated_on", "?")
            n_rows  = fm.get("n_train_rows",  "?")
            logger.info(
                "FlatMarketFilter: soglie da %s (calibrate su %s, %s righe) — "
                "vol=%.6f  wick=%.6f",
                THRESHOLDS_PATH.name, cal_on, n_rows, vol_th, wick_th
            )
            if vol_th <= 0 or wick_th <= 0:
                raise ValueError(
                    f"filter_thresholds.json contiene soglie non valide: "
                    f"vol={vol_th}, wick={wick_th}. "
                    f"Esegui retrain_binary03_fixed.py per rigenerarlo."
                )
        else:
            raise FileNotFoundError(
                f"File soglie non trovato: {THRESHOLDS_PATH}\n"
                f"Esegui prima: python scripts/training/combo/retrain_binary03_fixed.py\n"
                f"Oppure specifica le soglie via CLI: --vol-threshold X --wick-threshold Y"
            )

    chain = FilterChain([
        FlatMarketFilter(
            volatility_threshold=vol_th,
            wick_threshold=wick_th,
            require_both=True,
            enabled=not getattr(args, "no_flat_filter", False),
        ),
        SessionFilter(
            allowed_session_codes=[1, 2, 3, 4],
            enabled=not getattr(args, "no_session_filter", False),
        ),
        LiquidityFilter(
            min_liquidity=0.5,
            enabled=not getattr(args, "no_liquidity_filter", False),
        ),
    ])
    logger.info("Active filters: %s", chain.filter_names)
    return chain


# ---------------------------------------------------------------------------
# Decision policy factory
# ---------------------------------------------------------------------------

def build_policy(args):
    """Build the decision policy from CLI arguments."""
    if getattr(args, "policy", "weighted") == "and_gate":
        logger.info("Using legacy ANDGatePolicy")
        return ANDGatePolicy(required_models=["reversal", "binary03"])

    logger.info("Using WeightedScorePolicy (recommended)")
    return WeightedScorePolicy(
        weights={"reversal": 1.0, "binary03": 1.5},
        score_threshold=getattr(args, "score_threshold", 0.55),
        hard_gate_models=["reversal"],
        min_model_confidence=0.40,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Combo Ensemble Bot v2")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--reversal-model", default=DEFAULT_REVERSAL_MODEL)
    parser.add_argument("--binary03-model", default=DEFAULT_BINARY03_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--threshold",
        choices=["static", "walk_forward"],
        default="static",
        help="Threshold strategy for binary_03",
    )
    parser.add_argument(
        "--static-value",
        type=float,
        default=0.50,
        help="Threshold value when --threshold=static",
    )
    parser.add_argument(
        "--policy",
        choices=["weighted", "and_gate"],
        default="weighted",
        help="Decision policy",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.55,
        help="Composite score threshold for WeightedScorePolicy",
    )
    parser.add_argument(
        "--reversal-gate",
        type=float,
        default=0.5,
        help="Probability threshold for the reversal hard gate (default: 0.5). "
             "Lower values (e.g. 0.10) allow more reversal signals through.",
    )
    parser.add_argument(
        "--vol-threshold",
        type=float,
        default=0.0,
        help="Soglia volatility_10 per FlatMarketFilter (default: auto-calibrata al p25 del dataset).",
    )
    parser.add_argument(
        "--wick-threshold",
        type=float,
        default=0.0,
        help="Soglia wick_size per FlatMarketFilter (default: auto-calibrata al p25 del dataset).",
    )
    parser.add_argument(
        "--no-flat-filter", action="store_true", help="Disable flat market filter"
    )
    parser.add_argument(
        "--no-session-filter", action="store_true", help="Disable session filter"
    )
    parser.add_argument(
        "--no-liquidity-filter", action="store_true", help="Disable liquidity filter"
    )
    parser.add_argument(
        "--live", action="store_true", help="Run in live mode (no backtest execution)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--diag-filters", action="store_true",
        help="Stampa diagnostica dettagliata per ogni filtro (barre bloccate, motivi, colonne mancanti)",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading dataset: %s", args.dataset)
    df = pd.read_csv(args.dataset)
    logger.info("Dataset shape: %s", df.shape)

    # ── Build components ──────────────────────────────────────────────────
    threshold_mgr = build_threshold_manager(args, df, args.binary03_model)
    filter_chain  = build_filter_chain(args, df)
    policy        = build_policy(args)

    # ── Build orchestrator ────────────────────────────────────────────────
    orchestrator = EnsembleOrchestrator(
        reversal_model_path=args.reversal_model,
        binary03_model_path=args.binary03_model,
        threshold_manager=threshold_mgr,
        filter_chain=filter_chain,
        decision_policy=policy,
        take_profit_pct=TAKE_PROFIT_PCT,
        stop_loss_pct=STOP_LOSS_PCT,
        look_ahead_steps=LOOK_AHEAD_STEPS,
        timestamp_col="timestamp" if "timestamp" in df.columns else None,
        backtest_mode=not args.live,
        output_path=args.output if not args.live else None,
        reversal_gate_threshold=args.reversal_gate,
        diag_filters=args.diag_filters,
    )

    # ── Run ───────────────────────────────────────────────────────────────
    logger.info("Running ensemble pipeline …")
    result_df = orchestrator.run(df)

    # ── Summary ───────────────────────────────────────────────────────────
    traded_df = result_df[result_df.get("traded", pd.Series(dtype=bool)) == True]

    print("\n" + "=" * 60)
    print("  COMBO ENSEMBLE v2 — RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Policy          : {policy.__class__.__name__}")
    print(f"  Threshold type  : {threshold_mgr.__class__.__name__}")
    print(f"  Reversal gate   : {args.reversal_gate:.2f}")
    print(f"  Active filters  : {filter_chain.filter_names}")
    print(f"  Total bars      : {len(result_df)}")

    if "reversal__prediction" in result_df.columns:
        print(f"  Reversal fires  : {result_df['reversal__prediction'].sum()}")
    if "binary03__prediction" in result_df.columns:
        print(f"  Binary03 fires  : {result_df['binary03__prediction'].sum()}")

    print(f"  Trades taken    : {len(traded_df)}")

    if "pnl_pct" in traded_df.columns and not traded_df.empty:
        pnl = traded_df["pnl_pct"].dropna()
        wins   = (pnl > 0).sum()
        losses = (pnl <= 0).sum()
        total_pnl = pnl.sum()
        win_rate = wins / max(wins + losses, 1) * 100
        print(f"  Win rate        : {win_rate:.1f}%  ({wins}W / {losses}L)")
        print(f"  Total PnL       : {total_pnl:.3f}%")
        if losses > 0:
            profit_factor = (pnl[pnl > 0].sum() / abs(pnl[pnl <= 0].sum()))
            print(f"  Profit factor   : {profit_factor:.3f}")
    else:
        print("  (No PnL data — live mode or no trades)")

    if not args.live:
        print(f"\n  Results saved to: {args.output}")

    print("=" * 60)


if __name__ == "__main__":
    main()
