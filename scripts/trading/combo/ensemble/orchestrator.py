"""
orchestrator.py – EnsembleOrchestrator: top-level wiring and execution flow.

This is the single public entry point for the entire combo ensemble.
It owns:
  * Component construction (adapters, aggregator, threshold manager,
    decision engine, filter chain)
  * The canonical execution flow:
      load data → score models → aggregate signals → filter → decide → execute
  * The bridge to bot_universal.py (TP/SL/look-ahead simulation)
  * Result persistence

Design principles
-----------------
* All configuration lives in one place (the orchestrator constructor or
  a config dict).  No config scattered across pipeline scripts.
* bot_universal.py is wrapped with guards: capped dynamic SL, no forced
  price manipulation.
* The orchestrator has two modes:
    - backtest_mode=True  → simulate on historical data, save to CSV
    - backtest_mode=False → live mode, process one bar at a time

RL upgrade path
---------------
When RL is added:
  1. Instantiate an RLPolicy and pass it to DecisionEngine(policy=RLPolicy(...)).
  2. Extend the state observation in SignalBundle.bar_context.
  3. Wire the replay buffer to orchestrator._record_transition().
  Everything else stays unchanged.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ── Project root on sys.path ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.trading.combo.ensemble.adapters import (
    AdapterRegistry,
    Binary03Adapter,
    ReversalAdapter,
)
from scripts.trading.combo.ensemble.aggregator import SignalAggregator
from scripts.trading.combo.ensemble.decision import (
    DecisionEngine,
    TradeDecision,
    WeightedScorePolicy,
)
from scripts.trading.combo.ensemble.filters import (
    FilterChain,
    FlatMarketFilter,
    LiquidityFilter,
    SessionFilter,
)
from scripts.trading.combo.ensemble.signal import SignalBundle
from scripts.trading.combo.ensemble.threshold import (
    BaseThresholdManager,
    StaticThresholdManager,
    WalkForwardThresholdManager,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExecutionResult — what the orchestrator returns per bar
# ---------------------------------------------------------------------------

class ExecutionResult:
    """
    Combines a TradeDecision with execution outcome (from bot_universal).
    """

    def __init__(
        self,
        decision: TradeDecision,
        filter_passed: bool,
        execution_result: Optional[int] = None,   # 1=TP, -1=SL, 0=time-expired
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        trade_duration: Optional[int] = None,
        pnl_pct: Optional[float] = None,
    ) -> None:
        self.decision = decision
        self.filter_passed = filter_passed
        self.execution_result = execution_result
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.trade_duration = trade_duration
        self.pnl_pct = pnl_pct

    def to_dict(self) -> Dict[str, Any]:
        d = self.decision.to_dict()
        d.update(
            {
                "filter_passed": self.filter_passed,
                "execution_result": self.execution_result,
                "entry_price": self.entry_price,
                "exit_price": self.exit_price,
                "trade_duration": self.trade_duration,
                "pnl_pct": self.pnl_pct,
                "traded": self.decision.should_trade and self.filter_passed,
            }
        )
        return d


# ---------------------------------------------------------------------------
# EnsembleOrchestrator
# ---------------------------------------------------------------------------

class EnsembleOrchestrator:
    """
    Wires all ensemble components into a single runnable pipeline.

    Parameters
    ----------
    reversal_model_path : str
        Path to the reversal XGBoost .pkl.
    binary03_model_path : str
        Path to the binary_03 combo XGBoost .pkl.
    threshold_manager : BaseThresholdManager | None
        If None, defaults to StaticThresholdManager(0.50).
    filter_chain : FilterChain | None
        If None, uses FilterChain.default_combo_chain().
    decision_policy : BaseDecisionPolicy | None
        If None, uses WeightedScorePolicy with sensible defaults.
    take_profit_pct : float
        TP percentage for bot_universal execution.
    stop_loss_pct : float
        Base SL percentage.  Dynamic SL is capped at 3× this.
    look_ahead_steps : int
        Maximum candles to look ahead in simulation.
    context_cols : list[str] | None
        Columns to carry into SignalBundle.bar_context.
    timestamp_col : str | None
        Column name for timestamp alignment.
    backtest_mode : bool
        True = vectorised backtest, False = live bar-by-bar.
    output_path : str | None
        If set, results are saved to this CSV path.
    """

    def __init__(
        self,
        reversal_model_path: str = "models/reversal/best_model_xgb.pkl",
        binary03_model_path: str = "models/combo/binary03_combo_model_xgb.pkl",
        threshold_manager: Optional[BaseThresholdManager] = None,
        filter_chain: Optional[FilterChain] = None,
        decision_policy=None,
        take_profit_pct: float = 0.006,
        stop_loss_pct: float = 0.004,
        look_ahead_steps: int = 20,
        context_cols: Optional[List[str]] = None,
        timestamp_col: Optional[str] = "timestamp",
        backtest_mode: bool = True,
        output_path: Optional[str] = None,
        reversal_gate_threshold: float = 0.5,
        diag_filters: bool = False,
    ) -> None:
        # ── Threshold manager ─────────────────────────────────────────────
        self._threshold_manager = threshold_manager or StaticThresholdManager(0.50)

        # ── Adapters & aggregator ─────────────────────────────────────────
        binary03_adapter = Binary03Adapter(
            model_path=binary03_model_path,
            weight=1.5,
            confidence_threshold=self._threshold_manager.get_threshold(),
        )
        reversal_adapter = ReversalAdapter(
            model_path=reversal_model_path,
            weight=1.0,
            gate_threshold=reversal_gate_threshold,
        )
        logger.info("ReversalAdapter gate_threshold set to %.4f", reversal_gate_threshold)

        registry = AdapterRegistry()
        registry.register(reversal_adapter)
        registry.register(binary03_adapter)

        self._registry = registry
        self._aggregator = SignalAggregator.from_registry(registry, context_cols)
        self._timestamp_col = timestamp_col

        # ── Decision engine ───────────────────────────────────────────────
        policy = decision_policy or WeightedScorePolicy(
            weights=registry.weights(),
            score_threshold=0.55,
            hard_gate_models=["reversal"],
            min_model_confidence=0.40,
        )
        self._engine = DecisionEngine(policy)

        # ── Filter chain ──────────────────────────────────────────────────
        self._filters = filter_chain or FilterChain.default_combo_chain()

        # ── Execution parameters ──────────────────────────────────────────
        self._take_profit_pct = take_profit_pct
        self._stop_loss_pct = stop_loss_pct
        self._look_ahead_steps = look_ahead_steps
        self._backtest_mode = backtest_mode
        self._output_path = output_path

        self._diag_filters = diag_filters

        logger.info(
            "EnsembleOrchestrator initialised — policy=%s, threshold=%s, filters=%s",
            policy.name,
            self._threshold_manager.__class__.__name__,
            self._filters.filter_names,
        )

    # ------------------------------------------------------------------
    # Primary execution method
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full ensemble pipeline on a feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Canonical feature DataFrame.  Must contain all features
            required by all adapters.  Must contain 'close' for execution.
            May contain 'timestamp' for alignment.

        Returns
        -------
        pd.DataFrame
            One row per input bar with:
              * all signal columns (per-model prediction + confidence)
              * decision columns (should_trade, composite_score, policy)
              * filter column (filter_passed)
              * execution columns (entry_price, exit_price, result, pnl_pct)
              * traded column (should_trade AND filter_passed)
        """
        df = df.copy().reset_index(drop=True)

        # ── Step 1: Update binary_03 threshold before scoring ─────────────
        #    Use the timestamp of the first bar as the "as_of" time,
        #    so the threshold is always calibrated from past data only.
        ts_col = self._timestamp_col
        if ts_col and ts_col in df.columns:
            first_ts = pd.Timestamp(df.iloc[0][ts_col])
        else:
            first_ts = None

        new_threshold = self._threshold_manager.get_threshold(as_of=first_ts)
        self._registry.get("binary03").confidence_threshold = new_threshold
        logger.info("Binary03 confidence threshold set to %.4f", new_threshold)

        # ── Step 2: Score all models → SignalBundles ──────────────────────
        bundles: List[SignalBundle] = self._aggregator.aggregate(
            df, timestamp_col=ts_col
        )

        # ── Step 3: Apply decision engine ─────────────────────────────────
        decisions: List[TradeDecision] = self._engine.evaluate(bundles)

        # ── Step 4: Apply filter chain ────────────────────────────────────
        # ── Step 4: Apply filter chain (con diagnostica opzionale)
        if self._diag_filters:
            filter_results, diagnostics = self._filters.apply_batch_with_diagnostics(bundles)
            self._filters.print_diagnostics(diagnostics, total_bars=len(df))
            self._print_context_column_report(df)
        else:
            filter_results = self._filters.apply_batch(bundles)

        # ── Step 5: Execution (bot_universal bridge) ──────────────────────
        if self._backtest_mode:
            exec_results = self._run_backtest_execution(
                df, decisions, filter_results
            )
        else:
            # Live mode: execution is handled externally (e.g. broker API)
            exec_results = [
                ExecutionResult(
                    decision=d,
                    filter_passed=fp,
                )
                for d, fp in zip(decisions, filter_results)
            ]

        # ── Step 6: Flatten to DataFrame ──────────────────────────────────
        result_df = pd.DataFrame([r.to_dict() for r in exec_results])

        # Preserve original OHLCV columns for inspection
        for col in ["open", "high", "low", "close", "timestamp"]:
            if col in df.columns and col not in result_df.columns:
                result_df[col] = df[col].values

        if self._output_path:
            os.makedirs(Path(self._output_path).parent, exist_ok=True)
            result_df.to_csv(self._output_path, index=False)
            logger.info("Results saved to %s", self._output_path)

        traded_count = result_df["traded"].sum() if "traded" in result_df.columns else 0
        logger.info(
            "Orchestrator run complete: %d / %d bars traded.",
            traded_count,
            len(df),
        )
        return result_df

    # ------------------------------------------------------------------
    # Execution bridge to bot_universal.py
    # ------------------------------------------------------------------

    def _run_backtest_execution(
        self,
        df: pd.DataFrame,
        decisions: List[TradeDecision],
        filter_results: List[bool],
    ) -> List[ExecutionResult]:
        """
        Simulate TP/SL/look-ahead execution using bot_universal logic,
        but only for bars where the decision + filters approved a trade.

        Improvements over raw bot_universal
        ------------------------------------
        * Dynamic SL is capped at 3× base SL (prevents unbounded widening).
        * No forced price manipulation on time-expired trades.
        * Returns structured ExecutionResult instead of mutating the DataFrame.
        """
        if "close" not in df.columns:
            raise KeyError("'close' column required for backtest execution.")

        close = df["close"].values
        vol_mean = (
            df["volatility_10"].mean()
            if "volatility_10" in df.columns
            else None
        )

        results: List[ExecutionResult] = []
        n = len(df)
        in_position = False
        entry_idx: Optional[int] = None
        entry_price: Optional[float] = None

        for i, (decision, fp) in enumerate(zip(decisions, filter_results)):
            bar_result = ExecutionResult(decision=decision, filter_passed=fp)

            if in_position:
                # Compute dynamic SL (capped)
                vol_i = (
                    df.iloc[i].get("volatility_10", vol_mean)
                    if vol_mean is not None
                    else None
                )
                if vol_i is not None and vol_mean and vol_mean > 0:
                    dyn_factor = min(float(vol_i) / float(vol_mean), 3.0)  # cap at 3×
                else:
                    dyn_factor = 1.0

                tp = entry_price * (1 + self._take_profit_pct)
                sl = entry_price * (1 - self._stop_loss_pct * dyn_factor)

                # Look ahead within bounds
                look_end = min(i + self._look_ahead_steps, n)
                closed = False
                for j in range(i, look_end):
                    p = float(close[j])
                    if p >= tp:
                        results[-1].execution_result = 1  # TP
                        results[-1].exit_price = p
                        results[-1].trade_duration = j - entry_idx
                        results[-1].pnl_pct = (p - entry_price) / entry_price * 100
                        in_position = False
                        closed = True
                        break
                    elif p <= sl:
                        results[-1].execution_result = -1  # SL
                        results[-1].exit_price = p
                        results[-1].trade_duration = j - entry_idx
                        results[-1].pnl_pct = (p - entry_price) / entry_price * 100
                        in_position = False
                        closed = True
                        break

                if not closed and (i - entry_idx) >= self._look_ahead_steps:
                    # Time-expired: use actual price, no manipulation
                    p = float(close[i])
                    results[-1].execution_result = 0  # expired
                    results[-1].exit_price = p
                    results[-1].trade_duration = i - entry_idx
                    results[-1].pnl_pct = (p - entry_price) / entry_price * 100
                    in_position = False

            elif decision.should_trade and fp:
                # Open a new position
                in_position = True
                entry_price = float(close[i])
                entry_idx = i
                bar_result.entry_price = entry_price

            results.append(bar_result)

        return results

    # ------------------------------------------------------------------
    # Filter diagnostics helper
    # ------------------------------------------------------------------

    def _print_context_column_report(self, df: pd.DataFrame) -> None:
        """Stampa un confronto tra le colonne attese dal contesto e quelle presenti nel dataset."""
        from scripts.trading.combo.ensemble.aggregator import SignalAggregator
        expected = SignalAggregator._DEFAULT_CONTEXT_COLS
        present  = set(df.columns)

        print("\n" + "=" * 65)
        print("  CONTEXT COLUMNS: attese vs presenti nel dataset")
        print("=" * 65)
        all_ok = True
        for col in expected:
            status = "OK  " if col in present else "MANCA"
            if col not in present:
                all_ok = False
            print(f"  {status}  {col}")

        extra = sorted(present - set(expected))
        if extra:
            print(f"\n  Colonne extra nel dataset (non usate dai filtri): {len(extra)}")
            for col in extra:
                print(f"        {col}")

        if not all_ok:
            missing = [c for c in expected if c not in present]
            print(f"\n  ATTENZIONE: {len(missing)} colonne mancanti.")
            print(f"  I filtri che le usano passeranno silenziosamente (graceful skip).")
            print(f"  Per abilitarli, aggiungi le colonne al dataset o a starter_combo.py.")
        else:
            print("\n  Tutte le colonne attese sono presenti.")
        print("=" * 65)

    # ------------------------------------------------------------------
    # Live (single-bar) mode
    # ------------------------------------------------------------------

    def evaluate_bar(self, row: pd.Series) -> ExecutionResult:
        """
        Score a single bar and return an ExecutionResult.

        Use this in live trading — call once per new 5-minute candle.

        Parameters
        ----------
        row : pd.Series
            A single row of the feature DataFrame, with all required columns.
        """
        df_single = row.to_frame().T.reset_index(drop=True)
        bundles = self._aggregator.aggregate(df_single, timestamp_col=self._timestamp_col)
        assert len(bundles) == 1
        bundle = bundles[0]

        # Update threshold (time-aware)
        ts = bundle.timestamp
        new_threshold = self._threshold_manager.get_threshold(as_of=ts)
        self._registry.get("binary03").confidence_threshold = new_threshold

        decision = self._engine.evaluate_single(bundle)
        fp = self._filters.apply(bundle)

        return ExecutionResult(decision=decision, filter_passed=fp)

    # ------------------------------------------------------------------
    # Component accessors (for inspection / hot-swap)
    # ------------------------------------------------------------------

    @property
    def engine(self) -> DecisionEngine:
        return self._engine

    @property
    def filter_chain(self) -> FilterChain:
        return self._filters

    @property
    def threshold_manager(self) -> BaseThresholdManager:
        return self._threshold_manager

    @threshold_manager.setter
    def threshold_manager(self, tm: BaseThresholdManager) -> None:
        self._threshold_manager = tm

    @property
    def registry(self) -> AdapterRegistry:
        return self._registry
