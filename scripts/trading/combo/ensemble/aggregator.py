"""
aggregator.py – SignalAggregator: timestamp-keyed multi-model merge.

This module REPLACES the fragile positional pd.concat in the old combo bot:

    OLD (BROKEN):
        min_len = min(len(df_reversal), len(df_binary03))
        df_combo = pd.concat([
            df_reversal.iloc[:min_len].reset_index(drop=True),
            df_binary03[["binary03_signal"]].iloc[:min_len].reset_index(drop=True)
        ], axis=1)

    NEW (SAFE):
        agg = SignalAggregator([reversal_adapter, binary03_adapter])
        bundles = agg.aggregate(df, timestamp_col="timestamp")

Key guarantees
--------------
* Alignment is done on TIMESTAMP (preferred) or INTEGER INDEX (fallback).
* If timestamps exist but don't match across adapters, an error is raised
  immediately — silent misalignment is impossible.
* Each adapter scores its own columns independently; they can have different
  feature sets.
* The aggregator accepts a SINGLE canonical DataFrame as input; adapters
  extract whatever columns they need from it.
* Returns a list of SignalBundles — one per bar — in temporal order.

RL upgrade path
---------------
The aggregator does not change when an RL policy replaces the decision engine.
The same SignalBundles feed directly into the RL policy.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .adapters import AdapterRegistry
from .signal import BaseModelAdapter, ModelSignal, SignalBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestamp alignment helpers
# ---------------------------------------------------------------------------

def _extract_timestamps(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
) -> pd.Series:
    """
    Return a Series of pd.Timestamp (same length as df), or raise if
    no timestamp source is available.
    """
    # 1. Explicit timestamp column
    if timestamp_col and timestamp_col in df.columns:
        try:
            return pd.to_datetime(df[timestamp_col])
        except Exception as exc:
            raise ValueError(
                f"Cannot parse '{timestamp_col}' column as timestamps: {exc}"
            ) from exc

    # 2. DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.to_series().reset_index(drop=True)

    # 3. No timestamp available — warn and fall back to integer index
    logger.warning(
        "No timestamp column or DatetimeIndex found. "
        "Alignment will use integer row index.  "
        "This is safe only when all adapters receive the SAME DataFrame slice."
    )
    return pd.Series(range(len(df)), name="_row_index")


def _validate_alignment(
    timestamps_a: pd.Series,
    name_a: str,
    timestamps_b: pd.Series,
    name_b: str,
) -> None:
    """
    Assert that two timestamp Series are identical.
    Raises ValueError with a detailed report on mismatch.
    """
    if len(timestamps_a) != len(timestamps_b):
        raise ValueError(
            f"Alignment error: '{name_a}' produced {len(timestamps_a)} signals "
            f"but '{name_b}' produced {len(timestamps_b)}.  "
            "Both adapters must receive the same DataFrame."
        )

    mismatches = (timestamps_a.values != timestamps_b.values).sum()
    if mismatches > 0:
        bad_idx = (timestamps_a.values != timestamps_b.values).nonzero()[0]
        sample = bad_idx[:5]
        raise ValueError(
            f"Alignment error: {mismatches} timestamp mismatches between "
            f"'{name_a}' and '{name_b}'.  "
            f"First mismatches at positions {sample.tolist()}: "
            f"{timestamps_a.iloc[sample].tolist()} vs "
            f"{timestamps_b.iloc[sample].tolist()}"
        )


# ---------------------------------------------------------------------------
# SignalAggregator
# ---------------------------------------------------------------------------

class SignalAggregator:
    """
    Collects signals from multiple model adapters and merges them into
    a list of SignalBundles — one per bar — using timestamp alignment.

    Parameters
    ----------
    adapters : list[BaseModelAdapter]
        The model wrappers to run.  Order does not affect the result.
    context_cols : list[str] | None
        Columns from the input DataFrame to include verbatim in each
        SignalBundle's `bar_context` (e.g. 'close', 'volatility_10').
        Defaults to a sensible set.

    Usage
    -----
    agg = SignalAggregator([ReversalAdapter(), Binary03Adapter()])
    bundles = agg.aggregate(df, timestamp_col="timestamp")
    """

    _DEFAULT_CONTEXT_COLS = [
        "open", "high", "low", "close",
        "volatility_10", "wick_size", "body_size",
        "rsi_14", "adx", "market_session_code",
        "is_lon_ny_overlap", "liquidity_proxy",
    ]

    def __init__(
        self,
        adapters: List[BaseModelAdapter],
        context_cols: Optional[List[str]] = None,
    ) -> None:
        if not adapters:
            raise ValueError("SignalAggregator requires at least one adapter.")
        self._adapters = adapters
        self._context_cols = context_cols or self._DEFAULT_CONTEXT_COLS

    @classmethod
    def from_registry(
        cls,
        registry: AdapterRegistry,
        context_cols: Optional[List[str]] = None,
    ) -> "SignalAggregator":
        """Convenience constructor from an AdapterRegistry."""
        return cls(registry.all(), context_cols)

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def aggregate(
        self,
        df: pd.DataFrame,
        timestamp_col: Optional[str] = "timestamp",
    ) -> List[SignalBundle]:
        """
        Score `df` through every adapter and return one SignalBundle per row.

        Parameters
        ----------
        df : pd.DataFrame
            Canonical feature DataFrame.  Must contain all features required
            by every registered adapter.  No positional assumptions.
        timestamp_col : str | None
            Column name to use for row alignment.  Pass None to force
            integer-index alignment.

        Returns
        -------
        list[SignalBundle]
            Length == len(df), in the same row order as the input.
        """
        df = df.reset_index(drop=True)   # guarantee clean 0-based index
        timestamps = _extract_timestamps(df, timestamp_col)

        # ---- Score every adapter ----------------------------------------
        all_signals: Dict[str, Tuple[pd.Series, List[ModelSignal]]] = {}

        for adapter in self._adapters:
            logger.debug("Scoring adapter: %s", adapter.name)
            signals = adapter.score_batch(df)

            if len(signals) != len(df):
                raise RuntimeError(
                    f"Adapter '{adapter.name}' returned {len(signals)} signals "
                    f"for {len(df)} input rows.  Adapter must return exactly one "
                    "signal per row."
                )

            all_signals[adapter.name] = (timestamps, signals)

        # ---- Validate cross-adapter alignment ---------------------------
        adapter_names = list(all_signals.keys())
        for i in range(1, len(adapter_names)):
            name_a = adapter_names[0]
            name_b = adapter_names[i]
            ts_a, _ = all_signals[name_a]
            ts_b, _ = all_signals[name_b]
            _validate_alignment(ts_a, name_a, ts_b, name_b)

        # ---- Build SignalBundles -----------------------------------------
        bundles: List[SignalBundle] = []
        available_ctx_cols = [c for c in self._context_cols if c in df.columns]

        for row_idx in range(len(df)):
            ts = timestamps.iloc[row_idx]
            # Convert integer "timestamp" (fallback) to None
            ts_value = ts if isinstance(ts, pd.Timestamp) else None

            bundle = SignalBundle(
                timestamp=ts_value,
                row_index=row_idx,
                bar_context={
                    col: df.iloc[row_idx][col]
                    for col in available_ctx_cols
                },
            )

            for adapter_name, (_, signals) in all_signals.items():
                bundle.signals[adapter_name] = signals[row_idx]

            bundles.append(bundle)

        logger.info(
            "SignalAggregator: produced %d bundles from %d adapters.",
            len(bundles),
            len(self._adapters),
        )
        return bundles

    # ------------------------------------------------------------------
    # Utility: materialise bundles into a DataFrame (for analysis / saving)
    # ------------------------------------------------------------------

    @staticmethod
    def to_dataframe(bundles: List[SignalBundle]) -> pd.DataFrame:
        """
        Convert a list of SignalBundles to a flat DataFrame.
        Useful for saving results or feeding into evaluation scripts.
        """
        if not bundles:
            return pd.DataFrame()
        rows = [b.to_dict() for b in bundles]
        return pd.DataFrame(rows)
