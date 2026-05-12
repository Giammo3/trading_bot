"""
adapters.py – Concrete model adapters for the combo ensemble.

Each adapter wraps ONE trained model, loads it once at construction,
handles its own feature selection, and produces a list of ModelSignals.

Design principles
-----------------
* Feature alignment is done by COLUMN NAME, never by position.
* No CSV files are read here — data is passed in by the orchestrator.
* Adding a new model = adding one new Adapter class.  Nothing else changes.

RL upgrade path
---------------
An RL policy adapter would subclass BaseModelAdapter exactly like these do.
It would receive the same `df` slice and produce ModelSignals.  The rest of
the pipeline (aggregator, decision engine, filters) is unchanged.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd

from .signal import BaseModelAdapter, ModelSignal


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_model(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p.resolve()}")
    return joblib.load(p)


# ---------------------------------------------------------------------------
# 1. ReversalAdapter
# ---------------------------------------------------------------------------

class ReversalAdapter(BaseModelAdapter):
    """
    Wraps the XGBoost reversal model (best_model_xgb.pkl).

    Produces binary predictions directly (no probability threshold — the
    reversal model acts as a hard gate: it must fire 1 for a trade to
    be considered).

    Confidence
    ----------
    We use predict_proba[:, 1] as the confidence score.  This lets the
    DecisionEngine use reversal strength as a weighting factor even though
    the gate itself is hard.
    """

    _DEFAULT_PATH = "models/reversal/best_model_xgb.pkl"

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_PATH,
        weight: float = 1.0,
        gate_threshold: float = 0.5,
    ):
        self._model = _load_model(model_path)
        self._weight = weight
        self._gate_threshold = gate_threshold
        self._features: Optional[List[str]] = self._resolve_features(self._model)

    @property
    def gate_threshold(self) -> float:
        return self._gate_threshold

    @gate_threshold.setter
    def gate_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"gate_threshold must be in [0, 1], got {value}")
        self._gate_threshold = value

    @property
    def name(self) -> str:
        return "reversal"

    @property
    def weight(self) -> float:
        return self._weight

    def score_batch(self, df: pd.DataFrame) -> List[ModelSignal]:
        """
        Score all rows in `df`.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all features the reversal model was trained on.
            May optionally contain a 'timestamp' column.
        """
        if self._features is None:
            raise RuntimeError(
                "Reversal model has no stored feature list. "
                "Re-train with XGBoost >= 1.3 or set feature_names manually."
            )

        missing = set(self._features) - set(df.columns)
        if missing:
            raise KeyError(
                f"[{self.name}] Missing features in input data: {missing}"
            )

        X = df[self._features].copy()

        # Vectorised prediction — threshold configurabile via gate_threshold
        proba = self._safe_proba(self._model, X)   # shape (N, 2)
        hard_preds = (proba[:, 1] >= self._gate_threshold).astype(int)

        signals: List[ModelSignal] = []
        for i, row in enumerate(df.itertuples()):
            ts = None
            if "timestamp" in df.columns:
                try:
                    ts = pd.Timestamp(df.iloc[i]["timestamp"])
                except Exception:
                    pass
            elif isinstance(df.index, pd.DatetimeIndex):
                ts = df.index[i]

            signals.append(
                ModelSignal(
                    model_name=self.name,
                    timestamp=ts,
                    prediction=int(hard_preds[i]),
                    confidence=float(proba[i, 1]),
                    raw_proba=proba[i],
                )
            )

        return signals


# ---------------------------------------------------------------------------
# 2. Binary03Adapter
# ---------------------------------------------------------------------------

class Binary03Adapter(BaseModelAdapter):
    """
    Wraps the XGBoost binary_03 combo model (binary03_combo_model_xgb.pkl).

    Unlike the reversal adapter, this model uses a *tunable probability
    threshold* managed externally by the ThresholdManager.  The adapter
    always returns the raw probability; thresholding happens in the
    DecisionEngine so it can be updated at runtime without reloading.

    The adapter explicitly blocks look-ahead columns (`future_return`,
    `future_close`, `target_binary_03`) from being fed to the model —
    even if they are present in the input DataFrame.
    """

    _DEFAULT_PATH = "models/combo/binary03_combo_model_xgb.pkl"

    # Columns that must NEVER be fed to the model
    _FORBIDDEN = frozenset(
        ["future_return", "future_return_pct", "future_close", "target_binary_03",
         "reversal", "reversal_real", "target_reversal", "target_binary_02"]
    )

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_PATH,
        weight: float = 1.5,
        confidence_threshold: float = 0.5,
    ):
        self._model = _load_model(model_path)
        self._weight = weight
        self._confidence_threshold = confidence_threshold   # default, overridden by ThresholdManager

        raw_features = self._resolve_features(self._model)
        if raw_features is None:
            raise RuntimeError(
                "Binary03 model has no stored feature list. "
                "Re-train with XGBoost >= 1.3."
            )
        # Strip forbidden columns from whatever the model recorded
        self._features: List[str] = [
            f for f in raw_features if f not in self._FORBIDDEN
        ]
        if len(self._features) != len(raw_features):
            removed = set(raw_features) - set(self._features)
            warnings.warn(
                f"[{self.name}] Removed look-ahead / label columns from feature list: "
                f"{removed}"
            )

    @property
    def name(self) -> str:
        return "binary03"

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {value}")
        self._confidence_threshold = value

    def score_batch(self, df: pd.DataFrame) -> List[ModelSignal]:
        """
        Score all rows in `df`.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all features the binary_03 model was trained on
            (minus forbidden look-ahead columns).
            May optionally contain a 'timestamp' column.

        Notes
        -----
        The `prediction` in the returned ModelSignal is based on
        `self._confidence_threshold` (set by ThresholdManager), not on the
        model's default 0.5 cutoff.  The raw `confidence` (proba[:, 1]) is
        always included so the DecisionEngine can re-threshold if needed.
        """
        missing = set(self._features) - set(df.columns)
        if missing:
            raise KeyError(
                f"[{self.name}] Missing features in input data: {missing}"
            )

        X = df[self._features].copy()
        proba = self._safe_proba(self._model, X)   # shape (N, 2)
        hard_preds = (proba[:, 1] >= self._confidence_threshold).astype(int)

        signals: List[ModelSignal] = []
        for i in range(len(df)):
            ts = None
            if "timestamp" in df.columns:
                try:
                    ts = pd.Timestamp(df.iloc[i]["timestamp"])
                except Exception:
                    pass
            elif isinstance(df.index, pd.DatetimeIndex):
                ts = df.index[i]

            signals.append(
                ModelSignal(
                    model_name=self.name,
                    timestamp=ts,
                    prediction=int(hard_preds[i]),
                    confidence=float(proba[i, 1]),
                    raw_proba=proba[i],
                    metadata={"threshold_used": self._confidence_threshold},
                )
            )

        return signals


# ---------------------------------------------------------------------------
# 3. AdapterRegistry — lets the orchestrator discover adapters by name
# ---------------------------------------------------------------------------

class AdapterRegistry:
    """
    Simple registry mapping name → adapter instance.

    Usage
    -----
    registry = AdapterRegistry()
    registry.register(ReversalAdapter())
    registry.register(Binary03Adapter())
    adapter = registry.get("reversal")
    """

    def __init__(self) -> None:
        self._adapters: dict[str, BaseModelAdapter] = {}

    def register(self, adapter: BaseModelAdapter) -> None:
        if adapter.name in self._adapters:
            warnings.warn(
                f"AdapterRegistry: overwriting existing adapter '{adapter.name}'"
            )
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> BaseModelAdapter:
        if name not in self._adapters:
            raise KeyError(f"No adapter registered under name '{name}'")
        return self._adapters[name]

    def all(self) -> list[BaseModelAdapter]:
        return list(self._adapters.values())

    def names(self) -> list[str]:
        return list(self._adapters.keys())

    def weights(self) -> dict[str, float]:
        return {name: a.weight for name, a in self._adapters.items()}
