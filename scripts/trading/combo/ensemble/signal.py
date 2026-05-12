"""
signal.py – Core data contracts for the ensemble system.

Design goals
------------
* Every model produces exactly one ModelSignal per row.
* A SignalBundle collects all signals for a single timestamp, giving
  downstream components a clean, typed surface to reason about.
* BaseModelAdapter enforces a consistent contract for all model wrappers,
  making it trivial to add / swap models without touching the aggregator.

RL upgrade path
---------------
ModelSignal already carries a `confidence` (float in [0, 1]) alongside
the binary `prediction`.  When an RL policy replaces the DecisionEngine,
it will consume the same SignalBundle — no schema changes needed.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. ModelSignal — the atomic output of a single model on a single row
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSignal:
    """
    Immutable prediction record produced by one model for one bar.

    Attributes
    ----------
    model_name : str
        Unique identifier for the model that generated this signal.
    timestamp : pd.Timestamp | None
        Bar timestamp.  None is allowed when working with index-only data,
        but the aggregator will raise if timestamps are inconsistent.
    prediction : int
        Binary prediction: 1 = signal, 0 = no signal.
    confidence : float
        Probability of the positive class in [0.0, 1.0].
        For hard classifiers, set confidence = float(prediction).
    raw_proba : np.ndarray | None
        Full probability vector [P(0), P(1)] if available.
    metadata : dict
        Optional bag for anything model-specific (e.g. feature importances,
        volatility regime, session tag).
    """

    model_name: str
    timestamp: Optional[pd.Timestamp]
    prediction: int
    confidence: float
    raw_proba: Optional[np.ndarray] = field(default=None, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ModelSignal.confidence must be in [0, 1], got {self.confidence}"
            )
        if self.prediction not in (0, 1):
            raise ValueError(
                f"ModelSignal.prediction must be 0 or 1, got {self.prediction}"
            )

    @property
    def is_positive(self) -> bool:
        return self.prediction == 1


# ---------------------------------------------------------------------------
# 2. SignalBundle — all signals for a single bar, keyed by model name
# ---------------------------------------------------------------------------

@dataclass
class SignalBundle:
    """
    Container for all ModelSignals produced for one bar.

    The aggregator constructs one SignalBundle per bar by merging signals
    from all adapters on a shared timestamp / index key — never by position.

    Attributes
    ----------
    timestamp : pd.Timestamp | None
        Shared bar timestamp (validated at construction time).
    row_index : int | None
        Integer position in the original DataFrame (useful for debugging).
    signals : dict[str, ModelSignal]
        Mapping model_name → ModelSignal.
    bar_context : dict
        Raw feature values for this bar (close, volatility_10, etc.).
        Used by filters and the RL policy — never by models (they score
        their own data).
    """

    timestamp: Optional[pd.Timestamp]
    row_index: Optional[int]
    signals: Dict[str, ModelSignal] = field(default_factory=dict)
    bar_context: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get(self, model_name: str) -> Optional[ModelSignal]:
        return self.signals.get(model_name)

    def all_positive(self, model_names: Optional[List[str]] = None) -> bool:
        """Return True only when every requested model fired a positive signal."""
        names = model_names or list(self.signals.keys())
        return all(
            self.signals[n].is_positive
            for n in names
            if n in self.signals
        )

    def any_positive(self, model_names: Optional[List[str]] = None) -> bool:
        names = model_names or list(self.signals.keys())
        return any(
            self.signals[n].is_positive
            for n in names
            if n in self.signals
        )

    def weighted_confidence(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Weighted average of confidence scores across all models.

        Parameters
        ----------
        weights : dict | None
            {model_name: weight}.  Defaults to equal weighting.
            Weights are normalised internally, so they don't need to sum to 1.
        """
        if not self.signals:
            return 0.0

        names = list(self.signals.keys())
        if weights is None:
            w = np.ones(len(names))
        else:
            w = np.array([weights.get(n, 1.0) for n in names])

        w = w / w.sum()
        confidences = np.array([self.signals[n].confidence for n in names])
        return float(np.dot(w, confidences))

    def to_dict(self) -> Dict[str, Any]:
        """Flatten bundle to a plain dict for DataFrame construction."""
        out: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "row_index": self.row_index,
        }
        for name, sig in self.signals.items():
            out[f"{name}__prediction"] = sig.prediction
            out[f"{name}__confidence"] = sig.confidence
        out["weighted_confidence"] = self.weighted_confidence()
        out.update(self.bar_context)
        return out

    @property
    def model_names(self) -> List[str]:
        return list(self.signals.keys())


# ---------------------------------------------------------------------------
# 3. BaseModelAdapter — contract every model wrapper must satisfy
# ---------------------------------------------------------------------------

class BaseModelAdapter(ABC):
    """
    Abstract base for model wrappers used by the SignalAggregator.

    Subclasses must implement `score_batch()`.  All other helpers are
    provided here so adapters stay thin.

    Contract
    --------
    * `score_batch` receives a DataFrame whose index is already aligned
      (either by timestamp or integer position) and returns a list of
      ModelSignals in the SAME ORDER as the input rows.
    * Adapters must NOT modify the input DataFrame.
    * Adapters are responsible for selecting / reordering their own columns.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier (must be stable across runs)."""

    @property
    @abstractmethod
    def weight(self) -> float:
        """
        Relative importance weight in [0, ∞).
        Used by SignalBundle.weighted_confidence() if no explicit weights
        are provided.
        """

    @abstractmethod
    def score_batch(self, df: pd.DataFrame) -> List[ModelSignal]:
        """
        Score every row in `df` and return one ModelSignal per row.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a column or index that allows timestamp extraction.

        Returns
        -------
        list[ModelSignal]
            Length == len(df), ordered identically.
        """

    # ------------------------------------------------------------------
    # Non-abstract helpers available to all subclasses
    # ------------------------------------------------------------------

    def _extract_timestamp(
        self, row: pd.Series, index: Any
    ) -> Optional[pd.Timestamp]:
        """
        Try to get a pd.Timestamp from the row or its index.
        Returns None gracefully if no timestamp is available.
        """
        if "timestamp" in row.index:
            try:
                return pd.Timestamp(row["timestamp"])
            except Exception:
                pass
        if isinstance(index, pd.Timestamp):
            return index
        return None

    def _safe_proba(
        self, model: Any, X: pd.DataFrame
    ) -> np.ndarray:
        """
        Return predict_proba output, falling back to hard predictions.
        Shape: (n_rows, 2)
        """
        if hasattr(model, "predict_proba"):
            try:
                return model.predict_proba(X)
            except Exception as exc:
                warnings.warn(
                    f"[{self.name}] predict_proba failed ({exc}); "
                    "falling back to hard predictions."
                )
        preds = model.predict(X).astype(float)
        return np.column_stack([1 - preds, preds])

    def _resolve_features(self, model: Any) -> Optional[List[str]]:
        """
        Try to extract the feature list the model was trained on.
        Works for XGBoost (get_booster().feature_names) and sklearn pipelines.
        """
        # XGBoost
        if hasattr(model, "get_booster"):
            return model.get_booster().feature_names
        # sklearn with named steps (pipeline)
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        return None
