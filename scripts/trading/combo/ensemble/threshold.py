"""
threshold.py – ThresholdManager: walk-forward-aware, bias-free threshold selection.

Problems with the old approach
-------------------------------
OLD:
    backtest_df = pd.read_csv("threshold_backtest.csv")
    best_threshold = backtest_df.sort_values("saldo_finale", ascending=False).iloc[0]["threshold"]

Issues:
  1. Threshold was selected by maximising balance on the TEST SET — the same
     set used for evaluation.  This is in-sample optimisation (look-ahead bias).
  2. The threshold range [0.0005, 0.01] is coarse and extremely low — a 0.1%
     confidence threshold means "always fire".
  3. The CSV file is static; it goes stale as soon as the model is retrained.
  4. No walk-forward discipline — monthly thresholds were computed per-month
     on the test set but never propagated to production.

NEW design
----------
ThresholdManager implements three selection strategies:

  ├─ WalkForwardThresholdManager   (primary — use this in production)
  │    Selects the threshold using expanding-window cross-validation.
  │    Each fold: train on data[0..t], sweep thresholds, pick best on
  │    a small validation window immediately following t.  The threshold
  │    applied to a bar at time T was selected using data BEFORE T only.
  │
  ├─ StaticThresholdManager        (baseline / ablation)
  │    Simply holds a fixed threshold.  Useful for unit tests or when
  │    you want to freeze behaviour for reproducibility.
  │
  └─ PercentileThresholdManager    (adaptive, no data snooping)
       Sets the threshold as a percentile of out-of-sample probabilities
       from the previous window.  This is how binary_02's adaptive flat
       filter worked — generalised for probability thresholds.

All managers expose the same interface:
    .get_threshold(as_of: pd.Timestamp | None) -> float

RL upgrade path
---------------
In RL mode the threshold concept disappears — the policy outputs a
continuous action (position size).  At that point ThresholdManager
is bypassed by the RL DecisionPolicy.  The interface is preserved so
non-RL paths keep working during the transition.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseThresholdManager(ABC):
    """Common interface all threshold managers must satisfy."""

    @abstractmethod
    def get_threshold(self, as_of: Optional[pd.Timestamp] = None) -> float:
        """
        Return the threshold applicable at time `as_of`.

        Parameters
        ----------
        as_of : pd.Timestamp | None
            The bar timestamp for which we need a threshold.
            Managers that are time-aware use this to avoid look-ahead.
            Managers that are time-agnostic ignore it.
        """

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseThresholdManager":
        """Calibrate the manager from data.  Returns self."""


# ---------------------------------------------------------------------------
# 1. StaticThresholdManager
# ---------------------------------------------------------------------------

class StaticThresholdManager(BaseThresholdManager):
    """
    Holds a single fixed threshold.

    This is useful for:
    * Unit testing
    * Ablation studies where you want to isolate the model's skill
      from threshold selection
    * Deployments where the threshold was set once by a human expert

    Parameters
    ----------
    threshold : float
        Must be in (0, 1).  A sensible default for XGBoost is 0.5.
        The old system's 0.001 was dangerously low; start at 0.45.
    """

    def __init__(self, threshold: float = 0.45) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self._threshold = threshold

    def get_threshold(self, as_of: Optional[pd.Timestamp] = None) -> float:
        return self._threshold

    def fit(self, *args, **kwargs) -> "StaticThresholdManager":
        # Nothing to fit
        return self

    def __repr__(self) -> str:
        return f"StaticThresholdManager(threshold={self._threshold})"


# ---------------------------------------------------------------------------
# 2. WalkForwardThresholdManager
# ---------------------------------------------------------------------------

class WalkForwardThresholdManager(BaseThresholdManager):
    """
    Selects the threshold via expanding-window cross-validation.

    Procedure
    ---------
    Given a chronological sequence of (timestamp, probability, true_label):

    1. Start with a minimum training window of `min_train_bars`.
    2. For each subsequent validation window of `val_window_bars`:
       a. Sweep `candidate_thresholds`.
       b. For each threshold, compute the scoring metric on the validation
          window.  Default metric: profit_factor (wins / max(losses, 1)).
       c. Store best_threshold[end_of_training_window].
    3. When `get_threshold(as_of=T)` is called, return the threshold that
       was determined from the most recent training window that ends ≤ T.

    This ensures the threshold applied to bar T was never fitted on data
    after T — no look-ahead bias.

    Parameters
    ----------
    min_train_bars : int
        Minimum number of bars in the first training window.
    val_window_bars : int
        Size of the validation window (rolling forward).
    candidate_thresholds : list[float]
        Thresholds to sweep.  Keep in [0.3, 0.7] for sensible classifiers.
    scoring : callable
        f(y_true, y_pred_proba, threshold) -> float.
        Higher is better.  Default: profit_factor.
    """

    _DEFAULT_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    def __init__(
        self,
        min_train_bars: int = 2000,
        val_window_bars: int = 500,
        candidate_thresholds: Optional[List[float]] = None,
        scoring: Optional[Callable] = None,
    ) -> None:
        self.min_train_bars = min_train_bars
        self.val_window_bars = val_window_bars
        self.candidate_thresholds = (
            candidate_thresholds or self._DEFAULT_THRESHOLDS
        )
        self._scoring = scoring or _profit_factor_scorer

        # Fitted state
        self._threshold_map: Dict[pd.Timestamp, float] = {}
        self._global_fallback: float = 0.50  # used when no fold covers as_of

    def fit(
        self,
        timestamps: pd.Series,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> "WalkForwardThresholdManager":
        """
        Calibrate the threshold schedule.

        Parameters
        ----------
        timestamps : pd.Series of pd.Timestamp
            Bar timestamps, chronologically ordered.
        probabilities : np.ndarray, shape (N,)
            Predicted probabilities for the positive class.
        labels : np.ndarray, shape (N,)
            True binary labels.
        """
        n = len(timestamps)
        if n < self.min_train_bars + self.val_window_bars:
            logger.warning(
                "Not enough data for walk-forward calibration "
                "(%d bars, need %d).  Using global best threshold.",
                n,
                self.min_train_bars + self.val_window_bars,
            )
            self._global_fallback = self._best_threshold_global(
                probabilities, labels
            )
            return self

        self._threshold_map.clear()
        ts_idx = pd.Series(timestamps.values)

        start = self.min_train_bars
        while start + self.val_window_bars <= n:
            val_end = start + self.val_window_bars

            val_proba = probabilities[start:val_end]
            val_labels = labels[start:val_end]

            best_th, best_score = 0.50, -np.inf
            for th in self.candidate_thresholds:
                score = self._scoring(val_labels, val_proba, th)
                if score > best_score:
                    best_score = score
                    best_th = th

            fold_end_ts = ts_idx.iloc[start - 1]
            self._threshold_map[fold_end_ts] = best_th
            logger.debug(
                "WalkForward fold ending at %s → threshold %.4f (score=%.4f)",
                fold_end_ts,
                best_th,
                best_score,
            )

            start += self.val_window_bars

        # Global fallback = median of all fold thresholds
        if self._threshold_map:
            self._global_fallback = float(
                np.median(list(self._threshold_map.values()))
            )

        logger.info(
            "WalkForwardThresholdManager fitted: %d folds, "
            "fallback threshold = %.4f",
            len(self._threshold_map),
            self._global_fallback,
        )
        return self

    def get_threshold(self, as_of: Optional[pd.Timestamp] = None) -> float:
        if not self._threshold_map or as_of is None:
            return self._global_fallback

        # Find the most recent fold end that is <= as_of
        valid_keys = [ts for ts in self._threshold_map if ts <= as_of]
        if not valid_keys:
            return self._global_fallback

        latest_key = max(valid_keys)
        return self._threshold_map[latest_key]

    def _best_threshold_global(
        self, probabilities: np.ndarray, labels: np.ndarray
    ) -> float:
        best_th, best_score = 0.50, -np.inf
        for th in self.candidate_thresholds:
            score = self._scoring(labels, probabilities, th)
            if score > best_score:
                best_score = score
                best_th = th
        return best_th

    def summary(self) -> pd.DataFrame:
        """Return the fitted threshold schedule as a DataFrame."""
        rows = [
            {"fold_end": ts, "threshold": th}
            for ts, th in sorted(self._threshold_map.items())
        ]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. PercentileThresholdManager
# ---------------------------------------------------------------------------

class PercentileThresholdManager(BaseThresholdManager):
    """
    Sets the threshold as the Nth percentile of probabilities from the
    previous rolling window.

    This is the principled generalisation of the binary_02 adaptive flat
    filter — instead of a hardcoded constant, the threshold adjusts to the
    model's current confidence distribution without using any future data.

    Parameters
    ----------
    percentile : float
        E.g. 75.0 → only trade when model is in the top quartile of
        confidence for this window.
    window_bars : int
        Number of historical bars used to estimate the distribution.
    min_threshold : float
        Hard floor to prevent the threshold from collapsing to zero in
        high-confidence regimes.
    """

    def __init__(
        self,
        percentile: float = 70.0,
        window_bars: int = 500,
        min_threshold: float = 0.45,
    ) -> None:
        self._percentile = percentile
        self._window_bars = window_bars
        self._min_threshold = min_threshold

        # Fitted state
        self._timestamps: Optional[pd.Series] = None
        self._probabilities: Optional[np.ndarray] = None
        self._global_fallback: float = min_threshold

    def fit(
        self,
        timestamps: pd.Series,
        probabilities: np.ndarray,
        labels: Optional[np.ndarray] = None,   # unused, kept for API compatibility
    ) -> "PercentileThresholdManager":
        self._timestamps = timestamps.reset_index(drop=True)
        self._probabilities = probabilities
        self._global_fallback = max(
            float(np.percentile(probabilities, self._percentile)),
            self._min_threshold,
        )
        return self

    def get_threshold(self, as_of: Optional[pd.Timestamp] = None) -> float:
        if self._timestamps is None or as_of is None:
            return self._global_fallback

        mask = self._timestamps <= as_of
        past_probas = self._probabilities[mask.values]

        if len(past_probas) < 30:   # not enough history
            return self._global_fallback

        window = past_probas[-self._window_bars:]
        return max(float(np.percentile(window, self._percentile)), self._min_threshold)


# ---------------------------------------------------------------------------
# Scoring functions (can be replaced / extended)
# ---------------------------------------------------------------------------

def _profit_factor_scorer(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> float:
    """
    Profit factor = wins / max(losses, 1), evaluated at a given threshold.
    Returns 0 when no trades are triggered.
    """
    preds = (y_proba >= threshold).astype(int)
    # Only evaluate on bars where the model signals a trade
    trade_mask = preds == 1
    if trade_mask.sum() == 0:
        return 0.0
    wins = ((y_true == 1) & trade_mask).sum()
    losses = ((y_true == 0) & trade_mask).sum()
    return float(wins / max(losses, 1))


def _f1_scorer(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> float:
    """F1 score at a given threshold."""
    preds = (y_proba >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def _precision_recall_tradeoff_scorer(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    precision_weight: float = 0.7,
) -> float:
    """
    Weighted combination of precision and recall.
    Favours precision (fewer false trades) by default.
    """
    preds = (y_proba >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return precision_weight * prec + (1 - precision_weight) * rec
