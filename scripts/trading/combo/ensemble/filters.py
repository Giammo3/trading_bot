"""
filters.py – Modular, composable trade filter chain.

Problems with the old approach
-------------------------------
OLD:
    # reversal flat_filter.py (hardcoded magic numbers)
    flat_mask = (df["volatility_10"] < 0.00055) & (df["wick_size"] < 0.0007)

    # binary_02 flat_filter.py (adaptive, but on the live test set — data snooping)
    vol_threshold = df["volatility_10"].quantile(0.25)  # quantile of the TEST set

Issues:
  1. Hardcoded thresholds go stale as market regimes change.
  2. Adaptive thresholds computed on the live window introduce look-ahead.
  3. Session/liquidity filters scattered across entry_filters_reversal.py,
     entry_filters_auto.py, and bot_universal.py without a shared interface.
  4. Filter order matters but is implicit — changing execution order changes
     which trades get filtered.
  5. No way to disable a filter for A/B testing without modifying the code.

NEW design
----------
Each filter is an independent class with a clear interface:
    .apply(bundle: SignalBundle) -> bool    (True = PASS, False = REJECT)

Filters are composed into a FilterChain:
    chain = FilterChain([FlatMarketFilter(), SessionFilter(), LiquidityFilter()])
    passed = chain.apply(bundle)

Each filter:
* Reads from SignalBundle.bar_context — no raw DataFrame dependency.
* Uses thresholds calibrated from HISTORICAL data only (passed at construction).
* Is individually toggleable via `enabled=False`.
* Records its rejection reason so the orchestrator can log it.

RL upgrade path
---------------
In RL mode, filters can be presented as constraints rather than hard gates.
The RL policy receives the full SignalBundle + filter scores as part of the
state observation and learns to weight them as soft penalties.  For now they
remain hard filters.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .signal import SignalBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FilterResult — what a filter returns
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterResult:
    passed: bool
    filter_name: str
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------------
# BaseFilter
# ---------------------------------------------------------------------------

class BaseFilter(ABC):
    """Common interface every filter must satisfy."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique filter identifier."""

    @abstractmethod
    def _check(self, bundle: SignalBundle) -> FilterResult:
        """Perform the actual check.  Called only when enabled=True."""

    def apply(self, bundle: SignalBundle) -> FilterResult:
        if not self.enabled:
            return FilterResult(passed=True, filter_name=self.name, reason="disabled")
        result = self._check(bundle)
        if not result.passed:
            logger.debug(
                "Filter REJECT [%s] at %s: %s",
                self.name,
                bundle.timestamp,
                result.reason,
            )
        return result


# ---------------------------------------------------------------------------
# Filter 1: FlatMarketFilter
# ---------------------------------------------------------------------------

class FlatMarketFilter(BaseFilter):
    """
    Rejects bars where the market is too flat to trade.

    Replaces the hardcoded `vol < 0.00055 AND wick < 0.0007` in the old
    reversal/combo pipeline.

    Thresholds are calibrated from historical data:
    * volatility_threshold: any percentile of the training set's volatility_10.
    * wick_threshold: any percentile of the training set's wick_size.

    Crucially, these are FIXED at construction time — they never recalculate
    from the live test window (which was the binary_02 data-snooping issue).

    Parameters
    ----------
    volatility_threshold : float
        Bars with volatility_10 BELOW this are considered flat.
    wick_threshold : float
        Bars with wick_size BELOW this are considered flat.
    require_both : bool
        True → reject only when BOTH conditions are met (original logic).
        False → reject when EITHER condition is met (stricter).
    """

    def __init__(
        self,
        volatility_threshold: float = 0.00055,
        wick_threshold: float = 0.0007,
        require_both: bool = True,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled)
        self.volatility_threshold = volatility_threshold
        self.wick_threshold = wick_threshold
        self.require_both = require_both

    @property
    def name(self) -> str:
        return "FlatMarketFilter"

    @classmethod
    def from_historical(
        cls,
        volatility_series,
        wick_series,
        vol_percentile: float = 25.0,
        wick_percentile: float = 25.0,
        require_both: bool = True,
    ) -> "FlatMarketFilter":
        """
        Calibrate thresholds from historical (training) data.

        Parameters
        ----------
        volatility_series, wick_series : array-like
            Training set values of volatility_10 and wick_size.
        vol_percentile, wick_percentile : float
            Percentile to use as the flat-market cutoff.
        """
        vol_thresh = float(np.percentile(volatility_series, vol_percentile))
        wick_thresh = float(np.percentile(wick_series, wick_percentile))
        return cls(
            volatility_threshold=vol_thresh,
            wick_threshold=wick_thresh,
            require_both=require_both,
        )

    def _check(self, bundle: SignalBundle) -> FilterResult:
        ctx = bundle.bar_context
        vol = ctx.get("volatility_10")
        wick = ctx.get("wick_size")

        if vol is None or wick is None:
            return FilterResult(
                passed=True,
                filter_name=self.name,
                reason="missing context columns — filter skipped",
            )

        vol_flat = float(vol) < self.volatility_threshold
        wick_flat = float(wick) < self.wick_threshold

        if self.require_both:
            is_flat = vol_flat and wick_flat
        else:
            is_flat = vol_flat or wick_flat

        if is_flat:
            return FilterResult(
                passed=False,
                filter_name=self.name,
                reason=(
                    f"flat market: vol={vol:.6f} < {self.volatility_threshold}, "
                    f"wick={wick:.6f} < {self.wick_threshold}"
                ),
            )
        return FilterResult(passed=True, filter_name=self.name)


# ---------------------------------------------------------------------------
# Filter 2: SessionFilter
# ---------------------------------------------------------------------------

class SessionFilter(BaseFilter):
    """
    Restricts trading to preferred market sessions.

    Old code spread session logic across entry_filters_reversal.py,
    entry_filters_auto.py, and feature engineering — now centralised here.

    Parameters
    ----------
    allowed_session_codes : list[int]
        Market session codes matching utils/feature_engineering.py:
          0 = Asia, 1 = London_Open, 2 = London_Morning,
          3 = LON_NY_Overlap, 4 = NY, 5 = Off_Hours
    require_lon_ny_overlap : bool
        If True, additionally require is_lon_ny_overlap == 1.
    """

    # Default: London Morning, LON/NY Overlap, NY (skip Asia + Off_Hours)
    _DEFAULT_SESSIONS = [2, 3, 4]

    def __init__(
        self,
        allowed_session_codes: Optional[List[int]] = None,
        require_lon_ny_overlap: bool = False,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled)
        self.allowed_session_codes = allowed_session_codes or self._DEFAULT_SESSIONS
        self.require_lon_ny_overlap = require_lon_ny_overlap

    @property
    def name(self) -> str:
        return "SessionFilter"

    def _check(self, bundle: SignalBundle) -> FilterResult:
        ctx = bundle.bar_context
        session_code = ctx.get("market_session_code")
        lon_ny = ctx.get("is_lon_ny_overlap")

        # If session info is absent, let it pass (graceful degradation)
        if session_code is None:
            return FilterResult(passed=True, filter_name=self.name,
                                reason="no session info — filter skipped")

        if int(session_code) not in self.allowed_session_codes:
            return FilterResult(
                passed=False,
                filter_name=self.name,
                reason=f"session code {session_code} not in allowed {self.allowed_session_codes}",
            )

        if self.require_lon_ny_overlap and lon_ny is not None and int(lon_ny) != 1:
            return FilterResult(
                passed=False,
                filter_name=self.name,
                reason="require_lon_ny_overlap=True but is_lon_ny_overlap=0",
            )

        return FilterResult(passed=True, filter_name=self.name)


# ---------------------------------------------------------------------------
# Filter 3: LiquidityFilter
# ---------------------------------------------------------------------------

class LiquidityFilter(BaseFilter):
    """
    Rejects bars where the liquidity_proxy is too low.

    The liquidity_proxy feature from utils/feature_engineering.py combines
    wick/body ratio, session weight, and inverse volatility into a single
    score.  This filter guards against illiquid conditions without requiring
    separate checks for each component.

    Parameters
    ----------
    min_liquidity : float
        Minimum liquidity_proxy value to allow a trade.
        Calibrate via from_historical() using training-set data.
    """

    def __init__(self, min_liquidity: float = 0.6, enabled: bool = True) -> None:
        super().__init__(enabled)
        self.min_liquidity = min_liquidity

    @property
    def name(self) -> str:
        return "LiquidityFilter"

    @classmethod
    def from_historical(
        cls,
        liquidity_series,
        percentile: float = 25.0,
    ) -> "LiquidityFilter":
        """Calibrate min_liquidity from training data at the given percentile."""
        return cls(min_liquidity=float(np.percentile(liquidity_series, percentile)))

    def _check(self, bundle: SignalBundle) -> FilterResult:
        liq = bundle.bar_context.get("liquidity_proxy")
        if liq is None:
            return FilterResult(passed=True, filter_name=self.name,
                                reason="liquidity_proxy missing — filter skipped")

        if float(liq) < self.min_liquidity:
            return FilterResult(
                passed=False,
                filter_name=self.name,
                reason=f"liquidity_proxy={liq:.4f} < {self.min_liquidity}",
            )
        return FilterResult(passed=True, filter_name=self.name)


# ---------------------------------------------------------------------------
# Filter 4: MinConfidenceFilter
# ---------------------------------------------------------------------------

class MinConfidenceFilter(BaseFilter):
    """
    Rejects bundles where the composite confidence is below a floor.

    This is a post-decision-engine guard: even if the policy said "trade",
    this filter can veto if confidence is too low.

    Parameters
    ----------
    min_composite_confidence : float
        Minimum weighted confidence from SignalBundle.weighted_confidence().
    """

    def __init__(
        self, min_composite_confidence: float = 0.50, enabled: bool = True
    ) -> None:
        super().__init__(enabled)
        self.min_composite_confidence = min_composite_confidence

    @property
    def name(self) -> str:
        return "MinConfidenceFilter"

    def _check(self, bundle: SignalBundle) -> FilterResult:
        conf = bundle.weighted_confidence()
        if conf < self.min_composite_confidence:
            return FilterResult(
                passed=False,
                filter_name=self.name,
                reason=f"composite confidence {conf:.4f} < {self.min_composite_confidence}",
            )
        return FilterResult(passed=True, filter_name=self.name)


# ---------------------------------------------------------------------------
# FilterChain
# ---------------------------------------------------------------------------

class FilterChain:
    """
    Applies a sequence of filters to a SignalBundle.

    All filters must pass (logical AND) for the trade to be approved.
    Returns on the first rejection for efficiency, but records all results
    if `short_circuit=False`.

    Parameters
    ----------
    filters : list[BaseFilter]
        Applied in order.  Order matters: cheap filters should come first.
    short_circuit : bool
        If True (default), stop at the first rejection.
        If False, run all filters and collect all results.

    Usage
    -----
    chain = FilterChain([
        FlatMarketFilter(volatility_threshold=0.0004, wick_threshold=0.0006),
        SessionFilter(allowed_session_codes=[2, 3, 4]),
        LiquidityFilter(min_liquidity=0.5),
    ])
    if chain.apply(bundle):
        # trade passes all filters
    """

    def __init__(
        self,
        filters: Optional[List[BaseFilter]] = None,
        short_circuit: bool = True,
    ) -> None:
        self._filters = filters or []
        self._short_circuit = short_circuit

    def add(self, f: BaseFilter) -> "FilterChain":
        self._filters.append(f)
        return self

    def apply(self, bundle: SignalBundle) -> bool:
        """Return True if all filters pass, False otherwise."""
        for f in self._filters:
            result = f.apply(bundle)
            if not result.passed:
                return False
            if self._short_circuit:
                continue
        return True

    def apply_with_results(self, bundle: SignalBundle) -> List[FilterResult]:
        """Run all filters and return the full list of results."""
        results = []
        for f in self._filters:
            result = f.apply(bundle)
            results.append(result)
        return results

    def apply_batch(self, bundles: List[SignalBundle]) -> List[bool]:
        """Apply the chain to every bundle in a list."""
        return [self.apply(b) for b in bundles]

    def apply_batch_with_diagnostics(
        self, bundles: List[SignalBundle]
    ) -> tuple[List[bool], dict]:
        """
        Apply the chain to every bundle and collect per-filter rejection stats.

        Returns
        -------
        passed_list : list[bool]
            One bool per bundle (True = all filters passed).
        diagnostics : dict
            {
              filter_name: {
                "rejected": int,        # barre bloccate DA QUESTO filtro
                "skipped":  int,        # barre non raggiunte (short-circuit)
                "passed":   int,        # barre passate
                "top_reasons": list[tuple[str, int]]  # (motivo, conteggio) top-5
              }
            }
        """
        from collections import Counter, defaultdict

        # Inizializza contatori per ogni filtro
        stats: dict = {
            f.name: {"rejected": 0, "skipped": 0, "passed": 0, "reasons": Counter()}
            for f in self._filters
        }

        passed_list: List[bool] = []

        for bundle in bundles:
            bar_passed = True
            rejected_at: Optional[str] = None

            for f in self._filters:
                if rejected_at is not None:
                    # short-circuit: questo filtro non viene valutato
                    stats[f.name]["skipped"] += 1
                    continue

                result = f.apply(bundle)
                if result.passed:
                    stats[f.name]["passed"] += 1
                else:
                    stats[f.name]["rejected"] += 1
                    # normalizza il motivo: rimuovi valori numerici variabili
                    # per aggregare ragioni simili (es. "vol=0.000123 < 0.00055" → "flat market")
                    reason_key = result.reason.split(":")[0].strip() if result.reason else "unknown"
                    stats[f.name]["reasons"][reason_key] += 1
                    rejected_at = f.name
                    bar_passed = False

            passed_list.append(bar_passed)

        # Costruisci output finale
        n = len(bundles)
        diagnostics: dict = {}
        for f in self._filters:
            s = stats[f.name]
            diagnostics[f.name] = {
                "rejected": s["rejected"],
                "skipped":  s["skipped"],
                "passed":   s["passed"],
                "top_reasons": s["reasons"].most_common(5),
                "reject_pct": s["rejected"] / n * 100 if n else 0,
            }

        return passed_list, diagnostics

    def print_diagnostics(self, diagnostics: dict, total_bars: int) -> None:
        """Stampa un report leggibile delle statistiche di filtro."""
        print("\n" + "=" * 65)
        print("  FILTER CHAIN DIAGNOSTICS")
        print("=" * 65)
        print(f"  Totale barre analizzate: {total_bars}")

        cumulative_passed = total_bars
        for filter_name, d in diagnostics.items():
            evaluated = d["passed"] + d["rejected"]
            print(f"\n  [{filter_name}]")
            print(f"    Barre valutate : {evaluated}")
            print(f"    PASSATE        : {d['passed']:5d}  ({d['passed']/total_bars*100:.1f}% del totale)")
            print(f"    BLOCCATE       : {d['rejected']:5d}  ({d['reject_pct']:.1f}% del totale)")
            if d["skipped"]:
                print(f"    Non raggiunte  : {d['skipped']:5d}  (short-circuit da filtro precedente)")
            if d["top_reasons"]:
                print(f"    Motivi principali:")
                for reason, count in d["top_reasons"]:
                    print(f"      {count:5d}x  {reason}")

        # Colonne mancanti nel contesto
        print("\n  COLONNE CONTESTO (SignalAggregator._DEFAULT_CONTEXT_COLS):")
        from scripts.trading.combo.ensemble.aggregator import SignalAggregator
        expected_ctx = SignalAggregator._DEFAULT_CONTEXT_COLS
        if diagnostics:
            # Prova a ricostruire quali colonne erano presenti
            # dalle statistiche (non abbiamo accesso al df qui, ma possiamo stampare l'atteso)
            for col in expected_ctx:
                print(f"    {col}")
        print("=" * 65)

    @property
    def filter_names(self) -> List[str]:
        return [f.name for f in self._filters]

    @staticmethod
    def default_combo_chain(
        vol_threshold: float = 0.00055,
        wick_threshold: float = 0.0007,
        allowed_sessions: Optional[List[int]] = None,
    ) -> "FilterChain":
        """
        Factory that reproduces the original combo filter logic
        but through the new modular interface.

        Drop-in replacement for the old hardcoded flat_filter.
        """
        return FilterChain([
            FlatMarketFilter(
                volatility_threshold=vol_threshold,
                wick_threshold=wick_threshold,
                require_both=True,
            ),
            SessionFilter(
                allowed_session_codes=allowed_sessions or [1, 2, 3, 4],
            ),
        ])
