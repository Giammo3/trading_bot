"""
decision.py – Unified probabilistic decision framework.

Problems with the old approach
-------------------------------
OLD:
    df_combo["trade"] = (df_combo["binary03_signal"] == 1) & (df_combo["reversal_signal"] == 1)

Issues:
  1. Pure AND-gate: no weighting, no confidence, no partial signals.
  2. Reversal model fires 0 or 1 at default 0.5 cutoff — confidence is lost.
  3. Binary_03 threshold (0.001) is so low it's nearly always ON.
  4. Adding a third model requires editing the AND-gate logic directly.
  5. No concept of signal strength — a barely-passing signal is treated
     the same as a very-high-confidence signal.

NEW design
----------
The DecisionEngine takes a SignalBundle and produces a TradeDecision.
It supports multiple pluggable DecisionPolicies:

  ├─ ANDGatePolicy          – reproduces old behaviour exactly (for A/B testing)
  ├─ WeightedScorePolicy    – weighted average of confidences with a score threshold
  ├─ MajorityVotePolicy     – n-of-m models must signal positive
  └─ [future] RLPolicy      – delegates to an RL agent's action network

This design makes the old system a special case of the new one, so it can
be A/B tested and replaced incrementally.

TradeDecision
-------------
Not just a boolean — it carries:
  * go / no-go signal
  * composite confidence score (used for position sizing later)
  * the policy name that made the decision
  * all intermediate scores (for logging, analysis, RL replay buffer)

RL upgrade path
---------------
Replace the DecisionEngine's policy with an RLPolicy that receives the
SignalBundle as the state observation and outputs a continuous action.
The TradeDecision schema stays the same — everything downstream is unaffected.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .signal import SignalBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TradeDecision — the output of the DecisionEngine for one bar
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    """
    Immutable record of the decision made for one bar.

    Attributes
    ----------
    should_trade : bool
        True → open a position.
    composite_score : float
        Aggregate confidence in [0, 1].  Used for position sizing later.
    policy_name : str
        Which policy produced this decision.
    signal_breakdown : dict
        Per-model prediction and confidence, for logging / analysis.
    metadata : dict
        Extra diagnostics (e.g. active threshold, reason for rejection).
    timestamp : pd.Timestamp | None
        Bar timestamp from the SignalBundle.
    """

    should_trade: bool
    composite_score: float
    policy_name: str
    signal_breakdown: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Any = None   # pd.Timestamp or None

    @property
    def direction(self) -> str:
        """Placeholder for future directional trading."""
        return "LONG" if self.should_trade else "NO_TRADE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "should_trade": self.should_trade,
            "composite_score": self.composite_score,
            "policy_name": self.policy_name,
            **{f"breakdown__{k}": v for k, v in self.signal_breakdown.items()},
            **{f"meta__{k}": v for k, v in self.metadata.items()},
        }


# ---------------------------------------------------------------------------
# Base policy
# ---------------------------------------------------------------------------

class BaseDecisionPolicy(ABC):
    """
    Abstract base for all decision policies.

    A policy receives a SignalBundle (all model signals for one bar)
    and returns a TradeDecision.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy identifier."""

    @abstractmethod
    def decide(self, bundle: SignalBundle) -> TradeDecision:
        """Produce a TradeDecision from a SignalBundle."""

    def _build_breakdown(self, bundle: SignalBundle) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for model_name, sig in bundle.signals.items():
            out[f"{model_name}__pred"] = sig.prediction
            out[f"{model_name}__conf"] = round(sig.confidence, 4)
        return out


# ---------------------------------------------------------------------------
# Policy 1: ANDGatePolicy (reproduces legacy behaviour exactly)
# ---------------------------------------------------------------------------

class ANDGatePolicy(BaseDecisionPolicy):
    """
    Replicates the old combo bot: every required model must predict 1.

    Use this for A/B testing against the new policies.

    Parameters
    ----------
    required_models : list[str] | None
        Which models must all fire.  None = all registered models.
    """

    def __init__(self, required_models: Optional[List[str]] = None) -> None:
        self._required = required_models

    @property
    def name(self) -> str:
        return "ANDGatePolicy"

    def decide(self, bundle: SignalBundle) -> TradeDecision:
        models = self._required or bundle.model_names
        fire = bundle.all_positive(models)
        breakdown = self._build_breakdown(bundle)

        return TradeDecision(
            should_trade=fire,
            composite_score=bundle.weighted_confidence(),
            policy_name=self.name,
            signal_breakdown=breakdown,
            metadata={"required_models": models},
            timestamp=bundle.timestamp,
        )


# ---------------------------------------------------------------------------
# Policy 2: WeightedScorePolicy (primary recommended policy)
# ---------------------------------------------------------------------------

class WeightedScorePolicy(BaseDecisionPolicy):
    """
    Trades when the weighted average of model confidences exceeds a threshold.

    This is a soft AND: both models must contribute to the score, but a
    very high-confidence signal from one can partially compensate for a
    moderate signal from another.  Setting `min_score_threshold` ≥ 0.5
    effectively prevents either model from being too low.

    Parameters
    ----------
    weights : dict[str, float] | None
        Per-model weight.  Defaults to adapter.weight from the signals.
    score_threshold : float
        Minimum weighted confidence to trigger a trade.
    hard_gate_models : list[str] | None
        Models that must predict 1 regardless of score.  These act as
        hard guards (e.g. reversal must still fire).
    min_model_confidence : float
        Per-model minimum: if ANY model's confidence is below this, reject.
        Guards against a strong signal from one model masking a near-zero
        signal from another.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        score_threshold: float = 0.55,
        hard_gate_models: Optional[List[str]] = None,
        min_model_confidence: float = 0.40,
    ) -> None:
        self._weights = weights
        self._score_threshold = score_threshold
        self._hard_gates = hard_gate_models or []
        self._min_model_conf = min_model_confidence

    @property
    def name(self) -> str:
        return "WeightedScorePolicy"

    def decide(self, bundle: SignalBundle) -> TradeDecision:
        breakdown = self._build_breakdown(bundle)
        rejection_reason: Optional[str] = None

        # ── Hard gate check ────────────────────────────────────────────
        for model_name in self._hard_gates:
            sig = bundle.get(model_name)
            if sig is None:
                rejection_reason = f"hard_gate model '{model_name}' missing from bundle"
                break
            if not sig.is_positive:
                rejection_reason = (
                    f"hard_gate model '{model_name}' fired 0 "
                    f"(conf={sig.confidence:.4f})"
                )
                break

        # ── Per-model minimum confidence ───────────────────────────────
        if rejection_reason is None:
            for model_name, sig in bundle.signals.items():
                if sig.confidence < self._min_model_conf:
                    rejection_reason = (
                        f"model '{model_name}' confidence too low "
                        f"({sig.confidence:.4f} < {self._min_model_conf})"
                    )
                    break

        # ── Weighted composite score ───────────────────────────────────
        weights = self._weights or {
            n: s.metadata.get("weight", 1.0) for n, s in bundle.signals.items()
        }
        composite = bundle.weighted_confidence(weights)

        if rejection_reason is None and composite < self._score_threshold:
            rejection_reason = (
                f"composite score {composite:.4f} < threshold {self._score_threshold}"
            )

        fire = rejection_reason is None

        return TradeDecision(
            should_trade=fire,
            composite_score=composite,
            policy_name=self.name,
            signal_breakdown=breakdown,
            metadata={
                "score_threshold": self._score_threshold,
                "hard_gates": self._hard_gates,
                "rejection_reason": rejection_reason,
                "weights_used": weights,
            },
            timestamp=bundle.timestamp,
        )


# ---------------------------------------------------------------------------
# Policy 3: MajorityVotePolicy
# ---------------------------------------------------------------------------

class MajorityVotePolicy(BaseDecisionPolicy):
    """
    Requires at least `n_required` out of all registered models to predict 1.

    Useful when you add a third model and want 2-of-3 rather than 3-of-3.

    Parameters
    ----------
    n_required : int | None
        Minimum number of positive votes.  None → majority (⌈N/2⌉).
    weights : dict[str, float] | None
        Used only for computing composite_score; voting itself is unweighted.
    """

    def __init__(
        self,
        n_required: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._n_required = n_required
        self._weights = weights

    @property
    def name(self) -> str:
        return "MajorityVotePolicy"

    def decide(self, bundle: SignalBundle) -> TradeDecision:
        n_models = len(bundle.signals)
        required = self._n_required or int(np.ceil(n_models / 2))
        votes = sum(s.is_positive for s in bundle.signals.values())
        fire = votes >= required

        composite = bundle.weighted_confidence(self._weights)
        breakdown = self._build_breakdown(bundle)

        return TradeDecision(
            should_trade=fire,
            composite_score=composite,
            policy_name=self.name,
            signal_breakdown=breakdown,
            metadata={
                "n_required": required,
                "votes": votes,
                "n_models": n_models,
            },
            timestamp=bundle.timestamp,
        )


# ---------------------------------------------------------------------------
# DecisionEngine — wires a policy to the aggregator's output
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Top-level decision component.

    Applies a DecisionPolicy to each SignalBundle and returns a list of
    TradeDecisions.

    Parameters
    ----------
    policy : BaseDecisionPolicy
        The decision policy to use.

    Usage
    -----
    engine = DecisionEngine(policy=WeightedScorePolicy(
        weights={"reversal": 1.0, "binary03": 1.5},
        score_threshold=0.55,
        hard_gate_models=["reversal"],
    ))
    decisions = engine.evaluate(bundles)
    """

    def __init__(self, policy: BaseDecisionPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> BaseDecisionPolicy:
        return self._policy

    @policy.setter
    def policy(self, new_policy: BaseDecisionPolicy) -> None:
        """Hot-swap the policy without re-initialising the engine."""
        logger.info(
            "DecisionEngine: switching policy from '%s' to '%s'",
            self._policy.name,
            new_policy.name,
        )
        self._policy = new_policy

    def evaluate(self, bundles: List[SignalBundle]) -> List[TradeDecision]:
        """
        Apply the policy to every bundle and return one TradeDecision per bar.

        Parameters
        ----------
        bundles : list[SignalBundle]
            Output of SignalAggregator.aggregate().
        """
        decisions = [self._policy.decide(b) for b in bundles]
        n_trades = sum(d.should_trade for d in decisions)
        logger.info(
            "DecisionEngine [%s]: %d / %d bars → trade",
            self._policy.name,
            n_trades,
            len(decisions),
        )
        return decisions

    def evaluate_single(self, bundle: SignalBundle) -> TradeDecision:
        """Convenience wrapper for live bar-by-bar usage."""
        return self._policy.decide(bundle)
