"""
Ensemble trading package.

Architecture:
    signal.py       – ModelSignal dataclass, BaseModelAdapter ABC, SignalBundle
    adapters.py     – Concrete adapters: ReversalAdapter, Binary03Adapter
    aggregator.py   – SignalAggregator: timestamp-keyed merge, no positional tricks
    threshold.py    – ThresholdManager: walk-forward-aware, no CSV look-ahead bias
    decision.py     – DecisionEngine: unified probabilistic scoring + policies
    filters.py      – TradeFilter chain: flat-market, session, liquidity, etc.
    orchestrator.py – EnsembleOrchestrator: top-level wiring + bot_universal bridge

RL upgrade path (NOT implemented yet):
    In the future, replace DecisionEngine.decide() with an RL policy that
    receives the same SignalBundle and outputs a TradeDecision.  Everything
    above the DecisionEngine stays unchanged.
"""

from .signal import ModelSignal, SignalBundle
from .adapters import ReversalAdapter, Binary03Adapter
from .aggregator import SignalAggregator
from .threshold import  WalkForwardThresholdManager
from .decision import DecisionEngine, TradeDecision
from .filters import FilterChain, FlatMarketFilter, SessionFilter, LiquidityFilter
from .orchestrator import EnsembleOrchestrator

__all__ = [
    "ModelSignal",
    "SignalBundle",
    "ReversalAdapter",
    "Binary03Adapter",
    "SignalAggregator",
    "WalkForwardThresholdManager",
    "DecisionEngine",
    "TradeDecision",
    "FilterChain",
    "FlatMarketFilter",
    "SessionFilter",
    "LiquidityFilter",
    "EnsembleOrchestrator",
]
