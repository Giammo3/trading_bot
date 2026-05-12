# Combo Ensemble Architecture

## Overview

This document describes the modular ensemble trading system that replaces
`bot_combo_reversal_binary03.py`.  The new system lives entirely in the
`ensemble/` package and is invoked through `bot_combo_v2.py`.

---

## File Map

```
scripts/trading/combo/
├── bot_combo_reversal_binary03.py   ← OLD bot (keep for A/B reference)
├── bot_combo_v2.py                  ← NEW entry point (thin CLI wrapper)
└── ensemble/
    ├── __init__.py                  ← Public API re-exports
    ├── signal.py                    ← ModelSignal, SignalBundle, BaseModelAdapter
    ├── adapters.py                  ← ReversalAdapter, Binary03Adapter, AdapterRegistry
    ├── aggregator.py                ← SignalAggregator (timestamp-keyed merge)
    ├── threshold.py                 ← ThresholdManager family
    ├── decision.py                  ← DecisionEngine + policies
    ├── filters.py                   ← FilterChain + individual filters
    └── orchestrator.py              ← EnsembleOrchestrator (top-level wiring)
```

---

## Component Responsibilities

### `signal.py` — Data Contracts

| Class | Role |
|---|---|
| `ModelSignal` | Immutable output of one model for one bar. Carries `prediction`, `confidence`, `raw_proba`, `timestamp`. |
| `SignalBundle` | All signals for one bar. Carries per-model signals + `bar_context` (raw feature values). |
| `BaseModelAdapter` | ABC that every model wrapper must implement. |

### `adapters.py` — Model Wrappers

| Class | Role |
|---|---|
| `ReversalAdapter` | Wraps `best_model_xgb.pkl`. Hard binary gate. Confidence = predict_proba[:,1]. |
| `Binary03Adapter` | Wraps `binary03_combo_model_xgb.pkl`. Threshold is set externally by ThresholdManager. Blocks forbidden columns (future_return, etc.). |
| `AdapterRegistry` | Dictionary of adapters. Lets orchestrator discover adapters by name. |

### `aggregator.py` — Safe Merge

Replaces the fragile positional `pd.concat`:

```python
# OLD (BROKEN — silent misalignment):
df_combo = pd.concat([
    df_reversal.iloc[:min_len].reset_index(drop=True),
    df_binary03[["binary03_signal"]].iloc[:min_len].reset_index(drop=True)
], axis=1)

# NEW (timestamp-keyed — raises on mismatch):
bundles = SignalAggregator([reversal_adapter, binary03_adapter]).aggregate(
    df, timestamp_col="timestamp"
)
```

Both adapters score the **same DataFrame** → alignment is guaranteed.
If they somehow produce different row counts, an exception is raised immediately.

### `threshold.py` — Bias-Free Threshold Selection

| Class | When to use |
|---|---|
| `StaticThresholdManager` | Unit tests, frozen deployments, ablation studies. |
| `WalkForwardThresholdManager` | **Production.** Threshold selected from data strictly before the bar being evaluated. No look-ahead. |
| `PercentileThresholdManager` | Adaptive markets. Threshold = Nth percentile of past probabilities. |

The old system picked the threshold by maximising balance on the **test set** itself —
the same data it was evaluated on.  All new managers are trained on data before `as_of`.

### `decision.py` — Unified Probabilistic Framework

| Policy | Description |
|---|---|
| `ANDGatePolicy` | Exact replica of the old bot's `(rev==1) & (bin03==1)`. Use for A/B testing. |
| `WeightedScorePolicy` | Weighted average of confidences ≥ threshold. Hard gate for reversal. Per-model min confidence floor. **Recommended.** |
| `MajorityVotePolicy` | N-of-M models must vote 1. Useful when a third model is added. |

All policies produce a `TradeDecision` with:
- `should_trade` (bool)
- `composite_score` (float in [0,1])
- `signal_breakdown` (per-model details for logging)
- `rejection_reason` (why a trade was rejected)

### `filters.py` — Composable Filter Chain

| Filter | Replaces |
|---|---|
| `FlatMarketFilter` | Hardcoded `vol < 0.00055 AND wick < 0.0007` in flat_filter.py. Now calibrated from training data via `from_historical()`. |
| `SessionFilter` | Session logic scattered across entry_filters_reversal.py and entry_filters_auto.py. |
| `LiquidityFilter` | Relies on `liquidity_proxy` feature from unified feature engineering. |
| `MinConfidenceFilter` | New: rejects low-confidence decisions before execution. |

Filters are independently toggleable:
```python
FlatMarketFilter(enabled=False)   # disable for testing
```

### `orchestrator.py` — Execution Flow

```
load df
  │
  ▼
ThresholdManager.get_threshold(as_of=first_bar_ts)
  │  → sets binary03_adapter.confidence_threshold
  ▼
SignalAggregator.aggregate(df, timestamp_col="timestamp")
  │  → [SignalBundle, SignalBundle, ...]  (one per bar)
  ▼
DecisionEngine.evaluate(bundles)
  │  → [TradeDecision, TradeDecision, ...]
  ▼
FilterChain.apply_batch(bundles)
  │  → [bool, bool, ...]
  ▼
_run_backtest_execution(df, decisions, filter_results)
  │  TP/SL with capped dynamic SL (3× max), no price manipulation
  ▼
result_df  →  CSV
```

---

## Critical Fixes vs. Old System

| # | Old Problem | New Solution |
|---|---|---|
| 1 | **Positional merge** — silent row misalignment between reversal and binary_03 DataFrames | Both adapters score the same DataFrame; timestamp-keyed validation in aggregator |
| 2 | **Look-ahead threshold bias** — threshold selected on the test set it was evaluated on | `WalkForwardThresholdManager` uses only data before `as_of` |
| 3 | **Fake P&L** — `future_return` used instead of real TP/SL | `orchestrator._run_backtest_execution` simulates real TP/SL/look-ahead |
| 4 | **Target leakage** — `future_return` injected into feature matrix | `Binary03Adapter._FORBIDDEN` blocks look-ahead columns at score time |
| 5 | **Label misalignment** — y extracted by positional tail, not by key | Labels now aligned by timestamp/index key in the training pipeline |
| 6 | **Three divergent feature implementations** | Architecture expects `utils/feature_engineering.apply_all_features()` exclusively |
| 7 | **Hardcoded flat filter thresholds** | `FlatMarketFilter.from_historical()` calibrates from training data |
| 8 | **Unbounded dynamic SL** | Capped at `3× base_sl_pct` in orchestrator |
| 9 | **Three training scripts clobber the same .pkl** | Not fixed here (training layer concern) — see Training Conventions below |
| 10 | **Row-by-row iterrows() prediction loop** | All adapters use vectorised `predict_proba(X)` + list comprehension |

---

## How to Use

### Backtest (default)

```bash
python scripts/trading/combo/bot_combo_v2.py
```

### With walk-forward threshold (recommended for production)

```bash
python scripts/trading/combo/bot_combo_v2.py --threshold walk_forward
```

### A/B test: legacy AND-gate policy

```bash
python scripts/trading/combo/bot_combo_v2.py --policy and_gate
```

### Disable a filter

```bash
python scripts/trading/combo/bot_combo_v2.py --no-flat-filter
```

### Live bar-by-bar (broker integration)

```python
from scripts.trading.combo.ensemble import EnsembleOrchestrator

orch = EnsembleOrchestrator(backtest_mode=False)

# Called once per new 5-min candle:
result = orch.evaluate_bar(feature_row)
if result.decision.should_trade and result.filter_passed:
    place_order(direction="LONG", confidence=result.decision.composite_score)
```

---

## RL Upgrade Path

The architecture is designed so that adding an RL policy requires **zero changes**
to everything except the `DecisionEngine` policy.

**Step 1** — Create an RL policy (when ready):
```python
# scripts/trading/combo/ensemble/rl_policy.py  (not implemented yet)
class RLPolicy(BaseDecisionPolicy):
    def decide(self, bundle: SignalBundle) -> TradeDecision:
        state = self._bundle_to_state(bundle)   # flatten to numpy
        action = self._agent.act(state)          # continuous action
        return TradeDecision(
            should_trade=action > 0,
            composite_score=float(action),
            policy_name="RLPolicy",
        )
```

**Step 2** — Plug it in:
```python
orchestrator.engine.policy = RLPolicy(model_path="models/rl/policy.pt")
```

**Everything else stays the same.**

The SignalBundle already carries:
- Per-model confidence scores (model state)
- Bar context (market state: close, vol, session, liquidity)

These form the state observation vector for the RL agent.
The replay buffer can be wired to `orchestrator._record_transition()` (stub exists).

---

## Training Pipeline Conventions (not changed here)

To prevent the "three scripts clobber one .pkl" problem, adopt this convention:

| Script | Output |
|---|---|
| `retrain_combo_binary03_model.py` | `models/combo/binary03_combo_model_xgb.pkl` (canonical) |
| `smote_train_model.py` | `models/combo/binary03_combo_model_smote_xgb.pkl` (variant) |
| `train_model_binary03.py` | `models/combo/binary03_combo_model_scale_xgb.pkl` (variant) |

The bot always loads the **canonical** path.  Variants are for comparison only.
