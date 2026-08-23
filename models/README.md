# `models/` — ML confirmation models for the live regime daemon

Read by `realtime/regime_daemon.py::MasterRegimeDaemon.evaluate_ml_gate`. This
directory holds the Version B classifiers that confirm or veto a live entry
after the regime firewall has already permitted the symbol.

## Naming

| File | Registers for |
|---|---|
| `<strategy_id>.pkl` | every symbol that strategy trades |
| `<strategy_id>__<SYMBOL>.pkl` | that symbol alone, overriding the shared model |

The separator is a **double** underscore, because strategy ids in this
repository contain single ones (`sma_momentum_crossover`). `.pkl`, `.joblib`
and `.onnx` are recognised; `.onnx` additionally needs `onnxruntime`, which is
not in `requirements.txt` — pin it before shipping one.

## The sidecar is not optional

Each model needs `<same stem>.json`:

```json
{"features": ["atr_pct", "rsi_14", "dist_from_ema"], "threshold": 0.55}
```

**`features` is the column ORDER the model was fitted on, and the gate refuses
to run without it.** `evaluate_ml_gate` is handed a `dict`, and a dict has no
meaningful order; a classifier fed its columns permuted does not raise, it
returns a confident probability computed from a matrix that means nothing. The
sidecar is what orders the row, so the list must match the training matrix
column for column.

`threshold` is the probability the positive class must reach. It defaults to
`0.50` — the classifier's own decision boundary, which is the only value
defensible on behalf of a model whose author did not state one. A model
selected at a different operating point must declare it here.

The sidecar is committed and the model binary is not (see `.gitignore`): the
feature order and the threshold change what the filter does and belong in
review; the weights are a rebuildable artifact and a pickle is arbitrary code
at load time.

## What happens when a model is absent, broken, or under-declared

| State | `evaluate_ml_gate` |
|---|---|
| No model registered for the strategy | returns `True` — the documented pass-through, so a rule-based Version A keeps trading |
| Model registered and loads | `predict_proba(...)[1] >= threshold` |
| Model file present but fails to load | **raises `MLGateError`** |
| No sidecar, or no `features` list | **raises `MLGateError`** |
| A declared feature missing from the caller's dict | **raises `MLGateError`** |

The three raises are the point. Returning `True` there would trade unfiltered
while every log line read "ML confirmed"; returning `False` is
indistinguishable from a model that looked at the features and vetoed. Both are
silent. `MasterRegimeDaemon.describe()` lists what registered and what failed,
so a half-installed model is visible at start-up rather than at the first
signal.
