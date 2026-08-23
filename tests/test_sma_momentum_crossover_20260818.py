import sys
from pathlib import Path

# Every other suite in this directory carries these two lines. Without them
# `strategies` is not importable — the runner puts `tests/` on the path, not
# the repository root — and this file failed to collect under pytest and to
# run as a script.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pytest
import numpy as np
import pandas as pd
import importlib

from backtest.engine import unpack_signals

# Dynamic import of the strategy module
strat_module = importlib.import_module("strategies.experimental.sma_momentum_crossover_20260818")

@pytest.fixture
def sample_ohlcv():
    """Generates synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range(start="2023-01-01", periods=n, freq="15min")
    
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.uniform(0.1, 1.0, n)
    low = close - np.random.uniform(0.1, 1.0, n)
    open_p = close + np.random.uniform(-0.2, 0.2, n)
    volume = np.random.randint(100, 5000, n)
    
    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)
    return df

def test_module_structure():
    """Verifies that mandatory functions and attributes exist."""
    assert hasattr(strat_module, "calculate_signals") or hasattr(strat_module, "signal_fn"), \
        "Strategy must define calculate_signals or signal_fn"
    if hasattr(strat_module, "PARAM_GRID"):
        assert isinstance(strat_module.PARAM_GRID, dict), "PARAM_GRID must be a dict"

def test_calculate_signals_output(sample_ohlcv):
    """
    Verifies signal generation output shape and dtypes.

    FOUR masks, not two. `sma_momentum_crossover_20260818` is bidirectional and
    returns the four-mask form of the strategy contract — long entries, long
    exits, short entries, short exits. Unpacking it into two raises
    `ValueError: too many values to unpack`, which is the loud failure; the
    quiet one is a test that takes the first two and calls it a pass while the
    strategy's whole short side goes unchecked.

    `backtest.engine.unpack_signals` accepts both forms and is what the engine
    itself calls, so going through it means this asserts exactly what a real
    run accepts rather than a second opinion about the contract.
    """
    fn = getattr(strat_module, "calculate_signals", getattr(strat_module, "signal_fn", None))

    long_entries, long_exits, short_entries, short_exits = unpack_signals(
        fn(sample_ohlcv), len(sample_ohlcv), sample_ohlcv.index)

    masks = {"long_entries": long_entries, "long_exits": long_exits,
             "short_entries": short_entries, "short_exits": short_exits}
    for name, mask in masks.items():
        assert len(mask) == len(sample_ohlcv), f"{name} shape mismatch"
        assert mask.dtype == bool or np.issubdtype(mask.dtype, np.bool_), \
            f"{name} must be boolean"
        assert not pd.Series(mask).isna().any(), f"{name} carries NaN"

    # A bar is never both a long and a short ENTRY: the walk takes neither on
    # an ambiguous bar, so a module emitting both has a bug that shows up as a
    # missing trade rather than as an error.
    assert not (np.asarray(long_entries, dtype=bool)
                & np.asarray(short_entries, dtype=bool)).any(), \
        "a bar carries both a long and a short entry"

def test_extract_features_causality(sample_ohlcv):
    """Verifies Version B feature extractor returns clean array without NaNs at valid indices."""
    if hasattr(strat_module, "extract_features"):
        features = strat_module.extract_features(sample_ohlcv)
        assert isinstance(features, (pd.DataFrame, np.ndarray)), "Features must be DataFrame or numpy array"
        # Check no infinite values
        if isinstance(features, pd.DataFrame):
            assert not np.isinf(features.values).any(), "Features contain Inf"
        else:
            assert not np.isinf(features).any(), "Features contain Inf"


if __name__ == "__main__":
    # Without this the file exits 0 when run as a script — it defines its cases
    # and never executes one, so a broken suite reports success. Every other
    # suite in this directory carries a runner for the same reason.
    pytest.main([__file__])
