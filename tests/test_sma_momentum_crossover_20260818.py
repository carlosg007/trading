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
    """Verifies signal generation output shape and dtypes."""
    fn = getattr(strat_module, "calculate_signals", getattr(strat_module, "signal_fn", None))
    
    # Run with default kwargs
    entries, exits = fn(sample_ohlcv)
    
    assert len(entries) == len(sample_ohlcv), "Entries shape mismatch"
    assert len(exits) == len(sample_ohlcv), "Exits shape mismatch"
    assert entries.dtype == bool or np.issubdtype(entries.dtype, np.bool_), "Entries must be boolean"
    assert exits.dtype == bool or np.issubdtype(exits.dtype, np.bool_), "Exits must be boolean"

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

