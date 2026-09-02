"""
tests/test_energy_intraday_nonlinear_ar_20260902.py

ASSERT-BASED, so `tests/conftest.py` collects it normally and every case keeps
its own granularity. There is deliberately no `def check(` helper in this file:
that marker is what routes a suite to the subprocess runner, and a suite that
recorded results in a list instead of asserting would report green while
failing.

Inner helpers are named `_...` rather than `test_...` for the reason recorded
in CLAUDE.md: pytest collects any module-level `test_*` it can call, including
one whose only argument is defaulted, and in `test_regime_profiler.py` that ran
the sections without their artifact redirect and wrote real JSON onto the NFS
mount. The tell was the count - 8 passed where 4 were written.

Nothing here touches the lake, a mount or a broker. Every fixture is
deterministic and constructed in-process.

WHAT THIS SUITE IS ACTUALLY GUARDING
====================================
Not "is the ADX right" - `_adx` is copied verbatim from
`sma_momentum_crossover_20260818` and is validated against TA-Lib there. The
failures worth catching here are the ones specific to this module:

  * the quadrant id drifting back to the request's incorrect "Q2"
  * a short mask appearing in a strategy whose premise is bull-only
  * lookahead entering through `ml_features` - the one place causality is
    this module's own responsibility
  * a toggle that does not actually toggle, so Version A's ablation is
    measuring nothing
  * the risk keys drifting off the three names `promote.py` writes
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = (REPO_ROOT / "strategies" / "experimental"
               / "energy_intraday_nonlinear_ar_20260902.py")


def _load():
    """The module, loaded from its FILE PATH exactly as the engine loads it."""
    spec = importlib.util.spec_from_file_location(
        "energy_intraday_nonlinear_ar_20260902", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load()


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _bars(n: int = 3000, seed: int = 11, drift: float = 0.02,
          start: str = "2024-01-02 13:00", freq: str = "30min",
          volume: str = "random") -> pd.DataFrame:
    """
    Deterministic 30m bars in the shape the engine yields: `ts` as a COLUMN,
    positional index, UTC.

    13:00 UTC is 08:00 ET in January, so the sample starts an hour before the
    outcry window opens and spans it every day. No randomness that is not
    seeded: a strategy test whose fixture moves is one that will fail some day
    for a reason nobody can reproduce.
    """
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 60.0 + np.cumsum(rng.normal(drift, 0.25, n))
    if volume == "random":
        vol = rng.uniform(500.0, 3000.0, n)
    elif volume == "flat":
        vol = np.full(n, 1000.0)
    else:
        raise ValueError(volume)
    return pd.DataFrame({
        "ts": ts,
        "open": close,
        "high": close + rng.uniform(0.05, 0.40, n),
        "low": close - rng.uniform(0.05, 0.40, n),
        "close": close,
        "volume": vol,
    })


def _code_only(source: str) -> str:
    """
    `source` with every comment and string literal removed.

    A textual lookahead sweep has to run on code. Prose describing the property
    - "there is no `.shift(-1)` anywhere in this module" - contains the pattern
    it is asserting the absence of, so a plain substring check fails on the
    documentation and would be "fixed" by deleting the sentence.
    """
    import io
    import tokenize

    kept = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(tok.string)
    # Joined with "" rather than " ": `close.shift(-1)` must survive as the
    # contiguous text the patterns below match, not as `close . shift ( - 1 )`,
    # which would make the sweep pass on code that genuinely reads the future.
    return "".join(kept)


def _candidates(bars: pd.DataFrame, **params) -> np.ndarray:
    """
    Layer 1 AND 2 AND 3 before the walk - the CANDIDATE triggers, not the
    realised entries.

    THE DISTINCTION IS LOAD-BEARING AND IS RECORDED IN CLAUDE.md. A filter can
    only remove candidate TRIGGERS, never realised trades: the walk holds one
    position at a time and ignores a trigger arriving while one is open, so
    declining an early trigger can leave the strategy flat for a later one it
    would otherwise have been holding through - and the realised entry count
    goes UP. Measured elsewhere in this tree: enabling a filter removed 17
    candidates and ADDED 11 realised entries. Never use a trade count to decide
    whether a filter binds.
    """
    p = {**M.DEFAULT_PARAMS, **params}
    L = M._layers(bars, int(p["sma_L_len"]), float(p["adx_thresh"]),
                  int(p["vol_sma_len"]), p["session_start_et"],
                  p["session_end_et"], bool(p["use_baseline_filter"]),
                  bool(p["use_alpha_trigger"]), bool(p["use_volume_filter"]))
    return (L["baseline_long"] & L["trigger_long"]
            & L["confirm_long"]).to_numpy()


# --------------------------------------------------------------------------
# 1. Output shape, alignment and dtype
# --------------------------------------------------------------------------
def test_signal_fn_returns_two_aligned_boolean_series():
    """
    The two-mask long-only form. `engine.unpack_signals` accepts two or four
    and RAISES on anything else - silently taking the first two masks of a
    three-tuple is how a strategy's short side disappears into a plausible
    long-only equity curve.
    """
    bars = _bars()
    out = M.signal_fn(bars)

    assert isinstance(out, tuple) and len(out) == 2, (
        f"long-only returns (entries, exits); got {type(out)} of "
        f"{len(out) if isinstance(out, tuple) else '?'}")
    entries, exits = out
    for name, series in (("entries", entries), ("exits", exits)):
        assert isinstance(series, pd.Series), f"{name} is not a Series"
        assert series.dtype == bool, f"{name} dtype is {series.dtype}, not bool"
        assert len(series) == len(bars), f"{name} is not bar-aligned"
        assert series.index.equals(bars.index), f"{name} lost the bars' index"
        assert not series.isna().any(), f"{name} carries NaN"


def test_the_engine_can_unpack_what_signal_fn_returns():
    """The masks go straight into the engine's own unpacker, not just a shape
    this suite happens to like."""
    from backtest.engine import unpack_signals

    bars = _bars()
    unpacked = unpack_signals(M.signal_fn(bars), len(bars), index=bars.index)
    assert len(unpacked) == 4, "unpack_signals normalises to four masks"
    _le, _lx, se, sx = unpacked
    assert not np.asarray(se, dtype=bool).any(), "a short entry appeared"
    assert not np.asarray(sx, dtype=bool).any(), "a short exit appeared"


def test_the_module_declares_all_four_required_hooks():
    """`signal_fn`, `indicators`, `LOGIC`, `PARAM_GRID` - plus the two this
    request names, `make_signal_fn` and `ml_features`."""
    for name in ("signal_fn", "indicators", "make_signal_fn", "ml_features"):
        assert callable(getattr(M, name, None)), f"{name} is not declared"
    assert isinstance(M.LOGIC, dict) and M.LOGIC
    assert isinstance(M.PARAM_GRID, dict) and M.PARAM_GRID


def test_the_loader_accepts_the_module_and_binds_both_hooks():
    """`load_strategy` is what the engine actually calls."""
    from agents.tier3_workers import load_strategy

    fn, info = load_strategy(str(MODULE_PATH))
    assert callable(fn)
    assert info["timeframe"] == "30m"
    assert info["symbols"] == ["HO", "NG", "MCL"]
    assert callable(info["indicator_fn"])
    assert callable(info["ml_feature_fn"]), (
        "ml_feature_fn is None - Version B would silently fall back to the "
        "shared causal_features and be fitted on different columns from the "
        "ones this module declares")


def test_indicators_are_full_length_and_bar_aligned():
    bars = _bars()
    drawn = M.indicators(bars)
    assert drawn, "no indicator series declared"
    for name, series in drawn.items():
        assert isinstance(series, pd.Series), f"{name} is not a Series"
        assert len(series) == len(bars), f"{name} is not full length"
        assert series.index.equals(bars.index), f"{name} lost the index"


def test_it_accepts_a_datetime_indexed_frame_as_well_as_a_ts_column():
    """
    The engine hands a long-format frame with `ts` as a COLUMN; a caller
    holding a time-indexed fixture has the opposite shape. Both have to work or
    one of them silently produces garbage.
    """
    bars = _bars(n=1200)
    indexed = bars.set_index(pd.DatetimeIndex(bars["ts"])).drop(columns=["ts"])

    e_col, _ = M.signal_fn(bars)
    e_idx, _ = M.signal_fn(indexed)
    assert np.array_equal(e_col.to_numpy(), e_idx.to_numpy()), (
        "the two frame shapes produced different signals")


def test_a_frame_with_neither_ts_nor_a_datetime_index_raises():
    bars = _bars(n=400).drop(columns=["ts"])
    with pytest.raises(ValueError, match="ts.*DatetimeIndex|DatetimeIndex"):
        M.signal_fn(bars)


# --------------------------------------------------------------------------
# 2. The strategy is long-only, and that is the premise
# --------------------------------------------------------------------------
def test_no_short_is_ever_produced_under_any_grid_cell():
    """
    The premise is that the predictability exists ONLY in bull states. A
    mirrored short would trade the half of the paper that reports no edge,
    under this module's certification.
    """
    bars = _bars(n=1500, drift=-0.05)          # a falling tape, where a
    for sma_l in M.PARAM_GRID["sma_L_len"]:    # sign-flipped rule would fire
        for adx in M.PARAM_GRID["adx_thresh"]:
            out = M.signal_fn(bars, sma_L_len=sma_l, adx_thresh=adx)
            assert len(out) == 2, "a short mask appeared in the return shape"


def test_entries_only_fire_while_price_is_above_the_macro_ema():
    """Layer 1's bull gate. ADX is direction-agnostic and cannot supply it."""
    bars = _bars()
    cand = _candidates(bars)
    L = M._layers(bars, 20, 25.0, 20, "09:00", "14:30", True, True, True)
    above = (L["close"] > L["ema_macro"]).to_numpy()
    assert cand.any(), "no candidates at all - the fixture proves nothing"
    assert not (cand & ~above).any(), (
        "a candidate fired with the close below EMA(200)")


def test_entries_only_fire_above_the_adx_threshold():
    bars = _bars()
    L = M._layers(bars, 20, 25.0, 20, "09:00", "14:30", True, True, True)
    adx = L["adx"].to_numpy()
    cand = _candidates(bars)
    assert cand.any()
    assert np.all(adx[cand] > 25.0), "a candidate fired at or below the ADX gate"


def test_the_adx_gate_is_strictly_greater_than_never_greater_or_equal():
    """
    `mdlib/regimes.py` uses `>` and every cached quadrant, Stage 1 designation
    and Gate R verdict is drawn on it. A module using `>=` here would place its
    own entries on the other side of the boundary its certification is scored
    against - at ADX exactly 25.00000 the two disagree.
    """
    source = MODULE_PATH.read_text()
    assert "adx > float(adx_thresh)" in source, (
        "the ADX comparison is not the strict `>` mdlib/regimes.py uses")
    assert "adx >= float(adx_thresh)" not in source


# --------------------------------------------------------------------------
# 3. The session window
# --------------------------------------------------------------------------
def test_entries_are_confined_to_the_outcry_window_in_new_york_time():
    """
    09:00-14:30 America/New_York, through a NAMED ZONE. A fixed offset is an
    hour wrong from March to November - and silently, because it just selects
    a different five and a half hours.
    """
    bars = _bars(n=6000)                       # spans a DST transition
    cand = _candidates(bars)
    assert cand.any()
    et = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True)).tz_convert(
        "America/New_York")
    minutes = (et.hour * 60 + et.minute).to_numpy()
    fired = minutes[cand]
    assert fired.min() >= 9 * 60, f"an entry at {fired.min()} minutes ET"
    assert fired.max() < 14 * 60 + 30, (
        f"an entry at {fired.max()} minutes ET, at or past the flatten")


def test_the_session_window_survives_the_dst_transition():
    """
    The window is the same five and a half LOCAL hours in March and in July. A
    fixed UTC offset would shift it by an hour across the transition and the
    two halves of the sample would be trading different sessions.
    """
    for start in ("2024-02-05 12:00", "2024-07-08 12:00"):   # EST then EDT
        bars = _bars(n=800, start=start)
        L = M._layers(bars, 20, 25.0, 20, "09:00", "14:30", True, True, True)
        et = L["ts_et"]
        minutes = (et.hour * 60 + et.minute).to_numpy()
        gate = L["in_session"].to_numpy()
        assert gate.any(), f"no in-session bars at all for {start}"
        assert minutes[gate].min() >= 9 * 60
        assert minutes[gate].max() < 14 * 60 + 30


def test_the_flatten_bar_is_the_complement_of_the_entry_window_end():
    """
    Half-open [start, end). Bars are stamped when they OPEN, so on a 30m feed
    the 14:00 bar covers 14:00-14:30 and is the last one wholly inside the
    session; the 14:30 bar is the flatten bar. An inclusive end would make one
    bar both.
    """
    bars = _bars(n=800)
    L = M._layers(bars, 20, 25.0, 20, "09:00", "14:30", True, True, True)
    in_session = L["in_session"].to_numpy()
    flat = L["flat_bar"].to_numpy()
    assert not (in_session & flat).any(), (
        "a bar is both an entry bar and the flatten bar")


def test_a_position_is_always_closed_by_the_session_flatten():
    """
    Layer 4's unconditional exit. Every entry has to be matched by an exit at
    or before the session boundary - a strategy that can hold overnight is not
    the one specified.
    """
    bars = _bars(n=4000)
    entries, exits = M.signal_fn(bars)
    e = entries.to_numpy()
    x = exits.to_numpy()
    assert e.sum() > 5, f"only {int(e.sum())} entries - too few to prove this"
    assert int(x.sum()) == int(e.sum()), (
        f"{int(e.sum())} entries but {int(x.sum())} exits - a position was "
        f"left open")

    # And every exit is strictly after its entry, one position at a time.
    ei = np.flatnonzero(e)
    xi = np.flatnonzero(x)
    assert np.all(xi > ei), "an exit landed on or before its own entry bar"
    assert np.all(ei[1:] > xi[:-1]), "a second entry opened before the first "\
                                     "position closed"


def test_an_inverted_session_window_is_refused():
    with pytest.raises(ValueError, match="before"):
        M.signal_fn(_bars(n=400), session_start_et="14:30",
                    session_end_et="09:00")


# --------------------------------------------------------------------------
# 4. The modular toggles, each in isolation
# --------------------------------------------------------------------------
def test_the_baseline_filter_toggle_actually_binds():
    """
    Compared on CANDIDATES, never on realised trades. See `_candidates`: a
    filter can only remove triggers, and removing an early one can ADD a later
    realised entry.
    """
    bars = _bars()
    on = _candidates(bars, use_baseline_filter=True)
    off = _candidates(bars, use_baseline_filter=False)
    assert not (on & ~off).any(), "the filter ADDED candidates"
    assert int(off.sum()) > int(on.sum()), (
        "disabling the baseline filter changed nothing - the toggle is inert")


def test_the_alpha_trigger_toggle_actually_binds():
    bars = _bars()
    on = _candidates(bars, use_alpha_trigger=True)
    off = _candidates(bars, use_alpha_trigger=False)
    assert int(on.sum()) != int(off.sum()), (
        "the alpha trigger toggle is inert")


def test_the_alpha_trigger_off_is_an_event_not_a_state():
    """
    With the trigger off the strategy is the baseline alone, which is a STATE.
    Expressed as a state it would make EVERY bar of an uptrend a candidate and
    the Version A ablation would be meaningless. It is expressed as the bar
    price regains SMA(L).
    """
    bars = _bars()
    off = _candidates(bars, use_alpha_trigger=False)
    L = M._layers(bars, 20, 25.0, 20, "09:00", "14:30", True, False, True)
    state = (L["close"] > L["sma_l"]).to_numpy()
    assert int(off.sum()) < int(state.sum()) / 2, (
        "with the trigger off, candidates track the STATE rather than its "
        "onset")


def test_the_volume_filter_toggle_actually_binds():
    bars = _bars()
    on = _candidates(bars, use_volume_filter=True)
    off = _candidates(bars, use_volume_filter=False)
    assert not (on & ~off).any(), "the volume filter ADDED candidates"
    assert int(off.sum()) > int(on.sum()), "the volume filter is inert"


def test_the_volume_filter_carries_the_normalised_atr_floor():
    """Layer 3 is BOTH conditions; a volume-only reading drops the vol floor."""
    bars = _bars()
    cand = _candidates(bars)
    L = M._layers(bars, 20, 25.0, 20, "09:00", "14:30", True, True, True)
    assert cand.any()
    assert np.all(L["norm_atr"].to_numpy()[cand] > M.MIN_NORM_ATR), (
        "a candidate fired below the normalised-ATR floor")
    assert np.all(L["volume"].to_numpy()[cand]
                  > L["vol_sma"].to_numpy()[cand]), (
        "a candidate fired on below-average volume")


def test_every_toggle_is_independent_of_every_other():
    """
    Each is switched alone against the all-on baseline. A toggle that only
    binds when another is also set is not a modular toggle, and the ablation
    table it produces would be attributing one filter's effect to another.
    """
    bars = _bars()
    base = int(_candidates(bars).sum())
    for name in ("use_baseline_filter", "use_alpha_trigger",
                 "use_volume_filter"):
        alone = int(_candidates(bars, **{name: False}).sum())
        assert alone != base, f"{name} does nothing on its own"


def test_the_news_filter_is_wired_to_the_event_calendar():
    """
    `use_news_filter` reaches `backtest.event_calendar.apply_entry_filters` and
    unpacks its THREE return values. Four modules in this directory unpack two
    and raise `ValueError: too many values to unpack` the moment the flag is
    set; this one follows the older working form.
    """
    bars = _bars(n=2000)
    base = M.signal_fn(bars, use_news_filter=False)[0].to_numpy()
    gated = M.signal_fn(bars, use_news_filter=True)[0].to_numpy()

    assert base.any(), "no entries without the filter - proves nothing"
    # Candidates can only be removed; realised entries may move either way.
    lo_base = _candidates(bars)
    lo_gated, _short = M._news_suppress(bars, lo_base.copy(),
                                        np.zeros(len(bars), dtype=bool))
    assert not (lo_gated & ~lo_base).any(), "the news filter ADDED candidates"
    assert gated.dtype == bool


def test_the_news_filter_refuses_a_frame_with_no_timestamps():
    """Filtering a frame the calendar cannot place is filtering nothing, and it
    must say so rather than pass everything through."""
    bars = _bars(n=400).drop(columns=["ts"])
    indexed = _bars(n=400)
    bars.index = pd.DatetimeIndex(indexed["ts"])
    with pytest.raises(ValueError, match="ts"):
        M.signal_fn(bars, use_news_filter=True)


# --------------------------------------------------------------------------
# 5. Risk-parameter key integrity
# --------------------------------------------------------------------------
def test_the_risk_keys_are_exactly_the_three_the_pipeline_writes():
    """
    `run.py`'s RISK_PARAMS and `promote.py`'s RISK_KEYS are both exactly
    ("sl_atr_mult", "tp_atr_mult", "trailing"), and `promote._risk_block`
    writes those three and nothing else. A stop distance under any other name
    would be ABSENT from the promoted risk block, and the card would report
    `sl_atr_mult` as the stop while a different number bound every trade.
    """
    from backtest.promote import RISK_KEYS
    from backtest.run import RISK_PARAMS

    assert RISK_PARAMS == RISK_KEYS == ("sl_atr_mult", "tp_atr_mult",
                                        "trailing")
    for key in RISK_KEYS:
        assert key in M.DEFAULT_PARAMS, f"{key} is not a declared parameter"

    import inspect
    accepted = set(inspect.signature(M.signal_fn).parameters)
    for key in RISK_KEYS:
        assert key in accepted, f"signal_fn does not accept {key}"


def test_the_defaults_match_the_specification():
    assert M.DEFAULT_PARAMS["sma_L_len"] == 20
    assert M.DEFAULT_PARAMS["adx_thresh"] == 25.0
    assert M.DEFAULT_PARAMS["vol_sma_len"] == 20
    assert M.DEFAULT_PARAMS["sl_atr_mult"] == 1.5
    assert M.DEFAULT_PARAMS["tp_atr_mult"] == 3.0
    assert M.DEFAULT_PARAMS["trailing"] is True


def test_the_param_grid_matches_the_specification():
    assert M.PARAM_GRID["sma_L_len"] == [10, 20, 30]
    assert M.PARAM_GRID["adx_thresh"] == [20.0, 25.0, 30.0]
    assert M.PARAM_GRID["sl_atr_mult"] == [1.0, 1.5, 2.0]
    assert M.PARAM_GRID["tp_atr_mult"] == [2.0, 3.0, None]
    assert M.PARAM_GRID["trailing"] == [False, True]


def test_every_grid_key_is_a_parameter_signal_fn_accepts():
    """
    `load_strategy` rejects unknown parameter names, so a stale grid key raises
    rather than being quietly ignored - but only if it is actually swept. This
    catches it at the declaration.
    """
    import inspect
    accepted = set(inspect.signature(M.signal_fn).parameters)
    unknown = set(M.PARAM_GRID) - accepted
    assert not unknown, f"PARAM_GRID sweeps parameters signal_fn rejects: "\
                        f"{sorted(unknown)}"


def test_make_signal_fn_refuses_an_unknown_parameter():
    with pytest.raises(ValueError, match="unknown parameter"):
        M.make_signal_fn(trail_atr_mult=2.0)


def test_make_signal_fn_binds_and_returns_a_one_argument_callable():
    bars = _bars(n=1200)
    fn = M.make_signal_fn(sma_L_len=10, adx_thresh=20.0)
    bound = fn(bars)
    direct = M.signal_fn(bars, sma_L_len=10, adx_thresh=20.0)
    assert np.array_equal(bound[0].to_numpy(), direct[0].to_numpy())
    assert np.array_equal(bound[1].to_numpy(), direct[1].to_numpy())


# --------------------------------------------------------------------------
# 6. The tp_atr_mult=None runner guard
# --------------------------------------------------------------------------
def test_no_target_without_a_trailing_stop_is_refused():
    """
    The request's explicit guard. With no target and a stop that never follows
    the price, the bracket has no upside exit at all.
    """
    with pytest.raises(ValueError, match="trailing=True"):
        M.signal_fn(_bars(n=400), tp_atr_mult=None, trailing=False)


def test_no_target_with_a_trailing_stop_is_allowed():
    entries, exits = M.signal_fn(_bars(n=1500), tp_atr_mult=None,
                                 trailing=True)
    assert entries.dtype == bool and exits.dtype == bool


def test_the_refused_cells_are_exactly_the_twenty_seven_documented():
    """
    27 of the grid's 162 cells raise. `scan.py` counts the combination it
    EVALUATED, not the ones that survived, so this shrinks neither the search
    nor the honesty of `variants_tested` - but the count has to match what the
    PARAM_GRID note claims.
    """
    import itertools
    keys = list(M.PARAM_GRID)
    cells = list(itertools.product(*(M.PARAM_GRID[k] for k in keys)))
    assert len(cells) == 162

    refused = sum(1 for cell in cells
                  if dict(zip(keys, cell))["tp_atr_mult"] is None
                  and dict(zip(keys, cell))["trailing"] is False)
    assert refused == 27, f"{refused} cells are refused, the note claims 27"


def test_a_target_of_none_never_fires_rather_than_firing_at_the_fill():
    """
    NaN, not 0.0. A zero target sits AT the fill and would close every trade on
    its own entry bar; a huge sentinel is a level the search could reach.
    """
    assert np.isnan(M._tp_distance(None))
    assert M._tp_distance(3.0) == 3.0


@pytest.mark.parametrize("kwargs,match", [
    ({"sma_L_len": 1}, "sma_L_len"),
    ({"vol_sma_len": 0}, "vol_sma_len"),
    ({"adx_thresh": 150.0}, "adx_thresh"),
    ({"sl_atr_mult": 0.01}, "sl_atr_mult"),
    ({"tp_atr_mult": -1.0}, "tp_atr_mult"),
    ({"trailing": "yes"}, "trailing"),
    ({"session_start_et": "9am"}, "session_start_et"),
])
def test_a_parameter_set_the_module_cannot_describe_honestly_is_refused(
        kwargs, match):
    with pytest.raises(ValueError, match=match):
        M.signal_fn(_bars(n=400), **kwargs)


# --------------------------------------------------------------------------
# 7. Causality — the part that is this module's own responsibility
# --------------------------------------------------------------------------
def test_ml_features_has_the_declared_shape_and_columns():
    bars = _bars()
    features = M.ml_features(bars)
    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(bars), "the matrix is not bar-aligned"
    assert features.index.equals(bars.index)
    assert list(features.columns) == [
        "norm_atr", "volume_z", "adx_14", "vol_pct_rank", "ut_atr",
        "ar1_ret", "hour_et"]
    assert np.isfinite(features.to_numpy()).all(), (
        "a non-finite value survived into the feature matrix")


def test_no_negative_shift_or_reversed_slice_anywhere_in_the_module():
    """
    Lookahead written as an index shift is the form that actually shows up, and
    it is invisible in the equity curve - it just makes the strategy look
    brilliant. Checked on the AST rather than by eye.
    """
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = node.func.attr if isinstance(node.func,
                                                 ast.Attribute) else ""
            if fname in ("shift", "pct_change", "diff"):
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(arg, ast.UnaryOp) and isinstance(
                            arg.op, ast.USub):
                        raise AssertionError(
                            f"negative {fname}() at line {node.lineno} - that "
                            f"reads the future")
    # The textual sweep runs on CODE ONLY, with comments and string literals
    # tokenized away. The module's own docstring says "there is no `.shift(-1)`
    # anywhere in this module", and a naive `in source` check fails on the
    # sentence asserting the property it is checking for.
    code = _code_only(MODULE_PATH.read_text())
    assert "[::-1]" not in code, "a reversed slice"
    assert "shift(-" not in code, "a negative shift"
    assert "center=True" not in code, "a centred rolling window"


def test_every_feature_at_bar_i_is_unchanged_by_bars_after_i():
    """
    THE DIRECT TEST, and the one that catches what a shift-based audit cannot.
    Truncate the frame at bar i and the row for bar i must be identical to the
    row computed over the whole frame. A rolling percentile taken over the full
    sample, or a scaler fitted end to end, fails here and passes every
    grep-shaped check.
    """
    bars = _bars(n=900, seed=5)
    full = M.ml_features(bars)

    for cut in (400, 600, 875):
        truncated = M.ml_features(bars.iloc[:cut].copy())
        assert len(truncated) == cut
        np.testing.assert_allclose(
            truncated.iloc[cut - 1].to_numpy(dtype=float),
            full.iloc[cut - 1].to_numpy(dtype=float),
            rtol=1e-9, atol=1e-9,
            err_msg=(f"row {cut - 1} changed when bars after it were added - "
                     f"the matrix reads the future"))


def test_every_signal_at_bar_i_is_unchanged_by_bars_after_i():
    """The same test on the masks. A causal feature matrix over a
    future-reading signal would still be a future-reading strategy."""
    bars = _bars(n=900, seed=5)
    full_entries, _ = M.signal_fn(bars)

    for cut in (500, 700, 880):
        truncated, _ = M.signal_fn(bars.iloc[:cut].copy())
        assert np.array_equal(
            truncated.to_numpy()[:cut - 1],
            full_entries.to_numpy()[:cut - 1]), (
            f"signals before bar {cut} changed when later bars were added")


def test_the_volatility_percentile_is_a_trailing_rank_not_a_full_sample_one():
    """
    Named separately because it is the single most likely way lookahead enters
    this particular matrix: `rank(pct=True)` over the whole column is the
    obvious spelling and it is a statistic of the test period leaking into the
    training rows.
    """
    bars = _bars(n=900, seed=5)
    features = M.ml_features(bars)
    rank = features["vol_pct_rank"]
    assert rank.between(0.0, 1.0).all(), "a percentile outside [0, 1]"

    # A full-sample rank would place the maximum of the whole column at exactly
    # 1.0 at the bar where the series peaks. A trailing rank need not.
    truncated = M.ml_features(bars.iloc[:600].copy())
    assert truncated["vol_pct_rank"].iloc[599] == pytest.approx(
        rank.iloc[599], rel=1e-9), "the rank window extends past bar i"


def test_the_ar1_feature_is_one_lag_backwards():
    """`ar1_ret` is the autoregressive term the premise is named for. One lag,
    and the direction of that lag is the whole question."""
    bars = _bars(n=600, seed=3)
    features = M.ml_features(bars)
    close = bars["close"].astype(float)
    expected = close.pct_change(1).replace([np.inf, -np.inf],
                                           np.nan).fillna(0.0)
    np.testing.assert_allclose(features["ar1_ret"].to_numpy(),
                               expected.to_numpy(), rtol=1e-12, atol=1e-12)


def test_the_functional_coefficient_spread_is_close_minus_sma():
    """`Ut = Close - SMA(L)`, the request's spelling, and the same series the
    rules test the sign of."""
    bars = _bars(n=600, seed=3)
    L = M._layers(bars, 20, 25.0, 20, "09:00", "14:30", True, True, True)
    expected = bars["close"].astype(float) - bars["close"].astype(
        float).rolling(20, min_periods=20).mean()
    np.testing.assert_allclose(L["ut"].to_numpy(), expected.to_numpy(),
                               rtol=1e-12, atol=1e-12, equal_nan=True)


def test_version_b_vetoes_through_the_declared_feature_matrix():
    """
    Version B end to end: the shared expanding-window filter, fitted on THIS
    module's seven columns, acting as a veto and nothing else. The two
    properties that make it a veto: it may only turn entries OFF, and it may
    never turn one on.
    """
    from agents.tier3_workers import apply_ml_signal_filter

    bars = _bars(n=6000, seed=17)
    entries, exits = M.signal_fn(bars)
    assert int(entries.sum()) > 20, (
        f"only {int(entries.sum())} entries - too few for the filter to have "
        f"anything to fit")

    filtered, filtered_exits = apply_ml_signal_filter(
        bars, entries, exits, symbol="MCL", direction="long",
        features=M.ml_features)

    f = np.asarray(filtered, dtype=bool)
    a = np.asarray(entries, dtype=bool)
    assert not (f & ~a).any(), "Version B turned entries ON - it is not a veto"
    assert int(f.sum()) <= int(a.sum()), "Version B added entries"
    assert np.array_equal(np.asarray(filtered_exits, dtype=bool),
                          np.asarray(exits, dtype=bool)), (
        "Version B modified the exits")


# --------------------------------------------------------------------------
# 8. Metadata, and the quadrant correction
# --------------------------------------------------------------------------
def test_the_strategy_id_is_pinned_exactly_as_the_request_spells_it():
    """Pinned for bt-stage1 logging and JSON pipeline tracking."""
    assert M.STRATEGY_ID == "energy_intraday_nonlinear_ar"
    assert M.LOGIC["strategy_id"] == "energy_intraday_nonlinear_ar"


def test_the_loader_does_not_carry_strategy_id_into_the_tear_sheet_logic():
    """
    Recorded so nobody looks for it downstream and concludes it was dropped by
    mistake: `_describe_strategy` hard-codes ("concept", "entry", "exit"), so
    `LOGIC["strategy_id"]` lives on the MODULE and is not in
    `module_info["logic"]`.
    """
    from agents.tier3_workers import load_strategy

    _fn, info = load_strategy(str(MODULE_PATH))
    assert set(info["logic"]) <= {"concept", "entry", "exit"}
    assert "strategy_id" not in info["logic"]


def test_the_target_quadrant_is_q1_not_the_requests_q2():
    """
    THE CORRECTION, PINNED. The request asked for "Q2: High Vol / Trending".
    Q2 is High-Volatility/RANGING; the label it gave names Q1. A strategy
    registered under the wrong quadrant id is stood down in the environment it
    was certified for and turned loose in the one it never traded, with every
    log line reading correctly.
    """
    assert M.TARGET_QUADRANTS == ("Q1",)
    assert M.TARGET_REGIMES == ("High Volatility / Trending",)


def test_the_quadrant_ids_agree_with_mdlib_regimes():
    """
    Pinned against the authority rather than restated, so a future edit cannot
    drift them. `mdlib/regimes.py` is the only place the encoding is written
    down.
    """
    from mdlib.regimes import QUADRANT_LABELS

    assert QUADRANT_LABELS[1] == "High Volatility / Trending"
    assert QUADRANT_LABELS[2] == "High Volatility / Ranging"
    for quadrant, label in zip(M.TARGET_QUADRANTS, M.TARGET_REGIMES):
        assert QUADRANT_LABELS[int(quadrant[1:])] == label, (
            f"{quadrant} is not {label} in mdlib/regimes.py")


def test_the_declared_symbols_are_verified_contracts():
    """
    `backtest/specs.py` carries UNVERIFIED rows for several symbols. A
    multiplier or tick size that was never checked makes every P&L figure for
    that contract wrong by a constant factor, silently.
    """
    from backtest.specs import verify_specs

    problems = verify_specs()
    bad = [s for s in M.SYMBOLS if s in set(problems["symbol"])]
    assert not bad, f"declared symbols with spec problems: {bad}"


def test_the_ast_validator_objects_only_to_the_event_calendar_import():
    """
    The module deliberately imports `backtest.event_calendar`, which puts it
    outside `ALLOWED_IMPORTS`. That exception is granted for ONE import, and
    pinning the FULL objection list is what keeps it from covering a later edit
    reaching for `open`, `eval` or a network library.
    """
    from agents.tier3_workers import _audit_ast

    problems = sorted(set(_audit_ast(ast.parse(MODULE_PATH.read_text()))))
    assert problems == ["import from 'backtest.event_calendar' is not allowed"]


def test_the_walk_kernel_is_identical_to_its_siblings():
    """
    `_walk_loop` is duplicated verbatim across this directory and
    `tests/test_risk_params.py` requires identical output. Run both copies on
    the same arrays; a divergence means one was "improved" alone.
    """
    sibling_path = (REPO_ROOT / "strategies" / "experimental"
                    / "sma_momentum_crossover_20260818.py")
    spec = importlib.util.spec_from_file_location("_sibling", sibling_path)
    sibling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sibling)

    rng = np.random.default_rng(4)
    n = 500
    close = 60.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    args = (
        rng.random(n) < 0.05,                    # long_entry_ok
        rng.random(n) < 0.05,                    # short_entry_ok
        rng.random(n) < 0.03,                    # long_sig_exit
        rng.random(n) < 0.03,                    # short_sig_exit
        close,                                   # open
        close + rng.uniform(0.1, 0.5, n),        # high
        close - rng.uniform(0.1, 0.5, n),        # low
        np.full(n, 0.4),                         # atr
        np.zeros(n, dtype=bool),                 # flat_bar
        1.5, 3.0, True,
    )
    mine = M._walk_loop(*args)
    theirs = sibling._walk_loop(*args)
    for i, (a, b) in enumerate(zip(mine, theirs)):
        np.testing.assert_array_equal(
            a, b, err_msg=f"walk output {i} diverged from the shared kernel")


# --------------------------------------------------------------------------
# 9. Degenerate input
# --------------------------------------------------------------------------
def test_a_dead_flat_tape_produces_no_entries_and_does_not_raise():
    """
    A halted or holiday contract: zero range, zero ATR, ADX undefined. It must
    produce no trades rather than a divide-by-zero or a stop at the fill.
    """
    n = 600
    ts = pd.date_range("2024-03-04 13:00", periods=n, freq="30min", tz="UTC")
    flat = pd.DataFrame({"ts": ts, "open": 70.0, "high": 70.0, "low": 70.0,
                         "close": 70.0, "volume": 0.0})
    entries, exits = M.signal_fn(flat)
    assert not entries.any(), "an entry on a tape that never moved"
    assert not exits.any()

    features = M.ml_features(flat)
    assert np.isfinite(features.to_numpy()).all(), (
        "the flat tape produced non-finite features")


def test_a_frame_shorter_than_the_warmup_produces_no_entries():
    """ADX(14) does not exist until ~bar 27 and EMA(200) is meaningless before
    ~bar 200. A short frame is a warm-up, not an error."""
    entries, exits = M.signal_fn(_bars(n=40))
    assert not entries.any()
    assert not exits.any()


def test_a_zero_volume_column_does_not_divide_by_zero():
    bars = _bars(n=800, volume="flat")
    bars["volume"] = 0.0
    features = M.ml_features(bars)
    assert np.isfinite(features["volume_z"].to_numpy()).all()


def test_bars_with_no_volume_column_still_classify():
    """`volume` is not guaranteed by every feed. Its absence must disable the
    volume half of Layer 3, not raise."""
    bars = _bars(n=800).drop(columns=["volume"])
    entries, exits = M.signal_fn(bars, use_volume_filter=False)
    assert entries.dtype == bool and exits.dtype == bool
    features = M.ml_features(bars)
    assert np.isfinite(features.to_numpy()).all()
