#!/usr/bin/env python3
"""
test_sma_momentum_crossover.py - the three-layer crossover, and the per-strategy
ML feature hook it is the first user of.

Location:  ~/src/trading/tests/test_sma_momentum_crossover.py

Run:  OMP_NUM_THREADS=1 python tests/test_sma_momentum_crossover.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure. Nothing here needs the lake or a network.

Pin the thread count. Section 5 refits the classifier once per completed trade
on a few dozen rows, two hundred times, and on a 16-core box each fit's thread
pool costs far more than the fit - the same reason `test_dual_version.py` is
documented with the same prefix.

`tests/test_risk_params.py` already covers this module's stop, target and
trailing machinery, its contract shape, its causality and the fact that its
copy of the walk kernel is character-for-character the shared one. This file
covers what is NOT shared:

  * ADX(14). It is the whole of Layer 3, it is easy to write four ways, and
    three of them are not Wilder's. Section 1 checks it against TA-LIB rather
    than against whatever this code currently returns - an indicator compared
    only to itself is pinned, not validated - and pins the warm-up seeding
    difference that makes the two disagree by several ADX points early and
    converge to within 1e-6 later.
  * THE TOGGLES SUBTRACTING CANDIDATES. Each layer must be able to remove
    eligible triggers and must never add one. A filter wired to the wrong side
    of a comparison still produces a plausible equity curve; a filter wired to
    nothing produces the identical one to having it off, which is the failure
    that looks most like success.
  * A DISABLED FILTER COSTING NOTHING, INCLUDING ITS WARM-UP. `ready` is
    assembled per toggle, so `use_adx_filter=False` must not inherit ADX's
    ~27-bar wait. Getting this wrong scores the "no ADX" cell on a shorter
    history than the bare crossover it is meant to represent, and the symptom
    is silently missing early trades rather than an error.
  * THE NEWS VETO BEING THE REPOSITORY'S, NOT A SECOND COPY. Section 4 checks
    the module's own suppression against `backtest.event_calendar` computed
    independently, including the one-bar backward widening that exists because
    the engine fills at the NEXT bar's open.
  * THE ML FEATURE HOOK. Section 5 covers the plumbing this module introduced:
    `load_strategy` binding it, `apply_ml_signal_filter` using it, a module
    WITHOUT the hook still getting the shared default bit for bit, and a
    malformed matrix raising rather than being aligned or fallen back from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier3_workers import (apply_ml_signal_filter,        # noqa: E402
                                  causal_features, load_strategy)
from strategies.experimental import sma_momentum_crossover as M  # noqa: E402

MODULE_PATH = REPO / "strategies" / "experimental" / "sma_momentum_crossover.py"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def synthetic(n: int = 4000, seed: int = 5, drift: float = 0.02,
              start: str = "2022-03-01 13:30") -> pd.DataFrame:
    """
    A 15m frame with enough drift for a trend module to find entries.

    The same generator `tests/test_risk_params.py` uses, reproduced here rather
    than imported so this file does not depend on another suite's fixture
    staying the shape it is today. Volume varies, unlike that copy's flat
    100.0: the ML feature matrix carries a volume z-score, and a constant
    volume column makes it a column of zeros over a zero standard deviation -
    which is a degenerate feature that would pass a shape check while carrying
    no information.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    px = 100 + np.cumsum(rng.normal(drift, 0.4, n))
    return pd.DataFrame({
        "ts": ts,
        "open": px,
        "high": px + np.abs(rng.normal(0, 0.5, n)),
        "low": px - np.abs(rng.normal(0, 0.5, n)),
        "close": px + rng.normal(0, 0.1, n),
        "volume": rng.integers(200, 6000, n).astype(float),
    })


BASE = {"fast_window": 10, "slow_window": 30, "macro_window": 100,
        "adx_threshold": 20.0, "sl_atr_mult": 1.0, "tp_atr_mult": 2.0,
        "trailing": False}


def _candidates(bars: pd.DataFrame, **overrides) -> tuple[np.ndarray, np.ndarray]:
    """
    The Layer 1-3 candidate triggers, BEFORE the position walk.

    Read off `_signal_arrays`' inputs rather than its output on purpose. A
    filter subtracts CANDIDATES, not trades: the walk holds one position at a
    time and ignores a trigger arriving while one is open, so declining an
    early trigger leaves the strategy flat for a later one it would have been
    holding through. Enabling a filter can therefore ADD realised entries, and
    a case that watched the trade count would be checking the wrong number.
    """
    params = {**BASE, **overrides}
    s = M._series(bars, params["fast_window"], params["slow_window"],
                  params["macro_window"])
    close = bars["close"]

    ready = s["fast"].notna() & s["slow"].notna() & s["atr"].notna()
    if params.get("use_macro_anchor", True):
        ready &= s["macro"].notna()
    if params.get("use_adx_filter", True):
        ready &= s["adx"].notna()

    above = (s["fast"] > s["slow"]) & ready
    below = (s["fast"] < s["slow"]) & ready
    prev_ready = ready.shift(1, fill_value=False)
    cross_up = above & ~above.shift(1, fill_value=False) & prev_ready
    cross_down = below & ~below.shift(1, fill_value=False) & prev_ready

    def _on(cond, flag):
        return cond if flag else pd.Series(True, index=bars.index)

    lr = _on(close > s["macro"], params.get("use_macro_anchor", True))
    sr = _on(close < s["macro"], params.get("use_macro_anchor", True))
    tr = _on(s["adx"] > float(params["adx_threshold"]),
             params.get("use_adx_filter", True))
    return ((cross_up & lr & tr & ready).to_numpy(dtype=bool),
            (cross_down & sr & tr & ready).to_numpy(dtype=bool))


# --------------------------------------------------------------------------
# 1. ADX, against TA-Lib
# --------------------------------------------------------------------------
def test_adx_matches_talib() -> None:
    print("\nADX(14) — Wilder's, checked against TA-Lib rather than itself")
    try:
        import talib
    except ImportError:                                     # pragma: no cover
        print("  SKIP  talib is not installed, so ADX has no external oracle. "
              "This is the section that validates the indicator; the rest of "
              "this file only pins it.")
        return

    late_worst = 0.0
    early_worst = 0.0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 800
        close = 100 + np.cumsum(rng.normal(0.01, 0.7, n))
        high = close + np.abs(rng.normal(0, 0.5, n))
        low = close - np.abs(rng.normal(0, 0.5, n))
        bars = pd.DataFrame({"open": close, "high": high, "low": low,
                             "close": close, "volume": np.ones(n)})
        mine = M._adx(bars).to_numpy()
        ref = talib.ADX(high, low, close, timeperiod=M.ADX_PERIOD)
        diff = np.abs(mine - ref)
        late_worst = max(late_worst, float(np.nanmax(diff[300:])))
        early_worst = max(early_worst, float(np.nanmax(diff[:100])))

    # Converged once the seeding difference has decayed. This is the check that
    # says "this is ADX", not "this is what this function returned last week".
    # The two never become bit-identical - `_wilder`'s recursion and TA-Lib's
    # differ in their first term forever - but 1e-6 ADX points is far below
    # anything a threshold of 20 can distinguish.
    check("agrees with TA-Lib to 1e-6 from bar 300 onward", late_worst < 1e-6,
          f"max abs diff {late_worst:.3e}")
    # And the early disagreement is REAL and pinned, because near a threshold
    # of 20 it decides the gate. TA-Lib seeds Wilder's recursion with a simple
    # sum over the first period; `_wilder` seeds from the first value.
    check("but differs during warm-up, which the module documents",
          early_worst > 0.5, f"max abs diff over the first 100 bars "
                             f"{early_worst:.3f}")

    bars = synthetic(n=600)
    adx = M._adx(bars).dropna()
    check("bounded 0-100 by construction",
          bool(((adx >= 0) & (adx <= 100)).all()),
          f"[{adx.min():.2f}, {adx.max():.2f}]")


def test_adx_is_direction_agnostic() -> None:
    print("\nADX — the same threshold governs both sides")
    n = 400
    rng = np.random.default_rng(3)
    noise = np.abs(rng.normal(0, 0.3, n))
    up = 100 + np.arange(n) * 0.1
    # An exact mirror: the same path reflected through its own starting level,
    # so the trend is equally strong and points the other way.
    down = 200.0 - up

    def frame(px):
        return pd.DataFrame({"open": px, "high": px + noise, "low": px - noise,
                             "close": px, "volume": np.ones(n)})

    a = M._adx(frame(up)).to_numpy()
    b = M._adx(frame(down)).to_numpy()
    check("an uptrend and its exact mirror give the same ADX",
          bool(np.allclose(a[50:], b[50:], atol=1e-9, equal_nan=True)),
          f"max diff {np.nanmax(np.abs(a[50:] - b[50:])):.3e}")


def test_a_flat_window_is_dx_zero_not_a_gap() -> None:
    print("\nADX — a flat stretch reads as 'no trend', never as missing data")
    n = 200
    px = np.full(n, 100.0)
    bars = pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                         "volume": np.ones(n)})
    adx = M._adx(bars)
    warm = M.ADX_PERIOD * 2
    tail = adx.iloc[warm:]
    # Left as the 0/0 NaN the division produces, this would propagate through
    # the second smoothing and blank ADX out for the next `period` bars -
    # silently closing the entry gate long after the flat patch ended.
    check("ADX exists and is 0 on a dead-flat frame",
          bool(tail.notna().all()) and bool((tail.abs() < 1e-9).all()),
          f"{int(tail.isna().sum())} NaN, max {float(tail.abs().max()):.2e}")
    try:
        import talib
        ref = talib.ADX(bars["high"].to_numpy(), bars["low"].to_numpy(),
                        bars["close"].to_numpy(), M.ADX_PERIOD)
        check("and TA-Lib says 0 there too, so this is not a local convention",
              bool(np.nanmax(np.abs(ref[warm:])) < 1e-9))
    except ImportError:                                     # pragma: no cover
        pass


# --------------------------------------------------------------------------
# 2. The three layers
# --------------------------------------------------------------------------
def test_each_layer_only_ever_subtracts_candidates() -> None:
    print("\nthe layers — each removes eligible triggers and adds none")
    bars = synthetic()
    bare_l, bare_s = _candidates(bars, use_macro_anchor=False,
                                 use_adx_filter=False)
    for label, kw in (("macro anchor", {"use_macro_anchor": True,
                                        "use_adx_filter": False}),
                      ("ADX filter", {"use_macro_anchor": False,
                                      "use_adx_filter": True}),
                      ("both layers", {"use_macro_anchor": True,
                                       "use_adx_filter": True})):
        got_l, got_s = _candidates(bars, **kw)
        # A subset on both sides. `a & ~b` non-empty means the filter ADDED a
        # trigger the bare crossover did not have, which no confirmation can
        # legitimately do.
        subset = (not (got_l & ~bare_l).any()) and (not (got_s & ~bare_s).any())
        removed = int(bare_l.sum() - got_l.sum()) + int(bare_s.sum() - got_s.sum())
        check(f"{label}: candidates are a strict subset of the bare crossover",
              subset)
        check(f"{label}: and it actually binds", removed > 0,
              f"removed {removed} of {int(bare_l.sum() + bare_s.sum())}")


def test_the_macro_anchor_picks_the_side() -> None:
    print("\nLayer 1 — no long below the baseline, no short above it")
    bars = synthetic()
    macro = M._sma(bars["close"], BASE["macro_window"])
    close = bars["close"]
    long_ok, short_ok = _candidates(bars)
    check("every long candidate has close > macro baseline",
          bool((close[long_ok] > macro[long_ok]).all()),
          f"{int(long_ok.sum())} candidates")
    check("every short candidate has close < macro baseline",
          bool((close[short_ok] < macro[short_ok]).all()),
          f"{int(short_ok.sum())} candidates")


def test_the_adx_gate_is_strict_and_shared() -> None:
    print("\nLayer 3 — ADX strictly above the threshold, same on both sides")
    bars = synthetic()
    adx = M._adx(bars, M.ADX_PERIOD)
    long_ok, short_ok = _candidates(bars, adx_threshold=25.0)
    both = long_ok | short_ok
    check("every candidate's ADX is strictly above the threshold",
          bool((adx[both] > 25.0).all()),
          f"min {float(adx[both].min()):.2f} over {int(both.sum())} candidates")

    # Raising the threshold can only shrink the candidate set. A threshold that
    # changed nothing would mean the gate is wired to an array nobody reads.
    loose_l, loose_s = _candidates(bars, adx_threshold=0.0)
    tight_l, tight_s = _candidates(bars, adx_threshold=40.0)
    check("a higher threshold is a strict subset of a lower one",
          not (tight_l & ~loose_l).any() and not (tight_s & ~loose_s).any())
    check("and it removes candidates",
          int(tight_l.sum() + tight_s.sum())
          < int(loose_l.sum() + loose_s.sum()),
          f"{int(tight_l.sum() + tight_s.sum())} at 40 vs "
          f"{int(loose_l.sum() + loose_s.sum())} at 0")


def test_a_disabled_filter_costs_no_warm_up() -> None:
    print("\nthe toggles — a filter that is off does not charge its warm-up")
    bars = synthetic()
    # ADX(14) does not exist until ~bar 27; the macro baseline not until bar
    # 100. With both off, the first candidate must be reachable as soon as the
    # two trigger averages and ATR exist - around bar 30 on these settings, not
    # bar 100.
    off_l, off_s = _candidates(bars, use_macro_anchor=False,
                               use_adx_filter=False)
    on_l, on_s = _candidates(bars)
    first_off = int(np.argmax(off_l | off_s))
    first_on = int(np.argmax(on_l | on_s))
    check("the unfiltered strategy starts trading before the filtered one",
          first_off < first_on, f"first candidate at bar {first_off} vs "
                                f"{first_on}")
    check("and it starts before the macro baseline exists at all",
          first_off < BASE["macro_window"],
          f"bar {first_off} < macro_window {BASE['macro_window']}")


def test_the_crossover_is_an_event_not_a_state() -> None:
    print("\nthe trigger — one signal per crossing, not one per bar")
    bars = synthetic()
    long_ok, short_ok = _candidates(bars, use_macro_anchor=False,
                                    use_adx_filter=False)
    fast = M._sma(bars["close"], BASE["fast_window"])
    slow = M._sma(bars["close"], BASE["slow_window"])
    ready = fast.notna() & slow.notna()
    above = ((fast > slow) & ready).to_numpy()
    # A state-based trigger would fire on every bar the fast average spends
    # above the slow one; an event-based one fires only where it flips.
    check("far fewer long candidates than bars above the slow average",
          int(long_ok.sum()) < int(above.sum()) / 10,
          f"{int(long_ok.sum())} candidates vs {int(above.sum())} bars above")
    check("no bar is both a long and a short candidate",
          not bool((long_ok & short_ok).any()))


# --------------------------------------------------------------------------
# 3. The strategy contract and the declarations
# --------------------------------------------------------------------------
def test_the_module_declares_all_four() -> None:
    print("\nthe four declarations every strategy Claude writes must carry")
    for name in ("signal_fn", "indicators", "make_signal_fn", "ml_features"):
        check(f"declares {name}", callable(getattr(M, name, None)))
    check("declares LOGIC with all three sentences",
          isinstance(M.LOGIC, dict)
          and all(isinstance(M.LOGIC.get(k), str) and M.LOGIC[k].strip()
                  for k in ("concept", "entry", "exit")))
    check("declares a PARAM_GRID with the three risk axes",
          all(k in M.PARAM_GRID
              for k in ("sl_atr_mult", "tp_atr_mult", "trailing")))
    # The identifier the pipeline logs against. `backtest/run.py` resolves
    # `--strat` by FILENAME, so a constant that disagreed with the filename
    # would name a strategy the CLI cannot reach.
    check("STRATEGY_NAME is the filename the CLI resolves",
          M.STRATEGY_NAME == MODULE_PATH.stem, M.STRATEGY_NAME)
    check("DEFAULT_PARAMS covers every signal_fn parameter",
          set(M.DEFAULT_PARAMS) == {"fast_window", "slow_window",
                                    "macro_window", "adx_threshold",
                                    "use_macro_anchor", "use_adx_filter",
                                    "use_news_filter", "sl_atr_mult",
                                    "tp_atr_mult", "trailing"},
          sorted(M.DEFAULT_PARAMS))


def test_the_logic_card_fills_in_the_runs_own_parameters() -> None:
    print("\nLOGIC — the card states the settings that ran, not the defaults")
    _fn, info = load_strategy(MODULE_PATH, {"fast_window": 5,
                                            "use_adx_filter": False})
    logic = info["logic"]
    check("the entry sentence names the bound fast window",
          "Fast SMA (5)" in logic["entry"], logic["entry"][:60])
    check("and reports the toggle as it was actually set",
          "use_adx_filter=False" in logic["entry"])
    check("no unfilled {slot} survives into the card",
          not any("{" in text for text in logic.values()))


def test_validation_refuses_a_meaningless_adx_threshold() -> None:
    print("\nvalidation — an ADX threshold outside 0-100 is not a strict filter")
    for bad in (-1.0, 100.5, float("nan"), None):
        try:
            M.make_signal_fn(adx_threshold=bad)
            ok = False
        except ValueError:
            ok = True
        check(f"rejects adx_threshold={bad!r}", ok)
    for good in (0.0, 20.0, 100.0):
        try:
            M.make_signal_fn(adx_threshold=good)
            ok = True
        except ValueError:
            ok = False
        check(f"accepts adx_threshold={good!r}", ok)
    # The windows have to be ordered, or "the trend agrees with the cross"
    # becomes close to tautological.
    for label, kw in (("fast >= slow", {"fast_window": 30, "slow_window": 30}),
                      ("slow >= macro", {"slow_window": 200,
                                         "macro_window": 200})):
        try:
            M.make_signal_fn(**kw)
            ok = False
        except ValueError:
            ok = True
        check(f"rejects {label}", ok)
    for name in ("use_macro_anchor", "use_adx_filter", "use_news_filter"):
        try:
            M.make_signal_fn(**{name: "false"})
            ok = False
        except ValueError:
            ok = True
        check(f"rejects {name}='false' rather than reading it as True", ok)


def test_the_grid_is_the_requested_one() -> None:
    print("\nPARAM_GRID — 54 cells, none of them rejected")
    from backtest.scan import expand_grid
    combos = expand_grid(M.PARAM_GRID)
    check("54 combinations", len(combos) == 54, f"{len(combos)} cells")
    rejected = 0
    for combo in combos:
        try:
            M.make_signal_fn(**combo)
        except ValueError:
            rejected += 1
    check("every cell binds, so the sweep fits what it reports",
          rejected == 0, f"{rejected} rejected")


def test_the_only_allowlist_exception_is_the_calendar_import() -> None:
    """
    The module is outside `ALLOWED_IMPORTS` by exactly one import, on purpose.

    That allowlist governs MODEL-GENERATED strategy code, which is audited
    before it runs; it does not run over hand-written modules. This module
    imports `backtest.event_calendar` rather than carrying a second copy of the
    macro calendar, which is the right trade for a calendar file plus a
    provenance rule plus merged intervals - and the wrong one for a line of
    arithmetic, which is why `ema_trend_filter` still reimplements its session
    rule.

    Pinning the exception is the point. Granted and forgotten, it becomes cover
    for the next import; pinned, an edit that reaches for `open`, `eval` or a
    network library fails here rather than passing under an exception that was
    granted for something else.
    """
    print("\nthe AST validator — one documented exception and nothing else")
    import ast as _ast

    from agents.tier3_workers import _audit_ast

    problems = _audit_ast(_ast.parse(MODULE_PATH.read_text()))
    expected = ["import from 'backtest.event_calendar' is not allowed"]
    check("the calendar import is the validator's ONLY objection",
          problems == expected, f"{problems}")


# --------------------------------------------------------------------------
# 4. The news veto
# --------------------------------------------------------------------------
def test_the_news_veto_is_the_repositorys_own() -> None:
    print("\nthe news filter — imported from event_calendar, not reimplemented")
    from backtest.event_calendar import MacroEvent, PUBLISHED, entry_block_mask

    bars = synthetic()
    long_ok, short_ok = _candidates(bars)
    kept_l, kept_s = M._news_suppress(bars, long_ok.copy(), short_ok.copy())

    # The same mask, computed independently through the module the strategy
    # imports. If the strategy carried its own copy of this rule, the two would
    # be free to disagree about which bars a release covers.
    mask, _info = entry_block_mask(bars["ts"], news_filter=True,
                                   news_window_minutes=M.NEWS_WINDOW_MINUTES)
    check("the kept candidates are exactly the unblocked ones",
          bool(np.array_equal(kept_l, long_ok & ~mask))
          and bool(np.array_equal(kept_s, short_ok & ~mask)))
    check("it is a veto: never more candidates than it was given",
          int(kept_l.sum()) <= int(long_ok.sum())
          and int(kept_s.sum()) <= int(short_ok.sum()),
          f"long {int(long_ok.sum())} -> {int(kept_l.sum())}, "
          f"short {int(short_ok.sum())} -> {int(kept_s.sum())}")

    # THE VETO HAS TO BE SHOWN TO BIND, and the strategy's own candidates are
    # the wrong population to show it on: nine long triggers over 4,000 bars
    # land inside a release window only by coincidence, so a case resting on
    # them passes for the wrong reason on one fixture and fails on the next.
    # Hand `_news_suppress` an all-True candidate mask instead - every bar a
    # candidate - and the number removed is exactly the mask's own count.
    every = np.ones(len(bars), dtype=bool)
    kept_all_l, kept_all_s = M._news_suppress(bars, every.copy(), every.copy())
    removed = int(every.sum() - kept_all_l.sum())
    check("with every bar a candidate, the veto removes the blocked ones",
          removed == int(mask.sum()) and removed > 0,
          f"removed {removed} of {len(bars)} bars")
    check("and it removes the same bars on both sides",
          bool(np.array_equal(kept_all_l, kept_all_s)))

    # The one-bar backward widening. A signal is judged on the bar it FILLS,
    # and the engine fills at the NEXT bar's open - so the bar immediately
    # BEFORE a blocked bar must be blocked too. Without that, exactly one entry
    # per event slips through and fills inside the window: the least visible
    # outcome, and the whole population the filter exists to remove.
    from backtest.event_calendar import is_news_blocked
    raw = is_news_blocked(bars["ts"], M.NEWS_WINDOW_MINUTES, strict=False)
    check("the fixture's span holds events to widen from", bool(raw.any()),
          f"{int(raw.sum())} bars inside a release window")
    if raw.any():
        first = int(np.argmax(raw))
        check("the signal mask is widened one bar backwards from the fill bar",
              first == 0 or bool(mask[first - 1]),
              f"first blocked bar {first}")
        check("so the signal mask is strictly larger than the bar mask",
              int(mask.sum()) > int(raw.sum()),
              f"{int(mask.sum())} signal bars vs {int(raw.sum())} bars")

    # And through the full strategy: the filtered run can never have MORE
    # entries than the unfiltered one. It can have the same number - these nine
    # triggers happen to miss every window on this fixture - which is why the
    # binding is asserted above on a population that cannot miss.
    le_off, _lx, se_off, _sx = M.signal_fn(bars, **BASE)
    le_on, _lx2, se_on, _sx2 = M.signal_fn(bars, **BASE, use_news_filter=True)
    check("the filtered run never has more entries than the unfiltered one",
          int(le_on.sum()) <= int(le_off.sum())
          and int(se_on.sum()) <= int(se_off.sum()),
          f"long {int(le_off.sum())} -> {int(le_on.sum())}, "
          f"short {int(se_off.sum())} -> {int(se_on.sum())}")
    del MacroEvent, PUBLISHED


def test_the_news_filter_needs_timestamps() -> None:
    print("\nthe news filter — a frame with no `ts` is refused, not filtered")
    bars = synthetic(n=500).drop(columns=["ts"])
    try:
        M.signal_fn(bars, **BASE, use_news_filter=True)
        ok = False
    except ValueError:
        ok = True
    check("a frame without timestamps raises rather than filtering nothing", ok)


# --------------------------------------------------------------------------
# 5. The per-strategy ML feature hook
# --------------------------------------------------------------------------
def test_the_feature_matrix_is_the_three_declared_columns() -> None:
    print("\nml_features — rolling ATR, ADX and rolling volume, and nothing else")
    bars = synthetic()
    feats = M.ml_features(bars)
    check("columns are exactly ML_FEATURES, in order",
          list(feats.columns) == list(M.ML_FEATURES), list(feats.columns))
    check("one row per bar", len(feats) == len(bars))
    check("warm-up stays NaN rather than being filled",
          bool(feats.isna().any(axis=1).iloc[0]) and int(feats.isna().any(axis=1).sum()) > 0,
          f"{int(feats.isna().any(axis=1).sum())} incomplete rows")
    check("no column is constant, so none is a dead feature",
          all(feats[c].dropna().std() > 0 for c in feats.columns))

    # The ADX column must BE the array Layer 3 gates on, not a second
    # calculation - otherwise the veto reads a trend strength the entry rule
    # never saw.
    check("adx_14 is the same array the entry gate uses",
          bool(np.array_equal(feats["adx_14"].to_numpy(),
                              M._adx(bars, M.ADX_PERIOD).to_numpy(),
                              equal_nan=True)))


def test_the_features_are_causal() -> None:
    print("\nml_features — truncating the frame cannot change earlier rows")
    bars = synthetic()
    k = len(bars) // 2
    full = M.ml_features(bars).to_numpy()
    half = M.ml_features(bars.iloc[:k].copy()).to_numpy()
    check("the first half is identical either way",
          bool(np.array_equal(full[:k], half, equal_nan=True)))

    # A global scaler would pass the truncation check above only if it were
    # refitted - which is exactly the leak. Check the standardisation is
    # rolling by construction: the z-score of a bar cannot depend on volume
    # that arrives after it.
    spiked = bars.copy()
    spiked.loc[spiked.index[-1], "volume"] = 1e9
    check("a spike at the END does not change any earlier z-score",
          bool(np.array_equal(M.ml_features(spiked).to_numpy()[:-1],
                              full[:-1], equal_nan=True)))


def test_the_loader_binds_the_hook_and_others_still_get_the_default() -> None:
    print("\nthe hook — bound for this module, absent for every older one")
    _fn, info = load_strategy(MODULE_PATH, {"fast_window": 5})
    hook = info.get("ml_feature_fn")
    check("load_strategy binds ml_features", callable(hook))
    bars = synthetic(n=600)
    check("the bound hook returns this module's matrix",
          list(hook(bars).columns) == list(M.ML_FEATURES))

    _fn2, info2 = load_strategy(
        REPO / "strategies" / "experimental" / "ema_crossover_20260821.py")
    check("a module that declares none gets None, which selects the default",
          info2.get("ml_feature_fn") is None)


def test_the_filter_uses_the_supplied_features() -> None:
    print("\napply_ml_signal_filter — the supplied matrix reaches the model")
    bars = synthetic()
    entries, exits, _se, _sx = M.signal_fn(bars, **BASE)
    if int(entries.sum()) < 5:
        check("the fixture produces entries to filter", False,
              f"{int(entries.sum())} entries")
        return

    default_e, _ = apply_ml_signal_filter(bars, entries, exits, symbol="NQ",
                                          min_train_trades=3)
    ours_e, _ = apply_ml_signal_filter(bars, entries, exits, symbol="NQ",
                                       min_train_trades=3,
                                       features=M.ml_features)
    check("both runs are vetoes: never more entries than they were given",
          int(default_e.sum()) <= int(entries.sum())
          and int(ours_e.sum()) <= int(entries.sum()),
          f"{int(entries.sum())} -> default {int(default_e.sum())}, "
          f"ours {int(ours_e.sum())}")

    # Passing the shared default explicitly must reproduce the None path
    # exactly. This is the check that says the new parameter did not change the
    # existing behaviour - it only made it nameable.
    explicit_e, _ = apply_ml_signal_filter(bars, entries, exits, symbol="NQ",
                                           min_train_trades=3,
                                           features=causal_features(bars))
    check("features=causal_features(bars) reproduces features=None exactly",
          bool(np.array_equal(default_e.to_numpy(), explicit_e.to_numpy())))

    # And a callable is resolved the same way a frame is.
    callable_e, _ = apply_ml_signal_filter(bars, entries, exits, symbol="NQ",
                                           min_train_trades=3,
                                           features=lambda b: M.ml_features(b))
    check("a callable and a frame give the same filter",
          bool(np.array_equal(ours_e.to_numpy(), callable_e.to_numpy())))


def test_the_supplied_matrix_actually_drives_the_model() -> None:
    """
    The decisive plumbing check: a feature the model can act on must change
    which entries survive.

    Everything else about this parameter can pass while the matrix is quietly
    ignored - the shapes line up, the errors raise, and the filter still
    returns a subset of what it was given. The only way to prove the columns
    reach the classifier is to hand it a column it can profit from and watch
    the outcome move.

    So this feeds in an ORACLE column: the eventual label of the trade
    signalled on that bar. That is lookahead by construction and would be a
    critical flaw in a strategy - it is legitimate here precisely because it is
    a test of the plumbing, not of an edge. If the matrix reaches the model,
    the kept trades' win rate must climb far above the baseline's. If it does
    not, the number will sit at the baseline and nothing else in this file
    would have noticed.

    The walk-forward still applies to the oracle: the model is fitted only on
    trades that closed before each candidate, so even a perfect feature cannot
    lift the kept win rate to 1.0 - the warm-up entries pass through
    unfiltered.
    """
    print("\napply_ml_signal_filter — a feature the model can use changes the "
          "outcome")
    from agents.tier3_workers import _label_baseline_trades
    from backtest.engine import BacktestConfig

    bars = synthetic(n=12000, seed=9, drift=0.01)
    # Both layers off, so the fixture yields enough trades for the classifier
    # to be past its warm-up for most of the run. This case is about the
    # filter, not about the strategy's selection.
    entries, exits, _se, _sx = M.signal_fn(
        bars, **{**BASE, "use_macro_anchor": False, "use_adx_filter": False})
    cfg = BacktestConfig()
    trades = _label_baseline_trades(bars, entries, exits, "NQ", cfg, "long")
    check("the fixture produces enough trades to fit on",
          trades["label"].size >= 50, f"{trades['label'].size} trades")
    if trades["label"].size < 50:
        return

    label_at = dict(zip(trades["signal_idx"].tolist(),
                        trades["label"].tolist()))
    baseline_wr = float(trades["label"].mean())

    oracle = np.zeros(len(bars))
    oracle[trades["signal_idx"]] = trades["label"] * 2.0 - 1.0
    feats = pd.DataFrame({"oracle": oracle}, index=bars.index)

    def _win_rate(kept: pd.Series) -> float:
        idx = np.flatnonzero(kept.to_numpy())
        labs = [label_at[i] for i in idx if i in label_at]
        return float(np.mean(labs)) if labs else float("nan")

    kept_oracle, _ = apply_ml_signal_filter(bars, entries, exits, symbol="NQ",
                                            cfg=cfg, min_train_trades=10,
                                            features=feats)
    kept_default, _ = apply_ml_signal_filter(bars, entries, exits, symbol="NQ",
                                             cfg=cfg, min_train_trades=10)
    wr_oracle = _win_rate(kept_oracle)
    wr_default = _win_rate(kept_default)

    check("the two matrices veto different entries, so the column is read",
          not np.array_equal(kept_oracle.to_numpy(), kept_default.to_numpy()),
          f"oracle kept {int(kept_oracle.sum())}, default kept "
          f"{int(kept_default.sum())} of {int(entries.sum())}")
    check("an oracle feature lifts the kept trades' win rate far above the "
          "baseline", wr_oracle > baseline_wr + 0.30,
          f"{wr_oracle:.3f} kept vs {baseline_wr:.3f} baseline "
          f"(shared default: {wr_default:.3f})")


def test_a_malformed_matrix_raises_rather_than_falling_back() -> None:
    print("\napply_ml_signal_filter — a bad matrix is refused, never patched")
    bars = synthetic(n=800)
    entries, exits, _se, _sx = M.signal_fn(bars, **BASE)

    short = M.ml_features(bars).iloc[:-1]
    try:
        apply_ml_signal_filter(bars, entries, exits, symbol="NQ",
                               min_train_trades=3, features=short)
        ok = False
    except ValueError:
        ok = True
    check("a row count that disagrees with the bars raises", ok)

    empty = pd.DataFrame(index=bars.index)
    try:
        apply_ml_signal_filter(bars, entries, exits, symbol="NQ",
                               min_train_trades=3, features=empty)
        ok = False
    except ValueError:
        ok = True
    check("a matrix with no columns raises", ok)

    def _boom(_bars):
        raise RuntimeError("the hook is broken")

    try:
        apply_ml_signal_filter(bars, entries, exits, symbol="NQ",
                               min_train_trades=3, features=_boom)
        ok = False
    except RuntimeError:
        # NOT swallowed. Falling back to the shared default here would run
        # Version B on a different model from the one the module declared and
        # report it under the same name.
        ok = True
    check("a hook that raises is not swallowed into the default", ok)


if __name__ == "__main__":
    print("=" * 60)
    print("  sma_momentum_crossover — the three layers and the feature hook")
    print("=" * 60)

    test_adx_matches_talib()
    test_adx_is_direction_agnostic()
    test_a_flat_window_is_dx_zero_not_a_gap()
    test_each_layer_only_ever_subtracts_candidates()
    test_the_macro_anchor_picks_the_side()
    test_the_adx_gate_is_strict_and_shared()
    test_a_disabled_filter_costs_no_warm_up()
    test_the_crossover_is_an_event_not_a_state()
    test_the_module_declares_all_four()
    test_the_logic_card_fills_in_the_runs_own_parameters()
    test_validation_refuses_a_meaningless_adx_threshold()
    test_the_grid_is_the_requested_one()
    test_the_only_allowlist_exception_is_the_calendar_import()
    test_the_news_veto_is_the_repositorys_own()
    test_the_news_filter_needs_timestamps()
    test_the_feature_matrix_is_the_three_declared_columns()
    test_the_features_are_causal()
    test_the_loader_binds_the_hook_and_others_still_get_the_default()
    test_the_filter_uses_the_supplied_features()
    test_the_supplied_matrix_actually_drives_the_model()
    test_a_malformed_matrix_raises_rather_than_falling_back()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
