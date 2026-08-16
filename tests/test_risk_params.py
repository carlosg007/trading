#!/usr/bin/env python3
"""
test_risk_params.py - the swept TP / SL / trailing machinery, end to end.

Location:  ~/src/trading/tests/test_risk_params.py

Run:  python tests/test_risk_params.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure. Nothing here needs the lake or a network.

What this is defending against
------------------------------
Adding a take-profit and a fixed-stop mode to a strategy's position walk opens
a specific set of silent failures. Every one of them produces a plausible
equity curve:

  * ANCHORING THE FIXED STOP OR THE TARGET ON THE SIGNAL BAR'S CLOSE rather
    than on the fill bar's open. The engine fills at the next bar's open, so
    the signal bar's close is a price the position never traded at. The error
    is a fraction of a bar's range and it biases every trade the same way.
  * TREATING `tp_atr_mult=None` AS A NUMBER. A sentinel of 0 exits on the fill
    bar; a sentinel of 1e9 is a target the search could in principle reach.
    NaN is the only value that disables the comparison arithmetically, and the
    interpreted fallback and the numba build must agree that it does.
  * A `trailing` FLAG THAT DOES NOT CHANGE THE STOP. Truthiness would accept
    the string "false" as True; a mis-wired branch would leave both modes
    trailing. Either way the sweep reports two distinct columns that ran the
    same simulation, and half the grid is wasted while looking full.
  * THE TWO MODULES' WALKS DRIFTING APART. `ema_crossover` and
    `ema_trend_filter` each carry their own copy of the state machine, by the
    same convention that duplicates `_atr` and `_session_masks` across this
    directory. Copies drift. Section 3 runs both on identical arrays and
    requires identical output.
  * A `None` GRID VALUE THAT CANNOT BE MATCHED BACK. `tp_atr_mult: [..., None]`
    goes through a DataFrame, where None becomes NaN and `NaN == None` is
    False. The winning row would never be flagged `selected`, so the scan CSV
    would show a sweep with no winner beside a run that used one.

The hand-computed section is the load-bearing one: the levels are worked out on
paper from a frame built so the answer is known, rather than compared against
whatever the code currently returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier3_workers import load_strategy                   # noqa: E402
from backtest.scan import _same_value, expand_grid               # noqa: E402
from strategies.experimental import ema_crossover as EC          # noqa: E402
from strategies.experimental import ema_trend_filter as ETF      # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# 1. The walk, against levels computed by hand
# --------------------------------------------------------------------------
def _walk_fixture() -> dict:
    """
    Eight bars with ONE entry on bar 0, so every level is arithmetic.

    ATR is pinned to 1.0 everywhere, which makes a multiplier and a price
    distance the same number and keeps the hand-computed answers readable.

        bar   open   high    low
         0    100    100    100    entry signal fires here
         1    110    112    111    FILL at open = 110
         2    110    115    111
         3    110    114    109
         4    110    113    108
         5    110    112    107
         6    110    111    106
         7    110    111    105

    Worked out on paper, with sl = 2.0 (so the distance is 2.0):

        FIXED    level = 110 - 2 = 108 on every bar. Lows are 111, 111, 109,
                 108 -> first breach on BAR 4.
        TRAILING bar 1: hw = 112, level = 110, low 111 -> no breach.
                 bar 2: hw = 115, level = 113, low 111 -> BREACH on bar 2.
        TARGET   tp = 4.0 -> 110 + 4 = 114. Highs are 112, 115 -> fires on
                 BAR 2.

    The entry signal is on bar 0 and the fill is bar 1's open, 110. The signal
    bar's own prices are all 100, so an implementation anchoring there puts the
    fixed level at 98 and NEVER exits in this fixture — an off-by-one that
    cannot be mistaken for a rounding difference.
    """
    o = np.array([100., 110., 110., 110., 110., 110., 110., 110.])
    h = np.array([100., 112., 115., 114., 113., 112., 111., 111.])
    l = np.array([100., 111., 111., 109., 108., 107., 106., 105.])
    n = len(o)
    entry_ok = np.zeros(n, dtype=bool)
    entry_ok[0] = True
    return {
        "entry_ok": entry_ok,
        "sig_exit": np.zeros(n, dtype=bool),
        "open_": o, "high": h, "low": l,
        "atr": np.ones(n),
        "flat_bar": np.zeros(n, dtype=bool),
    }


def _run(walk, fx: dict, sl: float, tp: float, trailing: bool):
    return walk(fx["entry_ok"], fx["sig_exit"], fx["open_"], fx["high"],
                fx["low"], fx["atr"], fx["flat_bar"], sl, tp, trailing)


def test_fixed_stop_anchors_on_the_fill_price() -> None:
    print("\nfixed stop — anchored on the fill bar's open, never the signal "
          "bar's close")
    fx = _walk_fixture()
    # sl=2.0, ATR=1.0, fill price 110  ->  level 108.0, constant.
    entries, exits, stop, target = _run(EC._walk, fx, 2.0, np.nan, False)

    check("entry on the signal bar, not the fill bar",
          bool(entries[0]) and not entries[1:].any())
    check("no stop level before the fill bar", np.isnan(stop[0]))
    check("the level is fill_price - sl x ATR = 108.0, constant",
          bool(np.allclose(stop[1:5], 108.0)),
          f"got {stop[1:5]}")
    # Lows are 111, 111, 109, 108 -> bar 4 is the first to reach 108.
    check("breach on the first bar whose low reaches it (bar 4)",
          bool(exits[4]) and not exits[1:4].any(),
          f"exits at {np.flatnonzero(exits).tolist()}")
    check("no take-profit line when tp is NaN", bool(np.isnan(target[1:5]).all()))

    # The specific bug: anchoring on the signal bar (all prices 100) would put
    # the level at 98, which no bar in the fixture reaches — so the position
    # would never close at all.
    check("a signal-bar anchor would never have exited, and this does",
          int(exits.sum()) == 1 and int(np.flatnonzero(exits)[0]) == 4)


def test_trailing_stop_ratchets_from_the_fill_bar() -> None:
    print("\ntrailing stop — ratchets with the high since the FILL bar")
    fx = _walk_fixture()
    # sl=2.0. hw after bar 1 = 112 -> level 110. After bar 2, hw=115 -> 113.
    _e, exits, stop, _t = _run(EC._walk, fx, 2.0, np.nan, True)

    check("level on the fill bar is high(bar1) - 2 = 110.0",
          bool(np.isclose(stop[1], 110.0)), f"got {stop[1]}")
    check("level ratchets up to high(bar2) - 2 = 113.0 on bar 2",
          bool(np.isclose(stop[2], 113.0)), f"got {stop[2]}")
    # bar 1's low is 111 > 110, so the fill bar survives; bar 2's low is 111,
    # which the ratcheted 113 now cuts through.
    check("survives the fill bar, then exits on bar 2 once the level ratchets",
          bool(exits[2]) and not bool(exits[1]),
          f"exits at {np.flatnonzero(exits).tolist()}")
    check("the high-water mark starts on the FILL bar, not the signal bar",
          bool(np.isclose(stop[1], 110.0)),
          "the signal bar's high is 100; starting there would give hw=100 and "
          "a level of 98, which nothing in the fixture reaches")

    # Trailing and fixed must actually differ, and here they exit on different
    # bars entirely: trailing on 2, fixed on 4.
    _e2, exits_fixed, stop_fixed, _t2 = _run(EC._walk, fx, 2.0, np.nan, False)
    check("trailing=True and trailing=False produce different stop paths",
          not bool(np.allclose(stop[1:3], stop_fixed[1:3])),
          f"trailing {stop[1:3]} vs fixed {stop_fixed[1:3]}")
    check("and different exit bars — 2 trailing, 4 fixed",
          int(np.flatnonzero(exits)[0]) == 2
          and int(np.flatnonzero(exits_fixed)[0]) == 4)


def test_take_profit_anchors_and_fires() -> None:
    print("\ntake-profit — fill price + tp x ATR, and None disables it")
    fx = _walk_fixture()
    # tp=4.0, fill 110 -> target 114. Bar 1 high 112 < 114; bar 2 high 115.
    # A wide stop (sl=20 -> level 90, and the lowest low is 105) keeps the
    # stop out of the way so the target is the only thing that can fire.
    _e, exits, _s, target = _run(EC._walk, fx, 20.0, 4.0, False)
    check("target is fill_price + tp x ATR = 114.0",
          bool(np.isclose(target[1], 114.0)), f"got {target[1]}")
    check("fires on the first bar whose HIGH reaches it (bar 2)",
          bool(exits[2]) and not bool(exits[1]),
          f"exits at {np.flatnonzero(exits).tolist()}")

    # tp=NaN: the same wide stop, and now nothing should exit at all within
    # the fixture, because no low reaches 90 and no signal exit fires.
    _e2, exits2, _s2, target2 = _run(EC._walk, fx, 20.0, np.nan, False)
    check("tp=NaN disables the target entirely — no exit in the fixture",
          not exits2.any(), f"exits at {np.flatnonzero(exits2).tolist()}")
    check("tp=NaN draws no target line", bool(np.isnan(target2).all()))

    # Bar 2 breaches both: the ratcheted trailing stop is 113 against a low of
    # 111, and the target is 114 against a high of 115. A real bracket order
    # would fill at one price or the other; here both are one exit on bar 2 at
    # the next bar's open, so the race changes nothing — which is the claim the
    # module docstring makes and this pins.
    _e3, exits3, _s3, _t3 = _run(EC._walk, fx, 2.0, 4.0, True)
    check("stop and target on the same bar produce exactly one exit",
          int(exits3.sum()) == 1 and bool(exits3[2]),
          f"exits at {np.flatnonzero(exits3).tolist()}")


def test_no_exit_is_checked_on_the_signal_bar() -> None:
    print("\nthe signal bar is never an exit bar")
    fx = _walk_fixture()
    # A stop so tight that the signal bar itself would breach it if checked:
    # signal-bar low is 100, and any level near 100 would fire there.
    for trailing in (True, False):
        _e, exits, _s, _t = _run(EC._walk, fx, 0.5, np.nan, trailing)
        check(f"trailing={trailing}: no exit on the signal bar",
              not bool(exits[0]),
              f"exits at {np.flatnonzero(exits).tolist()}")


# --------------------------------------------------------------------------
# 2. The parameter contract
# --------------------------------------------------------------------------
def synthetic(n: int = 1500, seed: int = 5) -> pd.DataFrame:
    """A 15m frame with enough drift for a trend module to find entries."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-03-01 13:30", periods=n, freq="15min", tz="UTC")
    px = 100 + np.cumsum(rng.normal(0.02, 0.4, n))
    return pd.DataFrame({
        "ts": ts,
        "open": px,
        "high": px + np.abs(rng.normal(0, 0.5, n)),
        "low": px - np.abs(rng.normal(0, 0.5, n)),
        "close": px + rng.normal(0, 0.1, n),
        "volume": 100.0,
    })


MODULES = (
    ("ema_crossover", EC, {"fast_period": 9, "slow_period": 21}),
    ("ema_trend_filter", ETF, {"trend_period": 100, "pullback_period": 20}),
)


def test_validation_rejects_what_it_should() -> None:
    print("\nvalidation — a bad risk parameter raises rather than trading")
    for name, mod, base in MODULES:
        bad = [
            ("sl_atr_mult=0", {**base, "sl_atr_mult": 0.0}),
            ("sl_atr_mult<0", {**base, "sl_atr_mult": -1.0}),
            ("sl_atr_mult=None", {**base, "sl_atr_mult": None}),
            ("tp_atr_mult=0", {**base, "sl_atr_mult": 2.0, "tp_atr_mult": 0.0}),
            ("tp_atr_mult<0", {**base, "sl_atr_mult": 2.0, "tp_atr_mult": -3.0}),
            # Truthiness would swallow this and trail the stop for a reason
            # invisible in the leaderboard's params column.
            ("trailing='false'", {**base, "sl_atr_mult": 2.0,
                                  "trailing": "false"}),
            ("trailing=1", {**base, "sl_atr_mult": 2.0, "trailing": 1}),
        ]
        for label, params in bad:
            try:
                mod.make_signal_fn(**params)
                ok = False
            except ValueError:
                ok = True
            check(f"{name}: rejects {label}", ok)

        # And the good ones bind.
        for label, params in (("tp=None", {**base, "sl_atr_mult": 2.0,
                                           "tp_atr_mult": None}),
                              ("tp=3.0", {**base, "sl_atr_mult": 2.0,
                                          "tp_atr_mult": 3.0})):
            try:
                mod.make_signal_fn(**params)
                ok = True
            except ValueError as e:
                ok, label = False, f"{label} ({e})"
            check(f"{name}: accepts {label}", ok)


def test_risk_params_change_the_trades() -> None:
    print("\nthe risk parameters actually reach the simulation")
    bars = synthetic()
    for name, mod, base in MODULES:
        _t0, tight_again = mod.signal_fn(bars, **base, sl_atr_mult=0.5,
                                         tp_atr_mult=None, trailing=True)
        _t, wide_x = mod.signal_fn(bars, **base, sl_atr_mult=5.0,
                                   tp_atr_mult=None, trailing=True)
        _t2, tight_x = mod.signal_fn(bars, **base, sl_atr_mult=0.5,
                                     tp_atr_mult=None, trailing=True)
        check(f"{name}: a tighter stop changes the exits",
              not tight_x.equals(wide_x),
              f"{int(tight_x.sum())} vs {int(wide_x.sum())} exits")

        _e, no_tp = mod.signal_fn(bars, **base, sl_atr_mult=2.0,
                                  tp_atr_mult=None, trailing=True)
        _e2, with_tp = mod.signal_fn(bars, **base, sl_atr_mult=2.0,
                                     tp_atr_mult=1.0, trailing=True)
        check(f"{name}: adding a take-profit changes the exits",
              not no_tp.equals(with_tp),
              f"{int(no_tp.sum())} vs {int(with_tp.sum())} exits")

        _e3, trail_x = mod.signal_fn(bars, **base, sl_atr_mult=2.0,
                                     tp_atr_mult=None, trailing=True)
        _e4, fixed_x = mod.signal_fn(bars, **base, sl_atr_mult=2.0,
                                     tp_atr_mult=None, trailing=False)
        check(f"{name}: trailing=False is a different simulation",
              not trail_x.equals(fixed_x),
              f"{int(trail_x.sum())} vs {int(fixed_x.sum())} exits")
        check(f"{name}: the same parameters twice give the same exits",
              tight_x.equals(tight_again))


def test_indicators_track_the_settings() -> None:
    print("\nindicators — the drawn lines follow the risk settings")
    bars = synthetic()
    for name, mod, base in MODULES:
        no_tp = mod.indicators(bars, **base, sl_atr_mult=2.0,
                               tp_atr_mult=None, trailing=True)
        with_tp = mod.indicators(bars, **base, sl_atr_mult=2.0,
                                 tp_atr_mult=3.0, trailing=True)
        check(f"{name}: no take-profit line at all when tp is None",
              not any("Take Profit" in k for k in no_tp))
        check(f"{name}: a take-profit line when tp is set",
              any("Take Profit" in k for k in with_tp))

        fixed = mod.indicators(bars, **base, sl_atr_mult=2.0,
                               tp_atr_mult=None, trailing=False)
        check(f"{name}: the stop line is labelled Trailing or Fixed to match",
              any(k.startswith("Trailing Stop") for k in no_tp)
              and any(k.startswith("Fixed Stop") for k in fixed),
              f"{list(no_tp)[-1]!r} / {list(fixed)[-1]!r}")

        for label, series in with_tp.items():
            check(f"{name}: {label} is the full length of the frame",
                  len(series) == len(bars), f"{len(series)} vs {len(bars)}")

        # The drawn stop must be the array the exit was taken from, not a
        # second reconstruction. Same params -> byte-identical both ways.
        s, _e, _x, stop, _tg = mod._signal_arrays(
            bars, *base.values(), 2.0, None, True)
        drawn = [v for k, v in no_tp.items() if "Stop" in k][0]
        check(f"{name}: the drawn stop IS the walk's stop array",
              bool(np.array_equal(drawn.to_numpy(), stop, equal_nan=True)))


def test_causality_holds_with_the_new_exits() -> None:
    print("\ncausality — truncating the frame cannot change earlier signals")
    bars = synthetic()
    k = len(bars) // 2
    for name, mod, base in MODULES:
        for tp, tr in ((None, True), (2.5, False)):
            full, _ = mod.signal_fn(bars, **base, sl_atr_mult=2.0,
                                    tp_atr_mult=tp, trailing=tr)
            half, _ = mod.signal_fn(bars.iloc[:k].copy(), **base,
                                    sl_atr_mult=2.0, tp_atr_mult=tp,
                                    trailing=tr)
            check(f"{name}: causal with tp={tp}, trailing={tr}",
                  bool(np.array_equal(full.to_numpy()[:k - 1],
                                      half.to_numpy()[:k - 1])))


# --------------------------------------------------------------------------
# 3. The two copies of the walk must not drift
# --------------------------------------------------------------------------
def test_the_two_walks_are_the_same_machine() -> None:
    print("\nthe duplicated walks — identical output on identical arrays")
    rng = np.random.default_rng(21)
    n = 400
    px = 100 + np.cumsum(rng.normal(0, 0.5, n))
    fx = {
        "entry_ok": rng.random(n) < 0.04,
        "sig_exit": rng.random(n) < 0.03,
        "open_": px,
        "high": px + np.abs(rng.normal(0, 0.6, n)),
        "low": px - np.abs(rng.normal(0, 0.6, n)),
        "atr": np.full(n, 0.8),
        "flat_bar": rng.random(n) < 0.02,
    }
    same = True
    for sl in (1.0, 2.5):
        for tp in (np.nan, 3.0):
            for tr in (True, False):
                a = _run(EC._walk, fx, sl, tp, tr)
                b = _run(ETF._walk, fx, sl, tp, tr)
                for x, y in zip(a, b):
                    if not np.array_equal(x, y, equal_nan=True):
                        same = False
    check("ema_crossover._walk == ema_trend_filter._walk on all 8 settings",
          same)

    # And the interpreted fallback must agree with the compiled build, or a
    # box without numba silently takes different trades.
    agree = True
    for tp in (np.nan, 3.0):
        for tr in (True, False):
            for x, y in zip(_run(EC._walk, fx, 2.0, tp, tr),
                            _run(EC._walk_loop, fx, 2.0, tp, tr)):
                if not np.array_equal(x, y, equal_nan=True):
                    agree = False
    check("the numba build agrees with the interpreted fallback", agree)


# --------------------------------------------------------------------------
# 4. The grid, and the scanner's None handling
# --------------------------------------------------------------------------
def test_grids_are_reportable_and_fully_valid() -> None:
    print("\nPARAM_GRID — size, and every cell bindable")
    for name, mod, _base in MODULES:
        combos = expand_grid(mod.PARAM_GRID)
        check(f"{name}: declares a risk grid",
              all(k in mod.PARAM_GRID for k in
                  ("sl_atr_mult", "tp_atr_mult", "trailing")),
              f"{sorted(mod.PARAM_GRID)}")
        check(f"{name}: None is a searched point in tp_atr_mult",
              None in list(mod.PARAM_GRID["tp_atr_mult"]))
        # The size bound is the honesty bound. Past ~200 cells the best result
        # is a search worth arguing about rather than a measurement, and these
        # grids are deliberately kept under it.
        check(f"{name}: {len(combos)} combinations, small enough to report",
              len(combos) <= 200, f"{len(combos)} cells")

        rejected = []
        for combo in combos:
            try:
                mod.make_signal_fn(**{**mod.DEFAULT_PARAMS, **combo})
            except ValueError as e:
                rejected.append((combo, str(e)))
        check(f"{name}: every declared cell binds",
              not rejected,
              f"{len(rejected)} rejected, first: {rejected[0] if rejected else ''}")

        # The loader must accept every cell too - it rejects unknown parameter
        # names, so a grid key the signature does not take raises there.
        path = REPO / "strategies" / "experimental" / f"{name}.py"
        _fn, info = load_strategy(path, combos[0])
        check(f"{name}: load_strategy binds a grid cell",
              info["bound_params"]["sl_atr_mult"] == combos[0]["sl_atr_mult"])


def test_scan_matches_a_none_winner() -> None:
    print("\nscan — a None grid value survives the round trip through pandas")
    # The exact failure: a float column with a None becomes float64 with NaN,
    # and NaN equals nothing, so `==` cannot find the winning row.
    table = pd.DataFrame([{"tp_atr_mult": 2.5}, {"tp_atr_mult": None}])
    naive = [row.get("tp_atr_mult") == None                     # noqa: E711
             for _, row in table.iterrows()]
    check("plain == cannot match a None winner (the bug this guards)",
          not any(naive), f"{naive}")
    fixed = [_same_value(row.get("tp_atr_mult"), None)
             for _, row in table.iterrows()]
    check("_same_value matches exactly the None row", fixed == [False, True],
          f"{fixed}")

    check("_same_value: NaN matches None", _same_value(float("nan"), None))
    check("_same_value: None matches None", _same_value(None, None))
    check("_same_value: 2.5 does not match None",
          not _same_value(2.5, None))
    check("_same_value: numpy bool matches python bool",
          _same_value(np.bool_(True), True)
          and not _same_value(np.bool_(True), False))
    check("_same_value: floats compare normally",
          _same_value(2.5, 2.5) and not _same_value(2.5, 5.0))


def test_leaderboard_and_promote_carry_the_risk_settings() -> None:
    print("\nthe risk settings reach the leaderboard and meta.json")
    from backtest.promote import risk_settings
    from backtest.run import LEADERBOARD_COLUMNS, leaderboard_row, risk_columns

    for k in ("sl_atr_mult", "tp_atr_mult", "trailing"):
        check(f"leaderboard declares a {k} column", k in LEADERBOARD_COLUMNS)
    check("the declared first fifteen columns are untouched",
          LEADERBOARD_COLUMNS[:15] == [
              "timestamp", "strategy", "symbol", "tf", "params",
              "sharpe_a", "pf_a", "win_rate_a", "max_dd_a", "trades_a",
              "gate1_a", "sharpe_b", "gate1_b", "selected_version",
              "html_report"])

    bound = {"fast_period": 9, "slow_period": 21, "sl_atr_mult": 1.5,
             "tp_atr_mult": None, "trailing": False}
    cols = risk_columns(bound)
    check("a swept-off take-profit is the word None, not a blank cell",
          cols["tp_atr_mult"] == "None", f"{cols['tp_atr_mult']!r}")
    check("sl and trailing carry through as values",
          cols["sl_atr_mult"] == 1.5 and cols["trailing"] is False,
          f"{cols}")
    check("a strategy with no risk params leaves the cells blank",
          risk_columns({"fast_window": 10}) ==
          {"sl_atr_mult": None, "tp_atr_mult": None, "trailing": None})

    row = leaderboard_row("20260816_000000", "ema_crossover", "NQ", "15m",
                          {"sharpe": 1.4}, None, **risk_columns(bound))
    check("the row carries them and keeps the column order",
          list(row) == LEADERBOARD_COLUMNS
          and row["tp_atr_mult"] == "None" and row["sl_atr_mult"] == 1.5)

    # promote.py's three-state risk block.
    rs = risk_settings(bound)
    check("promote: a swept-off take-profit records as null, not absent",
          rs["tp_atr_mult"] is None and rs["sl_atr_mult"] == 1.5)
    check("promote: a strategy with no such parameter records NOT DECLARED",
          risk_settings({"fast_window": 10}) ==
          {"sl_atr_mult": "NOT DECLARED", "tp_atr_mult": "NOT DECLARED",
           "trailing": "NOT DECLARED"})


def test_promote_records_the_runs_params_not_the_modules() -> None:
    print("\npromote — meta.json records the RUN's parameters")
    from backtest.promote import promote, snapshot_params

    check("snapshot_params reads meta.params out of a metrics block",
          snapshot_params({"meta": {"params": {"sl_atr_mult": 1.5}}})
          == {"sl_atr_mult": 1.5})
    check("a snapshot without params yields {} rather than a guess",
          snapshot_params({"meta": {}}) == {} and snapshot_params(None) == {})

    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # A scan winner that is NOT the module's DEFAULT_PARAMS: a tighter
        # stop, the take-profit swept ON, and trailing swept OFF.
        won = {"fast_period": 20, "slow_period": 50, "sl_atr_mult": 1.5,
               "tp_atr_mult": 5.0, "trailing": False}
        snap = td / "dual_metrics_NQ.json"
        snap.write_text(json.dumps({
            "version_a": {"metrics": {"sharpe": 1.9,
                                      "meta": {"params": won,
                                               "variants_tested": 108}},
                          "gate_audit": {"status": "PASS", "gates": {}}}}))

        out = promote("risk_probe", "A",
                      REPO / "strategies" / "experimental" / "ema_crossover.py",
                      metrics_path=snap, commit=False, incubator=td / "inc")
        meta = out["meta"]
        check("meta.params is the winning cell, not DEFAULT_PARAMS",
              meta["params"] == won, f"{meta['params']}")
        check("params_source names the snapshot",
              "dual_metrics_NQ.json" in meta["params_source"],
              meta["params_source"])
        check("meta.risk records the winning stop, target and trailing flag",
              meta["risk"] == {"sl_atr_mult": 1.5, "tp_atr_mult": 5.0,
                               "trailing": False}, f"{meta['risk']}")

        # --params still outranks the snapshot: an operator correcting the
        # record on purpose beats a file.
        out2 = promote("risk_probe2", "A",
                       REPO / "strategies" / "experimental" / "ema_crossover.py",
                       metrics_path=snap, params={"sl_atr_mult": 3.0},
                       commit=False, incubator=td / "inc")
        check("--params wins over the snapshot",
              out2["meta"]["params"]["sl_atr_mult"] == 3.0
              and out2["meta"]["params"]["tp_atr_mult"] == 5.0,
              f"{out2['meta']['params']}")

        # And with no snapshot the old behaviour stands: the module's defaults.
        out3 = promote("risk_probe3", "A",
                       REPO / "strategies" / "experimental" / "ema_crossover.py",
                       commit=False, incubator=td / "inc")
        check("no snapshot falls back to the module's DEFAULT_PARAMS",
              out3["meta"]["params"] == EC.DEFAULT_PARAMS
              and "DEFAULT_PARAMS" in out3["meta"]["params_source"],
              out3["meta"]["params_source"])
        check("a no-take-profit default records as null, not as missing",
              out3["meta"]["risk"]["tp_atr_mult"] is None,
              f"{out3['meta']['risk']}")


if __name__ == "__main__":
    test_fixed_stop_anchors_on_the_fill_price()
    test_trailing_stop_ratchets_from_the_fill_bar()
    test_take_profit_anchors_and_fires()
    test_no_exit_is_checked_on_the_signal_bar()
    test_validation_rejects_what_it_should()
    test_risk_params_change_the_trades()
    test_indicators_track_the_settings()
    test_causality_holds_with_the_new_exits()
    test_the_two_walks_are_the_same_machine()
    test_grids_are_reportable_and_fully_valid()
    test_scan_matches_a_none_winner()
    test_leaderboard_and_promote_carry_the_risk_settings()
    test_promote_records_the_runs_params_not_the_modules()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
