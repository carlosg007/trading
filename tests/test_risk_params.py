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
  * THE SHORT SIDE BEING A SIGN FLIP THAT NOBODY CHECKED. Every failure above
    has a mirror, and a short one is harder to see because the P&L still looks
    like P&L. A short stop placed BELOW the fill exits instantly; a short
    target placed above it never fires; a short trailing stop that ratchets the
    wrong way widens instead of tightening and turns every loser into a bigger
    one. Section 1b works all four out on paper against an exact mirror of the
    long fixture, so a sign error cannot pass as a plausible drawdown.
  * A SHORT TRADE BOOKED AS A LONG. The engine stamps `direction` from
    vectorbt's own trade record; inferring it from the sign of the P&L cannot
    tell a losing long from a winning short. Section 5 runs the engine over a
    frame with a known short trade and checks the stamp, the sign of the gross
    P&L, and that a long-only strategy is unaffected by any of this.
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
        "short_entry_ok": np.zeros(n, dtype=bool),
        "sig_exit": np.zeros(n, dtype=bool),
        "short_sig_exit": np.zeros(n, dtype=bool),
        "open_": o, "high": h, "low": l,
        "atr": np.ones(n),
        "flat_bar": np.zeros(n, dtype=bool),
    }


def _short_walk_fixture() -> dict:
    """
    The EXACT MIRROR of `_walk_fixture`, one short entry on bar 0.

    Every price is reflected through the entry so the hand-computed answers
    land on the same bars as the long case. If the long fixture exits on bar 4
    fixed and bar 2 trailing, so must this one — a short side that agrees with
    the long side bar for bar is the strongest cheap evidence that it is a
    mirror rather than an approximation of one.

        bar   open   high    low
         0    100    100    100    short entry signal fires here
         1     90     89     88    FILL at open = 90
         2     90     89     85
         3     90     91     86
         4     90     92     87
         5     90     93     88
         6     90     94     89
         7     90     95     90

    ATR is 1.0 everywhere, so a multiplier and a price distance are the same
    number. Worked out on paper with sl = 2.0:

        FIXED    level = 90 + 2 = 92 on every bar. Highs are 89, 89, 91, 92 ->
                 first breach on BAR 4.
        TRAILING bar 1: lw = 88, level = 90, high 89 -> no breach.
                 bar 2: lw = 85, level = 87, high 89 -> BREACH on bar 2.
        TARGET   tp = 4.0 -> 90 - 4 = 86. Lows are 88, 85 -> fires on BAR 2.

    The signal bar's prices are all 100 again, so anchoring there puts the
    fixed level at 102 — above every high in the fixture, so the position would
    never close. A wrong-side stop is the other unmistakable failure: at 90 - 2
    = 88 the FILL BAR's own high of 89 breaches it immediately.
    """
    o = np.array([100., 90., 90., 90., 90., 90., 90., 90.])
    h = np.array([100., 89., 89., 91., 92., 93., 94., 95.])
    l = np.array([100., 88., 85., 86., 87., 88., 89., 90.])
    n = len(o)
    short_entry_ok = np.zeros(n, dtype=bool)
    short_entry_ok[0] = True
    return {
        "entry_ok": np.zeros(n, dtype=bool),
        "short_entry_ok": short_entry_ok,
        "sig_exit": np.zeros(n, dtype=bool),
        "short_sig_exit": np.zeros(n, dtype=bool),
        "open_": o, "high": h, "low": l,
        "atr": np.ones(n),
        "flat_bar": np.zeros(n, dtype=bool),
    }


def _run(walk, fx: dict, sl: float, tp: float, trailing: bool):
    """
    Call the walk and return `(entries, exits, stop, target)` for the side the
    fixture trades.

    The kernel returns six arrays; collapsing them to the side under test keeps
    every hand-computed assertion below reading the same way in both
    directions, which is the point of mirroring the fixture. `_run_raw` is
    there for the cases that need all six.
    """
    le, lx, se, sx, stop, target = _run_raw(walk, fx, sl, tp, trailing)
    if fx["short_entry_ok"].any():
        return se, sx, stop, target
    return le, lx, stop, target


def _run_raw(walk, fx: dict, sl: float, tp: float, trailing: bool):
    return walk(fx["entry_ok"], fx["short_entry_ok"],
                fx["sig_exit"], fx["short_sig_exit"],
                fx["open_"], fx["high"], fx["low"], fx["atr"],
                fx["flat_bar"], sl, tp, trailing)


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
# 1b. The SHORT side of the walk, against levels computed by hand
# --------------------------------------------------------------------------
def test_short_fixed_stop_sits_above_the_fill() -> None:
    print("\nshort fixed stop — ABOVE the fill price, and anchored on it")
    fx = _short_walk_fixture()
    # sl=2.0, ATR=1.0, fill price 90  ->  level 92.0, constant.
    entries, exits, stop, target = _run(EC._walk, fx, 2.0, np.nan, False)

    check("short entry on the signal bar, not the fill bar",
          bool(entries[0]) and not entries[1:].any())
    check("no stop level before the fill bar", np.isnan(stop[0]))
    check("the level is fill_price + sl x ATR = 92.0, constant",
          bool(np.allclose(stop[1:5], 92.0)), f"got {stop[1:5]}")
    # The sign error that matters: 90 - 2 = 88 would be breached by the fill
    # bar's own high of 89, closing the trade instantly on every entry.
    check("the stop is ABOVE the fill, not below it",
          bool(stop[1] > fx["open_"][1]), f"{stop[1]} vs fill 90.0")
    # Highs are 89, 89, 91, 92 -> bar 4 is the first to reach 92.
    check("breach on the first bar whose HIGH reaches it (bar 4)",
          bool(exits[4]) and not exits[1:4].any(),
          f"exits at {np.flatnonzero(exits).tolist()}")
    check("no take-profit line when tp is NaN", bool(np.isnan(target[1:5]).all()))
    check("a signal-bar anchor would never have exited, and this does",
          int(exits.sum()) == 1 and int(np.flatnonzero(exits)[0]) == 4)


def test_short_trailing_stop_ratchets_down() -> None:
    print("\nshort trailing stop — ratchets DOWN with the low since the FILL "
          "bar")
    fx = _short_walk_fixture()
    # sl=2.0. lw after bar 1 = 88 -> level 90. After bar 2, lw=85 -> 87.
    _e, exits, stop, _t = _run(EC._walk, fx, 2.0, np.nan, True)

    check("level on the fill bar is low(bar1) + 2 = 90.0",
          bool(np.isclose(stop[1], 90.0)), f"got {stop[1]}")
    check("level ratchets DOWN to low(bar2) + 2 = 87.0 on bar 2",
          bool(np.isclose(stop[2], 87.0)), f"got {stop[2]}")
    # The direction of the ratchet is the whole check: a short stop that moves
    # UP is widening, which is the opposite of a trailing stop.
    check("the ratchet tightens rather than widens",
          bool(stop[2] < stop[1]), f"{stop[1]} -> {stop[2]}")
    # bar 1's high is 89 < 90, so the fill bar survives; bar 2's high is 89,
    # which the ratcheted 87 now cuts through.
    check("survives the fill bar, then exits on bar 2 once the level ratchets",
          bool(exits[2]) and not bool(exits[1]),
          f"exits at {np.flatnonzero(exits).tolist()}")
    check("the low-water mark starts on the FILL bar, not the signal bar",
          bool(np.isclose(stop[1], 90.0)),
          "the signal bar's low is 100; starting there would give lw=100 and "
          "a level of 102, which nothing in the fixture reaches")

    _e2, exits_fixed, stop_fixed, _t2 = _run(EC._walk, fx, 2.0, np.nan, False)
    check("trailing=True and trailing=False produce different stop paths",
          not bool(np.allclose(stop[1:3], stop_fixed[1:3])),
          f"trailing {stop[1:3]} vs fixed {stop_fixed[1:3]}")
    check("and different exit bars — 2 trailing, 4 fixed",
          int(np.flatnonzero(exits)[0]) == 2
          and int(np.flatnonzero(exits_fixed)[0]) == 4)


def test_short_take_profit_sits_below_the_fill() -> None:
    print("\nshort take-profit — fill price MINUS tp x ATR, and None disables "
          "it")
    fx = _short_walk_fixture()
    # tp=4.0, fill 90 -> target 86. Bar 1 low 88 > 86; bar 2 low 85.
    # A wide stop (sl=20 -> level 110, and the highest high is 95) keeps the
    # stop out of the way so the target is the only thing that can fire.
    _e, exits, _s, target = _run(EC._walk, fx, 20.0, 4.0, False)
    check("target is fill_price - tp x ATR = 86.0",
          bool(np.isclose(target[1], 86.0)), f"got {target[1]}")
    check("the target is BELOW the fill, not above it",
          bool(target[1] < fx["open_"][1]), f"{target[1]} vs fill 90.0")
    check("fires on the first bar whose LOW reaches it (bar 2)",
          bool(exits[2]) and not bool(exits[1]),
          f"exits at {np.flatnonzero(exits).tolist()}")

    _e2, exits2, _s2, target2 = _run(EC._walk, fx, 20.0, np.nan, False)
    check("tp=NaN disables the target entirely — no exit in the fixture",
          not exits2.any(), f"exits at {np.flatnonzero(exits2).tolist()}")
    check("tp=NaN draws no target line", bool(np.isnan(target2).all()))

    # Bar 2 breaches both: the ratcheted trailing stop is 87 against a high of
    # 89, and the target is 86 against a low of 85. One exit, same as the long.
    _e3, exits3, _s3, _t3 = _run(EC._walk, fx, 2.0, 4.0, True)
    check("stop and target on the same bar produce exactly one exit",
          int(exits3.sum()) == 1 and bool(exits3[2]),
          f"exits at {np.flatnonzero(exits3).tolist()}")


def test_short_exits_on_the_bell_and_never_on_the_signal_bar() -> None:
    print("\nshort session flatten, and the signal bar is never an exit bar")
    for name, fixture in (("long", _walk_fixture), ("short",
                                                    _short_walk_fixture)):
        fx = fixture()
        # A wide stop and no target, so ONLY the session flatten can fire.
        fx["flat_bar"] = np.zeros(len(fx["open_"]), dtype=bool)
        fx["flat_bar"][3] = True
        _e, exits, _s, _t = _run(EC._walk, fx, 20.0, np.nan, True)
        check(f"{name}: the flat bar is the only exit, and it is bar 3",
              int(exits.sum()) == 1 and bool(exits[3]),
              f"exits at {np.flatnonzero(exits).tolist()}")

        # A stop so tight the signal bar itself would breach it if checked.
        fx2 = fixture()
        for trailing in (True, False):
            _e2, exits2, _s2, _t2 = _run(EC._walk, fx2, 0.5, np.nan, trailing)
            check(f"{name}: trailing={trailing}: no exit on the signal bar",
                  not bool(exits2[0]),
                  f"exits at {np.flatnonzero(exits2).tolist()}")


def test_the_walk_never_reverses_or_holds_both_sides() -> None:
    print("\nthe walk holds one position at a time and never reverses")
    n = 8
    both = np.ones(n, dtype=bool)
    px = np.full(n, 100.0)
    fx = {
        "entry_ok": both,
        "short_entry_ok": both,
        "sig_exit": np.zeros(n, dtype=bool),
        "short_sig_exit": np.zeros(n, dtype=bool),
        "open_": px, "high": px + 0.1, "low": px - 0.1,
        "atr": np.ones(n),
        "flat_bar": np.zeros(n, dtype=bool),
    }
    le, lx, se, sx, _s, _t = _run_raw(EC._walk, fx, 20.0, np.nan, False)
    check("a bar signalling BOTH sides while flat takes neither",
          not le.any() and not se.any(),
          f"long {int(le.sum())}, short {int(se.sum())}")
    check("and therefore books no exits either",
          not lx.any() and not sx.any())

    # Long entry on bar 0, then a short trigger on every later bar. The short
    # must be ignored while the long is open: reversing on the spot would open
    # a position with no flat bar in between, which the engine's chunking
    # assumes cannot happen.
    fx["entry_ok"] = np.zeros(n, dtype=bool)
    fx["entry_ok"][0] = True
    fx["short_entry_ok"] = np.zeros(n, dtype=bool)
    fx["short_entry_ok"][2:] = True
    le, lx, se, sx, _s, _t = _run_raw(EC._walk, fx, 20.0, np.nan, False)
    check("a short trigger while long is ignored, not a reversal",
          int(le.sum()) == 1 and int(se.sum()) == 0,
          f"long entries {np.flatnonzero(le).tolist()}, "
          f"short entries {np.flatnonzero(se).tolist()}")

    # And once the long is flattened, the next short trigger IS taken.
    fx["flat_bar"] = np.zeros(n, dtype=bool)
    fx["flat_bar"][3] = True
    le, lx, se, sx, _s, _t = _run_raw(EC._walk, fx, 20.0, np.nan, False)
    check("once flat again the short is taken on the next trigger",
          int(lx.sum()) == 1 and bool(lx[3]) and int(se.sum()) == 1
          and int(np.flatnonzero(se)[0]) == 4,
          f"long exits {np.flatnonzero(lx).tolist()}, "
          f"short entries {np.flatnonzero(se).tolist()}")


# --------------------------------------------------------------------------
# 2. The parameter contract
# --------------------------------------------------------------------------
def synthetic(n: int = 4000, seed: int = 5, drift: float = 0.02) -> pd.DataFrame:
    """
    A 15m frame with enough drift for a trend module to find entries.

    `n` is 4000 because 1500 was not enough, and the way it was not enough is
    the point. Both modules gate entries to 09:30-15:30 ET, which is a quarter
    of a 24-hour futures session, and `ema_trend_filter` additionally needs a
    crossover, the anchor trend and expanding volatility to line up on the same
    bar. At 1500 bars that yielded TWO trades, and neither of them ever reached
    a 1.0 x ATR target — so `test_risk_params_change_the_trades` compared a
    take-profit run against a no-take-profit run, got identical exits, and the
    check passed or failed on whether two arbitrary trades happened to hit a
    level rather than on whether the parameter reaches the simulation.

    A check that cannot fail when the code is broken is worse than no check.
    4000 bars gives seven entries at these settings, which is enough for the
    stop, the target and the trailing flag each to change the exit array.

    `drift` is the per-bar mean. The default +0.02 is the frame every existing
    case here was calibrated on. NEGATIVE drift is what the short side needs:
    `ema_trend_filter` only shorts below its anchor EMA, so on an up-drifting
    frame it finds one short in 4000 bars and every short assertion would be
    resting on whether that single trade happened to reach a level. Same
    generator, same seed, mirrored slope - see `SHORT_DRIFT`.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-03-01 13:30", periods=n, freq="15min", tz="UTC")
    px = 100 + np.cumsum(rng.normal(drift, 0.4, n))
    return pd.DataFrame({
        "ts": ts,
        "open": px,
        "high": px + np.abs(rng.normal(0, 0.5, n)),
        "low": px - np.abs(rng.normal(0, 0.5, n)),
        "close": px + rng.normal(0, 0.1, n),
        "volume": 100.0,
    })


# The non-risk parameters each module is bound with. Order matters as well as
# content: `test_indicators_track_the_settings` calls `_signal_arrays` with
# `*base.values()`, so these must be listed in the module's own positional
# order.
#
# `ema_trend_filter` is exercised at trend_period=100 rather than at its
# DEFAULT_PARAMS value of 800. That is a fixture decision, not a claim about
# the strategy: EMA(800) stays NaN until bar 799, so on a 4000-bar frame it
# spends a fifth of the sample warming up and yields four entries instead of
# seven. Nothing here tests the anchor length — these cases test that the risk
# parameters reach the simulation — so the shorter anchor buys trades to
# measure that on. The 800-bar default is exercised where it matters, against
# real bars, by the runner.
MODULES = (
    ("ema_crossover", EC, {"fast_period": 9, "slow_period": 21}),
    ("ema_trend_filter", ETF, {"fast_period": 9, "slow_period": 21,
                               "trend_period": 100}),
)

# Which modules trade both ways. `ema_crossover` is long only by its own
# specification and returns the two-mask form of the contract; `ema_trend_filter`
# is symmetric and returns four. Declared rather than detected, for the same
# reason `TP_NONE_SEARCHED` is: a table somebody chose fails in both directions,
# so a module quietly losing its short side is a failure here rather than a
# check that silently stops checking anything.
BIDIRECTIONAL = {
    "ema_crossover": False,
    "ema_trend_filter": True,
}

# The per-bar drift that gives the short side something to work with. See
# `synthetic`.
SHORT_DRIFT = -0.02


def _masks(mod, bars: pd.DataFrame, **params):
    """
    `(long_entries, long_exits, short_entries, short_exits)` from either
    contract form, through the engine's own unpacker.

    Going through `unpack_signals` rather than unpacking here means these cases
    accept exactly what the engine accepts - a module returning a three-tuple
    would fail in this helper the same way it fails in a run, instead of being
    silently truncated to its first two masks.
    """
    from backtest.engine import unpack_signals
    return unpack_signals(mod.signal_fn(bars, **params), len(bars))

# Does each module's PARAM_GRID include the `None` (no take-profit) point?
#
# `ema_crossover` does, and that is the convention: a sweep that never tries
# "no target" cannot tell you the target earned its place, only which target
# scored best among the ones offered.
#
# `ema_trend_filter` does NOT, by an explicit decision recorded at its
# PARAM_GRID. It matters more there than it would elsewhere, because that
# module has no signal exit at all — the stop and the target ARE its exit rule
# — so "is a target better than no target" is close to the central question
# about it, and its 162-cell grid does not ask. Answering it needs one
# out-of-band run at the winning cell with tp_atr_mult=None.
TP_NONE_SEARCHED = {
    "ema_crossover": True,
    "ema_trend_filter": False,
}


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
    for label, bars in (("long", synthetic()),
                        ("short", synthetic(drift=SHORT_DRIFT))):
        for name, mod, base in MODULES:
            # Index 1 is the long exits, index 3 the short exits. On the
            # down-drifting frame the short side is the one carrying trades, so
            # that is the array these cases have to watch - checking the long
            # exits there would be checking a handful of trades that survived a
            # falling market, which is not what the parameter is being tested
            # for.
            side = 3 if (label == "short" and BIDIRECTIONAL[name]) else 1
            if label == "short" and not BIDIRECTIONAL[name]:
                continue

            def _x(sl, tp, tr, _s=side, _m=mod, _b=base, _bars=bars):
                return _masks(_m, _bars, **_b, sl_atr_mult=sl, tp_atr_mult=tp,
                              trailing=tr)[_s]

            tag = f"{name} ({label} side)"
            tight_again = _x(0.5, None, True)
            wide_x = _x(5.0, None, True)
            tight_x = _x(0.5, None, True)
            check(f"{tag}: a tighter stop changes the exits",
                  not tight_x.equals(wide_x),
                  f"{int(tight_x.sum())} vs {int(wide_x.sum())} exits")

            no_tp = _x(2.0, None, True)
            with_tp = _x(2.0, 1.0, True)
            check(f"{tag}: adding a take-profit changes the exits",
                  not no_tp.equals(with_tp),
                  f"{int(no_tp.sum())} vs {int(with_tp.sum())} exits")

            trail_x = _x(2.0, None, True)
            fixed_x = _x(2.0, None, False)
            check(f"{tag}: trailing=False is a different simulation",
                  not trail_x.equals(fixed_x),
                  f"{int(trail_x.sum())} vs {int(fixed_x.sum())} exits")
            check(f"{tag}: the same parameters twice give the same exits",
                  tight_x.equals(tight_again))


def test_the_contract_shape_matches_what_each_module_claims() -> None:
    print("\nthe contract — two masks or four, and the short side is real")
    for name, mod, base in MODULES:
        bars = synthetic(drift=SHORT_DRIFT)
        out = mod.signal_fn(bars, **base, sl_atr_mult=2.0, tp_atr_mult=3.0,
                            trailing=True)
        want = 4 if BIDIRECTIONAL[name] else 2
        check(f"{name}: signal_fn returns {want} masks", len(out) == want,
              f"got {len(out)}")

        le, lx, se, sx = _masks(mod, bars, **base, sl_atr_mult=2.0,
                                tp_atr_mult=3.0, trailing=True)
        for lbl, m in (("long entries", le), ("long exits", lx),
                       ("short entries", se), ("short exits", sx)):
            check(f"{name}: {lbl} is a full-length bool mask",
                  len(m) == len(bars) and m.dtype == bool)

        if BIDIRECTIONAL[name]:
            # On a down-drifting frame a symmetric module must actually take
            # shorts. Zero here would mean the mirror is wired but unreachable -
            # a strategy that reports a short side and never trades it.
            check(f"{name}: takes shorts on a down-drifting frame",
                  int(se.sum()) > 0, f"{int(se.sum())} short entries")
            check(f"{name}: every short entry is matched by an exit",
                  int(se.sum()) == int(sx.sum()) or
                  int(se.sum()) == int(sx.sum()) + 1,
                  f"{int(se.sum())} entries, {int(sx.sum())} exits")
            check(f"{name}: never long and short on the same bar",
                  not bool((le & se).any()))
        else:
            check(f"{name}: declares no short side and emits none",
                  int(se.sum()) == 0 and int(sx.sum()) == 0)

        # And the short side is not merely the long side relabelled: on the
        # same bars the two masks must differ.
        if BIDIRECTIONAL[name]:
            check(f"{name}: the short entries are not a copy of the long ones",
                  not le.equals(se))


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
        s, _e, _x, se, sx, stop, _tg = mod._signal_arrays(
            bars, *base.values(), 2.0, None, True)
        drawn = [v for k, v in no_tp.items() if "Stop" in k][0]
        check(f"{name}: the drawn stop IS the walk's stop array",
              bool(np.array_equal(drawn.to_numpy(), stop, equal_nan=True)))
        # A long-only module must be returning the short pair for shape only.
        if not BIDIRECTIONAL[name]:
            check(f"{name}: _signal_arrays returns an empty short pair",
                  not se.any() and not sx.any())


def test_causality_holds_with_the_new_exits() -> None:
    print("\ncausality — truncating the frame cannot change earlier signals")
    # Both drifts, so the short side is exercised too: a lookahead bug living
    # only in the short branch would be invisible on the up-drifting frame.
    for label, bars in (("long", synthetic()),
                        ("short", synthetic(drift=SHORT_DRIFT))):
        k = len(bars) // 2
        for name, mod, base in MODULES:
            for tp, tr in ((None, True), (2.5, False)):
                full = _masks(mod, bars, **base, sl_atr_mult=2.0,
                              tp_atr_mult=tp, trailing=tr)
                half = _masks(mod, bars.iloc[:k].copy(), **base,
                              sl_atr_mult=2.0, tp_atr_mult=tp, trailing=tr)
                ok = all(np.array_equal(f.to_numpy()[:k - 1],
                                        h.to_numpy()[:k - 1])
                         for f, h in zip(full, half))
                check(f"{name} ({label} frame): causal with tp={tp}, "
                      f"trailing={tr}", ok)


# --------------------------------------------------------------------------
# 3. The two copies of the walk must not drift
# --------------------------------------------------------------------------
def test_the_two_walks_are_the_same_machine() -> None:
    print("\nthe duplicated walks — identical output on identical arrays")
    rng = np.random.default_rng(21)
    n = 400
    px = 100 + np.cumsum(rng.normal(0, 0.5, n))
    # BOTH sides signalled, densely and independently, so the comparison covers
    # the short branches, the reversal-refusal branch and the ambiguous-bar
    # branch rather than only the long path the old fixture reached.
    fx = {
        "entry_ok": rng.random(n) < 0.04,
        "short_entry_ok": rng.random(n) < 0.04,
        "sig_exit": rng.random(n) < 0.03,
        "short_sig_exit": rng.random(n) < 0.03,
        "open_": px,
        "high": px + np.abs(rng.normal(0, 0.6, n)),
        "low": px - np.abs(rng.normal(0, 0.6, n)),
        "atr": np.full(n, 0.8),
        "flat_bar": rng.random(n) < 0.02,
    }
    # The fixture is only worth what it exercises: assert it actually produces
    # trades on both sides before concluding the two copies agree about them.
    le, _lx, se, _sx, _s, _t = _run_raw(EC._walk, fx, 2.0, 3.0, True)
    check("the drift fixture trades both sides",
          int(le.sum()) > 0 and int(se.sum()) > 0,
          f"{int(le.sum())} long, {int(se.sum())} short entries")

    same = True
    for sl in (1.0, 2.5):
        for tp in (np.nan, 3.0):
            for tr in (True, False):
                a = _run_raw(EC._walk, fx, sl, tp, tr)
                b = _run_raw(ETF._walk, fx, sl, tp, tr)
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
            for x, y in zip(_run_raw(EC._walk, fx, 2.0, tp, tr),
                            _run_raw(EC._walk_loop, fx, 2.0, tp, tr)):
                if not np.array_equal(x, y, equal_nan=True):
                    agree = False
    check("the numba build agrees with the interpreted fallback", agree)

    # The copies must be identical as SOURCE too, not merely agree on one
    # fixture. Behavioural equality over 400 random bars is strong evidence and
    # not a proof - a branch neither side's fixture reaches would pass it.
    import inspect
    check("the two walk copies are character-for-character the same source",
          inspect.getsource(EC._walk_loop) == inspect.getsource(ETF._walk_loop))


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
        # `None` in the target axis is what makes "does the take-profit earn
        # its place at all?" a question the sweep ANSWERS rather than one the
        # grid assumes. It used to be required of every module here. It is now
        # declared per module, because `ema_trend_filter`'s grid was specified
        # without it deliberately and after the point was raised.
        #
        # The check is DECLARED rather than deleted on purpose. A dropped
        # assertion is indistinguishable from a convention nobody noticed
        # eroding; this way the exemption is a line of code somebody chose,
        # and it fails in both directions — adding `None` back to that grid
        # without updating this table is also a failure, so the table cannot
        # go stale while claiming to describe the grids.
        want_none = TP_NONE_SEARCHED[name]
        has_none = None in list(mod.PARAM_GRID["tp_atr_mult"])
        check(f"{name}: tp_atr_mult "
              f"{'searches' if want_none else 'deliberately omits'} the "
              f"no-take-profit point",
              has_none == want_none,
              f"None in grid={has_none}, declared={want_none}")
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


# --------------------------------------------------------------------------
# 5. The engine: the short side survives the simulation
# --------------------------------------------------------------------------
def _engine_bars(px: list[float]) -> pd.DataFrame:
    """A frame whose opens ARE the given prices. Fills happen at the open."""
    n = len(px)
    a = np.asarray(px, dtype=float)
    return pd.DataFrame({
        "ts": pd.date_range("2022-03-01 14:00", periods=n, freq="15min",
                            tz="UTC"),
        "open": a, "high": a + 1.0, "low": a - 1.0, "close": a,
        "volume": 100.0,
    })


def test_unpack_signals_accepts_both_contracts_and_refuses_the_rest() -> None:
    print("\nunpack_signals — two masks or four, and nothing in between")
    from backtest.engine import unpack_signals

    n = 5
    t = pd.Series([True] * n)
    f = pd.Series([False] * n)

    le, lx, se, sx = unpack_signals((t, f), n)
    check("a two-tuple gets all-False short masks",
          le.all() and not lx.any() and not se.any() and not sx.any())

    le, lx, se, sx = unpack_signals((t, f, f, t), n)
    check("a four-tuple passes all four through",
          le.all() and not lx.any() and not se.any() and sx.all())

    check("NaN counts as no signal",
          not unpack_signals((pd.Series([np.nan] * n), f), n)[0].any())

    for bad, why in (((t, f, f), "a three-tuple"),
                     ((t,), "a one-tuple"),
                     ((t, f, f, t, f), "a five-tuple"),
                     (t, "a bare Series")):
        try:
            unpack_signals(bad, n)
            ok = False
        except ValueError:
            ok = True
        # Silently taking the first two masks of a three-tuple is how a
        # strategy's short side disappears into a plausible long-only curve.
        check(f"rejects {why} rather than truncating it", ok)

    try:
        unpack_signals((t, f, f, pd.Series([True] * (n + 1))), n)
        ok = False
    except ValueError:
        ok = True
    check("rejects a short mask of the wrong length", ok)


def test_engine_books_a_short_with_the_right_sign_and_stamp() -> None:
    print("\nthe engine — a short trade is stamped short and signed short")
    from backtest.engine import BacktestConfig, _simulate, clean_signals_ls

    # Price falls 100 -> 90 and then recovers. The short is signalled on bar 0
    # (filled at bar 1's open, 100) and closed on bar 3 (filled at bar 4's
    # open, 90), so it makes 10 points on a falling market. A long over the
    # same bars would LOSE 10. Nothing but the direction stamp distinguishes
    # the two, which is the point.
    bars = _engine_bars([102., 100., 96., 92., 90., 95., 100., 104.])
    n = len(bars)
    z = pd.Series(np.zeros(n, dtype=bool))

    se = z.copy(); se.iloc[0] = True
    sx = z.copy(); sx.iloc[3] = True

    cfg = BacktestConfig(commission_per_side=0.0, slippage_ticks=0.0)
    le, lx, se, sx = clean_signals_ls(z, z, se, sx)
    trades = _simulate(bars, le, lx, "ES", cfg, se, sx)

    check("one closed short trade", len(trades) == 1, f"{len(trades)} trades")
    if len(trades) == 1:
        t = trades.iloc[0]
        check("stamped direction='short'", t["direction"] == "short",
              f"{t['direction']!r}")
        check("filled at the NEXT bar's open on both ends",
              t["entry_price"] == 100.0 and t["exit_price"] == 90.0,
              f"{t['entry_price']} -> {t['exit_price']}")
        # ES multiplier is 50: 10 points x 50 = 500, POSITIVE on a short.
        check("gross P&L is entry - exit, so a falling market is a WIN",
              t["gross_pnl"] > 0 and np.isclose(t["gross_pnl"], 500.0),
              f"gross {t['gross_pnl']}")
        check("net P&L matches gross with costs zeroed",
              np.isclose(t["pnl"], t["gross_pnl"]),
              f"{t['pnl']} vs {t['gross_pnl']}")

    # The same bars traded LONG must lose exactly what the short made.
    le2 = z.copy(); le2.iloc[0] = True
    lx2 = z.copy(); lx2.iloc[3] = True
    le2, lx2, _se2, _sx2 = clean_signals_ls(le2, lx2, z, z)
    longs = _simulate(bars, le2, lx2, "ES", cfg)
    check("the mirrored long loses exactly what the short made",
          len(longs) == 1 and np.isclose(longs.iloc[0]["gross_pnl"], -500.0),
          f"{longs['gross_pnl'].tolist()}")
    check("and is stamped long", longs.iloc[0]["direction"] == "long")


def test_the_four_mask_path_matches_the_long_only_path() -> None:
    print("\nthe engine — adding shorts changes nothing when there are none")
    from backtest.engine import BacktestConfig, _simulate, clean_signals_ls

    rng = np.random.default_rng(11)
    n = 300
    px = 100 + np.cumsum(rng.normal(0, 0.6, n))
    bars = _engine_bars(px.tolist())
    z = pd.Series(np.zeros(n, dtype=bool))
    e = pd.Series(rng.random(n) < 0.05)
    x = pd.Series(rng.random(n) < 0.05)
    e, x, _se, _sx = clean_signals_ls(e, x, z, z)

    cfg = BacktestConfig()
    long_only = _simulate(bars, e, x, "ES", cfg)
    # An all-False short pair still routes through the long-only branch, so
    # force the four-mask branch with a short that is signalled after the last
    # bar any long trade touches - it can never fill, so it cannot add a trade.
    tail = z.copy()
    tail.iloc[-1] = True
    with_shorts = _simulate(bars, e, x, "ES", cfg, tail, tail)

    check("the long trade list is identical either way",
          long_only.drop(columns=["direction"]).equals(
              with_shorts.drop(columns=["direction"])),
          f"{len(long_only)} vs {len(with_shorts)} trades")
    check("every trade is still stamped long",
          set(with_shorts["direction"]) == {"long"})


def test_the_legacy_oracle_agrees_with_vectorbt_on_shorts() -> None:
    print("\nthe engine — the legacy loop and vectorbt agree, short included")
    from backtest.engine import (BacktestConfig, _simulate, _simulate_legacy,
                                 clean_signals_ls)

    # `_simulate_legacy` is the obviously-correct reference `_simulate` is
    # checked against. It walks the same three states, so it is an oracle for
    # shorts too - but only if something actually compares them on a frame that
    # HAS shorts, which tests/test_engine_vbt.py's long-only fixtures do not.
    rng = np.random.default_rng(7)
    n = 400
    px = 100 + np.cumsum(rng.normal(0, 0.7, n))
    bars = _engine_bars(px.tolist())

    le = pd.Series(rng.random(n) < 0.05)
    lx = pd.Series(rng.random(n) < 0.06)
    se = pd.Series(rng.random(n) < 0.05)
    sx = pd.Series(rng.random(n) < 0.06)
    le, lx, se, sx = clean_signals_ls(le, lx, se, sx)

    # The legacy loop charges one scalar round-turn cost per trade, so the two
    # only agree on costs with slippage and commission both zeroed. What is
    # being compared is the trade LIST and the gross P&L - which bars were
    # entered and exited, on which side, at what prices.
    cfg = BacktestConfig(commission_per_side=0.0, slippage_ticks=0.0)
    fast = _simulate(bars, le, lx, "ES", cfg, se, sx).reset_index(drop=True)
    slow = _simulate_legacy(bars, le, lx, "ES", cfg, se, sx).reset_index(drop=True)

    check("the fixture produces trades on both sides",
          len(fast) > 10 and set(fast["direction"]) == {"long", "short"},
          f"{len(fast)} trades, sides {sorted(set(fast['direction']))}")
    check("the two implementations agree on the trade COUNT",
          len(fast) == len(slow), f"vbt {len(fast)} vs legacy {len(slow)}")
    if len(fast) == len(slow) and len(fast):
        cols = ["entry_time", "exit_time", "direction", "entry_price",
                "exit_price"]
        check("and trade for trade on times, side and fill prices",
              fast[cols].equals(slow[cols]),
              f"first divergence at row "
              f"{int((fast[cols] != slow[cols]).any(axis=1).idxmax())}")
        check("and on gross P&L, so the short sign matches",
              bool(np.allclose(fast["gross_pnl"], slow["gross_pnl"])),
              f"max diff "
              f"{float(np.max(np.abs(fast['gross_pnl'] - slow['gross_pnl']))):.6f}")


def test_the_ml_filter_labels_a_short_by_its_own_sign() -> None:
    print("\nthe ML filter — a short's label is entry - exit, not exit - entry")
    from agents.tier3_workers import _label_baseline_trades

    # Same falling market as above: the short WINS, so its label must be 1.
    bars = _engine_bars([102., 100., 96., 92., 90., 95., 100., 104.])
    n = len(bars)
    e = pd.Series(np.zeros(n, dtype=bool)); e.iloc[0] = True
    x = pd.Series(np.zeros(n, dtype=bool)); x.iloc[3] = True

    lng = _label_baseline_trades(bars, e, x, None, None, direction="long")
    sht = _label_baseline_trades(bars, e, x, None, None, direction="short")

    check("the long over a falling market is labelled a loss",
          lng["label"].tolist() == [0], f"{lng['label'].tolist()}")
    check("the short over the same bars is labelled a win",
          sht["label"].tolist() == [1], f"{sht['label'].tolist()}")
    # Mislabelling a side does not raise; it teaches the classifier the edge
    # inverted and Version B comes back smooth and backwards.
    check("the two labels are opposites, not copies",
          lng["label"].tolist() != sht["label"].tolist())

    try:
        _label_baseline_trades(bars, e, x, None, None, direction="both")
        ok = False
    except ValueError:
        ok = True
    check("an unrecognised direction raises rather than defaulting to long", ok)


if __name__ == "__main__":
    test_fixed_stop_anchors_on_the_fill_price()
    test_trailing_stop_ratchets_from_the_fill_bar()
    test_take_profit_anchors_and_fires()
    test_no_exit_is_checked_on_the_signal_bar()
    test_short_fixed_stop_sits_above_the_fill()
    test_short_trailing_stop_ratchets_down()
    test_short_take_profit_sits_below_the_fill()
    test_short_exits_on_the_bell_and_never_on_the_signal_bar()
    test_the_walk_never_reverses_or_holds_both_sides()
    test_validation_rejects_what_it_should()
    test_risk_params_change_the_trades()
    test_the_contract_shape_matches_what_each_module_claims()
    test_indicators_track_the_settings()
    test_causality_holds_with_the_new_exits()
    test_the_two_walks_are_the_same_machine()
    test_grids_are_reportable_and_fully_valid()
    test_scan_matches_a_none_winner()
    test_leaderboard_and_promote_carry_the_risk_settings()
    test_promote_records_the_runs_params_not_the_modules()
    test_unpack_signals_accepts_both_contracts_and_refuses_the_rest()
    test_engine_books_a_short_with_the_right_sign_and_stamp()
    test_the_four_mask_path_matches_the_long_only_path()
    test_the_legacy_oracle_agrees_with_vectorbt_on_shorts()
    test_the_ml_filter_labels_a_short_by_its_own_sign()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
