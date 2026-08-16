#!/usr/bin/env python3
"""
test_intraday_vol_mr.py - the band-fade strategy's indicators and position walk.

Location:  ~/src/trading/tests/test_intraday_vol_mr.py

Run:  python tests/test_intraday_vol_mr.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure.

Why this module gets its own suite
----------------------------------
Every other strategy here is a stateless mask over bars: compute two series,
compare them, return booleans. This one is not. It carries a hand-rolled
two-state machine, because a trailing stop's level depends on the high since
ENTRY and therefore on which earlier bar opened the position — which no mask
can express. A state machine is where off-by-one errors live, and the specific
ones available here are all silent:

  * starting the high-water mark on the SIGNAL bar rather than the FILL bar,
    which sets the first stop too far away and quietly widens every trade;
  * checking the exit on the entry bar, which closes positions that were never
    open;
  * reading a future bar's high into the stop, which is lookahead and makes the
    equity curve better rather than raising;
  * emitting overlapping entries, which `clean_signals` would silently absorb —
    hiding a bug rather than surfacing it.

None of these announce themselves. All four are checked below against
hand-computed answers, plus a truncation test that is the actual proof of
causality: signals computed on the first k bars must be byte-identical to the
same positions computed on the whole frame. A strategy that reads ahead cannot
pass that, whatever its docstring claims.

The indicators are checked against Wilder's recursion by hand, because "ATR" in
this repo has to be the ATR every other tool computes; a span EMA is roughly
twice as fast and would produce a band nobody else agrees with.

The lake is needed only for the final section, which says so and skips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.tier3_workers import load_strategy
from backtest.engine import clean_signals
from strategies.experimental import intraday_vol_mr as S

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def synthetic(n: int = 600, seed: int = 3) -> pd.DataFrame:
    """
    A 15m frame that actually produces this strategy's setup.

    A plain random walk does not. The entry needs three things at once — a low
    piercing the band, a close recovering back inside it, and RSI already
    washed out — and on a random walk that coincidence is rare enough that a
    600-bar frame yields nothing. Every invariant below would then pass
    vacuously on zero trades, which is worse than no test: "the walk never
    emits overlapping entries" is trivially true when it emits none.

    So the setup is constructed rather than hoped for: slow decline-and-recover
    cycles push RSI under the threshold, and every nineteenth bar carries a deep
    low wick whose close recovers. `test_signal_contract` asserts the frame
    traded, so this generator drifting into producing nothing fails loudly
    instead of quietly hollowing out the suite.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + 6 * np.sin(t / 23.0) - 0.004 * t + rng.normal(0, 0.25, n)
    spread = np.abs(rng.normal(0.30, 0.08, n))
    high, low = close + spread, close - spread

    dip = t % 19 == 0
    low[dip] = close[dip] - rng.uniform(2.0, 3.5, int(dip.sum()))

    return pd.DataFrame({
        "ts": pd.date_range("2022-01-03 14:30", periods=n, freq="15min",
                            tz="UTC"),
        "symbol": "NQ",
        "open": np.r_[close[0], close[:-1]],
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1_000, 10_000, n).astype(float),
    })


# --------------------------------------------------------------------------
def test_indicators() -> None:
    print("\nindicators are the ones everyone else computes")

    bars = synthetic(60)
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()

    tr = [high[0] - low[0]]
    for i in range(1, len(bars)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]),
                      abs(low[i] - close[i - 1])))
    tr = np.asarray(tr)

    atr = S._atr(bars, 14).to_numpy()
    check("ATR warm-up is NaN until 14 bars exist",
          bool(np.isnan(atr[:13]).all() and np.isfinite(atr[13])))
    # Wilder's recursion from the first defined value. Seeded from the
    # implementation's own first value so this checks the RECURSION, which is
    # what distinguishes Wilder from a span EMA, not the seeding convention.
    rec = [atr[13]]
    for i in range(14, len(bars)):
        rec.append((rec[-1] * 13 + tr[i]) / 14)
    check("ATR follows Wilder's recursion exactly",
          bool(np.allclose(atr[13:], rec)), "not a span EMA")

    span = bars["close"].ewm(span=14, adjust=False).mean().to_numpy()
    check("ATR is not a span-14 EMA of anything",
          not bool(np.allclose(atr[20:], span[20:])))

    rsi_up = S._rsi(pd.Series(np.arange(1, 41, dtype=float)), 14)
    check("RSI of a monotonic rise is exactly 100",
          float(rsi_up.iloc[-1]) == 100.0,
          "a zero average loss is RSI 100, not a division error")
    rsi = S._rsi(bars["close"], 14).dropna()
    check("RSI stays inside [0, 100]",
          bool((rsi >= 0).all() and (rsi <= 100).all()))
    check("RSI warm-up is NaN until 14 bars exist",
          bool(S._rsi(bars["close"], 14).iloc[:13].isna().all()))

    ind = S.indicators(bars, ema_period=20, atr_mult=2.5)
    check("indicators() returns full-length series",
          all(len(v) == len(bars) for v in ind.values()),
          "a series that is not the frame's length is DROPPED by the report")
    check("indicators() returns only price-scale series",
          not any("RSI" in k for k in ind),
          "a 0-100 oscillator on the price axis is a flat line at the bottom")
    band = ind[f"Lower Band (−2.5×ATR{S.ATR_PERIOD})"]
    ema = ind["EMA (20)"]
    # Compared where BOTH are defined. The band inherits the later of the two
    # warm-ups (EMA 20 here, not ATR 14), so dropping NaN from each separately
    # would line up two different stretches of the series.
    gap = (ema - band)
    both = gap.notna()
    check("the lower band sits below the EMA by exactly atr_mult x ATR",
          bool(np.allclose(gap[both], (2.5 * S._atr(bars, 14))[both])))
    check("the band inherits the later of the two warm-ups",
          int(both.idxmax()) == 19, f"first defined at {int(both.idxmax())}")


def test_session_masks() -> None:
    print("\nsession windows are New York wall-clock, both sides of DST")

    for label, day, in (("EDT (summer)", "2022-06-01"),
                        ("EST (winter)", "2022-01-05")):
        ts = pd.Series(pd.date_range(f"{day} 00:00", periods=96, freq="15min",
                                     tz="UTC"))
        window, flat = S._session_masks(ts)
        et = pd.DatetimeIndex(ts).tz_convert("America/New_York")
        got = (f"{et[window][0]:%H:%M}", f"{et[window][-1]:%H:%M}")
        check(f"{label}: entry window is 09:30-15:30 ET", got == ("09:30", "15:30"),
              f"{got[0]}..{got[1]}")
        flats = [f"{t:%H:%M}" for t in et[flat]]
        check(f"{label}: the flatten bar is the last one starting before 16:00",
              flats == ["15:45"], str(flats))

    # Timeframe-agnostic: found by comparing each bar to the next, not by
    # matching a wall-clock string, so a 5m frame flattens at 15:55.
    ts5 = pd.Series(pd.date_range("2022-06-01 00:00", periods=288, freq="5min",
                                  tz="UTC"))
    _, flat5 = S._session_masks(ts5)
    et5 = pd.DatetimeIndex(ts5).tz_convert("America/New_York")
    check("the flatten bar tracks the timeframe (15:55 on 5m bars)",
          [f"{t:%H:%M}" for t in et5[flat5]] == ["15:55"])


def test_walk_against_hand_computed_answers() -> None:
    print("\nthe position walk, against answers computed by hand")

    # Entry signal on bar 1 -> position live from bar 2. ATR 2.0 everywhere, so
    # the stop trails 1.75 * 2.0 = 3.5 below the high since the FILL bar.
    ok = np.array([False, True, False, False, False, False, False, False])
    high = np.array([100., 100., 102., 106., 105., 104., 103., 102.])
    low = np.array([100., 100., 101., 104., 103., 101.9, 100., 99.])
    close = np.array([100., 100., 101.5, 105., 104., 102., 101., 100.])
    ema = np.full(8, 999.)              # target unreachable
    atr = np.full(8, 2.0)
    flat = np.zeros(8, dtype=bool)

    e, x = S._walk_loop(ok, high, low, close, ema, atr, flat, 1.75)
    check("the entry signal is emitted on its own bar",
          list(np.flatnonzero(e)) == [1])
    # high-water peaks at 106 on bar 3, so the stop is 102.5; bar 4's low of
    # 103 survives it and bar 5's low of 101.9 does not.
    check("the trailing stop fires on the first bar whose low breaches it",
          list(np.flatnonzero(x)) == [5], "hw 106 - 3.5 = 102.5, low 101.9")

    spike = high.copy()
    spike[1] = 200.0                    # a huge high on the SIGNAL bar
    _, x2 = S._walk_loop(ok, spike, low, close, ema, atr, flat, 1.75)
    check("the high-water mark starts at the FILL bar, not the signal bar",
          list(np.flatnonzero(x2)) == [5],
          "a spike before the fill must not move the stop")

    ema_t = np.array([999.] * 3 + [104.5] + [999.] * 4)
    _, x3 = S._walk_loop(ok, high, low, close, ema_t, atr, flat, 1.75)
    check("the target exits when the close returns to the EMA",
          list(np.flatnonzero(x3)) == [3], "close 105 >= ema 104.5")

    flat_b = np.zeros(8, dtype=bool)
    flat_b[2] = True
    _, x4 = S._walk_loop(ok, high, low, close, ema, atr, flat_b, 1.75)
    check("the session flatten exits before the stop or the target",
          list(np.flatnonzero(x4)) == [2])

    ok_first = np.zeros(8, dtype=bool)
    ok_first[0] = True
    flat_first = np.zeros(8, dtype=bool)
    flat_first[0] = True
    _, x5 = S._walk_loop(ok_first, high, low, close, ema, atr, flat_first, 1.75)
    check("no exit is ever emitted on the entry signal bar itself",
          not bool(x5[0]), "the position is not open until the next bar")

    # A wider stop cannot exit earlier than a tighter one.
    firsts = []
    for mult in (0.5, 1.75, 5.0):
        _, xm = S._walk_loop(ok, high, low, close, ema, atr, flat, mult)
        hit = np.flatnonzero(xm)
        firsts.append(int(hit[0]) if len(hit) else 10**9)
    check("a wider stop never exits sooner than a tighter one",
          firsts == sorted(firsts), str(firsts))

    if S._walk is not S._walk_loop:
        ec, xc = S._walk(ok, high, low, close, ema, atr, flat, 1.75)
        check("the compiled walk matches the interpreted one",
              bool(np.array_equal(ec, e) and np.array_equal(xc, x)),
              "a missing compiler must not change which trades are taken")
    else:
        print("  SKIP  numba is absent, so there is no compiled walk to compare")


def test_signal_contract() -> None:
    print("\nthe signal contract, and the invariants the engine relies on")

    bars = synthetic(600)
    e, x = S.signal_fn(bars)

    check("two boolean Series on the frame's index",
          e.dtype == bool and x.dtype == bool and len(e) == len(bars)
          and e.index.equals(bars.index) and x.index.equals(bars.index))
    check("the strategy actually traded on this frame", int(e.sum()) > 0,
          f"{int(e.sum())} entries")
    check("entries and exits are paired",
          int(e.sum()) - int(x.sum()) in (0, 1),
          "at most one position may still be open at the last bar")

    ce, cx = clean_signals(e, x)
    check("clean_signals is a no-op — the walk already alternates",
          bool((ce == e).all() and (cx == x).all()),
          "overlapping entries would be absorbed, hiding the bug")

    et = pd.DatetimeIndex(bars["ts"]).tz_convert("America/New_York")
    mins = (et.hour * 60 + et.minute)[e.to_numpy()]
    check("every entry signal is inside the 09:30-15:30 ET window",
          bool((mins >= 570).all() and (mins <= 930).all()) if len(mins) else True)

    for bad, why in (({"ema_period": 1}, "ema_period"),
                     ({"atr_mult": 0}, "atr_mult"),
                     ({"rsi_thresh": 0}, "rsi_thresh")):
        try:
            S.signal_fn(bars, **bad)
            check(f"an impossible {why} raises", False, "it was accepted")
        except ValueError:
            check(f"an impossible {why} raises", True)

    fn, info = load_strategy(REPO / "strategies" / "experimental"
                             / "intraday_vol_mr.py",
                             {"ema_period": 15, "atr_mult": 2.0,
                              "rsi_thresh": 25})
    le, lx = fn(bars)
    de, dx = S.signal_fn(bars, ema_period=15, atr_mult=2.0, rsi_thresh=25)
    check("the loader binds the parameters it is given",
          bool((le == de).all() and (lx == dx).all()))
    check("the loader reads the declared grid and logic",
          set(info["param_grid"]) == {"ema_period", "atr_mult", "rsi_thresh"}
          and "{" not in info["logic"]["entry"],
          "the {param} slots are filled with the bound values")
    check("the logic card names the bound parameters, not the defaults",
          "15" in info["logic"]["entry"] and "2.0" in info["logic"]["entry"],
          info["logic"]["entry"][:70] + "…")


def test_no_lookahead() -> None:
    print("\ncausality — the check a docstring cannot make")

    bars = synthetic(600)
    e, x = S.signal_fn(bars)
    ea, xa = e.to_numpy(), x.to_numpy()

    mismatched = 0
    for k in (150, 300, 450, 599):
        e2, x2 = S.signal_fn(bars.iloc[:k].copy())
        # The last bar of a truncated frame has no next bar for the walk's
        # fill, so everything strictly before it must match.
        if not (np.array_equal(ea[:k - 1], e2.to_numpy()[:k - 1])
                and np.array_equal(xa[:k - 1], x2.to_numpy()[:k - 1])):
            mismatched += 1
    check("truncating the frame never changes an earlier signal",
          mismatched == 0,
          "a strategy that reads ahead cannot pass this")

    # Perturb the future and nothing before it may move.
    tail = bars.copy()
    tail.loc[tail.index[400:], ["open", "high", "low", "close"]] *= 1.5
    e3, x3 = S.signal_fn(tail)
    check("rewriting future bars never changes a past signal",
          bool(np.array_equal(ea[:399], e3.to_numpy()[:399])
               and np.array_equal(xa[:399], x3.to_numpy()[:399])))


def test_real_bars() -> None:
    print("\non real lake bars")

    try:
        from mdlib.lake import iter_bars
        bars = next(f for s, f in iter_bars(["NQ"], "15m", "2022-01-01",
                                            "2022-04-01")).reset_index(drop=True)
    except Exception as e:                                      # noqa: BLE001
        print(f"  SKIP  needs the lake ({type(e).__name__}: {e})")
        return

    e, x = S.signal_fn(bars)
    check(f"signals computed on {len(bars):,} real bars", int(e.sum()) > 0,
          f"{int(e.sum())} entries")

    e2, _ = S.signal_fn(bars.iloc[:len(bars) // 2].copy())
    check("still causal on real bars",
          bool(np.array_equal(e.to_numpy()[:len(bars) // 2 - 1],
                              e2.to_numpy()[:len(bars) // 2 - 1])))

    et = pd.DatetimeIndex(bars["ts"]).tz_convert("America/New_York")
    ei = np.flatnonzero(e.to_numpy())
    xi = np.flatnonzero(x.to_numpy())
    pairs = min(len(ei), len(xi))
    overnight = int((et[xi[:pairs]].normalize().values
                     != et[ei[:pairs]].normalize().values).sum())
    check("no position is carried past its own Eastern date",
          overnight == 0, f"{overnight} of {pairs}")

    s = S._series(bars, 20, 2.5)
    warm = max(s["ema"].first_valid_index(), s["atr"].first_valid_index(),
               s["rsi"].first_valid_index())
    check("nothing fires before every indicator exists",
          int(ei[0]) >= warm, f"first entry {int(ei[0])}, warm-up ends {warm}")


if __name__ == "__main__":
    test_indicators()
    test_session_masks()
    test_walk_against_hand_computed_answers()
    test_signal_contract()
    test_no_lookahead()
    test_real_bars()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
