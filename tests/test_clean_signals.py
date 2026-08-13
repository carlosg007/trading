#!/usr/bin/env python3
"""
test_clean_signals.py - is the compiled clean_signals the same function?

Location:  ~/src/trading/tests/test_clean_signals.py

Run:  python tests/test_clean_signals.py

There is no pytest config in this repo, so this is a plain script that exits
non-zero on failure.

What is being proved
--------------------
clean_signals decides which entry and exit signals actually execute. Every
trade in every backtest passes through it, so a discrepancy here does not
announce itself - it silently adds or removes trades and the equity curve
still looks plausible. That is worth proving rather than spot-checking.

The claim is exact equality with the pre-refactor Python loop on every input,
and it is attacked three ways:

1. EXHAUSTIVELY for short inputs. Every possible (entries, exits) pair up to
   length 10 is enumerated - all 2^n * 2^n of them, 1,398,100 pairs in total.
   Nothing is sampled and nothing is assumed.

   Exhaustion at short lengths is stronger than it looks. clean_signals is a
   two-state machine (flat / long) whose transition reads only the current
   bar, so it has no hidden state that needs a long input to reach. Every
   reachable (state, e[i], x[i]) combination and every transition between
   them already occurs many times over within 10 bars. A bug that needs 11
   bars to appear would have to depend on something the algorithm does not
   have.

2. RANDOMISED at full scale. 5.6M rows - the largest single symbol in the
   lake - across densities from 1-in-100,000 to 999-in-1000, so both the
   sparse regime real strategies live in and the degenerate near-every-bar
   regime are covered.

3. ADVERSARIALLY. All-true, all-false, entry and exit set on the same bar,
   perfectly alternating, long runs, empty input, single bar. These are the
   cases where an off-by-one in a rewritten loop actually shows up.

The oracle is transcribed independently in this file, in plain Python with no
numpy, from the implementation as it stood before the refactor. It is
deliberately NOT imported from backtest.engine: an oracle that shares code
with the thing it checks proves nothing.

It also checks what a signals rewrite is most likely to break silently: the
returned index, the dtype, and NaN handling.
"""

from __future__ import annotations

import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from backtest.engine import _clean_signals_loop, clean_signals

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def oracle(e: list[bool], x: list[bool]) -> tuple[list[bool], list[bool]]:
    """
    The pre-refactor implementation, transcribed by hand.

    Plain Python lists, no numpy, no shared code with backtest.engine. This is
    the definition of correct for this test.
    """
    n = len(e)
    ke = [False] * n
    kx = [False] * n
    in_pos = False
    for i in range(n):
        if not in_pos and e[i]:
            ke[i] = True
            in_pos = True
        elif in_pos and x[i]:
            kx[i] = True
            in_pos = False
    return ke, kx


def compiled(e: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The public entry point, through pandas, as the engine calls it."""
    ke, kx = clean_signals(pd.Series(e), pd.Series(x))
    return ke.to_numpy(), kx.to_numpy()


# --------------------------------------------------------------------------
def test_exhaustive_short_inputs() -> None:
    """Every possible input up to length 10. No sampling."""
    print("\n[1] exhaustive: every (entries, exits) pair up to length 10")
    total = 0
    t0 = time.perf_counter()

    for n in range(1, 11):
        bad = None
        for e_bits in itertools.product((False, True), repeat=n):
            e_arr = np.array(e_bits, dtype=bool)
            for x_bits in itertools.product((False, True), repeat=n):
                x_arr = np.array(x_bits, dtype=bool)
                ke, kx = compiled(e_arr, x_arr)
                oke, okx = oracle(list(e_bits), list(x_bits))
                total += 1
                if ke.tolist() != oke or kx.tolist() != okx:
                    bad = (e_bits, x_bits, ke.tolist(), kx.tolist(), oke, okx)
                    break
            if bad:
                break
        check(f"n={n}: all {2**n * 2**n:,} pairs match", bad is None,
              "" if bad is None else f"counterexample {bad}")

    print(f"    {total:,} input pairs verified in {time.perf_counter()-t0:.1f}s")


def test_random_full_scale() -> None:
    """5.6M rows - the largest single symbol - across every density regime."""
    print("\n[2] randomised at full scale (5.6M rows)")
    n = 5_600_000
    rng = np.random.default_rng(20260813)

    for d in (0.00001, 0.0005, 0.004, 0.05, 0.3, 0.75, 0.999):
        e = rng.random(n) < d
        x = rng.random(n) < d

        t0 = time.perf_counter()
        ke, kx = compiled(e, x)
        fast_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        oke, okx = _clean_signals_loop(e, x)
        loop_s = time.perf_counter() - t0

        ok = np.array_equal(ke, oke) and np.array_equal(kx, okx)
        check(f"density {d:<8}: identical to the loop", ok,
              f"{int(ke.sum()):,} entries kept, "
              f"{fast_s*1000:.0f}ms vs {loop_s*1000:.0f}ms "
              f"({loop_s/fast_s:.0f}x)")


def test_adversarial_patterns() -> None:
    """The shapes an off-by-one actually shows up on."""
    print("\n[3] adversarial patterns")
    n = 50_000
    ar = np.arange(n)

    cases = {
        "all entries, no exits":      (np.ones(n, bool), np.zeros(n, bool)),
        "no entries, all exits":      (np.zeros(n, bool), np.ones(n, bool)),
        "all entries and all exits":  (np.ones(n, bool), np.ones(n, bool)),
        "entry and exit same bar":    (ar % 7 == 0, ar % 7 == 0),
        "perfectly alternating":      (ar % 2 == 0, ar % 2 == 1),
        "exit before any entry":      (ar > n // 2, ar < n // 2),
        "single entry at last bar":   (ar == n - 1, np.zeros(n, bool)),
        "single exit at first bar":   (np.zeros(n, bool), ar == 0),
        "long runs":                  ((ar // 1000) % 2 == 0, (ar // 1500) % 2 == 1),
        "nothing at all":             (np.zeros(n, bool), np.zeros(n, bool)),
    }

    for label, (e, x) in cases.items():
        ke, kx = compiled(e, x)
        oke, okx = _clean_signals_loop(e, x)
        ok = np.array_equal(ke, oke) and np.array_equal(kx, okx)
        check(f"{label}", ok, f"{int(ke.sum()):,} entries / {int(kx.sum()):,} exits")

    # Degenerate lengths, which numba is entitled to dislike.
    for n_small in (0, 1, 2):
        e = np.ones(n_small, bool)
        x = np.ones(n_small, bool)
        try:
            ke, kx = compiled(e, x)
            oke, okx = oracle(list(e), list(x))
            ok = ke.tolist() == oke and kx.tolist() == okx
        except Exception as exc:                     # noqa: BLE001
            ok, oke = False, str(exc)
        check(f"length {n_small} input", ok, str(oke))


def test_invariants() -> None:
    """
    Properties that must hold whatever the input, checked independently of
    the oracle - a second opinion on what "clean" means.
    """
    print("\n[4] invariants")
    n = 2_000_000
    rng = np.random.default_rng(7)
    e = rng.random(n) < 0.01
    x = rng.random(n) < 0.01
    ke, kx = compiled(e, x)

    check("kept entries are a subset of the input entries",
          bool((~e[ke]).sum() == 0))
    check("kept exits are a subset of the input exits",
          bool((~x[kx]).sum() == 0))
    check("no bar is both a kept entry and a kept exit",
          bool(~(ke & kx).any()))

    # Entries and exits must strictly alternate, starting with an entry: the
    # whole point of the function.
    pos = np.concatenate([np.flatnonzero(ke), np.flatnonzero(kx)])
    kind = np.concatenate([np.ones(int(ke.sum()), np.int8),
                           -np.ones(int(kx.sum()), np.int8)])
    order = np.argsort(pos, kind="stable")
    seq = kind[order]
    check("signals strictly alternate entry/exit",
          bool((seq[::2] == 1).all() and (seq[1::2] == -1).all()),
          f"{len(seq):,} signals")
    check("entries == exits, or exactly one more (a position left open)",
          int(ke.sum()) - int(kx.sum()) in (0, 1),
          f"{int(ke.sum()):,} entries, {int(kx.sum()):,} exits")


def test_pandas_contract() -> None:
    """Index, dtype and NaN handling - the quiet ways a rewrite breaks."""
    print("\n[5] pandas contract")
    n = 1000
    rng = np.random.default_rng(3)
    e = pd.Series(rng.random(n) < 0.05)
    x = pd.Series(rng.random(n) < 0.05)

    # A non-default index, as a per-symbol slice would carry.
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    ke, kx = clean_signals(e.set_axis(idx), x.set_axis(idx))
    check("index is preserved", ke.index.equals(idx) and kx.index.equals(idx))
    check("dtype is bool", ke.dtype == bool and kx.dtype == bool,
          f"{ke.dtype}, {kx.dtype}")

    # NaN must be treated as False, as the original fillna(False) did.
    ef = pd.Series(np.where(rng.random(n) < 0.05, 1.0, np.nan))
    xf = pd.Series(np.where(rng.random(n) < 0.05, 1.0, np.nan))
    ke2, kx2 = clean_signals(ef, xf)
    oke, okx = _clean_signals_loop(ef.fillna(False).astype(bool).to_numpy(),
                                   xf.fillna(False).astype(bool).to_numpy())
    check("NaN is treated as False",
          np.array_equal(ke2.to_numpy(), oke) and np.array_equal(kx2.to_numpy(), okx),
          f"{int(ke2.sum())} entries kept")

    # Object dtype with real None, which get_bars-derived signals can produce.
    eo = pd.Series([True, None, False, True, None, True], dtype=object)
    xo = pd.Series([False, True, None, False, True, None], dtype=object)
    ke3, kx3 = clean_signals(eo, xo)
    oke3, okx3 = _clean_signals_loop(eo.fillna(False).astype(bool).to_numpy(),
                                     xo.fillna(False).astype(bool).to_numpy())
    check("object dtype with None works",
          np.array_equal(ke3.to_numpy(), oke3) and np.array_equal(kx3.to_numpy(), okx3))


def test_real_lake_signals() -> None:
    """Real bars, real crossover signals, across contract types."""
    print("\n[6] real lake data, SMA crossover signals")
    from mdlib.lake import get_bars

    for sym in ["ES", "GC", "ZN", "6E"]:
        bars = get_bars([sym], "1m", "2022-01-01", "2024-01-01").reset_index(drop=True)
        if bars.empty:
            check(f"{sym}: bars loaded", False, "empty")
            continue
        close = bars["close"]
        fast, slow = close.rolling(200).mean(), close.rolling(800).mean()
        e = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).fillna(False)
        x = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).fillna(False)

        ke, kx = clean_signals(e, x)
        oke, okx = _clean_signals_loop(e.to_numpy(), x.to_numpy())
        ok = (np.array_equal(ke.to_numpy(), oke)
              and np.array_equal(kx.to_numpy(), okx))
        check(f"{sym}: identical on {len(bars):,} real bars", ok,
              f"{int(ke.sum()):,} entries kept")


if __name__ == "__main__":
    test_exhaustive_short_inputs()
    test_random_full_scale()
    test_adversarial_patterns()
    test_invariants()
    test_pandas_contract()
    test_real_lake_signals()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
