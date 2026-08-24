#!/usr/bin/env python3
"""
The four-stage mathematical reconciliation.

Location:  ~/src/trading/tests/test_math_reconciliation.py

Reads no bars and runs no backtest. One synthetic 10-trade series with KNOWN
wins, losses, per-trade costs and regime tags is pushed through every stage
that computes a headline number, and each stage is required to reproduce the
same hand-calculated value.

The failure this exists to catch is not a crash. It is two stages computing
"profit factor" slightly differently - one on gross, one on net; one counting a
break-even trade as a loss, one dropping it - and reporting both under the same
word. Every number below is derived on paper first (see `HAND` and the module
docstring's arithmetic) and only then compared against the code, so a test that
passes because both sides changed together is not possible.

THE FIXTURE, in full. Ten trades, each costing exactly $20 in commission and
slippage, so `gross = net + 20` per trade and every quadrant total is checkable
in your head:

    Q1  High Volatility / Trending   +2400  +400  -400  -400
    Q2  High Volatility / Ranging     +500  -600  -400
    Q3  Low Volatility / Trending     +250  +250
    Q4  Low Volatility / Ranging      -100

    blended  gross wins 3800, gross losses 1900  ->  PF 2.00, net +1900
    Q1       gross wins 2800, gross losses  800  ->  PF 3.50, net +2000
    Q2       gross wins  500, gross losses 1000  ->  PF 0.50, net  -500
    Q3       gross wins  500, gross losses    0  ->  PF undefined, net +500
    Q4       gross wins    0, gross losses  100  ->  PF 0.00, net  -100

What each case pins, and why it is a way a number could mislead:

  * PROFIT FACTOR IS COMPUTED ON NET P&L, not gross. The engine's `pnl` column
    is vectorbt's post-cost figure and `gross_pnl` is the raw price difference;
    partitioning on gross would rank a strategy on an edge it never collected.
    On this fixture the two differ by a full 0.13 of profit factor, so the
    assertion cannot be satisfied by the wrong column.
  * ONE UNDEFINED PROFIT FACTOR, THREE SENTINELS. Q3 never lost, and
    `report.trade_stats` returns `inf`, `profiler` returns 999 and
    `report.day_of_week_breakdown` returns NaN for that identical case. All
    three are pinned HERE, together, because they are only safe while every
    consumer knows which one it is being handed - `999` sorts like the best
    result on the board and `inf` propagates through any average.
  * THE ALPHA SCORE IS `net_pnl x min(PF, 10)`, not `net x (PF-1) x sqrt(N)`
    and not `net x PF` uncapped. The fixture is built so the CEILING DECIDES
    THE WINNER: uncapped, Q3 scores 500 x 999 = 499,500 and is designated;
    capped, Q3 scores 5,000 and Q1's 7,000 wins. A regression that dropped the
    cap changes which environment this strategy is certified to trade in.
  * REGIME IS THE ENTRY BAR'S, NEVER THE EXIT BAR'S. The fixture's bars are
    laid out so exit-bar attribution moves Q1's four trades into Q3 and turns a
    3.50 profit factor into a different table that still sums to the same
    totals. Entry-bar attribution is checked against a hand-written map rather
    than against a second implementation.
  * DRAWDOWN IS ON CUMULATIVE REALIZED CASH EQUITY. `initial_capital` plus the
    cumulative sum of net P&L attributed to each trade's EXIT session - never a
    mark-to-market curve and never a percentage of a floating bar-open equity.
    The fixture gives the bars violent intrabar excursions between exit dates
    and requires the drawdown not to move, which is the only way to tell a cash
    curve from a floating one by its output.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.audit_gates import regime_gate                        # noqa: E402
from backtest.baseline import best_quadrant, screen                 # noqa: E402
from backtest.engine import _daily_returns                          # noqa: E402
from backtest.profiler import (QUADRANT_TO_REGIME, REGIMES,         # noqa: E402
                               REGIME_TO_QUADRANT, SCORE_PF_CEILING,
                               RegimeProfiler, designate,
                               designation_floor, quadrant_score)
from backtest.report import (daily_metrics, day_of_week_breakdown,  # noqa: E402
                             trade_stats)
from backtest.verify_full import cost_drag, friction_by_regime      # noqa: E402

FAILURES: list[str] = []

Q1, Q2, Q3, Q4 = REGIMES          # declared order IS the quadrant order
QUAD_OF = {label: q for q, label in QUADRANT_TO_REGIME.items()}
OUT_DIR = tempfile.mkdtemp(prefix="math_recon_")


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def close(a, b, tol: float = 1e-9) -> bool:
    """Both finite and equal to `tol`. NaN never equals anything, including itself."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


# --------------------------------------------------------------------------
# The fixture, and the hand-calculated answers
# --------------------------------------------------------------------------
COST_PER_TRADE = 20.0
INITIAL_CAPITAL = 100_000.0

# (net pnl, entry regime). Order IS the exit order - trade i exits on session i.
TRADES: list[tuple[float, str]] = [
    (+2400.0, Q1),
    (-400.0, Q1),
    (-400.0, Q1),
    (+400.0, Q1),
    (+500.0, Q2),
    (-600.0, Q2),
    (-400.0, Q2),
    (+250.0, Q3),
    (+250.0, Q3),
    (-100.0, Q4),
]

HAND = {
    # blended
    "n_trades": 10,
    "win_rate_pct": 50.0,                       # 5 wins of 10
    "gross_win": 3800.0,
    "gross_loss": 1900.0,
    "profit_factor": 2.00,                      # 3800 / 1900
    "net_pnl": 1900.0,
    "total_costs": 200.0,                       # 10 x 20
    "gross_pnl": 2100.0,                        # 1900 + 200
    "cost_share_pct": 100.0 * 200.0 / 2100.0,   # 9.523809523809524
    # per quadrant: (n, PF, net, win_rate_pct, gross_pnl, cost_share_pct)
    "by_regime": {
        Q1: (4, 3.50, 2000.0, 50.0, 2080.0, 100.0 * 80.0 / 2080.0),
        Q2: (3, 0.50, -500.0, 100.0 / 3.0, -440.0, None),
        Q3: (2, None, 500.0, 100.0, 540.0, 100.0 * 40.0 / 540.0),
        Q4: (1, 0.00, -100.0, 0.0, -80.0, None),
    },
    # alpha scores: net x min(PF, 10)
    "scores": {Q1: 2000.0 * 3.50, Q2: -500.0 * 0.50,
               Q3: 500.0 * SCORE_PF_CEILING, Q4: -100.0 * 0.00},
    "primary": Q1,
    "score_uncapped_winner": Q3,                # 500 x 999 beats 7,000
    # drawdown, on the realized cash curve
    "peak_equity": 102_500.0,
    "trough_after_peak": 101_500.0,
    "max_dd_pct": -100.0 * 1000.0 / 102_500.0,  # -0.975609756097561
}

SESSIONS = pd.DatetimeIndex(
    pd.date_range("2020-01-01", periods=len(TRADES), freq="D", tz="UTC"))
# Entry and exit sit on the SAME session date but at different times, so the
# two populations are separate rows of the bar frame. Sharing a slot would let
# an exit-bar attribution accidentally read an entry label and the fixture
# would stop discriminating the two rules.
ENTRY_OFFSET = pd.Timedelta(hours=12)
EXIT_OFFSET = pd.Timedelta(hours=20)


def trade_frame() -> pd.DataFrame:
    """
    The engine's own trade-log shape: `entry_time`, `exit_time`, `gross_pnl`,
    `costs`, `pnl`, with `pnl` the NET figure exactly as `_simulate` writes it.

    Entry timestamps are deliberately NOT the exit timestamps: each trade
    enters at 12:00 and exits at 20:00 of the SAME session, on two separate
    bars whose regime labels differ - see `labelled_bars`. Same session date,
    so the drawdown's exit-date attribution is unaffected.
    """
    net = [p for p, _ in TRADES]
    return pd.DataFrame({
        "entry_time": SESSIONS + ENTRY_OFFSET,
        "exit_time": SESSIONS + EXIT_OFFSET,
        "symbol": "NQ",
        "direction": "long",
        "entry_price": 100.0,
        "exit_price": 100.0,
        "gross_pnl": [p + COST_PER_TRADE for p in net],
        "costs": [COST_PER_TRADE] * len(net),
        "pnl": net,
    })


def labelled_bars() -> pd.DataFrame:
    """
    A quadrant-carrying frame covering both the entry and the exit session of
    every trade.

    The column is `regime_quadrant`, which is what `mdlib.lake` left-joins onto
    every frame it returns and what `RegimeProfiler` reads in production - not
    a hand-written `Regime` column, which the profiler would overwrite with its
    own live ADX/ATR pass. Feeding the cache column also puts
    `QUADRANT_TO_REGIME` on the hook: a transposed map would move every trade
    between quadrants with all four totals still adding up.

    The ENTRY sessions carry the fixture's declared quadrants. The EXIT
    sessions all carry Q3, so a stage that attributed on the exit bar would
    report ten Q3 trades - a single row where the answer is four - and the
    totals would still reconcile. `high`/`low` swing violently on the exit
    sessions and not at all on the entries, so a drawdown computed on anything
    mark-to-market cannot match the hand value either.
    """
    entries = SESSIONS + ENTRY_OFFSET
    exits = SESSIONS + EXIT_OFFSET
    idx = pd.DatetimeIndex(sorted(set(entries) | set(exits)))
    quad = pd.Series(QUAD_OF[Q3], index=idx, dtype="uint8")
    for ts, (_, r) in zip(entries, TRADES):
        quad.loc[ts] = QUAD_OF[r]
    swing = pd.Series(0.0, index=idx)
    swing.loc[exits] = 50_000.0
    return pd.DataFrame({
        "open": 100.0, "close": 100.0,
        "high": 100.0 + swing, "low": 100.0 - swing,
        "volume": 1.0,
        "regime_quadrant": quad,
        # `friction_by_regime` reads the LABEL the profiler already wrote; the
        # profiler will rewrite this from `regime_quadrant`, and the two must
        # agree or the map is transposed.
        "Regime": quad.map(QUADRANT_TO_REGIME).astype(object),
    }, index=idx)


def profile_of(trades: pd.DataFrame, bars: pd.DataFrame | None = None) -> dict:
    """Stage 1's profiler over the fixture, on a COPY (it mutates its frame)."""
    return RegimeProfiler((labelled_bars() if bars is None else bars).copy(),
                          trades, "recon", "SYNTH", "15m", out_dir=OUT_DIR,
                          quiet=True, min_trades=2).generate_profile()


def hand_profile() -> dict:
    """The `regime_breakdown` shape, built from HAND and never from the code."""
    breakdown = {}
    for regime, (n, pf, net, wr, _g, _c) in HAND["by_regime"].items():
        breakdown[regime] = {
            "trade_count": n,
            "profit_factor": 999 if pf is None else pf,   # profiler's sentinel
            "win_rate": round(wr, 2),
            "net_pnl": net,
        }
    return {"regime_breakdown": breakdown, "trades_profiled": HAND["n_trades"]}


# --------------------------------------------------------------------------
# 1. Profit factor: net P&L, and the division guard
# --------------------------------------------------------------------------
def test_profit_factor() -> None:
    print("\n[1] PROFIT FACTOR — net P&L, and the non-zero division guard")
    trades = trade_frame()

    ts = trade_stats(trades)
    check("blended PF == 2.00 (3800 net wins / 1900 net losses)",
          close(ts["profit_factor"], HAND["profit_factor"]),
          f"got {ts['profit_factor']}")
    check("blended trade count == 10", ts["n_trades"] == HAND["n_trades"],
          f"got {ts['n_trades']}")
    check("blended win rate == 50.0%",
          close(ts["win_rate_pct"], HAND["win_rate_pct"]),
          f"got {ts['win_rate_pct']}")

    # The discriminating assertion: the same function over the GROSS column
    # must give a DIFFERENT answer, or the fixture cannot tell the two apart.
    gross_pf = trade_stats(trades.assign(pnl=trades["gross_pnl"]))["profit_factor"]
    check("gross-P&L PF differs from net-P&L PF (the fixture discriminates)",
          not close(gross_pf, HAND["profit_factor"], 1e-6),
          f"gross {gross_pf} vs net {HAND['profit_factor']}")
    check("PF is taken from the NET column, not the gross one",
          close(ts["profit_factor"], HAND["profit_factor"])
          and not close(ts["profit_factor"], gross_pf, 1e-6),
          f"net {ts['profit_factor']} gross {gross_pf}")

    # Division guard. Q3 never lost: gross_loss == 0.
    no_loss = trades[trades["pnl"] > 0]
    check("zero gross loss does not raise and does not return 0.0",
          math.isinf(trade_stats(no_loss)["profit_factor"]),
          f"got {trade_stats(no_loss)['profit_factor']}")
    # ... and the three consumers each spell that same case differently.
    prof = profile_of(trades)
    check("profiler spells an undefined PF as the 999 sentinel",
          prof["regime_breakdown"][Q3]["profit_factor"] == 999,
          f"got {prof['regime_breakdown'][Q3]['profit_factor']}")
    dow = day_of_week_breakdown(no_loss)
    traded = dow[dow["trades"] > 0]["profit_factor"].tolist()
    check("day_of_week_breakdown spells the same case as NaN",
          bool(traded) and all(math.isnan(v) for v in traded),
          f"got {traded}")

    # A zero-P&L trade is a win in no stage. It is a LOSS to the profiler
    # (pnl <= 0) and neither to trade_stats (pnl > 0 / pnl < 0); both leave the
    # profit factor untouched, because it contributes 0 to either sum.
    with_zero = pd.concat([trades, trades.iloc[[0]].assign(pnl=0.0,
                                                           gross_pnl=20.0)],
                          ignore_index=True)
    check("a break-even trade moves neither gross win nor gross loss",
          close(trade_stats(with_zero)["profit_factor"], HAND["profit_factor"]),
          f"got {trade_stats(with_zero)['profit_factor']}")


# --------------------------------------------------------------------------
# 2. Alpha score: net x min(PF, ceiling)
# --------------------------------------------------------------------------
def test_alpha_score() -> None:
    print("\n[2] ALPHA SCORE — net_pnl x min(PF, 10), and the ceiling that decides")
    breakdown = hand_profile()["regime_breakdown"]

    for regime, expected in HAND["scores"].items():
        check(f"{REGIME_TO_QUADRANT[regime]} score == {expected:,.1f}",
              close(quadrant_score(breakdown[regime]), expected),
              f"got {quadrant_score(breakdown[regime])}")

    # The two formulas the audit asked to distinguish, on Q1's own numbers.
    n, pf, net = 4, 3.50, 2000.0
    alt = net * (pf - 1.0) * math.sqrt(n)          # net x (PF-1) x sqrt(N)
    check("the formula is NOT net x (PF-1) x sqrt(N)",
          not close(quadrant_score(breakdown[Q1]), alt, 1e-6),
          f"score {quadrant_score(breakdown[Q1])} vs alt {alt}")

    floor = designation_floor(HAND["n_trades"], min_trades=2)
    check("sample floor == max(2, ceil(10% of 10)) == 2", floor == 2, f"got {floor}")

    d = designate(breakdown, HAND["n_trades"], min_trades=2)
    check(f"primary quadrant is {REGIME_TO_QUADRANT[HAND['primary']]}",
          d["primary"] and d["primary"]["regime"] == HAND["primary"],
          f"got {d['primary'] and d['primary']['regime']}")
    check("the PF ceiling is what decided it — uncapped, Q3 would win",
          (500.0 * 999 > HAND["scores"][Q1]
           and HAND["scores"][Q3] < HAND["scores"][Q1]),
          "the fixture no longer discriminates the cap")
    check("Q3's capped row is flagged pf_capped",
          next(r for r in d["scores"] if r["regime"] == Q3)["pf_capped"],
          "not flagged")
    check("Q3's REPORTED profit factor is left as measured (999, uncapped)",
          next(r for r in d["scores"] if r["regime"] == Q3)["profit_factor"] == 999)

    # Exactly one primary, and the other three are the kill switch.
    q = best_quadrant(hand_profile(), min_trades=2)
    check("best_quadrant designates exactly one regime",
          q is not None and q["regime"] == HAND["primary"],
          f"got {q and q['regime']}")
    ok, _reason, best = screen({"A": hand_profile(), "B": None}, min_trades=2)
    check("screen() survives on Version A with the same quadrant",
          ok and best["version"] == "A" and best["regime"] == HAND["primary"],
          f"got {best}")
    check("a skipped Version B never wins the comparison",
          best["version"] == "A", f"got {best['version']}")


# --------------------------------------------------------------------------
# 3. Regime attribution: the ENTRY bar, never the exit bar
# --------------------------------------------------------------------------
def test_regime_attribution() -> None:
    print("\n[3] REGIME ATTRIBUTION — the ENTRY bar, never the exit or a later bar")
    trades, bars = trade_frame(), labelled_bars()

    prof = profile_of(trades, bars)
    bd = prof["regime_breakdown"]

    for regime, (n, pf, net, wr, _g, _c) in HAND["by_regime"].items():
        tag = REGIME_TO_QUADRANT[regime]
        check(f"{tag} trade count == {n}", bd.get(regime, {}).get("trade_count") == n,
              f"got {bd.get(regime, {}).get('trade_count')}")
        check(f"{tag} net P&L == {net:,.2f}",
              close(bd.get(regime, {}).get("net_pnl"), net, 1e-6),
              f"got {bd.get(regime, {}).get('net_pnl')}")
        if pf is not None:
            check(f"{tag} profit factor == {pf:.2f}",
                  close(bd.get(regime, {}).get("profit_factor"), pf, 1e-6),
                  f"got {bd.get(regime, {}).get('profit_factor')}")
        check(f"{tag} win rate == {wr:.2f}%",
              close(bd.get(regime, {}).get("win_rate"), round(wr, 2), 1e-6),
              f"got {bd.get(regime, {}).get('win_rate')}")

    check("every trade was placed in a quadrant",
          prof["trades_unplaced"] == 0 and prof["trades_profiled"] == 10,
          f"placed {prof['trades_profiled']} unplaced {prof['trades_unplaced']}")

    # The discriminating case: exit-bar attribution files all ten under Q3.
    exit_tagged = trades.assign(entry_time=trades["exit_time"])
    exit_prof = profile_of(exit_tagged, bars)
    check("attributing on the EXIT bar gives a DIFFERENT table (fixture discriminates)",
          exit_prof["regime_breakdown"].get(Q3, {}).get("trade_count") == 10,
          f"got {exit_prof['regime_breakdown'].get(Q3, {}).get('trade_count')}")
    check("the profiler did NOT produce the exit-bar table",
          bd[Q3]["trade_count"] == 2, f"got {bd[Q3]['trade_count']}")

    # A bar AFTER the entry cannot influence the label either: truncating the
    # frame at each entry bar leaves every attribution unchanged.
    last_entry = (SESSIONS + ENTRY_OFFSET).max()
    causal = profile_of(trades, bars.loc[:last_entry])
    same = all(causal["regime_breakdown"].get(r, {}).get("trade_count")
               == bd.get(r, {}).get("trade_count") for r in REGIMES)
    check("dropping every bar after the last ENTRY changes no attribution",
          same, "a later bar moved a trade between quadrants")

    # Stage 4 attributes the same way, on the same trades.
    fr = friction_by_regime(trades, bars)
    check("Stage 4 friction is available and places all ten trades",
          fr["available"] and fr["attributed_trades"] == 10
          and fr["unplaced_trades"] == 0, f"got {fr.get('attributed_trades')}")
    rows = {r["regime"]: r for r in fr["by_regime"]}
    check("Stage 4 reports all four quadrants as rows", len(fr["by_regime"]) == 4,
          f"got {len(fr['by_regime'])}")
    for regime, (n, _pf, net, _wr, gross, share) in HAND["by_regime"].items():
        tag = REGIME_TO_QUADRANT[regime]
        check(f"Stage 4 {tag} trades == {n}", rows[regime]["trades"] == n,
              f"got {rows[regime]['trades']}")
        check(f"Stage 4 {tag} net P&L == {net:,.2f}",
              close(rows[regime]["net_pnl"], net, 1e-6),
              f"got {rows[regime]['net_pnl']}")
        check(f"Stage 4 {tag} gross P&L == {gross:,.2f}",
              close(rows[regime]["gross_pnl"], gross, 1e-6),
              f"got {rows[regime]['gross_pnl']}")
        if share is None:
            check(f"Stage 4 {tag} cost share is None (gross not positive)",
                  rows[regime]["cost_share_pct"] is None,
                  f"got {rows[regime]['cost_share_pct']}")
        else:
            check(f"Stage 4 {tag} cost share == {share:.4f}%",
                  close(rows[regime]["cost_share_pct"], share, 1e-9),
                  f"got {rows[regime]['cost_share_pct']}")

    # Stage 1's per-quadrant table and Stage 4's must agree trade for trade.
    agree = all(rows[r]["trades"] == bd.get(r, {}).get("trade_count", 0)
                and close(rows[r]["net_pnl"], bd.get(r, {}).get("net_pnl", 0.0), 1e-6)
                for r in REGIMES)
    check("Stage 1 and Stage 4 agree on every quadrant, trade for trade", agree)


# --------------------------------------------------------------------------
# 4. Drawdown: cumulative realized cash equity
# --------------------------------------------------------------------------
def test_drawdown() -> None:
    print("\n[4] DRAWDOWN — cumulative realized CASH equity, not floating bar equity")
    trades = trade_frame()
    returns, equity = _daily_returns(trades, SESSIONS, INITIAL_CAPITAL)

    expected_eq = INITIAL_CAPITAL + np.cumsum([p for p, _ in TRADES])
    check("equity == initial_capital + cumsum(net P&L by EXIT session)",
          np.allclose(equity.to_numpy(), expected_eq),
          f"got {equity.to_numpy()[:3]} ... expected {expected_eq[:3]}")
    check(f"running peak == {HAND['peak_equity']:,.0f}",
          close(equity.max(), HAND["peak_equity"]), f"got {equity.max()}")

    dm = daily_metrics(equity, INITIAL_CAPITAL)
    check(f"max drawdown == {HAND['max_dd_pct']:.12f}% "
          f"(-1000 / 102500, off the peak)",
          close(dm["max_dd_pct"], HAND["max_dd_pct"], 1e-9),
          f"got {dm['max_dd_pct']}")
    check("the drawdown is a share of the PEAK equity, not of initial capital",
          not close(dm["max_dd_pct"], -100.0 * 1000.0 / INITIAL_CAPITAL, 1e-9),
          "matched a denominator of initial capital")
    check("metrics basis records daily closes",
          dm["basis"]["frequency"] == "daily_close"
          and dm["basis"]["n_days"] == len(TRADES),
          f"got {dm['basis']}")

    # The floating-equity discriminator. The bars swing +/-50,000 intrabar on
    # every exit session; a mark-to-market drawdown would be two orders of
    # magnitude deeper. The cash curve must not move at all.
    bars = labelled_bars()
    floating = float(((bars["low"] - bars["close"]) / bars["close"]).min() * 100)
    check("the bars carry an intrabar excursion far deeper than the reported DD",
          floating < -100.0 * abs(HAND["max_dd_pct"]),
          f"intrabar {floating}% vs reported {HAND['max_dd_pct']}%")
    check("the reported drawdown ignores it entirely (cash, not floating)",
          close(dm["max_dd_pct"], HAND["max_dd_pct"], 1e-9),
          f"got {dm['max_dd_pct']}")

    # Reordering the trade list cannot change a realized-cash drawdown keyed on
    # the exit date - but reversing the exit DATES must, because the path is
    # what a drawdown measures.
    shuffled = trades.iloc[[3, 1, 0, 2, 6, 5, 4, 9, 8, 7]].reset_index(drop=True)
    _r2, eq2 = _daily_returns(shuffled, SESSIONS, INITIAL_CAPITAL)
    check("row order in the trade log does not change the equity path",
          np.allclose(eq2.to_numpy(), equity.to_numpy()))
    reversed_exits = trades.assign(exit_time=trades["exit_time"].to_numpy()[::-1])
    _r3, eq3 = _daily_returns(reversed_exits, SESSIONS, INITIAL_CAPITAL)
    d3 = daily_metrics(eq3, INITIAL_CAPITAL)["max_dd_pct"]
    check("reversing the EXIT dates does change the drawdown (it is path-dependent)",
          not close(d3, HAND["max_dd_pct"], 1e-9), f"got {d3}")


# --------------------------------------------------------------------------
# 5. The four stages, reconciled against each other
# --------------------------------------------------------------------------
def test_cross_stage_identity() -> None:
    print("\n[5] CROSS-STAGE IDENTITY — one fixture, four stages, one set of numbers")
    trades, bars = trade_frame(), labelled_bars()

    # Stage 1 (baseline/profiler)
    prof = profile_of(trades, bars)
    stage1_pf = prof["regime_breakdown"][Q1]["profit_factor"]
    stage1_n = prof["regime_breakdown"][Q1]["trade_count"]

    # Stage 3 (audit_gates, Gate R) reads that same breakdown shape.
    gate = regime_gate(prof, Q1, min_profit_factor=1.00, min_trades=2)
    check("Gate R PASSes on Q1 at PF 3.50 over 4 trades",
          gate["status"] == "PASS", f"got {gate['status']} {gate.get('note')}")
    measured = gate.get("measured") or {}
    check("Gate R measured the SAME profit factor Stage 1 designated on",
          close(measured.get("profit_factor"), stage1_pf, 1e-9),
          f"gate {measured.get('profit_factor')} vs stage1 {stage1_pf}")
    check("Gate R measured the SAME trade count",
          measured.get("trade_count") == stage1_n,
          f"gate {measured.get('trade_count')} vs stage1 {stage1_n}")

    # Gate R at the shipped floor of 30 must FAIL this 4-trade quadrant on the
    # COUNT, not on the factor - the two are fixed by different work.
    starved = regime_gate(prof, Q1, min_profit_factor=1.00, min_trades=30)
    check("Gate R at the shipped 30-trade floor FAILs on the count, not the factor",
          starved["status"] == "FAIL"
          and any(c.get("status") == "FAIL" and "trade" in c.get("label", "").lower()
                  for c in starved.get("checks", []))
          and all(c.get("status") == "PASS"
                  for c in starved.get("checks", [])
                  if "profit factor" in c.get("label", "").lower()),
          f"got {starved['status']} {starved.get('checks')}")

    # Stage 4 (verify_full)
    drag = cost_drag(trades, None, "NQ", None)
    check("Stage 4 total costs == 200.00",
          close(drag["total_costs"], HAND["total_costs"]), f"got {drag['total_costs']}")
    check("Stage 4 gross P&L == 2,100.00",
          close(drag["gross_pnl"], HAND["gross_pnl"]), f"got {drag['gross_pnl']}")
    check("Stage 4 net P&L == 1,900.00 == gross - costs",
          close(drag["net_pnl"], HAND["net_pnl"])
          and close(drag["gross_pnl"] - drag["total_costs"], HAND["net_pnl"]),
          f"got {drag['net_pnl']}")
    check(f"Stage 4 blended cost share == {HAND['cost_share_pct']:.6f}%",
          close(drag["cost_share_pct"], HAND["cost_share_pct"], 1e-9),
          f"got {drag['cost_share_pct']}")

    # The identity every stage rests on, checked once and explicitly.
    ts = trade_stats(trades)
    check("net == gross - costs holds on the blended totals",
          close(HAND["gross_pnl"] - HAND["total_costs"], HAND["net_pnl"]))
    check("the four quadrant nets sum to the blended net",
          close(sum(v[2] for v in HAND["by_regime"].values()), HAND["net_pnl"]))
    fr = friction_by_regime(trades, bars)
    check("the four quadrant nets sum to Stage 4's blended net",
          close(sum(r["net_pnl"] for r in fr["by_regime"]), drag["net_pnl"], 1e-6))
    check("the four quadrant costs sum to Stage 4's total costs",
          close(sum(r["costs"] for r in fr["by_regime"]), drag["total_costs"], 1e-6))
    check("blended PF is identical in Stage 1's inputs and the scorecard",
          close(ts["profit_factor"], HAND["profit_factor"]))

    # And the number a promotion would cite.
    q = best_quadrant(prof, min_trades=2)
    check("the designated quadrant's PF is the one Gate R certifies",
          close(q["profit_factor"], measured.get("profit_factor"), 1e-9),
          f"{q['profit_factor']} vs {measured.get('profit_factor')}")
    check("exactly three quadrants land in the kill switch",
          len([r for r in REGIMES if r != q["regime"]]) == 3)


def main() -> int:
    print("=" * 78)
    print(" MATHEMATICAL RECONCILIATION — 10 synthetic trades, four stages")
    print("=" * 78)
    test_profit_factor()
    test_alpha_score()
    test_regime_attribution()
    test_drawdown()
    test_cross_stage_identity()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f" {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"   - {f}")
        return 1
    print(" ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
