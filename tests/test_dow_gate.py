#!/usr/bin/env python3
"""
Stage 4.5 - the day-of-week gate, and what each of its numbers can hide.

Location:  ~/src/trading/tests/test_dow_gate.py

SCRIPT-STYLE (`check(name, ok)` + `main()`), so `tests/conftest.py` routes it
to `tests/test_suite_runners.py` and it runs as its own subprocess asserting an
exit code. Read that file before converting this one: under bare pytest a
collector-style suite reports GREEN while its checks fail.

    .venv/bin/python3 tests/test_dow_gate.py

READS NO BARS AND RUNS NO BACKTEST. Every case is a small synthetic fixture
whose answer is worked out by hand above the code, which is the only way to
tell a weekday table that is right from one that merely produces plausible
dollars. The counterfactual is the one thing here that needs an engine, and it
is deliberately not exercised: it is one `run_dual_version_backtest` call whose
only difference from the baseline is `cfg.exclude_days`, and
`tests/test_batch_runner.py` already pins that mask against an oracle trade for
trade.

What is pinned, and why each one is a way the gate could mislead:

  * Attribution is by the ENTRY, on the CME SESSION date. A trade entered at
    19:00 ET on Sunday belongs to MONDAY's session; keyed on the calendar date
    it lands on a Sunday row and the weekday the filter would actually cut is
    a different one.
  * `expectancy` is the mean NET P&L per trade and is exactly `avg_pnl`. Two
    names computed twice are two numbers free to drift.
  * The drawdown contribution is a share of the run's DEEPEST episode, not a
    per-weekday drawdown - a weekday has no equity curve of its own.
  * A weekday below the trade floor is NEVER ranked, however badly it scored.
    That floor is the whole difference between a filter and a curve fit.
  * The worst weekday is always IDENTIFIED and is BLOCKED only when its
    expectancy is negative. Five profitable weekdays have a worst one too, and
    blocking it removes realised edge for nothing.
  * The promotion gate promotes EVERYTHING. Stage 4.5 produces an instruction
    for the live supervisor, never a verdict about a strategy.
  * `NOT EVALUATED` and "blocked nothing" stay distinct end to end - through
    the handoff reader, through `promote.load_dow_gate` and into meta.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.dow_gate import (DOW_MIN_TRADES,                     # noqa: E402
                               daily_return_profile,
                               drawdown_contributions,
                               format_weekday_profile,
                               promotion_gate,
                               select_worst_weekday,
                               weekday_profile,
                               worst_drawdown_episode)
from backtest.pipeline import (DOW_GATE_FILE, STAGE45,             # noqa: E402
                               stage45_blocked_days, write_stage)
from backtest.promote import load_dow_gate                         # noqa: E402

FAILURES: list[str] = []

MON, TUE, WED, THU, FRI = 0, 1, 2, 3, 4


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def _trades(rows) -> pd.DataFrame:
    """`[(entry, exit, pnl)]` -> the columns every function here reads."""
    return pd.DataFrame({
        "entry_time": [pd.Timestamp(r[0], tz="UTC") for r in rows],
        "exit_time": [pd.Timestamp(r[1], tz="UTC") for r in rows],
        "pnl": [float(r[2]) for r in rows],
        "direction": ["long"] * len(rows),
        "symbol": ["NQ"] * len(rows),
    })


# --------------------------------------------------------------------------
# 1. Grouping
# --------------------------------------------------------------------------
def test_attribution_is_the_entry_session() -> None:
    """
    2026-08-23 is a SUNDAY. 23:00 UTC is 19:00 EDT, past the 18:00 roll, so
    that bar belongs to MONDAY's session - and a trade entered there is a
    Monday trade however Sunday its calendar date looks.

    2026-08-27 is a THURSDAY. 22:00 UTC is 18:00 EDT, so that entry belongs to
    FRIDAY's session; 21:00 UTC is 17:00 EDT, inside Thursday's, and stays.

    The EXIT dates are deliberately all over the week: a table keyed on the
    exit would put every one of these rows somewhere else, and would point at
    a weekday whose removal does not remove the trades that made it worst.
    """
    print("\nattribution: entry, on the CME session date")
    t = _trades([
        ("2026-08-23 23:00", "2026-08-31 15:00", 10.0),   # Sun 19:00 ET -> MON
        ("2026-08-27 21:00", "2026-09-01 15:00", 20.0),   # Thu 17:00 ET -> THU
        ("2026-08-27 22:00", "2026-09-02 15:00", 30.0),   # Thu 18:00 ET -> FRI
    ])
    p = weekday_profile(t).set_index("weekday")
    check("a Sunday 19:00 ET entry is a MONDAY trade",
          int(p.loc[MON, "trades"]) == 1 and float(p.loc[MON, "net_pnl"]) == 10.0,
          f"{p.loc[MON, 'trades']} trades, {p.loc[MON, 'net_pnl']}")
    check("a Thursday 17:00 ET entry stays on THURSDAY",
          int(p.loc[THU, "trades"]) == 1 and float(p.loc[THU, "net_pnl"]) == 20.0)
    check("a Thursday 18:00 ET entry is a FRIDAY trade",
          int(p.loc[FRI, "trades"]) == 1 and float(p.loc[FRI, "net_pnl"]) == 30.0)
    check("nothing landed on a weekend row",
          int(p.loc[MON:FRI, "trades"].sum()) == 3)


def test_the_metrics_are_the_ones_written_on_the_row() -> None:
    """
    Monday, by hand: +100, +50, -30, -20.

        net P&L    = 100
        win rate   = 2/4 = 0.50
        profit factor = 150 / 50 = 3.00
        expectancy = 100 / 4 = 25.00     (and IS `avg_win x wr - ...`)
        avg win    = 75.0     avg loss = -25.0
    """
    print("\nthe per-weekday metrics")
    t = _trades([("2026-08-24 14:00", "2026-08-24 15:00", 100.0),
                 ("2026-08-24 15:00", "2026-08-24 16:00", 50.0),
                 ("2026-08-24 16:00", "2026-08-24 17:00", -30.0),
                 ("2026-08-24 17:00", "2026-08-24 18:00", -20.0)])
    r = weekday_profile(t).set_index("weekday").loc[MON]
    check("net P&L", float(r["net_pnl"]) == 100.0, str(r["net_pnl"]))
    check("win rate", float(r["win_rate"]) == 0.50, str(r["win_rate"]))
    check("profit factor", float(r["profit_factor"]) == 3.0,
          str(r["profit_factor"]))
    check("expectancy is the mean NET P&L per trade",
          float(r["expectancy"]) == 25.0, str(r["expectancy"]))
    check("expectancy == win_rate x avg_win + loss_rate x avg_loss",
          abs(float(r["expectancy"])
              - (0.5 * float(r["avg_win"]) + 0.5 * float(r["avg_loss"]))) < 1e-9)
    check("avg win / avg loss", float(r["avg_win"]) == 75.0
          and float(r["avg_loss"]) == -25.0)


def test_a_weekday_that_never_traded_is_a_row_of_zeros_not_an_absence() -> None:
    """
    An absent row reads as missing data when it means "this strategy never
    entered on a Friday", which is a finding. Its EXPECTANCY is NaN and not
    0.0, because a zero expectancy is a break-even day and would sort above a
    losing one - the wrong answer to "which weekday is worst" when the truth
    is that there is no evidence at all.
    """
    print("\nan untraded weekday")
    t = _trades([("2026-08-24 14:00", "2026-08-24 15:00", 10.0)])
    p = weekday_profile(t).set_index("weekday")
    check("Mon-Fri are all rows", set(range(5)) <= set(p.index))
    check("an untraded weekday shows zero trades",
          int(p.loc[FRI, "trades"]) == 0)
    check("its expectancy is NaN, never 0.0",
          pd.isna(p.loc[FRI, "expectancy"]))


# --------------------------------------------------------------------------
# 2. The drawdown contribution
# --------------------------------------------------------------------------
def _dd_fixture() -> pd.DataFrame:
    """
    Five trades in exit order, cumulative equity worked out by hand:

        pnl        +100   -30    -40    -30    +50
        equity      100    70     30      0     50
        peak        100   100    100    100    100
        underwater    0   -30    -70   -100    -50

    The deepest episode runs from index 0 (the peak) to index 3 (the trough)
    and gives back 100. The three trades INSIDE it are entered on Monday,
    Tuesday and Monday, so:

        Monday   -30 + -30 = -60   ->  60% of the decline
        Tuesday        -40         ->  40%
    """
    return _trades([
        ("2026-08-26 14:00", "2026-08-26 15:00", 100.0),   # Wed, the peak
        ("2026-08-24 14:00", "2026-08-27 15:00", -30.0),   # Mon
        ("2026-08-25 14:00", "2026-08-28 15:00", -40.0),   # Tue
        ("2026-08-31 14:00", "2026-08-31 15:00", -30.0),   # Mon, the trough
        ("2026-09-03 14:00", "2026-09-03 15:00", 50.0),    # Thu, after it
    ])


def test_the_deepest_episode_is_located_on_the_realisation_curve() -> None:
    print("\nthe deepest drawdown episode")
    e = worst_drawdown_episode(_dd_fixture())
    check("it is available", bool(e["available"]))
    check("the decline is -100", float(e["decline"]) == -100.0,
          str(e["decline"]))
    check("peak at index 0, trough at index 3",
          e["peak_index"] == 0 and e["trough_index"] == 3,
          f"{e['peak_index']} -> {e['trough_index']}")


def test_drawdown_contribution_is_a_share_of_that_episode() -> None:
    print("\nthe per-weekday drawdown contribution")
    c = drawdown_contributions(_dd_fixture())
    check("Monday contributed -60", c[MON]["dollars"] == -60.0,
          str(c[MON]["dollars"]))
    check("Monday is 60% of the decline", c[MON]["pct"] == 60.0,
          str(c[MON]["pct"]))
    check("Tuesday is 40%", c[TUE]["pct"] == 40.0, str(c[TUE]["pct"]))
    check("the Wednesday PEAK trade is outside the episode",
          c[WED]["dollars"] == 0.0, str(c[WED]["dollars"]))
    check("so is the Thursday trade after the trough",
          c[THU]["dollars"] == 0.0, str(c[THU]["dollars"]))
    check("every weekday is a key, traded or not",
          set(c) == set(range(7)))
    check("the profile carries the same numbers",
          float(weekday_profile(_dd_fixture())
                .set_index("weekday").loc[MON, "max_dd_contribution_pct"]) == 60.0)


def test_a_curve_that_never_drew_down_says_so() -> None:
    """
    Zero contributions, and `available` True with a REASON - not `available`
    False, which is what a run with no trades reports. "Nothing to attribute"
    and "no drawdown ever happened" are different findings.
    """
    print("\na curve with no drawdown")
    t = _trades([("2026-08-24 14:00", "2026-08-24 15:00", 10.0),
                 ("2026-08-25 14:00", "2026-08-25 15:00", 20.0)])
    e = worst_drawdown_episode(t)
    check("available, with a reason", e["available"] and "never drew down" in e["reason"])
    check("decline is 0.0, not None", e["decline"] == 0.0)
    check("every contribution is zero",
          all(v["dollars"] == 0.0 for v in drawdown_contributions(t).values()))


# --------------------------------------------------------------------------
# 3. The daily realised returns
# --------------------------------------------------------------------------
def test_daily_returns_group_by_session_weekday() -> None:
    """
    The engine's `returns` series is one point per SESSION, decimal, keyed on
    the date the P&L was realised. Grouped through `session_weekday` - the
    same rule the trade table uses - rather than through `.dayofweek`, which
    would agree here and disagree the day somebody hands this an intraday
    index.

    Mon 2026-08-24 .. Fri 2026-08-28, returns +1%, -2%, +3%, 0%, -1%.
    """
    print("\nthe daily realised-return table")
    idx = pd.to_datetime(["2026-08-24", "2026-08-25", "2026-08-26",
                          "2026-08-27", "2026-08-28"], utc=True)
    r = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01], index=idx)
    d = daily_return_profile(r)
    check("Monday is one session at +1%",
          d[MON]["sessions"] == 1 and abs(d[MON]["mean_pct"] - 1.0) < 1e-9)
    check("Tuesday is -2%", abs(d[TUE]["mean_pct"] + 2.0) < 1e-9)
    check("a flat session is not a winning one",
          d[THU]["win_rate"] == 0.0, str(d[THU]["win_rate"]))
    check("an untraded weekday reports zero sessions",
          d[5]["sessions"] == 0 and d[6]["sessions"] == 0)


# --------------------------------------------------------------------------
# 4. The verdict
# --------------------------------------------------------------------------
def _ranked_fixture(pnl_by_day: dict[int, list[float]]) -> pd.DataFrame:
    """A profile built from a P&L list per weekday, entered at 14:00 UTC."""
    base = {MON: "2026-08-24", TUE: "2026-08-25", WED: "2026-08-26",
            THU: "2026-08-27", FRI: "2026-08-28"}
    rows = []
    for day, pnls in pnl_by_day.items():
        for i, v in enumerate(pnls):
            # Minutes apart so every trade has its own entry and exit stamp;
            # they stay inside the same session, which is what is being fixed.
            rows.append((f"{base[day]} 14:{i % 60:02d}",
                         f"{base[day]} 15:{i % 60:02d}", v))
    return weekday_profile(_trades(rows))


def test_the_worst_weekday_is_the_lowest_expectancy() -> None:
    print("\nselecting the worst weekday")
    p = _ranked_fixture({
        MON: [10.0] * 25,
        TUE: [-1.0] * 25,          # expectancy -1.00
        WED: [-5.0] * 25,          # expectancy -5.00  <- worst
        THU: [2.0] * 25,
        FRI: [1.0] * 25,
    })
    v = select_worst_weekday(p, min_trades=DOW_MIN_TRADES)
    check("Wednesday is the worst session", v["worst_weekday"] == WED,
          str(v["worst_weekday"]))
    check("and it is BLOCKED, because its expectancy is negative",
          v["blocked"] and v["blocked_weekday"] == WED)
    check("the block rule is on the record",
          v["block_rule"] == "negative_expectancy")
    check("the metrics that condemned it travel with it",
          v["metrics"]["expectancy"] == -5.0 and v["metrics"]["trades"] == 25)
    check("every eligible weekday is listed",
          sorted(e["weekday"] for e in v["eligible"]) == [0, 1, 2, 3, 4])


def test_a_thin_weekday_is_never_ranked_however_badly_it_scored() -> None:
    """
    THE FLOOR IS THE DIFFERENCE BETWEEN A FILTER AND A CURVE FIT. Friday here
    loses ten times what Wednesday does, over three trades. Ranking it would
    cut a session on a sample that cannot separate an edge from a run of luck,
    which is precisely how a day-of-week filter manufactures an in-sample
    Sharpe.
    """
    print("\nthe trade floor")
    p = _ranked_fixture({
        MON: [10.0] * 25,
        TUE: [3.0] * 25,
        WED: [-5.0] * 25,
        THU: [2.0] * 25,
        FRI: [-50.0] * 3,          # far worse, far too thin
    })
    v = select_worst_weekday(p, min_trades=DOW_MIN_TRADES)
    check("Wednesday is chosen, not Friday", v["worst_weekday"] == WED,
          str(v["worst_weekday"]))
    check("Friday is REPORTED below the floor rather than dropped in silence",
          [e["weekday"] for e in v["below_floor"]] == [FRI])
    check("its trade count is on that record",
          v["below_floor"][0]["trades"] == 3)


def test_no_weekday_clearing_the_floor_blocks_nothing_and_says_why() -> None:
    print("\nnothing reached the floor")
    p = _ranked_fixture({MON: [-5.0] * 3, TUE: [-9.0] * 2})
    v = select_worst_weekday(p, min_trades=DOW_MIN_TRADES)
    check("no weekday is identified", v["worst_weekday"] is None)
    check("nothing is blocked", v["blocked"] is False
          and v["blocked_weekday"] is None)
    check("the reason names the floor", "trade floor" in v["reason"],
          v["reason"])


def test_a_profitable_worst_weekday_is_named_but_not_blocked() -> None:
    """
    Five profitable weekdays have a worst one too. Blocking it removes
    realised edge in exchange for nothing, and "fifth of five" is not evidence
    against a session - so the identification and the block are two separate
    answers, and this is the case that would fail if they were collapsed.
    """
    print("\na worst weekday that still makes money")
    p = _ranked_fixture({MON: [10.0] * 25, TUE: [8.0] * 25, WED: [6.0] * 25,
                         THU: [4.0] * 25, FRI: [1.0] * 25})
    v = select_worst_weekday(p, min_trades=DOW_MIN_TRADES)
    check("Friday is identified as the weakest", v["worst_weekday"] == FRI)
    check("and it is NOT blocked", v["blocked"] is False
          and v["blocked_weekday"] is None)
    check("the reason says the expectancy is not negative",
          "not negative" in v["reason"], v["reason"])

    forced = select_worst_weekday(p, min_trades=DOW_MIN_TRADES,
                                  block_always=True)
    check("--block-worst-always blocks it anyway",
          forced["blocked"] and forced["blocked_weekday"] == FRI)
    check("and the override is recorded on the verdict",
          forced["block_rule"] == "worst_always")


def test_a_tie_on_expectancy_breaks_on_the_win_rate() -> None:
    """
    Two weekdays with the same expectancy are not equally bad: the one that
    gets there on fewer, larger wins is the one whose next quarter is less
    predictable. Tuesday and Wednesday both average -1.00 here; Wednesday hits
    on 20% and Tuesday on 40%.

    The numbers are chosen so both sums are EXACT in float - 10x5 - 15x5 and
    5x15 - 20x5 are both -25 over 25 trades. A fixture that was a tie only to
    nine decimals would be broken by the sort long before the tie-break ran,
    and the case would pass or fail on rounding rather than on the rule.
    """
    print("\nthe tie-break")
    p = _ranked_fixture({
        MON: [5.0] * 25,
        TUE: [5.0] * 10 + [-5.0] * 15,        # sum -25, mean -1.0, wr 0.40
        WED: [15.0] * 5 + [-5.0] * 20,        # sum -25, mean -1.0, wr 0.20
        THU: [5.0] * 25,
        FRI: [5.0] * 25,
    })
    row = p.set_index("weekday")
    check("the fixture really is an EXACT tie on expectancy",
          float(row.loc[TUE, "expectancy"]) == float(row.loc[WED, "expectancy"]),
          f"{row.loc[TUE, 'expectancy']} vs {row.loc[WED, 'expectancy']}")
    v = select_worst_weekday(p, min_trades=DOW_MIN_TRADES)
    check("the lower win rate loses the tie", v["worst_weekday"] == WED,
          str(v["worst_weekday"]))


def test_the_promotion_gate_promotes_everything() -> None:
    """
    STAGE 4.5 PRUNES NOTHING, and the rule is a function rather than an
    absence so it can be found and tested. A weekday chosen in-sample on the
    bars it is scored on is not evidence that an edge is absent, and pruning
    on it would drop configurations Gate R certified on the strength of a
    calendar.
    """
    print("\nthe promotion criterion")
    for verdict in ({}, {"blocked_weekday": None}, {"blocked_weekday": FRI}):
        g = promotion_gate(verdict)
        check(f"promote is True for {verdict}", g["promote"] is True)
        check(f"status is ADVANCED for {verdict}", g["status"] == "ADVANCED")


def test_the_table_marks_the_worst_row_and_the_thin_ones() -> None:
    print("\nthe console table")
    p = _ranked_fixture({MON: [10.0] * 25, TUE: [-5.0] * 25, WED: [1.0] * 3})
    v = select_worst_weekday(p, min_trades=DOW_MIN_TRADES)
    text = format_weekday_profile(p, v)
    check("the worst row is marked", "WORST" in text)
    check("the blocked row is marked", "BLOCKED" in text)
    check("a thin row is flagged rather than dropped",
          "thin (<20)" in text and "Wed" in text)
    check("the entry-attribution rule is stated under the table",
          "ENTRY session" in text)


# --------------------------------------------------------------------------
# 5. The handoff, and what Stage 5 does with it
# --------------------------------------------------------------------------
def _write_handoff(root: Path, strat: str, symbol: str, tf: str,
                   blocked: int | None) -> Path:
    d = root / "pipeline" / strat
    d.mkdir(parents=True, exist_ok=True)
    path = d / DOW_GATE_FILE.format(symbol=symbol, tf=tf)
    write_stage(path, STAGE45, strat, {
        "symbol": symbol, "timeframe": tf, "version": "A",
        "blocked_weekday": blocked,
        "blocked_day": (None if blocked is None
                        else ["Mon", "Tue", "Wed", "Thu", "Fri"][blocked]),
        "blocked_day_name": (None if blocked is None
                             else ["Monday", "Tuesday", "Wednesday",
                                   "Thursday", "Friday"][blocked]),
        "selected_in_sample": True,
        "verdict": {"worst_weekday": FRI, "worst_day": "Fri",
                    "block_rule": "negative_expectancy", "min_trades": 20,
                    "reason": "fixture"},
    })
    return path


def test_the_summary_reader_is_keyed_on_the_pair(tmp: Path) -> None:
    """
    Which weekday loses is a fact about a contract AT A TIMEFRAME. One blocked
    weekday flattened across every timeframe would stand a strategy down on a
    session two of them trade profitably, with every log line reading
    correctly.
    """
    print("\nstage45_blocked_days")
    blob = {"results": [
        {"symbol": "NQ", "timeframe": "15m", "blocked_weekday": FRI},
        {"symbol": "NQ", "timeframe": "30m", "blocked_weekday": MON},
        {"symbol": "ES", "timeframe": "15m", "blocked_weekday": None},
    ]}
    m = stage45_blocked_days(blob)
    check("each pair keeps its own weekday",
          m == {("NQ", "15m"): (FRI,), ("NQ", "30m"): (MON,)}, str(m))
    check("a pair that blocked nothing is ABSENT, not mapped to ()",
          ("ES", "15m") not in m)
    check("an empty blob is an empty mapping, not an error",
          stage45_blocked_days({}) == {} and stage45_blocked_days(None) == {})

    bad = {"results": [{"symbol": "NQ", "timeframe": "15m",
                        "blocked_weekday": 9}]}
    try:
        stage45_blocked_days(bad)
        check("a weekday outside 0-6 RAISES", False, "it did not raise")
    except ValueError as e:
        check("a weekday outside 0-6 RAISES", "0-6" in str(e))


def test_promote_reads_three_distinct_states(tmp: Path) -> None:
    """
    `NOT EVALUATED` is written rather than omitted, and is NOT an empty block
    list. "Nobody looked" and "the stage looked and every session cleared" are
    different facts about a package, and the live handle keeps them apart on
    the way back in.
    """
    print("\npromote.load_dow_gate")
    root = tmp / "artifacts"
    _write_handoff(root, "fixture_strat", "NQ", "15m", FRI)
    _write_handoff(root, "fixture_strat", "ES", "15m", None)

    blocked = load_dow_gate("fixture_strat", "NQ", "15m", out_dir=None,
                            path=(root / "pipeline" / "fixture_strat"
                                  / DOW_GATE_FILE.format(symbol="NQ", tf="15m")))
    check("a blocked pair reports EVALUATED with the weekday",
          blocked["status"] == "EVALUATED"
          and blocked["blocked_weekdays"] == [FRI]
          and blocked["blocked_weekday"] == FRI, str(blocked))
    check("the verdict's own hash travels with it",
          len(blocked["source_sha256"]) == 64)

    cleared = load_dow_gate("fixture_strat", "ES", "15m",
                            path=(root / "pipeline" / "fixture_strat"
                                  / DOW_GATE_FILE.format(symbol="ES", tf="15m")))
    check("a cleared pair is EVALUATED with an EMPTY list",
          cleared["status"] == "EVALUATED"
          and cleared["blocked_weekdays"] == [], str(cleared))
    check("and it still names the worst session it declined to block",
          cleared["worst_day"] == "Fri")

    absent = load_dow_gate("fixture_strat", "GC", "15m",
                           path=(root / "pipeline" / "fixture_strat"
                                 / DOW_GATE_FILE.format(symbol="GC", tf="15m")))
    check("a pair with no file is NOT EVALUATED, not an empty verdict",
          absent["status"] == "NOT EVALUATED"
          and absent["blocked_weekdays"] == [], str(absent))

    junk = root / "pipeline" / "fixture_strat" / "dow_gate_ZZ_15m.json"
    junk.write_text("{ not json", encoding="utf-8")
    broken = load_dow_gate("fixture_strat", "ZZ", "15m", path=junk)
    check("an unreadable verdict is UNREADABLE and blocks nothing",
          broken["status"] == "UNREADABLE"
          and broken["blocked_weekdays"] == [], str(broken))


def test_the_days_card_catches_a_package_out_of_step(tmp: Path) -> None:
    """
    meta.json is written at PROMOTION time and never revisited, so a package
    promoted before Stage 4.5 ran for its pair trades the session it was stood
    down from - with every log line reading correctly. Neither `bt-inventory`
    nor `signal-check` can see it: each reads one of the two files.
    """
    print("\nscripts/check_strategy_days.py")
    import os                                                      # noqa: PLC0415

    from scripts.check_strategy_days import scan                   # noqa: PLC0415

    root = tmp / "artifacts2"
    _write_handoff(root, "fixture_strat", "NQ", "15m", WED)

    inc = tmp / "incubator"
    for name, block in (("fixture_strat_NQ_15m_VA",
                         {"status": "EVALUATED", "blocked_weekdays": [FRI],
                          "blocked_weekday": FRI, "blocked_day": "Fri"}),
                        ("fixture_strat_GC_15m_VA", None)):
        d = inc / name
        d.mkdir(parents=True, exist_ok=True)
        meta = {"name": name, "strategy": "fixture_strat",
                "symbol": name.split("_")[2], "timeframe": "15m",
                "version": "A"}
        if block is not None:
            meta["day_of_week_gate"] = block
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # $BT_ARTIFACTS rather than `out_dir=`: `pipeline_dir`'s second argument
    # is a full directory OVERRIDE for one strategy, not an artifacts root,
    # and this card walks many strategies at once. Restored afterwards - a
    # suite that left it set would send the next one's writes to a temp
    # directory that no longer exists.
    prior = os.environ.get("BT_ARTIFACTS")
    os.environ["BT_ARTIFACTS"] = str(root)
    try:
        rows = {r["strategy_id"]: r for r in scan(incubator=inc)}
    finally:
        if prior is None:
            os.environ.pop("BT_ARTIFACTS", None)
        else:
            os.environ["BT_ARTIFACTS"] = prior
    stale = rows["fixture_strat_NQ_15m_VA"]
    check("a package whose meta disagrees with the handoff is flagged",
          stale["agrees"] is False, str(stale["agrees"]))
    check("the card reports what the live loop will ACTUALLY enforce",
          stale["blocked"] == [FRI] and stale["handoff"]["blocked"] == [WED])
    check("its active days exclude the blocked one",
          stale["active_days"] == [MON, TUE, WED, THU])

    none_yet = rows["fixture_strat_GC_15m_VA"]
    check("a package with no verdict is tri-state None, never False",
          none_yet["agrees"] is None)
    check("and it blocks nothing", none_yet["blocked"] == []
          and none_yet["status"] == "NOT EVALUATED")


if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("  STAGE 4.5 - the day-of-week gate")
    print("=" * 70)
    test_attribution_is_the_entry_session()
    test_the_metrics_are_the_ones_written_on_the_row()
    test_a_weekday_that_never_traded_is_a_row_of_zeros_not_an_absence()
    test_the_deepest_episode_is_located_on_the_realisation_curve()
    test_drawdown_contribution_is_a_share_of_that_episode()
    test_a_curve_that_never_drew_down_says_so()
    test_daily_returns_group_by_session_weekday()
    test_the_worst_weekday_is_the_lowest_expectancy()
    test_a_thin_weekday_is_never_ranked_however_badly_it_scored()
    test_no_weekday_clearing_the_floor_blocks_nothing_and_says_why()
    test_a_profitable_worst_weekday_is_named_but_not_blocked()
    test_a_tie_on_expectancy_breaks_on_the_win_rate()
    test_the_promotion_gate_promotes_everything()
    test_the_table_marks_the_worst_row_and_the_thin_ones()
    with tempfile.TemporaryDirectory() as td:
        # A REAL TEMPORARY DIRECTORY, never $BT_ARTIFACTS. These cases write
        # handoff files, and a suite that wrote them under the artifacts root
        # would put fixture JSON on the NFS mount beside real campaign
        # evidence - see tests/test_regime_profiler.py, where exactly that
        # happened because pytest called a helper whose only argument was
        # defaulted.
        test_the_summary_reader_is_keyed_on_the_pair(Path(td))
        test_promote_reads_three_distinct_states(Path(td))
        test_the_days_card_catches_a_package_out_of_step(Path(td))

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")
