"""
tests/test_stage2_charter.py - the Regime-Switching Incubator Stage 2 Charter.

Location: ~/src/trading/tests/test_stage2_charter.py

Covers the two modules the charter binds at Stage 2:

    backtest/scan.py              ingestion, the in-sample window, the
                                  no-pruning guarantee, plateau selection and
                                  the summary handoff
    backtest/discord_reporter.py  the Stage 2 parameter-optimization card

The five clauses, and how each one fails silently when nothing pins it:

  1. **Ingestion.** Stage 2's input is `surviving_assets.json`, as EXACT
     (symbol, timeframe) pairs. `--symbols` and `--tf` are two independent
     axes, so handing the survivors over as their cross product sweeps
     configurations the screen dropped - and once parameters exist for a pair
     nobody screened, nothing downstream records that it was never screened.
  2. **The in-sample window.** 2013-01-01..2022-12-31 by default and the
     holdout is never read. Stage 2 FITS what it loads, so a window running
     into 2023 spends the holdout before Gate 3 is evaluated, and the audit
     that follows is well-formed and meaningless.
  3. **No pruning.** No contract, timeframe, quadrant or parameter set is
     eliminated here on an aggregate metric, and no prop-firm rule is applied.
     A grid where nothing clears Gate 1 still produces a winner, still writes
     `best_params_<SYMBOL>_<TF>.json`, and still advances.
  4. **Plateau detection.** The winner is the best Sharpe that survives one
     step in any direction on the grid, not the grid's single best cell. A
     spike is what an overfit looks like on a parameter surface.
  5. **The handoff.** `best_params_<SYMBOL>_<TF>.json` per configuration, plus
     `stage2_summary.json` and `stage2_summary_matrix.csv` covering every
     configuration the stage was asked to optimise - errors included, because
     a shorter table reads as a complete one.

Runs without the lake and without a network: the sweep is exercised on
synthetic bars through a synthetic strategy module, and the Discord transport
is exercised in --dry-run only.

    python tests/test_stage2_charter.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import backtest.discord_reporter as dr                            # noqa: E402
from backtest.engine import BacktestConfig                        # noqa: E402
from backtest.pipeline import (CHARTER_IS_END, CHARTER_IS_START,  # noqa: E402
                               HOLDOUT_START, STAGE2_MATRIX_FILE,
                               STAGE2_SUMMARY_FILE, stage1_pairs,
                               write_stage)
from backtest.scan import (CLEAN_SELECTIONS, PLATEAU_COLUMNS,     # noqa: E402
                           RANK_PLATEAU, RANK_SHARPE, ScanError,
                           axis_order, build_parser,
                           check_in_sample_window, expand_grid,
                           plateau_scores, resolve_targets,
                           scan_from_csv, scan_symbol,
                           summary_matrix_rows, winners_leaderboard,
                           write_best_params, write_scan_table,
                           write_stage2_summary)

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}"
          + (f"  [{detail}]" if detail else ""))
    if not ok:
        _failures.append(label)
    return bool(ok)


def raises(fn, exc=Exception) -> tuple[bool, str]:
    try:
        fn()
    except exc as e:
        return True, f"{type(e).__name__}: {e}"
    except Exception as e:                                        # noqa: BLE001
        return False, f"wrong exception: {type(e).__name__}: {e}"
    return False, "did not raise"


STRATEGY_SRC = '''
"""A synthetic crossover, so the sweep has something to reject as well."""
import pandas as pd


def make_signal_fn(fast=5, slow=20):
    if fast >= slow:
        raise ValueError("fast must be below slow")

    def signal_fn(bars):
        f = bars["close"].rolling(fast).mean()
        s = bars["close"].rolling(slow).mean()
        entries = (f > s) & (f.shift(1) <= s.shift(1))
        exits = (f < s) & (f.shift(1) >= s.shift(1))
        return entries.fillna(False), exits.fillna(False)

    return signal_fn


PARAM_GRID = {"fast": [3, 5, 10], "slow": [10, 20, 40]}
'''


def synthetic_bars(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2015-01-05 14:30", periods=n, freq="15min", tz="UTC")
    close = 4000 + np.cumsum(rng.normal(0, 3.0, n))
    return pd.DataFrame({"ts": ts, "symbol": "ES", "open": close,
                         "high": close + 2.0, "low": close - 2.0,
                         "close": close, "volume": 1000})


def stage1_blob() -> dict:
    """A Stage 1 handoff with RAGGED survivors - NQ at two timeframes, GC at
    one. Ragged on purpose: it is the shape whose cross product is wrong."""
    return {
        "surviving_pairs": [
            {"symbol": "NQ", "tf": "5m", "version": "A", "status": "PROMOTED",
             "optimal_regime": "High Volatility / Trending", "quadrant": "Q1",
             "regime_pf": 1.42, "regime_trade_count": 210,
             "kill_switch_regimes": ["High Volatility / Ranging",
                                     "Low Volatility / Trending",
                                     "Low Volatility / Ranging"]},
            {"symbol": "NQ", "tf": "15m", "version": "B", "status": "PROMOTED",
             "optimal_regime": "Low Volatility / Trending", "quadrant": "Q3",
             "regime_pf": 1.11, "regime_trade_count": 64,
             "kill_switch_regimes": ["High Volatility / Trending",
                                     "High Volatility / Ranging",
                                     "Low Volatility / Ranging"]},
            {"symbol": "GC", "tf": "15m", "version": "A", "status": "PROMOTED",
             "optimal_regime": "High Volatility / Trending", "quadrant": "Q1",
             "regime_pf": 1.03, "regime_trade_count": 88,
             "kill_switch_regimes": []},
        ],
        "surviving": ["GC", "NQ"],
        "timeframes": ["5m", "15m"],
    }


# --------------------------------------------------------------------------
# 1 · ingestion
# --------------------------------------------------------------------------
def test_ingestion() -> None:
    print("\n1 · ingestion: exact Stage 1 pairs, never their cross product")

    blob = stage1_blob()
    pairs = stage1_pairs(blob)
    check("stage1_pairs reads every surviving pair", len(pairs) == 3,
          str([(p["symbol"], p["tf"]) for p in pairs]))
    check("each pair carries the regime scope it was screened in",
          all(p["quadrant"] and p["optimal_regime"] for p in pairs))

    targets, source, unscreened = resolve_targets(blob, None, None, ["15m"])
    got = [(t["symbol"], t["tf"]) for t in targets]
    check("no flags: the sweep targets the exact surviving pairs",
          got == [("NQ", "5m"), ("NQ", "15m"), ("GC", "15m")], str(got))
    check("the cross product GC·5m — dropped by Stage 1 — is NOT swept",
          ("GC", "5m") not in got)
    check("nothing is flagged as unscreened", unscreened == [], str(unscreened))
    check("the source names the handoff", "stage 1" in source, source)
    check("the scope travels onto the target",
          targets[0]["stage1"]["quadrant"] == "Q1"
          and targets[0]["in_stage1"] is True)

    # An operator naming contracts gets them, at the timeframes each survived
    # at - and a contract Stage 1 never promoted is swept and FLAGGED.
    targets, _src, unscreened = resolve_targets(blob, ["NQ", "ES"], None,
                                                ["15m"])
    got = [(t["symbol"], t["tf"]) for t in targets]
    check("--symbols alone keeps each contract's own surviving timeframes",
          got == [("NQ", "5m"), ("NQ", "15m"), ("ES", "15m")], str(got))
    check("a contract Stage 1 never promoted is swept but flagged",
          unscreened == ["ES·15m"], str(unscreened))
    check("an unscreened pair carries no regime scope",
          targets[-1]["stage1"] is None and targets[-1]["in_stage1"] is False)

    targets, src, unscreened = resolve_targets(blob, ["NQ"], ["30m"], ["15m"])
    check("--tf is an explicit override and crosses the axes",
          [(t["symbol"], t["tf"]) for t in targets] == [("NQ", "30m")],
          src)
    check("the override's unscreened pairs are still flagged",
          unscreened == ["NQ·30m"], str(unscreened))

    check("a handoff with no survivors yields no targets",
          resolve_targets({"surviving_pairs": []}, None, None, ["15m"])[0] == [])


# --------------------------------------------------------------------------
# 2 · the in-sample window
# --------------------------------------------------------------------------
def test_in_sample_window() -> None:
    print("\n2 · in-sample window: the holdout is never read")

    args = build_parser().parse_args(["--strat", "x"])
    check(f"--start defaults to the charter's {CHARTER_IS_START}",
          args.start == CHARTER_IS_START == "2013-01-01", str(args.start))
    check(f"--end defaults to the charter's {CHARTER_IS_END}",
          args.end == CHARTER_IS_END == "2022-12-31", str(args.end))

    note = check_in_sample_window(CHARTER_IS_START, CHARTER_IS_END)
    check("the charter window is accepted and named as the charter's",
          "charter" in note and "holdout untouched" in note, note)

    ok, detail = raises(
        lambda: check_in_sample_window("2013-01-01", "2023-06-30"), ScanError)
    check("an --end inside the holdout is REFUSED", ok, detail)
    check("the refusal explains why, not just that", "holdout" in detail
          and "optimis" in detail.lower(), detail)

    ok, detail = raises(
        lambda: check_in_sample_window("2013-01-01", None), ScanError)
    check("an omitted --end is refused — it runs into the holdout", ok, detail)

    ok, detail = raises(
        lambda: check_in_sample_window("2023-02-01", "2023-06-30"), ScanError)
    check("a window wholly inside the holdout is refused", ok, detail)

    ok, detail = raises(
        lambda: check_in_sample_window("2020-01-01", "2019-01-01"), ScanError)
    check("--start after --end is refused", ok, detail)

    # The one date that has to hold across three stages at once.
    check("the holdout begins the day after the charter's in-sample end",
          HOLDOUT_START == "2023-01-01" and CHARTER_IS_END == "2022-12-31")

    # There is deliberately no escape hatch: a flag that spent the holdout
    # would be used, and it can only be spent once.
    flags = " ".join(a.option_strings[0] for a in build_parser()._actions
                     if a.option_strings)
    check("no flag offers to sweep past the holdout boundary",
          "--allow-holdout" not in flags and "--force-window" not in flags)


# --------------------------------------------------------------------------
# 3 · no pruning
# --------------------------------------------------------------------------
def test_no_pruning(tmp: Path) -> None:
    print("\n3 · no pruning: every survivor is optimised and advances")

    path = tmp / "probe.py"
    path.write_text(STRATEGY_SRC, encoding="utf-8")
    bars = synthetic_bars()
    grid = {"fast": [3, 5, 10], "slow": [10, 20, 40]}

    scan = scan_symbol(path, bars, "ES", BacktestConfig(), grid,
                       strat_name="probe")

    # The synthetic series is a random walk, so nothing clears Gate 1 and every
    # aggregate is bad. Under a screening rule this configuration dies here.
    check("the sweep produced a winner even though no cell cleared Gate 1",
          scan["winner"] is not None
          and scan["selection"] not in CLEAN_SELECTIONS, scan["selection"])
    check("the selection SAYS nothing cleared Gate 1 rather than implying a pass",
          "NO COMBINATION CLEARED GATE 1" in scan["selection"],
          scan["selection"])
    check("the winner's own Gate 1 status is reported honestly",
          scan["winner"]["gate1"] != "PASS", str(scan["winner"]["gate1"]))

    out = tmp / "nodrop"
    written = write_best_params(scan, "probe", "ES", "15m", CHARTER_IS_START,
                                CHARTER_IS_END, {}, out, timeframes=["15m"],
                                stage1_pair={"quadrant": "Q1",
                                             "optimal_regime": "High "
                                                               "Volatility / "
                                                               "Trending"})
    check("a configuration that cleared no gate STILL gets best_params files",
          len(written) == 2 and all(p.exists() for p in written))

    blob = json.loads(written[0].read_text())
    check("the handoff carries the parameters it advances with",
          isinstance(blob["params"], dict) and blob["params"])
    check("the regime scope travels onto the handoff",
          (blob.get("stage1_regime") or {}).get("quadrant") == "Q1")
    check("and it is recorded as NOT applied to the sweep",
          blob.get("regime_applied_to_sweep") is False)
    check("the window travels with it, holdout marked untouched",
          blob["in_sample_window"]["holdout_touched"] is False
          and blob["in_sample_window"]["holdout_starts"] == HOLDOUT_START)

    # Prop-firm governance belongs to CrossTrade, against a live balance. None
    # of it may appear in a research handoff.
    text = written[0].read_text().lower()
    check("no prop-firm rule leaked into the Stage 2 handoff",
          not any(k in text for k in ("trailing_drawdown", "daily_loss_limit",
                                      "consistency", "prop_firm")))


# --------------------------------------------------------------------------
# 4 · plateau detection
# --------------------------------------------------------------------------
def test_plateau() -> None:
    print("\n4 · plateau detection: a shelf beats a spike")

    grid = {"fast": [3, 5, 10], "slow": [10, 20, 40]}
    combos = expand_grid(grid)

    # One cell is spectacular and its neighbours are worthless: the shape of an
    # overfit on a parameter surface.
    spike = [0.1] * 9
    spike[combos.index({"fast": 5, "slow": 20})] = 3.0
    scored = plateau_scores(combos, spike, grid)
    at = {tuple(sorted(c.items())): s for c, s in zip(combos, scored)}
    peak = at[tuple(sorted({"fast": 5, "slow": 20}.items()))]
    check("the isolated peak is flagged as a spike", peak["is_spike"] is True,
          str(peak))
    check("the spike's plateau score is its NEIGHBOURS, not its own Sharpe",
          abs(peak["plateau_score"] - 0.1) < 1e-12, str(peak["plateau_score"]))
    best = max(range(9), key=lambda i: scored[i]["plateau_score"])
    check("ranking on the plateau does not hand the sweep to the spike",
          combos[best] != {"fast": 5, "slow": 20}, str(combos[best]))

    # A genuine shelf: the winner and everything one step from it hold up.
    shelf = [0.1, 1.0, 0.1, 1.0, 1.1, 1.05, 0.1, 1.0, 0.1]
    scored = plateau_scores(combos, shelf, grid)
    best = max(range(9), key=lambda i: scored[i]["plateau_score"])
    check("a shelf's centre wins on the plateau rank",
          combos[best] == {"fast": 5, "slow": 20}, str(combos[best]))
    check("and it is not flagged as a spike",
          scored[best]["is_spike"] is False)

    # A neighbour that never traded is evidence, not an absence.
    holed = [float("nan")] * 9
    holed[combos.index({"fast": 5, "slow": 20})] = 2.0
    scored = plateau_scores(combos, holed, grid)
    peak = scored[combos.index({"fast": 5, "slow": 20})]
    check("an untraded neighbour counts as 0.0, so a hole is not a plateau",
          peak["plateau_score"] == 0.0 and peak["is_spike"] is True, str(peak))

    check("axis order is numeric, not declaration order",
          axis_order({"a": [30, 15, 20]})["a"] == [15, 20, 30])
    check("None sits at the far end of a risk axis, never dropped",
          axis_order({"tp": [2.5, 5.0, None]})["tp"] == [2.5, 5.0, None])
    check("a boolean axis keeps its declared order — nothing to sort by",
          axis_order({"trailing": [True, False]})["trailing"] == [True, False])

    single = plateau_scores([{"a": 1}], [1.5], {"a": [1]})
    check("a one-cell grid degenerates to the Sharpe rule, with no neighbours",
          single[0]["plateau_score"] == 1.5
          and single[0]["plateau_neighbours"] == 0)


def test_plateau_in_the_sweep(tmp: Path) -> None:
    print("\n4b · the sweep carries the plateau through the table and the CSV")

    path = tmp / "probe.py"
    path.write_text(STRATEGY_SRC, encoding="utf-8")
    bars = synthetic_bars()
    grid = {"fast": [3, 5, 10], "slow": [10, 20, 40]}

    scan = scan_symbol(path, bars, "ES", BacktestConfig(), grid,
                       strat_name="probe")
    check("every plateau column reaches the scan table",
          all(c in scan["table"].columns for c in PLATEAU_COLUMNS),
          str(list(scan["table"].columns)))
    check("the sweep records which rank it used", scan["rank"] == RANK_PLATEAU)
    check("the winner carries its own plateau record",
          set(scan["winner"]["plateau"]) == set(PLATEAU_COLUMNS))

    sharpe_ranked = scan_symbol(path, bars, "ES", BacktestConfig(), grid,
                                strat_name="probe", rank=RANK_SHARPE)
    top = sharpe_ranked["table"]["sharpe"].max()
    check("--select sharpe restores the single-best-cell rule",
          abs(sharpe_ranked["winner"]["sharpe"] - float(top)) < 1e-12)
    check("and the two rules are recorded as different selections",
          sharpe_ranked["selection"] != scan["selection"],
          f"{sharpe_ranked['selection']} vs {scan['selection']}")

    # The rebuild path has to agree with the sweep about who won, or the
    # `selected` flag in the CSV and the best_params file name different rows.
    csv = write_scan_table(scan, tmp / "reuse")
    rebuilt = scan_from_csv(csv, "ES")
    check("--reuse-scan re-derives the SAME winner from the table",
          rebuilt["winner"]["params"] == scan["winner"]["params"],
          f"{rebuilt['winner']['params']} vs {scan['winner']['params']}")
    check("the rebuild says which rank the table supported",
          rebuilt["rank"] == RANK_PLATEAU)

    # A table written before the plateau columns existed must fall back to the
    # Sharpe rule and SAY it did, rather than ranking on a column it lacks.
    legacy = scan["table"].drop(columns=list(PLATEAU_COLUMNS))
    legacy = legacy.assign(selected=[
        bool(v) for v in (legacy["sharpe"] == legacy["sharpe"].max())])
    legacy_csv = tmp / "legacy_scan_ES.csv"
    legacy.to_csv(legacy_csv, index=False)
    old = scan_from_csv(legacy_csv, "ES")
    check("a pre-plateau table is ranked on Sharpe and labelled as such",
          old["rank"] == RANK_SHARPE and "PLATEAU" not in old["selection"],
          f"{old['rank']} · {old['selection']}")


# --------------------------------------------------------------------------
# 5 · the handoff
# --------------------------------------------------------------------------
def test_summary_handoff(tmp: Path) -> None:
    print("\n5 · handoff: the summary matrix covers everything, errors included")

    rows = [
        {"symbol": "NQ", "timeframe": "5m", "selection": "GATE 1 PASS · best "
                                                         "Sharpe plateau",
         "winner": {"fast": 5, "slow": 20}, "sharpe": 1.31,
         "profit_factor": 1.28, "max_drawdown_pct": -8.4, "trades": 412,
         "plateau_score": 1.02, "plateau_neighbours": 4, "is_spike": False,
         "variants_tested": 9, "exclude_days": [], "exclude_days_named": [],
         "exclude_days_source": "none", "in_stage1": True,
         "stage1": {"quadrant": "Q1", "version": "A",
                    "optimal_regime": "High Volatility / Trending"}},
        {"symbol": "GC", "timeframe": "15m", "selection": "BEST SHARPE "
                                                          "PLATEAU · NO "
                                                          "COMBINATION "
                                                          "CLEARED GATE 1",
         "winner": {"fast": 3, "slow": 40}, "sharpe": -0.4,
         "profit_factor": 0.91, "max_drawdown_pct": -21.0, "trades": 120,
         "plateau_score": -0.9, "plateau_neighbours": 3, "is_spike": True,
         "variants_tested": 9, "exclude_days": [], "exclude_days_named": [],
         "exclude_days_source": "none", "in_stage1": True,
         "stage1": {"quadrant": "Q1", "version": "A",
                    "optimal_regime": "High Volatility / Trending"}},
    ]
    errors = [{"symbol": "NQ", "timeframe": "15m", "in_stage1": True,
               "stage1": {"quadrant": "Q3", "version": "B",
                          "optimal_regime": "Low Volatility / Trending"},
               "error": "ScanError: no bars for NQ - nothing to sweep"}]

    matrix = summary_matrix_rows(rows, errors)
    check("the matrix has one row per configuration ASKED for, errors included",
          len(matrix) == 3, str(len(matrix)))
    errored = [m for m in matrix if m["status"] == "ERROR"]
    check("a failed sweep is a visible row, not a shorter table",
          len(errored) == 1 and errored[0]["error"].startswith("ScanError"))
    check("a failed sweep reports NOT OPTIMIZED, never '(no winner)'",
          errored[0]["params"] == "NOT OPTIMIZED", errored[0]["params"])
    check("the target quadrant is on every row that has one",
          {m["quadrant"] for m in matrix} == {"Q1", "Q3"})
    check("the spike flag survives into the matrix",
          [m["is_spike"] for m in matrix if m["symbol"] == "GC"] == [True])

    out = tmp / "handoff"
    written = write_stage2_summary("probe", rows, errors, out,
                                   CHARTER_IS_START, CHARTER_IS_END,
                                   ["5m", "15m"], "stage 1 survivors · exact "
                                                  "pairs", [], RANK_PLATEAU,
                                   {"fast": [3, 5, 10], "slow": [10, 20, 40]})
    check("both forms of the summary are written",
          [p.name for p in written] == [STAGE2_SUMMARY_FILE,
                                        STAGE2_MATRIX_FILE],
          str([p.name for p in written]))

    blob = json.loads(written[0].read_text())
    check("the summary is stamped as Stage 2 and as this strategy",
          blob["stage"] == 2 and blob["strategy"] == "probe")
    check("the in-sample window is on the handoff, holdout untouched",
          blob["in_sample_window"]["charter_default"] is True
          and blob["in_sample_window"]["holdout_touched"] is False)
    check("coverage counts the targets and the shortfall",
          blob["coverage"]["targets"] == 3
          and blob["coverage"]["optimized"] == 2
          and blob["coverage"]["errors"] == 1
          and blob["coverage"]["complete"] is False)
    check("coverage states that a shortfall is a run failure, not a screen",
          "drops nothing" in blob["coverage"]["rule"])

    # What was asked for and what was applied are two fields, because a
    # rebuild of a table with no plateau columns is ranked on Sharpe however
    # the sweep was invoked - and the card prints the applied one.
    check("the summary records the rank that was requested",
          blob["rank_requested"] == RANK_PLATEAU)
    mixed = write_stage2_summary(
        "probe", [dict(rows[0], rank=RANK_PLATEAU),
                  dict(rows[1], rank=RANK_SHARPE)], [], tmp / "mixed",
        CHARTER_IS_START, CHARTER_IS_END, ["5m"], "test", [], RANK_PLATEAU)
    check("a run whose rows were ranked differently reports 'mixed', not the "
          "request", json.loads(mixed[0].read_text())["rank"] == "mixed")
    agreed = write_stage2_summary(
        "probe", [dict(rows[0], rank=RANK_SHARPE)], [], tmp / "agreed",
        CHARTER_IS_START, CHARTER_IS_END, ["5m"], "test", [], RANK_PLATEAU)
    check("and a rebuild that could only rank on Sharpe says Sharpe",
          json.loads(agreed[0].read_text())["rank"] == RANK_SHARPE)

    frame = pd.read_csv(written[1])
    check("the CSV matrix carries the same rows", len(frame) == 3)
    check("the CSV names the winning parameters per configuration",
          "fast=5, slow=20" in set(frame["params"].astype(str)))

    table = winners_leaderboard(rows)
    check("the leaderboard shows the target quadrant", " QUAD " in table)
    check("the leaderboard shows the plateau, with the spike marked",
          "PLATEAU" in table and "!" in table)
    return blob


# --------------------------------------------------------------------------
# the Stage 2 Discord card
# --------------------------------------------------------------------------
def test_stage2_card(blob: dict) -> None:
    print("\nthe Stage 2 card: every field the charter asks for")

    embed = dr.build_stage2_embed("probe", blob,
                                  source="/mnt/backtest/x/stage2_summary.json")
    text = embed["description"]

    check("the card names the stage and the strategy",
          "Stage 2" in embed["title"] and "probe" in embed["title"],
          embed["title"])
    check("the in-sample window is on the card",
          f"{CHARTER_IS_START} → {CHARTER_IS_END}" in text, text[:120])
    check("the card states the holdout was untouched",
          HOLDOUT_START in text and "untouched" in text)
    # The per-configuration table and the full-parameter blocks are GONE.
    # Both were fixed-width monospace inside a container that reflows, and on
    # a 47-configuration run they took 5,212 of Discord's 6,000 characters.
    # Every value they carried is a column of stage2_summary_matrix.csv.
    check("no monospace block survives anywhere on the card - not in the "
          "description and not in a field",
          "```" not in text
          and not any("```" in str(f["value"]) for f in embed["fields"]),
          text)
    check("...and the card names the file that carries the detail instead",
          "stage2_summary_matrix.csv" in text, text)
    # Parameters, per-row drawdowns and the quadrant legend all moved to
    # stage2_summary_matrix.csv. What the card keeps is the range, because a
    # reader needs to know whether the sweep produced anything usable before
    # deciding to open the file.
    check("no per-row parameter set is left on the card",
          "fast=5, slow=20" not in text
          and not any("fast=5" in str(f["value"]) for f in embed["fields"]),
          text)
    check("the optimised profit factor RANGE is on the card, so the sweep's "
          "outcome is legible without opening the CSV",
          "Optimised PF" in text and "1.28" in text, text)
    check("...as a range and a median, never a mean - averaging profit "
          "factors across contracts blends separate simulations on different "
          "multipliers into a number that describes no instrument",
          "median" in text and "mean" not in text.lower(), text)

    fields = {f["name"]: f["value"] for f in embed["fields"]}
    check("the card counts the configurations it covers",
          fields["Configurations"] == "3", str(fields))
    check("it reports optimised → Stage 3, not 'promoted'",
          "Optimised → Stage 3" in fields and fields["Optimised → Stage 3"] == "2")
    check("a failed sweep is counted on the card",
          fields["Failed to sweep"] == "1")
    check("the card states that nothing was pruned",
          "none" in fields["Pruning"].lower(), fields["Pruning"])
    check("the embed fits Discord's limits",
          dr._embed_size(embed) <= dr.MAX_EMBED_TOTAL
          and len(text) <= dr.MAX_EMBED_DESCRIPTION,
          str(dr._embed_size(embed)))

    # Nothing on the card may be recomputed: it is a transcription.
    tampered = json.loads(json.dumps(blob))
    tampered["results"][0]["profit_factor"] = 9.99
    check("the card prints the profit factor it was given, recomputing nothing",
          "9.99" in dr.build_stage2_embed("probe", tampered)["description"])

    empty = dr.build_stage2_embed("probe", {"results": []})
    check("a stage that optimised nothing gets an amber card, not a crash",
          empty["color"] == dr.AMBER
          and [f["value"] for f in empty["fields"]
               if f["name"] == "Configurations"] == ["0"],
          str(empty["fields"]))

    many = {"results": [{"symbol": f"S{i}", "timeframe": "15m",
                         "status": "OPTIMIZED", "quadrant": "Q1",
                         "optimal_regime": "High Volatility / Trending",
                         "params": "fast=5, slow=20", "profit_factor": 1.1,
                         "max_drawdown_pct": -5.0} for i in range(60)]}
    wide = dr.build_stage2_embed("probe", many, max_rows=dr.STAGE2_MAX_ROWS)
    check("a long matrix no longer grows the description at all - there is "
          "no table to truncate, so there is no truncation to under-report",
          len(wide["description"]) < 512, str(len(wide["description"])))
    check("the totals still describe the whole matrix",
          [f["value"] for f in wide["fields"]
           if f["name"] == "Configurations"] == ["60"])

    # The card no longer carries the parameter sets, so the one part of it
    # that used to grow without bound cannot push an embed past 6000. The
    # worst case is now a constant.
    heavy = {"results": [
        {"symbol": f"SYM{i}", "timeframe": "15m", "status": "OPTIMIZED",
         "quadrant": "Q1", "optimal_regime": "High Volatility / Trending",
         "params": ", ".join(f"param_number_{j}={j}.0" for j in range(12)),
         "profit_factor": 1.1, "max_drawdown_pct": -5.0} for i in range(60)]}
    big = dr.build_stage2_embed("probe", heavy, source="/mnt/x/s.json")
    check("60 configurations of wide parameter sets stay far inside the "
          "embed limit",
          dr._embed_size(big) <= dr.MAX_EMBED_TOTAL // 2,
          str(dr._embed_size(big)))
    check("...and no parameter field is emitted at all",
          not [f for f in big["fields"]
               if f["name"].startswith(dr.STAGE2_PARAM_FIELD_NAME)],
          str([f["name"] for f in big["fields"]]))

    # format_stage2_param_fields is retained though the card no longer calls
    # it - it keeps its own coverage so it is ready if the block ever returns
    # behind a flag.
    # A budget too small for even one field must not leave the description
    # pointing at a block that is not there.
    starved = dr.format_stage2_param_fields(
        [{"symbol": "NQ", "timeframe": "15m", "params": "fast=5"}], budget=40)
    check("a card with no room for the block reports every row as not shown",
          starved == ([], 1), str(starved))


def test_stage1_version_spellings() -> None:
    print("\nthe Version B trigger cannot be disarmed by a spelling")
    from backtest.scan import stage1_version_of

    for scope, want, why in [
        ({"version": "B"}, "B", "what surviving_assets.json has always written"),
        ({"version": "VB"}, "B", "a display prefix must not disarm a gate"),
        ({"stage1_version": "B"}, "B", "the name every reader downstream uses"),
        ({"stage1_version": "VB"}, "B", "both, together"),
        ({"version": " b "}, "B", "whitespace and case are not a version"),
        ({"version": "A"}, "A", None),
        ({"version": "VA"}, "A", None),
        ({}, "", "absent is NOT Version A - it leaves the pass to --ml"),
        (None, "", "and neither is a missing scope"),
    ]:
        check(f"{str(scope):26} -> {want!r}" + (f"  ({why})" if why else ""),
              stage1_version_of(scope) == want, stage1_version_of(scope))


def test_baseline_pf_carry() -> None:
    print("\nStage 1's PF carried into the matrix, with its scope")
    from backtest.scan import summary_matrix_rows

    rows = [
        {"symbol": "6A", "timeframe": "15m", "status": "OPTIMIZED",
         "in_stage1": True, "winner": {"fast_period": 9},
         "stage1": {"quadrant": "Q1", "version": "B", "regime_pf": 1.03,
                    "optimal_regime": "High Volatility / Trending"},
         "profit_factor": 0.9439, "trades": 603},
        {"symbol": "XX", "timeframe": "1h", "status": "OPTIMIZED",
         "in_stage1": False, "stage1": None, "winner": {},
         "profit_factor": 1.2},
    ]
    out = {(r["symbol"], r["timeframe"]): r
           for r in summary_matrix_rows(rows, [])}
    surv, unscreened = out[("6A", "15m")], out[("XX", "1h")]

    check("Stage 1's regime_pf lands in the matrix as baseline_pf",
          surv["baseline_pf"] == 1.03, str(surv["baseline_pf"]))
    check("...beside the sweep's own factor as optimized_pf",
          surv["optimized_pf"] == 0.9439, str(surv["optimized_pf"]))
    check("...and profit_factor is KEPT as its alias, because this same dict "
          "is `results` in stage2_summary.json and the card and Stage 3's "
          "target resolution both bind to that key",
          surv["profit_factor"] == surv["optimized_pf"])
    check("the SCOPE travels with the number - the two factors are measured "
          "on different bars (one quadrant against the whole window, since "
          "regime_applied_to_sweep is False) and subtracting them is wrong",
          surv["baseline_pf_scope"] == "stage 1 designated quadrant only",
          str(surv["baseline_pf_scope"]))
    check("a pair with no Stage 1 record carries no baseline and no scope "
          "either - a scope beside an absent number would describe nothing",
          (unscreened["baseline_pf"], unscreened["baseline_pf_scope"])
          == (None, None), str(unscreened))


def test_mode_resolution() -> None:
    print("\nmode resolution: --stage 2 and --mode scan are one choice")

    check("--stage 2 selects the scan card", dr.resolve_mode(None, "2") == "scan")
    check("--mode scan selects it too", dr.resolve_mode("scan", None) == "scan")
    check("agreeing flags are accepted", dr.resolve_mode("scan", "2") == "scan")
    ok, detail = raises(lambda: dr.resolve_mode("scan", "5"), ValueError)
    check("disagreeing flags are refused rather than guessed at", ok, detail)
    check("the other two modes still resolve",
          dr.resolve_mode(None, "1") == "baseline"
          and dr.resolve_mode(None, None) == "promotion")


def test_cli(tmp: Path, blob: dict) -> None:
    print("\nthe CLI: --stage 2 reads the handoff and refuses the wrong one")

    pipeline = tmp / "cli"
    write_stage(pipeline / STAGE2_SUMMARY_FILE, 2, "probe", blob)

    out = subprocess.run(
        [sys.executable, str(REPO / "backtest" / "discord_reporter.py"),
         "--stage", "2", "--strat", "probe",
         "--summary", str(pipeline / STAGE2_SUMMARY_FILE), "--dry-run"],
        capture_output=True, text=True, timeout=120)
    check("--stage 2 --dry-run builds a payload and sends nothing",
          out.returncode == 0 and "DRY RUN" in out.stdout, out.stderr[-300:])
    payload = json.loads(out.stdout.split("DRY RUN")[0])
    check("the payload is one Stage 2 embed",
          len(payload["embeds"]) == 1
          and "Stage 2" in payload["embeds"][0]["title"])

    # Stage 1's handoff read as Stage 2's: refused, not posted under the wrong
    # heading. A Discord card is exactly the artifact nobody cross-checks.
    write_stage(pipeline / "wrong.json", 1, "probe", {"surviving_pairs": []})
    out = subprocess.run(
        [sys.executable, str(REPO / "backtest" / "discord_reporter.py"),
         "--stage", "2", "--strat", "probe",
         "--summary", str(pipeline / "wrong.json"), "--dry-run"],
        capture_output=True, text=True, timeout=120)
    check("a Stage 1 file handed to --stage 2 is refused",
          out.returncode == 1 and "stage 1" in out.stderr.lower(),
          out.stderr[-200:])

    out = subprocess.run(
        [sys.executable, str(REPO / "backtest" / "discord_reporter.py"),
         "--stage", "2", "--strat", "other",
         "--summary", str(pipeline / STAGE2_SUMMARY_FILE), "--dry-run"],
        capture_output=True, text=True, timeout=120)
    check("another strategy's summary is refused",
          out.returncode == 1 and "strategy" in out.stderr.lower(),
          out.stderr[-200:])

    out = subprocess.run(
        [sys.executable, str(REPO / "backtest" / "scan.py"),
         "--strat", "ema_crossover_20260821", "--symbols", "NQ",
         "--start", "2013-01-01", "--end", "2023-06-30"],
        capture_output=True, text=True, timeout=120)
    check("scan.py refuses a holdout window before it reads a bar",
          out.returncode == 1 and "holdout" in out.stderr.lower(),
          out.stderr[-200:])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage2charter_") as td:
        tmp = Path(td)
        test_ingestion()
        test_in_sample_window()
        test_no_pruning(tmp)
        test_plateau()
        test_plateau_in_the_sweep(tmp)
        blob = test_summary_handoff(tmp)
        test_stage2_card(blob)
        test_baseline_pf_carry()
        test_stage1_version_spellings()
        test_mode_resolution()
        test_cli(tmp, blob)

    print("\n" + "=" * 60)
    if _failures:
        print(f"  {len(_failures)} CHECK(S) FAILED")
        for f in _failures:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
