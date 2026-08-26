#!/usr/bin/env python3
"""
tests/test_check_portfolio_assets.py - the portfolio allocation card.

Location: ~/src/trading/tests/test_check_portfolio_assets.py

    .venv/bin/python3 -m pytest tests/test_check_portfolio_assets.py -v

ASSERT-BASED so `tests/conftest.py` collects it case by case. Every helper is
named `_...`: pytest collects any module-level `test_*` it can call, including
one whose only argument is defaulted, and `tests/test_regime_profiler.py` was
bitten by exactly that - a helper collected that way ran without its redirect
and wrote onto the NFS mount.

WHAT IS WORTH PINNING
---------------------
This card's job is to say who may trade what, so the failures that matter are
the ones where it would quietly mis-state a permission:

  * **The routing matrix must come from `contract_alias`.** Two spellings of
    "MNQ means NQ" that drifted would put an order on the wrong contract with
    every line of this card reading correctly.
  * **Empty portfolios must be SHOWN.** Three of the four on this box are
    empty by design; filtering them out makes "nothing is allocated here"
    indistinguishable from "this portfolio does not exist".
  * **A disagreement between the routing table and the certification** about
    which quadrant a strategy is for is invisible downstream - the daemon
    gates on one and a reader trusts the other.
  * **Unallocated streams come from the SPOOL**, not the regime file, which
    lists only the daemon's registered targets and would compare a set against
    itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from realtime import check_portfolio_assets as pa                  # noqa: E402


# ==========================================================================
# fixtures
# ==========================================================================

def _portfolio(name="Incubator-Odd", account="SimIncubator1", active=(),
               assets=("MNQ", "MCL"), regime_filter="Q2"):
    return {name: {
        "portfolio_id": name, "account_type": "incubator_sim",
        "target_account": account, "default_account_size": 50000,
        "risk_profile": {"fixed_risk_budget_usd": 250.0,
                         "max_trailing_drawdown_usd": 2500.0,
                         "clamping": {"min_contracts": 1, "max_contracts": 5}},
        "basket": {"assets": list(assets),
                   "correlation_group": "Index_Energy_Uncorrelated"},
        "active_strategies": list(active),
        "strategy_allocations": {a: {"symbol": "NQ", "timeframe": "1h",
                                     "allocation": 1, "status": "incubating",
                                     "regime_filter": regime_filter}
                                 for a in active}}}


def _gate(entries=False, status="MUTED", reason="indicator_warmup",
          certified="Q2"):
    return {"strategy_id": "demo_NQ_1h", "symbol": "NQ", "timeframe": "1h",
            "optimal_regime": certified,
            "optimal_regime_label": "Q2_HIGH_VOL_CHOP",
            "live_quadrant": "Q0", "live_regime": "Q0_UNDEFINED_WARMUP",
            "status": status, "reason": reason, "entries_allowed": entries,
            "exits_allowed": True}


def _regime(n_bars=19, gate=None):
    return {"symbols": {"NQ": {"quadrant": "Q0", "n_bars": n_bars,
                               "tf": "1h"}},
            "strategies": {"demo_NQ_1h": gate if gate is not None else _gate()}}


def _meta():
    return {"name": "demo_NQ_1h", "symbols": ["NQ"], "timeframe": "1h",
            "version": "A",
            "risk": {"sl_atr_mult": 2.0, "tp_atr_mult": 2.0,
                     "trailing": False}}


def _wire(tmp: Path, *, cfg=None, regime=None, meta=_meta(),
          spool=("NQ_1m.csv", "MNQ_1m.csv")) -> None:
    """
    Rebind every module-level path to `tmp`.

    Rebound rather than set through os.environ because several are resolved at
    IMPORT time, and a test setting the variable afterwards would read the real
    files under `data/` - which on this box are the LIVE ones.
    """
    pa.PORTFOLIO_CONFIG = tmp / "portfolios.json"
    if cfg is not None:
        pa.PORTFOLIO_CONFIG.write_text(json.dumps(cfg))
    pa.REGIME_STATE = tmp / "live_regime_state.json"
    if regime is not None:
        pa.REGIME_STATE.write_text(json.dumps(regime))
    pa.INCUBATOR = tmp / "incubator"
    if meta is not None:
        d = pa.INCUBATOR / "demo_NQ_1h"
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta))
    sp = tmp / "spool"
    sp.mkdir(parents=True, exist_ok=True)
    for n in spool:
        (sp / n).write_text("ts,open,high,low,close,volume\n")
    os.environ["BT_NT8_SPOOL"] = str(sp)


def _unwire() -> None:
    os.environ.pop("BT_NT8_SPOOL", None)


def _card(tmp: Path, **kw) -> str:
    _wire(tmp, **kw)
    try:
        snap = pa.collect()
    finally:
        _unwire()
    snap["interlock"] = {**snap["interlock"], "running": True,
                         "effective": True, "unit_active": "active"}
    return pa.render(snap)


# ==========================================================================
# the copies this file keeps
# ==========================================================================

def test_the_micro_names_match_the_contract_specs():
    """
    THE COPY IS CHECKED, NOT TRUSTED. Six strings are restated because
    `backtest.specs` pulls numpy and pandas for them.
    """
    from backtest.specs import SPECS                        # noqa: PLC0415
    for micro, name in pa.MICRO_NAMES.items():
        assert micro in SPECS, f"{micro} is named here and absent from SPECS"
        assert SPECS[micro].name == name, (
            f"{micro}: this file says {name!r}, SPECS says "
            f"{SPECS[micro].name!r}")


def test_the_routing_matrix_is_the_alias_tables_own(tmp_path):
    """
    The MAPPING is never restated - only the display names are. Two spellings
    of "MNQ means NQ" that drifted would route an order to the wrong contract
    with every line of this card reading correctly.
    """
    from realtime.contract_alias import MICRO_TO_PARENT     # noqa: PLC0415
    card = _card(tmp_path, cfg={"portfolios": _portfolio()},
                 regime=_regime())
    for micro, parent in MICRO_TO_PARENT.items():
        assert f"{parent:<4} → {micro:<4}" in card, f"{parent}->{micro} missing"
    assert set(pa.MICRO_NAMES) == set(MICRO_TO_PARENT), \
        "a micro gained or lost a name without the mapping changing"


# ==========================================================================
# multi-portfolio parsing
# ==========================================================================

def test_every_portfolio_is_listed_including_the_empty_ones(tmp_path):
    """
    Three of the four on this box are empty BY DESIGN. Filtering them out
    makes "nothing is allocated here" indistinguishable from "this portfolio
    does not exist", and the second is a configuration fault.
    """
    cfg = {"version": "1.1.0", "portfolios": {
        **_portfolio("Incubator-Odd", "SimIncubator1", ("demo_NQ_1h",)),
        **_portfolio("Incubator-Even", "SimIncubator2"),
        **_portfolio("Prop-Odd", "SimProp1"),
        **_portfolio("Prop-Even", "SimProp2")}}
    card = _card(tmp_path, cfg=cfg, regime=_regime())

    assert "4 (1 with an active allocation, 3 empty by design)" in card
    for name in ("Incubator-Odd", "Incubator-Even", "Prop-Odd", "Prop-Even"):
        assert f"--- Portfolio: {name} ---" in card
    assert card.count("none — this portfolio is empty") == 3
    assert "SimIncubator1" in card and "SimProp2" in card


def test_the_allocation_carries_the_promoted_parameters(tmp_path):
    card = _card(tmp_path,
                 cfg={"portfolios": _portfolio(active=("demo_NQ_1h",))},
                 regime=_regime())
    assert "Full-size tape : NQ @ 1h (resampled from the 1m spool)" in card
    assert "Execution      : MNQ · allocation 1 · incubating" in card
    assert "SL=2.0 ATR, TP=2.0 ATR, trailing=False · version A" in card
    assert "Target regime  : Q2 (Q2_HIGH_VOL_CHOP)" in card


def test_warmup_progress_is_shown_against_the_threshold(tmp_path):
    card = _card(tmp_path,
                 cfg={"portfolios": _portfolio(active=("demo_NQ_1h",))},
                 regime=_regime(n_bars=19))
    assert f"WARM-UP 19/{pa.MIN_BARS_FOR_REGIME} bars" in card
    assert "Q0_UNDEFINED_WARMUP" in card


def test_a_warm_strategy_shows_its_gate_rather_than_a_bar_count(tmp_path):
    card = _card(tmp_path,
                 cfg={"portfolios": _portfolio(active=("demo_NQ_1h",))},
                 regime=_regime(n_bars=40, gate=_gate(
                     entries=True, status="LIVE", reason="regime_match")))
    assert "ENTRIES ALLOWED (LIVE · regime_match)" in card
    assert "WARM-UP" not in card


# ==========================================================================
# the faults that are otherwise invisible
# ==========================================================================

def test_an_allocated_strategy_that_was_never_promoted_is_named(tmp_path):
    card = _card(tmp_path,
                 cfg={"portfolios": _portfolio(active=("demo_NQ_1h",))},
                 regime=_regime(), meta=None)
    assert "NOT FOUND" in card
    assert "allocates a strategy nothing promoted" in card


def test_a_strategy_with_no_switchboard_entry_is_named(tmp_path):
    card = _card(tmp_path,
                 cfg={"portfolios": _portfolio(active=("demo_NQ_1h",))},
                 regime={"symbols": {}, "strategies": {}})
    assert "NO SWITCHBOARD ENTRY" in card


def test_a_routing_table_that_disagrees_with_the_certification_is_flagged(tmp_path):
    """
    `regime_filter` in the routing table and `optimal_regime` on the
    switchboard are two records of the same decision. Disagreeing, the daemon
    gates on one and a reader trusts the other, and nothing names it.
    """
    card = _card(tmp_path,
                 cfg={"portfolios": _portfolio(active=("demo_NQ_1h",),
                                               regime_filter="Q1")},
                 regime=_regime(gate=_gate(certified="Q2")))
    assert "[!] regime_filter in the routing table is Q1" in card
    assert "certification says Q2" in card


def test_matching_regimes_are_not_flagged(tmp_path):
    card = _card(tmp_path,
                 cfg={"portfolios": _portfolio(active=("demo_NQ_1h",),
                                               regime_filter="Q2")},
                 regime=_regime(gate=_gate(certified="Q2")))
    assert "[!] regime_filter" not in card


# ==========================================================================
# unallocated streams
# ==========================================================================

def test_unallocated_streams_come_from_the_spool(tmp_path):
    """
    The regime file lists only the daemon's REGISTERED targets, so sourcing
    this there compares a set against itself and reports nothing, every time.
    """
    card = _card(tmp_path,
                 cfg={"portfolios": _portfolio(active=("demo_NQ_1h",))},
                 regime=_regime(),
                 spool=("NQ_1m.csv", "MNQ_1m.csv", "ZB_1m.csv", "ZW_1m.csv"))
    assert "2 contract(s) arriving that no allocated strategy watches" in card
    assert "ZB, ZW" in card
    tail = card.split("no allocated strategy watches")[1]
    assert "NQ" not in tail.split("Spooled")[0], "NQ is allocated"
    assert "MNQ" not in tail.split("Spooled")[0], "MNQ is its execution micro"


def test_a_basket_asset_counts_as_watched_even_with_no_strategy(tmp_path):
    """
    A contract in the permitted basket is configured to be traded, so it is
    not "unallocated" in the sense that matters - it is waiting for a
    strategy, not arriving unasked.
    """
    card = _card(tmp_path,
                 cfg={"portfolios": _portfolio(assets=("MNQ", "MCL"))},
                 regime=_regime(), spool=("MNQ_1m.csv", "MCL_1m.csv"))
    assert "every arriving contract is watched" in card


def test_a_declared_symbol_is_separated_from_the_unallocated(tmp_path):
    """
    FDAX is recorded in `contract_alias.NOT_TRADED`, so it belongs in the
    declared bucket with its reason - not in the list of contracts nobody has
    accounted for. Asserted on the COLLECTED DATA rather than by slicing the
    rendered card: the FDAX line itself contains the word DECLARED, so a
    string split lands mid-line and proves nothing.
    """
    _wire(tmp_path, cfg={"portfolios": _portfolio(active=("demo_NQ_1h",))},
          regime=_regime(),
          spool=("NQ_1m.csv", "MNQ_1m.csv", "FDAX_1m.csv", "ZW_1m.csv"))
    try:
        un = pa.collect()["unallocated"]
    finally:
        _unwire()

    assert un["unallocated"] == ["ZW"], "only ZW is unaccounted for"
    assert [s for s, _why in un["declared"]] == ["FDAX"]
    assert "EUR 25/point" in un["declared"][0][1]

    card = _card(tmp_path, cfg={"portfolios": _portfolio(active=("demo_NQ_1h",))},
                 regime=_regime(),
                 spool=("NQ_1m.csv", "MNQ_1m.csv", "FDAX_1m.csv", "ZW_1m.csv"))
    assert "DECLARED NOT TRADED" in card
    assert "1 contract(s) arriving that no allocated strategy watches" in card


# ==========================================================================
# missing and malformed state
# ==========================================================================

def test_an_unreadable_config_is_reported_not_raised(tmp_path):
    pa.PORTFOLIO_CONFIG = tmp_path / "portfolios.json"
    pa.PORTFOLIO_CONFIG.write_text("{not json")
    snap = pa.collect()
    assert snap["portfolios"]["error"]
    card = pa.render(snap)
    assert "unreadable" in card
    assert "Traceback" not in card


def test_a_missing_regime_file_leaves_the_allocation_visible(tmp_path):
    """
    The routing table is still readable with no live state, and what it says
    is still worth showing - the gate columns are what go unknown, not the
    allocation.
    """
    card = _card(tmp_path,
                 cfg={"portfolios": _portfolio(active=("demo_NQ_1h",))},
                 regime=None)
    assert "demo_NQ_1h" in card
    assert "NO SWITCHBOARD ENTRY" in card
    assert "Traceback" not in card


def test_no_portfolios_at_all_does_not_crash(tmp_path):
    card = _card(tmp_path, cfg={"portfolios": {}}, regime=_regime())
    assert "0 (0 with an active allocation" in card
    assert "Traceback" not in card


# ==========================================================================
# cost and the executable
# ==========================================================================

def test_the_tool_does_not_import_pandas():
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "import realtime.check_portfolio_assets;"
         "print(','.join(m for m in ('pandas','numpy','vectorbtpro')"
         "                if m in sys.modules))" % str(REPO)],
        capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"pulled in: {out.stdout.strip()}"


def test_it_runs_from_any_directory():
    out = subprocess.run(
        [sys.executable, str(REPO / "realtime" / "check_portfolio_assets.py")],
        capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=180)
    assert out.returncode == 0
    assert "PORTFOLIO ASSETS & ALLOCATION STATUS" in out.stdout
    assert "Micro Execution Routing Matrix" in out.stdout
    assert "Traceback" not in out.stderr


def test_it_writes_nothing(tmp_path):
    _wire(tmp_path, cfg={"portfolios": _portfolio()}, regime=_regime())
    try:
        before = sorted(p.name for p in tmp_path.iterdir())
        pa.collect()
        assert sorted(p.name for p in tmp_path.iterdir()) == before
    finally:
        _unwire()
