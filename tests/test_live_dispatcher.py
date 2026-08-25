"""
tests/test_live_dispatcher.py - the end-to-end live execution loop.

ASSERT-BASED, so pytest collects it case by case rather than routing it to
`tests/test_suite_runners.py` (see `tests/conftest.py`).

    .venv/bin/pytest tests/test_live_dispatcher.py -q

NOTHING HERE OPENS A SOCKET. Every case either runs in `dry_run` or injects a
sender, and one case asserts specifically that dry-run never calls the sender
at all - a live-execution test suite that could reach a network on a bad day is
not a test suite anybody should run.

Beyond the four items the specification asks for, this pins the properties that
fail silently:

  * a strategy whose `strat.py` no longer matches its recorded SHA-256 is
    refused, because that hash is the only thing making a promoted file
    provably the file the metrics describe;
  * a strategy certified on one contract cannot trade another because a config
    line put it in the same basket;
  * a TIMEOUT is never retried - a retried market order is a duplicate position
    this process cannot undo;
  * live mode refuses to START without a webhook, rather than evaluating every
    gate and then silently sending nothing;
  * `payloads` equals what `PortfolioManager.build_order_payloads` returns for
    the same input, so the dispatcher's reduction cannot drift from the
    manager's.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.config_loader import clear_cache                     # noqa: E402
from portfolio.portfolio_manager import PortfolioManager            # noqa: E402
from realtime.live_dispatcher import (                              # noqa: E402
    LiveDispatchError,
    LiveExecutionDispatcher,
    _is_safe_to_retry,
    load_env_file,
    resolve_credentials,
)

REAL_CONFIG = REPO_ROOT / "config" / "portfolios.json"

# Incubator-Odd holds MNQ/MCL and declares ALL FOUR quadrants since
# 2026-08-24, so nothing routed there can be stood down on regime any more -
# the partition split assets and quadrants on one axis, and MNQ lives only in
# that basket, so a strategy certified in a HIGH-volatility quadrant was
# routable to no account at all.
PERMITTED_QUADRANT = "Q4"
PERMITTED_LABEL = "Q4_LOW_VOL_MEAN_REVERSION"

# The stand-down cases therefore run on the EVEN track, which still declares
# two. MGC rather than MES because the sizing constants above are MNQ's and
# these cases assert a DECLINE - no contract count is reached at all.
GATED_PORTFOLIO = "Incubator-Even"
GATED_SYMBOL = "MGC"
# ...and the fixture has to be CERTIFIED on that contract, or it is declined by
# the symbol gate before the regime gate is ever reached and the case silently
# stops testing what it is named for.
GATED_CERTIFIED = ("GC",)
FORBIDDEN_QUADRANT = "Q3"
FORBIDDEN_LABEL = "Q3_LOW_VOL_TREND"

# MNQ point_value is 2.0. ATR 25.0 x stop 1.5 x 2.0 = $75 per contract, and a
# $250 budget buys floor(250/75) = 3 - inside the 1..5 clamp, so the number
# under test is the SIZER's and not the clamp's.
ATR = 25.0
SL_ATR_MULT = 1.5
EXPECTED_CONTRACTS = 3


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
STRATEGY_SRC = '''"""A fixture strategy whose direction is a bound parameter."""
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {"side": "long"}


def make_signal_fn(side="long", sl_atr_mult=1.5, tp_atr_mult=2.0,
                   exit_on_last=False):
    def signal_fn(bars):
        n = len(bars)
        false = pd.Series(np.zeros(n, dtype=bool))
        entries = false.copy()
        short_entries = false.copy()
        exits = false.copy()
        if side == "long":
            entries.iloc[-1] = True
        elif side == "short":
            short_entries.iloc[-1] = True
        elif side == "both":
            entries.iloc[-1] = True
            short_entries.iloc[-1] = True
        if exit_on_last:
            exits.iloc[-1] = True
        return entries, exits, short_entries, false.copy()
    return signal_fn
'''


def write_strategy(root: Path, strategy_id: str, *, side="long",
                   symbols=("NQ",), sl_atr_mult=SL_ATR_MULT, tp_atr_mult=2.0,
                   exit_on_last=False, corrupt_hash=False) -> Path:
    """A promoted-strategy directory: strat.py, meta.json, honest SHA-256."""
    directory = root / strategy_id
    directory.mkdir(parents=True, exist_ok=True)
    module = directory / "strat.py"
    module.write_text(STRATEGY_SRC)
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    if corrupt_hash:
        digest = "0" * 64
    (directory / "meta.json").write_text(json.dumps({
        "name": strategy_id,
        "version": "A",
        "symbols": list(symbols),
        "timeframe": "15m",
        "params": {"side": side, "sl_atr_mult": sl_atr_mult,
                   "tp_atr_mult": tp_atr_mult, "exit_on_last": exit_on_last},
        "risk": {"sl_atr_mult": sl_atr_mult, "tp_atr_mult": tp_atr_mult,
                 "trailing": False},
        "promoted_sha256": digest,
        "source_sha256": digest,
        "gate_audit_status": "PASS",
    }, indent=2))
    return directory


def write_config(tmp_path: Path, assignments: dict) -> Path:
    """
    The real config with `active_strategies` filled in per portfolio.

    Every portfolio is CLEARED first, and only then are `assignments` applied.
    These cases count handles and errors — "one strategy loaded, one error" —
    and a portfolio the case did not name still carries whatever
    `config/portfolios.json` holds today. Once `backtest/promote.py` started
    registering promotions automatically that stopped being empty, and a real
    allocation on an untouched portfolio silently became a second handle in
    every count. The fixture is meant to fix the assignment set, not to inherit
    the operator's.
    """
    blob = json.loads(REAL_CONFIG.read_text())
    for portfolio in blob["portfolios"].values():
        portfolio["active_strategies"] = []
        portfolio.pop("strategy_allocations", None)
    for pid, strategies in assignments.items():
        blob["portfolios"][pid]["active_strategies"] = list(strategies)
    path = tmp_path / "portfolios.json"
    path.write_text(json.dumps(blob, indent=2))
    clear_cache()
    return path


def write_state(tmp_path: Path, per_symbol: dict) -> Path:
    """
    A live regime state file in the shape the daemon publishes.

    Written by hand rather than by running the daemon: these tests are about
    the DISPATCHER's reaction to a regime, and generating one through
    pandas_ta would make the quadrant under test a function of a fixture's
    volatility instead of a value the case chose.
    """
    now = datetime.now(timezone.utc)
    symbols = {}
    for symbol, spec in per_symbol.items():
        quadrant = spec["quadrant"]
        symbols[symbol] = {
            "symbol": symbol, "tf": "15m",
            "quadrant": quadrant,
            "regime": spec.get("regime", {
                "Q1": FORBIDDEN_LABEL, "Q4": PERMITTED_LABEL}.get(quadrant)),
            "adx_14": spec.get("adx_14", 15.0),
            "atr_14": spec.get("atr_14", ATR),
            "theta_vol": 7.9,
            "is_high_vol": quadrant in ("Q1", "Q2"),
            "is_trending": quadrant in ("Q1", "Q3"),
            "bar_ts": (now - timedelta(seconds=30)).isoformat(),
            "updated_at": now.isoformat(),
            "written_at": now.isoformat(),
        }
    path = tmp_path / "live_regime_state.json"
    path.write_text(json.dumps({"schema_version": "1.0.0",
                                "updated_at": now.isoformat(),
                                "symbols": symbols}, indent=2))
    return path


def make_bars(n: int = 60, symbol: str = "MNQ") -> pd.DataFrame:
    ts = pd.date_range("2026-08-20", periods=n, freq="15min", tz="UTC")
    close = 15000.0 + np.arange(n) * 1.0
    return pd.DataFrame({"ts": ts, "open": close, "high": close + 2.0,
                         "low": close - 2.0, "close": close, "volume": 100})


class RecordingSender:
    """A stand-in for `live.dispatcher.send_execution_signal`."""

    def __init__(self, results=None):
        self.calls = []
        self._results = list(results or [])

    def __call__(self, payload, webhook_url=None, timeout_seconds=None):
        self.calls.append({"payload": payload, "url": webhook_url,
                           "timeout": timeout_seconds})
        if self._results:
            return self._results.pop(0)
        return {"ok": True, "http_status": 200, "error": None,
                "response_body": "{}", "url": webhook_url, "payload": payload}


class ExplodingSender:
    """Fails the test if anything tries to send."""

    def __call__(self, *a, **k):                       # pragma: no cover
        raise AssertionError("dry run must not reach the sender")


def build(tmp_path: Path, *, assignments, state, dry_run=True, sender=None,
          strategies=None, **kwargs) -> LiveExecutionDispatcher:
    root = tmp_path / "strategies"
    for strategy_id, spec in (strategies or {}).items():
        write_strategy(root, strategy_id, **spec)
    return LiveExecutionDispatcher(
        config_path=str(write_config(tmp_path, assignments)),
        state_file=str(write_state(tmp_path, state)),
        dry_run=dry_run,
        strategy_root=str(root),
        ml_model_dir=str(tmp_path / "models"),
        env_file=str(tmp_path / "nonexistent.env"),
        sender=sender if sender is not None else ExplodingSender(),
        **kwargs)


@pytest.fixture(autouse=True)
def _isolate_config_cache():
    clear_cache()
    yield
    clear_cache()


# --------------------------------------------------------------------------
# 1. Regime gating in dispatch
# --------------------------------------------------------------------------
def test_a_signal_outside_its_quadrant_is_declined_with_a_reason(tmp_path):
    """
    Incubator-Even trades Q1/Q2. With MGC in Q3 the strategy must produce no
    order AND a decline naming the quadrant - an empty payload list cannot tell
    a basket standing down from a day with no signals.
    """
    d = build(tmp_path,
              assignments={GATED_PORTFOLIO: ["fixture_long"]},
              state={GATED_SYMBOL: {"quadrant": FORBIDDEN_QUADRANT}},
              strategies={"fixture_long": dict(side="long",
                                               symbols=GATED_CERTIFIED)})
    report = d.process_bar_cycle({GATED_SYMBOL: make_bars()})

    assert report["payloads"] == []
    assert report["signals"] == []
    assert report["dispatches"] == []
    declines = [x for x in report["declines"] if x["symbol"] == GATED_SYMBOL]
    assert len(declines) == 1
    reason = declines[0]["reason"]
    assert FORBIDDEN_QUADRANT in reason
    assert "Q1" in reason and "Q2" in reason        # what it DOES trade
    assert declines[0]["strategy_id"] == "fixture_long"


def test_the_same_strategy_trades_once_its_quadrant_is_permitted(tmp_path):
    """The mirror of the case above: only the regime differs."""
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert len(report["payloads"]) == 1
    assert report["payloads"][0]["action"] == "BUY"
    assert report["payloads"][0]["account"] == "Incubator-Odd"
    assert report["declines"] == []


def test_a_missing_regime_reading_stands_the_symbol_down(tmp_path):
    """An unknown environment is not a permitted one."""
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MCL": {"quadrant": PERMITTED_QUADRANT}},   # no MNQ
              strategies={"fixture_long": dict(side="long", symbols=("NQ",))})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert report["payloads"] == []
    assert any("no live regime reading" in x["reason"]
               for x in report["declines"])


def test_a_stale_regime_can_be_refused(tmp_path):
    """`max_age_s` turns a stale quadrant into a decline rather than an
    order priced on a market that has since moved."""
    state = write_state(tmp_path, {"MNQ": {"quadrant": PERMITTED_QUADRANT}})
    blob = json.loads(state.read_text())
    old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    blob["symbols"]["MNQ"]["written_at"] = old
    blob["symbols"]["MNQ"]["bar_ts"] = old
    state.write_text(json.dumps(blob))

    root = tmp_path / "strategies"
    write_strategy(root, "fixture_long", side="long")
    d = LiveExecutionDispatcher(
        config_path=str(write_config(tmp_path,
                                     {"Incubator-Odd": ["fixture_long"]})),
        state_file=str(state), dry_run=True, strategy_root=str(root),
        ml_model_dir=str(tmp_path / "models"),
        env_file=str(tmp_path / "none.env"), sender=ExplodingSender(),
        max_age_s=60)
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert report["payloads"] == []
    assert any("stale" in x["reason"] for x in report["declines"])


# --------------------------------------------------------------------------
# 2. Signal netting in dispatch
# --------------------------------------------------------------------------
def test_opposing_signals_on_mnq_net_out_before_any_payload(tmp_path):
    """
    Two strategies in Incubator-Odd, one long MNQ and one short MNQ. The net is
    zero, so NO CrossTrade payload is formatted - and the plan says the
    position netted flat rather than leaving an unexplained empty list.

    It must NOT become a FLATTEN either: this process does not know whether
    anything is open, and closing on that assumption shuts positions it never
    opened.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long", "fixture_short"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long"),
                          "fixture_short": dict(side="short")})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert len(report["signals"]) == 2
    assert {s["direction"] for s in report["signals"]} == {"long", "short"}

    assert report["payloads"] == []
    assert report["dispatches"] == []

    mnq = [r for r in report["plan"] if r["symbol"] == "MNQ"]
    assert len(mnq) == 1
    assert mnq[0]["net_units"] == 0
    assert "netted flat" in mnq[0]["skipped_reason"]
    assert "FLATTEN" in mnq[0]["skipped_reason"]
    assert len(mnq[0]["contributors"]) == 2


def test_the_execution_summary_counts_what_the_stages_recorded(tmp_path):
    """
    The four-key summary is DERIVED from the stage records, so it cannot
    disagree with them. Three strategies on one contract: one long, one short
    (they net flat), and one whose quadrant is not permitted - so the cycle
    evaluates three, approves two and sends nothing.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long", "fixture_short"],
                           "Incubator-Even": ["fixture_blocked"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long"),
                          "fixture_short": dict(side="short"),
                          "fixture_blocked": dict(side="long")})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    # Incubator-Even does not hold MNQ, so its strategy is never evaluated on
    # it; the two that are net flat.
    assert report["evaluated_signals"] == (
        len(report["signals"])
        + len([x for x in report["declines"] if x.get("strategy_id")])
        + len(report["ml_vetoes"])
        + len(report["errors"]))
    assert report["approved_signals"] == 2
    assert report["orders"] == []
    assert report["processed_at"] == report["finished_at"]


def test_orders_and_payloads_are_the_same_list(tmp_path):
    """Two names for what was sent must not be able to disagree about it."""
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert len(report["orders"]) == 1
    assert report["orders"] is report["payloads"]
    assert report["approved_signals"] == 1
    assert report["evaluated_signals"] == 1


def test_a_declined_signal_is_evaluated_but_not_approved(tmp_path):
    """A cycle that gated everything away must still report that it looked -
    an evaluated count of zero reads as a feed that delivered nothing."""
    d = build(tmp_path,
              assignments={GATED_PORTFOLIO: ["fixture_long"]},
              state={GATED_SYMBOL: {"quadrant": FORBIDDEN_QUADRANT}},
              strategies={"fixture_long": dict(side="long",
                                               symbols=GATED_CERTIFIED)})
    report = d.process_bar_cycle({GATED_SYMBOL: make_bars()})

    assert report["evaluated_signals"] == 1
    assert report["approved_signals"] == 0
    assert report["orders"] == []


def test_an_empty_roster_is_not_counted_as_an_evaluation(tmp_path):
    """The 'no active strategies' note carries no strategy_id: it is a warning
    about the config, not a (strategy, symbol) the loop looked at."""
    d = build(tmp_path,
              assignments={"Incubator-Odd": []},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert report["declines"]                      # it still says so
    assert report["evaluated_signals"] == 0
    assert report["approved_signals"] == 0
    assert report["orders"] == []


def test_reinforcing_signals_stack_into_one_larger_order(tmp_path):
    """Two longs net to +2, and conviction scales the size - which is what
    makes stacking observable in the order rather than only in a field."""
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_a", "fixture_b"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_a": dict(side="long"),
                          "fixture_b": dict(side="long")})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert len(report["payloads"]) == 1
    mnq = [r for r in report["plan"] if r["symbol"] == "MNQ"][0]
    assert mnq["net_units"] == 2
    assert mnq["sizing"]["conviction"] == 2
    # 3 per unit x 2 units = 6, clamped to the config's max_contracts of 5.
    assert mnq["sizing"]["unit_contracts"] == EXPECTED_CONTRACTS
    assert report["payloads"][0]["quantity"] == 5
    assert mnq["sizing"]["budget_breached"] is True


def test_an_ml_veto_changes_the_net_and_says_so(tmp_path, monkeypatch):
    """
    The documented consequence that is not obvious: vetoing one side of an
    opposing pair turns a position that would have netted flat into a live
    order. `ml_vetoes` is what makes it visible rather than surprising.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long", "fixture_short"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long"),
                          "fixture_short": dict(side="short")})

    def veto_shorts(strategy_id, symbol, features):
        return strategy_id != "fixture_short"
    monkeypatch.setattr(d.daemon, "evaluate_ml_gate", veto_shorts)

    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert len(report["ml_vetoes"]) == 1
    assert report["ml_vetoes"][0]["strategy_id"] == "fixture_short"
    assert len(report["payloads"]) == 1
    assert report["payloads"][0]["action"] == "BUY"


def test_a_bar_signalling_both_sides_takes_neither(tmp_path):
    """Mirrors `backtest.engine`'s state machine. Guessing a winner here would
    make the live loop trade a rule the backtest never simulated."""
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_both"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_both": dict(side="both")})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert report["signals"][0]["direction"] == "flat"
    assert "NEITHER" in report["signals"][0]["detail"]["note"]
    assert report["payloads"] == []


def test_an_exit_with_no_tracked_position_is_reported_and_sends_nothing(tmp_path):
    """
    AN EXIT IS NOT A LICENCE TO FLATTEN AN ACCOUNT.

    The loop flattens only a position it has a record of OPENING. With an
    empty `PositionBook` - a fresh process, or a position opened by hand in
    NT8, by a previous run, or by another tool on the same account - the exit
    is reported with its reason and nothing reaches the wire. A blind flatten
    here closes whatever happens to be on the account, silently and
    unrecoverably.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_exit"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_exit": dict(side="flat", exit_on_last=True)})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert len(report["exit_signals"]) == 1
    assert report["exit_signals"][0]["symbol"] == "MNQ"
    assert report["payloads"] == []
    assert not any(str(p.get("action", "")).upper() == "FLATTEN"
                   for p in report["payloads"])


# --------------------------------------------------------------------------
# 3. Sizing in dispatch
# --------------------------------------------------------------------------
def test_atr_sizing_reaches_the_final_payload(tmp_path):
    """
    contracts = floor(risk_budget / (ATR x stop_atr_mult x point_value))
              = floor(250 / (25.0 x 1.5 x 2.0)) = floor(3.33) = 3
    and that integer must be the quantity on the wire, not a default.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT, "atr_14": ATR}},
              strategies={"fixture_long": dict(side="long",
                                               sl_atr_mult=SL_ATR_MULT)})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    payload = report["payloads"][0]
    assert payload["quantity"] == EXPECTED_CONTRACTS

    sizing = [r for r in report["plan"] if r["symbol"] == "MNQ"][0]["sizing"]
    assert sizing["stop_distance_pts"] == pytest.approx(ATR * SL_ATR_MULT)
    assert sizing["risk_per_contract_usd"] == pytest.approx(75.0)
    assert sizing["risk_usd"] == pytest.approx(225.0)
    assert sizing["budget_breached"] is False
    # And the quantity survived the formatter unchanged.
    assert f"qty={EXPECTED_CONTRACTS};" in report["dispatches"][0]["command"]


def test_the_strategys_own_stop_is_used_not_the_sizers_default(tmp_path):
    """
    A position sized on the sizer's 1.0 default when the strategy stops at 1.5
    is sized for a stop the strategy will not use. Halving the ATR must halve
    the stop distance and double the size, which only holds if the STRATEGY's
    multiplier reached the sizer.
    """
    wide = build(tmp_path / "wide",
                 assignments={"Incubator-Odd": ["s"]},
                 state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
                 strategies={"s": dict(side="long", sl_atr_mult=2.5)})
    tight = build(tmp_path / "tight",
                  assignments={"Incubator-Odd": ["s"]},
                  state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
                  strategies={"s": dict(side="long", sl_atr_mult=1.0)})

    w = wide.process_bar_cycle({"MNQ": make_bars()})
    t = tight.process_bar_cycle({"MNQ": make_bars()})

    # floor(250 / (25 x 2.5 x 2)) = 2   vs   floor(250 / (25 x 1 x 2)) = 5
    assert w["payloads"][0]["quantity"] == 2
    assert t["payloads"][0]["quantity"] == 5


def test_when_contributors_disagree_the_widest_stop_wins_and_is_recorded(tmp_path):
    """
    Two strategies netting into one position can declare different stops.
    Sizing on the tightest would over-size relative to the strategy holding the
    wide one and breach its budget; the widest errs toward less risk. Either
    way the disagreement is on the record.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["tight_one", "wide_one"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"tight_one": dict(side="long", sl_atr_mult=1.0),
                          "wide_one": dict(side="long", sl_atr_mult=2.5)})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    record = [r for r in report["plan"] if r["symbol"] == "MNQ"][0]
    source = record["regime"]["stop_atr_mult_source"]
    assert source["value"] == 2.5
    assert source["from"] == "wide_one"
    assert source["in_contention"] == [1.0, 2.5]
    assert record["sizing"]["stop_distance_pts"] == pytest.approx(ATR * 2.5)


def test_payloads_match_the_managers_own_reduction(tmp_path):
    """
    The dispatcher takes `payloads` from the plan. That must be exactly what
    `build_order_payloads` returns for the same input - two reductions of one
    plan are two places for the answer to differ, and nobody compares them.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    manager = PortfolioManager(config_path=d.config_path, config=d.config)
    net = manager.aggregate_signals(report["signals"])
    regimes = {"MNQ": {**report["regime_readings"]["MNQ"],
                       "stop_atr_mult": SL_ATR_MULT}}
    assert report["payloads"] == manager.build_order_payloads(net, regimes)


# --------------------------------------------------------------------------
# 4. Dry-run dispatch
# --------------------------------------------------------------------------
def test_dry_run_formats_both_wire_forms_and_sends_nothing(tmp_path):
    """The sender explodes if touched, so this asserts the absence of a
    network call rather than merely its success."""
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")},
              dry_run=True, sender=ExplodingSender())
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert len(report["dispatches"]) == 1
    attempt = report["dispatches"][0]
    assert attempt["ok"] is True
    assert attempt["dry_run"] is True
    assert attempt["attempts"] == 0
    assert "DRY RUN" in attempt["note"]

    # The plain-text form, field for field.
    assert attempt["command"].startswith("key=")
    assert "command=place;" in attempt["command"]
    assert "account=Incubator-Odd;" in attempt["command"]
    assert "instrument=MNQ;" in attempt["command"]
    assert "action=BUY;" in attempt["command"]
    assert "order_type=MARKET;" in attempt["command"]

    # And the JSON form, lower-cased as the endpoint declares.
    assert attempt["json"]["command"] == "place"
    assert attempt["json"]["action"] == "buy"
    assert attempt["json"]["instrument"] == "MNQ"
    assert attempt["json"]["qty"] == EXPECTED_CONTRACTS


def test_a_dry_run_command_is_logged_redacted(tmp_path):
    """A dry run is the mode most likely to be pasted into a ticket, and the
    key is a bearer credential for a live account."""
    root = tmp_path / "strategies"
    write_strategy(root, "fixture_long", side="long")
    d = LiveExecutionDispatcher(
        config_path=str(write_config(tmp_path,
                                     {"Incubator-Odd": ["fixture_long"]})),
        state_file=str(write_state(tmp_path,
                                   {"MNQ": {"quadrant": PERMITTED_QUADRANT}})),
        dry_run=True, strategy_root=str(root),
        ml_model_dir=str(tmp_path / "models"),
        env_file=str(tmp_path / "none.env"),
        crosstrade_key="SUPER_SECRET_KEY", sender=ExplodingSender())
    report = d.process_bar_cycle({"MNQ": make_bars()})

    command = report["dispatches"][0]["command"]
    assert "SUPER_SECRET_KEY" not in command
    assert "REDACTED" in command


def test_live_mode_sends_the_json_body_to_the_configured_webhook(tmp_path):
    sender = RecordingSender()
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")},
              dry_run=False, sender=sender,
              crosstrade_url="https://crosstrade.invalid/hook",
              crosstrade_key="K")
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert len(sender.calls) == 1
    assert sender.calls[0]["url"] == "https://crosstrade.invalid/hook"
    assert sender.calls[0]["payload"]["qty"] == EXPECTED_CONTRACTS
    assert report["dispatches"][0]["ok"] is True
    assert report["ok"] is True


# --------------------------------------------------------------------------
# 5. Retries, credentials and the integrity checks
# --------------------------------------------------------------------------
def test_a_timeout_is_never_retried(tmp_path):
    """
    A timeout leaves the outcome UNKNOWN - the order may be live. Retrying it
    is how one uncertain send becomes two positions, and this process cannot
    undo the second.
    """
    sender = RecordingSender([
        {"ok": False, "http_status": None,
         "error": "timeout after 2.0s: timed out"}])
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")},
              dry_run=False, sender=sender,
              crosstrade_url="https://crosstrade.invalid/hook",
              max_attempts=3)
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert len(sender.calls) == 1
    attempt = report["dispatches"][0]
    assert attempt["ok"] is False
    assert attempt["attempts"] == 1
    assert "duplicate position" in attempt["retry_note"]


def test_a_server_error_is_never_retried(tmp_path):
    """A 5xx means the request arrived. What the broker did with it is not
    knowable from here."""
    sender = RecordingSender([{"ok": False, "http_status": 503,
                               "error": "HTTP 503: Service Unavailable"}])
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")},
              dry_run=False, sender=sender,
              crosstrade_url="https://crosstrade.invalid/hook", max_attempts=3)
    d.process_bar_cycle({"MNQ": make_bars()})
    assert len(sender.calls) == 1


def test_a_refused_connection_is_retried_because_nothing_was_sent(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr("realtime.live_dispatcher.RETRY_BACKOFF_S", 0.0)
    sender = RecordingSender([
        {"ok": False, "http_status": None,
         "error": "network error: [Errno 111] Connection refused"},
        {"ok": False, "http_status": None,
         "error": "network error: [Errno 111] Connection refused"},
        {"ok": True, "http_status": 200, "error": None},
    ])
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")},
              dry_run=False, sender=sender,
              crosstrade_url="https://crosstrade.invalid/hook", max_attempts=3)
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert len(sender.calls) == 3
    assert report["dispatches"][0]["ok"] is True
    assert report["dispatches"][0]["attempts"] == 3


def test_an_unrecognised_error_defaults_to_not_retrying():
    """A new failure mode added upstream must not silently become a duplicate
    order."""
    assert _is_safe_to_retry({"error": "something nobody has seen before"}) is False
    assert _is_safe_to_retry({"error": None}) is False
    assert _is_safe_to_retry({"http_status": 200, "error": "connection refused"}) is False
    assert _is_safe_to_retry({"error": "network error: Connection refused"}) is True


def test_live_mode_refuses_to_start_without_a_webhook(tmp_path):
    """
    Discovering this at the first order means every gate ran and produced
    nothing, which on a console reads exactly like a quiet market.
    """
    root = tmp_path / "strategies"
    write_strategy(root, "fixture_long", side="long")
    with pytest.raises(LiveDispatchError, match="no CrossTrade webhook URL"):
        LiveExecutionDispatcher(
            config_path=str(write_config(tmp_path,
                                         {"Incubator-Odd": ["fixture_long"]})),
            state_file=str(write_state(tmp_path,
                                       {"MNQ": {"quadrant": PERMITTED_QUADRANT}})),
            dry_run=False, strategy_root=str(root),
            ml_model_dir=str(tmp_path / "models"),
            env_file=str(tmp_path / "none.env"))


def test_edited_strategy_code_is_refused(tmp_path):
    """
    `promoted_sha256` is what makes a promoted file provably the file the
    metrics describe. A live loop that ignored it would trade an edited module
    under a certified strategy's name.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["tampered"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"tampered": dict(side="long", corrupt_hash=True)})

    assert d.strategies == []
    assert len(d.strategy_errors) == 1
    assert "SHA-256" in d.strategy_errors[0]["error"]
    # And it is reported, not silently dropped.
    assert "FAILED TO LOAD" in d.describe()


def test_a_strategy_certified_elsewhere_cannot_trade_this_basket(tmp_path):
    """A strategy certified on soybeans has no evidence about a Nasdaq micro,
    and the only thing that put them together is a line in a config file."""
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["soybeans"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"soybeans": dict(side="long", symbols=("ZS",))})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert report["payloads"] == []
    assert any("certification does not cover" in x["reason"]
               for x in report["declines"])


def test_a_micro_is_covered_by_its_full_size_certification(tmp_path):
    """MNQ and NQ are the same price series at the same tick size - the same
    alias the regime daemon resolves theta_vol through."""
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["nasdaq"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"nasdaq": dict(side="long", symbols=("NQ",))})
    report = d.process_bar_cycle({"MNQ": make_bars()})
    assert len(report["payloads"]) == 1


def test_a_missing_strategy_directory_is_an_error_not_a_skip(tmp_path):
    """A loop quietly running three of the four strategies somebody assigned is
    the failure that gets noticed at the end of the month."""
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long", "does_not_exist"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")})

    assert [h.strategy_id for h in d.strategies] == ["fixture_long"]
    assert len(d.strategy_errors) == 1
    assert d.strategy_errors[0]["strategy_id"] == "does_not_exist"


def test_no_active_strategies_is_reported_rather_than_looking_quiet(tmp_path):
    """`active_strategies` is empty in the shipped config, and that is the
    intended state. It must not read as a market with nothing to do."""
    d = build(tmp_path, assignments={},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert d.strategies == []
    assert any("no active strategies" in x["reason"]
               for x in report["declines"])
    assert "ACTIVE STRATEGIES: none" in d.describe()


def test_credentials_resolve_from_the_env_file_and_are_never_printed(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\n"
                   "export CROSSTRADE_WEBHOOK_URL='https://hook.invalid/x'\n"
                   'CROSSTRADE_API_KEY="ENVKEY"\n')
    assert load_env_file(env)["CROSSTRADE_API_KEY"] == "ENVKEY"

    url, key, source = resolve_credentials(env_file=env)
    assert url == "https://hook.invalid/x"
    assert key == "ENVKEY"
    assert "CROSSTRADE_API_KEY" in source

    # An explicit argument wins over the file.
    url2, key2, _ = resolve_credentials(crosstrade_key="ARG", env_file=env)
    assert key2 == "ARG" and url2 == "https://hook.invalid/x"

    root = tmp_path / "strategies"
    write_strategy(root, "fixture_long", side="long")
    d = LiveExecutionDispatcher(
        config_path=str(write_config(tmp_path,
                                     {"Incubator-Odd": ["fixture_long"]})),
        state_file=str(write_state(tmp_path,
                                   {"MNQ": {"quadrant": PERMITTED_QUADRANT}})),
        dry_run=True, strategy_root=str(root),
        ml_model_dir=str(tmp_path / "models"), env_file=str(env),
        sender=ExplodingSender())
    described = d.describe()
    assert "ENVKEY" not in described
    assert "hook.invalid" in described          # host only, never the full URL
    assert "https://hook.invalid/x" not in described


def test_load_env_file_does_not_touch_the_process_environment(tmp_path,
                                                              monkeypatch):
    """A credential in os.environ is inherited by every subprocess this loop
    ever spawns."""
    monkeypatch.delenv("CROSSTRADE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("CROSSTRADE_API_KEY=LEAKY\n")
    load_env_file(env)
    import os
    assert os.environ.get("CROSSTRADE_API_KEY") is None


# --------------------------------------------------------------------------
# 6. master_live.py — the CLI, the loop and the shutdown
# --------------------------------------------------------------------------
def test_the_cli_defaults_are_the_documented_ones():
    import master_live
    args = master_live.build_parser().parse_args([])
    assert args.interval_sec == 60
    assert args.config.endswith("portfolios.json")
    assert args.state_file.endswith("live_regime_state.json")
    assert args.once is False


def test_a_command_naming_no_mode_is_a_dry_run():
    """
    The safe mode is the one an operator gets by forgetting a flag. A market
    order this process cannot see is not undone by noticing the mistake, so the
    default has to be the recoverable one.
    """
    import master_live
    parse = master_live.build_parser().parse_args
    assert master_live.resolve_dry_run(parse([])) is True
    assert master_live.resolve_dry_run(parse(["--dry-run"])) is True
    assert master_live.resolve_dry_run(parse(["--live"])) is False


def test_asking_for_both_modes_is_refused_rather_than_resolved():
    """Either resolution leaves half the command describing a run that did not
    happen - and the wrong half is the one about whether real orders went."""
    import master_live
    args = master_live.build_parser().parse_args(["--dry-run", "--live"])
    with pytest.raises(ValueError):
        master_live.resolve_dry_run(args)


def test_bars_for_a_micro_are_read_from_its_full_size_parent(monkeypatch):
    """
    The four baskets hold micros; the lake holds only the full-size contracts.
    Without the alias this loop reads nothing and no-ops forever - and it would
    look exactly like a market with no signals.
    """
    import master_live

    captured = {}

    def fake_iter_bars(symbols, tf, start, end):
        captured["symbols"] = list(symbols)
        for s in symbols:
            yield s, make_bars(80, s)

    monkeypatch.setattr("mdlib.lake.iter_bars", fake_iter_bars)
    bars, sources = master_live.load_symbol_bars(
        ["MNQ", "MES", "MCL", "MGC"], "15m", 50)

    # It asked the lake for the PARENTS...
    assert captured["symbols"] == ["CL", "ES", "GC", "NQ"]
    # ...and handed the pipeline the MICROS, with the substitution recorded.
    assert sorted(bars) == ["MCL", "MES", "MGC", "MNQ"]
    assert sources == {"MNQ": "NQ", "MES": "ES", "MCL": "CL", "MGC": "GC"}
    assert len(bars["MNQ"]) == 50


def test_a_symbol_with_no_bars_is_absent_rather_than_empty(monkeypatch):
    import master_live

    def only_nq(symbols, tf, start, end):
        for s in symbols:
            if s == "NQ":
                yield s, make_bars(80, s)

    monkeypatch.setattr("mdlib.lake.iter_bars", only_nq)
    bars, sources = master_live.load_symbol_bars(["MNQ", "MES"], "15m", 50)
    assert sorted(bars) == ["MNQ"]
    assert sources == {"MNQ": "NQ"}


def test_shutdown_is_requested_not_forced(monkeypatch):
    """
    A signal sets a flag; it never interrupts a cycle. An exception raised
    between the send and the print leaves an order on a broker with no line in
    the log saying so.
    """
    import master_live
    flag = master_live.ShutdownFlag()
    assert flag.requested is False

    flag._handle(int(__import__("signal").SIGTERM), None)
    assert flag.requested is True
    assert flag.signal_name == "SIGTERM"

    # A second signal exits at once: the first is "stop when safe".
    with pytest.raises(SystemExit) as exc:
        flag._handle(int(__import__("signal").SIGINT), None)
    assert exc.value.code == 130


def test_shutdown_sleep_returns_immediately_once_requested():
    """The loop must notice a signal within a slice, not after a full
    --interval-sec."""
    import time as _time
    import master_live
    flag = master_live.ShutdownFlag()
    flag.requested = True
    started = _time.perf_counter()
    flag.sleep(30.0)
    assert _time.perf_counter() - started < 1.0


def test_basket_symbols_covers_all_four_accounts(tmp_path):
    import master_live
    d = build(tmp_path, assignments={},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}})
    assert master_live.basket_symbols(d) == ["MCL", "MES", "MGC", "MNQ"]


def test_the_strategy_tag_names_every_contributor(tmp_path):
    """
    A netted position belongs to every strategy that contributed. Tagging it
    with one would attribute the whole position to a strategy that asked for
    part of it, and NT8 fill reconciliation would report the others as having
    placed nothing.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["alpha_one", "beta_two"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"alpha_one": dict(side="long"),
                          "beta_two": dict(side="long")})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    tag = report["dispatches"][0]["json"]["strategy_tag"]
    assert tag == "Incubator-Odd:alpha_one+beta_two"
    # The payload itself is untouched — it must stay the manager's five keys.
    assert set(report["payloads"][0]) == {"account", "action", "symbol",
                                          "orderType", "quantity"}


def test_the_webhook_url_never_survives_into_the_attempt_record(tmp_path):
    """
    A webhook URL is a bearer credential - anyone holding it can place orders
    on the account. The sender echoes it back on every result, and the record
    is printed, logged and serialised.
    """
    sender = RecordingSender()
    secret = "https://crosstrade.invalid/hooks/SECRET-PATH-TOKEN"
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")},
              dry_run=False, sender=sender, crosstrade_url=secret,
              crosstrade_key="K")
    report = d.process_bar_cycle({"MNQ": make_bars()})

    blob = json.dumps(report, default=str)
    assert "SECRET-PATH-TOKEN" not in blob
    assert "K" not in report["dispatches"][0]["command"].split("key=")[1][:20]
    assert report["dispatches"][0]["result"]["url"] == "crosstrade.invalid"


# --------------------------------------------------------------------------
# 9. The exit actuator: muting blocks entries and never an exit
# --------------------------------------------------------------------------
def test_a_muted_strategy_still_flattens_a_position_this_process_opened(tmp_path):
    """
    THE GAP THIS CLOSES. A position is opened in the certified quadrant, the
    market leaves it, and the strategy is MUTED. Before the actuator existed
    the loop returned at the regime decline BEFORE the exit was even computed,
    so the position was stranded: entries correctly blocked, and no way out.

    Muting exists to stop new ENTRIES. An open position must still be
    closeable, so the exit is evaluated ahead of every regime check and becomes
    a real FLATTEN on the account the entry was opened on.
    """
    sender = RecordingSender()
    d = build(tmp_path, dry_run=False, sender=sender,
              assignments={"Incubator-Odd": ["fixture_exit"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_exit": dict(side="flat", exit_on_last=True)})

    # This process opened it, so this process may close it.
    d.positions.record_fill("Incubator-Odd", "MNQ", "long", 2,
                            ["fixture_exit"])
    # ...and the market has left the quadrant the basket trades.
    write_state(tmp_path, {"MNQ": {"quadrant": FORBIDDEN_QUADRANT}})

    report = d.process_bar_cycle({"MNQ": make_bars()})

    # The exit was SEEN despite the mute - that is the fix.
    assert len(report["exit_signals"]) == 1, report["declines"]
    assert report["exit_signals"][0]["exit_permitted"] is True

    emitted = [e for e in report["exit_orders"] if e["emitted"]]
    assert len(emitted) == 1, report["exit_orders"]
    assert emitted[0]["action"] == "FLATTEN"
    assert emitted[0]["symbol"] == "MNQ"
    assert emitted[0]["ok"] is True

    # It reached the wire, on the account the entries use, with no side and no
    # quantity - a flatten that guessed either would open the opposite position.
    assert len(sender.calls) == 1
    body = sender.calls[0]["payload"]
    assert body["command"] == "flatten"
    assert body["account"] == "Incubator-Odd"
    assert body["instrument"] == "MNQ"
    assert "action" not in body and "qty" not in body

    # And the book no longer claims it, so a repeat exit does not re-flatten.
    assert d.positions.direction("Incubator-Odd", "MNQ") == "flat"


def test_entries_stay_blocked_while_muted_even_though_exits_are_not(tmp_path):
    """
    The other half of the same rule. Moving the exit ahead of the regime gate
    must not let an ENTRY past it: the mute is what stops a strategy trading
    the environment nobody certified it in, and an exit path that also opened
    positions would be worse than the stranding it fixed.
    """
    sender = RecordingSender()
    d = build(tmp_path, dry_run=False, sender=sender,
              assignments={GATED_PORTFOLIO: ["fixture_long"]},
              state={GATED_SYMBOL: {"quadrant": FORBIDDEN_QUADRANT}},
              strategies={"fixture_long": dict(side="long",
                                               symbols=GATED_CERTIFIED)})
    d.positions.record_fill(GATED_PORTFOLIO, GATED_SYMBOL, "long", 1,
                            ["fixture_long"])

    report = d.process_bar_cycle({GATED_SYMBOL: make_bars()})

    assert report["signals"] == [], "a muted strategy produced an entry signal"
    assert report["payloads"] == []
    assert report["dispatches"] == []
    assert not sender.calls, "a muted strategy reached the wire"
    declines = [x for x in report["declines"] if x["symbol"] == GATED_SYMBOL]
    assert declines and "Standing down" in declines[0]["reason"]
    # The position it already held is untouched: this strategy signalled no
    # exit, and a mute is not itself an instruction to close.
    assert d.positions.direction(GATED_PORTFOLIO, GATED_SYMBOL) == "long"
