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

import datetime as dt
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
from realtime.crosstrade_formatter import ACTIVE_CONTRACT           # noqa: E402
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

# The NT8 account `Incubator-Odd` executes on, READ from the routing table
# rather than retyped. This suite asserts that the wire carries whatever
# `target_account` says; `tests/test_portfolio_config.py` is where that field
# is checked against the specification. Retyped here, this suite would fail a
# second time for the same rename, and one of the two failures would be noise.
ODD_EXECUTION_ACCOUNT = json.loads(REAL_CONFIG.read_text())[
    "portfolios"]["Incubator-Odd"]["target_account"]

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


#: The same fixture, plus the `ml_features` hook the indicator telemetry
#: reads. A SEPARATE source rather than a flag on the one above, because
#: the default fixture declaring NO feature matrix is itself a case under
#: test: a module with no indicators to publish must record nothing AND
#: no error.
STRATEGY_SRC_WITH_FEATURES = STRATEGY_SRC + '''

def ml_features(bars, **_ignored):
    n = len(bars)
    return pd.DataFrame({
        "fast_rsi": np.linspace(10.0, 48.25, n),
        "macd_hist": np.linspace(-1.0, -0.5, n),
        "flag": np.zeros(n, dtype=bool),
    })
'''

#: A feature matrix that will not build. Telemetry must record the failure
#: and let the cycle finish - the strict `_ml_features` call in the ML gate
#: path is the one that has to raise, and only when a model is judging.
STRATEGY_SRC_BROKEN_FEATURES = STRATEGY_SRC + '''

def ml_features(bars, **_ignored):
    raise RuntimeError("this matrix will not build")
'''


def write_strategy(root: Path, strategy_id: str, *, side="long",
                   symbols=("NQ",), sl_atr_mult=SL_ATR_MULT, tp_atr_mult=2.0,
                   exit_on_last=False, corrupt_hash=False,
                   src: str | None = None, timeframe: str = "15m") -> Path:
    """A promoted-strategy directory: strat.py, meta.json, honest SHA-256."""
    directory = root / strategy_id
    directory.mkdir(parents=True, exist_ok=True)
    module = directory / "strat.py"
    module.write_text(src or STRATEGY_SRC)
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    if corrupt_hash:
        digest = "0" * 64
    (directory / "meta.json").write_text(json.dumps({
        "name": strategy_id,
        "version": "A",
        "symbols": list(symbols),
        "timeframe": timeframe,
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


def make_bars(n: int = 60, symbol: str = "MNQ",
              freq: str = "15min") -> pd.DataFrame:
    ts = pd.date_range("2026-08-20", periods=n, freq=freq, tz="UTC")
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
    # The ACCOUNT on the wire is the portfolio's `target_account`, which is
    # NinjaTrader's name for it and not the portfolio id: NT8 prefixes a
    # simulation account with `Sim`. An order addressed to the id would be
    # rejected by a broker that has no such account.
    assert report["payloads"][0]["account"] == ODD_EXECUTION_ACCOUNT
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
    assert f"account={ODD_EXECUTION_ACCOUNT};" in attempt["command"]
    # THE CONTRACT MONTH, not the bare root. NinjaTrader answers a bare root
    # with "Instrument 'MNQ' not found", and CrossTrade returns 200 before it
    # ever gets there - so this was refused DOWNSTREAM of anything this
    # repository logs, on orders whose dispatch record said OK. Asserted
    # against `ACTIVE_CONTRACT` rather than a literal so the table is the one
    # place a roll is recorded.
    assert f"instrument={ACTIVE_CONTRACT['MNQ']};" in attempt["command"]
    assert "action=BUY;" in attempt["command"]
    assert "order_type=MARKET;" in attempt["command"]

    # And the JSON form, lower-cased as the endpoint declares.
    assert attempt["json"]["command"] == "place"
    assert attempt["json"]["action"] == "buy"
    assert attempt["json"]["instrument"] == ACTIVE_CONTRACT["MNQ"]
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


def test_live_mode_sends_the_text_command_to_the_configured_webhook(tmp_path):
    """The SEMICOLON TEXT command goes on the wire, not the JSON object.

    This test asserted the JSON body until 2026-08-31, and it passed the whole
    time the live loop could not place a single order: the configured endpoint
    is CrossTrade's `/v1/send/` WEBHOOK, which parses the text form, and 125
    consecutive live orders were refused HTTP 400 while this stayed green.
    A test can only pin the form the code sends; it took the endpoint to say
    which form was right, via `realtime/send_test_probe.py`.
    """
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
    payload = sender.calls[0]["payload"]
    assert isinstance(payload, str), (
        f"the webhook takes the text command; got {type(payload).__name__}")
    assert "command=place" in payload
    assert f"qty={EXPECTED_CONTRACTS}" in payload
    assert "key=K" in payload, "the wire form carries the key"
    assert report["dispatches"][0]["ok"] is True
    assert report["ok"] is True


def test_the_key_never_survives_into_the_dispatch_record(tmp_path):
    """The wire form carries `key=`, and the sender echoes the payload back.

    The record is printed, logged and serialised onward, so both the echoed
    payload and any response body that quotes the request have to be scrubbed
    - redacting only at the log line leaves the credential in every other
    consumer of the report.
    """
    sender = RecordingSender([
        {"ok": False, "http_status": 400, "error": "HTTP 400: Bad Request",
         "response_body": "rejected: key=K; command=place;"}])
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")},
              dry_run=False, sender=sender,
              crosstrade_url="https://crosstrade.invalid/hook",
              crosstrade_key="SEKRET")
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert sender.calls, "nothing was sent"
    assert "key=SEKRET" in sender.calls[0]["payload"], (
        "the real key must reach the wire")
    blob = str(report["dispatches"])
    assert "SEKRET" not in blob, "the key survived into the dispatch record"


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


def test_a_micro_basket_is_gated_on_its_parents_published_regime(tmp_path):
    """
    THE STATE FILE IS KEYED ON THE PARENT, AND THE BASKET ON THE MICRO.

    Every other case here publishes the regime under the basket's own symbol,
    so none of them exercised the seam that actually runs: the daemon
    publishes NQ, because that is where the history and the pinned theta_vol
    anchor live, and the basket holds MNQ. Before the reader resolved the
    alias this declined with "no live regime reading" on a state file that
    held the answer - which on a console is the same line a dead daemon
    produces.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["nasdaq"]},
              state={"NQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"nasdaq": dict(side="long", symbols=("NQ",))})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert report["regime_missing"] == {}
    reading = report["regime_readings"]["MNQ"]
    assert reading["resolved_symbol"] == "NQ"
    assert reading["symbol_aliased"] is True
    assert reading["quadrant"] == PERMITTED_QUADRANT
    assert len(report["payloads"]) == 1
    assert report["payloads"][0]["symbol"] == "MNQ"


def test_a_stand_down_names_the_contract_the_quadrant_was_measured_on(tmp_path):
    """
    The order is for the micro and the reading is the parent's. An operator
    reading a stand-down has to be able to see both without going to the
    alias table themselves.
    """
    # On the EVEN track, for the reason at the top of this file: it is the one
    # that declares two quadrants, so FORBIDDEN_QUADRANT actually stands down.
    d = build(tmp_path,
              assignments={GATED_PORTFOLIO: ["gold"]},
              state={"GC": {"quadrant": FORBIDDEN_QUADRANT}},
              strategies={"gold": dict(side="long", symbols=GATED_CERTIFIED)})
    report = d.process_bar_cycle({GATED_SYMBOL: make_bars()})

    assert report["payloads"] == []
    reasons = [x["reason"] for x in report["declines"]
               if x["symbol"] == GATED_SYMBOL]
    assert len(reasons) == 1
    assert f"{GATED_SYMBOL} (regime read from GC)" in reasons[0]
    assert FORBIDDEN_QUADRANT in reasons[0]


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
    The four baskets hold micros; the tape is the full-size contract's.
    Without the alias this loop reads nothing and no-ops forever - and it would
    look exactly like a market with no signals.

    The feed is passed EXPLICITLY. `load_symbol_bars` now delegates to
    `realtime/feed.py`, whose default resolves to the live vendor when one is
    configured - so a case that means "read the lake" has to say so, or it
    silently tests the network instead.
    """
    import master_live
    from realtime.feed import LakeFeed

    captured = {}

    def fake_iter_bars(symbols, tf, start, end):
        captured["symbols"] = list(symbols)
        for s in symbols:
            yield s, make_bars(80, s)

    monkeypatch.setattr("mdlib.lake.iter_bars", fake_iter_bars)
    bars, sources = master_live.load_symbol_bars(
        ["MNQ", "MES", "MCL", "MGC"], "15m", 50, feed=LakeFeed())

    # It asked the lake for the PARENTS...
    assert captured["symbols"] == ["CL", "ES", "GC", "NQ"]
    # ...and handed the pipeline the MICROS, with the substitution recorded.
    assert sorted(bars) == ["MCL", "MES", "MGC", "MNQ"]
    assert sources == {"MNQ": "NQ", "MES": "ES", "MCL": "CL", "MGC": "GC"}
    assert len(bars["MNQ"]) == 50


def test_a_symbol_with_no_bars_is_absent_rather_than_empty(monkeypatch):
    import master_live
    from realtime.feed import LakeFeed

    def only_nq(symbols, tf, start, end):
        for s in symbols:
            if s == "NQ":
                yield s, make_bars(80, s)

    monkeypatch.setattr("mdlib.lake.iter_bars", only_nq)
    bars, sources = master_live.load_symbol_bars(["MNQ", "MES"], "15m", 50,
                                                 feed=LakeFeed())
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
    assert body["account"] == ODD_EXECUTION_ACCOUNT
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


# ==========================================================================
# Indicator telemetry
#
# `describe_cycle` logs the readings each strategy was looking at, so a bar
# that traded nothing says WHY in indicator terms and not only in gate terms.
# The three things worth pinning are that it survives a decline, that it
# cannot break a cycle, and that it publishes the MODULE's own columns.
# ==========================================================================

def test_indicators_are_recorded_even_when_the_strategy_stands_down(tmp_path):
    """
    THE CASE THE FEATURE EXISTS FOR. A signal that fires is already visible in
    the payload; a bar that traded nothing is the one nobody can explain later,
    and it is exactly the bar the regime gate returns early on.
    """
    d = build(tmp_path,
              assignments={GATED_PORTFOLIO: ["fixture_long"]},
              state={GATED_SYMBOL: {"quadrant": FORBIDDEN_QUADRANT}},
              strategies={"fixture_long": dict(
                  side="long", symbols=GATED_CERTIFIED,
                  src=STRATEGY_SRC_WITH_FEATURES)})
    report = d.process_bar_cycle({GATED_SYMBOL: make_bars()})

    assert report["payloads"] == [], "the quadrant is forbidden; nothing trades"
    assert report["declines"], "and it declined"
    readings = [i for i in report["indicators"] if i["symbol"] == GATED_SYMBOL]
    assert len(readings) == 1, "the stood-down bar still reports its readings"
    assert readings[0]["strategy_id"] == "fixture_long"
    assert set(readings[0]["values"]) == {"fast_rsi", "macd_hist", "flag"}
    assert readings[0]["values"]["fast_rsi"] == 48.25
    assert report["indicator_errors"] == []


def test_the_columns_are_the_modules_own_not_a_fixed_list(tmp_path):
    """
    There is no universal indicator set. `double_rsi_macd_scalp` declares
    rsi_fast/macd_hist; `t3_braid_scalp` - the only strategy actually
    allocated on this box - declares t3_slope/braid_hist/stiffness. A hardcoded
    "fast RSI and MACD" line would print nothing for the one that is running.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(
                  side="long", src=STRATEGY_SRC_WITH_FEATURES)})
    report = d.process_bar_cycle({"MNQ": make_bars()})
    values = report["indicators"][0]["values"]
    # Whatever the module declared, verbatim - including a non-float column.
    assert list(values) == ["fast_rsi", "macd_hist", "flag"]


def test_a_module_with_no_feature_matrix_records_nothing_and_no_error(tmp_path):
    """
    Silence is correct here and is NOT a failure: a module that declares no
    `ml_features` has no indicators to publish. Recording an error would put a
    permanent complaint in the log of every cycle of a working strategy.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")})
    report = d.process_bar_cycle({"MNQ": make_bars()})
    assert report["indicators"] == []
    assert report["indicator_errors"] == []
    assert report["payloads"], "and the cycle still traded normally"


def test_broken_indicators_do_not_break_a_cycle_they_are_only_watching(tmp_path):
    """
    TELEMETRY MUST NEVER COST A DECISION.

    The gated portfolio declines on the regime BEFORE the ML path, so the only
    thing touching `ml_features` here is the telemetry call. A matrix that
    will not build is recorded and the cycle finishes.
    """
    d = build(tmp_path,
              assignments={GATED_PORTFOLIO: ["fixture_long"]},
              state={GATED_SYMBOL: {"quadrant": FORBIDDEN_QUADRANT}},
              strategies={"fixture_long": dict(
                  side="long", symbols=GATED_CERTIFIED,
                  src=STRATEGY_SRC_BROKEN_FEATURES)})
    report = d.process_bar_cycle({GATED_SYMBOL: make_bars()})

    assert report["indicators"] == []
    assert len(report["indicator_errors"]) == 1
    assert "will not build" in report["indicator_errors"][0]["error"]
    # The cycle completed and reached its normal verdict. A telemetry failure
    # is NOT an evaluation error.
    assert report["errors"] == []
    assert report["declines"], "the regime decline still happened"
    assert d.describe_cycle(report), "and the summary still renders"


def test_a_broken_matrix_still_stops_the_ML_GATE_and_that_is_not_telemetry(
        tmp_path):
    """
    THE OTHER HALF, PINNED SO THE TWO ARE NEVER CONFUSED.

    On a bar that actually signals, `_evaluate` calls `_ml_features` STRICTLY
    for the ML gate, and that call must keep raising: a model asked to judge a
    signal on features that will not build has to refuse rather than guess.
    The trade is stopped there, in behaviour that predates the telemetry, and
    it surfaces as an evaluation ERROR rather than as an indicator one.

    Written down because the obvious reading of "broken indicators stopped my
    trade" is that the telemetry did it. It did not; both records appear, and
    they mean different things.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(
                  side="long", src=STRATEGY_SRC_BROKEN_FEATURES)})
    report = d.process_bar_cycle({"MNQ": make_bars()})

    assert report["payloads"] == [], "the ML gate could not be evaluated"
    assert len(report["errors"]) == 1
    assert "will not build" in report["errors"][0]["error"]
    # ...and the telemetry recorded its own, separately.
    assert len(report["indicator_errors"]) == 1


def test_the_cycle_summary_shows_the_readings_under_the_verdict(tmp_path):
    """
    `IND` lines render after the decision, so the verdict reads first and the
    numbers behind it read second. `realtime/check_live_signals.py` parses
    these back and separates them for the same reason.
    """
    d = build(tmp_path,
              assignments={GATED_PORTFOLIO: ["fixture_long"]},
              state={GATED_SYMBOL: {"quadrant": FORBIDDEN_QUADRANT}},
              strategies={"fixture_long": dict(
                  side="long", symbols=GATED_CERTIFIED,
                  src=STRATEGY_SRC_WITH_FEATURES)})
    text = d.describe_cycle(d.process_bar_cycle({GATED_SYMBOL: make_bars()}))
    lines = text.splitlines()

    hold = next(i for i, ln in enumerate(lines) if "HOLD" in ln)
    ind = next(i for i, ln in enumerate(lines) if "IND " in ln)
    assert ind > hold, "the decision is printed before the readings"
    assert "fast_rsi=48.25" in lines[ind]
    assert "macd_hist=" in lines[ind]


def test_a_reading_that_cannot_be_formatted_does_not_raise():
    """
    `f"{None:.2f}"` raises TypeError, and this runs inside a console line.
    That exact expression took `realtime/regime_daemon.py` down 163 times on
    2026-08-26 - publishing correctly, then dying while formatting its own log.
    Every branch of the formatter ends in a string.
    """
    from realtime.live_dispatcher import _fmt_reading         # noqa: PLC0415

    assert _fmt_reading(None) == "n/a"
    assert _fmt_reading(0.0) == "0", "a measured zero is not an absence"
    assert _fmt_reading(float("nan")) == "nan"
    assert _fmt_reading(float("inf")) == "inf"
    assert _fmt_reading(True) == "T"
    assert _fmt_reading(48.25) == "48.25"
    assert _fmt_reading(6.0) == "6", "counts lose the trailing .00"
    assert _fmt_reading(0.00097) == "0.00097", "small values keep their digits"
    assert _fmt_reading(15234.567) == "15234.57", "no scientific on price scale"
    assert _fmt_reading("weird") == "weird"


# ==========================================================================
# Multi-timeframe dispatch
#
# THE BUG THESE EXIST FOR, stated once: until 2026-08-27 `master_live.py`
# loaded ONE timeframe and handed those bars to every strategy, and
# `trades_symbol` checked only the symbol. A strategy swept, plateau-selected
# and Gate-R certified on 3m bars was therefore evaluated on 1h bars —
# producing real signals, against a certification describing a different tape,
# with every log line reading correctly. It was caught during pre-live arming
# and before any order went out.
# ==========================================================================

def test_the_handle_records_the_timeframe_it_was_certified_on(tmp_path):
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long", timeframe="3m")})
    assert d.strategies[0].certified_timeframe == "3m"


def test_a_handle_falls_back_to_the_id_suffix(tmp_path):
    """
    A meta.json written before `timeframe` existed carries no claim, and
    stranding it would regress every strategy promoted before the guard. The
    id is split with `pipeline.split_strategy_id`, not by counting
    underscores — `ma_anchoring_spread_20260820` would otherwise split to a
    symbol of `spread` at a timeframe of `20260820`.
    """
    from realtime.live_dispatcher import StrategyHandle       # noqa: PLC0415
    resolve = StrategyHandle._resolve_timeframe
    assert resolve("demo_NQ_1h", {}) == "1h"
    assert resolve("demo_NQ_1h", {"timeframe": "3m"}) == "3m", "meta wins"
    assert resolve("ma_anchoring_spread_20260820", {}) is None
    assert resolve("bare_name", {}) is None


def test_strategy_handle_timeframe_mismatch(tmp_path):
    """
    SPEC TEST 1. Hourly bars into a 3m strategy must raise, not decline.

    It raises rather than declining because `process_bar_cycle` has already
    filtered the roster to this bucket — a mismatched pair reaching `_evaluate`
    is a bug in the dispatch above it, not a configuration a strategy can be
    quietly stood down for. A decline would be one more silent HOLD in a stack
    whose entire failure mode is a silent HOLD that reads correctly.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long", timeframe="3m")})
    handle = d.strategies[0]

    with pytest.raises(ValueError, match="TIMEFRAME_MISMATCH"):
        d._evaluate(handle, "MNQ", make_bars(freq="1h"),
                    {"MNQ": {"quadrant": PERMITTED_QUADRANT}}, {},
                    [PERMITTED_QUADRANT],
                    {"declines": [], "signals": [], "exit_signals": [],
                     "ml_vetoes": [], "errors": [], "indicators": [],
                     "indicator_errors": []},
                    bar_timeframe="1h")

    # ...and the message names BOTH widths, because "mismatch" alone sends an
    # operator to look up which one the strategy wanted.
    try:
        d._evaluate(handle, "MNQ", make_bars(freq="1h"),
                    {"MNQ": {"quadrant": PERMITTED_QUADRANT}}, {},
                    [PERMITTED_QUADRANT],
                    {"declines": [], "signals": [], "exit_signals": [],
                     "ml_vetoes": [], "errors": [], "indicators": [],
                     "indicator_errors": []},
                    bar_timeframe="1h")
    except ValueError as e:
        assert "expects 3m" in str(e) and "received 1h" in str(e)


def test_a_bucket_evaluates_only_the_strategies_certified_on_it(tmp_path):
    """
    A strategy belonging to another bucket is SKIPPED, not declined and not
    errored. Under multi-timeframe dispatch most of the roster legitimately
    belongs elsewhere every cycle; recording each as a refusal would bury the
    ones that were actually refused.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fast", "slow"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fast": dict(side="long", timeframe="3m"),
                          "slow": dict(side="long", timeframe="1h")})

    report = d.process_bar_cycle({"MNQ": make_bars(freq="3min")},
                                 timeframe="3m")
    assert report["timeframe"] == "3m"
    skipped = {s["strategy_id"] for s in report["timeframe_skipped"]}
    assert skipped == {"slow"}, "the 1h strategy sits this bucket out"
    assert report["declines"] == [], "and it is not a decline"
    assert report["errors"] == [], "nor an error"
    assert report["payloads"], "the 3m strategy still traded"


def test_multi_timeframe_resampling_dispatch(tmp_path):
    """
    SPEC TEST 2. A 3m strategy and a 1h strategy, both evaluated in one cycle,
    each on its OWN bar width — which is the whole point of the change.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fast", "slow"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fast": dict(side="long", timeframe="3m"),
                          "slow": dict(side="long", timeframe="1h")})

    seen = {}
    for tf, freq in (("3m", "3min"), ("1h", "1h")):
        report = d.process_bar_cycle({"MNQ": make_bars(freq=freq)},
                                     timeframe=tf)
        traded = {s["strategy_id"] for s in report["signals"]}
        seen[tf] = traded
        assert report["errors"] == [], f"{tf} bucket errored: {report['errors']}"

    assert seen["3m"] == {"fast"}, seen
    assert seen["1h"] == {"slow"}, seen
    # Neither ever saw the other's bars: that is the defect, closed.
    assert "slow" not in seen["3m"] and "fast" not in seen["1h"]


def test_the_timeframe_is_inferred_when_a_caller_does_not_declare_it(tmp_path):
    """
    Backward compatibility that is NOT an exemption. A caller predating the
    parameter is still guarded — the width is measured off the frames — so the
    hole cannot reopen through an un-updated call site.
    """
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fast", "slow"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fast": dict(side="long", timeframe="3m"),
                          "slow": dict(side="long", timeframe="1h")})
    report = d.process_bar_cycle({"MNQ": make_bars(freq="3min")})
    assert report["timeframe"] == "3m", "measured, not assumed"
    assert {s["strategy_id"] for s in report["timeframe_skipped"]} == {"slow"}


def test_a_strategy_declaring_no_timeframe_is_still_evaluated(tmp_path):
    """
    A meta.json written before the key existed makes no claim to contradict.
    Refusing it would strand every strategy promoted before this guard.
    """
    from realtime.live_dispatcher import StrategyHandle       # noqa: PLC0415
    d = build(tmp_path,
              assignments={"Incubator-Odd": ["fixture_long"]},
              state={"MNQ": {"quadrant": PERMITTED_QUADRANT}},
              strategies={"fixture_long": dict(side="long")})
    d.strategies[0].certified_timeframe = None                # pre-guard meta
    report = d.process_bar_cycle({"MNQ": make_bars(freq="1h")}, timeframe="1h")
    assert report["timeframe_skipped"] == []
    assert report["payloads"], "it trades, as it did before the guard"


# --------------------------------------------------------------------------
# The contract roll table
# --------------------------------------------------------------------------
def test_a_bare_root_resolves_to_a_contract_month():
    """NinjaTrader answers a bare root with "Instrument 'MNQ' not found".

    CrossTrade returns 200 before NT8 ever sees it, so this failed DOWNSTREAM
    of everything this repository logs - the dispatch record said OK on orders
    that never reached an account.
    """
    from realtime.crosstrade_formatter import resolve_contract
    for root in ("MNQ", "MES", "MGC", "6E", "6J"):
        out = resolve_contract(root)
        assert out.startswith(root + " "), f"{root} -> {out}"
        assert out != root, f"{root} was not resolved"
    assert resolve_contract("mnq") == resolve_contract("MNQ"), "case matters"


def test_resolution_is_idempotent_for_an_explicit_contract():
    """A caller who names the month gets exactly what they typed.

    `send_test_probe.py --symbol "MNQ SEP26"` must not become
    "MNQ SEP26 SEP26", and a continuous or dash-formatted contract is the
    caller's choice to make.
    """
    from realtime.crosstrade_formatter import resolve_contract
    for explicit in ("MNQ SEP26", "MES 09-26", "MNQ 1!", "MGC DEC26"):
        assert resolve_contract(explicit) == explicit
    assert resolve_contract(resolve_contract("MNQ")) == resolve_contract("MNQ")


def test_an_unknown_root_is_refused_rather_than_guessed():
    from realtime.crosstrade_formatter import (CrossTradeFormatError,
                                               resolve_contract)
    with pytest.raises(CrossTradeFormatError, match="no active contract"):
        resolve_contract("ZZZ")


def test_the_table_refuses_once_it_is_out_of_date():
    """THE POINT OF THE EXPIRY DATE.

    `_clean_instrument` declined to map a root because "guessed here it would
    send an order to whichever contract month a stale rule named". The table
    answers that by expiring: past the date it RAISES, so a stale table costs
    a refused order - visible in the first cycle - rather than a FILLED order
    in an expired contract, which nothing downstream can detect.
    """
    from realtime.crosstrade_formatter import (CONTRACT_TABLE_VALID_UNTIL,
                                               CrossTradeFormatError,
                                               resolve_contract)
    assert resolve_contract("MNQ", on=CONTRACT_TABLE_VALID_UNTIL)
    day_after = (dt.date.fromisoformat(CONTRACT_TABLE_VALID_UNTIL)
                 + dt.timedelta(days=1)).isoformat()
    with pytest.raises(CrossTradeFormatError, match="expired"):
        resolve_contract("MNQ", on=day_after)


def test_the_table_covers_every_micro_the_router_can_reach():
    """A symbol a basket can hold but the table cannot map is an order that
    is refused at the formatter - so the two lists have to agree."""
    from realtime.contract_alias import MICRO_TO_PARENT
    from realtime.crosstrade_formatter import ACTIVE_CONTRACT
    missing = sorted(set(MICRO_TO_PARENT) - set(ACTIVE_CONTRACT))
    assert not missing, f"no contract month recorded for {missing}"
    parents = sorted(set(MICRO_TO_PARENT.values()) - set(ACTIVE_CONTRACT))
    assert not parents, f"no contract month recorded for {parents}"
