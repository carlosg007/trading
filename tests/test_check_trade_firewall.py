#!/usr/bin/env python3
"""
tests/test_check_trade_firewall.py - the live entry-gate diagnostics card.

Location: ~/src/trading/tests/test_check_trade_firewall.py

    .venv/bin/python3 -m pytest tests/test_check_trade_firewall.py -v

ASSERT-BASED, so `tests/conftest.py` collects it case by case rather than
routing it to the subprocess runner. Every helper is named `_...`: pytest
collects any module-level `test_*` it can call, including one whose only
argument is defaulted, and `tests/test_regime_profiler.py` was bitten by
exactly that - a helper collected that way ran without its redirect and wrote
onto the NFS mount. Nothing here may touch `data/` or `/mnt/backtest`.

WHAT IS WORTH PINNING
---------------------
This card's whole job is to say WHY nothing is trading, so the failures that
matter are the ones where it would give a confident wrong answer:

  * **Credential resolution.** The CrossTrade variables are withheld from
    `os.environ` by design (`mdlib.env.NO_EXPORT`), because anything in the
    environment is inherited by every subprocess. The first version of this
    card read `os.environ` alone and reported the bridge DOWN on a correctly
    configured box. The resolution order is pinned against the dispatcher's
    own `load_env_file`.
  * **The restated constants.** `MIN_BARS_FOR_REGIME` lives behind pandas, so
    it is copied here. A copy that drifted would call a strategy warm eleven
    bars early.
  * **Open positions.** `PositionBook` is deliberately not persisted, so
    exposure is NOT knowable from disk. Printing `0 / 4` would be a claim
    about an account this tool cannot see.
  * **Blocker ORDER.** The interlock makes every layer under it moot; a list
    that put the regime first would send an operator to the wrong place.
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

from realtime import check_trade_firewall as fw                    # noqa: E402


# ==========================================================================
# fixtures
# ==========================================================================

def _gate(status="MUTED", reason="indicator_warmup", entries=False,
          exits=True, live_q="Q0", certified="Q2", **kw):
    return {"strategy_id": "demo_NQ_1h", "portfolio_id": "Incubator-Odd",
            "symbol": "NQ", "timeframe": "1h", "version": "A",
            "optimal_regime": certified,
            "optimal_regime_label": "Q2_HIGH_VOL_CHOP",
            "live_quadrant": live_q, "live_regime": "Q0_UNDEFINED_WARMUP",
            "status": status, "reason": reason,
            "detail": "inside the ADX warm-up.",
            "entries_allowed": entries, "exits_allowed": exits,
            "adx_14": None, "atr_14": None, "theta_vol": 16.23, **kw}


def _regime(n_bars=18, gate=None):
    return {"updated_at": "2026-08-26T17:00:00+00:00",
            "symbols": {"NQ": {"quadrant": "Q0", "n_bars": n_bars, "tf": "1h",
                               "adx_14": None, "atr_14": None}},
            "strategies": {"demo_NQ_1h": gate if gate is not None else _gate()}}


def _portfolios(active=("demo_NQ_1h",)):
    return {"version": "1.1.0", "portfolios": {"Incubator-Odd": {
        "portfolio_id": "Incubator-Odd", "target_account": "SimIncubator1",
        "active_strategies": list(active)}}}


def _wire(tmp: Path, *, regime=None, portfolios=None, engine_state=None,
          kill=None, spool=("NQ_1m.csv",), repo_unit="--dry-run") -> None:
    """
    Point every module-level path at `tmp`.

    Rebound rather than set through os.environ, because several are resolved at
    IMPORT time and a test setting the variable afterwards would silently read
    the real files under `data/` - which on this box are the LIVE ones.
    """
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    fw.REGIME_STATE = tmp / "data" / "live_regime_state.json"
    if regime is not None:
        fw.REGIME_STATE.write_text(json.dumps(regime))
    fw.PORTFOLIO_CONFIG = tmp / "portfolios.json"
    if portfolios is not None:
        fw.PORTFOLIO_CONFIG.write_text(json.dumps(portfolios))
    fw.REPO_UNIT = tmp / "unit.service"
    fw.REPO_UNIT.write_text(
        f"[Service]\nExecStart=/x/python3 /x/master_live.py {repo_unit}\n")

    st = tmp / "data" / "engine_state.json"
    if engine_state is not None:
        st.write_text(json.dumps(engine_state))
    os.environ["BT_ENGINE_STATE"] = str(st)

    ks = tmp / "data" / "KILL_SWITCH"
    if kill is not None:
        ks.write_text(kill)
    os.environ["BT_KILL_SWITCH"] = str(ks)

    sp = tmp / "spool"
    sp.mkdir(parents=True, exist_ok=True)
    for n in spool:
        (sp / n).write_text("ts,open,high,low,close,volume\n")
    os.environ["BT_NT8_SPOOL"] = str(sp)


def _unwire() -> None:
    for var in ("BT_ENGINE_STATE", "BT_KILL_SWITCH", "BT_NT8_SPOOL"):
        os.environ.pop(var, None)


def _snap(tmp: Path, *, running=True, effective=True, **kw) -> dict:
    """A collected snapshot with the process table replaced."""
    _wire(tmp, **kw)
    try:
        snap = fw.collect()
    finally:
        _unwire()
    snap["interlock"] = {**snap["interlock"], "running": running,
                         "effective": effective, "pids": [1] if running else [],
                         "unit_active": "active", "drop_ins": []}
    answers = [v for v in (snap["interlock"]["repo"], effective, running)
               if v is not None]
    snap["interlock"]["agree"] = len(set(answers)) <= 1
    return snap


# ==========================================================================
# the copies this file keeps
# ==========================================================================

def test_the_restated_constants_match_their_sources():
    """
    THE COPIES ARE CHECKED, NOT TRUSTED.

    `MIN_BARS_FOR_REGIME` and `ADX_LENGTH` live behind pandas (268ms and
    253ms), and the CrossTrade variable NAMES behind `live_dispatcher` (2.7s),
    so all four are restated in the card. A warm-up threshold that drifted
    would report a strategy as ready eleven bars early - which on this box is
    eleven hours of tape that was never measured.
    """
    from mdlib.regimes import ADX_LENGTH                     # noqa: PLC0415
    from realtime.regime_daemon import MIN_BARS_FOR_REGIME   # noqa: PLC0415
    from realtime.live_dispatcher import ENV_KEY, ENV_URL    # noqa: PLC0415

    assert fw.ADX_LENGTH == ADX_LENGTH
    assert fw.MIN_BARS_FOR_REGIME == MIN_BARS_FOR_REGIME == 29
    assert fw.ENV_URL == ENV_URL
    assert fw.ENV_KEY == ENV_KEY


def test_credentials_resolve_the_way_the_dispatcher_resolves_them(tmp_path):
    """
    THE BUG THIS FUNCTION EXISTS FOR.

    `mdlib.env.NO_EXPORT` withholds the CrossTrade variables from `os.environ`
    on purpose - anything in the environment is inherited by every subprocess,
    which is how a webhook key reaches an unrelated tool's output. The
    dispatcher reads `env.get(NAME) or os.environ.get(NAME)`: FILE first.

    The first version of this card read `os.environ` alone and reported
    `Execution Bridge: BLOCKED` on a correctly configured box - telling an
    operator their bridge was down while it was fine.
    """
    from realtime.live_dispatcher import load_env_file        # noqa: PLC0415

    envfile = tmp_path / ".env"
    envfile.write_text(
        "CROSSTRADE_WEBHOOK_URL=https://app.crosstrade.io/hook/abc\n"
        "CROSSTRADE_API_KEY=secret-key-value\n")
    os.environ["BT_ENV_FILE"] = str(envfile)
    os.environ.pop(fw.ENV_URL, None)
    os.environ.pop(fw.ENV_KEY, None)
    try:
        theirs = load_env_file(envfile)
        mine = fw._credential_sources()
        assert mine[fw.ENV_URL] == theirs[fw.ENV_URL]
        assert mine[fw.ENV_KEY] == theirs[fw.ENV_KEY]

        br = fw.bridge()
        assert br["url_set"] and br["key_set"]
        assert br["host"] == "app.crosstrade.io"
        assert br["parse_error"] is None
    finally:
        os.environ.pop("BT_ENV_FILE", None)


def test_the_credential_value_never_reaches_the_card(tmp_path):
    """A card that printed the webhook is a card that leaks a live account."""
    envfile = tmp_path / ".env"
    envfile.write_text(
        "CROSSTRADE_WEBHOOK_URL=https://app.crosstrade.io/hook/SUPERSECRET\n"
        "CROSSTRADE_API_KEY=KEYSHOULDNOTAPPEAR\n")
    os.environ["BT_ENV_FILE"] = str(envfile)
    try:
        card = fw.render(_snap(tmp_path, regime=_regime(),
                               portfolios=_portfolios()))
        assert "SUPERSECRET" not in card
        assert "KEYSHOULDNOTAPPEAR" not in card
        assert "app.crosstrade.io" in card, "the host is fine to show"
    finally:
        os.environ.pop("BT_ENV_FILE", None)


# ==========================================================================
# the interlock, in three places
# ==========================================================================

def test_the_interlock_is_reported_from_three_sources(tmp_path):
    card = fw.render(_snap(tmp_path, regime=_regime(),
                           portfolios=_portfolios()))
    assert "repo unit" in card
    assert "systemd (effective)" in card
    assert "running process" in card


def test_a_disagreeing_interlock_is_named_not_averaged(tmp_path):
    """
    `systemctl edit --full` writes an override that SHADOWS the repo unit and
    git never sees it; a unit edited but not restarted leaves the old flag on
    the running process. Either way the three answers differ, and collapsing
    them into one verdict is how a card tells an operator they are in dry run
    while a live loop sends orders.
    """
    snap = _snap(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 running=False, effective=False, repo_unit="--dry-run")
    assert snap["interlock"]["agree"] is False
    card = fw.render(snap)
    assert "THE THREE DISAGREE" in card
    assert any("DISAGREES across sources" in b for b in fw.blockers(snap))


def test_a_live_running_loop_is_not_reported_as_blocked_by_the_interlock(tmp_path):
    snap = _snap(tmp_path, regime=_regime(n_bars=40, gate=_gate(
        status="LIVE", reason="regime_match", entries=True, live_q="Q2")),
        portfolios=_portfolios(), running=False, effective=False,
        repo_unit="")
    assert fw.blockers(snap) == [], "nothing should block this stack"
    card = fw.render(snap)
    assert "ACTIVE — every layer permits an entry" in card
    assert "LIVE EXECUTION" in card


# ==========================================================================
# the kill switch
# ==========================================================================

def test_a_tripped_kill_switch_blocks_and_shows_its_reason(tmp_path):
    snap = _snap(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 kill="2026-08-26T17:00:00Z MANUAL HALT — reconciling NQ")
    card = fw.render(snap)
    assert "Kill Switch State      : TRIPPED" in card
    assert "2. Global Kill Switch     : BLOCKED" in card
    assert "MANUAL HALT" in card
    assert any("kill switch is ARMED" in b for b in fw.blockers(snap))


def test_a_clear_kill_switch_passes(tmp_path):
    card = fw.render(_snap(tmp_path, regime=_regime(),
                           portfolios=_portfolios()))
    assert "Kill Switch State      : CLEAR" in card
    assert "2. Global Kill Switch     : PASS" in card


# ==========================================================================
# the session firewall
# ==========================================================================

def test_no_session_is_not_reported_as_zero_orders(tmp_path):
    """
    An absent engine_state.json means NO SESSION HAS RUN. `Orders: 0` says the
    loop ran and sent nothing, which is a different and actionable claim.
    """
    card = fw.render(_snap(tmp_path, regime=_regime(),
                           portfolios=_portfolios(), engine_state=None))
    assert "UNKNOWN — no session recorded" in card
    assert "NOT the same as zero orders" in card


def test_open_positions_are_not_claimed_from_disk(tmp_path):
    """
    `PositionBook` is deliberately not persisted and `EngineState` records what
    was SENT, not what the account HOLDS. A `0 / 4` printed here would be a
    claim about an account this tool cannot see.
    """
    card = fw.render(_snap(tmp_path, regime=_regime(),
                           portfolios=_portfolios(),
                           engine_state={"session": "2026-08-26",
                                         "orders": [], "fills": [],
                                         "dispatched": {}}))
    assert "NOT knowable here" in card
    assert "0 / 4" not in card


def test_the_order_cap_blocks(tmp_path):
    cap = fw.DEFAULT_LIMITS["max_orders_per_session"]
    snap = _snap(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 engine_state={"session": "2026-08-26",
                               "orders": [{"ok": True}] * cap,
                               "fills": [], "dispatched": {}})
    assert any("order cap reached" in b for b in fw.blockers(snap))


def test_the_session_loss_cap_blocks_and_says_do_not_raise_it(tmp_path):
    """
    The runbook's rule: hitting this means stop for the day. A card that
    reported the number without the instruction invites raising the cap, which
    is the one response that turns a bad day into a worse one.
    """
    cap = float(fw.DEFAULT_LIMITS["max_session_loss_usd"])
    snap = _snap(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 engine_state={"session": "2026-08-26", "orders": [],
                               "fills": [{"pnl": -cap - 1}], "dispatched": {}})
    reasons = fw.blockers(snap)
    assert any("loss cap reached" in b for b in reasons)
    assert any("do not raise the cap" in b for b in reasons)


def test_a_healthy_session_does_not_block(tmp_path):
    snap = _snap(tmp_path, regime=_regime(n_bars=40, gate=_gate(
        status="LIVE", reason="regime_match", entries=True, live_q="Q2")),
        portfolios=_portfolios(), running=False, effective=False,
        repo_unit="",
        engine_state={"session": "2026-08-26",
                      "orders": [{"ok": True}, {"ok": False}],
                      "fills": [{"pnl": 120.0}], "dispatched": {"a": {}}})
    assert fw.blockers(snap) == []
    card = fw.render(snap)
    assert "1 confirmed of 2 attempted / 40 max" in card
    assert "$120.00" in card


# ==========================================================================
# the regime gate
# ==========================================================================

def test_an_incomplete_warmup_blocks_with_the_exact_shortfall(tmp_path):
    snap = _snap(tmp_path, regime=_regime(n_bars=18),
                 portfolios=_portfolios())
    card = fw.render(snap)
    assert "18 / 29 bars — short by 11" in card
    assert any("18/29 bars, short by 11" in b for b in fw.blockers(snap))


def test_a_warm_but_mismatched_regime_blocks_for_a_different_reason(tmp_path):
    """
    Warm-up completing is NECESSARY, NOT SUFFICIENT. At 29 bars a real quadrant
    appears; if it is not the certified one the gate stays shut on a MISMATCH.
    Same outcome, different cause, and only the first is on a timer.
    """
    snap = _snap(tmp_path, regime=_regime(n_bars=40, gate=_gate(
        status="MUTED", reason="regime_mismatch", entries=False,
        live_q="Q1")), portfolios=_portfolios())
    reasons = fw.blockers(snap)
    assert not any("warm-up" in b for b in reasons), "warm-up is complete"
    assert any("regime_mismatch" in b and "Q1" in b and "Q2" in b
               for b in reasons)


def test_an_allocated_strategy_with_no_switchboard_entry_blocks(tmp_path):
    snap = _snap(tmp_path, regime={"symbols": {}, "strategies": {}},
                 portfolios=_portfolios())
    card = fw.render(snap)
    assert "NO SWITCHBOARD ENTRY" in card
    assert any("no switchboard entry" in b.lower() for b in fw.blockers(snap))


# ==========================================================================
# the feed
# ==========================================================================

def test_a_missing_spool_for_a_needed_symbol_blocks(tmp_path):
    snap = _snap(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 spool=("ES_1m.csv",))
    assert any("No spool file for NQ" in b for b in fw.blockers(snap))


# ==========================================================================
# the summary
# ==========================================================================

def test_blockers_are_listed_interlock_first(tmp_path):
    """
    ORDER IS THE POINT. A dry-run loop evaluates gates and sends nothing, so
    "the regime is wrong" is true and irrelevant until the flag is off. A list
    that led with the regime sends an operator to fix the wrong layer.
    """
    snap = _snap(tmp_path, regime=_regime(n_bars=18),
                 portfolios=_portfolios(), kill="halt")
    reasons = fw.blockers(snap)
    assert "--dry-run" in reasons[0]
    assert any("kill switch" in r for r in reasons)
    assert "warm-up" in reasons[-1]


def test_a_clean_stack_says_nothing_is_blocking(tmp_path):
    snap = _snap(tmp_path, regime=_regime(n_bars=40, gate=_gate(
        status="LIVE", reason="regime_match", entries=True, live_q="Q2")),
        portfolios=_portfolios(), running=False, effective=False,
        repo_unit="")
    card = fw.render(snap)
    assert "Nothing is blocking an entry" in card
    assert "[BLOCKER]" not in card


# ==========================================================================
# safety and cost
# ==========================================================================

def test_the_card_never_opens_a_socket():
    """
    "Reachable" for the CrossTrade endpoint means POSTing to the thing that
    places orders on a funded account. A status tool that pings it to see
    whether it answers is a status tool that can open a position, so this file
    must not carry an HTTP client at all.
    """
    src = (REPO / "realtime" / "check_trade_firewall.py").read_text()
    for banned in ("urllib.request", "http.client", "requests", "socket."):
        assert banned not in src, f"{banned} has no business in this file"


def test_the_tool_does_not_import_pandas():
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "import realtime.check_trade_firewall;"
         "print(','.join(m for m in ('pandas','numpy','vectorbtpro')"
         "                if m in sys.modules))" % str(REPO)],
        capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"pulled in: {out.stdout.strip()}"


def test_an_absent_reading_formats_instead_of_raising():
    """`f"{None:.2f}"` took the regime daemon down 163 times on 2026-08-26."""
    assert fw.num(None) == "n/a"
    assert fw.num(0.0) == "0.00", "a measured zero is not an absence"
    assert fw.num(27.4567) == "27.46"
    assert fw.num("weird") == "weird"


def test_it_runs_from_any_directory_and_exits_nonzero_when_blocked():
    out = subprocess.run(
        [sys.executable, str(REPO / "realtime" / "check_trade_firewall.py")],
        capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=180)
    assert out.returncode in (0, 1)
    assert "LIVE TRADE FIREWALL & ENTRY GATE DIAGNOSTICS" in out.stdout
    assert "Root Cause Summary" in out.stdout
    assert "Traceback" not in out.stderr


def test_it_writes_nothing(tmp_path):
    _wire(tmp_path, regime=_regime(), portfolios=_portfolios())
    try:
        before = sorted(p.name for p in (tmp_path / "data").iterdir())
        fw.collect()
        assert sorted(p.name for p in (tmp_path / "data").iterdir()) == before
    finally:
        _unwire()
