#!/usr/bin/env python3
"""
tests/test_check_live_signals.py - the live signal / gate status card.

Location: ~/src/trading/tests/test_check_live_signals.py

    .venv/bin/python3 -m pytest tests/test_check_live_signals.py -v

ASSERT-BASED, and pytest-shaped on purpose. `tests/conftest.py` routes suites
carrying the `check()` collector to a subprocess runner because pytest cannot
see their results; this one fails through `assert`, so both runners agree.

Every helper is named `_...`. `tests/test_regime_profiler.py` was bitten by the
other spelling: pytest collects any module-level `test_*` it can call,
INCLUDING one whose only argument is defaulted, and a helper collected that way
ran without its redirect and wrote onto the NFS mount. Nothing here may touch
`/mnt/backtest` or `data/`, so nothing here is named so it could be called by
accident.

WHAT IS WORTH PINNING
---------------------
Not the layout. The three things that would make this card LIE:

  * **claiming zero when nothing was counted.** An absent
    `data/engine_state.json` means no session has run. Rendering that as
    `Orders Dispatched: 0` is a different statement - it says the loop ran and
    sent nothing - and it is the one an operator would act on.
  * **an allocated strategy with no switchboard entry reading as permitted.**
    The routing table allocating an id the daemon never published a status for
    is a real and silent fault, and the safe rendering is the one that says so
    rather than defaulting `entries_allowed` to anything.
  * **`f"{None:.2f}"`.** `adx_14`/`atr_14` are None for the whole ADX warm-up.
    That exact format string took the regime daemon down 163 times on
    2026-08-26 and a status card reading the same fields will not repeat it.
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

from realtime import check_live_signals as cs                      # noqa: E402


# ==========================================================================
# fixtures: the three files this card reads
# ==========================================================================

def _gate(status="MUTED", reason="indicator_warmup", entries=False,
          exits=True, adx=None, atr=None, **kw):
    return {"strategy_id": "demo_NQ_1h", "portfolio_id": "Incubator-Odd",
            "symbol": "NQ", "timeframe": "1h", "version": "A",
            "optimal_regime": "Q2", "optimal_regime_label": "Q2_HIGH_VOL_CHOP",
            "live_quadrant": "Q0", "live_regime": "Q0_UNDEFINED_WARMUP",
            "bar_ts": "2026-08-26 14:00:00+00:00",
            "status": status, "reason": reason,
            "detail": "NQ is inside the ADX(14)/ATR(14) warm-up.",
            "entries_allowed": entries, "exits_allowed": exits,
            "adx_14": adx, "atr_14": atr, "theta_vol": 16.23, **kw}


def _regime(strategies=None, symbols=None):
    return {"schema_version": 3, "updated_at": "2026-08-26T15:09:29+00:00",
            "symbols": symbols if symbols is not None else {
                "NQ": {"quadrant": "Q0", "regime": "Q0_UNDEFINED_WARMUP",
                       "adx_14": None, "atr_14": None, "theta_vol": 16.23,
                       "n_bars": 16, "tf": "1h"}},
            "strategies": strategies if strategies is not None
            else {"demo_NQ_1h": _gate()}}


def _portfolios(active=("demo_NQ_1h",)):
    return {"version": "1.1.0", "portfolios": {"Incubator-Odd": {
        "portfolio_id": "Incubator-Odd", "account_type": "incubator_sim",
        "target_account": "SimIncubator1",
        "basket": {"assets": ["MNQ"]},
        "active_strategies": list(active),
        "strategy_allocations": {a: {"symbol": "NQ", "timeframe": "1h",
                                     "allocation": 1, "status": "incubating"}
                                 for a in active}}}}


def _meta():
    return {"name": "demo_NQ_1h", "symbols": ["NQ"], "timeframe": "1h",
            "version": "A",
            "risk": {"sl_atr_mult": 2.0, "tp_atr_mult": 2.0, "trailing": False}}


#: A REAL cycle block, captured verbatim from
#: `master_live.py --dry-run --once --tf 1h --feed nt8` on 2026-08-26. Not
#: hand-written: the point of these cases is that the parser reads what the
#: dispatcher actually emits, so the fixture has to be what it emitted.
REAL_CYCLE = """[master_live] newest closed bar 2026-08-26 14:00:00+00:00 \
— closed 21.4 min ago (1h bar = 60 min)
[2026-08-26T15:21:21+00:00] cycle DRY RUN  symbols=3  strategies=1  458.6ms
       HOLD t3_braid_scalp_20260823_NQ_1h MNQ — MNQ (regime read from NQ) is \
in Q0 (Q0_UNDEFINED_WARMUP); Incubator-Odd trades ['Q1', 'Q2', 'Q3', 'Q4']. \
Standing down rather than trading the environment nobody certified.
"""


def _wire(tmp: Path, *, regime=None, portfolios=None, meta=_meta(),
          engine_state=None, kill=None, spool=("NQ_1m.csv",),
          cycle_log="") -> None:
    """
    Point every module-level path at `tmp`.

    Rebound rather than monkeypatched into os.environ, because several of these
    are resolved at IMPORT time by the modules this one imports, and a test
    that set the variable afterwards would silently read the real files under
    `data/` - which on this box are the LIVE ones.
    """
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    (tmp / "config").mkdir(parents=True, exist_ok=True)

    cs.REGIME_STATE = tmp / "data" / "live_regime_state.json"
    if regime is not None:
        cs.REGIME_STATE.write_text(json.dumps(regime))

    cs.PORTFOLIO_CONFIG = tmp / "config" / "portfolios.json"
    if portfolios is not None:
        cs.PORTFOLIO_CONFIG.write_text(json.dumps(portfolios))

    cs.INCUBATOR = tmp / "incubator"
    if meta is not None:
        d = cs.INCUBATOR / "demo_NQ_1h"
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta))

    cs.MASTER_LOG = tmp / "logs" / "master_live.log"
    cs.MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    cs.MASTER_LOG.write_text(cycle_log)

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


def _card(tmp: Path, **kw) -> str:
    """Render with no engine process, which is this box's resting state."""
    _wire(tmp, **kw)
    try:
        snap = cs.collect()
        snap["engine"] = {"pids": [], "dry_run": None, "args": None,
                          "unit_active": "inactive", "ps_ok": True}
        return cs.render(snap)
    finally:
        _unwire()


# ==========================================================================
# the three ways this card could lie
# ==========================================================================

def test_no_session_is_not_reported_as_zero_orders(tmp_path):
    """
    An absent engine_state.json means NO SESSION HAS RUN. `Orders: 0` says
    something else - that the loop ran and sent nothing - and only one of those
    is a reason to go looking at the dispatcher.
    """
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 engine_state=None)
    assert "no session has run" in card
    assert "NOT the same as zero orders" in card
    assert "Orders Dispatched" not in card


def test_a_real_session_reports_its_counters(tmp_path):
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 engine_state={"schema_version": 1, "session": "2026-08-26",
                               "dispatched": {"a": {}, "b": {}},
                               "orders": [{"ok": True}, {"ok": False}],
                               "fills": [{"pnl": 125.5}, {"pnl": -25.0}]})
    assert "Session Date       : 2026-08-26" in card
    assert "$100.50 USD" in card
    assert "1 confirmed of 2 attempted / 40 max per session" in card
    assert "2 this session" in card


def test_an_allocated_strategy_with_no_switchboard_entry_is_not_permitted(tmp_path):
    """
    The routing table allocating an id the daemon never published a status for
    is a real and silent fault. The safe rendering says so; defaulting
    `entries_allowed` to anything would invent a permission.
    """
    card = _card(tmp_path, regime=_regime(strategies={}),
                 portfolios=_portfolios())
    assert "NOT ON THE SWITCHBOARD" in card
    assert "NOT EVALUATED" in card
    assert "ENTRIES PERMITTED" not in card


def test_an_absent_reading_formats_instead_of_raising(tmp_path):
    """
    `f"{None:.2f}"` raises TypeError. `adx_14`/`atr_14` are None for the whole
    ADX warm-up - which is the state this box is in right now - and that exact
    format string took the regime daemon down 163 times on 2026-08-26.
    """
    assert cs.num(None) == "n/a"
    assert cs.num(27.4567) == "27.46"
    assert cs.num(0.0) == "0.00", "a measured zero is not an absence"

    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios())
    assert "ADX=n/a ATR=n/a" in card
    assert "None" not in card.split("--- Risk")[0].replace(
        "trailing=False", ""), "a None leaked into the strategy block"


# ==========================================================================
# the gate, as the switchboard states it
# ==========================================================================

def test_a_muted_gate_reads_as_blocked(tmp_path):
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios())
    assert "MUTED (indicator_warmup) — entries BLOCKED, exits ALLOWED" in card
    assert "HOLD / NO NEW ENTRY" in card
    assert "Certified For  : Q2 (Q2_HIGH_VOL_CHOP)" in card
    assert "Market Regime  : Q0 (Q0_UNDEFINED_WARMUP)" in card


def test_a_live_gate_reads_as_permitted(tmp_path):
    """The other side of the same switch, so the card is not stuck on MUTED."""
    gate = _gate(status="LIVE", reason="regime_match", entries=True,
                 adx=31.2, atr=18.4, live_quadrant="Q2",
                 live_regime="Q2_HIGH_VOL_CHOP")
    card = _card(tmp_path, regime=_regime(strategies={"demo_NQ_1h": gate}),
                 portfolios=_portfolios())
    assert "entries ALLOWED" in card
    assert "ENTRIES PERMITTED" in card
    assert "ADX=31.20" in card


def test_the_promoted_risk_parameters_are_reported(tmp_path):
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios())
    assert "SL=2.0 ATR, TP=2.0 ATR, trailing=False" in card
    assert "Certified Tape : ['NQ'] at 1h · version A" in card
    assert "account SimIncubator1" in card


def test_an_allocated_strategy_that_was_never_promoted_is_named(tmp_path):
    """
    `approved_incubator/<id>/meta.json` missing means the routing table
    allocates something no promotion ever produced. Silently omitting the risk
    line would leave that looking like a formatting gap.
    """
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 meta=None)
    assert "NOT FOUND" in card
    assert "never promoted" in card


def test_nothing_allocated_says_so(tmp_path):
    card = _card(tmp_path, regime=_regime(strategies={}),
                 portfolios=_portfolios(active=()))
    assert "no portfolio lists an active strategy" in card
    assert "nothing is allocated, so nothing is evaluated" in card


# ==========================================================================
# the engine, running and not
# ==========================================================================

def test_a_stopped_loop_is_a_resting_state_not_an_error(tmp_path):
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios())
    assert "IDLE — master_live.py is not running" in card
    assert "resting state" in card
    assert "no cycle recorded" in card
    assert "Traceback" not in card


def test_the_mode_comes_from_the_live_command_line(tmp_path):
    """
    An operator who edited the unit to arm the loop and one who did not are
    exactly what this line has to tell apart, so it reads the RUNNING argv
    rather than the unit file.
    """
    _wire(tmp_path, regime=_regime(), portfolios=_portfolios())
    try:
        snap = cs.collect()
        snap["engine"] = {"pids": [4242], "dry_run": True, "ps_ok": True,
                          "args": "python3 master_live.py --dry-run",
                          "unit_active": "active"}
        assert "DRY RUN" in cs.render(snap)

        snap["engine"]["dry_run"] = False
        card = cs.render(snap)
        assert "LIVE EXECUTION — orders will be sent" in card
        assert "ACTIVE (PID: 4242)" in card
    finally:
        _unwire()


def test_latency_comes_from_the_dispatchers_own_cycle_line(tmp_path):
    """
    An earlier version of this card declared latency unavailable because
    `master_live.py` has no `perf_counter`. That was WRONG: the timing lives in
    `LiveExecutionDispatcher.describe_cycle`, which prints `<elapsed>ms` in a
    header once per cycle. It is parsed back rather than declared absent.
    """
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 cycle_log=REAL_CYCLE)
    assert "458.6 ms (DRY RUN, 3 symbol(s), 1 strategy)" in card
    assert "Last Cycle Ran     : 2026-08-26 15:21:21 UTC" in card
    assert "Last Evaluated Bar : 2026-08-26 14:00:00 UTC" in card


def test_no_cycle_logged_reports_nothing_timed_rather_than_zero(tmp_path):
    """A blank latency reads as a fast cycle; zero reads as an instant one."""
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios())
    assert "no cycle logged — nothing has been timed" in card
    assert "0.0 ms" not in card


def test_the_cycle_verdicts_are_the_dispatchers_own_words(tmp_path):
    """
    The reason belongs to the dispatcher, written when the decision was taken.
    Paraphrasing it here would offer this tool's account of a decision it did
    not make - and the reason is the whole value of the line.
    """
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 cycle_log=REAL_CYCLE)
    assert "Last cycle's decisions (DRY RUN" in card
    assert "HOLD  t3_braid_scalp_20260823_NQ_1h MNQ" in card
    assert "Standing down rather than trading the environment nobody certified." \
        in card.replace("\n", " ").replace("        ", " ")
    # The gate's view is kept and labelled as the OTHER question.
    assert "what would be permitted" in card


def test_the_parser_matches_describe_cycles_real_format():
    """
    THE COPY IS CHECKED, NOT TRUSTED.

    `CYCLE_HEADER_RE` and `CYCLE_VERDICTS` restate a vocabulary owned by
    `LiveExecutionDispatcher.describe_cycle`. This drives the real method with
    a synthetic report and asserts the parser recognises what comes out - so a
    change to that formatter fails here rather than silently turning every
    cycle into "decided nothing".
    """
    from types import SimpleNamespace                          # noqa: PLC0415
    from realtime.live_dispatcher import (                     # noqa: PLC0415
        LiveExecutionDispatcher)

    report = {
        "dry_run": True, "started_at": "2026-08-26T15:21:21+00:00",
        "symbols": ["MNQ", "MES", "MGC"], "dispatches": [], "plan": [],
        "payloads": [], "ml_vetoes": [], "errors": [], "exit_signals": [],
        "declines": [{"strategy_id": "demo_NQ_1h", "symbol": "MNQ",
                      "reason": "is in Q0; standing down."}],
        "elapsed_ms": 458.6,
    }
    text = LiveExecutionDispatcher.describe_cycle(
        SimpleNamespace(strategies={"demo_NQ_1h": object()}), report)
    lines = text.splitlines()

    m = cs.CYCLE_HEADER_RE.match(lines[0].strip())
    assert m is not None, f"the header no longer parses: {lines[0]!r}"
    assert m.group(2) == "DRY RUN"
    assert float(m.group(5)) == 458.6

    kinds = {ln.strip().split(None, 1)[0] for ln in lines[1:] if ln.strip()}
    unknown = kinds - set(cs.CYCLE_VERDICTS)
    assert not unknown, f"describe_cycle emits verdicts this parser drops: {unknown}"


# ==========================================================================
# the kill switch and the caps
# ==========================================================================

def test_an_armed_kill_switch_is_impossible_to_miss(tmp_path):
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 kill="2026-08-26T15:00:00Z MANUAL HALT — reconciling NQ")
    assert "ARMED — NO NEW ORDER WILL BE SENT" in card
    assert "MANUAL HALT" in card


def test_a_clear_kill_switch_says_so(tmp_path):
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios())
    assert "Kill Switch        : CLEAR" in card


def test_the_caps_are_the_firewalls_own(tmp_path):
    """
    Imported from `risk_firewall`, not restated. A card quoting a cap the
    firewall no longer enforces is worse than one quoting none.
    """
    from realtime.risk_firewall import DEFAULT_LIMITS          # noqa: PLC0415
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios())
    assert f"{DEFAULT_LIMITS['max_open_positions']} open positions" in card
    assert f"{DEFAULT_LIMITS['max_contracts_per_order']} contracts/order" in card


# ==========================================================================
# the feed, and what nothing is watching
# ==========================================================================

def test_unwatched_symbols_come_from_the_spool_not_the_regime_file(tmp_path):
    """
    The daemon classifies only its REGISTERED targets, so the regime file lists
    one symbol while the feed carries a dozen. Sourcing this from the regime
    file compares a set against itself and reports nothing, every time - as
    wrong as it is quiet, and it is what the first version of this did.
    """
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 spool=("NQ_1m.csv", "MNQ_1m.csv", "ES_1m.csv", "FDAX_1m.csv"))
    assert "Feed symbols no allocated strategy watches (2)" in card
    assert "ES, FDAX" in card
    # NQ is allocated and MNQ is its execution micro; neither is unwatched.
    assert "MNQ" not in card.split("no allocated strategy watches")[1]


def test_the_unwatched_line_does_not_claim_they_are_classified(tmp_path):
    """
    They are spooled and nothing more - no quadrant is computed and no gate is
    drawn. "Classified" would credit the stack with work it did not do.
    """
    card = _card(tmp_path, regime=_regime(), portfolios=_portfolios(),
                 spool=("NQ_1m.csv", "ES_1m.csv"))
    tail = card.split("no allocated strategy watches")[1]
    assert "spooled and nothing else" in tail
    assert "classified" not in tail


# ==========================================================================
# robustness
# ==========================================================================

def test_a_truncated_state_file_does_not_crash(tmp_path):
    _wire(tmp_path, regime=_regime(), portfolios=_portfolios())
    try:
        cs.REGIME_STATE.write_text('{"strategies": {"demo_NQ_1h":')
        snap = cs.collect()
        snap["engine"] = {"pids": [], "dry_run": None, "args": None,
                          "unit_active": None, "ps_ok": True}
        card = cs.render(snap)
        assert "STRATEGY & SIGNAL EVALUATION STATUS" in card
        assert "Traceback" not in card
    finally:
        _unwire()


def test_an_unreadable_portfolio_config_is_named(tmp_path):
    _wire(tmp_path, regime=_regime(), portfolios=_portfolios())
    try:
        cs.PORTFOLIO_CONFIG.write_text("{not json")
        snap = cs.collect()
        snap["engine"] = {"pids": [], "dry_run": None, "args": None,
                          "unit_active": None, "ps_ok": True}
        card = cs.render(snap)
        assert "unreadable" in card
        assert "Traceback" not in card
    finally:
        _unwire()


def test_a_stale_regime_file_is_flagged(tmp_path):
    """
    The daemon republishes every 5 minutes. Past twice that, the verdicts on
    this card describe a bar that is no longer current, and saying so is the
    difference between a status and a stale status that looks like one.
    """
    import time as _t
    _wire(tmp_path, regime=_regime(), portfolios=_portfolios())
    try:
        old = _t.time() - 3600
        os.utime(cs.REGIME_STATE, (old, old))
        snap = cs.collect()
        snap["engine"] = {"pids": [], "dry_run": None, "args": None,
                          "unit_active": None, "ps_ok": True}
        card = cs.render(snap)
        assert "no longer current" in card
    finally:
        _unwire()


def test_the_tool_does_not_import_pandas():
    """247ms against the ~50ms this needs, and nothing would fail."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "import realtime.check_live_signals;"
         "print(','.join(m for m in ('pandas','numpy','vectorbtpro')"
         "                if m in sys.modules))" % str(REPO)],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"pulled in: {out.stdout.strip()}"


def test_it_runs_from_any_directory(tmp_path):
    """
    The acceptance criterion as written, plus the exit code: 1 when the loop is
    NOT evaluating, so the alias chains. That means "is it evaluating", not
    "is something wrong" - an idle loop is a legitimate resting state here.
    """
    out = subprocess.run(
        [sys.executable, str(REPO / "realtime" / "check_live_signals.py")],
        capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=120)
    assert out.returncode in (0, 1)
    assert "STRATEGY & SIGNAL EVALUATION STATUS" in out.stdout
    assert "Traceback" not in out.stderr


def test_it_writes_nothing(tmp_path):
    _wire(tmp_path, regime=_regime(), portfolios=_portfolios())
    try:
        before = sorted(p.name for p in (tmp_path / "data").iterdir())
        cs.collect()
        assert sorted(p.name for p in (tmp_path / "data").iterdir()) == before
    finally:
        _unwire()
