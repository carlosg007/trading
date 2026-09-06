#!/usr/bin/env python3
"""
master_live.py - the live execution loop.

    python3 master_live.py                             # dry run, the DEFAULT
    python3 master_live.py --dry-run --interval-sec 30 --once
    python3 master_live.py --live --interval-sec 60     # LIVE. Sends orders.

One process, one job: every `--interval-sec`, read the newest closed bars for
the symbols the four baskets hold, hand them to
`realtime.live_dispatcher.LiveExecutionDispatcher`, and print what it decided.
The pipeline itself lives in that module; this file is the CLI, the loop and
the shutdown handling, so the wiring can be unit-tested without a clock.

TWO PROCESSES, ONE DIRECTION
============================
This loop does NOT classify regimes. `realtime/regime_daemon.py` publishes
`data/live_regime_state.json`; this reads it. Keeping them apart means a slow
indicator pass can never stall a dispatch, and it means this loop's view of the
market is a file somebody can inspect after the fact rather than a value that
existed for one millisecond inside a process that has since exited.

Run the daemon separately. With no state file this loop declines every signal
and says so on every line - which is the correct behaviour, because an unknown
environment is not a permitted one.

WHAT --dry-run ACTUALLY GUARANTEES
==================================
Every stage runs: strategies are loaded and hash-checked, signals are computed,
regime and ML gates are applied, positions are netted and sized, and payloads
are formatted and validated. The single thing that does not happen is the
socket. That is the mode to run first, and the mode to run after any config
change.

It is also the DEFAULT: a command that names neither mode sends nothing, and
`--live` is the only thing that arms the socket. The safe mode is the one you
get by forgetting a flag, because the unsafe one is unrecoverable - a market
order this process cannot see is not undone by noticing the mistake.

Live mode refuses to start without a webhook URL, rather than discovering it at
the first order - see `LiveExecutionDispatcher.__init__`.

MARKET HOURS: THE LOOP STOPS BEING A PROCESS, NOT JUST AN IDLE ONE
==================================================================
`--halt-when-closed` makes the loop EXIT when the CME is shut - the daily
17:00-18:00 ET maintenance halt, the Friday 17:00 ET weekend close, and any
exchange holiday `realtime/market_calendar.py` has a file for - and it exits
with a status systemd is told not to restart on.
`deploy/systemd/trading-master-live.timer` starts it again at 17:55 ET on
Sunday through Thursday, and the process waits out the last few minutes before
the open.

IT EXITS RATHER THAN SLEEPS, AND THAT IS THE WHOLE POINT. Measured 2026-09-06:
this loop reached 18.5 GB RSS after 47 hours on a 24 GB box and OOM-killed the
test gate. A sleeping process keeps every one of those bytes. `gc.collect()`
does not give them back either - CPython returns freed arenas to the OS only
when they happen to be entirely empty, and a fragmented 18 GB heap is not. The
only thing that reliably returns 18 GB to the machine is exit. So the weekend
costs nothing, and the daily halt caps unbounded growth at one session's worth
instead of a week's.

`--halt-when-closed` IS OFF BY DEFAULT and the unit file passes it. An operator
running a dry run by hand at 17:30 wants a loop that keeps printing, not one
that vanishes; a supervised process wants the opposite. The flag is where the
two are told apart.

NOTHING IS FLATTENED AT THE CLOSE, and nothing is cancelled. This loop sends
market orders inside one `send_execution_signal` call: it holds no resting
entry order to cancel and no socket to drain, and it never did. What the close
does is REPORT - the positions the book holds, the unverified claims
`EngineState` holds, and a loud line if either is non-empty going into a
49-hour weekend. Being flat over a weekend is a trading decision and belongs to
the operator and to CrossTrade NAM, not to a scheduler.

GRACEFUL SHUTDOWN
=================
SIGINT and SIGTERM set a flag; they never interrupt a cycle. A cycle is signals
-> gates -> sizing -> dispatch, and killing it midway can leave orders sent
with nothing printed about them. The handler therefore lets the cycle it is in
finish, skips the sleep, and exits after the loop - so the last thing on the
console is always a complete cycle. A SECOND signal exits immediately, because
an operator pressing Ctrl-C twice means it.

Sockets are opened and closed inside one `send_execution_signal` call with a
hard 2.0s timeout and are never held between cycles, so there is no connection
pool to drain and no orphan to leave behind.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ, so an operator
# opening a fresh terminal never has to `source .env` first. It runs at import
# time, above the imports below, because modules resolve their BT_* variables
# while being imported and loading the file inside main() would be too late for
# those - and would work here, which is the kind of difference nobody notices
# until one runner silently uses the default path. The rules live in ONE
# module: see mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import load_env                                     # noqa: E402

load_env()
# ---------------------------------------------------------------------------

import argparse
import gc
import signal
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.config_loader import DEFAULT_CONFIG_PATH          # noqa: E402
from realtime.live_dispatcher import (DEFAULT_ENV_FILE,          # noqa: E402
                                      DEFAULT_MAX_ATTEMPTS,
                                      DEFAULT_MODEL_DIR,
                                      DEFAULT_STRATEGY_ROOT,
                                      DEFAULT_TIMEOUT_S,
                                      LiveDispatchError,
                                      LiveExecutionDispatcher)
from realtime.feed import FeedError, resolve_feed                # noqa: E402
from realtime.lifecycle import (EngineState,                     # noqa: E402
                                emergency_halt,
                                startup_report)
from realtime.risk_firewall import RiskFirewall                  # noqa: E402
from realtime.nt8_positions import (PositionSnapshotError,       # noqa: E402
                                    describe as describe_positions,
                                    load_snapshot, reconcile)
from realtime.market_calendar import (MarketCalendarError,      # noqa: E402
                                      OPEN as MARKET_OPEN,
                                      describe as describe_market,
                                      load_holidays,
                                      seconds_until_open,
                                      session_phase)
from realtime.regime_reader import (DEFAULT_STATE_FILE,          # noqa: E402
                                    RegimeStateError, get_all_regimes)

DEFAULT_INTERVAL_S = 60

#: The exit status that means "the exchange is shut and this was on purpose".
#:
#: NOT 0, and the difference is operational rather than cosmetic. The unit runs
#: `Restart=always`, which restarts on a clean exit too, so a 0 here would put
#: the process straight back up and hold 18 GB across the entire weekend it was
#: told to release. A distinct code is what `RestartPreventExitStatus=` can
#: name. It is also distinct from 2 (REFUSING TO START) and from 1 (the run had
#: failures): "shut for the weekend", "misconfigured" and "degraded" are three
#: different mornings for whoever reads `systemctl status`.
#:
#: `--closed-exit-code 0` exists for a caller that supervises differently.
EXIT_MARKET_CLOSED = 3

#: How long the startup wait may sit in front of a market that is about to
#: open, before the process exits instead. The timer starts this unit at 17:55
#: ET for an 18:00 open, so five minutes is the real case and thirty is slack.
#: A WEEKEND is 49 hours and must never be waited out in memory, which is what
#: bounding this buys.
DEFAULT_OPEN_WAIT_S = 1800.0


class ShutdownFlag:
    """
    SIGINT / SIGTERM, recorded rather than acted on immediately.

    A flag instead of an exception because an exception raised inside a cycle
    can unwind between the send and the print - leaving an order on a broker
    and no line in the log saying so. A second signal is honoured at once: the
    first is "stop when it is safe", the second is "stop".
    """

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, _frame) -> None:
        name = signal.Signals(signum).name
        if self.requested:
            print(f"\n[master_live] second {name}: exiting immediately.",
                  flush=True)
            raise SystemExit(130)
        self.requested = True
        self.signal_name = name
        print(f"\n[master_live] {name} received. Finishing the current cycle, "
              f"then stopping. Press again to exit now.", flush=True)

    def sleep(self, seconds: float, step: float = 0.25) -> None:
        """Sleep in slices so a signal is noticed within `step`, not `seconds`."""
        deadline = time.monotonic() + seconds
        while not self.requested and time.monotonic() < deadline:
            time.sleep(min(step, max(0.0, deadline - time.monotonic())))


def load_symbol_bars(symbols, tf: str, lookback_bars: int,
                     feed=None, source: str | None = None
                     ) -> tuple[dict, dict]:
    """
    `({basket_symbol: bars}, {basket_symbol: which contract supplied them})`.

    THE SEAM, NOW WITH SOMETHING BEHIND IT. This used to be a lake read inlined
    here, which is why the loop could poll every sixty seconds against a tape
    that stopped weeks ago and report nothing wrong: the newest bar the lake
    can return is the newest bar somebody ingested. `realtime/feed.py` owns the
    choice now - a live vendor feed when one is configured, the lake otherwise -
    and this function is the adapter between it and the loop.

    **The bars MUST be closed bars.** The engine fills at the next bar's open,
    so acting on the last CLOSED bar with a market order now is that fill;
    acting on a bar still forming is lookahead, and it is lookahead that
    produces a live equity curve worse than the backtest for reasons nobody can
    find afterwards. A live feed always has a forming bar - the one the market
    is printing into right now - and `realtime.feed.drop_forming_bar` removes
    it before any of these frames reach a strategy. That rule is written once,
    there, and every feed passes through it.

    THE MICRO ALIAS. The four baskets hold MNQ/MES/MCL/MGC and the tape is the
    full-size contract's - same price series, same tick size, only the
    multiplier differs, and a multiplier appears in no indicator - so a micro's
    bars are read from its parent through `realtime/contract_alias.py`, the one
    table the regime reader and the dispatcher also resolve through. The
    substitution is REPORTED rather than silent: the order is still for the
    micro and still sized on the micro's own point value.

    `feed` is injected so the loop can be tested without a vendor and so the
    mode is decided once, at startup, rather than re-resolved every cycle.
    `source` names a mode instead - `load_symbol_bars(..., source="nt8")` is
    the same thing `--feed nt8` gets, resolved here for a caller that has a
    name rather than a feed. An injected `feed` WINS: it is the object the
    loop already described on the console, and re-resolving it per call is how
    a run ends up reading one tape while its startup banner names another.

    THE PUSH LISTENER ARRIVES THROUGH THIS SAME PATH.
    `realtime/nt8_bar_listener.py` receives bars over HTTP and APPENDS THEM TO
    THE NT8 SPOOL rather than holding its own store, so `source="nt8"` reads
    pushed bars and spooled bars identically and there is one format, one
    forming-bar drop, one micro alias and one `ts` conversion. A second bar
    store read directly here would have bypassed all four.
    """
    from realtime.feed import resolve_feed

    if feed is None:
        feed = resolve_feed(source or "auto")
    return feed.closed_bars(symbols, tf, lookback_bars)


def required_timeframes(dispatcher: LiveExecutionDispatcher,
                        fallback: str) -> list[str]:
    """
    The bar widths this roster actually needs, one bucket each.

    Until 2026-08-27 the loop read ONE timeframe and handed it to every
    strategy. A 3m certification was therefore evaluated on 1h bars: real
    signals, correct log lines, and a certification describing a different
    tape. `StrategyHandle.certified_timeframe` is what that bug cost, and this
    is what spends it - the loop now reads each width the roster names and
    gives each strategy only its own.

    A handle declaring NO timeframe falls back to `--tf`, which is the old
    behaviour and is correct for it: a meta.json written before the key
    existed carries no claim to contradict, and stranding it would be a
    regression for every strategy promoted before the guard.

    Sorted by WIDTH, narrowest first, so the console reads 3m before 1h and a
    reader can see the cheap buckets complete before the expensive ones.
    """
    from realtime.feed import tf_delta                              # noqa: PLC0415

    wanted = {h.certified_timeframe or fallback for h in dispatcher.strategies}
    wanted = {str(tf).strip().lower() for tf in wanted if tf}
    if not wanted:
        return [fallback]

    def width(tf: str):
        try:
            return tf_delta(tf)
        except Exception:                                      # noqa: BLE001
            # An unmeasurable width sorts last rather than killing the sort.
            return pd.Timedelta.max
    return sorted(wanted, key=width)


def basket_symbols(dispatcher: LiveExecutionDispatcher) -> list[str]:
    """Every contract the four baskets hold, deduplicated."""
    return sorted({asset
                   for p in dispatcher.portfolios.values()
                   for asset in p["basket"]["assets"]})


def rss_mb() -> float:
    """
    This process's resident set size in MB, from /proc, or 0.0 off Linux.

    Read straight out of `/proc/self/statm` rather than through psutil: the
    number is wanted in a shutdown path that must not be able to fail, and the
    second field there is resident pages, which is the same thing `ps -o rss`
    prints. It is EVIDENCE, not a control - nothing branches on it - and it is
    on the console at the close so the growth this whole gate exists to bound
    is a number somebody can read in a log rather than a claim.
    """
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        import resource                                        # noqa: PLC0415
        return pages * resource.getpagesize() / (1024 * 1024)
    except Exception:                                          # noqa: BLE001
        return 0.0


def release_caches() -> str:
    """
    Drop what this process is knowingly holding, and say what it bought.

    THIS IS NOT THE MECHANISM THAT FIXES THE MEMORY - exiting is. It is here
    because the caches it clears are the ones this repository can NAME
    (`mdlib.lake`'s four `lru_cache`s over the catalog, the coverage frame, the
    degraded-day set and the roll dates), and because reporting RSS on both
    sides of the collect turns "the loop leaks" into a measured before/after on
    every close. If a future run shows the collect recovering most of 18 GB,
    the leak is a reference this function can reach; if it recovers ~nothing,
    it is fragmentation or an extension allocation and only exit will do -
    which is what the 2026-09-06 measurement suggests and what nobody has
    confirmed either way.

    Never raises. A shutdown that failed while tidying up would lose the
    position report printed after it, which is the part an operator needs.
    """
    before = rss_mb()
    cleared = []
    try:
        from mdlib import lake                                  # noqa: PLC0415
        for name in ("available_symbols", "coverage", "degraded_days",
                     "roll_dates"):
            fn = getattr(lake, name, None)
            if fn is not None and hasattr(fn, "cache_clear"):
                fn.cache_clear()
                cleared.append(f"lake.{name}")
    except Exception as exc:                                    # noqa: BLE001
        cleared.append(f"lake caches NOT cleared ({type(exc).__name__}: {exc})")
    collected = gc.collect()
    after = rss_mb()
    return (f"[master_live] caches released: {', '.join(cleared) or 'none'}; "
            f"gc collected {collected} object(s); RSS {before:,.0f} MB -> "
            f"{after:,.0f} MB. Only the exit below returns the rest to the "
            f"machine.")


def market_close_report(dispatcher: LiveExecutionDispatcher,
                        engine_state: EngineState,
                        phase: str, reason: str) -> str:
    """
    What is still open as the exchange shuts, stated rather than acted on.

    Two independent records, and they are NOT the same thing:

      * `dispatcher.positions` — what THIS PROCESS believes it holds, after the
        last cycle reconciled it against the broker snapshot.
      * `engine_state.open_claims()` — what this process SENT and has not seen
        confirmed. A claim is not a position.

    Both are printed because they can disagree, and a disagreement at the close
    is the one worth looking at. NEITHER IS ACTED ON. A scheduler that
    flattened at 17:00 would be making a trading decision on a clock, and would
    do it on a Monday maintenance break as readily as on a Friday - so the
    stand-down says what is held and stops.

    The WEEKEND line is louder than the MAINTENANCE one on purpose: carrying
    inventory through a 60-minute halt is ordinary, carrying it through 49
    hours of gap risk with no process watching is a decision somebody should
    have made deliberately.
    """
    lines = [f"[master_live] MARKET CLOSED — {phase}: {reason}."]
    try:
        held = dispatcher.positions.open_positions()
    except Exception as exc:                                    # noqa: BLE001
        held = []
        lines.append(f"  position book UNREADABLE: {type(exc).__name__}: {exc}")
    claims = engine_state.open_claims()

    if held:
        lines.append(f"  {len(held)} position(s) OPEN in this process's book:")
        for rec in held[:10]:
            lines.append(f"    {rec.get('portfolio_id')}/{rec.get('symbol')} "
                         f"{rec.get('direction')} x{rec.get('quantity')}")
    else:
        lines.append("  position book: FLAT (nothing this process opened is "
                     "still recorded open).")
    if claims:
        lines.append(f"  {len(claims)} UNVERIFIED claim(s) in "
                     f"{engine_state.path} — sent, never confirmed. Reconcile "
                     f"in NinjaTrader.")

    if held or claims:
        lines.append("  NOTHING IS BEING FLATTENED. Closing a position is a "
                     "trading decision; this is a scheduler.")
        if phase == "WEEKEND":
            lines.append("  ^ THIS IS THE WEEKEND CLOSE. Inventory carried "
                         "here is unwatched for ~49 hours and reopens on a "
                         "Sunday gap. CrossTrade NAM is the only thing still "
                         "governing the account until Sunday 18:00 ET.")
    lines.append("  No sockets are held between cycles, and no resting entry "
                 "order exists to cancel: every order this loop sends is a "
                 "market order completed inside one call.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="master_live.py",
        description="The live execution loop: signals -> regime gate -> ML "
                    "gate -> netting/sizing -> CrossTrade.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run --dry-run first. Live mode sends real orders.")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                    help="path to config/portfolios.json")
    ap.add_argument("--state-file", default=DEFAULT_STATE_FILE,
                    help="path to data/live_regime_state.json")
    # DEFAULT TRUE, and `--live` is the only way off it. The two are separate
    # flags rather than one `--no-dry-run` because arming real money should be
    # a word an operator means, not a double negative typed by habit; and
    # `--dry-run` stays spellable so every documented command still parses and
    # still says on its face which mode it is in.
    ap.add_argument("--dry-run", action="store_true", default=None,
                    help="run every gate and format every payload, but open "
                         "no socket. THIS IS THE DEFAULT")
    ap.add_argument("--live", action="store_true",
                    help="SEND REAL ORDERS. Without it the loop is a dry run")
    ap.add_argument("--interval-sec", type=float, default=DEFAULT_INTERVAL_S,
                    help=f"seconds between cycles (default {DEFAULT_INTERVAL_S})")
    ap.add_argument("--once", action="store_true",
                    help="run a single cycle and exit")
    ap.add_argument("--tf", default="15m", help="bar timeframe to read")
    ap.add_argument("--feed", default="auto",
                    choices=("auto", "nt8", "live", "lake"),
                    help="where bars come from. auto: the NT8 broker feed when "
                         "it is publishing, the lake otherwise. nt8 (or its "
                         "alias live): REFUSES to start when NT8 is not "
                         "publishing, rather than falling back to historical "
                         "bars that read exactly like a quiet market. lake: "
                         "the historical store, as current as the last ingest. "
                         "Live data is the BROKER's; Databento is historical "
                         "only")
    ap.add_argument("--lookback-bars", type=int, default=500,
                    help="bars handed to each strategy per cycle")
    ap.add_argument("--positions-snapshot", default=None,
                    help="broker position snapshot to reconcile the book "
                         "against each cycle (default $BT_NT8_POSITIONS, else "
                         "/mnt/backtest/artifacts/nt8_positions/positions.json). "
                         "ABSENT IS NOT AN ERROR: with no publisher the loop "
                         "keeps only what it opened itself, exactly as before.")
    ap.add_argument("--no-position-reconcile", dest="reconcile_positions",
                    action="store_false",
                    help="do not read the broker snapshot. The loop then "
                         "cannot flatten a position it did not open in this "
                         "process - which is the pre-2026-09-03 behaviour and "
                         "is why every exit read 'position state is not known "
                         "here' after a restart.")
    ap.add_argument("--no-tag-manifest", dest="tag_manifest",
                    action="store_false",
                    help="do not refresh the CrossTrade strategy tag manifest "
                         "at startup. The manifest is what a journal "
                         "pre-registers locks from; skipping the refresh "
                         "leaves it describing whatever config/portfolios.json "
                         "said the last time it was written.")
    ap.add_argument("--max-regime-write-age-sec", type=float, default=900.0,
                    help="CRITICAL and stand every strategy down when the "
                         "regime daemon has not WRITTEN for this long "
                         "(default 900). This is the publisher's own clock, "
                         "not the market's, so it is timeframe-independent - "
                         "unlike --max-regime-age-sec, which compares "
                         "max(write age, bar age) and cannot be defaulted "
                         "because a 1h bar is legitimately ~3,600s old before "
                         "it closes. 0 disables the guard.")
    ap.add_argument("--max-regime-age-sec", type=float, default=None,
                    help="refuse a regime reading older than this. Left unset, "
                         "ages are reported and nothing is refused - how old "
                         "is too old depends on the timeframe")
    ap.add_argument("--strategy-root", default=DEFAULT_STRATEGY_ROOT)
    ap.add_argument("--models", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    ap.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                    help="attempts for failures that PROVE nothing was sent. "
                         "A timeout is never retried")
    ap.add_argument("--state-path", default=None,
                    help="durable record of what this process has SENT "
                         "(default data/engine_state.json, $BT_ENGINE_STATE). "
                         "It is what stops a restart re-sending an order for "
                         "a bar it already acted on")
    ap.add_argument("--session-cutoff-utc", default=None,
                    help="HH:MM UTC after which no NEW entry is sent, e.g. "
                         "20:55. Exits are unaffected")
    ap.add_argument("--max-session-loss-usd", type=float, default=None,
                    help="halt new orders once THIS LOOP has booked this much "
                         "realised loss today. A local circuit breaker, not "
                         "the funding program's rule — CrossTrade NAM enforces "
                         "that against the live balance")
    ap.add_argument("--no-risk-firewall", action="store_true",
                    help="DISABLE the pre-trade risk gate. There is no good "
                         "reason to pass this against a live account")
    # --- the market-hours gate -------------------------------------------
    # OFF BY DEFAULT, and `deploy/systemd/trading-master-live.service` is what
    # turns it on. A hand-run dry run at 17:30 wants a loop that keeps
    # printing; a supervised process wants one that gets out of memory. The
    # default is the one that surprises nobody at a console.
    ap.add_argument("--halt-when-closed", action="store_true",
                    help="EXIT when the CME is shut — the 17:00-18:00 ET "
                         "daily maintenance halt, the Friday 17:00 ET weekend "
                         "close, and any holiday $BT_CME_HOLIDAYS names — "
                         "instead of idling through it. Exits with "
                         f"--closed-exit-code (default {EXIT_MARKET_CLOSED}), "
                         "which the unit file tells systemd not to restart "
                         "on; the .timer starts it again before the open. "
                         "Nothing is flattened and nothing is cancelled")
    ap.add_argument("--open-wait-sec", type=float, default=DEFAULT_OPEN_WAIT_S,
                    help="with --halt-when-closed, wait for an open that is "
                         "this close rather than exiting "
                         f"(default {DEFAULT_OPEN_WAIT_S:.0f}). It is what "
                         "lets the timer start the unit at 17:55 for an 18:00 "
                         "open. A longer closure is never waited out: the "
                         "point of the gate is to stop holding memory")
    ap.add_argument("--closed-exit-code", type=int,
                    default=EXIT_MARKET_CLOSED,
                    help=f"status to exit with when the market is shut "
                         f"(default {EXIT_MARKET_CLOSED}). NOT 0: this unit "
                         f"runs Restart=always, which restarts on a clean "
                         f"exit too, so 0 would put the process straight back "
                         f"up and hold its memory all weekend")
    ap.add_argument("--holidays", default=None,
                    help="CME holiday CSV (default $BT_CME_HOLIDAYS, else "
                         "/mnt/backtest/reference/calendar/cme_holidays.csv). "
                         "ABSENT IS NOT AN ERROR and no holiday is then "
                         "modelled — a holiday nobody told us about costs one "
                         "idle day, an invented one costs a trading day. A "
                         "file that exists and does not parse REFUSES TO START")
    ap.add_argument("--no-verify-hash", action="store_true",
                    help="skip the meta.json SHA-256 check on strategy code. "
                         "Do not use this to trade an edited module")
    return ap


def resolve_dry_run(args: argparse.Namespace) -> bool:
    """
    The effective mode, from the two flags.

    `--dry-run` is the default and `--live` is the only thing that clears it,
    so a command that says neither sends nothing. Asking for BOTH is refused
    rather than resolved: whichever way it were resolved, half of the command
    would be describing a run that did not happen, and the half that is wrong
    is the half about whether real orders went out.
    """
    if args.live and args.dry_run:
        raise ValueError("--dry-run and --live contradict each other; pass one")
    return not args.live


def regime_write_age(state_file: str) -> tuple[float | None, str]:
    """
    How long ago the regime daemon last WROTE, in seconds, over every symbol.

    THE WRITE AGE, NOT THE BAR AGE, and the distinction is the whole guard.
    `age_seconds` is the DAEMON's clock - how long since it published - and is
    the same number whatever timeframe the loop runs. `bar_age_seconds` is the
    MARKET's, and on a 1h feed it climbs to ~3,600s in the ordinary course of
    an hour that has not finished yet. A single threshold over `max(write,
    bar)` - which is what `--max-regime-age-sec` compares - therefore cannot
    be defaulted: at 900s it would refuse every 1h reading for three quarters
    of every hour, and at 3,600s it would not notice a publisher that died
    fifty minutes ago. That is why that flag ships as None and this one is
    separate.

    Returns `(worst age, description)`; `(None, why)` when no record carries a
    readable write timestamp, which is treated as stale by the caller - an
    unreadable clock is not evidence of freshness.
    """
    try:
        state = get_all_regimes(state_file)
    except RegimeStateError as exc:
        return None, f"regime state unreadable: {exc}"
    if not state:
        return None, "regime state holds no symbols"
    ages = {sym: rec.get("age_seconds") for sym, rec in state.items()}
    readable = {s: a for s, a in ages.items() if a is not None}
    if not readable:
        return None, (f"no symbol carries a readable write timestamp "
                      f"({sorted(ages)})")
    worst_symbol = max(readable, key=lambda s: readable[s])
    worst = float(readable[worst_symbol])
    return worst, (f"oldest write {worst:.0f}s ago ({worst_symbol}); "
                   f"{len(readable)} of {len(ages)} symbols timestamped")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dry_run = resolve_dry_run(args)
    except ValueError as exc:
        parser.error(str(exc))

    # THE HOLIDAY CALENDAR IS RESOLVED FIRST, and a bad one stops the process
    # here. `load_holidays` returns {} for a MISSING file, which is not an
    # error and means no holiday is modelled; it RAISES for a file that exists
    # and does not parse. An operator who wrote that file meant to have
    # holidays, and a live loop that silently traded a closed session because
    # a comma was wrong is the failure they would never see. Loading it now,
    # rather than at the first cycle boundary, is what makes the refusal a
    # startup message instead of a 17:00 surprise.
    #
    # A BAD CALENDAR IS FATAL ONLY WHEN SOMETHING DEPENDS ON IT. With the gate
    # armed, or with --holidays named explicitly, an unparseable file stops the
    # process: the operator meant to have holidays and silently not having them
    # is the failure they would never see. With neither, the calendar is
    # decoration on a banner, and refusing to start a dry run over it would be
    # the gate standing down a loop that was never going to consult it.
    demands_holidays = bool(args.halt_when_closed or args.holidays)
    try:
        holidays = load_holidays(args.holidays)
    except MarketCalendarError as exc:
        if demands_holidays:
            print(f"[master_live] REFUSING TO START: {exc}", file=sys.stderr)
            return 2
        print(f"[master_live] holiday calendar unreadable ({exc}); no holiday "
              f"is modelled. The market-hours gate is not armed, so nothing "
              f"here depends on it.", file=sys.stderr, flush=True)
        holidays = {}

    # The durable record and the pre-trade gate, built BEFORE the dispatcher
    # so an unreadable state file stops the process here rather than at the
    # first order. `EngineState` refuses to start on a corrupt file precisely
    # because "this process has sent nothing" is the belief that re-sends an
    # order it already placed.
    try:
        engine_state = EngineState(args.state_path)
    except RuntimeError as exc:
        print(f"[master_live] REFUSING TO START: {exc}", file=sys.stderr)
        return 2

    firewall = None
    if not args.no_risk_firewall:
        limits = {}
        if args.session_cutoff_utc:
            limits["session_cutoff_utc"] = args.session_cutoff_utc
        if args.max_session_loss_usd is not None:
            limits["max_session_loss_usd"] = args.max_session_loss_usd
        firewall = RiskFirewall(limits=limits or None, state=engine_state)

    try:
        dispatcher = LiveExecutionDispatcher(
            firewall=firewall,
            state=engine_state,
            config_path=args.config,
            state_file=args.state_file,
            dry_run=dry_run,
            strategy_root=args.strategy_root,
            ml_model_dir=args.models,
            env_file=args.env_file,
            max_age_s=args.max_regime_age_sec,
            timeout_seconds=args.timeout_sec,
            max_attempts=args.max_attempts,
            verify_code_hash=not args.no_verify_hash)
    except (LiveDispatchError, OSError) as exc:
        print(f"[master_live] REFUSING TO START: {exc}", file=sys.stderr)
        return 2

    # The feed is resolved ONCE, here, and reported before the first cycle.
    # Resolved per cycle it could change under the loop; unreported, an
    # operator reading "no signal" on every line has no way to tell a quiet
    # market from a feed that fell back to bars from a fortnight ago.
    try:
        feed = resolve_feed(args.feed)
    except FeedError as exc:
        print(f"[master_live] REFUSING TO START: {exc}", file=sys.stderr)
        return 2

    print(dispatcher.describe(), flush=True)
    print(f"[master_live] bar feed: {feed.describe()}", flush=True)
    print(startup_report(engine_state), flush=True)
    if firewall is not None:
        print(firewall.describe(), flush=True)
    else:
        print("[master_live] RISK FIREWALL DISABLED (--no-risk-firewall). "
              "Nothing checks an order between the sizer and the socket.",
              flush=True)
    # ---- the strategy tag manifest ---------------------------------------
    # AUDITED AGAINST THE ROUTING TABLE THIS PROCESS JUST LOADED, at startup
    # rather than on a timer, because `active_strategies` is read once here and
    # a manifest generated from a different config would pre-register locks
    # this loop will never take out. Regenerated in full rather than diffed:
    # the manifest is derived, so rewriting it IS the audit.
    #
    # NEVER FATAL. `export` does not raise. A live loop that refused to start
    # because an NFS mount was busy would be down for a journal convenience,
    # and the loop trades correctly without the file - the tag on the wire is
    # composed at dispatch by the same function, not read from here.
    if args.tag_manifest:
        from scripts.strategy_tag_manifest import export       # noqa: PLC0415
        exported = export(config_path=args.config)
        if exported["ok"]:
            print(f"[master_live] tag manifest {exported['path']} refreshed "
                  f"({exported['singleton_tags']} singleton tag(s) over "
                  f"{exported['rows']} portfolio/symbol pair(s))", flush=True)
        else:
            print(f"[master_live] tag manifest NOT refreshed: "
                  f"{exported['error']} — the loop continues; tags on the "
                  f"wire are composed at dispatch, not read from the file.",
                  file=sys.stderr, flush=True)

    if not dry_run:
        print("[master_live] LIVE MODE — orders will be sent.", flush=True)

    symbols = basket_symbols(dispatcher)
    shutdown = ShutdownFlag()
    shutdown.install()

    # THE MARKET BANNER, printed whether or not the gate is armed. An operator
    # reading "no signal" on every line at 02:00 on a Saturday needs the phase
    # on the console; without it a closed exchange and a dead feed produce the
    # same quiet log, which is the confusion the watchdog already exists to
    # stop making people resolve by hand.
    print(describe_market(holidays=holidays), flush=True)
    if not args.halt_when_closed:
        print("[master_live] market-hours gate DISABLED (no "
              "--halt-when-closed): this process stays resident through the "
              "close and keeps whatever memory it has grown.", flush=True)
    else:
        # THE PRE-OPEN WAIT, and it is bounded. The timer starts this unit at
        # 17:55 ET for an 18:00 open so the dispatcher's load, the hash checks
        # and the tag manifest are all done and REPORTED before the market is
        # trading — a startup failure surfaces with five minutes to fix it
        # rather than at the open. Anything longer than --open-wait-sec is not
        # waited for at all: the cycle gate below then exits immediately,
        # because sitting on a heap for 49 hours is the thing this gate was
        # added to end.
        wait = seconds_until_open(holidays=holidays)
        if 0 < wait <= args.open_wait_sec:
            print(f"[master_live] holding for the open: {wait / 60:.1f} min. "
                  f"Everything above is already loaded; the first cycle runs "
                  f"when the exchange does. Ctrl-C or SIGTERM stops the wait.",
                  flush=True)
            shutdown.sleep(wait + 2.0)          # +2s so 17:59:59.9 is not 18:00

    cycles = failures = 0
    closed_phase: tuple[str, str] | None = None
    while True:
        # THE MARKET GATE, at the top of the cycle and before the position
        # book is touched. Standing down here rather than after
        # `begin_cycle()` means the closed path mints no cycle token, reads no
        # snapshot and evaluates no strategy: the last thing in the log is a
        # complete cycle, exactly as it is for a signal.
        if args.halt_when_closed:
            phase, why = session_phase(holidays=holidays)
            if phase != MARKET_OPEN:
                closed_phase = (phase, why)
                break
        # A signal that arrived during the pre-open WAIT stops the process
        # there. `cycles == 0` is what distinguishes it from a signal received
        # mid-session, which must still finish the cycle it is in - the break
        # at the bottom of the loop is the one that handles that, and this one
        # must not pre-empt it.
        if shutdown.requested and cycles == 0:
            break

        cycles += 1
        # THE CYCLE TOKEN, AND IT IS MINTED HERE RATHER THAN IN
        # `process_bar_cycle`. That method runs once per TIMEFRAME BUCKET
        # against ONE shared position book, so a token minted per bucket would
        # reset between the 15m bucket's exit and the 30m bucket's re-entry -
        # which is exactly the pair the cooldown exists to catch. Declaring it
        # here is what makes "the same cycle" mean the same 60 seconds the
        # operator sees on the console.
        dispatcher.positions.begin_cycle(cycles)

        # RECONCILE THE BOOK AGAINST THE BROKER, BEFORE ANY SIGNAL IS READ.
        # `PositionBook` holds what THIS PROCESS opened, so after a restart it
        # is empty while the account still holds inventory - and
        # `plan_exits`' first condition is `is_open`, so every stop the
        # strategies computed was dropped with "position state is not known
        # here". Reconciling first is what lets the loop close what it did not
        # open.
        #
        # EVERY CYCLE, not once at startup: a position closed by hand, by a
        # bracket or by the prop-firm layer between cycles has to be dropped
        # from the book too, or the loop keeps trying to flatten something
        # that is already gone.
        if getattr(args, "reconcile_positions", True):
            try:
                snapshot = load_snapshot(args.positions_snapshot)
                outcome = reconcile(dispatcher.positions, snapshot,
                                    dispatcher.account_for,
                                    dispatcher.portfolios,
                                    state=getattr(dispatcher, "state", None))
                # Printed only when it CHANGED something, or on the first
                # cycle. A reconciliation that confirmed the book is the
                # ordinary case and printing it every 60s would bury the one
                # that adopted a position.
                moved = any(outcome[k] for k in
                            ("adopted", "conflicts", "closed_elsewhere",
                             "skipped"))
                if moved or cycles == 1:
                    print(describe_positions(outcome), flush=True)
            except PositionSnapshotError as exc:
                # A publisher that is RUNNING and wrong is a fact about the
                # account. Reading it as "no positions" would be the loop
                # deciding it is flat because a file was malformed, so the
                # cycle proceeds on the unreconciled book and says so.
                # LOUD, BUT NOT A RUN FAILURE. `failures` becomes the
                # process's exit code, and it is CUMULATIVE over the run - one
                # refusal at hour one makes the SIGTERM at hour twelve exit 1
                # and systemd report the shutdown as FAILURE. A stale snapshot
                # is a degraded INPUT the loop is designed to handle, not a
                # fault in the run: the book stays unreconciled, which is the
                # state it was in before this feed existed. stderr and the
                # watchdog are the right channel; the exit code is not.
                print(f"[master_live] position snapshot REFUSED: {exc}",
                      file=sys.stderr, flush=True)

        # THE STALENESS GUARD. A regime reading is a PERMISSION, and one
        # granted on a snapshot nobody has refreshed is a permission for a
        # market that has since moved. The loop reads the file the daemon
        # publishes and cannot tell a quiet tape from a dead publisher by
        # looking at the quadrant - only the write clock separates them.
        #
        # It stands the WHOLE CYCLE down rather than declining per symbol:
        # every symbol reads the same file, so a stale file is not a fact
        # about one contract. Exits are unaffected - nothing here closes a
        # position, and `dispatch_exits` is not reached because no bars are
        # loaded, so open inventory keeps whatever brackets it already has.
        stale_limit = float(getattr(args, "max_regime_write_age_sec", 0) or 0)
        if stale_limit > 0:
            age, detail = regime_write_age(args.state_file)
            if age is None or age > stale_limit:
                # CRITICAL, on stderr, EVERY cycle it persists. Printed rather
                # than counted once: a guard that announced itself only on the
                # transition would be silent for the hours that matter, and
                # this line is what an operator greps for.
                print(f"[master_live] CRITICAL regime state is STALE — "
                      f"{detail}; limit {stale_limit:.0f}s. Standing ALL "
                      f"{len(dispatcher.strategies)} strategies down for this "
                      f"cycle: no entries, no exits, nothing sent. The "
                      f"publisher is trading-regime-daemon; check its timer.",
                      file=sys.stderr, flush=True)
                if args.once or shutdown.requested:
                    break
                shutdown.sleep(args.interval_sec)
                continue

        # ONE BUCKET PER BAR WIDTH THE ROSTER NEEDS, resolved every cycle
        # rather than once at startup: `active_strategies` can be edited under
        # a running loop, and a bucket list fixed at boot would keep feeding a
        # newly-promoted 3m strategy the hourly bars its certification says
        # nothing about.
        buckets = required_timeframes(dispatcher, args.tf)
        if len(buckets) > 1:
            print(f"[master_live] cycle {cycles}: {len(buckets)} timeframe "
                  f"bucket(s) — {', '.join(buckets)}", flush=True)
        cycle_bars = 0
        for bucket_tf in buckets:
            try:
                bars, sources = load_symbol_bars(symbols, bucket_tf,
                                                 args.lookback_bars, feed=feed)
            # HOW OLD IS THE NEWEST BAR. Printed every cycle, because the
            # whole failure this feed exists to end was a loop reporting "no
            # signal" against a tape that had stopped. A vendor publishing on
            # a lag, a feed that fell back to the lake and a market that is
            # simply closed all produce the same quiet console otherwise.
                newest = max((f["ts"].iloc[-1] for f in bars.values()
                               if len(f)), default=None)
                if newest is not None:
                    from realtime.feed import tf_delta
                    width = tf_delta(bucket_tf)
                # Measured from when the bar CLOSED, not when it opened. A bar
                # is stamped at its open, so an hourly bar is always at least
                # an hour "old" by that reading and a warning drawn on it would
                # fire on every healthy cycle - which is how an operator learns
                # to ignore the one line that matters.
                    since_close = (pd.Timestamp.now(tz="UTC")
                                   - (newest + width)).total_seconds()
                    behind = since_close > width.total_seconds()
                    print(f"[master_live] newest closed bar {newest} — closed "
                          f"{since_close / 60:.1f} min ago ({bucket_tf} bar = "
                          f"{width.total_seconds() / 60:.0f} min)"
                          + ("  <-- A WHOLE BAR BEHIND: feed lagging, stale, "
                             "or the market is shut" if behind else ""),
                          flush=True)

                aliased = {k: v for k, v in sources.items() if k != v}
                if aliased:
                    print(f"[master_live] bars sourced from the full-size "
                          f"tape: {aliased} (same price series, same tick "
                          f"size; orders are still for the micro and sized on "
                          f"its point value)", flush=True)
            except Exception as exc:
            # A feed failure must not end the loop: the next cycle may read
            # cleanly, and a process that exits on one bad read needs a
            # supervisor to do what a `continue` does here.
                failures += 1
                print(f"[master_live] {bucket_tf} bar load failed: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr,
                      flush=True)
                bars = {}

            if not bars:
                continue
            cycle_bars += 1
            before = len(dispatcher.risk_refusals)
            # The bucket's width travels WITH its bars. `process_bar_cycle`
            # evaluates only the strategies certified on it and refuses any
            # that reach the gate anyway, so a 3m strategy can no longer be
            # handed hourly bars by a loop that loaded one width for everyone.
            report = dispatcher.process_bar_cycle(bars, timeframe=bucket_tf)
            print(dispatcher.describe_cycle(report), flush=True)
            for refusal in dispatcher.risk_refusals[before:]:
                # A risk refusal is not a decline and not an error: the
                # strategy asked, the gate said no, and the reason is the
                # thing an operator needs on the console rather than in a file.
                print(f"       RISK BLOCKED {refusal['symbol']} "
                      f"[{refusal['rule']}] {refusal['detail']}", flush=True)
                failures += 1
            if not report.get("ok", True):
                failures += 1

        if not cycle_bars:
            print(f"[master_live] no bars for {symbols} at "
                  f"{', '.join(buckets)}; nothing evaluated.", flush=True)

        if args.once or shutdown.requested:
            break
        shutdown.sleep(args.interval_sec)
        if shutdown.requested:
            break

    reason = (f"{closed_phase[0]}" if closed_phase
              else f"{shutdown.signal_name}" if shutdown.requested
              else "--once" if args.once else "loop ended")
    print(f"[master_live] stopped after {cycles} cycle(s) ({reason}). "
          f"No sockets are held between cycles, so there is nothing to drain.",
          flush=True)

    if closed_phase is not None:
        # THE CLOSE, in this order: say what is held, then release, then exit.
        # The report goes out FIRST because it is the part an operator needs
        # and `release_caches` — however defensively written — is the part
        # that touches other modules.
        print(market_close_report(dispatcher, engine_state, *closed_phase),
              flush=True)
        print(release_caches(), flush=True)
        if failures:
            print(f"[master_live] {failures} failure(s) were recorded during "
                  f"the session; the exit status reports the CLOSE, not them "
                  f"— they are in the log above and in master_live.err.",
                  flush=True)
        reopen = seconds_until_open(holidays=holidays)
        print(f"[master_live] exiting {args.closed_exit_code} so the "
              f"supervisor leaves this stopped. Next open in "
              f"{reopen / 3600:.1f}h; trading-master-live.timer brings it "
              f"back. Restarting it by hand before then is fine — it will "
              f"stand down again.", flush=True)
        return int(args.closed_exit_code)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
