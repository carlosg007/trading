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
from realtime.regime_reader import DEFAULT_STATE_FILE            # noqa: E402

DEFAULT_INTERVAL_S = 60


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dry_run = resolve_dry_run(args)
    except ValueError as exc:
        parser.error(str(exc))

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
    if not dry_run:
        print("[master_live] LIVE MODE — orders will be sent.", flush=True)

    symbols = basket_symbols(dispatcher)
    shutdown = ShutdownFlag()
    shutdown.install()

    cycles = failures = 0
    while True:
        cycles += 1
        # THE CYCLE TOKEN, AND IT IS MINTED HERE RATHER THAN IN
        # `process_bar_cycle`. That method runs once per TIMEFRAME BUCKET
        # against ONE shared position book, so a token minted per bucket would
        # reset between the 15m bucket's exit and the 30m bucket's re-entry -
        # which is exactly the pair the cooldown exists to catch. Declaring it
        # here is what makes "the same cycle" mean the same 60 seconds the
        # operator sees on the console.
        dispatcher.positions.begin_cycle(cycles)
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

    reason = (f"{shutdown.signal_name}" if shutdown.requested
              else "--once" if args.once else "loop ended")
    print(f"[master_live] stopped after {cycles} cycle(s) ({reason}). "
          f"No sockets are held between cycles, so there is nothing to drain.",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
