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


def load_symbol_bars(symbols, tf: str,
                     lookback_bars: int) -> tuple[dict, dict]:
    """
    `({basket_symbol: bars}, {basket_symbol: which contract supplied them})`.

    THE LAKE IS A HISTORICAL STORE AND THIS IS THE SEAM WHERE A REAL FEED GOES.
    It is enough to exercise the whole pipeline against real bars and it is not
    a live feed: the newest bar it can return is the newest bar that has been
    ingested, which in this repository is a batch process. Everything
    downstream takes `{symbol: DataFrame}` and does not care where the frames
    came from, so replacing this one function is the whole integration.

    **The bars MUST be closed bars.** The engine fills at the next bar's open,
    so acting on the last closed bar with a market order now is that fill;
    acting on a bar still forming is lookahead. A real feed handed in here has
    to drop its forming bar - the lake has no partial bar to drop, so nothing
    is dropped, and the signal bar's timestamp travels onto every signal so the
    assumption is auditable rather than implicit.

    THE MICRO ALIAS, AND WHY IT IS NEEDED HERE AT ALL. The four baskets hold
    MNQ/MES/MCL/MGC; the lake holds only the full-size contracts. They quote
    the SAME price series at the SAME tick size - only the multiplier differs,
    and a multiplier appears in no indicator - so a micro's bars are read from
    its parent. This is the third use of `THETA_ANCHOR_ALIAS`, whose tick sizes
    are reconciled against `backtest/specs.py` on every daemon construction.

    The substitution is REPORTED rather than silent: the order is still for the
    micro, sized on the micro's own point value, and an operator has to be able
    to see that its signal came from the full-size tape.
    """
    from mdlib.lake import iter_bars
    from realtime.regime_daemon import THETA_ANCHOR_ALIAS

    wanted = {sym: THETA_ANCHOR_ALIAS.get(sym, sym) for sym in symbols}
    frames = {}
    for source_symbol, df in iter_bars(sorted(set(wanted.values())), tf,
                                       None, None):
        if df is not None and not df.empty:
            frames[source_symbol] = df

    bars, sources = {}, {}
    for symbol, source_symbol in wanted.items():
        df = frames.get(source_symbol)
        if df is None:
            continue
        bars[symbol] = df.tail(int(lookback_bars)).reset_index(drop=True)
        sources[symbol] = source_symbol
    return bars, sources


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

    try:
        dispatcher = LiveExecutionDispatcher(
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

    print(dispatcher.describe(), flush=True)
    if not dry_run:
        print("[master_live] LIVE MODE — orders will be sent.", flush=True)

    symbols = basket_symbols(dispatcher)
    shutdown = ShutdownFlag()
    shutdown.install()

    cycles = failures = 0
    while True:
        cycles += 1
        try:
            bars, sources = load_symbol_bars(symbols, args.tf,
                                             args.lookback_bars)
            aliased = {k: v for k, v in sources.items() if k != v}
            if aliased:
                print(f"[master_live] bars sourced from the full-size tape: "
                      f"{aliased} (same price series, same tick size; orders "
                      f"are still for the micro and sized on its point value)",
                      flush=True)
        except Exception as exc:
            # A feed failure must not end the loop: the next cycle may read
            # cleanly, and a process that exits on one bad read needs a
            # supervisor to do what a `continue` does here.
            failures += 1
            print(f"[master_live] bar load failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            bars = {}

        if bars:
            report = dispatcher.process_bar_cycle(bars)
            print(dispatcher.describe_cycle(report), flush=True)
            if not report.get("ok", True):
                failures += 1
        else:
            print(f"[master_live] no bars for {symbols}; nothing evaluated.",
                  flush=True)

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
