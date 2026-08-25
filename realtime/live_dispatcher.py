"""
realtime.live_dispatcher - the end-to-end live execution loop.

Wires four subsystems that were built standalone into one bar cycle:

    strategy module  ->  signal on the last CLOSED bar
    regime_reader    ->  is this symbol's live quadrant one this basket trades?
    regime_daemon    ->  does the ML confirmation model permit this entry?
    PortfolioManager ->  net the signals, size them from ATR, route to an account
    crosstrade_formatter + live.dispatcher  ->  the wire, or a dry-run log

THE PIPELINE OPENS POSITIONS AND CANNOT CLOSE THEM
==================================================
Read this before deploying anything here. `PortfolioManager` does not know what
is open - it never emits FLATTEN, and nothing in this module invents the
position state it would need to. So an EXIT signal on the last bar is COUNTED
AND REPORTED (`exit_signals` on the cycle report) and is not turned into an
order. A loop run without something reconciling positions on the CrossTrade
side accumulates entries and never leaves.

That is a deliberate stopping point rather than an oversight: closing a
position this process cannot see would shut positions it never opened, and the
failure is silent in exactly the direction that costs money. `format_flatten_command`
exists in the formatter for whoever owns that reconciliation.

WHERE THE REGIME IS CHECKED, AND WHY IT IS CHECKED TWICE
========================================================
Once here, per (strategy, symbol), before a signal is emitted at all - so a
decline is recorded against the STRATEGY, with its reason, rather than
disappearing into an empty payload list. Once again inside
`PortfolioManager.build_order_plan`, which is the authoritative gate between a
net position and an order.

They cannot disagree: both read the same `data/live_regime_state.json` through
the same reader, and both compare against the same
`portfolio["derived"]["canonical_quadrants"]`. The early check is an
observability layer over the late one, never a substitute - if it were removed,
every order would still be gated correctly.

**A regime decline does not change the net.** The gate is per (portfolio,
symbol) and every strategy trading that symbol in that portfolio gets the same
verdict, so declining cannot leave one side of an opposing pair standing.
**An ML veto DOES change the net**, and that is intended: a vetoed entry is not
a signal. One consequence is worth stating because it is not obvious - vetoing
one side of an opposing pair turns a position that would have netted flat into
a live order. `ml_vetoes` on the report is what makes that visible.

WHAT DECIDES A STRATEGY MAY TRADE
=================================
`active_strategies` in `config/portfolios.json` grants PERMISSION; the
`strategies/approved_incubator/<id>/` directory supplies the CODE. Being in the
directory is explicitly not permission to trade (see its README), and a
strategy named in the config with no directory is an error rather than a skip.

**The promoted code is hash-checked before it is run.** `meta.json` records the
SHA-256 of the file that was backtested, and this module refuses to trade a
`strat.py` that no longer matches it. That hash exists precisely so a promoted
file provably IS the file the metrics describe; a live loop that ignored it
would be trading an edited strategy under a certified strategy's name.

**The certified symbols are checked too**, through the shared micro/full-size
table in `realtime/contract_alias.py` (`MNQ`->`NQ`) - the same one the regime
reader resolves a micro's quadrant with, so a certification and a regime
reading can never disagree about which contract a symbol means. A strategy
certified on ZS routed into a basket holding MNQ is refused: same price series
is a defensible alias, same portfolio is not.

THE STOP MULTIPLIER, WHEN CONTRIBUTORS DISAGREE
===============================================
`PortfolioManager` sizes from `current_regimes[symbol]["stop_atr_mult"]` - one
value per symbol - while `sl_atr_mult` belongs to a STRATEGY. Two strategies
netting into one MNQ position can declare different stops.

The plan is built PER PORTFOLIO so the mapping never has to describe two
baskets at once, and within a portfolio the WIDEST stop wins. Sizing on the
tightest stop would over-size the position relative to the strategy holding the
wide one and breach its risk budget; the widest under-sizes the tight-stop
strategy, which errs toward less risk. The disagreement is never silent -
`stop_atr_mult_source` records the value, the strategy it came from, and every
value that was in contention.

RETRIES ARE NARROW ON PURPOSE
=============================
A retried market order is a DUPLICATE POSITION, and duplicates are not
recoverable by this process. So a retry happens only when the error proves
nothing reached the broker - a refused connection, a DNS failure, no route to
host, all of which occur before a byte of the request is written.

**A timeout is never retried, and neither is a 5xx.** Both leave the outcome
unknown: the order may be live. Reporting one uncertain send is strictly better
than turning it into two positions. `dispatch_order` returns the full attempt
record either way, so the ambiguity is on the record rather than resolved by a
guess.

NOTHING HERE OPENS A SOCKET
===========================
`live/dispatcher.py::send_execution_signal` remains the only thing in this
repository that sends an order. This module calls it, retries it under the rule
above, and injects a substitute only when a caller passes one (which is how the
tests avoid the network).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.tier3_workers import load_strategy                     # noqa: E402
from backtest.engine import unpack_signals                         # noqa: E402
from live.dispatcher import send_execution_signal                  # noqa: E402
from portfolio.config_loader import (DEFAULT_CONFIG_PATH,          # noqa: E402
                                     PortfolioConfigError,
                                     load_portfolio_config)
from portfolio.portfolio_manager import (PositionBook,             # noqa: E402
                                        LONG, SHORT, FLAT,
                                         PortfolioError,
                                         PortfolioManager)
from realtime.crosstrade_formatter import (                        # noqa: E402
    CrossTradeFormatError,
    format_crosstrade_command,
    format_crosstrade_json,
    format_flatten_command,
    format_flatten_json,
    redact,
)
from realtime.contract_alias import resolve_parent                 # noqa: E402
from realtime.regime_daemon import (MasterRegimeDaemon,            # noqa: E402
                                    MLGateError)
from realtime import regime_reader                                # noqa: E402
from realtime.risk_firewall import RiskViolation                   # noqa: E402
from realtime.regime_reader import (DEFAULT_STATE_FILE,            # noqa: E402
                                    RegimeStateError,
                                    get_current_regime)

# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------
DEFAULT_STRATEGY_ROOT = "strategies/approved_incubator"
DEFAULT_MODEL_DIR = "models/"
DEFAULT_ENV_FILE = ".env"

# The two names the specification gives the credentials. The formatter's own
# fallback variable is `CROSSTRADE_KEY`; this module passes the key EXPLICITLY
# to every format call rather than relying on that fallback, so the two spellings
# can never silently resolve to different keys.
ENV_URL = "CROSSTRADE_WEBHOOK_URL"
ENV_KEY = "CROSSTRADE_API_KEY"

# One send is 2.0s at most, matching `live/config.json`'s hard timeout.
DEFAULT_TIMEOUT_S = 2.0

# Attempts for the narrow class of failures that prove nothing was sent.
DEFAULT_MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 0.5

# Error fragments that mean the connection was never established. Matched
# against `send_execution_signal`'s `error` string, which is the only channel
# it reports through. Anything not on this list is NOT retried - see the module
# docstring; the default has to be "do not send it again".
_NEVER_SENT_MARKERS = (
    "connection refused",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname provided",
    "no route to host",
    "network is unreachable",
)


class LiveDispatchError(RuntimeError):
    """The live loop cannot be configured or cannot route what it was given."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
def load_env_file(path: str | Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    """
    A minimal `.env` reader: `KEY=value` per line, `#` comments, optional
    `export ` prefix, optional surrounding quotes.

    Hand-rolled rather than adding `python-dotenv`, because `requirements.txt`
    is pinned deliberately in this repository and a new runtime dependency for
    fifteen lines of parsing is not a trade worth making.

    It does NOT put anything into `os.environ`. A credential loaded into the
    process environment is inherited by every subprocess this loop ever spawns,
    which is how a webhook key ends up in an unrelated tool's debug output.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def resolve_credentials(crosstrade_url: str | None = None,
                        crosstrade_key: str | None = None,
                        env_file: str | Path = DEFAULT_ENV_FILE
                        ) -> tuple[str, str, str]:
    """
    `(url, key, source)` - explicit argument, then `.env`, then the process
    environment.

    Returns empty strings rather than raising: a dry run is a legitimate way to
    use this module with no credentials at all, and refusing to construct would
    make the safe mode the one that needs a live webhook configured. The
    refusal belongs at the point a LIVE dispatcher is built - see
    `LiveExecutionDispatcher.__init__`.
    """
    env = load_env_file(env_file)
    sources = []

    url = (crosstrade_url or "").strip()
    if url:
        sources.append("url:argument")
    else:
        url = (env.get(ENV_URL) or os.environ.get(ENV_URL) or "").strip()
        if url:
            sources.append(f"url:{ENV_URL}")

    key = (crosstrade_key or "").strip()
    if key:
        sources.append("key:argument")
    else:
        key = (env.get(ENV_KEY) or os.environ.get(ENV_KEY) or "").strip()
        if key:
            sources.append(f"key:{ENV_KEY}")

    return url, key, ", ".join(sources) or "none"


# --------------------------------------------------------------------------
# strategy handles
# --------------------------------------------------------------------------
class StrategyHandle:
    """
    One promoted strategy, bound to one portfolio, ready to be asked for a
    signal.

    Built once per cycle-loop rather than per bar: `load_strategy` executes the
    module, and re-importing it every minute would be both wasteful and a way
    for the code under a running loop to change without anybody deciding it
    should.
    """

    def __init__(self, strategy_id: str, portfolio_id: str, account: str,
                 signal_fn: Callable, module_info: dict, meta: dict,
                 directory: Path) -> None:
        self.strategy_id = strategy_id
        self.portfolio_id = portfolio_id
        self.account = account
        self.signal_fn = signal_fn
        self.module_info = module_info
        self.meta = meta
        self.directory = directory
        risk = meta.get("risk") or {}
        self.sl_atr_mult = _risk_value(risk.get("sl_atr_mult"))
        self.tp_atr_mult = _risk_value(risk.get("tp_atr_mult"))
        self.certified_symbols = tuple(meta.get("symbols") or ())

    def trades_symbol(self, symbol: str) -> tuple[bool, str]:
        """
        Whether this strategy's certification covers `symbol`.

        Resolved through `realtime/contract_alias.py`, the one micro/full-size
        table the regime daemon, the reader and the portfolio loader all use:
        MNQ and NQ are the same price series at the same tick size, so a
        certification on ['NQ'] authorizes trading MNQ, and one on ['MNQ']
        authorizes NQ. Anything else is refused - a strategy certified on
        soybeans has no evidence about a Nasdaq micro, and the only thing that
        put them together is a line in a config file.

        A strategy declaring NO symbols is allowed through with a note. Several
        pre-existing modules predate the field, and refusing them would make
        this loop unusable with the strategies the repository actually has.
        """
        if not self.certified_symbols:
            return True, "meta.json declares no `symbols`; certification scope unknown"
        sym = str(symbol).upper()
        parent = resolve_parent(sym)
        for certified in self.certified_symbols:
            c = str(certified).upper()
            if c == sym:
                return True, f"certified on {c}"
            if resolve_parent(c) == parent:
                return True, f"certified on {c}, same price series as {sym}"
        return False, (f"certified on {list(self.certified_symbols)}, which "
                       f"does not cover {sym} (nor as a micro of it)")


def _risk_value(value):
    """
    `meta.json`'s three-state risk field, preserved.

    `"NOT DECLARED"` means the strategy has no such parameter; `None` means it
    has one and the promoted run modelled it OFF. Both come back as None here
    because neither gives a multiplier - but they are NOT the same statement,
    so the raw token is kept on `meta` for anything that reports it.
    """
    if value is None or value == "NOT DECLARED":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# the dispatcher
# --------------------------------------------------------------------------
class LiveExecutionDispatcher:
    """
    One bar cycle: signals in, CrossTrade orders out, with every decline on the
    record.

    Construct once and call `process_bar_cycle` per bar close. Strategy modules
    are loaded at construction; the regime state is re-read every cycle,
    because that is the thing that changes.
    """

    def __init__(self,
                 config_path: str = DEFAULT_CONFIG_PATH,
                 state_file: str = DEFAULT_STATE_FILE,
                 crosstrade_url: str | None = None,
                 crosstrade_key: str | None = None,
                 dry_run: bool = False,
                 strategy_root: str = DEFAULT_STRATEGY_ROOT,
                 ml_model_dir: str = DEFAULT_MODEL_DIR,
                 env_file: str | Path = DEFAULT_ENV_FILE,
                 max_age_s: float | None = None,
                 timeout_seconds: float = DEFAULT_TIMEOUT_S,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 sender: Callable | None = None,
                 verify_code_hash: bool = True,
                 firewall: Any | None = None,
                 state: Any | None = None) -> None:
        self.config_path = config_path
        self.state_file = state_file
        self.dry_run = bool(dry_run)
        self.max_age_s = max_age_s
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = max(1, int(max_attempts))
        self.verify_code_hash = bool(verify_code_hash)
        # The default IS `live.dispatcher`. An injected sender is how the tests
        # stay off the network; it is not an alternative transport.
        self._sender = sender or send_execution_signal

        # The pre-trade gate and the durable record. Both default to None so a
        # test or a one-off script constructs a dispatcher exactly as before;
        # `master_live.py` supplies them, and that is the path that sends
        # orders.
        self.firewall = firewall
        self.state = state
        self.risk_refusals: list[dict[str, Any]] = []

        self.config = load_portfolio_config(config_path)
        self.portfolios = self.config["portfolios"]
        self.manager = PortfolioManager(config_path, config=self.config)
        # What THIS PROCESS opened. Not persisted: a restart begins believing
        # it holds nothing, which is the safe direction to be wrong in - the
        # loop then declines to flatten a position it cannot vouch for rather
        # than flattening one it has no record of opening. See PositionBook.
        self.positions = PositionBook()

        self.strategy_root = (Path(strategy_root) if Path(strategy_root).is_absolute()
                              else REPO_ROOT / strategy_root)

        self.crosstrade_url, self.crosstrade_key, self.credential_source = \
            resolve_credentials(crosstrade_url, crosstrade_key, env_file)

        # A LIVE dispatcher with nowhere to send is refused HERE, not at the
        # first order. Discovering it mid-cycle means the signals, the gates
        # and the sizing all ran and produced nothing, which reads exactly like
        # a quiet market.
        if not self.dry_run and not self.crosstrade_url:
            raise LiveDispatchError(
                f"live dispatch requested but no CrossTrade webhook URL is "
                f"configured. Set {ENV_URL} in {env_file} (and {ENV_KEY}), pass "
                f"crosstrade_url=, or run with dry_run=True. Refusing to start "
                f"a loop that would evaluate every gate and then silently send "
                f"nothing.")

        # The daemon is constructed for its ML gate alone; this loop never asks
        # it to classify. Classification belongs to the daemon PROCESS, which
        # publishes the state file this loop reads - two processes, one
        # direction, so a slow indicator pass can never stall a dispatch.
        self.daemon = MasterRegimeDaemon(
            config_path=config_path, state_file=state_file,
            ml_model_dir=ml_model_dir, strict_config=False)

        self.strategies: list[StrategyHandle] = []
        self.strategy_errors: list[dict] = []
        self._load_active_strategies()

    # -- strategies -------------------------------------------------------
    def _load_active_strategies(self) -> None:
        """
        One handle per (portfolio, strategy) named in `active_strategies`.

        A strategy on two portfolios of different tracks gets TWO handles, which
        is the documented normal case: the same edge incubating and evaluating
        at once. Each handle carries its own account, so the two never share an
        order.

        A named strategy that cannot be loaded is recorded in
        `strategy_errors`, never skipped silently: a live loop quietly running
        three of the four strategies somebody assigned is the failure that gets
        noticed at the end of the month.
        """
        for pid in sorted(self.portfolios):
            portfolio = self.portfolios[pid]
            for strategy_id in (portfolio.get("active_strategies") or []):
                try:
                    self.strategies.append(
                        self._build_handle(str(strategy_id), pid, portfolio))
                except Exception as exc:
                    self.strategy_errors.append({
                        "strategy_id": str(strategy_id),
                        "portfolio_id": pid,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    def _build_handle(self, strategy_id: str, pid: str,
                      portfolio: dict) -> StrategyHandle:
        directory = self.strategy_root / strategy_id
        if not directory.is_dir():
            raise LiveDispatchError(
                f"{strategy_id!r} is in {pid}'s active_strategies but there is "
                f"no {directory}. The config grants permission; the incubator "
                f"directory supplies the code, and there is no fallback that "
                f"would not be guessing at which module was meant.")

        meta_path = directory / "meta.json"
        if not meta_path.is_file():
            raise LiveDispatchError(
                f"{directory} has no meta.json, so the parameters, the risk "
                f"block and the certified symbols are all unknown. An "
                f"unlabelled strategy is worse than a missing one.")
        meta = json.loads(meta_path.read_text())

        module_path = directory / "strat.py"
        if not module_path.is_file():
            module_path = directory / "strategy.py"
        if not module_path.is_file():
            raise LiveDispatchError(
                f"{directory} holds neither strat.py nor strategy.py")

        if self.verify_code_hash:
            recorded = meta.get("promoted_sha256") or meta.get("source_sha256")
            if not recorded:
                raise LiveDispatchError(
                    f"{meta_path} records no promoted_sha256, so the code "
                    f"about to be traded cannot be shown to be the code that "
                    f"was backtested.")
            actual = _sha256_file(module_path)
            if actual != recorded:
                raise LiveDispatchError(
                    f"{module_path} does not match the SHA-256 in meta.json "
                    f"({actual[:12]} vs {str(recorded)[:12]}). That hash "
                    f"exists so a promoted file provably IS the file the "
                    f"metrics describe; trading an edited module under a "
                    f"certified name is the one thing it is there to stop.")

        signal_fn, module_info = load_strategy(module_path,
                                               params=meta.get("params") or {})
        return StrategyHandle(strategy_id, pid, portfolio["target_account"],
                              signal_fn, module_info, meta, directory)

    # -- regime -----------------------------------------------------------
    def read_regimes(self, symbols) -> tuple[dict, dict]:
        """
        `({symbol: reading}, {symbol: why it is missing})` from the live state
        file.

        Read once per cycle rather than per strategy, so every decision in one
        cycle is made against ONE snapshot. Re-reading per strategy would let
        the daemon publish mid-cycle and leave two strategies on the same
        symbol gated by different quadrants, with no record that they were.

        Keyed on the BASKET's symbol - the micro that will actually be traded.
        The reader resolves a micro to its full-size parent (MNQ -> NQ, the
        contract the daemon publishes) and stamps `resolved_symbol` on the
        reading, so the mapping is visible on every decline rather than being
        a lookup the operator has to do in their head.
        """
        readings, missing = {}, {}
        for symbol in sorted(set(symbols)):
            try:
                readings[symbol] = get_current_regime(
                    symbol, state_file=self.state_file,
                    max_age_s=self.max_age_s)
            except RegimeStateError as exc:
                missing[symbol] = str(exc)
        return readings, missing

    # -- signals ----------------------------------------------------------
    @staticmethod
    def direction_on_last_bar(signal_fn: Callable,
                              bars: pd.DataFrame) -> tuple[str, bool, dict]:
        """
        `(direction, exit_signalled, detail)` for the LAST bar of `bars`.

        `bars` MUST be closed bars. The engine fills at the next bar's open, so
        acting on the last CLOSED bar with a market order now is that fill;
        acting on a bar still forming is lookahead, and it is lookahead that
        produces a live equity curve worse than the backtest for reasons nobody
        can find afterwards. This function cannot detect a forming bar - the
        caller owns the feed - so the bar's timestamp travels onto every signal.

        The three-way resolution mirrors `backtest.engine`'s state machine
        exactly: a bar signalling BOTH sides takes NEITHER. Guessing which side
        wins here would make the live loop trade a rule the backtest never
        simulated.
        """
        out = signal_fn(bars)
        n = len(bars)
        entries, exits, short_entries, short_exits = unpack_signals(out, n)

        long_in = bool(entries.iloc[-1])
        short_in = bool(short_entries.iloc[-1])
        exit_out = bool(exits.iloc[-1]) or bool(short_exits.iloc[-1])

        if long_in and short_in:
            direction = FLAT
            note = ("both sides signalled on the same bar; taking NEITHER, "
                    "exactly as backtest.engine resolves it")
        elif long_in:
            direction, note = LONG, "long entry"
        elif short_in:
            direction, note = SHORT, "short entry"
        else:
            direction, note = FLAT, "no entry"

        return direction, exit_out, {"note": note,
                                     "long_entry": long_in,
                                     "short_entry": short_in,
                                     "exit_signalled": exit_out}

    def _ml_features(self, handle: StrategyHandle,
                     bars: pd.DataFrame) -> dict:
        """
        The last row of the strategy's own `ml_features` matrix, as a dict.

        Built from the module's hook so the classifier sees the columns it was
        fitted on. A module declaring none returns an empty dict, which is
        correct in both directions: with no model registered the gate passes
        through, and with a model registered the gate RAISES for the missing
        features rather than inventing them.
        """
        fn = handle.module_info.get("ml_feature_fn")
        if fn is None:
            return {}
        frame = fn(bars)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise LiveDispatchError(
                f"{handle.strategy_id}: ml_features returned "
                f"{type(frame).__name__}, not a non-empty DataFrame")
        if len(frame) != len(bars):
            raise LiveDispatchError(
                f"{handle.strategy_id}: ml_features returned {len(frame)} rows "
                f"for {len(bars)} bars. A feature matrix out of step with the "
                f"bars is a row from the wrong moment.")
        return {str(k): v for k, v in frame.iloc[-1].to_dict().items()}

    # -- the cycle --------------------------------------------------------
    def process_bar_cycle(self, symbol_bar_map: dict) -> dict:
        """
        One full pass: signals -> regime gate -> ML gate -> netting -> sizing
        -> dispatch.

        `symbol_bar_map` is `{symbol: DataFrame}` of CLOSED bars per contract,
        oldest to newest - the shape `mdlib.lake.iter_bars` yields.

        Returns a report carrying, for every stage, what happened AND what did
        not:

            signals        the raw signals emitted, in the order emitted
            declines       every (strategy, symbol) that produced none, with why
            ml_vetoes      entries a registered model refused
            exit_signals   exits observed and NOT acted on - see the docstring
            plan           `PortfolioManager`'s decision per net position
            payloads       the subset that became orders
            dispatches     one attempt record per payload
            errors         anything that raised, per strategy

        and the four-key execution summary over them, every value derived from
        the records above rather than counted a second time:

            processed_at        when the cycle finished (== `finished_at`)
            evaluated_signals   every (strategy, symbol) actually evaluated
            approved_signals    the entries that cleared BOTH gates and
                                reached netting - FLAT records are not
                                approvals
            orders              the payloads that were dispatched (the same
                                list object as `payloads`)

        Nothing in the report is derived twice: `payloads` is taken from the
        plan, which is exactly what `build_order_payloads` returns for the same
        input.
        """
        cycle_started = time.perf_counter()
        report: dict[str, Any] = {
            "started_at": _utcnow(),
            "dry_run": self.dry_run,
            "symbols": sorted(symbol_bar_map),
            "signals": [], "declines": [], "ml_vetoes": [],
            "exit_signals": [], "exit_orders": [], "errors": [],
            "regime_readings": {}, "regime_missing": {},
            "plan": [], "payloads": [], "dispatches": [],
        }

        if not self.strategies:
            report["declines"].append({
                "strategy_id": None, "symbol": None,
                "reason": (f"no active strategies. `active_strategies` is "
                           f"empty on all four portfolios in "
                           f"{self.config_path}; nothing is permitted to "
                           f"trade until a human assigns one.")})

        readings, missing = self.read_regimes(symbol_bar_map)
        report["regime_readings"] = readings
        report["regime_missing"] = missing

        # ---- 1-3: signals, regime gate, ML gate -------------------------
        for handle in self.strategies:
            portfolio = self.portfolios[handle.portfolio_id]
            permitted = list(portfolio["derived"]["canonical_quadrants"])
            for symbol in portfolio["basket"]["assets"]:
                if symbol not in symbol_bar_map:
                    continue
                try:
                    self._evaluate(handle, symbol, symbol_bar_map[symbol],
                                   readings, missing, permitted, report)
                except Exception as exc:
                    report["errors"].append({
                        "strategy_id": handle.strategy_id,
                        "portfolio_id": handle.portfolio_id,
                        "symbol": symbol,
                        "error": f"{type(exc).__name__}: {exc}"})

        # ---- 4: netting and sizing, PER PORTFOLIO -----------------------
        report["plan"] = self._build_plan(report["signals"], readings, report)
        report["payloads"] = [r["payload"] for r in report["plan"]
                              if r["payload"] is not None]

        # ---- 5: dispatch ------------------------------------------------
        # The tag is passed BESIDE the payload rather than added to it: the
        # payload is `build_order_payloads`' exact five keys and must stay
        # byte-comparable with what the manager produces for the same input.
        for record in report["plan"]:
            if record["payload"] is not None:
                attempt = self.dispatch_order(
                    record["payload"],
                    strategy_tag=_strategy_tag(record),
                    bar_ts=str(record.get("bar_ts") or ""))
                report["dispatches"].append(attempt)
                # The book records a position only on a send that SUCCEEDED,
                # and only for a real side and size. Recording on intent would
                # leave this process believing it holds a position a refused
                # order never opened, and the next exit would then flatten an
                # account that is already flat - which on a shared account is
                # not a no-op.
                if attempt.get("ok"):
                    direction = (LONG if record["payload"]["action"] == "BUY"
                                 else SHORT)
                    try:
                        self.positions.record_fill(
                            record["portfolio_id"], record["symbol"],
                            direction, int(record["payload"]["quantity"]),
                            strategies=record.get("strategies"))
                    except PortfolioError as exc:
                        report["errors"].append({
                            "stage": "position_book",
                            "error": f"{type(exc).__name__}: {exc}"})

        # ---- 6: the exit actuator ---------------------------------------
        # AFTER the entries, and after the netting that produced them: an exit
        # is a flatten only when no strategy still claims that (portfolio,
        # symbol) this cycle, and `report["plan"]` is where that claim is
        # visible. Running this first would flatten a position another
        # strategy was about to be sized into.
        try:
            self.dispatch_exits(report, self._claims(report["plan"]))
        except Exception as exc:                                  # noqa: BLE001
            report["errors"].append({
                "stage": "dispatch_exits",
                "error": f"{type(exc).__name__}: {exc}"})

        report["ok"] = (all(d.get("ok") for d in report["dispatches"])
                        and all(e.get("ok", True) for e in report["exit_orders"]
                                if e.get("emitted")))
        report["elapsed_ms"] = round((time.perf_counter() - cycle_started) * 1000, 1)
        report["finished_at"] = _utcnow()

        # ---- the four-key execution summary ------------------------------
        # DERIVED from the stage records above, never counted a second time
        # while the stages run: a counter incremented alongside a list is one
        # early `return` away from disagreeing with the list it summarises,
        # and the summary is the half a caller reads.
        #
        # Every evaluation terminates in exactly ONE of declines / ml_vetoes /
        # signals, or raises into errors, so the sum double-counts nothing.
        # The "no active strategies" note carries no strategy_id and is not an
        # evaluation, so it is excluded rather than inflating the count.
        report["processed_at"] = report["finished_at"]
        report["evaluated_signals"] = (
            len(report["signals"])
            + len([d for d in report["declines"] if d.get("strategy_id")])
            + len(report["ml_vetoes"])
            + len(report["errors"]))
        # APPROVED is the entries that cleared both gates and reached netting.
        # A FLAT record is an evaluation that produced no entry - counting it
        # would report approvals on a cycle where no strategy wanted a
        # position.
        report["approved_signals"] = len(
            [s for s in report["signals"] if s["direction"] in (LONG, SHORT)])
        # The same list, not a copy: `orders` and `payloads` must not be able
        # to disagree about what was sent.
        report["orders"] = report["payloads"]
        return report

    def _evaluate(self, handle: StrategyHandle, symbol: str, bars,
                  readings: dict, missing: dict, permitted: list,
                  report: dict) -> None:
        """One (strategy, symbol): gate it, run it, gate it again, emit it."""
        def decline(reason: str, **extra) -> None:
            report["declines"].append({"strategy_id": handle.strategy_id,
                                       "portfolio_id": handle.portfolio_id,
                                       "symbol": symbol, "reason": reason,
                                       **extra})

        covered, why = handle.trades_symbol(symbol)
        if not covered:
            decline(f"certification does not cover this contract: {why}")
            return

        # BARS AND THE EXIT COME FIRST, BEFORE EVERY REGIME CHECK BELOW.
        # A muted strategy used to return at the regime decline, so its exit
        # was never computed - which is the whole reason a position could be
        # opened in a certified quadrant and then stranded the moment the
        # market left it. Muting exists to stop new ENTRIES; an open position
        # still has to be closeable, and a regime reading that is missing or
        # unfavourable is not a reason to keep holding one.
        if bars is None or len(bars) == 0:
            decline("no bars for this symbol in this cycle")
            return

        direction, exit_signalled, detail = self.direction_on_last_bar(
            handle.signal_fn, bars)
        bar_ts = str(pd.to_datetime(bars["ts"].iloc[-1], utc=True)
                     if "ts" in bars.columns else bars.index[-1])

        if exit_signalled:
            self._record_exit(handle, symbol, bar_ts, report)

        if symbol in missing:
            decline(f"no live regime reading: {missing[symbol]}")
            return

        reading = readings[symbol]
        quadrant = reading.get("quadrant")
        # Compared against the id AND the schema label, the same two spellings
        # `is_regime_permitted` accepts - `derived.canonical_quadrants` holds
        # ids, and a config edited to hold labels must not silently gate to
        # nothing.
        label = reading.get("regime")
        # Named as "MNQ (regime read from NQ)" when the alias was used: the
        # order is still for the micro, and an operator reading a stand-down
        # has to be able to see which tape the quadrant was measured on.
        measured = reading.get("resolved_symbol") or symbol
        named = (symbol if measured == symbol
                 else f"{symbol} (regime read from {measured})")
        if quadrant not in permitted and label not in permitted:
            decline(f"{named} is in {quadrant} ({label}); "
                    f"{handle.portfolio_id} trades {permitted}. Standing down "
                    f"rather than trading the environment nobody certified.",
                    quadrant=quadrant)
            return

        if direction in (LONG, SHORT):
            features = self._ml_features(handle, bars)
            try:
                confirmed = self.daemon.evaluate_ml_gate(
                    handle.strategy_id, symbol, features)
            except MLGateError as exc:
                # A broken or under-declared model is a REFUSAL to trade, not a
                # pass-through. The daemon already refuses to answer; the loop
                # records it as a decline rather than letting one strategy's
                # misconfigured model stop the cycle for the others.
                decline(f"ML gate could not be evaluated: {exc}")
                return
            if not confirmed:
                report["ml_vetoes"].append({
                    "strategy_id": handle.strategy_id, "symbol": symbol,
                    "portfolio_id": handle.portfolio_id,
                    "direction": direction, "bar_ts": bar_ts})
                return

        report["signals"].append({
            "strategy_id": handle.strategy_id,
            "portfolio_id": handle.portfolio_id,
            "symbol": symbol,
            "direction": direction,
            "action": {LONG: "BUY", SHORT: "SELL"}.get(direction, "FLAT"),
            "timestamp": _utcnow(),
            "bar_ts": bar_ts,
            "sl_atr_mult": handle.sl_atr_mult,
            "tp_atr_mult": handle.tp_atr_mult,
            "units": 1,
            "detail": detail,
        })

    @staticmethod
    def _claims(plan: list[dict]) -> dict:
        """
        Which (portfolio, symbol) pairs a strategy still wants a position in.

        Built from the PLAN rather than from the raw signals, because the plan
        is what survived netting, sizing and the regime gate. A signal that was
        declined is not a claim on the account, and treating it as one would
        block a flatten for a position nothing is going to hold.
        """
        claims: dict[str, dict] = {}
        for record in plan:
            if record.get("payload") is None:
                continue
            direction = (LONG if record["payload"]["action"] == "BUY"
                         else SHORT)
            claims.setdefault(record["portfolio_id"], {})[record["symbol"]] = {
                "direction": direction}
        return claims

    def _account_for(self, portfolio_id: str) -> str:
        """
        The broker account a portfolio's orders go to.

        `target_account` from the routing table - the SAME field
        `PortfolioManager.build_order_payloads` puts on every entry. Exits must
        land on the account the entries opened, and reading a different field
        here is how a flatten reaches the wrong account and leaves the position
        it was meant to close still open.
        """
        record = (self.config.get("portfolios") or {}).get(portfolio_id) or {}
        account = str(record.get("target_account") or "").strip()
        if not account:
            raise PortfolioError(
                f"portfolio {portfolio_id!r} declares no target_account, so "
                f"there is no account to flatten on")
        return account

    def _send_with_retry(self, record: dict, command: str,
                         started: float) -> dict:
        """
        The send loop, shared with `dispatch_order` so both obey ONE retry rule.

        A retried market order is a duplicate position this process cannot
        undo, so a retry happens only where the error proves nothing reached
        the broker. A flatten is idempotent and could bear a looser rule, but
        it does not get one: `_is_safe_to_retry` stays the single predicate,
        because two retry policies is how the stricter one quietly stops
        applying to the path that needed it.
        """
        body = format_flatten_json(account=record["account"],
                                   instrument=record["symbol"],
                                   strategy_tag=record.get("strategy_id") or "")
        for attempt in range(1, self.max_attempts + 1):
            record["attempts"] = attempt
            result = self._sender(body, webhook_url=self.crosstrade_url,
                                  timeout_seconds=self.timeout_seconds)
            record["result"] = {**result,
                                "url": _host_only(result.get("url", ""))}
            record["ok"] = bool(result.get("ok"))
            record["error"] = result.get("error")
            record["http_status"] = result.get("http_status")
            if record["ok"] or not _is_safe_to_retry(result):
                break
            if attempt < self.max_attempts:
                time.sleep(RETRY_BACKOFF_S * attempt)
        if not record["ok"] and record["attempts"] == 1 and record["error"]:
            record["retry_note"] = (
                "not retried: the failure does not prove the flatten never "
                "reached the broker, and this process cannot see whether the "
                "position is still open")
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return record

    def _record_exit(self, handle, symbol: str, bar_ts: str,
                     report: dict) -> None:
        """
        Record one exit, gated by the switchboard's own exit permission.

        `is_exit_permitted` is asked rather than assumed. It answers True for a
        MUTED strategy - muting blocks entries and never an exit - and the one
        case it refuses is a strategy the switchboard does not know, where
        "permitted" would be a guess about an account. The verdict travels on
        the record so a cycle report says WHY an exit did or did not become an
        order.
        """
        permitted, why = True, "switchboard not consulted"
        try:
            permitted = bool(regime_reader.is_exit_permitted(
                handle.strategy_id, state_file=self.state_file))
            why = ("switchboard permits exits"
                   if permitted else "switchboard refused the exit")
        except Exception as exc:                                  # noqa: BLE001
            # An unreadable switchboard must not strand an open position. The
            # refusal to invent state applies to OPENING, not to closing: the
            # position book below is what decides whether a flatten is real,
            # and it is this process's own record rather than a guess.
            permitted, why = True, (f"switchboard unreadable ({type(exc).__name__}"
                                    f": {exc}); exits are permitted by default")
        report["exit_signals"].append({
            "strategy_id": handle.strategy_id, "symbol": symbol,
            "portfolio_id": handle.portfolio_id, "bar_ts": bar_ts,
            "exit_permitted": permitted, "permission_reason": why})

    # -- the exit actuator --------------------------------------------------
    def dispatch_exits(self, report: dict, net_positions: dict | None = None
                       ) -> list[dict]:
        """
        Turn this cycle's exits into FLATTEN orders, and send them.

        THE ORDER OF THE THREE GATES IS THE SAFETY PROPERTY. An exit becomes an
        order only when the switchboard permitted it, this process has a
        RECORDED open position for that (portfolio, symbol), and no strategy
        still claims one there. `PositionBook.plan_exits` owns the last two;
        this method owns the first and the wire.

        Every refusal is reported with its reason. "no position this process
        opened" and "another strategy still holds it" are different facts about
        an account, and an exit that produced no order must never be
        indistinguishable from an exit that was never signalled.

        A FLATTEN carries no side and no quantity - see
        `crosstrade_formatter.format_flatten_command`. That is also why it is
        IDEMPOTENT in a way an entry is not: flattening an already-flat account
        is a no-op, while a duplicated entry is a second position this process
        cannot see. The retry rule is unchanged regardless, because a duplicate
        is not the only failure a retry can cause.
        """
        allowed = [e for e in report.get("exit_signals", [])
                   if e.get("exit_permitted")]
        for blocked in (e for e in report.get("exit_signals", [])
                        if not e.get("exit_permitted")):
            report["exit_orders"].append({
                **{k: blocked[k] for k in ("strategy_id", "symbol",
                                           "portfolio_id")},
                "emitted": False,
                "reason": blocked.get("permission_reason",
                                      "switchboard refused the exit")})

        intents = self.positions.plan_exits(allowed, net_positions)
        sent: list[dict] = []
        for intent in intents:
            if not intent["emit"]:
                report["exit_orders"].append({
                    "strategy_id": intent.get("strategy_id"),
                    "symbol": intent["symbol"],
                    "portfolio_id": intent["portfolio_id"],
                    "emitted": False, "reason": intent["reason"]})
                continue
            account = self._account_for(intent["portfolio_id"])
            record = self.dispatch_flatten(account, intent["symbol"],
                                           intent["portfolio_id"])
            record.update({"strategy_id": intent.get("strategy_id"),
                           "portfolio_id": intent["portfolio_id"],
                           "emitted": True, "reason": intent["reason"],
                           "held_direction": intent.get("held_direction"),
                           "held_quantity": intent.get("held_quantity")})
            # The book is cleared only on a SEND THAT SUCCEEDED. Clearing on
            # intent would leave this process believing it is flat while the
            # position is still open, and the next exit would then decline to
            # flatten the thing it failed to flatten.
            if record.get("ok"):
                self.positions.record_flat(intent["portfolio_id"],
                                           intent["symbol"])
            report["exit_orders"].append(record)
            sent.append(record)
        return sent

    def dispatch_flatten(self, account: str, symbol: str,
                         portfolio_id: str = "") -> dict:
        """
        Format one FLATTEN and put it on the wire.

        Built by `crosstrade_formatter.format_flatten_command`, sent by
        `live.dispatcher.send_execution_signal` - the same and only sender
        every entry goes through. `dry_run` formats and validates everything
        and opens no socket; the logged command is REDACTED, because the key is
        a bearer credential for a live account.
        """
        started = time.perf_counter()
        record: dict[str, Any] = {
            "timestamp": _utcnow(), "dry_run": self.dry_run,
            "account": account, "symbol": symbol, "action": "FLATTEN",
            "quantity": None, "attempts": 0, "ok": False, "error": None,
        }
        try:
            command = format_flatten_command(account=account,
                                             instrument=symbol,
                                             key=self.crosstrade_key)
        except Exception as exc:                                  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record
        record["command"] = redact(command)

        if self.dry_run:
            record.update({"ok": True, "attempts": 0,
                           "note": "dry run — formatted, validated, not sent"})
            return record
        return self._send_with_retry(record, command, started)

    # -- netting and sizing ------------------------------------------------
    def _build_plan(self, signals: list[dict], readings: dict,
                    report: dict) -> list[dict]:
        """
        Net and size, ONE PORTFOLIO AT A TIME.

        `build_order_plan` takes `current_regimes` keyed by symbol alone, so a
        single call cannot carry two portfolios' different stop multipliers for
        the same contract. Splitting the call is the alternative to either
        inventing a per-portfolio regime mapping the manager does not accept,
        or silently sizing one basket on the other's stop.
        """
        if not signals:
            return []
        try:
            net = self.manager.aggregate_signals(signals)
        except PortfolioError as exc:
            report["errors"].append({"stage": "aggregate_signals",
                                     "error": f"{type(exc).__name__}: {exc}"})
            return []

        plan: list[dict] = []
        for pid, book in net.items():
            scoped = {symbol: dict(self._regime_for(readings, symbol,
                                                    book[symbol], signals, pid))
                      for symbol in book if symbol in readings}
            try:
                plan.extend(self.manager.build_order_plan({pid: book}, scoped))
            except (PortfolioError, PortfolioConfigError) as exc:
                report["errors"].append({"stage": "build_order_plan",
                                         "portfolio_id": pid,
                                         "error": f"{type(exc).__name__}: {exc}"})
        return plan

    @staticmethod
    def _regime_for(readings: dict, symbol: str, net_record: dict,
                    signals: list[dict], pid: str) -> dict:
        """
        The regime reading `build_order_plan` sizes from, with this portfolio's
        stop multiplier attached.

        WIDEST STOP WINS when contributors disagree - see the module docstring.
        The losing values are kept on the record rather than discarded, because
        "sized on 2.0 while one contributor wanted 1.0" is a fact about the
        position that is invisible in the contract count.
        """
        reading = dict(readings[symbol])
        contributors = {c.get("strategy_id")
                        for c in net_record.get("contributors", [])}
        candidates = [(s["sl_atr_mult"], s["strategy_id"]) for s in signals
                      if s["portfolio_id"] == pid and s["symbol"] == symbol
                      and s["strategy_id"] in contributors
                      and s.get("sl_atr_mult") is not None]

        if candidates:
            mult, owner = max(candidates)
            reading["stop_atr_mult"] = float(mult)
            reading["stop_atr_mult_source"] = {
                "value": float(mult), "from": owner,
                "in_contention": sorted({c[0] for c in candidates}),
                "rule": ("widest stop wins; the tightest would over-size the "
                         "position relative to the strategy holding the wide "
                         "one and breach its risk budget"),
            }
        else:
            reading["stop_atr_mult_source"] = {
                "value": 1.0, "from": None, "in_contention": [],
                "rule": ("no contributor declared sl_atr_mult; the sizer's "
                         "1.0 default applies and the position is sized for a "
                         "stop no strategy stated"),
            }
        return reading

    # -- dispatch ---------------------------------------------------------
    def dispatch_order(self, payload: dict, strategy_tag: str = "",
                       bar_ts: str = "") -> dict:
        """
        Format one `PortfolioManager` payload for CrossTrade and send it.

        The payload arrives in `live.dispatcher.format_crosstrade_payload`'s
        five keys (`account`, `action`, `symbol`, `orderType`, `quantity`) and
        is translated - not reformatted - into the two CrossTrade wire forms by
        `realtime.crosstrade_formatter`. The translation is a field rename and
        nothing else; the formatter re-validates every value, so a malformed
        order is refused twice rather than once.

        Returns an attempt record. It never raises on a transport failure: one
        bad send must not end a cycle that has other orders to place, and the
        record IS the evidence of what was attempted.

        `dry_run` formats everything and opens no socket. The logged command is
        REDACTED - a dry run is the mode most likely to be pasted into a
        ticket, and the key is a bearer credential for a live account.

        RETURNS THE ATTEMPT RECORD RATHER THAN A BARE BOOL. `record["ok"]` is
        the boolean; everything beside it - the attempt count, the HTTP status,
        the error, whether a retry was declined and why - is what a bool throws
        away, and this is the layer where an order either happened or did not.
        """
        started = time.perf_counter()
        record: dict[str, Any] = {
            "timestamp": _utcnow(), "dry_run": self.dry_run,
            "account": payload.get("account"), "symbol": payload.get("symbol"),
            "action": payload.get("action"), "quantity": payload.get("quantity"),
            "attempts": 0, "ok": False, "error": None,
        }

        # THE PRE-TRADE GATE, INSIDE THE SEND PATH. Every order this repository
        # places goes through this method, so the firewall sits here rather
        # than in the caller: a gate a caller can forget is a gate that will be
        # forgotten the day somebody adds a second dispatch path in a hurry. A
        # refusal is a RECORD, not an exception thrown at the loop - one
        # blocked order must not end a cycle that has others to place, and the
        # record is the evidence that it was blocked and why.
        if self.firewall is not None:
            try:
                self.firewall.check_order(payload, strategy_tag=strategy_tag,
                                          bar_ts=bar_ts)
            except RiskViolation as exc:
                record.update({"ok": False, "blocked_by": exc.rule,
                               "error": f"RISK REFUSED [{exc.rule}] {exc.detail}",
                               "elapsed_ms": round(
                                   (time.perf_counter() - started) * 1000, 1)})
                self.risk_refusals.append(record)
                return record

        if self.state is not None and strategy_tag and bar_ts:
            # Recorded BEFORE the socket. If the process dies between the two,
            # the restart declines to re-send: a missed entry is a trade not
            # taken, a double entry is a position nothing here can unwind.
            self.state.record_dispatch(strategy_tag, payload.get("symbol", ""),
                                       bar_ts, payload)

        try:
            command = format_crosstrade_command(
                account=payload["account"],
                instrument=payload["symbol"],
                action=payload["action"],
                qty=payload["quantity"],
                order_type=payload.get("orderType", "MARKET"),
                key=self.crosstrade_key)
            body = format_crosstrade_json(
                account=payload["account"],
                instrument=payload["symbol"],
                action=payload["action"],
                qty=payload["quantity"],
                order_type=payload.get("orderType", "MARKET"),
                strategy_tag=strategy_tag)
        except (CrossTradeFormatError, KeyError) as exc:
            record["error"] = f"payload refused by the formatter: {exc}"
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return record

        record["command"] = redact(command)
        record["json"] = body

        if self.dry_run:
            record["ok"] = True
            record["note"] = "DRY RUN - formatted, nothing sent"
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return record

        for attempt in range(1, self.max_attempts + 1):
            record["attempts"] = attempt
            result = self._sender(body, webhook_url=self.crosstrade_url,
                                  timeout_seconds=self.timeout_seconds)
            # The sender echoes the full webhook URL back on every result, and
            # a webhook URL IS the credential - anyone holding it can place
            # orders on the account. The record is printed, logged and
            # serialised, so the URL is reduced to its host before it is kept.
            record["result"] = {**result, "url": _host_only(result.get("url", ""))}
            record["ok"] = bool(result.get("ok"))
            record["error"] = result.get("error")
            record["http_status"] = result.get("http_status")
            if record["ok"] or not _is_safe_to_retry(result):
                break
            if attempt < self.max_attempts:
                time.sleep(RETRY_BACKOFF_S * attempt)

        if not record["ok"] and record["attempts"] == 1 and record["error"]:
            record["retry_note"] = (
                "not retried: the failure does not prove the order never "
                "reached the broker, and a retried market order is a duplicate "
                "position this process cannot undo")

        if self.state is not None and strategy_tag and bar_ts:
            self.state.record_order(record, strategy=strategy_tag,
                                    bar_ts=bar_ts)

        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return record

    # -- reporting --------------------------------------------------------
    def describe_cycle(self, report: dict) -> str:
        """The console summary. Orders, declines and vetoes all visible."""
        mode = "DRY RUN" if report["dry_run"] else "LIVE"
        lines = [f"[{report['started_at']}] cycle {mode}  "
                 f"symbols={len(report['symbols'])}  "
                 f"strategies={len(self.strategies)}  "
                 f"{report.get('elapsed_ms', 0)}ms"]

        for d in report["dispatches"]:
            status = "OK  " if d["ok"] else "FAIL"
            lines.append(f"  {status} {d['account']:<16} {d['action']:<5} "
                         f"{d['symbol']:<5} x{d['quantity']}"
                         + (f"   {d['error']}" if d.get("error") else ""))
        if report["plan"] and not report["payloads"]:
            lines.append("  no orders — every net position was declined:")
        for r in report["plan"]:
            if r["payload"] is None:
                lines.append(f"       SKIP {r['account']:<16} {r['symbol']:<5} "
                             f"{r['skipped_reason']}")
        for v in report["ml_vetoes"]:
            lines.append(f"       VETO {v['strategy_id']} {v['symbol']} "
                         f"{v['direction']} — ML gate declined")
        for d in report["declines"]:
            lines.append(f"       HOLD {d.get('strategy_id')} "
                         f"{d.get('symbol')} — {d['reason']}")
        for e in report["errors"]:
            lines.append(f"       ERROR {e}")
        for s in report["exit_signals"]:
            lines.append(f"       EXIT {s['strategy_id']} {s['symbol']} — "
                         f"observed, NOT acted on (no position state here)")
        return "\n".join(lines)

    def describe(self) -> str:
        """What this dispatcher is wired to, printed before the loop starts."""
        lines = [
            f"LiveExecutionDispatcher  mode="
            f"{'DRY RUN' if self.dry_run else 'LIVE'}",
            f"  config      {self.config_path}",
            f"  state file  {self.state_file}",
            f"  strategies  {self.strategy_root}",
            f"  webhook     {_host_only(self.crosstrade_url)}  "
            f"(credentials from: {self.credential_source})",
            f"  key         {'set' if self.crosstrade_key else 'NOT SET'}",
        ]
        if not self.strategies:
            lines.append("  ACTIVE STRATEGIES: none — `active_strategies` is "
                         "empty, so this loop will place no orders.")
        for h in self.strategies:
            lines.append(f"  {h.strategy_id:<34} -> {h.portfolio_id:<15} "
                         f"({h.account})  sl={h.sl_atr_mult} tp={h.tp_atr_mult} "
                         f"certified={list(h.certified_symbols)}")
        for e in self.strategy_errors:
            lines.append(f"  FAILED TO LOAD {e['strategy_id']} "
                         f"({e['portfolio_id']}): {e['error']}")
        if self.daemon.model_errors:
            lines.append(f"  ML MODELS FAILED TO LOAD: "
                         f"{sorted(self.daemon.model_errors)}")
        return "\n".join(lines)


def _strategy_tag(plan_record: dict) -> str:
    """
    `portfolio:strategy_a+strategy_b` - what ties an NT8 fill back to what
    asked for it (`live.dispatcher.evaluate_incubator_sync` reconciles on it).

    A NETTED POSITION BELONGS TO EVERY CONTRIBUTOR, so all of them are named.
    Tagging it with one would attribute the whole position to a strategy that
    asked for part of it, and the reconciliation would then report the others
    as having placed nothing.
    """
    contributors = sorted({str(c.get("strategy_id"))
                           for c in plan_record.get("contributors", [])
                           if c.get("strategy_id")})
    tag = f"{plan_record.get('portfolio_id', '')}:{'+'.join(contributors)}"
    # The plain-text command is `;`/`=` delimited and the tag is free text.
    return tag.replace(";", "_").replace("=", "_")


def _is_safe_to_retry(result: dict) -> bool:
    """
    True only when the error proves the request never left this machine.

    Matched on the error string because that is `send_execution_signal`'s only
    channel. THE DEFAULT IS FALSE: an unrecognised error is not retried, so a
    new failure mode added upstream cannot silently become a duplicate order.
    """
    if result.get("http_status") is not None:
        return False                      # a response came back; it was sent
    error = str(result.get("error") or "").lower()
    if not error:
        return False
    return any(marker in error for marker in _NEVER_SENT_MARKERS)


def _host_only(url: str) -> str:
    """A webhook URL is a credential. Only its host is ever printed."""
    if not url:
        return "NOT SET"
    from urllib.parse import urlparse
    try:
        return urlparse(url).hostname or "unparseable"
    except ValueError:
        return "unparseable"


__all__ = ["LiveExecutionDispatcher", "StrategyHandle", "LiveDispatchError",
           "load_env_file", "resolve_credentials"]
