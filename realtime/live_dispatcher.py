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
from portfolio.portfolio_manager import (                          # noqa: E402
                                        LONG, SHORT, FLAT,
                                         PortfolioError,
                                         PortfolioManager)
# THE SUBCLASS, not `portfolio_manager`'s own. Same state, same key, same
# `record_fill`/`record_flat`/`plan_exits` - plus `can_execute`, the gate on
# the way IN. See `realtime/position_book.py` for why the gate had to live on
# the object the fill and the flatten already update rather than beside it.
from realtime.position_book import (MAX_QTY,                       # noqa: E402
                                    PositionBook,
                                    quantity_refusal)
from realtime.crosstrade_formatter import (                        # noqa: E402
    CrossTradeFormatError,
    format_crosstrade_command,
    format_crosstrade_json,
    format_account_flatten_command,
    format_flatten_command,
    format_flatten_json,
    redact,
    refusal_reason,
    sanitize_strategy_tag,
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
#: How a promoted `meta.json` spells the Stage 4.5 verdict. One constant,
#: because `backtest/promote.py` writes this key and this module reads it, and
#: a rename on one side alone silently turns every gate off: the block would
#: simply never be found, every log line would read correctly, and the strategy
#: would trade the session it was stood down from.
DOW_GATE_KEY = "day_of_week_gate"

#: What that block says when Stage 4.5 was never run for the pair. NOT an
#: empty list - "the stage looked and blocked nothing" and "nobody looked" are
#: different facts, and an empty list reads as the first.
DOW_NOT_EVALUATED = "NOT EVALUATED"


def named_weekdays(days) -> str:
    """
    `(4,)` -> `Friday`. Spelled out, because `4` in a log line is unreadable.

    The names are Monday-first to match `datetime.weekday()`, which is what
    `backtest.event_calendar.session_weekday` returns and what every
    `exclude_days` in this repository is written in. A Sunday-first table here
    would move every stand-down by a day while both files still parsed.
    """
    names = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday")
    return ", ".join(names[int(d)] for d in days) or "no weekday"


def session_weekday_for_fill(bars, timeframe: str | None = None
                             ) -> dict[str, Any]:
    """
    The CME session weekday an order placed on this cycle would FILL on.

    THE FILL BAR, NOT THE LAST CLOSED ONE, and the difference is the whole
    reason this is a function rather than a `.dayofweek`. The live loop acts
    on the last CLOSED bar; the engine fills at the NEXT bar's open, and
    `backtest.event_calendar._widen_to_fill_bar` blocks a signal on bar `i`
    whenever bar `i+1` falls inside the blocked window - precisely so the one
    entry signalled just before the window opens does not get through. A live
    gate keyed on the last closed bar would let exactly that trade out, on the
    last bar of Thursday's session filling into Friday's, which is both the
    least visible outcome and the one trade the block exists to stop.

    THE SESSION, NOT THE CALENDAR DAY. `backtest.event_calendar.session_date`
    is the one rule, imported rather than restated: CME runs each session to
    17:00 ET and reopens at 18:00, so any bar at or after 18:00 ET belongs to
    the NEXT day's session. There are no bars in the 17:00-18:00 maintenance
    break, so the 17:00 rollover an operator describes and the 18:00 rule the
    backtests were computed on select the same weekday for every bar that
    exists - and using the backtest's own function is what guarantees a live
    stand-down and a backtest exclusion can never disagree about which day it
    is.

    The bar width comes from `realtime.feed.tf_delta`, which reads
    `mdlib.lake.DERIVED` - so the width used here is the width the bar was
    built at. An UNKNOWN timeframe is reported as unknown rather than guessed:
    the caller decides what to do with that, and this module's answer is to
    let the strategy through (see `StrategyHandle.trades_session_weekday`).

    Returns a record rather than an int, because both weekdays belong on the
    cycle report: `last_bar_weekday` is what the strategy DECIDED on and
    `fill_weekday` is what it would trade, and an operator reading a
    stand-down at 17:55 on a Thursday has to be able to see which is which.
    """
    from realtime.feed import infer_timeframe, tf_delta            # noqa: PLC0415

    out: dict[str, Any] = {
        "last_bar_ts": None, "last_bar_weekday": None,
        "fill_bar_ts": None, "fill_weekday": None,
        "timeframe": timeframe, "resolved": False, "reason": "",
    }
    if bars is None or len(bars) == 0:
        out["reason"] = "no bars in this cycle"
        return out

    from backtest.event_calendar import session_weekday            # noqa: PLC0415

    last = pd.to_datetime(
        bars["ts"].iloc[-1] if "ts" in getattr(bars, "columns", [])
        else bars.index[-1], utc=True)
    out["last_bar_ts"] = str(last)
    out["last_bar_weekday"] = int(session_weekday([last])[0])

    tf = timeframe or infer_timeframe(bars)
    out["timeframe"] = tf
    if not tf:
        out["reason"] = ("the bar width could not be measured, so the fill "
                         "bar cannot be located")
        return out
    try:
        width = tf_delta(tf)
    except Exception as exc:                                      # noqa: BLE001
        out["reason"] = f"unknown timeframe {tf!r}: {exc}"
        return out

    fill = last + width
    out["fill_bar_ts"] = str(fill)
    out["fill_weekday"] = int(session_weekday([fill])[0])
    out["resolved"] = True
    return out


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
        # THE TIMEFRAME THE CERTIFICATION WAS MEASURED ON.
        #
        # Absent until 2026-08-27, and its absence was a live defect rather
        # than an omission: `master_live.py` loaded ONE timeframe and handed
        # those bars to every strategy, `trades_symbol` checked only the
        # symbol, and nothing else compared anything. A strategy swept,
        # plateau-selected and Gate-R certified on 3m bars was therefore
        # evaluated on 1h bars, producing signals no backtest ever simulated
        # against a certification describing a different tape - with every log
        # line reading correctly.
        #
        # `meta["timeframe"]` is written by `backtest/promote.py` and is the
        # authority. The id suffix is a FALLBACK for a meta.json written
        # before that key existed, and it is split with
        # `backtest.pipeline.split_strategy_id` rather than by counting
        # underscores: strategy names carry underscores and a date suffix, so
        # `ma_anchoring_spread_20260820` splits to a symbol of `spread` at a
        # timeframe of `20260820` under any left-to-right rule.
        self.certified_timeframe = self._resolve_timeframe(strategy_id, meta)
        # THE WEEKDAYS STAGE 4.5 STOOD THIS PAIR DOWN ON.
        #
        # Per STRATEGY, read off this package's own meta.json, and never a
        # process-wide setting. Sixteen strategies trade NQ inside
        # Incubator-Odd and they were profiled separately; one global blocked
        # weekday would stand fifteen of them down on a session their own
        # tables say they make money on, and the console would show a quiet
        # market rather than a policy.
        self.blocked_weekdays, self.dow_gate = self._resolve_blocked_weekdays(meta)

    @staticmethod
    def _resolve_blocked_weekdays(meta: dict) -> tuple[tuple[int, ...], dict]:
        """
        `meta["day_of_week_gate"]` -> the weekday integers, and the block itself.

        THREE STATES, KEPT APART, exactly as the `risk` block keeps its three:

            key absent, or "NOT EVALUATED"  Stage 4.5 never ran for this pair.
                                            Nothing is blocked and the record
                                            says WHY - which is not the same
                                            statement as a stage that ran and
                                            found no losing session.
            blocked_weekday: null           the stage ran and blocked nothing.
            blocked_weekday: 4              Friday entries are stood down.

        A value outside 0-6 RAISES rather than being dropped. Dropped, the gate
        is simply off and every log line reads correctly; raised, the package
        fails to load and somebody fixes the file. A weekday nobody can read is
        not a weekday to trade through.
        """
        block = meta.get(DOW_GATE_KEY)
        if not isinstance(block, dict):
            return (), {"status": DOW_NOT_EVALUATED,
                        "reason": ("meta.json carries no `day_of_week_gate`; "
                                   "Stage 4.5 did not run for this pair")}
        raw = block.get("blocked_weekdays")
        if raw is None:
            one = block.get("blocked_weekday")
            raw = [] if one is None else [one]
        if isinstance(raw, str):
            # "NOT EVALUATED" reaching the integer parse below would come out
            # as a TypeError three frames away from the file that caused it.
            return (), {**block, "status": str(raw)}
        days: list[int] = []
        for value in raw:
            d = int(value)
            if not 0 <= d <= 6:
                raise LiveDispatchError(
                    f"meta.json blocks weekday {value!r}; weekdays are 0-6 "
                    f"(Mon-Sun). Read as anything else this stands a strategy "
                    f"down on a day nobody profiled.")
            days.append(d)
        return tuple(sorted(set(days))), dict(block)

    def trades_session_weekday(self, session: dict | None
                               ) -> tuple[bool, str]:
        """
        Whether this strategy may open a position in the session an order
        would FILL into.

        The weekday twin of `trades_symbol` and `trades_timeframe`, and it
        returns the same `(permitted, why)` pair so the caller records one
        shape whichever scope refused.

        AN UNRESOLVED SESSION IS PERMITTED, with a note, and that is the
        deliberate direction. The gate needs the bar width to locate the fill
        bar; a cycle that delivered one bar, or a timeframe this repository
        cannot build, is a fact about the FRAME rather than evidence about the
        weekday, and standing a strategy down on it would mute the roster
        every time a feed hiccuped - a failure that looks exactly like a quiet
        market. The regime gate takes the opposite view because an unknown
        environment is genuinely not a permitted one; an unknown bar width is
        not an unknown weekday.
        """
        if not self.blocked_weekdays:
            return True, ""
        if not session or not session.get("resolved"):
            return True, (f"blocked on "
                          f"{named_weekdays(self.blocked_weekdays)}, but this "
                          f"cycle's fill session could not be resolved "
                          f"({(session or {}).get('reason', 'no session record')})")
        fill = int(session["fill_weekday"])
        if fill not in self.blocked_weekdays:
            return True, ""
        return False, (
            f"Stage 4.5 blocked {named_weekdays(self.blocked_weekdays)} for this "
            f"pair; an entry on this cycle fills into the "
            f"{named_weekdays((fill,))} session "
            f"(bar {session.get('fill_bar_ts')}). Entries are suppressed and "
            f"no order is built; exits are unaffected.")

    @staticmethod
    def _resolve_timeframe(strategy_id: str, meta: dict) -> str | None:
        declared = str(meta.get("timeframe") or "").strip().lower()
        if declared:
            return declared
        from backtest.pipeline import split_strategy_id          # noqa: PLC0415
        _strategy, _symbol, tf = split_strategy_id(strategy_id)
        return tf

    def trades_timeframe(self, timeframe: str | None) -> tuple[bool, str]:
        """
        Whether this strategy's certification covers `timeframe`.

        The timeframe twin of `trades_symbol`, and refused for the same
        reason: a certification is evidence about ONE contract at ONE bar
        width. Parameters fitted to 3m bars and a holdout measured on 3m bars
        say nothing about how the same rules behave hourly.

        A handle that declares NO timeframe is allowed through with a note,
        exactly as an undeclared symbol is - a meta.json written before the
        key existed is a legitimate state, and refusing it would strand every
        strategy promoted before this guard.

        An UNKNOWN incoming timeframe is also allowed through: the caller
        could not measure the bars, which is a fact about the frame rather
        than evidence of a mismatch, and refusing on it would stand a
        strategy down whenever a cycle delivered a single bar.
        """
        if not self.certified_timeframe:
            return True, ("meta.json declares no `timeframe`; certification "
                          "scope unknown")
        if not timeframe:
            return True, "the incoming bars have no measurable timeframe"
        if str(timeframe).strip().lower() == self.certified_timeframe:
            return True, ""
        return False, (f"certified on {self.certified_timeframe} bars, which "
                       f"does not cover {timeframe}")

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
                 max_qty: int = MAX_QTY,
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
        # THE HARD SIZE CAP. The default IS `MAX_QTY` and nothing in
        # `master_live.py` overrides it, so every dispatcher that trades gets
        # 1. It is a parameter only so a test can raise it and still exercise
        # the send path - a suite that had to keep every fixture under the cap
        # would be testing the cap in every case and the wire in none.
        self.max_qty = int(max_qty)
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
        # Entries the netted book refused as a stack. A SEPARATE LIST from
        # `risk_refusals`: a risk refusal is a limit being hit and wants an
        # operator's attention, while a HOLD is the gate working normally on
        # every cycle a position stays open. Folded together, the one that
        # matters would be buried under the one that does not.
        self.held_orders: list[dict[str, Any]] = []

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
    def process_bar_cycle(self, symbol_bar_map: dict,
                          timeframe: str | None = None) -> dict:
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

        # WHICH BAR WIDTH THESE ARE. Declared by the caller when it knows -
        # `master_live.py` loads one timeframe per bucket and says so - and
        # otherwise measured off the frames themselves, so a caller that
        # predates the parameter is still guarded rather than exempt.
        if timeframe is None:
            from realtime.feed import infer_timeframe              # noqa: PLC0415
            for frame in symbol_bar_map.values():
                timeframe = infer_timeframe(frame)
                if timeframe:
                    break

        report: dict[str, Any] = {
            "started_at": _utcnow(),
            "dry_run": self.dry_run,
            "timeframe": timeframe,
            "symbols": sorted(symbol_bar_map),
            "signals": [], "declines": [], "ml_vetoes": [],
            "exit_signals": [], "exit_orders": [], "errors": [],
            "regime_readings": {}, "regime_missing": {},
            "plan": [], "payloads": [], "dispatches": [],
            # Entries the netted book refused as a stack. On the report because
            # an order that was NOT sent has to be as visible as one that was:
            # a cycle that held everything and a cycle that signalled nothing
            # produce the same empty `dispatches`, and they are different facts
            # about the account.
            "held": [],
            # Declared here rather than grown by `setdefault` in the evaluator,
            # so a reader of this dict sees the full shape in one place and an
            # empty cycle carries the keys rather than omitting them.
            "indicators": [], "indicator_errors": [],
            "timeframe_skipped": [],
            # Strategies carrying a Stage 4.5 blocked weekday that were let
            # through because this cycle's fill session could not be resolved.
            # On the report because a gate that degraded to permissive and a
            # gate that had nothing to say produce the same absence of
            # declines, and they are different facts about the account.
            "dow_unresolved": [],
            # Per-symbol: which session an order placed on this cycle would
            # FILL into. Filled in immediately below.
            "sessions": {},
        }

        # WHICH SESSION AN ORDER FROM THIS CYCLE WOULD FILL INTO, per symbol.
        #
        # Computed ONCE per cycle rather than per (strategy, symbol): every
        # strategy in this bucket is handed the same frames, so a
        # per-strategy recomputation could only produce the same answer more
        # slowly - or a different one, if a cycle straddled a session roll
        # while it ran, which is the failure mode a shared record removes.
        #
        # PER SYMBOL and not once for the bucket, because the contracts do not
        # have to be in step: a lagging feed leaves one symbol's newest closed
        # bar an interval behind the others, and on a Friday afternoon that is
        # exactly the difference between filling into Friday's session and
        # into Monday's. Taking whichever frame happened to be first in the
        # map would give every other contract that symbol's weekday.
        #
        # Off the BARS and not `datetime.now()` for the same reason the loop
        # acts on the last CLOSED bar: a cycle delayed by a slow lake read
        # must not change which weekday it thinks it is.
        report["sessions"] = {
            sym: session_weekday_for_fill(frame, timeframe)
            for sym, frame in symbol_bar_map.items()}

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
        # A strategy certified on ANOTHER bar width is not evaluated in this
        # bucket, and that is a SKIP rather than a decline or an error: under
        # multi-timeframe dispatch most of the roster legitimately belongs to
        # a different bucket every cycle, and recording each as a refusal
        # would bury the ones that were actually refused.
        for handle in self.strategies:
            covers_tf, tf_why = handle.trades_timeframe(timeframe)
            if not covers_tf:
                report["timeframe_skipped"].append({
                    "strategy_id": handle.strategy_id,
                    "portfolio_id": handle.portfolio_id,
                    "certified_timeframe": handle.certified_timeframe,
                    "cycle_timeframe": timeframe, "reason": tf_why})
                continue
            portfolio = self.portfolios[handle.portfolio_id]
            permitted = list(portfolio["derived"]["canonical_quadrants"])
            for symbol in portfolio["basket"]["assets"]:
                if symbol not in symbol_bar_map:
                    continue
                try:
                    self._evaluate(handle, symbol, symbol_bar_map[symbol],
                                   readings, missing, permitted, report,
                                   bar_timeframe=timeframe)
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
                before_held = len(self.held_orders)
                attempt = self.dispatch_order(
                    record["payload"],
                    strategy_tag=_strategy_tag(record),
                    bar_ts=str(record.get("bar_ts") or ""),
                    portfolio_id=record.get("portfolio_id", ""))
                report["dispatches"].append(attempt)
                report["held"].extend(self.held_orders[before_held:])
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
                            strategies=_contributor_ids(record))
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

    def _record_telemetry(self, handle: StrategyHandle, symbol: str, bars,
                          bar_ts: str, direction: str, report: dict) -> None:
        """
        The strategy's OWN declared feature row, recorded for the operator.

        THE STRATEGY'S OWN, and that is the design rather than an
        implementation detail. There is no universal indicator set here:
        `double_rsi_macd_scalp` declares `rsi_fast`/`rsi_slow`/`macd_hist`,
        while `t3_braid_scalp` - the only strategy currently allocated -
        declares `t3_slope`/`braid_hist`/`stiffness`. A hardcoded "fast RSI and
        MACD histogram" line would print nothing at all for the one strategy
        actually running, so what is logged is whatever the module declares
        through `ml_features`.

        `indicators()` is deliberately NOT used: that hook returns the
        PRICE-SCALE series for the tear sheet, and its own docstring says the
        RSIs and the histogram are excluded on purpose because they cannot be
        drawn on a price axis. `ml_features` is where the module says those
        reach a reader.

        BEST EFFORT, AND THAT IS NOT LAZINESS. The strict `_ml_features` call
        in the gate path below must keep raising - a model asked to judge a
        signal on features that will not build has to refuse. This call is
        telemetry, and telemetry that can abort a cycle would trade a real
        decision for a log line. Failures are recorded and the cycle continues.

        Cost, measured on 400 bars of the allocated strategy: 2.4ms against the
        25.4ms `signal_fn` already spends per evaluation, so this is ~10% on a
        step that runs anyway. Not gated behind a flag at that price.
        """
        fn = handle.module_info.get("ml_feature_fn")
        if fn is None:
            # A module declaring no feature matrix has no indicators to
            # publish. Silence here is correct and is not an error.
            return
        try:
            frame = fn(bars)
            values = {str(k): v for k, v in frame.iloc[-1].to_dict().items()}
        except Exception as exc:                                  # noqa: BLE001
            report.setdefault("indicator_errors", []).append({
                "strategy_id": handle.strategy_id, "symbol": symbol,
                "error": f"{type(exc).__name__}: {exc}"})
            return
        report.setdefault("indicators", []).append({
            "strategy_id": handle.strategy_id,
            "portfolio_id": handle.portfolio_id,
            "symbol": symbol, "bar_ts": bar_ts, "direction": direction,
            "values": values})

    def _evaluate(self, handle: StrategyHandle, symbol: str, bars,
                  readings: dict, missing: dict, permitted: list,
                  report: dict, bar_timeframe: str | None = None) -> None:
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

        # THE BACKSTOP, AND IT RAISES ON PURPOSE.
        #
        # `process_bar_cycle` already filtered the roster to this bucket, so
        # a mismatched pair reaching here is a BUG in the dispatch above, not
        # a configuration a strategy can be stood down for. Declining would
        # log one more quiet HOLD in a stack whose whole failure mode is a
        # quiet HOLD that reads correctly; raising surfaces it as an error row
        # naming both timeframes, and the outer handler keeps the rest of the
        # cycle running.
        covers_tf, tf_why = handle.trades_timeframe(bar_timeframe)
        if not covers_tf:
            raise ValueError(
                f"[TIMEFRAME_MISMATCH] {handle.strategy_id} expects "
                f"{handle.certified_timeframe} bars, received "
                f"{bar_timeframe} ({tf_why})")

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

        # BEFORE the regime gate, so a stood-down strategy still reports what
        # it was looking at. The whole value of this line is on the bars where
        # nothing traded: "flat" and "flat because the fast RSI sat at 48 all
        # session" send an operator to different places.
        self._record_telemetry(handle, symbol, bars, bar_ts, direction, report)

        # ---- STAGE 4.5's DAY-OF-WEEK GATE ------------------------------
        #
        # AFTER the exit above and BEFORE every entry gate below. Both halves
        # are the contract: `_record_exit` has already run, so a position
        # opened on Thursday is still closeable on a blocked Friday - a gate
        # that muted the exit too would strand inventory, which is the
        # expensive direction to be wrong in - and nothing below this line
        # produces a signal, so no payload, no netting and no webhook exists
        # for this strategy on a blocked session.
        #
        # THE GATE IS PER STRATEGY AND IS APPLIED EXACTLY ONCE, HERE. The
        # regime gate is deliberately checked twice - once here and once
        # inside `build_order_plan` - and this one is NOT, because the two
        # gates have different granularity. A regime is a fact about a SYMBOL,
        # so the netted plan can re-check it and reach the same verdict. A
        # blocked weekday is a fact about ONE PROMOTED PAIR: sixteen
        # strategies net into one NQ position, and re-applying this rule at
        # the plan level would stand every contributor on that pair down on
        # one strategy's blocked day. That is the global lock this gate exists
        # to avoid.
        #
        # It is recorded as a DECLINE rather than as a new report bucket.
        # `evaluated_signals` is derived as signals + declines + ml_vetoes +
        # errors on the guarantee that every evaluation terminates in exactly
        # one of them; a fifth list would have to be added to that sum in the
        # same edit or the summary would silently undercount the roster. The
        # `rule` key is what makes it findable - "blocked_weekday" and "the
        # market left the quadrant" are different investigations.
        session = (report.get("sessions") or {}).get(symbol) or {}
        allowed_day, day_why = handle.trades_session_weekday(session)
        if not allowed_day:
            decline(day_why, rule="blocked_weekday",
                    blocked_weekdays=list(handle.blocked_weekdays),
                    fill_weekday=session.get("fill_weekday"),
                    fill_bar_ts=session.get("fill_bar_ts"),
                    last_bar_weekday=session.get("last_bar_weekday"))
            return
        if day_why:
            # PERMITTED, but not silently: this strategy HAS a blocked weekday
            # and the fill session could not be resolved, so it is trading on
            # the fallback rather than on a verdict. Recorded on the report
            # because a gate that quietly degraded to off is the one nobody
            # notices.
            report["dow_unresolved"].append({
                "strategy_id": handle.strategy_id,
                "portfolio_id": handle.portfolio_id,
                "symbol": symbol, "reason": day_why,
                "blocked_weekdays": list(handle.blocked_weekdays)})

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

    def account_for(self, portfolio_id: str) -> str:
        """
        The broker account a portfolio's orders go to.

        PUBLIC because `lifecycle.emergency_halt` has to ask it. The halt holds
        position records keyed by PORTFOLIO and has to flatten on an ACCOUNT,
        and a halt that read `portfolio_id` as an account name would send its
        kill command to an account that does not exist - failing at the one
        moment nothing else is going to stop the loop.

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

    def _safe_result(self, result: dict) -> dict:
        """
        A sender result with both credentials taken out of it.

        THE WEBHOOK URL IS A CREDENTIAL - anyone holding it can place orders on
        the account - and so is the `key=` field the text command carries. The
        sender echoes the URL back on every result and echoes the request
        payload back beside it, and an endpoint that quotes the request in its
        error hands the key straight into `response_body`. The record is
        printed, logged and serialised onward, so both are stripped HERE, in
        the one place every send path passes through, rather than at each
        place a record is displayed.
        """
        safe = {**result, "url": _host_only(result.get("url", ""))}
        for field in ("payload", "response_body"):
            value = safe.get(field)
            if isinstance(value, str):
                safe[field] = redact(value)
        return safe

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
        # THE TEXT COMMAND, exactly as `dispatch_order` sends for an entry -
        # and it is now the CALLER that has built it, because this loop has no
        # business knowing which kind of flatten it is carrying.
        #
        # It used to build a JSON object here and POST that instead, which is
        # the form the `/v1/send/` WEBHOOK does not parse: the same mismatch
        # that had every entry refused HTTP 400 on 2026-08-31, left in place on
        # the one path whose job is to CLOSE a position. A flatten refused by
        # the endpoint is an open position with a cycle report saying it was
        # sent, which is the direction of failure that costs money.
        for attempt in range(1, self.max_attempts + 1):
            record["attempts"] = attempt
            result = self._sender(command, webhook_url=self.crosstrade_url,
                                  timeout_seconds=self.timeout_seconds)
            # AND THE PAYLOAD IS NOW THE COMMAND, WHICH CARRIES `key=`. While
            # this loop posted a JSON body there was no credential in the echo
            # for it to strip; there is now.
            record["result"] = self._safe_result(result)
            record["ok"] = bool(result.get("ok"))
            record["error"] = result.get("error")
            record["http_status"] = result.get("http_status")
            # A 2xx IS NOT A FILL. CrossTrade accepts the webhook and then
            # declines the trade in the BODY - a strategy-lock refusal comes
            # back 200 - so an order it refused read `OK` everywhere. See
            # `crosstrade_formatter.refusal_reason`.
            refusal = refusal_reason(result.get("response_body"))
            if refusal is not None:
                record["ok"] = False
                record["refused_by_broker"] = refusal
                record["error"] = record["error"] or refusal
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
            account = self.account_for(intent["portfolio_id"])
            # THE EXIT CARRIES THE ENTRY'S TAG, rebuilt from the position
            # book - which is this process's record of WHO opened it - and not
            # from the exiting strategy's own id. A netted position opened by
            # two strategies took out ONE lock named for both; flattening it
            # under the name of whichever one signalled the exit presents
            # CrossTrade with a tag that locked nothing, and the position stays
            # open behind a flatten that was sent, accepted and logged. Read
            # BEFORE the send, because `record_flat` below forgets it.
            held = self.positions.get(intent["portfolio_id"],
                                      intent["symbol"]) or {}
            record = self.dispatch_flatten(
                account, intent["symbol"], intent["portfolio_id"],
                strategy_tag=compose_strategy_tag(intent["portfolio_id"],
                                                  held.get("strategies")))
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
                         portfolio_id: str = "",
                         strategy_tag: str = "") -> dict:
        """
        Format one FLATTEN and put it on the wire.

        Built by `crosstrade_formatter.format_flatten_command`, sent by
        `live.dispatcher.send_execution_signal` - the same and only sender
        every entry goes through. `dry_run` formats and validates everything
        and opens no socket; the logged command is REDACTED, because the key is
        a bearer credential for a live account.

        `strategy_tag` is the ATTRIBUTION tag - who opened the position, read
        off the position book by `dispatch_exits`. It is no longer what
        releases the lock: the wire carries `compose_wire_tag(portfolio_id,
        symbol)`, which is derivable from the pair alone and is therefore the
        same string the entry sent WHETHER OR NOT this process is the one that
        opened the position. That is what makes a reconciled flatten work after
        a restart, when the book is empty and the contributors are unknowable.

        It defaults to empty because this method is also the manual/operator
        entry point, where an untagged flatten (act on whatever the account
        holds) is the honest thing to send; `dispatch_exits` never leaves it
        empty. WITHOUT A `portfolio_id` there is no pair to key a lock on and
        the caller's tag goes out unchanged.
        """
        started = time.perf_counter()
        record: dict[str, Any] = {
            "timestamp": _utcnow(), "dry_run": self.dry_run,
            "account": account, "symbol": symbol, "action": "FLATTEN",
            "quantity": None, "attempts": 0, "ok": False, "error": None,
            # ON THE RECORD BEFORE THE SEND, and read back by
            # `_send_with_retry` for the JSON body - so the two wire forms of
            # one flatten cannot carry different tags.
            "strategy_tag": strategy_tag,
        }
        # THE LOCK RELEASE. Keyed on the pair, so it matches what the entry
        # took out even when this process cannot name who opened the position -
        # which is every flatten after a restart, and the case where a tag
        # mismatch is silent and expensive: the flatten is sent, accepted and
        # logged, and the position stays open.
        wire_tag = (compose_wire_tag(portfolio_id, symbol)
                    if portfolio_id else strategy_tag)
        record["wire_strategy_tag"] = wire_tag
        try:
            command = format_flatten_command(account=account,
                                             instrument=symbol,
                                             key=self.crosstrade_key,
                                             strategy_tag=wire_tag)
        except Exception as exc:                                  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record
        record["command"] = redact(command)
        # KEPT ON THE RECORD, NOT SENT - exactly as `dispatch_order` keeps its
        # own. It is the evidence of what the JSON endpoint would have been
        # given, and a switch back is one line. `strategy_tag` comes off the
        # record rather than from `strategy_id`, which `dispatch_exits`
        # attaches only AFTER this method returns.
        record["json"] = format_flatten_json(
            account=account, instrument=symbol,
            strategy_tag=record.get("wire_strategy_tag") or "")

        if self.dry_run:
            record.update({"ok": True, "attempts": 0,
                           "note": "dry run — formatted, validated, not sent"})
            return record
        return self._send_with_retry(record, command, started)

    def dispatch_account_flatten(self, account: str) -> dict:
        """
        Close EVERYTHING on one account. The emergency kill, and nothing else.

        No instrument and no strategy tag: see
        `crosstrade_formatter.format_account_flatten_command` for why both
        absences are the point, and for what this closes that this process
        never opened. `realtime.lifecycle.emergency_halt` is the only caller.

        It is a SEPARATE METHOD from `dispatch_flatten` rather than that method
        with an empty symbol, so nothing on the ordinary exit path can reach
        the account-wide hammer by dropping a field.

        Same sender, same retry rule, same redaction as every other order this
        class puts on the wire.
        """
        started = time.perf_counter()
        record: dict[str, Any] = {
            "timestamp": _utcnow(), "dry_run": self.dry_run,
            "account": account, "symbol": None, "action": "FLATTEN_ACCOUNT",
            "scope": "account", "quantity": None, "strategy_tag": "",
            "attempts": 0, "ok": False, "error": None,
        }
        try:
            command = format_account_flatten_command(account=account,
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
                       bar_ts: str = "", portfolio_id: str = "") -> dict:
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
            "portfolio_id": portfolio_id, "strategy_tag": strategy_tag,
            "attempts": 0, "ok": False, "error": None,
        }

        # THE LOCK, AND IT IS NOT THE ATTRIBUTION TAG. CrossTrade keys a
        # strategy lock on the exact string that opened the position and
        # refuses a later order carrying a different one - and the contributor
        # tag changes whenever the set of strategies that signalled together
        # changes. See `compose_wire_tag`. Both are kept on the record: this
        # one is what the broker sees, `strategy_tag` is who asked for it.
        #
        # WITH NO PORTFOLIO THERE IS NO PAIR TO KEY ON, so the caller's tag
        # goes out unchanged. That is the manual/operator entry point, where
        # nothing composed the tag in the first place.
        wire_tag = (compose_wire_tag(portfolio_id, payload.get("symbol", ""))
                    if portfolio_id else strategy_tag)
        record["wire_strategy_tag"] = wire_tag

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
                # `rule` and `detail` are carried as their own keys, not just
                # folded into `error`. `master_live` prints the refusal line
                # from `risk_refusals` and reads exactly these two; before
                # they were stored the loop raised `KeyError: 'rule'` INSIDE
                # the handler for a refused order, so the first time the
                # firewall did its job the execution loop died and systemd
                # restarted it into the same refusal every 15 seconds.
                # `blocked_by` is kept for anything reading the old name.
                record.update({"ok": False, "blocked_by": exc.rule,
                               "rule": exc.rule, "detail": exc.detail,
                               "error": f"RISK REFUSED [{exc.rule}] {exc.detail}",
                               "elapsed_ms": round(
                                   (time.perf_counter() - started) * 1000, 1)})
                self.risk_refusals.append(record)
                return record

        # THE SIZE CAP, IN THE SEND PATH. It CLAMPS to `max_qty` rather than
        # dropping the order. The firewall carries a configurable
        # `max_contracts_per_order` too, but the firewall is OPTIONAL -
        # `firewall=None` is the default and every construction outside
        # `master_live.py` leaves it there - so this cap is here as well, where
        # nothing can be armed incorrectly.
        #
        # WHAT A CLAMP COSTS, since it is not free and the next reader will ask.
        # The order that goes out is NOT the order the sizer computed: the ATR
        # sizer routinely asks for 5 contracts on MNQ, and the position that
        # results is sized against a stop drawn for five. The risk model no
        # longer describes the trade, and it does so while the record reports
        # success. That is the trade accepted here - a 1-lot expression of the
        # edge is preferred to no expression of it - and it is survivable only
        # because both sizes are kept on the record and the console says so on
        # every clamp. Do not let `requested_quantity` or the warning quietly
        # drop out; they are the whole difference between this and a silent
        # mis-size.
        requested_qty = payload.get("quantity")
        refusal = quantity_refusal(requested_qty, self.max_qty)
        if refusal is not None:
            order_qty = min(int(requested_qty), self.max_qty)
            # THE MESSAGE NAMES `self.max_qty`, NOT THE MODULE CONSTANT. They
            # are the same value in production - nothing outside the suite
            # passes `max_qty=` - but a line that reported a cap other than the
            # one it clamped to would be worse than no line at all.
            warning = (f"CLAMPED {payload.get('symbol')} order from "
                       f"{requested_qty} to {self.max_qty} due to MAX_QTY cap.")
            print(f"[live_dispatcher] {warning}", file=sys.stderr, flush=True)
            # BOTH SIZES STAY ON THE RECORD. `quantity` is what went to the
            # broker, because everything downstream reads it as the size of the
            # position; `requested_quantity` is what the sizer computed, and
            # without it the clamp is invisible in the one artifact that says
            # what happened. `clamped` is the flag an audit can filter on.
            record.update({
                "quantity": order_qty,
                "requested_quantity": requested_qty,
                "clamped": True,
                "rule": refusal["rule"],
                "detail": refusal["detail"],
                "warning": warning})
            # A COPY, never a mutation of the caller's dict. The plan's payload
            # list is the cycle report's `orders`, and rewriting the size in
            # place would make the report claim the sizer asked for 1.
            payload = {**payload, "quantity": order_qty}

        # THE STACK GATE. `aggregate_signals` nets the signals arriving in ONE
        # cycle against each other and knows nothing about the position the
        # PREVIOUS cycle opened, so a strategy that stays long-signalled did not
        # enter once - it entered on every pass, one position per interval,
        # none of which this loop could see as one position. The book that
        # answers is the same object `record_fill` and `record_flat` update.
        action = str(payload.get("action") or "")
        symbol = str(payload.get("symbol") or "")
        if not self.positions.can_execute(portfolio_id, symbol, action):
            hold = self.positions.hold_reason(portfolio_id, symbol, action)
            # TWO RULES, NOT ONE. `position_open` is a stack - the strategy is
            # already in the trade it is asking for. `exit_cooldown` is a
            # churn - it was taken OUT of that trade earlier in this same
            # cycle and is asking straight back in. They send an operator to
            # different places, so anything filtering on `rule` has to be able
            # to tell them apart.
            cooling = self.positions.exited_this_cycle(portfolio_id, symbol)
            rule = "exit_cooldown" if cooling else "position_open"
            record.update({
                "ok": False, "blocked_by": rule, "rule": rule,
                "held_direction": self.positions.state(portfolio_id, symbol),
                "detail": hold, "error": hold,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
            # A RECORD, not a print. Every other refusal in this method is one
            # too - `describe_cycle` is what puts them on the console, and
            # printing here as well is the same line three times: once
            # directly, once as a failed dispatch, once as a hold.
            self.held_orders.append(record)
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
                key=self.crosstrade_key,
                # THE TEXT COMMAND IS THE ONE THAT IS SENT (see the loop
                # below), so the tag has to be here. It was on `body` alone
                # from the day it was added, which meant every live entry
                # went out untagged and CrossTrade held no lock for the
                # flatten to clear.
                #
                # `wire_tag`, NOT `strategy_tag` - the stable per-pair lock,
                # because a contributor set that changes between bars presents
                # CrossTrade with a second string for a contract it has
                # already locked and the order is refused outright.
                strategy_tag=wire_tag)
            body = format_crosstrade_json(
                account=payload["account"],
                instrument=payload["symbol"],
                action=payload["action"],
                qty=payload["quantity"],
                order_type=payload.get("orderType", "MARKET"),
                # The same lock. The JSON object is kept as evidence of what
                # the other endpoint would have been handed, and two wire
                # forms of one order carrying different tags is the bug this
                # module already shipped once.
                strategy_tag=wire_tag)
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
            # THE TEXT COMMAND GOES ON THE WIRE, not the JSON object beside
            # it. The configured endpoint is CrossTrade's `/v1/send/` WEBHOOK,
            # which parses the semicolon form; `format_crosstrade_json`'s
            # object is for the JSON endpoint. This module built both and sent
            # the wrong one, and every live order was refused HTTP 400 - 125
            # of them on 2026-08-31 across five symbols, none ever accepted,
            # while dry run reported success because dry run opens no socket.
            # `realtime/send_test_probe.py` confirmed the text form against
            # all four accounts before this was changed. `body` is still built
            # above and kept on the record, so a switch back is one line.
            result = self._sender(command, webhook_url=self.crosstrade_url,
                                  timeout_seconds=self.timeout_seconds)
            # Both credentials out - the webhook URL and the `key=` field the
            # command carries. See `_safe_result`, which is shared with the
            # flatten path so the two cannot drift apart on which of them
            # scrubs what.
            record["result"] = self._safe_result(result)
            record["ok"] = bool(result.get("ok"))
            record["error"] = result.get("error")
            record["http_status"] = result.get("http_status")
            # A 2xx IS NOT A FILL. CrossTrade accepts the webhook and then
            # declines the trade in the BODY - a strategy-lock refusal comes
            # back 200 - so an order it refused read `OK` everywhere. See
            # `crosstrade_formatter.refusal_reason`.
            refusal = refusal_reason(result.get("response_body"))
            if refusal is not None:
                record["ok"] = False
                record["refused_by_broker"] = refusal
                record["error"] = record["error"] or refusal
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
            # A HELD ORDER IS NOT A FAILED ONE. It gets its own line below;
            # printed here as well it reads as an order that was attempted and
            # rejected by the broker, which is a different thing to go and
            # investigate.
            if d.get("rule") in ("position_open", "exit_cooldown"):
                continue
            status = "OK  " if d["ok"] else "FAIL"
            # THE RESPONSE BODY, ON FAILURE ONLY. `send_execution_signal`
            # captures it on every 4xx/5xx and it was being thrown away here,
            # leaving "HTTP 400: Bad Request" as the whole of what an operator
            # could see. On 2026-08-31 that cost a diagnosis: 109 consecutive
            # live orders were refused and the only readable fact was the
            # status code, which names a class of error rather than the error.
            # The endpoint's own explanation is the thing worth printing.
            detail = ""
            if not d["ok"]:
                body = (d.get("result") or {}).get("response_body")
                if body:
                    # SANITISED BEFORE IT IS PRINTED. An endpoint that echoes
                    # the request back in its error body hands back the
                    # `key=` field verbatim, and this line goes to a log file
                    # that outlives the session. `redact` replaces the key and
                    # passes text without one through unchanged; the webhook
                    # PATH is scrubbed separately because it is the other half
                    # of the credential and `redact` knows nothing about it.
                    detail = redact(str(body).strip())
                    secret = str(getattr(self, "crosstrade_url", "") or "")
                    path = _url_path(secret)
                    if path:
                        detail = detail.replace(path, "/<redacted>")
                    detail = f"  <- {detail[:400]}"
            # A CLAMPED ORDER PRINTS `x1` AND IS NOT A 1-LOT THE SIZER ASKED
            # FOR. Without this the console cannot tell the two apart, and the
            # one that matters is the one where a 5-contract position went on
            # at 1 against a stop drawn for five.
            if d.get("clamped"):
                detail += (f"  (CLAMPED from {d.get('requested_quantity')} "
                           f"— MAX_QTY cap)")
            # THE TAG, ON THE LINE THAT SAYS AN ORDER WENT OUT. It is
            # CrossTrade's LOCK - the flatten is matched to it by string
            # equality - and until 2026-09-04 it appeared in no log at all, so
            # "which lock did that order take out" could only be answered by
            # recomposing it by hand from the plan. An order that carries no
            # tag is shown as `tag=NONE` rather than omitted: an untagged
            # entry is a position CrossTrade holds no lock for, and a blank
            # where a tag should be reads as a formatting quirk.
            tag = str(d.get("strategy_tag") or "")
            lock = str(d.get("wire_strategy_tag") or "")
            # BOTH, because they answer different questions and are no longer
            # the same string. `tag=` is WHO ASKED - the contributors to the
            # netted position, which is the attribution a promotion is decided
            # on. `lock=` is what CROSSTRADE HOLDS against the contract, keyed
            # on the pair so it cannot collide with itself between bars. It is
            # printed only when it differs: on the manual path there is no
            # portfolio, the two are one string, and repeating it reads as two
            # facts where there is one.
            lines.append(f"  {status} {d['account']:<16} {d['action']:<5} "
                         f"{d['symbol']:<5} x{d['quantity']}"
                         + (f'  | tag="{tag}"' if tag else "  | tag=NONE")
                         + (f' lock="{lock}"' if lock and lock != tag else "")
                         + (f"   {d['error']}" if d.get("error") else "")
                         + detail)
        for h in report.get("held", []):
            # AN ORDER THAT WAS NOT SENT, on the same summary as the ones that
            # were. A cycle that held every entry and a cycle that produced no
            # signal both print no dispatch line, and they are different facts
            # about the account.
            lines.append(f"       {h['error']}")
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
            # A BLOCKED WEEKDAY GETS ITS OWN VERB. `HOLD` is what every other
            # decline prints, and an operator scanning a Friday log for "why
            # is nothing trading" needs to separate "the market left the
            # certified quadrant" from "this pair is stood down all session by
            # a rule somebody wrote". They are fixed by completely different
            # work, and one of them is not a fault at all.
            verb = "DOW " if d.get("rule") == "blocked_weekday" else "HOLD"
            lines.append(f"       {verb} {d.get('strategy_id')} "
                         f"{d.get('symbol')} — {d['reason']}")
        for u in report.get("dow_unresolved") or []:
            # A GATE THAT DEGRADED TO PERMISSIVE, said out loud. This strategy
            # carries a blocked weekday and was let through because the fill
            # session could not be resolved; printed nowhere it is
            # indistinguishable from a cycle the gate had nothing to say about.
            lines.append(f"       DOW? {u['strategy_id']} {u['symbol']} — "
                         f"{u['reason']}")
        # AFTER the verdicts, so the decision reads first and the numbers
        # behind it read second. One line per (strategy, symbol) evaluated,
        # carrying whatever that module declares - see `_record_telemetry`.
        for ind in report.get("indicators") or []:
            lines.append(f"       IND  {ind['strategy_id']} {ind['symbol']} "
                         f"{ind['direction']}  "
                         + "  ".join(f"{k}={_fmt_reading(v)}"
                                     for k, v in ind["values"].items()))
        for e in report.get("indicator_errors") or []:
            # Reported, never fatal. A feature matrix that will not build is a
            # real finding about the module and says nothing about the cycle,
            # which completed.
            lines.append(f"       IND? {e['strategy_id']} {e['symbol']} — "
                         f"indicators unavailable: {e['error']}")
        # THE EXIT ACTUATOR'S ORDERS. Absent from this card until 2026-09-02,
        # which is why the MES churn took a log audit to find: the FLATTEN that
        # closed the position every cycle left NO LINE ANYWHERE, so the console
        # showed a bare re-entry with no explanation and the loop looked like a
        # stack gate that had stopped working. A flatten is the one order whose
        # job is to close a position; it is the last thing that should be
        # invisible. Refusals print too - "declined to flatten" and "never
        # signalled an exit" are different facts about an account.
        for x in report.get("exit_orders", []):
            if x.get("emitted"):
                status = "OK  " if x.get("ok") else "FAIL"
                # THE FLATTEN'S TAG IS THE ONE THAT MATTERS MOST. It is the
                # lock RELEASE, matched to the entry's by string equality, and
                # a flatten spelling it even slightly differently is sent,
                # accepted and logged while the position stays open. Printing
                # it beside the entry's is what makes that visible in a log
                # rather than only on the account.
                ftag = str(x.get("strategy_tag") or "")
                flock = str(x.get("wire_strategy_tag") or "")
                lines.append(
                    f"  {status} {x.get('portfolio_id','')}/"
                    f"{x.get('symbol','')} FLATTEN"
                    + (f'  | tag="{ftag}"' if ftag else "  | tag=NONE")
                    + (f' lock="{flock}"' if flock and flock != ftag else "")
                    + f"  ({x.get('reason')})"
                    + (f"   {x['error']}" if x.get("error") else ""))
            else:
                lines.append(f"       NO-EXIT {x.get('portfolio_id','')}/"
                             f"{x.get('symbol','')} — {x.get('reason')}")
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
            # THE BLOCKED WEEKDAY IS ON THE ROSTER LINE, printed before the
            # first cycle. It is a standing restriction on an account, and an
            # operator who does not know it is there reads Friday's empty log
            # as a market with no setups.
            dow = (f"  dow=BLOCKED {named_weekdays(h.blocked_weekdays)}"
                   if h.blocked_weekdays else "")
            lines.append(f"  {h.strategy_id:<34} -> {h.portfolio_id:<15} "
                         f"({h.account})  sl={h.sl_atr_mult} tp={h.tp_atr_mult} "
                         f"certified={list(h.certified_symbols)}{dow}")
        for e in self.strategy_errors:
            lines.append(f"  FAILED TO LOAD {e['strategy_id']} "
                         f"({e['portfolio_id']}): {e['error']}")
        if self.daemon.model_errors:
            lines.append(f"  ML MODELS FAILED TO LOAD: "
                         f"{sorted(self.daemon.model_errors)}")
        return "\n".join(lines)


def _contributor_ids(plan_record: dict) -> list[str]:
    """
    The strategy ids behind one netted position, sorted and de-duplicated.

    THE SAME NORMALISATION `PositionBook.record_fill` APPLIES to the list it
    stores, so a tag composed from the plan and a tag composed from the book
    for that position are the same string. That equality is what
    `compose_strategy_tag` relies on to tag an exit with the tag its entry
    carried.
    """
    return sorted({str(c.get("strategy_id"))
                   for c in plan_record.get("contributors", [])
                   if c.get("strategy_id")})


def compose_strategy_tag(portfolio_id: str, strategy_ids) -> str:
    """
    `portfolio:strategy_a+strategy_b` - what ties an NT8 fill back to what
    asked for it (`live.dispatcher.evaluate_incubator_sync` reconciles on it)
    and, since it now travels on the wire, the CrossTrade strategy LOCK.

    A NETTED POSITION BELONGS TO EVERY CONTRIBUTOR, so all of them are named.
    Tagging it with one would attribute the whole position to a strategy that
    asked for part of it, and the reconciliation would then report the others
    as having placed nothing.

    ONE FUNCTION, BOTH SIDES OF THE TRADE. The entry composes it from the plan
    and the exit from the position book, and CrossTrade matches a lock by
    string equality - so an exit that spelled the tag even slightly differently
    would not release what the entry took out, and the position would sit there
    with every log line reporting a flatten that was sent and accepted.
    """
    ids = sorted({str(sid) for sid in (strategy_ids or []) if sid})
    tag = f"{portfolio_id}:{'+'.join(ids)}"
    # `;` and `=` are the plain-text command's field separators and the tag is
    # free text. They are mapped to `_` HERE rather than left to the
    # formatter, which deletes them: two ids differing only in a separator
    # would otherwise collapse onto one lock. `sanitize_strategy_tag` still
    # runs after this and is what the wire form is guaranteed by - it is asked
    # here too so the tag on the cycle report is the tag on the wire.
    return sanitize_strategy_tag(tag.replace(";", "_").replace("=", "_"))


def compose_wire_tag(portfolio_id: str, symbol: str) -> str:
    """
    `portfolio:SYMBOL` - the STABLE tag that goes on the wire, and the only
    string CrossTrade's strategy lock ever sees.

    WHY THE WIRE TAG IS NOT THE ATTRIBUTION TAG
    ===========================================
    CrossTrade locks a contract to the exact string that OPENED the position
    and refuses any later order carrying a different one:

        Trade blocked: MGC is managed by strategy 'Incubator-Even:strat_a'

    `compose_strategy_tag` names the CONTRIBUTORS, and a contributor set is not
    stable across bars - it is whichever strategies signalled together on that
    bar. A position opened by `a` alone and reversed on the next bar by `a+b`
    presents a second string for a contract that is already locked, and the
    order is REFUSED. The exit path had the same exposure from the other
    direction: after a restart the position book is empty, so a reconciled
    flatten knows the pair but not who opened it, and the tag it could compose
    was not the one holding the lock.

    Keying the lock on (portfolio, symbol) removes both. The pair is what the
    position IS - `PositionBook`, `build_order_plan` and `plan_exits` are all
    keyed by it, one netted position per pair - so the lock now has exactly the
    granularity of the thing it locks, and it is derivable at any moment by any
    process, including one that has just started and holds no history.

    ATTRIBUTION DOES NOT LIVE HERE AND IS NOT LOST. The contributor tag is
    still composed for every order and is what goes on the dispatch record, the
    cycle card in `master_live.log`, and the durable `EngineState` ledger. See
    `describe_cycle`, which prints both: `tag=` is who asked, `lock=` is what
    CrossTrade holds.

    The symbol is upper-cased and used AS DISPATCHED - the micro, not its
    full-size parent. The lock is held against the contract the order names, so
    resolving MGC to GC here would name a contract no order was placed on.
    """
    sym = str(symbol or "").strip().upper()
    tag = f"{portfolio_id}:{sym}"
    # Separators mapped exactly as `compose_strategy_tag` maps them, for the
    # same reason: the formatter DELETES them, so two ids differing only in a
    # separator would collapse onto one lock.
    return sanitize_strategy_tag(tag.replace(";", "_").replace("=", "_"))


def _strategy_tag(plan_record: dict) -> str:
    """The tag for one netted ENTRY, composed from its plan record."""
    return compose_strategy_tag(plan_record.get("portfolio_id", ""),
                                _contributor_ids(plan_record))


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


def _fmt_reading(value: Any) -> str:
    """
    One indicator value, compactly, and WITHOUT EVER RAISING.

    A feature matrix can carry None, NaN, a bool or a numpy scalar, and this
    runs inside a console line. `f"{None:.2f}"` raises TypeError - that exact
    expression took `realtime/regime_daemon.py` down 163 times on 2026-08-26,
    publishing correctly and then dying while formatting its own log. A cycle
    summary is not going to repeat it, so every branch here ends in a string.

    Absence prints `n/a` rather than `0`: a zero RSI is a measurement, and a
    zero in a column of readings reads as one.
    """
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "T" if value else "F"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:                       # NaN, without importing math for it
        return "nan"
    if f in (float("inf"), float("-inf")):
        return "inf" if f > 0 else "-inf"
    if f == int(f) and abs(f) < 1e6:
        # hour_et=6.0 and day_of_week=1.0 are counts; a trailing .00 on them
        # is noise in a line already carrying five columns.
        return str(int(f))
    if abs(f) < 0.001:
        # `atr_norm` lands here. Scientific is the readable form for a value
        # whose leading digit is four places down; a fixed format would print
        # 0.0000 and lose it entirely.
        return f"{f:.4g}"
    if abs(f) >= 1000:
        # A price-scale reading. Plain decimals, because `1.523e+04` is harder
        # to compare against a chart than `15234.57` at exactly the moment
        # somebody is doing that.
        return f"{f:.2f}"
    return f"{f:.4f}".rstrip("0").rstrip(".")


def _url_path(url: str) -> str:
    """
    The PATH of a webhook URL, for scrubbing it out of text that may echo it.

    `_host_only` keeps the host because that is the safe half; this returns
    the other half, which is the half that is the credential - CrossTrade's
    `/v1/send/<token>/<token>` route authorises orders on the account by
    itself. Returns "" for a URL with no meaningful path, so a caller's
    `replace()` cannot blank out every "/" in the string it is cleaning.
    """
    if not url:
        return ""
    from urllib.parse import urlparse
    try:
        path = urlparse(url).path or ""
    except ValueError:
        return ""
    return path if len(path) > 1 else ""


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
