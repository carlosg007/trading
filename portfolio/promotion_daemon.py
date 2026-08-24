#!/usr/bin/env python3
"""
The forward-incubation promotion rules, and the file surgery that acts on them.

Location:  ~/src/trading/portfolio/promotion_daemon.py
Reads:     data/incubator_ledger.json, config/portfolios.json
Writes:    both of the above, and only through `promote_strategy`.

WHAT THIS DECIDES, AND WHAT IT DELIBERATELY DOES NOT
====================================================
A strategy reaches `Incubator-Odd` / `Incubator-Even` because Stage 3 certified
it on historical bars. That is evidence about the past. This module answers the
next question — did it keep working on bars nobody had when it was certified —
and it answers it from FORWARD PAPER TRADES recorded in the ledger, never from
a backtest.

The bar is deliberately LOOSE, and the four criteria say why in their own
names: a fortnight of live behaviour, enough closed trades to be more than a
coin flip, expectancy on the right side of 1.00 after costs, and a forward
drawdown inside the envelope the account will actually be governed by. It is
not a second certification. Stage 3's gates already asked whether there is an
edge; this asks whether the thing has been quietly falling apart since.

**Nothing here enforces anything.** Moving a strategy from an incubator
portfolio to a prop portfolio changes which account CrossTrade NAM will route
its orders to. The drawdown lockouts, the daily loss caps and the prop-firm
challenge state are enforced there, against a live balance this repository
cannot see — see `portfolio/config_loader.py` for the whole separation. What
this module writes is a routing decision, and a routing decision is exactly as
reversible as editing the JSON back.

THE ALLOWABLE FORWARD DRAWDOWN IS NOT COMPUTED HERE IF IT CAN BE READ
=====================================================================
`allowable_dd = max_trailing_drawdown_usd x max_forward_incubation_dd_pct`
($2,500 x 0.40 = $1,000 as shipped). `config_loader.load_portfolio_config`
already derives exactly that product onto `derived.allowable_forward_dd_usd`,
so `allowable_forward_dd` PREFERS the derived value and falls back to the
product only for a raw config that never went through the loader. Two
implementations of one multiplication is still two places for it to be wrong,
and the wrong one produces a plausible dollar figure rather than an error.

It is a fraction of the TRAILING LIMIT, not of the account. $1,000 is 2% of the
$50,000 profile; read as an account percentage the bar would sit at $20,000,
which no forward drawdown would ever breach and the criterion would silently
stop being a criterion.

WHERE THE METRICS COME FROM, AND WHY THAT IS RECORDED
=====================================================
A ledger entry may state its metrics (`days_active`, `trade_count`,
`realized_pf`, `max_drawdown`) or carry the closed `trades` they were computed
from. When it carries trades, they are DERIVED here and the declared summary is
checked against them; a disagreement FAILS the evaluation rather than being
resolved silently, because the only two readings of it are "the summary is
stale" and "the trade list is incomplete", and both are reasons not to move an
account. `metrics["source"]` records which path ran, on every report.

**A missing metric is a FAIL, never a zero.** An absent trade count is not a
strategy that placed no trades; it is a ledger nobody filled in, and defaulting
it to 0 would make those two indistinguishable at exactly the moment a real
account is about to be handed a strategy.

The one exception is `active_sessions`, the parenthetical half of criterion 1
(">= 14 calendar days, at least 10 active trading sessions"). It is checked
when the ledger records it or the trades imply it, and reported as NOT RECORDED
— not as a failure — when neither does. It is a texture check on the same
window the calendar-day bar already covers, and failing every entry written
before the field existed would stall the incubator on a bookkeeping detail
rather than on a result.

SESSION DATES ARE THE ENGINE'S, NOT A SECOND COPY OF THE RULE
==============================================================
"Active session" means a CME session, which starts at 18:00 ET the previous
evening. `backtest.event_calendar.session_date` is the one place that rule is
written down and it is imported here rather than reimplemented — a Sunday
evening fill counted as its own session inflates the count against a bar the
whole point of which is that ten distinct sessions happened. The import is lazy
because it pulls pandas in, and an entry that declares its metrics needs
neither.

TWO FILES, ONE DECISION, AND NO SUCH THING AS ONE ATOMIC WRITE ACROSS BOTH
===========================================================================
`promote_strategy` moves a strategy in `config/portfolios.json` and stamps it
in `data/incubator_ledger.json`. Each file is written atomically (temp file,
`os.replace`), and the config is VALIDATED through `load_portfolio_config`
while it is still the temp file — so a mutation that would produce an
unloadable routing table never reaches the real path.

The PAIR is not one transaction, and pretending otherwise would be the lie.
The config is written first and the ledger second, because that ordering has a
recovery and the reverse does not: if the ledger write fails, the config is
restored from the bytes that were read, and the function re-raises. A ledger
that said GRADUATED_PROP over a config that still routed to the incubator would
be invisible — every subsequent run would skip the strategy as already
graduated while its orders kept arriving on the sim account.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# Importable as `portfolio.promotion_daemon` from the repository root AND
# runnable from inside `portfolio/`, where Python puts this directory on the
# path instead of the root. The same two lines `config_loader.py` carries.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.config_loader import (  # noqa: E402
    ALLOCATIONS_KEY,
    DEFAULT_CONFIG_PATH,
    PortfolioConfigError,
    clear_cache,
    load_portfolio_config,
)

DEFAULT_LEDGER_PATH = "data/incubator_ledger.json"

# --------------------------------------------------------------------------
# the promotion criteria
# --------------------------------------------------------------------------
# Loose governance, on purpose. Stage 3 asked whether there is an edge; these
# ask whether it survived contact with forward bars.
MIN_DAYS_ACTIVE = 14          # calendar days, inclusive bar (>=)
MIN_ACTIVE_SESSIONS = 10      # CME sessions inside that window (>=)
MIN_TRADE_COUNT = 14          # closed forward trades (>=), ~1 per day
MIN_PROFIT_FACTOR = 1.00      # STRICT: 1.00 is break-even, not expectancy

# Status tokens. A strategy this module has never seen is INCUBATING; the one
# it writes is GRADUATED_PROP. Anything else is left alone and reported as-is —
# a token nobody here defined is a decision somebody else made.
STATUS_INCUBATING = "INCUBATING"
STATUS_GRADUATED = "GRADUATED_PROP"

# The routing table, spelled out rather than derived from the name. Deriving
# `Prop-` + suffix would promote a portfolio called `Incubator-Test` onto a
# `Prop-Test` account that does not exist, and the failure would surface as a
# KeyError several layers away from the typo.
PROMOTION_ROUTES = {
    "Incubator-Odd": "Prop-Odd",
    "Incubator-Even": "Prop-Even",
}

# Metric aliases accepted on a ledger entry, canonical name first. Kept short
# and closed: an open-ended alias map is how a typo'd key silently reads as a
# missing metric.
_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "days_active": ("days_active",),
    "active_sessions": ("active_sessions",),
    "trade_count": ("trade_count",),
    "realized_pf": ("realized_pf", "realized_profit_factor"),
    "max_drawdown": ("max_drawdown", "max_forward_drawdown_usd"),
}

# Tolerances for the declared-vs-derived reconciliation. Money and ratios are
# compared loosely enough that a rounded summary is not called a contradiction.
_PF_TOLERANCE = 0.01
_USD_TOLERANCE = 0.01


class PromotionError(Exception):
    """The ledger, the config, or the two together are not in a state a
    promotion can act on."""


# --------------------------------------------------------------------------
# paths and files
# --------------------------------------------------------------------------

def _resolve(path: str | Path) -> Path:
    """
    A relative path resolves against the REPOSITORY ROOT, not the working
    directory — the same rule `load_portfolio_config` follows, and for the same
    reason: a cron job, a test runner and an operator's shell disagree about
    `cwd`, and a daemon that wrote a promotion into a ledger somewhere else
    would report success.
    """
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _read_json(path: Path, what: str) -> dict:
    if not path.exists():
        raise PromotionError(f"no {what} at {path}")
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromotionError(
            f"{path} is not valid JSON: {exc.msg} at line {exc.lineno} "
            f"column {exc.colno}") from exc
    if not isinstance(blob, dict):
        raise PromotionError(f"{path}: the top level of a {what} must be an "
                             f"object; got {type(blob).__name__}")
    return blob


def _atomic_write_json(path: Path, blob: dict) -> None:
    """Temp file then `os.replace`, the rule every writer in this repository
    follows. A process killed mid-write leaves the previous file intact rather
    than a truncated one the next run reads as a short strategy list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_ledger(ledger_path: str | Path = DEFAULT_LEDGER_PATH) -> dict:
    """The incubator ledger as it is on disk, `strategies` guaranteed to be a
    dict."""
    path = _resolve(ledger_path)
    blob = _read_json(path, "incubator ledger")
    strategies = blob.get("strategies")
    if strategies is None:
        blob["strategies"] = {}
    elif not isinstance(strategies, dict):
        raise PromotionError(
            f"{path}: `strategies` must be an object keyed by strategy id; "
            f"got {type(strategies).__name__}")
    return blob


def load_raw_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    The routing table exactly as written, with no `derived` or `reconciliation`
    block added.

    `load_portfolio_config` is the right reader for asking questions of the
    config and the wrong one for editing it: it returns an ENRICHED copy, and
    writing that back would persist computed values into a file whose whole
    point is to distinguish what a human declared from what was derived from
    it. Validation still happens on every write — see `_write_config`.
    """
    path = _resolve(config_path)
    blob = _read_json(path, "portfolio configuration")
    if not isinstance(blob.get("portfolios"), dict):
        raise PromotionError(f"{path}: `portfolios` is missing or is not an "
                             f"object")
    return blob


def _write_config(path: Path, blob: dict) -> None:
    """
    Write the routing table, but only if it still loads.

    The temp file is validated through `load_portfolio_config` BEFORE
    `os.replace`, so a mutation that breaks the schema fails with the real file
    untouched. Skipping this would let a promotion leave behind a config that
    every subsequent run — including the live router — refuses to load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    try:
        load_portfolio_config(str(tmp), use_cache=False)
    except PortfolioConfigError as exc:
        tmp.unlink(missing_ok=True)
        raise PromotionError(
            f"the promotion would have made {path} unloadable, so it was not "
            f"written: {exc}") from exc
    os.replace(tmp, path)
    # The loader caches per resolved path. Without this, anything that already
    # read the config in this process keeps answering from the pre-promotion
    # copy — including `get_portfolio_for_strategy`, which is how the next
    # order finds its account.
    clear_cache()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _number(value: Any) -> float | None:
    """A float, or None for anything that is not a recorded number. `bool` is
    excluded deliberately: `True` is a 1.0 that means nothing."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _declared_metrics(entry: dict) -> dict[str, float | None]:
    """
    The metrics an entry states, read from the entry itself or from a nested
    `metrics` block. Both spellings exist in the wild — a hand-written ledger
    row puts them at the top level, a tracker-written one nests them — and
    refusing either would be a schema argument fought at the cost of a
    promotion.
    """
    nested = entry.get("metrics")
    nested = nested if isinstance(nested, dict) else {}
    out: dict[str, float | None] = {}
    for canonical, aliases in _METRIC_KEYS.items():
        found = None
        for alias in aliases:
            # Both places are tried for every alias rather than the top level
            # short-circuiting the nested one: an entry carrying `realized_pf:
            # null` beside a `metrics` block that holds the real figure is a
            # half-migrated row, and reading the null as the answer reports a
            # recorded metric as missing.
            found = _number(entry.get(alias))
            if found is None:
                found = _number(nested.get(alias))
            if found is not None:
                break
        out[canonical] = found
    return out


def _trade_pnl(trade: Any) -> float | None:
    if not isinstance(trade, dict):
        return None
    for key in ("pnl", "net_pnl", "realized_pnl"):
        value = _number(trade.get(key))
        if value is not None:
            return value
    return None


def _closed_trades(trades: Any) -> list[dict]:
    """
    Closed trades only. An OPEN position has no realised P&L, and counting it
    would let a strategy reach the 14-trade bar on positions it has not exited
    — the exact trades whose outcome is still unknown.
    """
    if not isinstance(trades, list):
        return []
    closed = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        status = str(trade.get("status", "CLOSED")).upper()
        if status != "CLOSED":
            continue
        if _trade_pnl(trade) is None:
            continue
        closed.append(trade)
    return closed


def _trade_timestamps(trades: list[dict]) -> list[str]:
    stamps = []
    for trade in trades:
        for key in ("closed_at", "exit_time", "timestamp"):
            value = trade.get(key)
            if isinstance(value, str) and value:
                stamps.append(value)
                break
    return stamps


def _session_dates(trades: list[dict]) -> set[str] | None:
    """
    The distinct CME sessions the closed trades exited in, or None when the
    trades carry no timestamps to attribute.

    `backtest.event_calendar.session_date` is imported here rather than
    reimplemented: the 18:00 ET roll is one line and it is written down once.
    An explicit `session_date` on a trade wins, because a recorder that already
    knows the session should not have it re-derived from a timestamp it may
    have rounded.
    """
    explicit = {str(t["session_date"]) for t in trades
                if isinstance(t.get("session_date"), str) and t["session_date"]}
    remaining = [t for t in trades if not isinstance(t.get("session_date"), str)]
    stamps = _trade_timestamps(remaining)
    if not explicit and not stamps:
        return None
    derived: set[str] = set()
    if stamps:
        from backtest.event_calendar import session_date  # lazy: pulls pandas
        for day in session_date(stamps):
            derived.add(day.strftime("%Y-%m-%d"))
    return {s[:10] for s in explicit} | derived


def derive_metrics_from_trades(entry: dict) -> dict[str, Any] | None:
    """
    The four criteria's inputs, computed from the entry's own closed trades.
    Returns None when the entry carries no usable trade list.

    `realized_pf` is gross profit over gross loss, and it is `None` — never a
    sentinel — when there was no losing trade. `backtest/profiler.py` writes
    999 in that case and the Discord card has to render it as `--`; a promotion
    decision must not inherit a number that means "undefined" but sorts like
    the best result on the board. `pf_undefined` says so, and the expectancy
    criterion then falls back to the sign of the net P&L, which is what "any
    net-positive expectancy" means when there is nothing to divide by.

    `max_drawdown` is the deepest peak-to-trough excursion of CUMULATIVE
    REALISED P&L, in dollars and reported POSITIVE. It is a floor on the true
    forward drawdown, not the true one: an open position's mark-to-market never
    enters this series, so a strategy that sat $2,000 underwater and closed
    flat contributes nothing here. That is the honest limit of what a closed
    trade list can say, and it is why the criterion is a safety check rather
    than the risk system.
    """
    trades = _closed_trades(entry.get("trades"))
    if not trades:
        return None
    pnls = [float(_trade_pnl(t)) for t in trades]
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    sessions = _session_dates(trades)
    stamps = _trade_timestamps(trades)
    days_active = None
    if len(stamps) >= 1:
        parsed = sorted(_parse_iso(s) for s in stamps if _parse_iso(s))
        if parsed:
            declared_start = _parse_iso(str(entry.get("started_at", "")))
            # The EARLIER of the two. A `started_at` stamped after the first
            # trade is a bookkeeping error, and subtracting from it yields a
            # window shorter than the trades prove — in the limit a negative
            # one, which clears no bar and reads as a brand new strategy.
            start = min(declared_start, parsed[0]) if declared_start else parsed[0]
            # Inclusive of both end days: a strategy that first traded on the
            # 1st and last traded on the 14th has been active for 14 days, not
            # 13. An exclusive count would hold every entry one day past the
            # bar it already cleared.
            days_active = (parsed[-1].date() - start.date()).days + 1

    return {
        "days_active": float(days_active) if days_active is not None else None,
        "active_sessions": float(len(sessions)) if sessions is not None else None,
        "trade_count": float(len(trades)),
        "realized_pf": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "pf_undefined": gross_loss <= 0,
        "net_pnl": float(sum(pnls)),
        "max_drawdown": float(max_dd),
    }


def _parse_iso(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def extract_metrics(entry: dict) -> dict[str, Any]:
    """
    The evaluation's inputs, and where each one came from.

    Derived values win when the entry carries trades, because the trade list is
    the primary record and a summary beside it is a cache. When BOTH are
    present and they disagree beyond a rounding tolerance, the disagreement is
    recorded in `inconsistencies` and the evaluation fails on it — the two
    readings are "the summary is stale" and "the trade list is incomplete", and
    neither is a state to move a live account in.
    """
    declared = _declared_metrics(entry)
    derived = derive_metrics_from_trades(entry)
    if derived is None:
        return {
            **declared,
            "pf_undefined": False,
            "net_pnl": None,
            "source": "declared",
            "inconsistencies": [],
        }

    inconsistencies = []
    for key, tolerance in (("days_active", 1.0), ("active_sessions", 1.0),
                           ("trade_count", 0.0), ("realized_pf", _PF_TOLERANCE),
                           ("max_drawdown", _USD_TOLERANCE)):
        stated, computed = declared.get(key), derived.get(key)
        if stated is None or computed is None:
            continue
        if abs(abs(stated) - abs(computed)) > tolerance:
            inconsistencies.append(
                f"{key}: the entry states {stated:,.2f} but its own closed "
                f"trades give {computed:,.2f}")
    # A derived value of None means "the trades could not say" — trades with
    # no timestamps cannot date a window — and the entry's own figure stands in
    # for it. `derived.get(k, declared[k])` would NOT do this: the key is
    # present and holds None, so the default never fires and a declared 16-day
    # window would read as NOT RECORDED beside a trade list that simply carried
    # no clocks.
    merged = {k: (derived.get(k) if derived.get(k) is not None
                  else declared.get(k))
              for k in _METRIC_KEYS}
    return {
        **merged,
        "pf_undefined": bool(derived.get("pf_undefined")),
        "net_pnl": derived.get("net_pnl"),
        "source": "derived_from_trades",
        "inconsistencies": inconsistencies,
    }


def allowable_forward_dd(portfolio_config: dict) -> float:
    """
    The forward drawdown a strategy may run up during incubation, in dollars.

    Prefers `derived.allowable_forward_dd_usd`, which `load_portfolio_config`
    computed, and falls back to the product for a raw config that never went
    through the loader. See the module docstring: one multiplication, one
    implementation, and it is a fraction of the trailing LIMIT rather than of
    the account.
    """
    derived = portfolio_config.get("derived")
    if isinstance(derived, dict):
        value = _number(derived.get("allowable_forward_dd_usd"))
        if value is not None and value > 0:
            return value
    profile = portfolio_config.get("risk_profile")
    if not isinstance(profile, dict):
        raise PromotionError(
            f"portfolio {portfolio_config.get('portfolio_id')!r} has no "
            f"risk_profile, so the allowable forward drawdown cannot be "
            f"computed. There is no default: a bar invented here would be a "
            f"live-account limit nobody wrote down.")
    limit = _number(profile.get("max_trailing_drawdown_usd"))
    pct = _number(profile.get("max_forward_incubation_dd_pct"))
    if limit is None or pct is None:
        raise PromotionError(
            f"portfolio {portfolio_config.get('portfolio_id')!r}: "
            f"risk_profile needs both max_trailing_drawdown_usd and "
            f"max_forward_incubation_dd_pct to derive the allowable forward "
            f"drawdown; got {limit!r} and {pct!r}")
    return limit * pct


# --------------------------------------------------------------------------
# the evaluation
# --------------------------------------------------------------------------

def _fmt(value: float | None, decimals: int = 2, prefix: str = "") -> str:
    if value is None:
        return "NOT RECORDED"
    return f"{prefix}{value:,.{decimals}f}"


def evaluate_strategy_promotion(strategy_id: str, ledger_entry: dict,
                                portfolio_config: dict) -> dict:
    """
    Score one incubating strategy against the four promotion criteria.

    Returns `{"strategy_id", "passed", "reasons", "metrics"}` — plus `criteria`
    and `failed`, which are the same verdict in a shape a table can read
    without parsing prose.

    `passed` is True only when every criterion cleared on a RECORDED number.
    A missing metric fails the criterion it belongs to; `active_sessions` is
    the single documented exception (see the module docstring) and reports NOT
    RECORDED without blocking.

    Nothing is written here and nothing is read from disk: the entry and the
    portfolio are handed in. That is what makes the rule testable against a
    mock entry, and it is why `scripts/incubator_tracker.py` can print a table
    of verdicts with `--dry-run` and touch nothing.
    """
    if not isinstance(ledger_entry, dict):
        raise PromotionError(
            f"{strategy_id!r}: a ledger entry must be an object; got "
            f"{type(ledger_entry).__name__}")

    metrics = extract_metrics(ledger_entry)
    allowable_dd = allowable_forward_dd(portfolio_config)
    metrics["allowable_dd"] = allowable_dd
    metrics["thresholds"] = {
        "min_days_active": MIN_DAYS_ACTIVE,
        "min_active_sessions": MIN_ACTIVE_SESSIONS,
        "min_trade_count": MIN_TRADE_COUNT,
        "min_profit_factor": MIN_PROFIT_FACTOR,
        "allowable_dd_usd": allowable_dd,
    }

    days = metrics.get("days_active")
    sessions = metrics.get("active_sessions")
    trades = metrics.get("trade_count")
    pf = metrics.get("realized_pf")
    max_dd = metrics.get("max_drawdown")

    criteria: list[dict[str, Any]] = []

    # 1 · the window ------------------------------------------------------
    if days is None:
        criteria.append(_criterion(
            "window", "Minimum active window", False,
            f"days active NOT RECORDED (>= {MIN_DAYS_ACTIVE} required). An "
            f"absent window is not a short one; the ledger is incomplete."))
    else:
        window_ok = days >= MIN_DAYS_ACTIVE
        session_note = "sessions NOT RECORDED"
        if sessions is not None:
            window_ok = window_ok and sessions >= MIN_ACTIVE_SESSIONS
            session_note = (f"{sessions:,.0f} active sessions "
                            f"(>= {MIN_ACTIVE_SESSIONS})")
        criteria.append(_criterion(
            "window", "Minimum active window", window_ok,
            f"{days:,.0f} calendar days (>= {MIN_DAYS_ACTIVE}); "
            f"{session_note}"))

    # 2 · the sample ------------------------------------------------------
    criteria.append(_criterion(
        "sample", "Minimum trade sample", trades is not None
        and trades >= MIN_TRADE_COUNT,
        f"{_fmt(trades, 0)} closed forward trades "
        f"(>= {MIN_TRADE_COUNT} required)"))

    # 3 · expectancy ------------------------------------------------------
    if pf is None and metrics.get("pf_undefined"):
        net = metrics.get("net_pnl")
        criteria.append(_criterion(
            "expectancy", "Positive expectancy", net is not None and net > 0,
            f"profit factor UNDEFINED (no losing trade); scored on net P&L "
            f"{_fmt(net, 2, '$')}. A 999 sentinel is not a measured factor and "
            f"is deliberately not used here."))
    else:
        criteria.append(_criterion(
            "expectancy", "Positive expectancy",
            pf is not None and pf > MIN_PROFIT_FACTOR,
            f"realized profit factor {_fmt(pf)} "
            f"(> {MIN_PROFIT_FACTOR:.2f} required)"))

    # 4 · forward drawdown safety ----------------------------------------
    # Compared on MAGNITUDE: a drawdown recorded as -1,250 is the same
    # excursion as 1,250, and comparing raw would let every negative figure
    # clear every limit.
    criteria.append(_criterion(
        "drawdown", "Forward drawdown safety",
        max_dd is not None and abs(max_dd) < allowable_dd,
        f"max forward drawdown {_fmt(None if max_dd is None else abs(max_dd), 2, '$')} "
        f"(< ${allowable_dd:,.2f} allowable)"))

    reasons = [f"{c['status']} {c['id']}: {c['detail']}" for c in criteria]
    failed = [c["id"] for c in criteria if c["status"] == "FAIL"]

    for note in metrics.get("inconsistencies", []):
        reasons.append(f"FAIL ledger: {note}")
        failed.append("ledger")

    status = str(ledger_entry.get("status", STATUS_INCUBATING)).upper()
    if status != STATUS_INCUBATING:
        reasons.append(
            f"FAIL status: the ledger records {status!r}, not "
            f"{STATUS_INCUBATING!r}. Only an incubating strategy is a "
            f"promotion candidate.")
        failed.append("status")

    return {
        "strategy_id": strategy_id,
        "passed": not failed,
        "reasons": reasons,
        "metrics": metrics,
        "criteria": criteria,
        "failed": failed,
        "portfolio": portfolio_config.get("portfolio_id"),
    }


def _criterion(cid: str, label: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"id": cid, "label": label, "status": "PASS" if ok else "FAIL",
            "detail": detail}


# --------------------------------------------------------------------------
# the promotion
# --------------------------------------------------------------------------

def target_portfolio_for(source_portfolio: str) -> str:
    """`Incubator-Odd` -> `Prop-Odd`, `Incubator-Even` -> `Prop-Even`, and a
    refusal for anything else."""
    try:
        return PROMOTION_ROUTES[source_portfolio]
    except KeyError:
        raise PromotionError(
            f"{source_portfolio!r} is not an incubator portfolio. Promotions "
            f"run {sorted(PROMOTION_ROUTES.items())} and nowhere else; a "
            f"target derived from the name would route orders to an account "
            f"that may not exist.") from None


def promote_strategy(strategy_id: str,
                     source_portfolio: str,
                     config_path: str = DEFAULT_CONFIG_PATH,
                     ledger_path: str = DEFAULT_LEDGER_PATH) -> bool:
    """
    Move one strategy from its incubator portfolio to the matching prop
    portfolio, and stamp the ledger.

    Returns True when both files were changed. Returns False for the ONE
    no-op that is not an error: the strategy is already graduated and already
    on the target portfolio, which is what a second run over the same ledger
    finds. Everything else raises `PromotionError` — a half-applied promotion,
    an unknown portfolio, an absent ledger entry — because those are states a
    caller has to see rather than count as "nothing to do".

    This applies the decision; it does not make it. `evaluate_strategy_promotion`
    is the rule, and calling this without it promotes a strategy on no evidence.
    """
    target_portfolio = target_portfolio_for(source_portfolio)
    cfg_path = _resolve(config_path)
    led_path = _resolve(ledger_path)

    config = load_raw_config(cfg_path)
    original_config_bytes = cfg_path.read_bytes()
    ledger = load_ledger(led_path)

    portfolios = config["portfolios"]
    for pid in (source_portfolio, target_portfolio):
        if pid not in portfolios:
            raise PromotionError(
                f"{cfg_path} has no portfolio {pid!r}. Known: "
                f"{sorted(portfolios)}")

    source = portfolios[source_portfolio]
    target = portfolios[target_portfolio]
    source_list = list(source.get("active_strategies") or [])
    target_list = list(target.get("active_strategies") or [])

    entry = ledger["strategies"].get(strategy_id)
    if not isinstance(entry, dict):
        raise PromotionError(
            f"{strategy_id!r} has no entry in {led_path}. A promotion writes a "
            f"graduation stamp onto a record of the forward trades it rests "
            f"on; there is nothing here to stamp.")

    already_graduated = (str(entry.get("status", "")).upper() == STATUS_GRADUATED
                         and strategy_id in target_list
                         and strategy_id not in source_list)
    if already_graduated:
        return False

    if strategy_id not in source_list:
        raise PromotionError(
            f"{strategy_id!r} is not on {source_portfolio}'s "
            f"active_strategies (it holds {source_list}). Promoting it would "
            f"append it to {target_portfolio} without removing it from "
            f"anywhere — which is not a move, it is a second live assignment "
            f"of the same strategy.")

    # One timestamp for both records. Two `now()` calls would put the ledger
    # and the routing table a few microseconds apart, and "which of these two
    # graduations is the same graduation" is then a judgement call.
    stamped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    source_list.remove(strategy_id)
    if strategy_id not in target_list:
        target_list.append(strategy_id)
    source["active_strategies"] = source_list
    target["active_strategies"] = target_list

    # The allocation record travels with the permission.
    #
    # `backtest/promote.py` writes a strategy into TWO places on a portfolio:
    # `active_strategies` grants the permission, and `strategy_allocations`
    # records the certified contract, timeframe, quadrant and contract count
    # behind it. Moving only the first would leave the record sitting on the
    # incubator account after the strategy graduated off it - a described
    # allocation on an account that no longer holds the strategy, beside a prop
    # account holding it with nothing describing it. Neither half routes an
    # order, so nothing would raise; `load_portfolio_config` would report the
    # drift and every reader after that would see two accounts each telling
    # half the truth.
    #
    # `status` is restamped rather than carried: the record said `incubating`
    # because it was, and this function is the moment that stopped being true.
    src_allocs = source.get(ALLOCATIONS_KEY)
    if isinstance(src_allocs, dict) and strategy_id in src_allocs:
        moved = copy.deepcopy(src_allocs.pop(strategy_id))
        moved["status"] = STATUS_GRADUATED
        moved["graduated_at"] = stamped_at
        moved["source_portfolio"] = source_portfolio
        tgt_allocs = target.get(ALLOCATIONS_KEY)
        if not isinstance(tgt_allocs, dict):
            tgt_allocs = {}
            target[ALLOCATIONS_KEY] = tgt_allocs
        tgt_allocs[strategy_id] = moved

    stamped = copy.deepcopy(entry)
    stamped["status"] = STATUS_GRADUATED
    stamped["graduated_at"] = stamped_at
    stamped["target_portfolio"] = target_portfolio
    # What it was promoted FROM, kept beside where it went. Without it the
    # ledger records a graduation with no way to say which incubator account
    # produced the forward trades the decision rested on.
    stamped.setdefault("source_portfolio", source_portfolio)
    ledger["strategies"][strategy_id] = stamped

    _write_config(cfg_path, config)
    try:
        _atomic_write_json(led_path, ledger)
    except OSError as exc:
        # The config moved and the ledger did not. Put the config back, so the
        # pair is consistent and the next run sees the promotion as still
        # outstanding rather than as silently done.
        cfg_path.write_bytes(original_config_bytes)
        clear_cache()
        raise PromotionError(
            f"{strategy_id!r}: the routing table was updated and the ledger "
            f"write then failed ({exc}). The routing change has been rolled "
            f"back, so the strategy is still on {source_portfolio} and the "
            f"promotion is still outstanding.") from exc
    return True
