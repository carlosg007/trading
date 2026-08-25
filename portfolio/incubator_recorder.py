#!/usr/bin/env python3
"""
portfolio.incubator_recorder - NT8 forward fills -> the incubator ledger.

Reads:   /mnt/backtest/artifacts/incubator_logs/*, config/portfolios.json
Writes:  data/incubator_ledger.json, and only when asked

THE INPUT SIDE OF THE PROMOTION GATE
====================================
`portfolio/promotion_daemon.py` grades a strategy on the CLOSED TRADES in its
ledger entry, and until this module existed nothing put any there. The ledger
shipped empty, the tracker printed "the ledger holds no strategies", and every
incubating strategy was invisible to the audit that decides whether it
graduates. This is the producer: NT8's export in, closed round turns out.

It computes the four criteria's INPUTS and never the criteria. Days, trades,
profit factor and drawdown are derived by `derive_metrics_from_trades` from
the trade list written here - so a recorder that wrote its own `realized_pf`
would be a second implementation of the gate, free to disagree with the one
that promotes. Nothing declared is written for that reason: the daemon flags a
summary that contradicts its own trades, and the only way to never trip that
check is to have no summary to contradict.

PAIRING, AND WHAT AN OPEN POSITION IS WORTH
===========================================
An NT8 EXECUTION log is one row per fill. A promotion criterion counts CLOSED
trades, so fills are paired into round turns FIFO per (strategy, symbol) - the
oldest open lot closes first, which is what the account did. Quantity is
matched in pieces: a 3-lot entry closed by a 1-lot and a 2-lot exit is two
trades, priced separately, because that is two realised outcomes and counting
it as one would average a winner into a loser.

**What is still open at the end of the log is NOT a trade and is not written.**
Its outcome is unknown, and a strategy allowed to reach the 14-trade bar on
positions it has not exited would be promoted on the trades whose result
nobody has seen yet. They are COUNTED in the report instead, so an operator
can see the difference between "flat" and "holding four".

A TRADE-LEVEL export skips all of that: NT8's Trades tab already carries the
round turn and its profit, and those rows are taken as the closed trades they
are.

COSTS ARE APPLIED, NEVER ASSUMED AWAY
=====================================
A forward P&L compared against a profit factor of 1.00 decides a promotion, so
commissions belong in it - the same rule the backtests run under. `--cost-
basis`:

    auto   an explicitly NET figure (`net_pnl`, `realized_pnl`) is used as-is;
           a gross `profit` has the log's own commission subtracted, or, if
           the export carries none, the round turn from `backtest/specs.py`.
    log    the log's numbers stand. For an export already net of fees, where
           subtracting again would charge the strategy twice.
    specs  the spec's round turn always, ignoring any commission column.

Every trade records the `cost_basis` it was priced under. Which of the three
is right depends on how the NT8 template was configured, and that is an
operator's knowledge - the wrong guess made silently is a profit factor off by
the commission on fourteen round turns, which at the 1.00 boundary is the
whole decision.

A price is never turned into money without a multiplier from
`backtest/specs.py`. A symbol with no spec has its rows REPORTED and skipped,
because a guessed multiplier scales every P&L figure for that contract and
nothing downstream would look wrong.

ATTRIBUTION IS EVIDENCE, NOT INFERENCE
======================================
A trade belongs to the strategy the log NAMES (`strategy`, `strategy_id`,
`strategy_tag` - the field `format_crosstrade_json` already sends). When the
export carries no such column, a row can still be attributed if the account it
filled on holds exactly ONE strategy certified for that symbol; two candidates
means the row is UNATTRIBUTED and is reported, never split, never assigned to
the first. A trade filed under the wrong strategy is a promotion decided on
somebody else's P&L.

The account column is NT8's (`Sim101`), the routing table's is
`Incubator-Odd`, and `NT8_ACCOUNT_ALIASES` is the only place the two are tied
together. Portfolio ids and `target_account` values are accepted as
themselves, so a log that already speaks in routing-table names needs no map.

RE-READING THE SAME LOG CHANGES NOTHING
=======================================
Every trade carries a `trade_id` derived from what it IS - strategy, symbol,
direction, quantity, both timestamps, both prices - so a cron run over an
export that grew by three rows adds three trades and not a duplicate of every
earlier one. Duplicates are the failure that makes a strategy look like it
cleared the 14-trade bar twice as fast as it did.

A GRADUATED entry is never appended to. Its forward trades are being taken on
a prop account and are not incubation evidence for a decision that has already
been made.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live.dispatcher import (                                      # noqa: E402
    INCUBATOR_LOG_DIR,
    fill_status,
    normalize_fill_row,
    parse_log_number,
    read_fill_log,
)
from portfolio.promotion_daemon import (                           # noqa: E402
    DEFAULT_LEDGER_PATH,
    STATUS_GRADUATED,
    STATUS_INCUBATING,
    load_ledger,
    write_ledger,
)

# NT8's account names against the routing table's. Spelled out rather than
# derived, for the same reason `PROMOTION_ROUTES` is: a rule that mapped
# `Sim1NN` by position would file a third simulation account's fills onto
# whichever portfolio the arithmetic landed on, and the trades would look
# real. Case-insensitive on lookup; a portfolio id or a `target_account`
# resolves to itself, so a log already written in routing-table names needs no
# entry here.
NT8_ACCOUNT_ALIASES: dict[str, str] = {
    "SIM101": "Incubator-Odd",
    "SIM102": "Incubator-Even",
}

# Recorder-only column spellings. The seven shared with
# `evaluate_incubator_sync` (symbol, ts, action, quantity, status,
# signal_price, realized_price) come from `live.dispatcher.FILL_ALIASES` and
# are deliberately NOT restated here.
EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "strategy":    ("strategy", "strategy_id", "strategy_tag", "strategy_name",
                    "signal_name"),
    "account":     ("account", "account_id", "account_name", "acct"),
    "net_pnl":     ("net_pnl", "realized_pnl", "realised_pnl"),
    "gross_pnl":   ("profit", "pnl", "gross_pnl", "trade_profit"),
    "commission":  ("commission", "commissions", "fees", "fee"),
    "entry_time":  ("entry_time", "entrytime", "entry_ts", "entry"),
    "exit_time":   ("exit_time", "exittime", "exit_ts", "exit"),
    "entry_price": ("entry_price", "entryprice"),
    "exit_price":  ("exit_price", "exitprice"),
}

COST_BASES = ("auto", "log", "specs")

LONG, SHORT = "LONG", "SHORT"
_BUY = {"buy", "buytocover", "buy_to_cover", "long", "b"}
_SELL = {"sell", "sellshort", "sell_short", "short", "s"}

RECORDER_TAG = "portfolio/incubator_recorder.py"


class RecorderError(Exception):
    """The logs, the routing table, or the ledger are not in a state fills can
    be recorded against."""


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------

def _extra_fields(raw: dict) -> dict[str, Any]:
    """The recorder-only columns off one raw row, canonicalised."""
    lowered = {str(k).strip().lower().replace(" ", "_"): v
               for k, v in raw.items()}
    out: dict[str, Any] = {}
    for canon, names in EXTRA_ALIASES.items():
        for name in names:
            if name in lowered and lowered[name] not in (None, ""):
                out[canon] = lowered[name]
                break
    return out


def root_symbol(instrument: Any) -> str:
    """
    `'MNQ 12-26'` -> `'MNQ'`. The expiry is DROPPED, deliberately.

    A NinjaTrader instrument name carries the contract month and a contract
    spec does not: MNQ's multiplier is MNQ's in every expiry, and the P&L on
    this trade needs the multiplier. Keeping the month would turn every roll
    into a new unknown symbol whose rows the recorder would refuse.
    """
    text = str(instrument or "").strip().upper()
    return text.split()[0] if text else ""


def _direction(action: Any) -> str | None:
    """LONG / SHORT from an action column, or None when it says neither."""
    token = str(action or "").strip().lower().replace(" ", "_")
    if token in _BUY:
        return LONG
    if token in _SELL:
        return SHORT
    return None


def parse_rows(path: str | Path) -> list[dict[str, Any]]:
    """
    One NT8 export -> canonical rows, in file order.

    The shared reader and the shared alias table do the common half; this adds
    the columns only a recorder needs. A row keeps its source file so a
    question about a trade ends at the line that produced it.
    """
    log_path = Path(path)
    rows = []
    for index, raw in enumerate(read_fill_log(log_path)):
        row = dict(normalize_fill_row(raw))
        row.update(_extra_fields(raw))
        row["_source_log"] = log_path.name
        row["_row"] = index
        rows.append(row)
    return rows


def discover_logs(target: str | Path) -> list[Path]:
    """
    Every readable export under `target`, sorted, or the single file it names.

    A MISSING directory raises and an EMPTY one does not: the first means the
    NT8 export was never wired up and the run should say so out loud, the
    second means a quiet session, which is an ordinary thing for a post-market
    job to find.
    """
    path = Path(target)
    if path.is_file():
        return [path]
    if not path.exists():
        raise RecorderError(
            f"no incubator log directory at {path}. Nothing has exported "
            f"fills there yet - this is not 'no trades', it is no feed. Point "
            f"--logs at the export, or configure NT8 to write into it.")
    if not path.is_dir():
        raise RecorderError(f"{path} is neither a file nor a directory")
    return sorted(p for p in path.iterdir()
                  if p.is_file() and p.suffix.lower() in
                  (".csv", ".txt", ".tsv", ".json"))


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------

def portfolio_for_account(account: Any, config: dict) -> str | None:
    """
    NT8's account name -> the portfolio id that routes it, or None.

    A portfolio id and a `target_account` both resolve to themselves before
    the alias table is consulted, so the map only has to carry the names that
    genuinely differ.
    """
    token = str(account or "").strip()
    if not token:
        return None
    portfolios = config.get("portfolios") or {}
    for pid, portfolio in portfolios.items():
        if token.upper() == str(pid).upper():
            return pid
        if token.upper() == str(portfolio.get("target_account", "")).upper():
            return pid
    mapped = NT8_ACCOUNT_ALIASES.get(token.upper())
    return mapped if mapped in portfolios else None


def _trades_symbol(portfolio: dict, strategy_id: str, symbol: str) -> bool:
    """
    Whether this strategy, on this portfolio, is the one trading `symbol`.

    The strategy's own `strategy_allocations` record answers it - that is the
    contract it was certified and registered on. Only when there is no record
    does the question fall back to the basket, which says what the PORTFOLIO
    trades and not which of its strategies.

    Both sides resolve through `realtime/contract_alias.py`, so a strategy
    registered on NQ matches a fill on MNQ: same price series, and the order
    was always for the micro.
    """
    from realtime.contract_alias import resolve_parent
    parent = resolve_parent(symbol)
    record = ((portfolio.get("strategy_allocations") or {})
              .get(strategy_id) or {})
    certified = record.get("symbol")
    if isinstance(certified, str) and certified:
        return resolve_parent(certified) == parent
    return any(resolve_parent(asset) == parent
               for asset in (portfolio.get("basket") or {}).get("assets") or [])


def attribute(row: dict, config: dict) -> tuple[str | None, str | None, str]:
    """
    `(strategy_id, portfolio_id, why)` for one row.

    The strategy the log NAMES wins. Without one, the account is resolved to a
    portfolio and its `active_strategies` are filtered to those whose basket
    holds the contract; exactly one candidate is an attribution and two are
    not. See the module docstring: a trade filed under the wrong strategy is a
    promotion decided on somebody else's P&L.
    """
    symbol = root_symbol(row.get("symbol"))
    portfolio_id = portfolio_for_account(row.get("account"), config)
    declared = str(row.get("strategy") or "").strip()

    if declared:
        if portfolio_id is None:
            # The strategy is named, so the trade is attributable; the account
            # only decides which risk envelope grades it, and the tracker
            # resolves that from the routing table anyway.
            return declared, None, "named by the log"
        return declared, portfolio_id, "named by the log"

    if portfolio_id is None:
        return None, None, (
            f"account {str(row.get('account') or '')!r} matches no portfolio "
            f"(known: {sorted(NT8_ACCOUNT_ALIASES)} plus portfolio ids and "
            f"target accounts) and the row names no strategy")

    portfolio = config["portfolios"][portfolio_id]
    if not symbol:
        return None, portfolio_id, "the row carries no instrument"
    candidates = [sid for sid in (portfolio.get("active_strategies") or [])
                  if _trades_symbol(portfolio, sid, symbol)]
    if len(candidates) == 1:
        return candidates[0], portfolio_id, (
            f"the only strategy on {portfolio_id} trading {symbol}")
    if not candidates:
        return None, portfolio_id, (
            f"{portfolio_id} has no active strategy trading {symbol}")
    return None, portfolio_id, (
        f"{portfolio_id} has {len(candidates)} strategies trading {symbol} "
        f"({sorted(candidates)}) and the row names none; refusing to guess")


# --------------------------------------------------------------------------
# money
# --------------------------------------------------------------------------

def _spec(symbol: str):
    from backtest.specs import get_spec
    return get_spec(symbol)


def contract_multiplier(symbol: str) -> float:
    """Dollars per point, from `backtest/specs.py`. Raises for an unknown
    symbol rather than assuming one - see the module docstring."""
    return float(_spec(symbol).multiplier)


def round_turn_cost(symbol: str) -> float:
    """Commission for one contract, both sides, from the spec."""
    return float(_spec(symbol).round_turn_cost)


def _priced(direction: str, entry: float, exit_: float, qty: float,
            symbol: str, cost_basis: str,
            log_commission: float | None) -> tuple[float, str]:
    """`(net_pnl, cost_note)` for one closed round turn."""
    sign = 1.0 if direction == LONG else -1.0
    gross = sign * (exit_ - entry) * contract_multiplier(symbol) * qty
    if cost_basis == "log":
        cost = log_commission or 0.0
        note = ("log commission" if log_commission is not None
                else "log, which carried none")
    elif cost_basis == "specs":
        cost = round_turn_cost(symbol) * qty
        note = "specs round turn"
    else:                                    # auto
        if log_commission is not None:
            cost, note = log_commission, "log commission"
        else:
            cost, note = round_turn_cost(symbol) * qty, "specs round turn"
    return gross - abs(cost), note


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------

def _stamp(row: dict, *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, datetime):
            return value.isoformat()
    return None


def trade_id(trade: dict) -> str:
    """
    A hash of what the trade IS, so the same round turn read twice is one
    trade. Prices and both timestamps are in it: two 1-lot MNQ trades on the
    same day at different prices are two trades, and must not collapse.
    """
    material = "|".join(str(trade.get(k, "")) for k in (
        "strategy_id", "symbol", "direction", "quantity",
        "entry_time", "exit_time", "entry_price", "exit_price"))
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _finish(trade: dict) -> dict:
    trade["trade_id"] = trade_id(trade)
    trade["status"] = "CLOSED"
    return trade


def _as_closed_trade(row: dict, strategy_id: str, symbol: str,
                     cost_basis: str) -> dict | None:
    """
    A TRADE-level row (NT8's Trades tab) as a closed trade, or None when the
    row is not one.

    A row qualifies when it carries a realised P&L and something to date the
    exit with. An explicitly NET figure is taken as net under `auto`; a gross
    `profit` has costs applied, because a promotion compared against a profit
    factor of 1.00 must not be decided on a figure that never paid commission.
    """
    net = parse_log_number(row.get("net_pnl"))
    gross = parse_log_number(row.get("gross_pnl"))
    if net is None and gross is None:
        return None
    exit_time = _stamp(row, "exit_time", "ts")
    if not exit_time:
        return None

    commission = parse_log_number(row.get("commission"))
    qty = parse_log_number(row.get("quantity")) or 1.0
    if net is not None and cost_basis != "specs":
        pnl, note = net, "log, declared net"
    elif net is not None:                        # cost_basis == "specs"
        pnl = net - round_turn_cost(symbol) * qty
        note = "specs round turn, over a declared-net figure"
    elif cost_basis == "log":
        pnl = gross - abs(commission or 0.0)
        note = ("log commission" if commission is not None
                else "log, which carried none")
    elif cost_basis == "specs":
        pnl, note = gross - round_turn_cost(symbol) * qty, "specs round turn"
    elif commission is not None:
        pnl, note = gross - abs(commission), "log commission"
    else:
        pnl, note = gross - round_turn_cost(symbol) * qty, "specs round turn"

    return _finish({
        "strategy_id": strategy_id,
        "symbol": symbol,
        "direction": _direction(row.get("action")) or "UNRECORDED",
        "quantity": qty,
        "entry_time": _stamp(row, "entry_time"),
        "exit_time": exit_time,
        "entry_price": parse_log_number(row.get("entry_price")),
        "exit_price": parse_log_number(row.get("exit_price")),
        "pnl": round(float(pnl), 2),
        "cost_basis": note,
        "source_log": row.get("_source_log"),
        "shape": "trade",
    })


def pair_fills(rows: list[dict], strategy_id: str, symbol: str,
               cost_basis: str) -> tuple[list[dict], list[dict], list[str]]:
    """
    `(closed_trades, still_open_lots, problems)` for one (strategy, symbol).

    FIFO, in timestamp order. A row with no usable price or side cannot be
    paired and is reported rather than dropped - an execution the recorder
    could not read is a trade the ledger is missing, and the count is how
    anybody notices.
    """
    trades: list[dict] = []
    problems: list[str] = []
    lots: list[dict] = []            # open lots, all the same direction

    def sort_key(row: dict) -> tuple:
        return (_stamp(row, "ts", "exit_time", "entry_time") or "", row["_row"])

    for row in sorted(rows, key=sort_key):
        if fill_status(row) is False:
            continue
        price = parse_log_number(row.get("realized_price"))
        side = _direction(row.get("action"))
        qty = parse_log_number(row.get("quantity"))
        stamp = _stamp(row, "ts", "exit_time")
        where = f"{row.get('_source_log')} row {row.get('_row')}"
        if price is None or side is None or not qty or qty <= 0:
            problems.append(
                f"{where}: unpairable ("
                + ", ".join(filter(None, [
                    None if price is not None else "no realised price",
                    None if side is not None else
                    f"side {str(row.get('action') or '')!r} not BUY/SELL",
                    None if qty and qty > 0 else "no positive quantity"]))
                + ")")
            continue

        remaining = qty
        while remaining > 0 and lots and lots[0]["direction"] != side:
            lot = lots[0]
            matched = min(remaining, lot["quantity"])
            pnl, note = _priced(lot["direction"], lot["price"], price, matched,
                                symbol, cost_basis,
                                _pro_rata(lot, row, matched, qty))
            trades.append(_finish({
                "strategy_id": strategy_id,
                "symbol": symbol,
                "direction": lot["direction"],
                "quantity": matched,
                "entry_time": lot["ts"],
                "exit_time": stamp,
                "entry_price": lot["price"],
                "exit_price": price,
                "pnl": round(float(pnl), 2),
                "cost_basis": note,
                "source_log": row.get("_source_log"),
                "shape": "paired",
            }))
            lot["quantity"] -= matched
            remaining -= matched
            if lot["quantity"] <= 0:
                lots.pop(0)

        if remaining > 0:
            lots.append({"direction": side, "quantity": remaining,
                         "price": price, "ts": stamp,
                         "commission": parse_log_number(row.get("commission")),
                         "row_quantity": qty})

    return trades, lots, problems


def _pro_rata(lot: dict, exit_row: dict, matched: float,
              exit_qty: float) -> float | None:
    """
    The log's own commission for the matched portion, both sides, or None.

    Allocated by size: a commission column describes the whole row, and a
    1-lot exit against a 3-lot entry paid a third of the entry's fee. None
    when either side carried no figure, so `_priced` falls back to the spec
    rather than charging half a round turn.
    """
    entry_fee = lot.get("commission")
    exit_fee = parse_log_number(exit_row.get("commission"))
    if entry_fee is None or exit_fee is None:
        return None
    entry_qty = lot.get("row_quantity") or matched
    return (abs(entry_fee) * matched / entry_qty
            + abs(exit_fee) * matched / (exit_qty or matched))


# --------------------------------------------------------------------------
# the pass
# --------------------------------------------------------------------------

def build_trades(rows: Iterable[dict], config: dict,
                 cost_basis: str = "auto") -> dict[str, Any]:
    """
    Every row -> `{strategy_id: [closed trades]}`, plus what could not be used.

    Trade-level rows are taken as they are; execution rows are paired per
    (strategy, symbol). The two shapes can appear in the same run - one export
    per tab is a normal NT8 configuration - and a row is a trade row iff it
    carries a realised P&L.
    """
    if cost_basis not in COST_BASES:
        raise RecorderError(
            f"cost_basis must be one of {COST_BASES}; got {cost_basis!r}")

    by_strategy: dict[str, list[dict]] = {}
    portfolios: dict[str, str] = {}
    unattributed: list[dict] = []
    unpriceable: dict[str, int] = {}
    problems: list[str] = []
    to_pair: dict[tuple[str, str], list[dict]] = {}
    open_lots: dict[str, int] = {}

    for row in rows:
        strategy_id, portfolio_id, why = attribute(row, config)
        symbol = root_symbol(row.get("symbol"))
        if strategy_id is None:
            unattributed.append({
                "source_log": row.get("_source_log"), "row": row.get("_row"),
                "symbol": symbol, "account": row.get("account"),
                "reason": why})
            continue
        if portfolio_id and strategy_id not in portfolios:
            portfolios[strategy_id] = portfolio_id
        if not symbol:
            unattributed.append({
                "source_log": row.get("_source_log"), "row": row.get("_row"),
                "symbol": "", "account": row.get("account"),
                "reason": "the row carries no instrument"})
            continue
        try:
            contract_multiplier(symbol)
        except KeyError:
            unpriceable[symbol] = unpriceable.get(symbol, 0) + 1
            continue

        trade = _as_closed_trade(row, strategy_id, symbol, cost_basis)
        if trade is not None:
            by_strategy.setdefault(strategy_id, []).append(trade)
        else:
            to_pair.setdefault((strategy_id, symbol), []).append(row)

    for (strategy_id, symbol), group in sorted(to_pair.items()):
        trades, lots, group_problems = pair_fills(group, strategy_id, symbol,
                                                  cost_basis)
        if trades:
            by_strategy.setdefault(strategy_id, []).extend(trades)
        if lots:
            open_lots[f"{strategy_id}/{symbol}"] = int(sum(
                lot["quantity"] for lot in lots))
        problems.extend(group_problems)

    for trades in by_strategy.values():
        trades.sort(key=lambda t: (str(t.get("exit_time") or ""),
                                   t["trade_id"]))
    _stamp_sessions(by_strategy)

    return {
        "trades": by_strategy,
        "portfolios": portfolios,
        "unattributed": unattributed,
        "unpriceable_symbols": unpriceable,
        "open_positions": open_lots,
        "problems": problems,
        "cost_basis": cost_basis,
    }


def _stamp_sessions(by_strategy: dict[str, list[dict]]) -> None:
    """
    Write each trade's CME session date onto it, from its exit stamp.

    `backtest.event_calendar.session_date` owns the 18:00 ET roll and is
    imported rather than reimplemented - the promotion daemon prefers an
    explicit `session_date` precisely so the recorder can settle it once,
    against the same calendar the backtests attribute trades with. Pandas is
    imported lazily because a caller that only wants the pairing should not
    pay for it, and a session date that cannot be derived is left off rather
    than guessed.
    """
    stamped = [(trades, index, str(trade["exit_time"]))
               for trades in by_strategy.values()
               for index, trade in enumerate(trades)
               if isinstance(trade.get("exit_time"), str) and trade["exit_time"]]
    if not stamped:
        return
    try:
        from backtest.event_calendar import session_date
        days = session_date([stamp for _, _, stamp in stamped])
    except Exception:                                    # noqa: BLE001
        return
    for (trades, index, _), day in zip(stamped, days):
        trades[index]["session_date"] = day.strftime("%Y-%m-%d")


def merge_into_ledger(ledger: dict, built: dict[str, Any],
                      now: str | None = None) -> dict[str, Any]:
    """
    Fold the built trades into the ledger IN PLACE, and report what changed.

    New trades are appended by `trade_id`; one already there is skipped, which
    is what makes a re-read of a growing export idempotent. A GRADUATED entry
    is refused - see the module docstring.
    """
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    strategies = ledger.setdefault("strategies", {})
    added: dict[str, int] = {}
    skipped: dict[str, int] = {}
    created: list[str] = []
    refused: list[str] = []

    for strategy_id, trades in sorted(built["trades"].items()):
        entry = strategies.get(strategy_id)
        if entry is None:
            entry = {
                "status": STATUS_INCUBATING,
                "portfolio": built["portfolios"].get(strategy_id),
                "recorded_by": RECORDER_TAG,
                "trades": [],
            }
            strategies[strategy_id] = entry
            created.append(strategy_id)
        if not isinstance(entry, dict):
            refused.append(f"{strategy_id}: the ledger entry is not an object")
            continue
        if str(entry.get("status", STATUS_INCUBATING)).upper() == STATUS_GRADUATED:
            refused.append(
                f"{strategy_id}: already {STATUS_GRADUATED}; its forward "
                f"trades are being taken on a prop account and are not "
                f"incubation evidence")
            continue

        existing = entry.setdefault("trades", [])
        if not isinstance(existing, list):
            refused.append(f"{strategy_id}: `trades` is not a list")
            continue
        seen = {t.get("trade_id") for t in existing if isinstance(t, dict)}
        fresh = [t for t in trades if t["trade_id"] not in seen]
        existing.extend(fresh)
        existing.sort(key=lambda t: (str(t.get("exit_time") or ""),
                                     str(t.get("trade_id") or "")))
        added[strategy_id] = len(fresh)
        skipped[strategy_id] = len(trades) - len(fresh)

        starts = [str(t.get("entry_time") or t.get("exit_time"))
                  for t in existing
                  if t.get("entry_time") or t.get("exit_time")]
        if starts:
            earliest = min(starts)
            declared = entry.get("started_at")
            # The EARLIER of the two survives. A `started_at` later than the
            # first trade shortens the window the trades themselves prove, and
            # the window is criterion one.
            entry["started_at"] = (min(str(declared), earliest)
                                   if isinstance(declared, str) and declared
                                   else earliest)
        if fresh:
            entry["last_recorded_at"] = stamp
            entry["recorded_by"] = RECORDER_TAG
        if entry.get("portfolio") is None:
            entry["portfolio"] = built["portfolios"].get(strategy_id)

    return {"added": added, "skipped": skipped, "created": created,
            "refused": refused, "stamp": stamp}


def record(log_target: str | Path = INCUBATOR_LOG_DIR,
           config: dict | None = None,
           config_path: str | Path | None = None,
           ledger_path: str | Path = DEFAULT_LEDGER_PATH,
           cost_basis: str = "auto",
           write: bool = False) -> dict[str, Any]:
    """
    One pass: read the exports, build the trades, merge, and write if asked.

    `write=False` is the default and does everything except the write, so the
    same call that a cron job makes with `--write` can be run first to see
    what it would do. The ledger is written atomically through the promotion
    daemon's own writer.
    """
    if config is None:
        from portfolio.config_loader import load_portfolio_config
        config = load_portfolio_config(config_path) if config_path else \
            load_portfolio_config()

    logs = discover_logs(log_target)
    rows: list[dict] = []
    read_errors: list[str] = []
    for log in logs:
        try:
            rows.extend(parse_rows(log))
        except (OSError, ValueError) as exc:
            read_errors.append(f"{log.name}: {type(exc).__name__}: {exc}")

    built = build_trades(rows, config, cost_basis=cost_basis)
    ledger = load_ledger(ledger_path)
    merged = merge_into_ledger(ledger, built)

    written = None
    if write and (any(merged["added"].values()) or merged["created"]):
        written = str(write_ledger(ledger, ledger_path))

    return {
        "logs": [str(p) for p in logs],
        "rows": len(rows),
        "read_errors": read_errors,
        "built": built,
        "merged": merged,
        "ledger": ledger,
        "written": written,
        "wrote": written is not None,
    }
