"""
The portfolio engine: turn many strategies' signals into one net position per
basket, size it from current volatility, and hand CrossTrade an order payload
per account.

Location:  ~/src/trading/portfolio/portfolio_manager.py

THE THREE STEPS, AND WHAT EACH ONE IS RESPONSIBLE FOR
=====================================================
    aggregate_signals   many signals -> ONE net position per (portfolio,
                        symbol). Opposing signals cancel; reinforcing ones
                        stack. No prices, no sizes, no accounts touched.
    build_order_plan    net positions + current volatility and regime -> a
                        decision per position, INCLUDING the ones it declines
                        and why.
    build_order_payloads  the same, reduced to the wire format.

The plan is the interesting one and it is separate from the payloads
deliberately. A list of payloads cannot say what it left out: a basket standing
down because the market is in the wrong regime and a basket with no signal at
all produce the same empty list, and those are completely different facts about
a trading day. `build_order_plan` records every skip with its reason;
`build_order_payloads` is `[r["payload"] for r in plan if r["payload"]]`, which
is what the spec asks for and what goes on the wire.

WHAT THIS DOES NOT DO
=====================
**No risk management.** No drawdown lockout, no daily loss cap, no prop-firm
rule, no kill switch. Those are CrossTrade NAM's, enforced against a live
account balance this process cannot see — see `portfolio/config_loader.py` for
the separation and why the drawdown figures in the config are a specification
rather than a gate. Nothing here reads a balance or a P&L.

**No knowledge of what is already open.** This builds the order implied by
today's signals; it does not diff against a current position, and it never
emits `FLATTEN`. A net that cancels to zero is recorded as a skip, not turned
into a close — the manager does not know whether anything is open to close, and
a FLATTEN sent on that assumption would shut positions it never opened. Whoever
holds the position state does that reconciliation.

**No order state.** Nothing here is idempotent across calls, nothing is
retried, and nothing is deduplicated against orders already sent. Calling it
twice builds the same payloads twice.

TWO DECISIONS THE SPECIFICATION LEFT OPEN
=========================================
1. **CONVICTION SCALES THE SIZE, AND THE CLAMP BOUNDS THE BREACH.** Two
   strategies both long MNQ net to +2. The size could be read two ways: the
   risk budget is FIXED per portfolio, so +2 could mean "same size, more
   confidence" — or stacking could mean what the word says and trade twice the
   contracts.

   This module multiplies: `contracts = clamp(unit_size x |net_units|)`. That
   is what makes "reinforcing signals stack" observable in the order rather
   than only in a field nobody reads. THE COST IS REAL AND IS REPORTED: two
   units risk twice the configured `fixed_risk_budget_usd`, so every record
   carries `risk_usd` beside `risk_budget_usd` and a `budget_breached` flag.
   `max_contracts` (5) is what bounds it — which is the clamp doing the job the
   config gave it, not an accident.

2. **A SYMBOL IN THE WRONG REGIME IS STOOD DOWN, NOT SIZED SMALLER.** Each
   portfolio declares the quadrants it is meant to trade, and Stage 1's
   designation is a claim about ONE environment. Trading the basket outside it
   is trading a strategy in the environment nobody certified — so a symbol
   whose current quadrant is not in the portfolio's permitted set produces no
   order at all, with `skipped_reason` naming the quadrant. That is the live
   counterpart of the regime firewall, and it is the whole point of carrying
   `regime_quadrants` on the portfolio.

   A symbol with NO regime reading is also stood down, for the same reason a
   missing ATR is: an unknown environment is not a permitted one. The two are
   distinguished in the reason string, because they are fixed by different
   work — one is a gap in the regime feed, the other is a market this basket is
   not for.

THE PAYLOAD IS THE DISPATCHER'S, NOT A SECOND COPY
==================================================
Every payload is built by `live.dispatcher.format_crosstrade_payload`, which is
the only place in this repository that formats an order. It validates the
action, refuses a fractional or non-positive quantity and rejects an order type
that carries no price. A second formatter here would be free to disagree with
it about any of those, and the disagreement would be discovered by a broker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live.dispatcher import format_crosstrade_payload            # noqa: E402
from portfolio.config_loader import (DEFAULT_CONFIG_PATH,        # noqa: E402
                                     INCUBATOR_ACCOUNT_TYPE,
                                     PROP_ACCOUNT_TYPE,
                                     PortfolioConfigError,
                                     canonical_quadrant,
                                     get_portfolio_for_strategy,
                                     load_portfolio_config)
from portfolio.volatility_sizer import size_detail               # noqa: E402

LONG, SHORT, FLAT = "long", "short", "flat"
DIRECTIONS = (LONG, SHORT, FLAT)
_SIGN = {LONG: 1, SHORT: -1, FLAT: 0}
# CrossTrade's own words for the two sides. FLATTEN is deliberately absent —
# see "No knowledge of what is already open" in the module docstring.
_ACTION = {1: "BUY", -1: "SELL"}


#: What a flatten intent looks like. FLATTEN carries no side and no quantity -
#: see `realtime.crosstrade_formatter.format_flatten_command` for why a guessed
#: side does not close a position, it opens the opposite one.
FLATTEN = "FLATTEN"


class PositionBook:
    """
    What THIS PROCESS opened, and nothing else.

    THE NARROWNESS IS THE SAFETY PROPERTY, NOT A LIMITATION
    =======================================================
    `PortfolioManager` still does not know what is open at the broker, and this
    class does not pretend to. It records only the fills this process
    dispatched, so a FLATTEN can be emitted for a position we can point at
    having opened - and never for one we merely infer. An account is shared: a
    blind flatten closes whatever is on it, including a position placed by
    NinjaTrader by hand, by a previous run of this loop, or by another tool.
    Closing someone else's position is silent, immediate and unrecoverable.

    **It is deliberately NOT persisted.** A restart therefore begins believing
    it holds nothing, which is the safe direction to be wrong in: the loop
    declines to flatten a position it cannot vouch for and says so, rather than
    flattening one it has no record of opening. The reconciliation that WOULD
    make persistence safe - reading real positions back from CrossTrade -
    does not exist yet, and a file that looks like position state would be
    trusted as though it did.

    WHY THE KEY IS (portfolio, symbol) AND NOT (strategy, symbol)
    =============================================================
    `aggregate_signals` nets many strategies into ONE position per
    (portfolio, symbol). Two strategies long NQ hold one position between them,
    so an exit from one of them is not a flatten - it is a smaller net. Keying
    the book by strategy would let the first exit close a position the second
    strategy is still in, which reads on the console as an orderly exit and is
    a silent liquidation of somebody else's trade.
    """

    def __init__(self) -> None:
        self._open: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _key(portfolio_id: str, symbol: str) -> tuple[str, str]:
        return (str(portfolio_id), str(symbol).upper())

    def record_fill(self, portfolio_id: str, symbol: str, direction: str,
                    quantity: int, strategies: list[str] | None = None) -> dict:
        """
        Record an order this process dispatched.

        Called AFTER a successful dispatch, never before: a book updated on
        intent believes it holds a position that a refused or failed order
        never opened, and the next exit then flattens an account that is
        already flat - which on a shared account is not a no-op.
        """
        if direction not in (LONG, SHORT):
            raise PortfolioError(
                f"record_fill got direction {direction!r}; only {LONG} and "
                f"{SHORT} open a position. FLAT closes one - use record_flat.")
        if int(quantity) <= 0:
            raise PortfolioError(
                f"record_fill got quantity {quantity!r}; an order that moved "
                f"no contracts did not open a position")
        record = {"portfolio_id": str(portfolio_id),
                  "symbol": str(symbol).upper(),
                  "direction": direction, "quantity": int(quantity),
                  "strategies": sorted(set(strategies or []))}
        self._open[self._key(portfolio_id, symbol)] = record
        return dict(record)

    def record_flat(self, portfolio_id: str, symbol: str) -> dict | None:
        """Forget a position, after a flatten this process dispatched."""
        return self._open.pop(self._key(portfolio_id, symbol), None)

    def is_open(self, portfolio_id: str, symbol: str) -> bool:
        return self._key(portfolio_id, symbol) in self._open

    def direction(self, portfolio_id: str, symbol: str) -> str:
        """`LONG`, `SHORT`, or `FLAT` when this process holds no record."""
        rec = self._open.get(self._key(portfolio_id, symbol))
        return rec["direction"] if rec else FLAT

    def get(self, portfolio_id: str, symbol: str) -> dict | None:
        rec = self._open.get(self._key(portfolio_id, symbol))
        return dict(rec) if rec else None

    def open_positions(self) -> list[dict]:
        """Every tracked position, portfolio-major, for the cycle report."""
        return [dict(v) for _, v in sorted(self._open.items())]

    # ------------------------------------------------------------------
    def _owns(self, portfolio_id: str, symbol: str,
              strategy_id: object) -> bool:
        """
        Is `strategy_id` one of the strategies this position was opened by?

        TRUE WHEN THE OWNER SET IS UNKNOWN. A position recorded without
        contributors - an older record, or a fill whose plan carried none -
        must stay closeable, because a flatten that cannot fire strands
        inventory and that is the expensive direction to be wrong in.
        """
        held = self._open.get(self._key(portfolio_id, symbol)) or {}
        owners = held.get("strategies") or []
        if not owners:
            return True
        return str(strategy_id) in set(owners)

    def plan_exits(self, exit_signals: list[dict],
                   net_positions: dict | None = None) -> list[dict]:
        """
        Turn exit signals into FLATTEN intents, and refuse the rest.

        An intent is emitted only when BOTH hold:

          1. this process has a recorded open position for that
             (portfolio, symbol), and
          2. no strategy still wants one there this cycle.

        The second condition is what stops one strategy's exit closing a
        position another is still in. `net_positions` is
        `aggregate_signals`' output; a (portfolio, symbol) with a non-FLAT net
        in it has a live claim, so the exit reduces that claim and the netting
        will size it - it is not a flatten. With no `net_positions` supplied
        nothing is assumed to be claimed, because the alternative is assuming a
        claim exists and never flattening at all.

        Every refusal is RETURNED with its reason rather than dropped. "no
        tracked position" and "another strategy still holds it" are different
        facts about an account, and an exit that produced no order must never
        be indistinguishable from an exit that was never signalled.
        """
        claimed: set[tuple[str, str]] = set()
        for pid, symbols in (net_positions or {}).items():
            for symbol, record in (symbols or {}).items():
                direction = (record or {}).get("direction", FLAT) \
                    if isinstance(record, dict) else record
                if direction in (LONG, SHORT):
                    claimed.add(self._key(pid, symbol))

        intents: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for signal in exit_signals:
            pid = str(signal.get("portfolio_id") or "")
            symbol = str(signal.get("symbol") or "").upper()
            key = self._key(pid, symbol)
            base = {"portfolio_id": pid, "symbol": symbol,
                    "strategy_id": signal.get("strategy_id"),
                    "action": FLATTEN, "emit": False, "reason": None}

            if not self.is_open(pid, symbol):
                base["reason"] = (
                    "no position opened by this process for "
                    f"{pid}/{symbol}; declining to flatten an account this "
                    "loop cannot vouch for")
            elif not self._owns(pid, symbol, signal.get("strategy_id")):
                # THE CROSS-BUCKET GUARD. `claimed` above is built from ONE
                # call's `net_positions`, and `master_live` calls the loop once
                # per TIMEFRAME BUCKET against a book that is shared by all of
                # them. So a 15m strategy's exit arrives in a cycle where the
                # 30m strategies that actually opened the position were never
                # evaluated, their claim is absent, and the pair looks
                # unclaimed. Measured on MES 2026-09-02 15:30-15:59: the 15m
                # bucket flattened the position the 30m bucket had opened
                # seconds earlier, every 60-second cycle, and the 30m bucket
                # re-opened it on the next pass - 30 entries, no holds, while
                # the identical setup an hour earlier with no 15m exit signal
                # held correctly 13 times running.
                #
                # A flatten closes the WHOLE netted position, so the only
                # strategy entitled to ask for one is a strategy that is in it.
                # An unknown owner set falls through to the claim check
                # unchanged: refusing there would strand inventory this process
                # cannot otherwise close, which is the failure direction that
                # costs money.
                held = self.get(pid, symbol) or {}
                base["reason"] = (
                    f"{signal.get('strategy_id')} did not open "
                    f"{pid}/{symbol} - it is held by "
                    f"{held.get('strategies')}, and a flatten closes the whole "
                    f"netted position. An exit from a strategy outside that "
                    f"set is an exit from a position it is not in, which on a "
                    f"multi-timeframe loop is one bucket closing another "
                    f"bucket's trade.")
            elif key in claimed:
                base["reason"] = (
                    f"another strategy still holds {pid}/{symbol} this cycle; "
                    "the exit reduces the net rather than flattening it")
            elif key in seen:
                base["reason"] = (
                    f"{pid}/{symbol} already has a flatten intent this cycle; "
                    "a flatten closes the whole position once")
            else:
                held = self.get(pid, symbol) or {}
                base.update({"emit": True, "reason": "position tracked open "
                             f"({held.get('direction')} x"
                             f"{held.get('quantity')}) and unclaimed",
                             "held_direction": held.get("direction"),
                             "held_quantity": held.get("quantity")})
                seen.add(key)
            intents.append(base)
        return intents


class PortfolioError(RuntimeError):
    """A signal or a net position that cannot be routed as given."""


class PortfolioManager:
    """
    One instance per configuration. Stateless between calls except for the
    config it loaded.

    The config is read ONCE, at construction, and every accessor on it returns
    a deep copy — so a caller mutating a returned record cannot change where
    the next order is routed. Rebuild the manager to pick up an edited file.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH,
                 config: dict | None = None,
                 strict_specs: bool = True) -> None:
        self.config_path = config_path
        self.config = (config if config is not None
                       else load_portfolio_config(config_path,
                                                  strict_specs=strict_specs))
        self.portfolios = self.config["portfolios"]

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _resolve_portfolio(self, signal: dict) -> str:
        """
        Which portfolio a signal belongs to.

        An explicit `portfolio_id` wins — a caller that already knows the
        account is not second-guessed. Otherwise the strategy is looked up in
        `active_strategies` on the requested track, which RAISES when it is
        unassigned: there is no default, because a portfolio inferred from a
        strategy's name would be a live account chosen by a rule nobody wrote
        down.
        """
        explicit = signal.get("portfolio_id")
        if explicit:
            if explicit not in self.portfolios:
                raise PortfolioError(
                    f"signal names portfolio {explicit!r}, which is not in "
                    f"{sorted(self.portfolios)}")
            return str(explicit)

        strategy_id = signal.get("strategy_id")
        if not strategy_id:
            raise PortfolioError(
                f"signal has neither portfolio_id nor strategy_id, so nothing "
                f"can route it: {signal!r}")
        is_incubating = bool(signal.get("is_incubating", True))
        try:
            return get_portfolio_for_strategy(str(strategy_id),
                                              is_incubating=is_incubating,
                                              config=self.config)
        except PortfolioConfigError as exc:
            # Re-raised as this module's own type, with the loader's message
            # intact. A caller of `aggregate_signals` is handling ROUTING
            # failures — an unknown account, an unassigned strategy, a symbol
            # outside its basket — and should catch one exception type for all
            # of them rather than knowing which layer noticed. The chain keeps
            # the original for anyone who does care.
            raise PortfolioError(str(exc)) from exc

    # ------------------------------------------------------------------
    # 1. Aggregation
    # ------------------------------------------------------------------
    def aggregate_signals(self, raw_signals: list[dict]) -> dict:
        """
        Net many strategies' signals into one position per (portfolio, symbol).

        Each signal is `{"symbol", "direction"}` plus one of `portfolio_id` or
        `strategy_id`:

            {"strategy_id": "double_rsi_macd_scalp", "symbol": "MNQ",
             "direction": "long", "is_incubating": True, "units": 1}

        `direction` is `long`, `short` or `flat`; `units` defaults to 1 and is
        the signal's weight. The net is the signed sum, so two longs make +2
        (reinforcing) and a long against a short makes 0 (opposing).

        Returns `{portfolio_id: {symbol: record}}`. Nested rather than keyed on
        a tuple so the result is JSON-serialisable and reads the way the
        accounts do.

        A symbol NOT in the portfolio's basket RAISES. Netting it would
        aggregate a position onto an account that does not trade that contract,
        and the correlation partition the four baskets exist for would be
        fictional — the whole point of a basket is that it names what may be
        held there.

        FLAT SIGNALS ARE COUNTED AND CONTRIBUTE ZERO. They appear in
        `signals` and in `contributors`, so "three strategies looked and none
        wanted a position" stays distinguishable from "nothing ran".
        """
        if not isinstance(raw_signals, Iterable) or isinstance(raw_signals,
                                                               (str, bytes)):
            raise PortfolioError(
                f"raw_signals must be a list of signal dicts; got "
                f"{type(raw_signals).__name__}")

        net: dict[str, dict[str, dict]] = {}
        for i, signal in enumerate(raw_signals):
            if not isinstance(signal, dict):
                raise PortfolioError(
                    f"signal {i} is {type(signal).__name__}, not a dict")

            direction = str(signal.get("direction", "")).strip().lower()
            if direction not in DIRECTIONS:
                raise PortfolioError(
                    f"signal {i} has direction {signal.get('direction')!r}; "
                    f"expected one of {list(DIRECTIONS)}")

            units = signal.get("units", 1)
            if isinstance(units, bool) or not isinstance(units, int):
                raise PortfolioError(
                    f"signal {i}: units must be a whole number; got {units!r}")
            if units < 0:
                raise PortfolioError(
                    f"signal {i}: units must be >= 0; got {units!r}. A "
                    f"negative weight would invert the direction it was "
                    f"paired with, which is a second way to say `short` and "
                    f"the one nobody reads.")

            symbol = str(signal.get("symbol", "")).strip().upper()
            if not symbol:
                raise PortfolioError(f"signal {i} has no symbol")

            pid = self._resolve_portfolio(signal)
            basket = self.portfolios[pid]["basket"]["assets"]
            if symbol not in basket:
                raise PortfolioError(
                    f"signal {i} routes {symbol} to {pid}, whose basket is "
                    f"{basket}. A position netted onto an account that does "
                    f"not trade the contract makes the correlation partition "
                    f"fictional.")

            book = net.setdefault(pid, {})
            record = book.setdefault(symbol, {
                "portfolio_id": pid,
                "symbol": symbol,
                "net_units": 0,
                "long_units": 0,
                "short_units": 0,
                "signals": 0,
                "contributors": [],
            })
            sign = _SIGN[direction]
            record["net_units"] += sign * units
            if sign > 0:
                record["long_units"] += units
            elif sign < 0:
                record["short_units"] += units
            record["signals"] += 1
            record["contributors"].append({
                "strategy_id": signal.get("strategy_id"),
                "direction": direction,
                "units": units,
            })

        for book in net.values():
            for record in book.values():
                n = record["net_units"]
                record["direction"] = LONG if n > 0 else SHORT if n < 0 else FLAT
        return net

    # ------------------------------------------------------------------
    # 2. Sizing and the plan
    # ------------------------------------------------------------------
    @staticmethod
    def _regime_reading(current_regimes: dict, symbol: str) -> dict | None:
        """
        `{"atr_14": float, "quadrant": "Q1".."Q4"}` for one symbol, or None.

        Accepts the quadrant as this repository's id (`Q1`) or as a schema
        label (`Q1_HIGH_VOL_TREND`), resolving the second through
        `canonical_quadrant` — the only supported translation. A caller holding
        Stage 1's handoff has the id; a caller reading the portfolio config has
        the label; neither should have to know which the other used.
        """
        reading = current_regimes.get(symbol) if current_regimes else None
        if reading is None:
            return None
        if not isinstance(reading, dict):
            raise PortfolioError(
                f"current_regimes[{symbol!r}] must be a dict carrying at "
                f"least `atr_14` and `quadrant`; got "
                f"{type(reading).__name__}. Sizing needs the ATR and the gate "
                f"needs the quadrant, and a bare quadrant string cannot carry "
                f"both.")
        out = dict(reading)
        quadrant = out.get("quadrant")
        if isinstance(quadrant, str) and "_" in quadrant:
            out["quadrant"] = canonical_quadrant(quadrant)
        return out

    def build_order_plan(self, net_positions: dict,
                         current_regimes: dict | None = None) -> list[dict]:
        """
        One decision per net position, INCLUDING the declined ones.

        `current_regimes` is `{symbol: {"atr_14": float, "quadrant": "Q1"}}`.
        The ATR sizes the position and the quadrant gates it; a symbol missing
        from the mapping is stood down rather than sized on a guess.

        Every record carries `payload` (None when declined), `skipped_reason`
        (None when not), the full `sizing` record and the `regime` reading.
        """
        current_regimes = current_regimes or {}
        plan: list[dict] = []

        for pid in sorted(net_positions):
            portfolio = self.portfolios.get(pid)
            if portfolio is None:
                raise PortfolioError(
                    f"net positions name portfolio {pid!r}, which is not in "
                    f"{sorted(self.portfolios)}")
            risk = portfolio["risk_profile"]
            clamp = risk["clamping"]
            permitted = list(portfolio["derived"]["canonical_quadrants"])

            for symbol in sorted(net_positions[pid]):
                record = dict(net_positions[pid][symbol])
                net_units = int(record.get("net_units", 0))
                base = {
                    "portfolio_id": pid,
                    "account": portfolio["target_account"],
                    "account_type": portfolio["account_type"],
                    "symbol": symbol,
                    "net_units": net_units,
                    "direction": record.get("direction", FLAT),
                    "contributors": record.get("contributors", []),
                    "permitted_quadrants": permitted,
                    "payload": None,
                    "sizing": None,
                    "regime": None,
                    "skipped_reason": None,
                }

                if net_units == 0:
                    # Netted flat. NOT a FLATTEN: this manager does not know
                    # whether anything is open, and closing on that assumption
                    # would shut positions it never opened.
                    base["skipped_reason"] = (
                        f"signals netted flat ({record.get('long_units', 0)} "
                        f"long against {record.get('short_units', 0)} short); "
                        f"no order, and no FLATTEN — position state is not "
                        f"known here")
                    plan.append(base)
                    continue

                reading = self._regime_reading(current_regimes, symbol)
                base["regime"] = reading
                if reading is None:
                    base["skipped_reason"] = (
                        f"no regime reading for {symbol}: an unknown "
                        f"environment is not a permitted one, and the ATR is "
                        f"what sizes the position")
                    plan.append(base)
                    continue

                quadrant = reading.get("quadrant")
                if quadrant not in permitted:
                    base["skipped_reason"] = (
                        f"{symbol} is in {quadrant}, and {pid} trades "
                        f"{permitted}. Standing down rather than sizing "
                        f"smaller: outside its quadrant the basket is a "
                        f"strategy in the environment nobody certified.")
                    plan.append(base)
                    continue

                atr = reading.get("atr_14")
                if atr is None:
                    base["skipped_reason"] = (
                        f"regime reading for {symbol} carries no atr_14, so "
                        f"the position cannot be sized")
                    plan.append(base)
                    continue

                sizing = size_detail(
                    symbol, atr,
                    risk_budget_usd=float(risk["fixed_risk_budget_usd"]),
                    min_contracts=int(clamp["min_contracts"]),
                    max_contracts=int(clamp["max_contracts"]),
                    stop_atr_mult=float(reading.get("stop_atr_mult", 1.0)),
                    config=self.config)

                # CONVICTION SCALING, then the clamp — see decision 1 in the
                # module docstring. The unit size is what the budget buys; the
                # net units are how many strategies agreed.
                conviction = abs(net_units)
                scaled = min(int(clamp["max_contracts"]),
                             sizing["contracts"] * conviction)
                sizing = {
                    **sizing,
                    "unit_contracts": sizing["contracts"],
                    "conviction": conviction,
                    "contracts": int(scaled),
                    "risk_usd": scaled * sizing["risk_per_contract_usd"],
                }
                sizing["budget_breached"] = bool(
                    sizing["risk_usd"] > sizing["risk_budget_usd"] + 1e-9)
                base["sizing"] = sizing

                base["payload"] = format_crosstrade_payload(
                    symbol=symbol,
                    action=_ACTION[1 if net_units > 0 else -1],
                    quantity=int(scaled),
                    order_type="MARKET",
                    account_id=portfolio["target_account"])
                plan.append(base)

        return plan

    def build_order_payloads(self, net_positions: dict,
                             current_regimes: dict | None = None
                             ) -> list[dict]:
        """
        The CrossTrade payloads for every net position that survived sizing and
        the regime gate.

        Exactly the five keys `live.dispatcher.format_crosstrade_payload`
        produces — `account`, `action`, `symbol`, `orderType`, `quantity` — and
        nothing else, so the result can be POSTed as it stands. `account` is
        the portfolio's `target_account` - NinjaTrader's name for the account
        and NOT the portfolio id: `SimIncubator1`, `SimIncubator2`, `SimProp1`
        or `SimProp2`, for `Incubator-Odd`, `Incubator-Even`, `Prop-Odd` and
        `Prop-Even` respectively. The field is read from the routing table, so
        renaming an account there is the whole change; an order addressed to a
        portfolio id would be rejected by a broker that has no such account.

        A DECLINED POSITION IS SIMPLY ABSENT, which is why `build_order_plan`
        exists: an empty list here cannot distinguish a basket standing down
        outside its regime from a day with no signals, and those are different
        facts. Read the plan when the answer matters.
        """
        return [record["payload"]
                for record in self.build_order_plan(net_positions,
                                                    current_regimes)
                if record["payload"] is not None]

    # ------------------------------------------------------------------
    def describe_plan(self, plan: list[dict]) -> str:
        """One line per decision, orders and declines alike, for the console."""
        lines = []
        for r in plan:
            if r["payload"]:
                s = r["sizing"]
                flag = "  BUDGET BREACHED" if s["budget_breached"] else ""
                lines.append(
                    f"  {r['account']:<16} {r['payload']['action']:<5} "
                    f"{r['symbol']:<5} x{r['payload']['quantity']} "
                    f"(net {r['net_units']:+d}, {s['unit_contracts']}/unit, "
                    f"ATR {s['atr_14']:g}, risk ${s['risk_usd']:,.0f} of "
                    f"${s['risk_budget_usd']:,.0f}){flag}")
            else:
                lines.append(f"  {r['account']:<16} SKIP  {r['symbol']:<5} "
                             f"{r['skipped_reason']}")
        return "\n".join(lines) or "  no positions"


__all__ = ["PortfolioManager", "PortfolioError", "LONG", "SHORT", "FLAT",
           "PortfolioConfigError", "INCUBATOR_ACCOUNT_TYPE",
           "PROP_ACCOUNT_TYPE"]
