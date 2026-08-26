"""
backtest.specs - contract specifications.

Location:  ~/src/trading/backtest/specs.py

Multiplier, tick size, and tick value per contract. Needed to convert a price
move into dollars, which every cost model and position sizer depends on.

    dollars_per_point = multiplier
    dollars_per_tick  = tick_value = multiplier * tick_size

VERIFY THESE BEFORE TRUSTING A BACKTEST
---------------------------------------
These are hand-entered from published CME specs. They are the single most
consequential set of constants in the whole system - a wrong multiplier
silently scales every P&L figure for that symbol, and the backtest will look
perfectly plausible.

The `definition` schema at /mnt/backtest/reference/futures/definitions/
carries the authoritative values. `python -m backtest.specs` reconciles this
table against it.

Reconciled 2026-08-13 against definitions through 2026-08-09:

  - Every multiplier agrees with the exchange definition.
  - Two tick corrections applied, both of which had been overcharging
    slippage by 2x on recent data:
      ZT  1/128 -> 1/256   (CME cut the increment to 1/8 of 1/32 on
                            2019-01-14; $15.625/tick -> $7.8125)
      6A  0.0001 -> 0.00005 (halved 2020-11-23; $10/tick -> $5)
  - Six contracts have changed tick at some point (ZT, 6A, 6C, 6E, 6J, 6S,
    plus ETH widening). A single scalar cannot price slippage correctly
    across those - see TICK_HISTORY and ContractSpec.tick_size_array.
  - Coverage caveat: only 2026 definitions were downloaded for the 17
    symbols added on 2026-08-13, so their definition files cannot show a
    mid-history change. Their tick history came from the lake price grid
    instead. SI's definitions stop at 2016 and CL's at 2025-12.

Do not reconcile `multiplier` against Databento's `unit_of_measure_qty`
without checking `unit_of_measure` first; for Treasuries that field is the
face value and the multiplier is face/100. See verify_specs().

Commissions
-----------
NinjaTrader personal account, as of 2026-08:
    $0.35 per micro contract per side
    $1.29 per non-micro contract per side
plus exchange, NFA and clearing fees.

The defaults below add roughly $1.00/side for those fees. Replace with actual
figures from a statement once you have one - estimated costs are the most
common reason a backtest overstates performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFINITIONS = Path("/mnt/backtest/reference/futures/definitions")

# How to turn Databento's `unit_of_measure_qty` into dollars per point.
#
#   multiplier = unit_of_measure_qty * QUOTE_SCALE[unit_of_measure]
#
# `unit_of_measure_qty` is the contract size in its natural unit - 50 index
# points, 1000 barrels, 5000 bushels, $100,000 of face. Converting that to
# dollars per quoted point needs the quote convention, which the field itself
# does not carry, so it is spelled out here:
#
#   1.00  price is quoted directly in dollars per unit (ES, CL, GC, FX, crypto)
#   0.01  price is quoted in cents per unit (grains, livestock), or as a
#         percent of par (Treasuries) - a point is 1/100 of the unit
#
# Getting this wrong is a silent 100x error in one direction or the other, so
# an unrecognised unit is reported as UNVERIFIED rather than assumed to be 1.
QUOTE_SCALE: dict[str, float] = {
    # quoted in dollars per unit
    "IPNT": 1.0,        # index points - equity index
    "BBL": 1.0,         # barrels - crude
    "GAL": 1.0,         # gallons - RBOB, heating oil
    "MMBTU": 1.0,       # natural gas
    "TRYOZ": 1.0,       # troy ounces - metals
    "EUR": 1.0, "GBP": 1.0, "AUD": 1.0, "CAD": 1.0,
    "JPY": 1.0, "CHF": 1.0,     # FX - USD per unit of foreign currency
    "BTC": 1.0, "ETH": 1.0,     # crypto - USD per coin
    # quoted in cents per unit, or percent of par
    "USD": 0.01,        # Treasuries - qty is face value, price is % of par
    "BU": 0.01,         # bushels - grains, quoted in cents
    "LBS": 0.01,        # pounds - livestock, quoted in cents
}


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    name: str
    exchange: str
    multiplier: float      # dollars per full point
    tick_size: float       # minimum price increment
    commission: float      # per contract PER SIDE, incl. estimated fees
    is_micro: bool = False

    @property
    def tick_value(self) -> float:
        """Dollars per tick."""
        return self.multiplier * self.tick_size

    @property
    def round_turn_cost(self) -> float:
        """Commission for a complete round trip, one contract."""
        return self.commission * 2

    def slippage_dollars(self, ticks: float) -> float:
        """Cost of `ticks` of slippage, one contract, one side."""
        return ticks * self.tick_value

    # -- date-aware variants ----------------------------------------------
    # `tick_size` above is the CURRENT tick. A contract whose tick has changed
    # needs the tick that was in force on the bar being priced, or slippage is
    # wrong for part of the backtest. See TICK_HISTORY.

    def tick_size_array(self, ts) -> np.ndarray:
        """
        Tick size in force at each timestamp, as a float array.

        This is the one to use for a slippage PRICE adjustment: a slippage
        fraction is (ticks * tick_size) / price. Multiplier must not appear -
        see tick_value_array.
        """
        t = pd.to_datetime(pd.Series(ts).to_numpy(), utc=True)
        history = TICK_HISTORY.get(self.symbol)
        if not history:
            return np.full(len(t), float(self.tick_size))

        out = np.full(len(t), float(history[0][1]))
        for effective, tick in history:
            out[t >= pd.Timestamp(effective, tz="UTC")] = float(tick)
        return out

    def tick_value_array(self, ts) -> np.ndarray:
        """
        DOLLARS per tick at each timestamp: multiplier * tick_size_array.

        Use this to express slippage in dollars. Do NOT divide it by price to
        get a slippage fraction - a fraction lives in price space, so it is
        tick_size/price, not tick_value/price. Dividing tick_value by price
        overstates slippage by a factor of the multiplier (ES: 50x).
        """
        return self.multiplier * self.tick_size_array(ts)


# Commission estimate: NT rate per side plus ~$1.00 exchange/NFA/clearing.
_NON_MICRO = 1.29 + 1.00
_MICRO = 0.35 + 0.35

# EVERY SPEC BELOW IS A US EXCHANGE AND A DOLLAR MULTIPLIER, AND THAT IS LOAD
# BEARING. `multiplier` is dollars per full point, `commission` is dollars per
# side, and `config/portfolios.json` declares `base_currency: USD`. There is no
# currency conversion anywhere in this repository, so a non-USD contract has
# nothing to pass through and its multiplier would be a dollar figure that is
# not one.
#
# That is why FDAX (Eurex DAX, EUR 25/point) is NOT here even though the NT8
# publisher spools it: the decision, and what it would take to reverse it, are
# recorded in `realtime.contract_alias.NOT_TRADED`. `get_spec` raising for an
# unknown symbol is what keeps it un-tradeable in the meantime, and that raise
# is the feature - a fabricated multiplier would not fail, it would scale every
# position and every P&L by a constant with every log line reading correctly.

SPECS: dict[str, ContractSpec] = {
    # Equity index
    "ES":  ContractSpec("ES",  "E-mini S&P 500",        "CME",   50,     0.25,      _NON_MICRO),
    "NQ":  ContractSpec("NQ",  "E-mini Nasdaq-100",     "CME",   20,     0.25,      _NON_MICRO),
    "RTY": ContractSpec("RTY", "E-mini Russell 2000",   "CME",   50,     0.10,      _NON_MICRO),
    "YM":  ContractSpec("YM",  "E-mini Dow",            "CBOT",   5,     1.00,      _NON_MICRO),

    # Rates
    # Treasuries are quoted as a percent of par, so dollars_per_point =
    # face_value / 100 (ZT: $200k face -> 2000; the rest: $100k face -> 1000).
    # Databento's `unit_of_measure_qty` for these is the FACE VALUE, not the
    # multiplier - see verify_specs(), which corrects for it. Do not "fix"
    # these to 100000; that scales every rates P&L figure by 100x.
    #
    # ZT tick: CME cut the minimum increment from 1/4 to 1/8 of 1/32 effective
    # 2019-01-14 (confirmed in the definition files and in lake prices, which
    # are 100% on the 1/256 grid after that date and ~50% off the 1/128 grid).
    # 1/256 is correct for anything current; a ZT backtest that starts before
    # 2019-01-14 understates slippage over its early years.
    "ZT":  ContractSpec("ZT",  "2-Year T-Note",         "CBOT", 2000,    1/256,     _NON_MICRO),
    "ZF":  ContractSpec("ZF",  "5-Year T-Note",         "CBOT", 1000,    1/128,     _NON_MICRO),
    "ZN":  ContractSpec("ZN",  "10-Year T-Note",        "CBOT", 1000,    1/64,      _NON_MICRO),
    "ZB":  ContractSpec("ZB",  "30-Year T-Bond",        "CBOT", 1000,    1/32,      _NON_MICRO),

    # Energy
    "CL":  ContractSpec("CL",  "Crude Oil WTI",         "NYMEX", 1000,   0.01,      _NON_MICRO),
    "NG":  ContractSpec("NG",  "Natural Gas",           "NYMEX", 10000,  0.001,     _NON_MICRO),
    "RB":  ContractSpec("RB",  "RBOB Gasoline",         "NYMEX", 42000,  0.0001,    _NON_MICRO),
    "HO":  ContractSpec("HO",  "Heating Oil",           "NYMEX", 42000,  0.0001,    _NON_MICRO),

    # Metals
    "GC":  ContractSpec("GC",  "Gold",                  "COMEX", 100,    0.10,      _NON_MICRO),
    "SI":  ContractSpec("SI",  "Silver",                "COMEX", 5000,   0.005,     _NON_MICRO),
    "PL":  ContractSpec("PL",  "Platinum",              "NYMEX", 50,     0.10,      _NON_MICRO),

    # Grains
    "ZC":  ContractSpec("ZC",  "Corn",                  "CBOT",  50,     0.25,      _NON_MICRO),
    "ZW":  ContractSpec("ZW",  "Wheat",                 "CBOT",  50,     0.25,      _NON_MICRO),
    "ZS":  ContractSpec("ZS",  "Soybeans",              "CBOT",  50,     0.25,      _NON_MICRO),

    # Livestock
    "LE":  ContractSpec("LE",  "Live Cattle",           "CME",   400,    0.025,     _NON_MICRO),

    # FX
    "6E":  ContractSpec("6E",  "Euro FX",               "CME",   125000, 0.00005,   _NON_MICRO),
    "6B":  ContractSpec("6B",  "British Pound",         "CME",   62500,  0.0001,    _NON_MICRO),
    # 6A tick halved to 0.00005 on 2020-11-23 (tick value $10 -> $5). Both the
    # 2026 definition file and the lake price grid agree; the old $10 figure
    # overstated 6A slippage by 2x on everything after 2020.
    "6A":  ContractSpec("6A",  "Australian Dollar",     "CME",   100000, 0.00005,   _NON_MICRO),
    "6C":  ContractSpec("6C",  "Canadian Dollar",       "CME",   100000, 0.00005,   _NON_MICRO),
    "6J":  ContractSpec("6J",  "Japanese Yen",          "CME",   12500000, 0.0000005, _NON_MICRO),
    "6S":  ContractSpec("6S",  "Swiss Franc",           "CME",   125000, 0.00005,   _NON_MICRO),

    # Crypto
    "BTC": ContractSpec("BTC", "Bitcoin",               "CME",   5,      5.00,      _NON_MICRO),
    "ETH": ContractSpec("ETH", "Ether",                 "CME",   50,     0.50,      _NON_MICRO),

    # Micros - same markets at 1/10 size. Included for position sizing only;
    # backtest on the full-size symbol and scale.
    "MES": ContractSpec("MES", "Micro E-mini S&P 500",  "CME",    5,     0.25, _MICRO, True),
    "MNQ": ContractSpec("MNQ", "Micro E-mini Nasdaq",   "CME",    2,     0.25, _MICRO, True),
    "MCL": ContractSpec("MCL", "Micro Crude Oil",       "NYMEX", 100,    0.01, _MICRO, True),
    "MGC": ContractSpec("MGC", "Micro Gold",            "COMEX",  10,    0.10, _MICRO, True),
    "M2K": ContractSpec("M2K", "Micro E-mini Russell",  "CME",     5,    0.10, _MICRO, True),
    "MYM": ContractSpec("MYM", "Micro E-mini Dow",      "CBOT",    0.50, 1.00, _MICRO, True),
}


# Contracts whose tick size has changed, oldest first, as
# (effective_date_utc, tick_size). The last entry must equal the symbol's
# `tick_size` in SPECS above; verify_specs() enforces that.
#
# Two independent sources, because neither is sufficient alone:
#
#   ZT   from the definition files, which cover 2010-2026 for that symbol. The
#        date is the first definition record carrying the new tick - the
#        Sunday-evening session open (2019-01-13 17:05 UTC) for CME trade date
#        2019-01-14. Using the record date keeps that session's Sunday-evening
#        bars on the correct tick.
#
#   FX   from the lake price grid: the first session whose prices fall off the
#        old tick grid. Only 2026 definitions have been downloaded for these
#        symbols, so the definition files show a single regime and would imply
#        - wrongly - that the tick never moved. The price grid is the harder
#        evidence anyway: it is what actually traded.
#
# Every other symbol in SPECS was checked the same way, year by year, and is a
# single regime, so the date-aware path is a no-op for them. Re-run the check
# after extending history: scripts do not exist for this yet, see Open Tasks.
TICK_HISTORY: dict[str, list[tuple[str, float]]] = {
    "ZT":  [("2010-06-06", 1 / 128),  ("2019-01-13", 1 / 256)],
    "6J":  [("2010-06-06", 0.000001), ("2015-06-23", 0.0000005)],
    "6E":  [("2010-06-06", 0.0001),   ("2016-01-11", 0.00005)],
    "6C":  [("2010-06-06", 0.0001),   ("2016-07-11", 0.00005)],
    "6A":  [("2010-06-06", 0.0001),   ("2020-11-23", 0.00005)],
    "6S":  [("2010-06-06", 0.0001),   ("2022-05-02", 0.00005)],
    # ETH went the other way: it listed at a 0.25 tick and widened to 0.50.
    "ETH": [("2021-02-08", 0.25),     ("2021-12-06", 0.50)],
}


def get_spec(symbol: str) -> ContractSpec:
    if symbol not in SPECS:
        raise KeyError(
            f"No contract spec for {symbol!r}. Add it to backtest/specs.py - "
            f"a backtest cannot compute P&L without the multiplier."
        )
    return SPECS[symbol]


def specs_table() -> pd.DataFrame:
    """All specs as a DataFrame, for eyeballing."""
    rows = [{
        "symbol": s.symbol, "name": s.name, "exchange": s.exchange,
        "multiplier": s.multiplier, "tick_size": s.tick_size,
        "tick_value": round(s.tick_value, 4),
        "commission_per_side": s.commission,
        "round_turn": round(s.round_turn_cost, 2),
        "is_micro": s.is_micro,
    } for s in SPECS.values()]
    return pd.DataFrame(rows)


def verify_specs(definitions_dir: Path = DEFINITIONS) -> pd.DataFrame:
    """
    Reconcile the table above against Databento's `definition` schema.

    Returns a frame with one row per problem, `status` being either:

        MISMATCH   - our value disagrees with the exchange definition
        UNVERIFIED - no definition data downloaded, nothing was checked

    An empty frame means every symbol in SPECS was checked and agreed.
    UNVERIFIED rows are reported rather than skipped: silently passing over a
    symbol with no data is how a wrong multiplier survives a "clean" run.

    Reading `unit_of_measure_qty` correctly requires `unit_of_measure`. For
    index and physical contracts (IPNT, BBL, GAL, MMBTU, TRYOZ) it is the
    dollars-per-point multiplier directly. For Treasuries it is USD of face
    value, and the multiplier is face / 100, because the contract is quoted as
    a percent of par. Comparing the raw quantity against the multiplier reports
    every rates contract as wrong by exactly 100x.
    """
    if not definitions_dir.exists():
        raise FileNotFoundError(
            f"{definitions_dir} not found. The definition schema download may "
            f"still be running."
        )

    rows = []

    def problem(sym, status, field, ours, theirs, note=""):
        rows.append({"symbol": sym, "status": status, "field": field,
                     "ours": ours, "databento": theirs, "note": note})

    # Self-consistency: the last entry of a tick schedule is the current tick,
    # so it must equal the scalar in SPECS. Needs no downloaded data.
    for sym, history in TICK_HISTORY.items():
        if sym not in SPECS:
            problem(sym, "MISMATCH", "tick_history", None, None,
                    "TICK_HISTORY entry for a symbol not in SPECS")
            continue
        latest = history[-1][1]
        if abs(latest - SPECS[sym].tick_size) > 1e-12:
            problem(sym, "MISMATCH", "tick_history", SPECS[sym].tick_size,
                    latest, "last TICK_HISTORY entry != SPECS tick_size")

    for sym, spec in SPECS.items():
        sym_dir = definitions_dir / f"symbol={sym}"
        files = sorted(sym_dir.glob("*.parquet")) if sym_dir.exists() else []
        if not files:
            problem(sym, "UNVERIFIED", "-", None, None,
                    "no definition data downloaded")
            continue
        try:
            df = pd.read_parquet(files[-1])
        except Exception as e:
            problem(sym, "UNVERIFIED", "read", None, None, f"unreadable: {e}")
            continue

        # Outright futures only. Spreads and user-defined instruments carry
        # their own, finer, tick sizes and would look like a mismatch.
        if "instrument_class" in df.columns:
            df = df[df["instrument_class"] == "F"]
        if "user_defined_instrument" in df.columns:
            df = df[df["user_defined_instrument"] == "N"]
        if df.empty:
            problem(sym, "UNVERIFIED", "-", None, None,
                    "no outright futures records")
            continue

        last = df.iloc[-1]          # most recent definition = current spec

        # --- tick size --------------------------------------------------
        if "min_price_increment" in df.columns:
            theirs = last["min_price_increment"]
            # Databento encodes prices as fixed-point integers, 1e-9 scale.
            # Already decoded to float in these files, so this is a guard for
            # a future re-pull that keeps the raw form.
            if theirs and abs(theirs) > 1e3:
                theirs = theirs / 1e9
            if abs(float(theirs) - float(spec.tick_size)) > 1e-12:
                problem(sym, "MISMATCH", "tick_size", spec.tick_size, theirs)

        # --- multiplier -------------------------------------------------
        if "unit_of_measure_qty" in df.columns:
            qty = last["unit_of_measure_qty"]
            uom = str(last.get("unit_of_measure", "") or "")
            scale = QUOTE_SCALE.get(uom)
            if pd.isna(qty):
                problem(sym, "UNVERIFIED", "multiplier", spec.multiplier, None,
                        "unit_of_measure_qty missing")
            elif scale is None:
                problem(sym, "UNVERIFIED", "multiplier", spec.multiplier, None,
                        f"unit_of_measure {uom!r} has no QUOTE_SCALE entry - "
                        f"add one rather than assuming 1.0")
            else:
                expected = float(qty) * scale
                if abs(expected - float(spec.multiplier)) > 1e-9:
                    problem(sym, "MISMATCH", "multiplier", spec.multiplier,
                            expected,
                            f"unit_of_measure_qty={qty:g} {uom} x {scale:g}")

    return pd.DataFrame(
        rows,
        columns=["symbol", "status", "field", "ours", "databento", "note"],
    )


if __name__ == "__main__":
    report = verify_specs()
    if report.empty:
        print("All contract specs reconcile against the definition files.")
    else:
        print(report.to_string(index=False))
