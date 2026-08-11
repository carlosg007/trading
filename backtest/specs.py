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

The `definition` schema being downloaded to
/mnt/backtest/reference/futures/definitions/ carries the authoritative values.
Run `verify_specs()` once that download completes and reconcile any
differences before running a real backtest.

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

import pandas as pd

DEFINITIONS = Path("/mnt/backtest/reference/futures/definitions")


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


# Commission estimate: NT rate per side plus ~$1.00 exchange/NFA/clearing.
_NON_MICRO = 1.29 + 1.00
_MICRO = 0.35 + 0.35

SPECS: dict[str, ContractSpec] = {
    # Equity index
    "ES":  ContractSpec("ES",  "E-mini S&P 500",        "CME",   50,     0.25,      _NON_MICRO),
    "NQ":  ContractSpec("NQ",  "E-mini Nasdaq-100",     "CME",   20,     0.25,      _NON_MICRO),
    "RTY": ContractSpec("RTY", "E-mini Russell 2000",   "CME",   50,     0.10,      _NON_MICRO),
    "YM":  ContractSpec("YM",  "E-mini Dow",            "CBOT",   5,     1.00,      _NON_MICRO),

    # Rates
    "ZT":  ContractSpec("ZT",  "2-Year T-Note",         "CBOT", 2000,    1/128,     _NON_MICRO),
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
    "6A":  ContractSpec("6A",  "Australian Dollar",     "CME",   100000, 0.0001,    _NON_MICRO),
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

    Returns a frame of differences. An empty frame means everything matches.
    Run this once the definition download finishes - a wrong multiplier is
    invisible in results but scales every P&L figure for that symbol.
    """
    if not definitions_dir.exists():
        raise FileNotFoundError(
            f"{definitions_dir} not found. The definition schema download may "
            f"still be running."
        )

    rows = []
    for sym, spec in SPECS.items():
        sym_dir = definitions_dir / f"symbol={sym}"
        if not sym_dir.exists():
            continue
        files = sorted(sym_dir.glob("*.parquet"))
        if not files:
            continue
        try:
            df = pd.read_parquet(files[-1])
        except Exception as e:
            rows.append({"symbol": sym, "field": "read", "ours": None,
                         "databento": f"unreadable: {e}"})
            continue
        if df.empty:
            continue

        last = df.iloc[-1]
        for field, ours in (("min_price_increment", spec.tick_size),
                            ("unit_of_measure_qty", spec.multiplier)):
            if field not in df.columns:
                continue
            theirs = last[field]
            # Databento encodes prices as fixed-point integers, 1e-9 scale
            if field == "min_price_increment" and theirs and abs(theirs) > 1e3:
                theirs = theirs / 1e9
            try:
                if abs(float(theirs) - float(ours)) > 1e-9:
                    rows.append({"symbol": sym, "field": field,
                                 "ours": ours, "databento": theirs})
            except (TypeError, ValueError):
                pass

    return pd.DataFrame(rows)
