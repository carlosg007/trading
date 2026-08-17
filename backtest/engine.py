"""
backtest.engine - turns signals into a tested result.

Location:  ~/src/trading/backtest/engine.py

A strategy says when to be long, short, or flat. This says what that would
have cost and whether the account would have survived.

    from backtest.engine import BacktestConfig, run_backtest

    def my_strategy(bars):          # one symbol's bars, positionally indexed
        fast = bars["close"].rolling(20).mean()
        slow = bars["close"].rolling(50).mean()
        return fast > slow, fast < slow

    cfg = BacktestConfig(flat_by_close=True, trailing_drawdown_pct=5.0)
    res = run_backtest(["ES", "NQ", "GC"], "1h", my_strategy,
                       start="2013-01-01", cfg=cfg)
    res.save("/mnt/backtest/artifacts/strat1")

The engine reads the bars. You pass the strategy, not the signals - see
`run_backtest` for why that is enforced rather than merely encouraged.

Then:

    python backtest/report.py \
      --returns /mnt/backtest/artifacts/strat1_returns.parquet \
      --trades  /mnt/backtest/artifacts/strat1_trades.parquet \
      --name "Strat 1" --variants-tested 1 --costs-included yes

Why this is separate from the strategy
--------------------------------------
If each strategy carried its own cost handling you would end up with twenty
slightly different cost models and no way to know whether strategy A beat
strategy B or merely assumed cheaper fills. One wrapper means every idea is
judged on identical terms.

Structure
---------
The vectorbt call is isolated in `_simulate`, which drives
`vbt.Portfolio.from_signals`. Everything around it - signal preparation,
session handling, cost computation, result formatting, the drawdown check - is
plain pandas, so it can be tested without vectorbt and swapped if the engine
ever changes.

Memory
------
The full 1-minute lake is 110M rows across 27 symbols, and on a 26 GB box
nothing may hold it all at once. Two levels of streaming keep that true:

  - `run_backtest` reads ONE symbol at a time via `mdlib.lake.iter_bars` and
    frees it before the next, so peak tracks the largest single symbol (5.6M
    rows) rather than the lake. Building the whole frame first cost 20.54 GiB
    before any simulation started; the per-symbol path peaks at 2.90 GiB.
  - `_simulate` feeds vectorbt `cfg.chunk_size` bars at a time and drops each
    batch's intermediates before starting the next, so peak RAM tracks the
    chunk rather than the symbol.

Chunk boundaries are placed only where the strategy is flat, so batching
cannot drop a trade that spans a boundary - slicing by calendar year would,
and would quietly flatter the results. The trade list is identical at any
chunk size; tests/test_engine_batching.py asserts that against the unchunked
engine.

Costs are passed into the simulation as per-bar arrays (`slippage` as a
fraction of price, `fees` as a fraction of order value) rather than subtracted
afterwards, so they broadcast across the index inside the compiled simulation
with no Python-level iteration. The unit conversions are fiddly and are
documented in `_cost_arrays` - in particular slippage is built from tick SIZE,
not tick VALUE.

The loop this replaced is kept as `_simulate_legacy` and is the oracle in
tests/test_engine_vbt.py: the two must agree trade-for-trade.

Long and short
--------------
A strategy may return two masks or four:

    signal_fn(bars) -> (entries, exits)
    signal_fn(bars) -> (entries, exits, short_entries, short_exits)

`unpack_signals` accepts both and fills the short pair with False for the
two-mask form, so every strategy written before this existed runs unchanged
AND produces the same numbers as it did - the long-only vectorbt call is kept
distinct rather than routed through the four-mask one. Signals are resolved by
a single three-state machine (`_clean_signals_ls_loop`: flat, long, short), so
there is no second resolver that could disagree with the first about what an
unmatched exit means, and every closed trade is stamped `direction` from
vectorbt's own record rather than from the sign of its P&L.

What the engine still does NOT do is any of the risk machinery. There are no
stop or target ORDERS in it, in either direction: a stop is an exit signal,
detected on the bar that breaches it and filled at the NEXT bar's open. A
strategy that wants a stop, a target or a trailing level - long or short -
computes it in its own module (see `strategies/experimental/ema_trend_filter`)
and emits the resulting exit.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .event_calendar import apply_entry_filters
from .specs import get_spec
# The ratio definitions live in backtest.report and are imported rather than
# reimplemented here. A second local Sharpe would be free to disagree with the
# one the tear sheet prints for the same run, with nothing raising - and the
# daily-close standardization these depend on is exactly the kind of detail
# that drifts between two copies. report.py imports nothing from this package,
# so the dependency is one-way.
from .report import daily_metrics as report_daily_metrics
from .report import daily_returns as report_daily_returns

# Imported lazily-ish: everything in this module except _simulate is plain
# pandas and stays importable (and testable) without vectorbt installed.
try:
    import vectorbtpro as vbt
    _VBT_IMPORT_ERROR: Exception | None = None
except ImportError as e:      # pragma: no cover - depends on the environment
    vbt = None
    _VBT_IMPORT_ERROR = e

# numba is pinned in requirements.txt and is a transitive dependency of
# vectorbt anyway, but clean_signals stays usable without it - see below. The
# fallback is silent and only costs speed, so _NUMBA_IMPORT_ERROR is the thing
# to inspect if a run is unexpectedly slow.
try:
    from numba import njit
    _NUMBA_IMPORT_ERROR: Exception | None = None
except ImportError as e:      # pragma: no cover - depends on the environment
    njit = None
    _NUMBA_IMPORT_ERROR = e


# --------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """
    Everything that is not the strategy itself.

    The constraint set is a parameter, not a separate codebase: Portfolio A
    (intraday, prop accounts) sets flat_by_close=True and a trailing drawdown
    limit; Portfolio B (swing, own capital) leaves both off. Same engine,
    same strategy code.
    """

    initial_capital: float = 100_000.0
    contracts: int = 1

    # Costs. Defaults come from backtest/specs.py per symbol; these override.
    commission_per_side: float | None = None   # dollars per contract
    slippage_ticks: float = 1.0                # each way

    # Annual risk-free rate as a fraction (0.04 for 4%), subtracted from the
    # DAILY returns as rf / 252 before Sharpe and Sortino. Zero by default:
    # these are futures strategies scored on the edge, and a non-zero rate
    # would quietly rank a 2015 backtest against a different hurdle than a
    # 2023 one. Set it deliberately, and it is recorded in stats["basis"].
    risk_free_rate: float = 0.0

    # Portfolio A constraints
    flat_by_close: bool = False
    session_close_utc: str = "20:00"           # 16:00 ET during EDT
    trailing_drawdown_pct: float | None = None  # e.g. 5.0 for FundedNext
    daily_loss_limit: float | None = None       # dollars

    # Data hygiene
    exclude_degraded: bool = True
    exclude_rolls: bool = True

    # Entry filters (backtest/event_calendar.py). Both suppress ENTRIES on both
    # sides and never touch an exit - blocking an exit would hold a position
    # through the very event the filter exists to avoid. They live on the
    # config rather than in the strategies because neither reads a price:
    # applied here they work for every module in the tree without one of them
    # being edited, and a sweep over strategy parameters cannot sweep them.
    #
    # `news_filter` needs a macro calendar. Absent a published one it falls
    # back to APPROXIMATE rule-generated dates and says so, in stats and on
    # every report - see the calendar module's docstring before reading a
    # news-filtered result as one that dodged the actual releases.
    news_filter: bool = False
    news_window_minutes: float = 30.0
    news_kinds: tuple[str, ...] | None = None      # None = all four
    # Weekday integers, Monday=0 .. Sunday=6, on the CME SESSION date rather
    # than the UTC date. (0, 4) suppresses Monday and Friday entries.
    exclude_days: tuple[int, ...] | None = None

    # Execution. Bars are handed to vectorbt this many at a time so a
    # full-lake run does not have to hold the whole simulation in RAM. This is
    # a memory knob ONLY: boundaries are snapped into the gaps between trades,
    # so the trade list is identical whatever it is set to (0 disables
    # chunking). See _chunk_bounds. 1m bars are ~5.6M rows per symbol at the
    # moment, so the default runs the largest symbol in three passes.
    chunk_size: int = 2_000_000

    # Bookkeeping - carried into the result so a number is never read without
    # the context needed to judge it.
    variants_tested: int | None = None
    notes: str = ""


@dataclass
class BacktestResult:
    """
    Standard output. Every strategy produces this shape, so results can be
    compared and correlated against each other.
    """

    returns: pd.Series          # daily, decimal (0.01 = 1%)
    trades: pd.DataFrame        # entry_time, exit_time, symbol, pnl, direction
    equity: pd.Series
    config: BacktestConfig
    breach: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def save(self, prefix: str | Path) -> tuple[Path, Path]:
        """Write in the format backtest/report.py consumes."""
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        r = prefix.with_name(prefix.name + "_returns.parquet")
        t = prefix.with_name(prefix.name + "_trades.parquet")

        (self.returns.rename("returns").rename_axis("date")
             .reset_index().to_parquet(r, index=False))
        self.trades.to_parquet(t, index=False)
        return r, t

    def summary(self) -> str:
        lines = [
            f"  trades          {len(self.trades):>10,}",
            f"  total return    {self.stats.get('total_return_pct', float('nan')):>10.2f} %",
            f"  sharpe          {self.stats.get('sharpe', float('nan')):>10.2f}",
            f"  max drawdown    {self.stats.get('max_dd_pct', float('nan')):>10.2f} %",
            f"  total costs     {self.stats.get('total_costs', float('nan')):>10,.0f}",
        ]
        if self.breach.get("breached"):
            lines.append(f"  PROP BREACH     {self.breach['first_breach']}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Session handling
# --------------------------------------------------------------------------
def apply_flat_by_close(bars: pd.DataFrame,
                        entries: pd.Series,
                        exits: pd.Series,
                        session_close_utc: str = "20:00") -> tuple[pd.Series, pd.Series]:
    """
    Force an exit at the last bar of each session, and block entries on it.

    Portfolio A trades prop accounts, which do not permit overnight holds.
    This is not cosmetic - it truncates exactly the moves a trend or breakout
    strategy would otherwise capture, so results change materially. That is
    the point: the constraint has to be in the backtest, not discovered live.

    An entry on the final bar would be opened and closed in the same bar, so
    those are suppressed rather than left to produce a guaranteed cost.

    This works on ONE side's pair of masks. A bidirectional strategy calls it
    twice - once for `(entries, exits)` and once for `(short_entries,
    short_exits)` - because the rule is identical per side: block the side's
    entry on the last bar of the session and force its exit there. Taking a
    short pair through the long-named arguments is correct and not a misuse.
    """
    ts = pd.to_datetime(bars["ts"], utc=True)
    close_t = pd.Timestamp(f"2000-01-01 {session_close_utc}", tz="UTC").time()

    # Session date: bars at or after the close belong to the next session.
    session = ts.dt.normalize()
    session = session.where(ts.dt.time < close_t, session + pd.Timedelta(days=1))

    is_last = session != session.shift(-1)
    is_last.iloc[-1] = True

    exits = exits.copy()
    entries = entries.copy()
    exits[is_last.values] = True
    entries[is_last.values] = False
    return entries, exits


def _clean_signals_ls_loop(le: np.ndarray, lx: np.ndarray,
                           se: np.ndarray, sx: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
    """
    Walk the signals as a THREE-state machine: flat (0), long (1), short (-1).

    This is the production resolver for every backtest, long-only ones
    included - `clean_signals` calls it with all-False short masks. There is
    deliberately only one state machine: two of them (one for longs, one for
    shorts) would be two things that can drift apart, and a drift here silently
    adds or removes trades from every backtest rather than raising.

    The rules, and what each one is refusing to guess:

      * From FLAT, the first long entry opens a long and the first short entry
        opens a short.
      * From FLAT with BOTH firing on the same bar, NEITHER is taken. The bar
        is ambiguous - the strategy has asked to be long and short at once -
        and picking a side by argument order would bury a coin flip inside the
        engine. A well-formed strategy cannot produce this (a close cannot be
        both above and below the same anchor EMA), so the branch exists to make
        a malformed one visible as a missing trade rather than as a plausible
        one-sided equity curve.
      * From LONG, only a long exit closes; from SHORT, only a short exit. A
        short entry arriving while long is DROPPED, not treated as a reversal.
        Reversing would open the new position at the same bar's fill with no
        flat bar in between, which `_pair_trades` (and therefore chunking)
        assumes cannot happen. A strategy that wants to reverse must emit the
        closing exit itself.
      * Everything else is dropped: an entry while already in a position would
        double-count it, and an exit while flat would book a trade that was
        never opened.

    With `se` and `sx` all False this reduces EXACTLY to `_clean_signals_loop`
    below - the short branches are unreachable and the ambiguity branch cannot
    fire - which is what `tests/test_clean_signals.py` checks exhaustively for
    every input up to length 10, since it compares `clean_signals` against that
    interpreted two-state oracle.

    Note the state checks are `elif`: an exit on the same bar as the entry does
    not close it, which is what `_simulate_legacy` and `_pair_trades` also
    assume.
    """
    n = le.shape[0]
    kle = np.zeros(n, dtype=np.bool_)
    klx = np.zeros(n, dtype=np.bool_)
    kse = np.zeros(n, dtype=np.bool_)
    ksx = np.zeros(n, dtype=np.bool_)

    state = 0                       # 0 flat, 1 long, -1 short
    for i in range(n):
        if state == 0:
            if le[i] and se[i]:
                continue            # ambiguous bar: take neither side
            if le[i]:
                kle[i] = True
                state = 1
            elif se[i]:
                kse[i] = True
                state = -1
        elif state == 1:
            if lx[i]:
                klx[i] = True
                state = 0
        else:
            if sx[i]:
                ksx[i] = True
                state = 0

    return kle, klx, kse, ksx


def _clean_signals_loop(e: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk the signals as a two-state machine: flat, or long.

    From flat, the first entry opens a position. From long, the first exit
    closes it. Everything else is dropped - an entry while already long would
    double-count the position, and an exit while flat would book a trade that
    was never opened. Note the `elif`: an exit on the same bar as the entry
    does not close it, which is what `_simulate_legacy` and `_pair_trades`
    also assume.

    This stays a bar-by-bar loop because the decision at bar i depends on the
    state left by bar i-1, so there is nothing to vectorise - numpy has no
    primitive for a sequential two-state scan. The searchsorted alternative
    (jump entry-to-exit-to-entry) is only a win when trades are rare: measured
    on 5.6M rows it beat this loop 46x at 1-in-10,000 density, but at 50% it
    was 9x SLOWER than even this interpreted loop, because it degenerates into
    a Python loop per trade.

    So the loop is kept and compiled instead - see _clean_signals_ls_fast.

    THIS FUNCTION IS NO LONGER THE PRODUCTION PATH. `clean_signals` routes
    through the three-state `_clean_signals_ls_loop` above, which reduces to
    this one when there are no short signals. It is kept as the long-only
    ORACLE: tests/test_clean_signals.py checks the production resolver against
    it exhaustively at every length up to 10, so the two-state behaviour that
    every existing strategy depends on cannot change when the short branches
    are edited.
    """
    n = e.shape[0]
    ke = np.zeros(n, dtype=np.bool_)
    kx = np.zeros(n, dtype=np.bool_)

    in_pos = False
    for i in range(n):
        if not in_pos and e[i]:
            ke[i] = True
            in_pos = True
        elif in_pos and x[i]:
            kx[i] = True
            in_pos = False

    return ke, kx


# The compiled path is the SAME function object handed to numba, not a
# reimplementation of it. That is deliberate: two hand-written copies of a
# state machine are two things that can drift apart, and a drift here would
# silently add or remove trades from every backtest rather than raise. This
# way the pure-Python oracle in tests/test_clean_signals.py is checking the
# exact source that runs in production.
#
# Falls back to the interpreted function if numba is missing, so this module
# stays importable and correct - just slower - in an environment without it.
_clean_signals_ls_fast = (njit(cache=True, nogil=True)(_clean_signals_ls_loop)
                          if njit is not None else _clean_signals_ls_loop)


def _bool_array(s) -> np.ndarray:
    """A signal series as a contiguous bool array. NaN counts as no signal."""
    return np.ascontiguousarray(pd.Series(s).fillna(False).astype(bool).to_numpy())


def clean_signals_ls(entries: pd.Series,
                     exits: pd.Series,
                     short_entries: pd.Series,
                     short_exits: pd.Series
                     ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Resolve four masks into executable long AND short signals.

    The bidirectional form of `clean_signals`. See `_clean_signals_ls_loop`
    for the state machine and for what it refuses to guess - in particular
    that a short entry arriving while long is dropped rather than treated as a
    reversal.

    NaN counts as no signal.
    """
    kle, klx, kse, ksx = _clean_signals_ls_fast(
        _bool_array(entries), _bool_array(exits),
        _bool_array(short_entries), _bool_array(short_exits))

    return (pd.Series(kle, index=entries.index),
            pd.Series(klx, index=exits.index),
            pd.Series(kse, index=short_entries.index),
            pd.Series(ksx, index=short_exits.index))


def clean_signals(entries: pd.Series, exits: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Remove signals that cannot execute: an entry with no prior exit, and an
    exit with no open position. Prevents double-counting.

    The long-only form, and the one every pre-existing strategy goes through.
    It delegates to `clean_signals_ls` with empty short masks rather than
    carrying a second copy of the state machine, so the two cannot disagree
    about what an unmatched exit means.

    Every trade in every backtest passes through here, and a full-lake run
    calls it on 109.8M bars, so the scan is compiled - 12-44x faster than the
    interpreted loop depending on signal density, and unlike a searchsorted
    rewrite it does not degrade as signals get denser.
    tests/test_clean_signals.py proves the equivalence exhaustively for every
    input up to length 10, and at full scale beyond it - which now also pins
    the three-state resolver's long-only reduction.

    NaN counts as no signal.
    """
    empty = pd.Series(np.zeros(len(entries), dtype=bool), index=entries.index)
    ke, kx, _se, _sx = clean_signals_ls(entries, exits, empty, empty)
    return ke, kx


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------
def round_turn_cost(symbol: str, cfg: BacktestConfig) -> float:
    """
    Total cost of one round trip, one contract: commission both sides plus
    slippage both sides.

    Slippage is charged in ticks each way on the assumption you pay the
    spread. Optimistic fill assumptions are the most common reason a backtest
    overstates performance, so the default is deliberately conservative.
    """
    spec = get_spec(symbol)
    commission = (cfg.commission_per_side
                  if cfg.commission_per_side is not None else spec.commission)
    slip = spec.slippage_dollars(cfg.slippage_ticks)
    return (commission + slip) * 2


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
TRADE_COLUMNS = ["entry_time", "exit_time", "symbol", "direction",
                 "entry_price", "exit_price", "gross_pnl", "costs", "pnl"]


def _cost_arrays(bars: pd.DataFrame,
                 entries: np.ndarray,
                 exits: np.ndarray,
                 symbol: str,
                 cfg: BacktestConfig,
                 short_entries: np.ndarray | None = None,
                 short_exits: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Per-bar slippage and fee arrays for vectorbt, plus the position size.

    Both are expressed the way from_signals wants them, which is not the way
    they are quoted:

    slippage - a fraction of PRICE. vectorbt fills at price*(1+s) to buy and
        price*(1-s) to sell, so s must move the price by `ticks` ticks:

            s = ticks * tick_size / price

        tick_size, not tick_value. tick_value is dollars (multiplier *
        tick_size); dividing it by price would move the fill by `multiplier`
        ticks - 50x on ES, 1000x on ZN. The multiplier enters through `size`
        below and cancels, so it must not appear here as well.

    fees - a fraction of order value, and order value is size*fill_price. Our
        commission is a flat dollar amount per contract per side, so:

            f = commission * contracts / (size * fill_price)

        Dividing by the FILL price rather than the raw price is what makes the
        charge come out at exactly `commission` per side; the fill is a tick
        away from the raw price and vectorbt bills against the fill.

    tick_size is looked up per bar, so a contract whose tick changed mid-
    history (ZT, 2019-01-13) is charged the tick that was actually in force.

    Direction and which way the fill is adjusted
    -------------------------------------------
    The fee denominator is the FILL price, and the fill is a tick on the side
    the order actually crossed. That side is set by whether the order BUYS or
    SELLS, not by whether it opens or closes a position:

        buys   long entries  and short exits    ->  px * (1 + slippage)
        sells  long exits    and short entries  ->  px * (1 - slippage)

    Passing a short entry through the long-entry branch would put the fill a
    tick on the wrong side. The error is a fraction of a tick per trade -
    far too small to notice in a total, wrong in every one of them - which is
    why the short masks are arguments here rather than assumed away.

    `short_entries` and `short_exits` default to None, which means a long-only
    run; the arithmetic then reduces exactly to the pre-bidirectional version.
    """
    spec = get_spec(symbol)
    px = bars["open"].to_numpy(dtype=float)
    ts = pd.to_datetime(bars["ts"], utc=True)

    size = float(spec.multiplier) * cfg.contracts

    tick_size = spec.tick_size_array(ts)
    slippage = cfg.slippage_ticks * tick_size / px

    commission = (cfg.commission_per_side
                  if cfg.commission_per_side is not None else spec.commission)

    zeros = np.zeros(len(px), dtype=bool)
    se = zeros if short_entries is None else np.asarray(short_entries, dtype=bool)
    sx = zeros if short_exits is None else np.asarray(short_exits, dtype=bool)

    buys = np.asarray(entries, dtype=bool) | sx
    sells = np.asarray(exits, dtype=bool) | se

    fill = np.where(buys, px * (1 + slippage),
                    np.where(sells, px * (1 - slippage), px))
    fees = (commission * cfg.contracts) / (size * fill)

    return slippage, fees, size


def _shift_to_fill(signals) -> np.ndarray:
    """
    A signal on bar i executes on bar i+1.

    Acting on the bar that produced the signal is lookahead bias, and it is the
    single most common way a backtest lies. `from_signals` fills on the bar it
    sees a signal, so every mask is shifted forward one bar before it is handed
    over and `price` is that bar's open.

    np.roll wraps the last element to the front, where it is cleared - which is
    exactly the "no next bar to fill on" case: a signal on the final bar cannot
    be executed and is dropped.
    """
    out = np.roll(np.asarray(signals, dtype=bool), 1)
    out[0] = False
    return out


def _pair_trades(ent: np.ndarray, exi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve fill-bar signals into (entry_bar, exit_bar) pairs.

    Same rule the simulation uses: from the first entry, take the first exit
    STRICTLY after it, then look for the next entry after that. An exit on the
    entry bar itself does not close the position, matching `_simulate_legacy`.

    This is not used to compute P&L - vectorbt does that. It is used only to
    find bars where the strategy is flat, which is where a chunk may be cut.
    The loop runs once per trade, not once per bar, so it stays cheap on a
    multi-million-bar symbol.

    A final entry with no matching exit is left out: that position never
    closes, so it never realises a P&L and no chunk boundary needs to protect
    it.
    """
    e_pos = np.flatnonzero(ent)
    x_pos = np.flatnonzero(exi)
    if e_pos.size == 0 or x_pos.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    e_out: list[int] = []
    x_out: list[int] = []
    cursor = 0
    while True:
        i = np.searchsorted(e_pos, cursor, side="left")
        if i >= e_pos.size:
            break
        e = int(e_pos[i])
        j = np.searchsorted(x_pos, e, side="right")   # first exit after entry
        if j >= x_pos.size:
            break                                     # never closes
        x = int(x_pos[j])
        e_out.append(e)
        x_out.append(x)
        cursor = x + 1

    return (np.asarray(e_out, dtype=np.int64),
            np.asarray(x_out, dtype=np.int64))


def _chunk_bounds(n: int,
                  chunk_size: int,
                  e_idx: np.ndarray,
                  x_idx: np.ndarray) -> list[tuple[int, int]]:
    """
    Split [0, n) into slices that never cut through a trade.

    Chunking exists so a 110-million-bar run does not have to hold the whole
    simulation in RAM. The naive way to do that - slice by calendar year - is
    silently wrong: a position open on 31 December is never closed, so the
    trade is dropped from the results entirely. Dropped trades are not
    neutral. They remove losers as readily as winners and the equity curve
    still looks plausible, which is precisely the kind of quietly flattering
    backtest this engine is supposed to make impossible.

    So a boundary is only ever placed where the strategy is flat. A target
    boundary that falls inside a trade is pushed forward to the bar after that
    trade's exit. Trades are non-overlapping and sorted, so the only trade
    that can straddle `target` is the first one whose exit is at or after it.

    The result is that chunking is a pure memory optimisation: the trade list
    is bit-identical to the unchunked run at any chunk size. That equivalence
    is what tests/test_engine_batching.py asserts.

    Chunks come out at roughly `chunk_size` bars, longer only where a single
    trade is longer than that. A very small chunk_size is correct but slow -
    each chunk is a separate compiled vectorbt call with fixed overhead.
    """
    if chunk_size <= 0 or n <= chunk_size:
        return [(0, n)]

    bounds: list[tuple[int, int]] = []
    lo = 0
    while lo < n:
        target = lo + chunk_size
        if target >= n:
            bounds.append((lo, n))
            break

        j = np.searchsorted(x_idx, target, side="left")
        hi = target
        if j < x_idx.size and e_idx[j] < target:
            hi = int(x_idx[j]) + 1        # carry the straddling trade whole

        bounds.append((lo, hi))
        lo = hi

    return bounds


def _simulate(bars: pd.DataFrame,
              entries: pd.Series,
              exits: pd.Series,
              symbol: str,
              cfg: BacktestConfig,
              short_entries: pd.Series | None = None,
              short_exits: pd.Series | None = None) -> pd.DataFrame:
    """
    Produce a trade list with vectorbt Pro, one batch of bars at a time.

    Entries and exits fill at the NEXT bar's open, never the signal bar's
    close. Acting on the bar that produced the signal is lookahead bias, and
    it is the single most common way a backtest lies. from_signals fills on
    the bar it sees a signal, so the signals are shifted forward one bar and
    `price` is that bar's open. A signal on the final bar has no next bar and
    is dropped, as it cannot be executed.

    Costs are handed to vectorbt as per-bar arrays rather than applied
    afterwards, so they broadcast across the index inside the compiled
    simulation. See _cost_arrays for the unit conversions.

    Direction
    ---------
    `short_entries` / `short_exits` are optional. Omitted (or all False), the
    call is the long-only one it has always been - `direction="longonly"`,
    every trade stamped `"long"`. Supplied with any signal in them, vectorbt is
    driven in its LONG/SHORT SIGNAL MODE: all four masks are passed and
    `direction` is NOT passed at all, because vectorbtpro refuses the two
    together ("Direction and short signal arrays cannot be used together"). The
    four-mask form IS the both-directions form; there is no
    `direction="both"` to combine with it.

    The two paths are kept distinct rather than always running the four-mask
    call, so that adding shorts to the engine cannot change a single number in
    an existing long-only backtest. `tests/test_risk_params.py` pins the
    equivalence in the other direction: an all-False short pair through the
    four-mask path produces the same trades as the long-only path.

    Each closed trade is stamped `direction` from vectorbt's own trade record
    (0 = Long, 1 = Short), never inferred from the sign of the P&L - a losing
    long and a winning short are indistinguishable that way. `gross_pnl` is
    signed with it: a short's gross is `entry_price - exit_price` per contract.

    Batching
    --------
    vectorbt allocates several full-length float64 arrays per call (cash,
    position, value, the order and trade records), so peak RAM scales with the
    number of bars in one call. Bars are therefore fed in `cfg.chunk_size`
    slices, and each slice's intermediates are dropped before the next one
    starts. Boundaries are placed only where the strategy is flat, so the
    output does not depend on the chunk size - see _chunk_bounds.

    Returns: entry_time, exit_time, symbol, direction, entry_price,
             exit_price, gross_pnl, costs, pnl
    """
    if vbt is None:
        raise ImportError(
            "vectorbtpro is required by backtest.engine._simulate. "
            f"Import failed with: {_VBT_IMPORT_ERROR}"
        ) from _VBT_IMPORT_ERROR

    spec = get_spec(symbol)
    px = bars["open"].to_numpy(dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))

    # Signal on bar i executes on bar i+1; see _shift_to_fill.
    ent = _shift_to_fill(entries.to_numpy(dtype=bool))
    exi = _shift_to_fill(exits.to_numpy(dtype=bool))
    zeros = np.zeros(len(px), dtype=bool)
    s_ent = zeros if short_entries is None else _shift_to_fill(
        short_entries.to_numpy(dtype=bool))
    s_exi = zeros if short_exits is None else _shift_to_fill(
        short_exits.to_numpy(dtype=bool))

    bidirectional = bool(s_ent.any())

    if not ent.any() and not bidirectional:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    # Chunk boundaries are placed where the strategy is FLAT, which after
    # `clean_signals_ls` means flat on both sides: entries and exits strictly
    # alternate across the union of the two directions, so the union is what
    # `_pair_trades` has to see. Pairing the long masks alone would treat a bar
    # inside an open short as flat and cut a chunk through it, dropping the
    # trade - the exact failure chunking is built to avoid.
    e_idx, x_idx = _pair_trades(ent | s_ent, exi | s_exi)
    bounds = _chunk_bounds(len(px), cfg.chunk_size, e_idx, x_idx)
    del e_idx, x_idx

    frames: list[pd.DataFrame] = []

    for lo, hi in bounds:
        ent_c = ent[lo:hi]
        s_ent_c = s_ent[lo:hi]
        if not ent_c.any() and not s_ent_c.any():   # no trade starts here
            continue
        exi_c = exi[lo:hi]
        s_exi_c = s_exi[lo:hi]
        px_c = px[lo:hi]
        index_c = index[lo:hi]

        slippage, fees, size = _cost_arrays(bars.iloc[lo:hi], ent_c, exi_c,
                                            symbol, cfg, s_ent_c, s_exi_c)
        price = pd.Series(px_c, index=index_c)

        common = dict(
            close=price,
            entries=pd.Series(ent_c, index=index_c),
            exits=pd.Series(exi_c, index=index_c),
            price=price,
            size=size,
            size_type="amount",
            fees=pd.Series(fees, index=index_c),
            slippage=pd.Series(slippage, index=index_c),
            # Futures are margined, not paid for in full. Cash is not the
            # binding constraint here and the account curve is built from
            # realised P&L in _daily_returns, so an unbounded balance keeps
            # vectorbt from rejecting an order whose notional exceeds the
            # account. It also makes chunks independent: no cash balance
            # carries across a boundary to change later position sizing.
            init_cash=np.inf,
            accumulate=False,
        )
        if bidirectional:
            # No `direction=` here: vectorbtpro raises "Direction and short
            # signal arrays cannot be used together". The four masks ARE the
            # long/short mode.
            pf = vbt.Portfolio.from_signals(
                **common,
                short_entries=pd.Series(s_ent_c, index=index_c),
                short_exits=pd.Series(s_exi_c, index=index_c),
            )
        else:
            pf = vbt.Portfolio.from_signals(**common, direction="longonly")

        rec = pf.trades.records
        rec = rec[rec["status"] == 1]    # closed only; an open position at the
        if not rec.empty:                # end of the data never realised a P&L
            entry_i = rec["entry_idx"].to_numpy()
            exit_i = rec["exit_idx"].to_numpy()

            # Report the raw open prices and carry the cost separately, so the
            # trade list stays comparable across cost assumptions. vectorbt's
            # own entry/exit prices are slippage-adjusted; the difference IS
            # the cost.
            entry_px = px_c[entry_i]
            exit_px = px_c[exit_i]

            # vectorbt's TradeDirection: 0 = Long, 1 = Short. Read from the
            # record rather than inferred - the sign of the P&L cannot tell a
            # losing long from a winning short.
            is_short = rec["direction"].to_numpy() == 1
            per_contract = np.where(is_short, entry_px - exit_px,
                                    exit_px - entry_px)
            gross = per_contract * spec.multiplier * cfg.contracts
            pnl = rec["pnl"].to_numpy()

            frames.append(pd.DataFrame({
                "entry_time": index_c[entry_i],
                "exit_time": index_c[exit_i],
                "symbol": symbol,
                "direction": np.where(is_short, "short", "long"),
                "entry_price": entry_px,
                "exit_price": exit_px,
                "gross_pnl": gross,
                "costs": gross - pnl,
                "pnl": pnl,
            }))

        # The portfolio holds the whole simulation. Drop it before building
        # the next one rather than letting two coexist at the peak.
        del pf, rec, common, price, slippage, fees
        del ent_c, exi_c, s_ent_c, s_exi_c, px_c, index_c
        if len(bounds) > 1:
            gc.collect()

    if not frames:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def _simulate_legacy(bars: pd.DataFrame,
                     entries: pd.Series,
                     exits: pd.Series,
                     symbol: str,
                     cfg: BacktestConfig,
                     short_entries: pd.Series | None = None,
                     short_exits: pd.Series | None = None) -> pd.DataFrame:
    """
    The pre-vectorbt loop, kept as the reference implementation.

    Not used in a run. It is the oracle the vectorbt path is tested against -
    slow but obviously correct, which is what makes it worth keeping. See
    tests/test_engine_vbt.py.

    It walks the same three states `_clean_signals_ls_loop` does, so it is an
    oracle for short trades as well as long ones. A short's gross P&L is
    `entry_price - exit_price` per contract: written out explicitly here rather
    than folded into a sign flip, because this loop exists to be read and
    checked by hand.

    Note it charges a single scalar tick value for every bar, so it cannot
    reproduce the vectorbt path on a contract whose tick changed mid-history.
    """
    spec = get_spec(symbol)
    cost = round_turn_cost(symbol, cfg) * cfg.contracts

    ts = pd.to_datetime(bars["ts"], utc=True).to_numpy()
    op = bars["open"].to_numpy(dtype=float)
    e = entries.to_numpy(dtype=bool)
    x = exits.to_numpy(dtype=bool)
    n = len(ts)
    zeros = np.zeros(n, dtype=bool)
    se = zeros if short_entries is None else short_entries.to_numpy(dtype=bool)
    sx = zeros if short_exits is None else short_exits.to_numpy(dtype=bool)

    trades = []
    state = 0                        # 0 flat, 1 long, -1 short
    entry_i = -1

    def _close(exit_i: int) -> None:
        long_side = state == 1
        per_contract = (op[exit_i] - op[entry_i] if long_side
                        else op[entry_i] - op[exit_i])
        gross = per_contract * spec.multiplier * cfg.contracts
        trades.append({
            "entry_time": ts[entry_i],
            "exit_time": ts[exit_i],
            "symbol": symbol,
            "direction": "long" if long_side else "short",
            "entry_price": op[entry_i],
            "exit_price": op[exit_i],
            "gross_pnl": gross,
            "costs": cost,
            "pnl": gross - cost,
        })

    for i in range(n - 1):
        if state == 0:
            if e[i] and se[i]:
                continue             # ambiguous bar, same rule as the resolver
            if e[i]:
                entry_i = i + 1      # fill next bar open
                state = 1
            elif se[i]:
                entry_i = i + 1
                state = -1
        elif state == 1:
            if x[i]:
                _close(i + 1)
                state = 0
        else:
            if sx[i]:
                _close(i + 1)
                state = 0

    return pd.DataFrame(trades, columns=TRADE_COLUMNS if not trades else None)


def _daily_returns(trades: pd.DataFrame,
                   days: pd.DatetimeIndex,
                   initial_capital: float) -> tuple[pd.Series, pd.Series]:
    """
    Convert a trade list into a DAILY equity curve and a daily return series.

    This is where the engine's metric frequency is fixed, and it is fixed at
    one point per session regardless of the timeframe the bars were read at. A
    15m backtest and a 1d backtest of the same strategy therefore produce
    return series that are directly comparable and that sqrt(252) correctly
    annualizes - see `backtest.report.daily_returns` for what goes wrong when
    a ratio is annualized at the wrong root.

    P&L is attributed to the exit date, which is when it is realised. Every
    session in the backtest window appears, including flat days - a strategy
    that trades rarely should show that in its return series rather than
    compressing time. Sessions the market did not trade never appear at all.

    `days` is the sorted set of session dates covered by the bars. It is
    accumulated per symbol in run_backtest rather than derived here, so
    normalising 110 million timestamps at once never happens.
    """
    idx = pd.DatetimeIndex(days)

    daily_pnl = pd.Series(0.0, index=idx)
    if not trades.empty:
        by_day = (trades.assign(d=pd.to_datetime(trades["exit_time"], utc=True)
                                  .dt.normalize())
                        .groupby("d")["pnl"].sum())
        daily_pnl = daily_pnl.add(by_day.reindex(idx).fillna(0.0), fill_value=0.0)

    equity = initial_capital + daily_pnl.cumsum()
    # Seeded on the starting capital so the first session is scored rather than
    # consumed as the base of the series. The returned series keeps one entry
    # per session, exactly as before.
    returns = report_daily_returns(equity, initial_capital)
    return returns, equity


def check_trailing_drawdown(equity: pd.Series, limit_pct: float) -> dict:
    """
    Prop firm trailing drawdown check.

    The limit trails the high water mark, so the PATH of returns matters more
    than the total. A strategy with a strong Sharpe and a deep drawdown fails
    the account regardless of eventual profitability - which is why this is a
    first-class output rather than an afterthought.
    """
    peak = equity.cummax()
    floor = peak * (1 - limit_pct / 100)
    breached = equity < floor
    return {
        "breached": bool(breached.any()),
        "first_breach": str(breached.idxmax().date()) if breached.any() else None,
        "worst_margin_pct": float(((equity - floor) / peak).min() * 100),
        "limit_pct": limit_pct,
    }


def _merge_filter_info(parts: list[dict]) -> dict:
    """
    Per-symbol entry-filter reports -> one run-level report.

    Counts add up across symbols; the settings do not, so they are taken from
    the first symbol and the provenance is collapsed to a single token only
    when every symbol agrees. A run where one symbol's span had published dates
    and another's fell back to rule-generated ones reports MIXED rather than
    whichever came last.
    """
    if not parts:
        return {}
    out = dict(parts[0])
    counters = [k for k in out
                if k.endswith(("_blocked", "_before", "_suppressed", "_in_span"))
                or k == "bars"]
    for k in counters:
        out[k] = int(sum(int(p.get(k, 0) or 0) for p in parts))
    provs = {p.get("news_provenance") for p in parts if p.get("news_provenance")}
    if provs:
        out["news_provenance"] = provs.pop() if len(provs) == 1 else "MIXED"
    out["symbols"] = len(parts)
    return out


def _assemble_result(all_trades: list[pd.DataFrame],
                     days: pd.DatetimeIndex,
                     cfg: BacktestConfig,
                     filter_info: dict | None = None) -> BacktestResult:
    """Pool per-symbol trade lists into the standard result."""
    trades = (pd.concat(all_trades, ignore_index=True)
              if all_trades else
              pd.DataFrame(columns=TRADE_COLUMNS))
    if not trades.empty:
        # Sorted on more than exit_time, and stably, so the row order does not
        # depend on the order the symbols were requested in. Many trades share
        # an exit timestamp once several symbols are in play, and a single-key
        # unstable sort would leave their relative order down to the caller's
        # symbol list.
        trades = (trades.sort_values(["exit_time", "symbol", "entry_time"],
                                     kind="stable")
                        .reset_index(drop=True))

        # Every closed trade carries its side. The stamp itself is made in
        # `_simulate`, from vectorbt's own trade record, because that is the
        # only place the record exists - by the time the frames arrive here the
        # information would have to be re-derived, and the only thing left to
        # re-derive it from is the sign of the P&L, which cannot tell a losing
        # long from a winning short. This is the check that the stamp survived:
        # an unlabelled or mislabelled row would flow into the tear sheet, the
        # leaderboard and the dispatcher as a long.
        bad = set(pd.unique(trades["direction"].astype("object"))) - {"long",
                                                                      "short"}
        if bad:
            raise ValueError(
                f"trades carry unrecognised direction values {sorted(bad)}; "
                f"every closed trade must be stamped 'long' or 'short'.")

    returns, equity = _daily_returns(trades, days, cfg.initial_capital)

    breach = (check_trailing_drawdown(equity, cfg.trailing_drawdown_pct)
              if cfg.trailing_drawdown_pct else {})

    # Sharpe, Sortino and Calmar all come out of ONE call on ONE daily equity
    # series, so they cannot end up sampled at different frequencies. `basis`
    # records which frequency, annualization factor and risk-free rate
    # produced them and is carried into the scorecard JSON - a ratio whose
    # sampling is not stated is not reproducible.
    m = report_daily_metrics(equity, cfg.initial_capital, cfg.risk_free_rate)
    stats = {
        "n_trades": len(trades),
        # The long/short split. A bidirectional strategy whose edge is entirely
        # on one side is a one-sided strategy paying for a second set of
        # signals, and the pooled Sharpe cannot show that. Zero on a long-only
        # run, which is the truth about it rather than a missing field.
        "n_long": int((trades["direction"] == "long").sum()) if not trades.empty else 0,
        "n_short": int((trades["direction"] == "short").sum()) if not trades.empty else 0,
        "total_return_pct": float((equity.iloc[-1] / cfg.initial_capital - 1) * 100),
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "calmar": m["calmar"],
        "annualized_return_pct": m["annualized_return_pct"],
        "max_dd_pct": m["max_dd_pct"],
        "basis": m["basis"],
        "total_costs": float(trades["costs"].sum()) if not trades.empty else 0.0,
        "gross_pnl": float(trades["gross_pnl"].sum()) if not trades.empty else 0.0,
        "net_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
    }
    # What the entry filters removed before the simulation saw the signals.
    # Recorded whenever either filter was configured - including when it
    # removed nothing, because "the news filter ran and cut 0 entries" and "no
    # news filter ran" are different results and the equity curve is identical.
    if filter_info:
        stats["entry_filters"] = filter_info

    return BacktestResult(returns=returns, trades=trades, equity=equity,
                          config=cfg, breach=breach, stats=stats)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def unpack_signals(out, n: int, index=None
                   ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Accept either strategy contract and return all four masks.

        (entries, exits)                                   long-only
        (entries, exits, short_entries, short_exits)       bidirectional

    A two-tuple gets all-False short masks, so every strategy written before
    the engine could go short keeps running unchanged and unchanged in its
    numbers. That backward compatibility is load-bearing rather than polite:
    `strategies/approved_incubator/` holds promoted modules that are recorded
    byte for byte against the metrics they were backtested on, and a contract
    change that forced an edit to those files would break the one property
    that makes a promotion evidence.

    A tuple of any other length raises. Silently taking the first two elements
    of a three-tuple is how a strategy's short side disappears into a plausible
    long-only equity curve.

    Every mask is validated to the bar count here rather than deeper in, so a
    strategy that returns a short side of the wrong length is caught before its
    signals are mixed into a simulation.
    """
    if not isinstance(out, tuple) or len(out) not in (2, 4):
        got = (f"a {len(out)}-tuple" if isinstance(out, tuple)
               else type(out).__name__)
        raise ValueError(
            f"signal_fn must return (entries, exits) or (entries, exits, "
            f"short_entries, short_exits); got {got}.")

    if len(out) == 2:
        e, x = out
        se = sx = np.zeros(n, dtype=bool)
    else:
        e, x, se, sx = out

    names = ("entries", "exits", "short_entries", "short_exits")
    masks = []
    for name, raw in zip(names, (e, x, se, sx)):
        if len(raw) != n:
            raise ValueError(
                f"signal_fn returned {len(raw)} {name} for {n} bars.")
        s = pd.Series(raw).reset_index(drop=True).fillna(False).astype(bool)
        masks.append(s if index is None else s.set_axis(index))
    return tuple(masks)


def run_backtest(symbols: str | list[str],
                 tf: str,
                 signal_fn,
                 start: str | None = None,
                 end: str | None = None,
                 cfg: BacktestConfig | None = None,
                 **lake_kwargs) -> BacktestResult:
    """
    Run one strategy over one or more symbols.

    The engine reads the bars itself, one symbol at a time, and calls the
    strategy on each. It is deliberately not possible to hand it a
    multi-symbol frame with signals already computed - see below.

    Parameters
    ----------
    symbols
        One symbol or a list. A list is the normal case: a daily strategy on
        ES alone over 16 years is ~100-200 trades, too thin to separate skill
        from luck.
    tf
        Any timeframe `mdlib.lake` serves - "1m", "1d" natively, the rest
        derived.
    signal_fn
        The `strategies/` contract, in either of its two forms:

            signal_fn(bars) -> (entries, exits)
            signal_fn(bars) -> (entries, exits, short_entries, short_exits)

        It receives ONE symbol's bars, positionally indexed from 0, and returns
        boolean Series aligned to them. The two-mask form is long-only and is
        what every strategy written before the engine could go short returns;
        it is not deprecated. See `unpack_signals`.
    **lake_kwargs
        Passed to `iter_bars`: session_merge, exclude_degraded,
        exclude_rolls, respect_coverage.

    Why the strategy is a callable and not precomputed signals
    ----------------------------------------------------------
    This engine used to take `(bars, entries, exits)`, where `bars` was a long
    frame from `get_bars`. That frame is sorted by (ts, symbol), so it
    INTERLEAVES symbols: row i is ES, row i+1 is NQ, row i+2 is GC. A strategy
    computing `close.rolling(200).mean()` over it was therefore averaging
    across 27 different instruments, and the resulting signals were noise. The
    engine could not detect this - the signals were the right length and the
    right dtype, and the backtest returned a plausible-looking equity curve.
    On this lake it produced 608,079 trades where the correct per-symbol
    signals give 86,035.

    A signal_fn cannot express that mistake. It is handed one symbol at a
    time, so a rolling window has nothing to bleed into. The old signature was
    removed rather than deprecated because a trap that only misleads - never
    raises - is not one to leave lying around for the next person or agent to
    find.

    Memory
    ------
    Reading one symbol at a time also means the whole lake is never resident.
    Measured on all 27 symbols of 1-minute data (110M rows), against the old
    build-the-frame-then-split approach:

        whole-frame   peak 15.82 GiB   150s
        per-symbol    peak  2.90 GiB   110s

    and the frame the old path built cost 20.54 GiB to assemble before any
    simulation started. Peak now tracks the largest single symbol (5.6M rows),
    not the number of symbols requested.
    """
    # Imported here rather than at module scope: everything else in this
    # module works on bars it is handed, and only this function needs the
    # reader. Keeps `import backtest.engine` free of an NFS-backed dependency.
    from mdlib.lake import iter_bars

    cfg = cfg or BacktestConfig()

    all_trades: list[pd.DataFrame] = []
    day_parts: list[np.ndarray] = []
    filter_parts: list[dict] = []
    n_symbols = 0

    for sym, g in iter_bars(symbols, tf, start, end, **lake_kwargs):
        n_symbols += 1
        try:
            e, x, se, sx = unpack_signals(signal_fn(g), len(g))
        except ValueError as err:
            raise ValueError(f"{sym}: {err}") from err

        # Entry filters, applied to the strategy's own masks before anything
        # else touches them. Entries only, both sides, never the exits: a
        # blocked exit would hold a position through the release rather than
        # keeping one out of it. Neither filter reads a price, so neither can
        # add trades or create an edge - see backtest/event_calendar.py.
        if cfg.news_filter or cfg.exclude_days:
            e, se, finfo = apply_entry_filters(
                g["ts"], e, se,
                news_filter=cfg.news_filter,
                news_window_minutes=cfg.news_window_minutes,
                news_kinds=cfg.news_kinds,
                exclude_days=cfg.exclude_days)
            finfo["symbol"] = sym
            filter_parts.append(finfo)

        day_parts.append(
            np.unique(pd.DatetimeIndex(g["ts"]).values.astype("datetime64[D]")))

        if cfg.flat_by_close:
            # Once per side - the rule is the same for both, and a short left
            # open through the bell is the same breach as a long.
            e, x = apply_flat_by_close(g, e, x, cfg.session_close_utc)
            se, sx = apply_flat_by_close(g, se, sx, cfg.session_close_utc)

        e, x, se, sx = clean_signals_ls(e, x, se, sx)
        t = _simulate(g, e, x, sym, cfg, se, sx)
        if not t.empty:
            all_trades.append(t)

        del g, e, x, se, sx, t
        gc.collect()

    if n_symbols == 0:
        raise ValueError("No bars supplied.")

    days = pd.DatetimeIndex(np.unique(np.concatenate(day_parts))).tz_localize("UTC")
    del day_parts
    gc.collect()

    return _assemble_result(all_trades, days, cfg,
                            filter_info=_merge_filter_info(filter_parts))
