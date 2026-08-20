import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os

from backtest.pipeline import pipeline_dir

# The four quadrants, in the order every table prints them. A module constant
# rather than a list built inside `generate_profile`, because Stage 1 screens
# on these names and derives each survivor's kill-switch set as "the other
# three". Two spellings of the same quadrant - one here and one in the caller -
# would make a kill switch name a regime that never appears in a breakdown, and
# nothing would raise.
REGIMES = (
    "High Volatility / Trending",
    "High Volatility / Ranging",
    "Low Volatility / Trending",
    "Low Volatility / Ranging",
)


def _resolve_out_dir(out_dir, strat_name: str) -> str:
    """
    Where this profile is written: the caller's directory, or the strategy's
    own pipeline directory under `$BT_ARTIFACTS`.

    Resolved through `pipeline.pipeline_dir` rather than off the environment
    here, because `$BT_ARTIFACTS` is the artifacts ROOT and the handoffs live
    one level in, at `<root>/pipeline/<strategy>/`. Reading the variable
    directly would default to the pipeline directory but return the root the
    moment it is set, scattering regime profiles one level above every other
    artifact of the same run.

    Read at CALL time, never at import - `pipeline.artifacts_root` documents
    why, and a module-level default would bind the variable before a test that
    sets it has run.

    An explicit `out_dir` is the FINAL directory and is used verbatim. Stage 4
    passes its own resolved directory so `--out-dir` reaches the profile, and
    that path already ends in the strategy name.
    """
    return str(out_dir) if out_dir else str(pipeline_dir(strat_name))


def _entry_timestamps(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    The bar timestamps, normalised the way `backtest/engine.py` normalises
    them (`pd.to_datetime(bars["ts"], utc=True)`).

    The regime of a trade is looked up by its entry timestamp, so both sides
    of that lookup have to be built the same way. A naive index on one side
    and a UTC-aware one on the other matches nothing, and an unmatched trade
    is dropped from every regime bucket without raising - the profile comes
    back empty and reads as a strategy that never traded.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "ts" in df.columns:
        idx = pd.DatetimeIndex(df["ts"])
    else:
        raise ValueError("the bar frame carries neither a DatetimeIndex nor a "
                         "'ts' column, so trades cannot be placed on it")
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True))


def _closed_trades(portfolio) -> pd.DataFrame:
    """
    Closed trades as `entry_ts` (UTC) and `pnl`, whatever produced them.

    **There is no `vbt.Portfolio` object to hand this class in this repo.**
    `engine._simulate` builds one per CHUNK, extracts the closed-trade records
    and deletes it before building the next, so a run produces several and
    keeps none. What survives is `BacktestResult.trades`, and that is what the
    pipeline passes here. A real portfolio is still accepted - a caller
    holding one is not wrong - so both shapes resolve to the same two columns.

    An unrecognised shape RAISES. Returning an empty frame instead would print
    a regime profile of nothing and write a kill-switch artifact naming all
    four regimes, which is a live-trading instruction derived from no trades.
    """
    if isinstance(portfolio, pd.DataFrame):
        trades = portfolio
    elif hasattr(portfolio, "trades") and isinstance(
            getattr(portfolio, "trades"), pd.DataFrame):
        trades = portfolio.trades                    # BacktestResult
    elif hasattr(portfolio, "trades") and hasattr(portfolio.trades,
                                                  "records_readable"):
        trades = portfolio.trades.records_readable   # vbt.Portfolio
    else:
        raise TypeError(f"cannot read a trade list off a "
                        f"{type(portfolio).__name__}")

    trades = trades.copy()
    for ts_col, pnl_col in (("entry_time", "pnl"),            # engine
                            ("Entry Timestamp", "PnL")):      # vectorbt
        if ts_col in trades.columns and pnl_col in trades.columns:
            return pd.DataFrame({
                "entry_ts": pd.to_datetime(trades[ts_col], utc=True),
                "pnl": trades[pnl_col].astype(float),
            })
    raise KeyError(f"no entry-timestamp / P&L column pair in "
                   f"{list(trades.columns)}")


class RegimeProfiler:
    def __init__(self, df, portfolio, strat_name, symbol, tf, out_dir=None,
                 version: str = "", quiet: bool = False):
        """
        `version` suffixes the artifact filename ("a" ->
        `regime_profile_NQ_15m_version_a.json`) and is empty by default, so a
        caller profiling one result per configuration keeps the original path.
        Stage 1 profiles BOTH versions of the same configuration, and without
        the suffix Version B's profile would overwrite Version A's at a path
        named only for the contract - a file whose name says NQ 15m holding the
        other version's regime breakdown, with nothing raising.

        `quiet` suppresses the console breakdown and nothing else - the same
        dict is returned and the same artifact written. Stage 1 screens 27
        contracts x 4 timeframes x 2 versions, and 216 ten-line tables on a
        console whose stated job is one progress line per configuration is how
        the last one goes unread.
        """
        self.df = df.copy()
        self.df.index = _entry_timestamps(self.df)
        self.portfolio = portfolio
        self.strat_name = strat_name
        self.symbol = symbol
        self.tf = tf
        self.version = str(version or "")
        self.quiet = bool(quiet)
        self.out_dir = _resolve_out_dir(out_dir, strat_name)
        os.makedirs(self.out_dir, exist_ok=True)

    def _say(self, *a, **kw):
        if not self.quiet:
            print(*a, **kw)

    @property
    def artifact_path(self) -> str:
        """Where `generate_profile` writes, version suffix included."""
        suffix = f"_version_{self.version.lower()}" if self.version else ""
        return (f"{self.out_dir}/regime_profile_"
                f"{self.symbol}_{self.tf}{suffix}.json")

    def generate_profile(self):
        """
        The four-quadrant breakdown, RETURNED as well as printed and written.

        The return value is the same dict the artifact holds, so a caller that
        screens on the profile reads the numbers it wrote rather than parsing
        the JSON back off an NFS mount. A run with NO trades still returns a
        dict - `optimal_regime` "None", an empty breakdown, an EMPTY kill
        switch - and writes NO artifact. An empty kill switch rather than all
        four regimes is the point: "trade nowhere" is a live-trading
        instruction, and deriving one from a strategy that never traded is the
        failure `_closed_trades` refuses to make quietly.
        """
        self._say(f"\n[PROFILER] Analyzing {self.symbol} on {self.tf} for {self.strat_name}...")
        
        # 1. Calculate ADX (Trend) and ATR (Volatility)
        self.df.ta.adx(length=14, append=True)
        self.df.ta.atr(length=14, append=True)
        
        # Find exact column names generated by pandas-ta
        adx_col = [c for c in self.df.columns if c.startswith('ADX')][0]
        atr_col = [c for c in self.df.columns if c.startswith('ATRe') or c.startswith('ATR')][0]
        
        # 2. Define the Volatility Median for this specific asset
        atr_median = self.df[atr_col].median()
        
        # 3. Classify Every Bar into a Quadrant
        conditions = [
            (self.df[atr_col] > atr_median) & (self.df[adx_col] > 25),  # High Vol, Trending
            (self.df[atr_col] > atr_median) & (self.df[adx_col] <= 25), # High Vol, Ranging
            (self.df[atr_col] <= atr_median) & (self.df[adx_col] > 25), # Low Vol, Trending
            (self.df[atr_col] <= atr_median) & (self.df[adx_col] <= 25) # Low Vol, Ranging
        ]
        choices = list(REGIMES)
        self.df['Regime'] = np.select(conditions, choices, default="UNKNOWN")
        
        # 4. Extract the closed trades and tag them
        trades = _closed_trades(self.portfolio)
        if len(trades) == 0:
            self._say("No trades found to profile.")
            return {
                "strategy": self.strat_name,
                "symbol": self.symbol,
                "timeframe": self.tf,
                "version": self.version,
                "optimal_regime": "None",
                "optimal_profit_factor": None,
                "optimal_trade_count": 0,
                "kill_switch_conditions": [],
                "regime_breakdown": {},
                "trades_profiled": 0,
                "trades_unplaced": 0,
                "artifact": None,
            }
            
        # Map the entry timestamp to the DataFrame index to get the regime at entry
        regime_map = self.df['Regime'].to_dict()
        trades['Entry_Regime'] = trades['entry_ts'].map(regime_map)
        
        # A trade in no bucket is a trade the four rows below quietly leave
        # out, and they are read as the whole run. Two ways it happens: an
        # entry timestamp that is not a bar in this frame (NaN), and an entry
        # inside the ADX/ATR warm-up, where the indicators are undefined and
        # np.select falls through to "UNKNOWN". Counted together and named,
        # because a breakdown that sums to less than the trade list has to say
        # so on the same screen.
        unplaced = int((~trades['Entry_Regime'].isin(choices)).sum())
        if unplaced:
            self._say(f"[PROFILER] {unplaced} of {len(trades)} trades are in no "
                  f"regime (entry outside this frame, or inside the 14-bar "
                  f"indicator warm-up) and are excluded from every row below.")
        
        # 5. Calculate Metrics per Regime
        profile = {}
        best_regime = "None"
        best_pf = 0
        
        self._say("\n" + "="*80)
        self._say(f" REGIME PROFILE: {self.symbol} {self.tf} | {self.strat_name}")
        self._say("="*80)
        self._say(f" {'REGIME':<30} | {'TRADES':<8} | {'WIN %':<8} | {'PROFIT FACTOR':<14} | {'NET PNL'}")
        self._say("-" * 80)
        
        for regime in choices:
            regime_trades = trades[trades['Entry_Regime'] == regime]
            count = len(regime_trades)
            if count == 0: continue
                
            wins = regime_trades[regime_trades['pnl'] > 0]
            losses = regime_trades[regime_trades['pnl'] <= 0]
            
            gross_prof = wins['pnl'].sum()
            gross_loss = abs(losses['pnl'].sum())
            
            pf = gross_prof / gross_loss if gross_loss > 0 else (999 if gross_prof > 0 else 0)
            win_rate = (len(wins) / count) * 100
            net_pnl = gross_prof - gross_loss
            
            profile[regime] = {
                "trade_count": count,
                "profit_factor": round(pf, 2),
                "win_rate": round(win_rate, 2),
                "net_pnl": round(net_pnl, 2)
            }
            
            self._say(f" {regime:<30} | {count:<8} | {win_rate:>5.1f}%  | {pf:>13.2f} | ${net_pnl:,.2f}")
            
            # Select Optimal Regime (Must have > 30 trades and highest PF)
            if pf > best_pf and count >= 30:
                best_pf = pf
                best_regime = regime
                
        self._say("="*80)
        self._say(f"✅ OPTIMAL ENVIRONMENT: {best_regime} (PF: {best_pf:.2f})\n")

        # 6. Save Artifact for the Live Supervisor
        file_path = self.artifact_path
        out_data = {
            "strategy": self.strat_name,
            "symbol": self.symbol,
            "timeframe": self.tf,
            "version": self.version,
            "optimal_regime": best_regime,
            # The optimal quadrant's own numbers, beside its name. Stage 1
            # screens on the PAIR (profit factor at a trade count), and a name
            # with no numbers under it forces every reader to re-derive them
            # from the breakdown - where a reader is free to apply a different
            # trade floor than the one that chose the name.
            "optimal_profit_factor": (profile.get(best_regime) or {}
                                      ).get("profit_factor"),
            "optimal_trade_count": (profile.get(best_regime) or {}
                                    ).get("trade_count", 0),
            "kill_switch_conditions": ([r for r in choices if r != best_regime]
                                       if best_regime in profile else []),
            "regime_breakdown": profile,
            "trades_profiled": int(len(trades) - unplaced),
            "trades_unplaced": int(unplaced),
            "artifact": file_path,
        }

        with open(file_path, "w") as f:
            json.dump(out_data, f, indent=4)
        return out_data
