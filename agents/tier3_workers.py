"""
agents.tier3_workers - the tier that actually runs things.

Location:  ~/src/trading/agents/tier3_workers.py

The execution and quantitative-testing tools are implemented. The agent
orchestration entry points (`run_variant`, `run_dual_version`,
`generate_ml_filter`, `main`) are still scaffold and raise NotImplementedError.

What this tier is for
---------------------
Tier 3 does the work Tier 1 asked for and Tier 2 will judge: running
backtests, generating ML-filtered strategy variants, and writing artifacts.
It makes no decisions about whether a result is good. It runs the thing and
reports what happened, including when what happened was a failure.

Calling the engine
------------------
`backtest.engine.run_backtest` takes SYMBOLS AND A STRATEGY, not bars and
precomputed signals:

    run_backtest(symbols, tf, signal_fn, start, end, cfg) -> BacktestResult

The engine reads the bars itself, one symbol at a time, and calls `signal_fn`
with that symbol's bars alone.

This is not a stylistic preference and it should not be worked around. The
engine used to accept `(bars, entries, exits)`, where `bars` came from
`get_bars` - a frame sorted by (ts, symbol), which INTERLEAVES instruments.
A strategy computing `close.rolling(200).mean()` over that frame was averaging
across 27 different contracts. Nothing raised: the signals were the right
length and dtype and the equity curve looked plausible. On the full lake it
produced 608,079 trades where the correct per-symbol signals give 86,035.
The signature was deleted so the mistake cannot be made again.

So a worker must never assemble a multi-symbol frame, compute signals over it,
and feed those in. If a future agent finds itself reaching for `get_bars` to
build inputs for a backtest, that is the bug, not the missing convenience.

Costs and constraints are not the worker's to choose. They come from
`BacktestConfig` and `backtest/specs.py`; a worker that runs without costs to
"see if the strategy works" has produced nothing worth reading.

The Dual-Version Mandate
------------------------
Every strategy is expected to produce two variants, run identically:

    Version A   pure rule-based baseline (e.g. an SMA crossover)
    Version B   the same, plus an ML filter on the signals

B is only adopted if it beats A out of sample without breaching the risk
limits. Running only B, or comparing B against a differently-configured A, is
how an ML result gets adopted on no evidence.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np
import pandas as pd

# Importable both as `agents.tier3_workers` and as a script; the latter puts
# agents/ on sys.path rather than the repo root, so backtest/ would not resolve.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backtest.engine import (BacktestConfig, BacktestResult,  # noqa: E402
                             _pair_trades, round_turn_cost, run_backtest)
from backtest.report import sortino as report_sortino  # noqa: E402
from backtest.report import (annualized_return_pct  # noqa: E402
                             as report_annualized_return_pct)
from backtest.report import calmar as report_calmar  # noqa: E402
from backtest.report import day_of_week_breakdown  # noqa: E402
from backtest.report import to_daily_equity as report_to_daily_equity  # noqa: E402
from backtest.specs import get_spec  # noqa: E402

ARTIFACTS = Path("/mnt/backtest/artifacts")
EXPERIMENTAL = _REPO / "strategies" / "experimental"

TRADING_DAYS = 252

# Default when neither the caller nor the strategy module names one. Daily is
# the safe default: an intraday timeframe silently applied to a strategy
# written for daily bars produces a plausible, wrong result.
DEFAULT_TIMEFRAME = "1d"


class StrategyLoadError(Exception):
    """Raised when a strategy module cannot be loaded or does not conform."""


class SignalFn(Protocol):
    """
    The strategies/ contract.

    Receives ONE symbol's bars, positionally indexed from 0, and returns two
    boolean Series aligned to them.
    """

    def __call__(self, bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]: ...


@dataclass
class WorkOrder:
    """One unit of work handed down from Tier 1."""

    name: str
    symbols: list[str]
    timeframe: str
    start: str | None = None
    end: str | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    variants_tested: int = 1


@dataclass
class WorkReport:
    """
    What happened. Reports failure as readily as success.

    A worker that swallows an exception and returns an empty result is
    indistinguishable, downstream, from a strategy that simply never traded.
    """

    order: WorkOrder
    ok: bool
    stats: dict[str, Any] = field(default_factory=dict)
    breach: dict[str, Any] = field(default_factory=dict)
    artifact_prefix: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------
# Strategy loading
# --------------------------------------------------------------------------
def load_strategy(strategy_path: str | Path,
                  params: dict[str, Any] | None = None) -> tuple[SignalFn, dict]:
    """
    Import a strategy module from a file path and bind its parameters.

    A module may expose either:

        make_signal_fn(**params) -> signal_fn      preferred when parameterised
        signal_fn(bars)                            when it takes none

    Returns `(bound_signal_fn, module_info)`. `module_info` carries the
    module's declared TIMEFRAME, SYMBOLS and PARAM_GRID if it sets them.

    Two optional declarations are picked up for the tear sheet, and only for
    the tear sheet - neither can change a signal:

        LOGIC = {"concept": ..., "entry": ..., "exit": ...}
            Plain-English sentences with `{param}` slots, filled here with the
            parameters that were actually bound. `module_info["logic"]`.
        indicators(bars, **params) -> {name: series}
            The strategy's own calculated series, bound the same way as the
            signal function and returned as `module_info["indicator_fn"]`. The
            report draws these over the trade inspector's candles, so they have
            to come from the module rather than be recomputed downstream where
            they could drift out of step with the signals.

    Every failure here raises. A worker that returns a null strategy on a bad
    import produces a backtest with no trades, which downstream is
    indistinguishable from a strategy that simply never triggered.
    """
    params = dict(params or {})
    path = Path(strategy_path).expanduser().resolve()
    if not path.exists():
        raise StrategyLoadError(f"strategy file not found: {path}")
    if path.suffix != ".py":
        raise StrategyLoadError(f"not a Python module: {path}")

    spec = importlib.util.spec_from_file_location(f"_strategy_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise StrategyLoadError(f"{path.name} raised on import: "
                                f"{type(e).__name__}: {e}") from e

    info = {
        "path": str(path),
        "module": path.stem,
        "timeframe": getattr(module, "TIMEFRAME", None),
        "symbols": getattr(module, "SYMBOLS", None),
        "default_params": dict(getattr(module, "DEFAULT_PARAMS", {}) or {}),
        # The search space `backtest.scan` sweeps under --scan. Read here for
        # the same reason as SYMBOLS and TIMEFRAME - the module is the one
        # place that knows what its own parameters mean, and a grid written
        # anywhere else would drift from the signature it has to bind against.
        "param_grid": dict(getattr(module, "PARAM_GRID", {}) or {}),
    }

    factory = getattr(module, "make_signal_fn", None)
    raw = getattr(module, "signal_fn", None)

    if factory is not None:
        merged = {**info["default_params"], **params}
        _reject_unknown_params(factory, merged, path.name)
        try:
            fn = factory(**merged)
        except Exception as e:
            raise StrategyLoadError(f"{path.name}: make_signal_fn(**{merged}) "
                                    f"raised {type(e).__name__}: {e}") from e
        info["bound_params"] = merged
    elif raw is not None:
        if params:
            raise StrategyLoadError(
                f"{path.name} defines signal_fn but no make_signal_fn, so it "
                f"cannot accept params {sorted(params)}. Add a "
                f"make_signal_fn(**params) factory."
            )
        fn = raw
        info["bound_params"] = {}
    else:
        raise StrategyLoadError(
            f"{path.name} defines neither signal_fn nor make_signal_fn"
        )

    if not callable(fn):
        raise StrategyLoadError(f"{path.name}: resolved strategy is not callable")

    info["logic"] = _describe_strategy(module, info["bound_params"])
    info["indicator_fn"] = _bind_indicators(module, info["bound_params"])
    return fn, info


def _describe_strategy(module: Any, params: dict) -> dict[str, str]:
    """
    A module's LOGIC block with the run's parameters filled in.

    Presentation only, so a malformed template costs the sentence rather than
    the backtest: an unfilled slot is left verbatim instead of raising, and a
    module with no LOGIC returns {} - which the report renders as "not
    declared", never as an invented description.
    """
    logic = getattr(module, "LOGIC", None)
    if not isinstance(logic, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("concept", "entry", "exit"):
        text = logic.get(key)
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            out[key] = text.format(**params)
        except (KeyError, IndexError, ValueError):
            out[key] = text
    return out


def _bind_indicators(module: Any, params: dict) -> Callable | None:
    """
    Bind a module's `indicators(bars, **params)` hook, or None if it has none.

    Only the parameters the hook actually accepts are passed. A strategy whose
    indicators depend on a subset of its parameters is normal, and an unknown
    keyword here would break a report over a cosmetic function - the strict
    check belongs on the signal path, where a dropped parameter changes trades.
    """
    fn = getattr(module, "indicators", None)
    if not callable(fn):
        return None
    try:
        sig = inspect.signature(fn)
        takes_all = any(p.kind is inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values())
        accepted = (dict(params) if takes_all
                    else {k: v for k, v in params.items() if k in sig.parameters})
    except (TypeError, ValueError):
        accepted = {}

    def _bound(bars):
        return fn(bars, **accepted)

    return _bound


def _reject_unknown_params(factory: Callable, params: dict, name: str) -> None:
    """
    Fail on a parameter the factory does not accept.

    Silently ignoring an unknown key is how a sensitivity sweep ends up
    reporting that a parameter does not matter, when in truth it was never
    applied.
    """
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return
    unknown = set(params) - set(sig.parameters)
    if unknown:
        raise StrategyLoadError(
            f"{name}: make_signal_fn does not accept {sorted(unknown)} "
            f"(accepts {sorted(sig.parameters)})"
        )


def _resolve_timeframe(explicit: str | None, info: dict) -> str:
    return explicit or info.get("timeframe") or DEFAULT_TIMEFRAME


def _build_config(params: dict[str, Any] | None,
                  cfg: BacktestConfig | None) -> BacktestConfig:
    """
    Config overrides come from `params["config"]`, never from strategy params.

    Costs and prop-firm constraints are not the strategy's to choose - see the
    module docstring. Keeping them in a separate sub-dict means a parameter
    sweep over strategy inputs cannot accidentally sweep away the cost model.
    """
    if cfg is not None:
        return cfg
    overrides = dict((params or {}).get("config") or {})
    unknown = set(overrides) - set(BacktestConfig.__dataclass_fields__)
    if unknown:
        raise StrategyLoadError(
            f"unknown BacktestConfig field(s): {sorted(unknown)}"
        )
    return BacktestConfig(**overrides)


def _strategy_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Strategy params are everything except the reserved `config` key."""
    return {k: v for k, v in (params or {}).items() if k != "config"}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return float("nan")
    return float((trades["pnl"] > 0).sum() / len(trades))


def _profit_factor(trades: pd.DataFrame) -> float:
    """
    Gross profit / gross loss.

    Returns inf when there are winners and no losers - a real but useless
    number, and better than a silent 0 or a crash. NaN when there are no
    trades at all, which is a different thing and should not be confused
    with a bad profit factor.
    """
    if trades.empty:
        return float("nan")
    pnl = trades["pnl"]
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return gross_profit / gross_loss


def _annualized_return_pct(equity: pd.Series, initial_capital: float) -> float:
    """
    CAGR from the equity curve, in percent. Delegated to
    `backtest.report.annualized_return_pct`.

    Delegated for the same reason `_sortino` is: the equity curve is first
    collapsed onto daily closes, and a second local copy of that step would be
    free to drift from the one the engine and the tear sheet use.
    """
    if equity is None or len(equity) < 2:
        return float("nan")
    return report_annualized_return_pct(report_to_daily_equity(equity),
                                        initial_capital)


def _sortino(returns: pd.Series) -> float:
    """
    Sortino ratio, delegated to `backtest.report.sortino`.

    Delegated rather than reimplemented because the denominator convention is a
    real choice - shortfalls averaged over all periods, versus over losing
    periods only - and the two differ by roughly sqrt(n_all / n_down), which on
    a sparse trader is large enough to invert the ranking. A second local
    implementation would make the dashboard and the CLI report disagree about
    the same backtest.
    """
    if returns is None or len(returns) < 2:
        return float("nan")
    return report_sortino(returns)


def _calmar(annualized_return_pct: float, max_drawdown_pct: float) -> float:
    """
    CAGR over the absolute max drawdown, both already in percent. Delegated to
    `backtest.report.calmar`.
    """
    return report_calmar(annualized_return_pct, max_drawdown_pct)


def summarize_result(result: BacktestResult,
                     include_trades: bool = True,
                     include_trade_records: bool = False) -> dict[str, Any]:
    """
    Flatten a BacktestResult into the structured dict the agent tiers consume.

    `include_trades=False` drops the trade log. Walk-forward and sensitivity
    runs use that: holding every fold's trades at once is what turns a bounded
    sweep into an unbounded one.

    `include_trade_records` additionally emits the trades as a list of dicts.
    It defaults to False because it is a second, far more expensive copy of
    data already present in `trades`: on a 27-symbol 1-minute run producing
    535,563 trades, materialising the records list is most of a gigabyte on its
    own. Ask for it only when something genuinely needs JSON-shaped rows.
    """
    trades = result.trades
    stats = result.stats or {}
    final_equity = (float(result.equity.iloc[-1])
                    if result.equity is not None and len(result.equity) else float("nan"))
    max_dd_pct = float(stats.get("max_dd_pct", float("nan")))
    # The engine already computed these off its daily equity curve and recorded
    # the basis they were sampled on. Preferring its numbers - rather than
    # recomputing from `result.returns` - is what keeps the scorecard JSON
    # identical to the stats the run was judged on. The fallbacks are for a
    # BacktestResult assembled by hand, which is how several tests build one.
    annualized_pct = float(stats.get(
        "annualized_return_pct",
        _annualized_return_pct(result.equity, result.config.initial_capital)))
    sortino_val = float(stats.get("sortino", _sortino(result.returns)))
    calmar_val = float(stats.get("calmar", _calmar(annualized_pct, max_dd_pct)))
    out: dict[str, Any] = {
        "ok": True,
        # Equity at or below zero is a blown account, not a bad quarter. It is
        # surfaced explicitly because the derived figures go quiet at exactly
        # that point: CAGR of a non-positive final equity is undefined, so a
        # total wipeout otherwise shows up only as a NaN that reads like
        # missing data.
        "ruined": bool(final_equity <= 0) if not math.isnan(final_equity) else False,
        "final_equity": final_equity,
        "total_pnl": float(stats.get("net_pnl", float("nan"))),
        "gross_pnl": float(stats.get("gross_pnl", float("nan"))),
        "total_costs": float(stats.get("total_costs", float("nan"))),
        "total_return_pct": float(stats.get("total_return_pct", float("nan"))),
        "annualized_return_pct": annualized_pct,
        "sharpe": float(stats.get("sharpe", float("nan"))),
        "sortino": sortino_val,
        "calmar": calmar_val,
        # How the three ratios above were sampled. Travels into the scorecard
        # JSON so a Sharpe is never read without knowing the frequency and the
        # risk-free rate behind it.
        "metrics_basis": dict(stats.get("basis", {})),
        "max_drawdown_pct": max_dd_pct,
        "win_rate": _win_rate(trades),
        "profit_factor": _profit_factor(trades),
        "trade_count": int(stats.get("n_trades", len(trades))),
        # The long/short split, carried so a two-sided result can be read as
        # one. A strategy whose trades are 90% one side is a one-sided strategy
        # paying for a second set of signals, and no pooled ratio above can
        # show that. Both are 0 on a long-only run, which is the truth about it
        # rather than a missing field.
        "long_trades": int(stats.get("n_long", 0)),
        "short_trades": int(stats.get("n_short", 0)),
        "breach": dict(result.breach or {}),
        "n_days": int(len(result.equity)) if result.equity is not None else 0,
        # P&L, win rate and trade count by weekday, attributed by entry
        # session. Records rather than a DataFrame because this dict is
        # serialized into `dual_metrics.json`, and `_jsonable` DROPS pandas
        # objects - a breakdown stored as a frame would be present on screen
        # and silently absent from the snapshot a promotion cites. Seven rows,
        # so it costs nothing to carry.
        "dow_breakdown": day_of_week_breakdown(trades).to_dict("records"),
        # What the entry filters removed before the simulation ran, including
        # the macro calendar's provenance. Empty when neither filter was
        # configured; a filter that ran and cut nothing reports itself with
        # zero counts, which is a different statement.
        "entry_filters": dict(stats.get("entry_filters", {}) or {}),
    }
    if include_trades:
        out["trades"] = trades
        # The daily equity curve, so a compliance audit can measure drawdown on
        # the real path rather than reconstructing it from trade exits. Bounded
        # regardless of timeframe - one point per trading day, so ~4,000 floats
        # over the full 16-year lake.
        out["equity"] = result.equity
        if include_trade_records:
            out["trade_log"] = trades.to_dict("records")
    return out


def _empty_metrics(reason: str) -> dict[str, Any]:
    """A failed run, shaped like a successful one so callers can aggregate."""
    return {
        "ok": False, "error": reason,
        "total_pnl": float("nan"), "gross_pnl": float("nan"),
        "total_costs": float("nan"), "total_return_pct": float("nan"),
        "annualized_return_pct": float("nan"), "sharpe": float("nan"),
        "sortino": float("nan"), "calmar": float("nan"), "metrics_basis": {},
        "max_drawdown_pct": float("nan"), "win_rate": float("nan"),
        "profit_factor": float("nan"), "trade_count": 0,
        "long_trades": 0, "short_trades": 0,
        "breach": {}, "n_days": 0,
    }


# --------------------------------------------------------------------------
# 1. Backtest
# --------------------------------------------------------------------------
def run_strategy_backtest(strategy_path: str | Path,
                          symbols: str | list[str],
                          start_date: str | None = None,
                          end_date: str | None = None,
                          params: dict[str, Any] | None = None,
                          tf: str | None = None,
                          cfg: BacktestConfig | None = None,
                          include_trades: bool = True,
                          include_trade_records: bool = False,
                          **lake_kwargs) -> dict[str, Any]:
    """
    Load a strategy and run it through the streaming engine.

    Parameters
    ----------
    strategy_path
        Path to a module exposing `make_signal_fn(**params)` or `signal_fn`.
    symbols
        One symbol or a list. A list is the normal case - a daily strategy on
        ES alone over 16 years is ~100-200 trades, too thin to separate skill
        from luck.
    params
        Strategy parameters. The reserved key `params["config"]` holds
        BacktestConfig overrides; everything else is passed to the strategy.
    tf
        Timeframe. Falls back to the module's TIMEFRAME, then "1d". The engine
        requires one and there is no way to infer it from the strategy code.

    Returns a dict with total P&L, Sharpe, max drawdown, win rate, profit
    factor, trade count and the raw trade log.

    Load failures raise StrategyLoadError. A strategy that loads but produces
    no trades returns ok=True with trade_count=0 - that is a result, not an
    error, and the two must stay distinguishable.
    """
    fn, info = load_strategy(strategy_path, _strategy_params(params))
    config = _build_config(params, cfg)
    timeframe = _resolve_timeframe(tf, info)

    result = run_backtest(symbols, timeframe, fn,
                          start=start_date, end=end_date, cfg=config,
                          **lake_kwargs)

    out = summarize_result(result, include_trades=include_trades,
                           include_trade_records=include_trade_records)
    out["meta"] = {
        "strategy": info["module"],
        "strategy_path": info["path"],
        "params": info.get("bound_params", {}),
        "symbols": [symbols] if isinstance(symbols, str) else list(symbols),
        "timeframe": timeframe,
        "start": start_date,
        "end": end_date,
        "costs_included": True,
        "initial_capital": config.initial_capital,
    }
    return out


# --------------------------------------------------------------------------
# 2. Walk-forward
# --------------------------------------------------------------------------
def _fold_windows(start_year: int, end_year: int,
                  train_years: int, test_years: int) -> list[dict]:
    """Sequential, non-overlapping test windows rolling forward."""
    folds = []
    year = start_year
    while year + train_years + test_years - 1 <= end_year:
        folds.append({
            "train_start": f"{year}-01-01",
            "train_end": f"{year + train_years - 1}-12-31",
            "test_start": f"{year + train_years}-01-01",
            "test_end": f"{year + train_years + test_years - 1}-12-31",
        })
        year += test_years
    return folds


def run_walk_forward_analysis(strategy_path: str | Path,
                              symbols: str | list[str],
                              train_years: int = 2,
                              test_years: int = 1,
                              start_year: int = 2010,
                              end_year: int = 2023,
                              params: dict[str, Any] | None = None,
                              param_grid: list[dict] | None = None,
                              tf: str | None = None,
                              cfg: BacktestConfig | None = None,
                              **lake_kwargs) -> dict[str, Any]:
    """
    Roll train/test segments forward and report the WFO efficiency ratio.

    Efficiency = annualized OOS return / annualized IS return. Below ~0.5 is
    the usual warning that in-sample performance is not surviving the step
    forward.

    What the ratio does and does not measure
    ----------------------------------------
    **Without `param_grid` this is not an overfitting test.** With nothing
    being selected in-sample, the train window is just an earlier slice of
    history run with the same fixed parameters, and the ratio compares two time
    periods rather than fitted-versus-unseen. It is reported either way, with
    `optimized` recording which was done, because a ratio of 0.4 means
    something very different in the two cases.

    Pass `param_grid=[{...}, {...}]` to select the best in-sample combination
    per fold by Sharpe and carry it into the test window. That is the version
    that says something about overfitting.

    The efficiency ratio is undefined when the in-sample return is <= 0:
    dividing a negative OOS return by a negative IS return yields a healthy
    looking positive number for a strategy that lost money in both windows.
    Those folds report `efficiency=None` and are excluded from the aggregate.
    """
    folds_spec = _fold_windows(start_year, end_year, train_years, test_years)
    if not folds_spec:
        return {
            "ok": False,
            "error": (f"no folds fit in {start_year}-{end_year} with "
                      f"train={train_years}y test={test_years}y"),
            "folds": [], "efficiency_ratio": None,
        }

    base_params = _strategy_params(params)
    optimized = bool(param_grid)
    folds: list[dict] = []

    for spec in folds_spec:
        fold = dict(spec)
        try:
            chosen = dict(base_params)
            if param_grid:
                best, best_sharpe = None, -math.inf
                for combo in param_grid:
                    trial = {**base_params, **combo}
                    m = run_strategy_backtest(
                        strategy_path, symbols, spec["train_start"],
                        spec["train_end"], {**trial, "config": (params or {}).get("config", {})},
                        tf=tf, cfg=cfg, include_trades=False, **lake_kwargs)
                    s = m.get("sharpe", float("nan"))
                    if not math.isnan(s) and s > best_sharpe:
                        best, best_sharpe = trial, s
                if best is None:
                    fold.update(error="no parameter combination produced a "
                                      "finite in-sample Sharpe",
                                efficiency=None, is_metrics=None, oos_metrics=None)
                    folds.append(fold)
                    continue
                chosen = best
            fold["params"] = chosen

            run_params = {**chosen, "config": (params or {}).get("config", {})}
            is_m = run_strategy_backtest(
                strategy_path, symbols, spec["train_start"], spec["train_end"],
                run_params, tf=tf, cfg=cfg, include_trades=False, **lake_kwargs)
            oos_m = run_strategy_backtest(
                strategy_path, symbols, spec["test_start"], spec["test_end"],
                run_params, tf=tf, cfg=cfg, include_trades=False, **lake_kwargs)

            is_ann = is_m["annualized_return_pct"]
            oos_ann = oos_m["annualized_return_pct"]
            # Either side being NaN means an annualized return could not be
            # formed - normally because the account was ruined and CAGR is
            # undefined. A NaN ratio must not reach the aggregate, where it
            # would turn the whole mean into NaN.
            if (math.isnan(is_ann) or math.isnan(oos_ann) or is_ann <= 0):
                eff = None
            else:
                eff = float(oos_ann / is_ann)

            fold.update(is_metrics=is_m, oos_metrics=oos_m,
                        is_annualized_pct=is_ann, oos_annualized_pct=oos_ann,
                        efficiency=eff, error=None)
        except Exception as e:
            fold.update(error=f"{type(e).__name__}: {e}",
                        efficiency=None, is_metrics=None, oos_metrics=None)
        folds.append(fold)

    scored = [f["efficiency"] for f in folds
              if f.get("efficiency") is not None and not math.isnan(f["efficiency"])]
    failed = [f for f in folds if f.get("error")]
    undefined = [f for f in folds
                 if not f.get("error") and f.get("efficiency") is None]

    oos_returns = [f["oos_metrics"]["annualized_return_pct"] for f in folds
                   if f.get("oos_metrics")]
    positive_oos = sum(1 for r in oos_returns if not math.isnan(r) and r > 0)
    ruined = [f for f in folds
              if (f.get("oos_metrics") or {}).get("ruined")
              or (f.get("is_metrics") or {}).get("ruined")]

    return {
        "ok": bool(scored) or not failed,
        "optimized": optimized,
        "warning": None if optimized else (
            "No param_grid supplied: nothing was selected in-sample, so this "
            "ratio compares two time periods rather than fitted-versus-unseen "
            "performance. It is not evidence about overfitting."
        ),
        "n_folds": len(folds),
        "n_scored": len(scored),
        "n_failed": len(failed),
        "n_undefined": len(undefined),
        "undefined_reason": ("in-sample return <= 0, or an account was ruined "
                             "so no annualized return exists - the ratio would "
                             "be misleading" if undefined else None),
        "n_ruined_folds": len(ruined),
        "ruin_warning": (
            f"{len(ruined)} fold(s) ended with equity at or below zero. No "
            f"efficiency ratio is meaningful for a strategy that blows the "
            f"account." if ruined else None
        ),
        "efficiency_ratio": float(np.mean(scored)) if scored else None,
        "efficiency_median": float(np.median(scored)) if scored else None,
        "efficiency_min": float(np.min(scored)) if scored else None,
        "oos_positive_folds": positive_oos,
        "oos_mean_annualized_pct": (float(np.nanmean(oos_returns))
                                    if oos_returns else float("nan")),
        "folds": folds,
    }


# --------------------------------------------------------------------------
# 3. Parameter sensitivity
# --------------------------------------------------------------------------
def run_parameter_sensitivity(strategy_path: str | Path,
                              base_params: dict[str, Any],
                              perturbation_pct: float = 0.10,
                              symbols: str | list[str] | None = None,
                              start_date: str | None = None,
                              end_date: str | None = None,
                              metric: str = "sharpe",
                              tf: str | None = None,
                              cfg: BacktestConfig | None = None,
                              **lake_kwargs) -> dict[str, Any]:
    """
    Shift each numeric parameter by +/- `perturbation_pct` and measure the hit.

    A strategy whose edge evaporates when a lookback moves from 20 to 22 has
    found a feature of this particular history, not of the market. The point of
    this sweep is that a fragile optimum and a robust one look identical in a
    single backtest.

    Integer parameters are perturbed as integers and rounded away from the base
    value, so a 10% shift on a lookback of 20 gives 18 and 22 rather than
    collapsing back onto 20. Parameters that cannot move (a bool, a string, an
    int whose perturbation rounds to itself) are reported as skipped rather
    than silently dropped - "not tested" and "insensitive" are different
    findings.
    """
    if symbols is None:
        raise ValueError("run_parameter_sensitivity requires symbols")
    if not 0 < perturbation_pct < 1:
        raise ValueError(f"perturbation_pct must be in (0, 1), got {perturbation_pct}")

    strat_params = _strategy_params(base_params)
    config_block = (base_params or {}).get("config", {})

    base = run_strategy_backtest(strategy_path, symbols, start_date, end_date,
                                 base_params, tf=tf, cfg=cfg,
                                 include_trades=False, **lake_kwargs)
    base_metric = base.get(metric, float("nan"))

    variations: list[dict] = []
    skipped: list[dict] = []

    for name, value in strat_params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            skipped.append({"param": name, "value": value,
                            "reason": "not a numeric parameter"})
            continue

        for direction, sign in (("up", 1.0), ("down", -1.0)):
            shifted = value * (1.0 + sign * perturbation_pct)
            if isinstance(value, int):
                # Round to nearest, then nudge if the shift rounded back onto
                # the base. Nearest rather than ceil/floor because binary
                # floating point makes 50 * 1.1 == 55.00000000000001, and
                # ceil() would turn a 10% step into 56.
                shifted = int(round(shifted))
                if shifted == value:
                    shifted = value + (1 if sign > 0 else -1)
            if isinstance(value, int) and shifted <= 0:
                skipped.append({"param": name, "direction": direction,
                                "value": value,
                                "reason": "perturbation would be non-positive"})
                continue

            trial = {**strat_params, name: shifted, "config": config_block}
            entry = {"param": name, "direction": direction,
                     "base_value": value, "value": shifted}
            try:
                m = run_strategy_backtest(strategy_path, symbols, start_date,
                                          end_date, trial, tf=tf, cfg=cfg,
                                          include_trades=False, **lake_kwargs)
                entry["metrics"] = m
                entry["metric_value"] = m.get(metric, float("nan"))
                entry["degradation_pct"] = _degradation_pct(
                    base_metric, entry["metric_value"])
                entry["error"] = None
            except Exception as e:
                entry.update(metrics=None, metric_value=float("nan"),
                             degradation_pct=float("nan"),
                             error=f"{type(e).__name__}: {e}")
            variations.append(entry)

    degradations = [v["degradation_pct"] for v in variations
                    if not math.isnan(v.get("degradation_pct", float("nan")))]
    worst = max(degradations) if degradations else float("nan")
    mean_deg = float(np.mean(degradations)) if degradations else float("nan")

    # A sign flip is the loudest fragility signal there is: the edge did not
    # shrink under a small shift, it inverted.
    sign_flips = [v for v in variations
                  if not math.isnan(v.get("metric_value", float("nan")))
                  and not math.isnan(base_metric)
                  and np.sign(v["metric_value"]) != np.sign(base_metric)
                  and base_metric != 0]

    return {
        "ok": True,
        "metric": metric,
        "perturbation_pct": perturbation_pct,
        "base_metric": float(base_metric),
        "base_metrics": base,
        "n_variations": len(variations),
        "n_skipped": len(skipped),
        "mean_degradation_pct": mean_deg,
        "worst_degradation_pct": worst,
        "sign_flips": len(sign_flips),
        "fragile": bool(
            (not math.isnan(worst) and worst > 50.0) or sign_flips
        ),
        "fragility_note": (
            "A >50% drop or a sign flip from a 10% parameter move means the "
            "result depends on the exact parameter value, which is the "
            "signature of a fitted optimum rather than an edge."
        ),
        "variations": variations,
        "skipped": skipped,
    }


def _degradation_pct(base: float, trial: float) -> float:
    """
    Percentage fall from base to trial. Positive means worse.

    Normalised by |base| so a base Sharpe of -0.2 does not invert the sign of
    the degradation. Undefined when base is zero or either side is NaN.
    """
    if math.isnan(base) or math.isnan(trial) or base == 0:
        return float("nan")
    return float((base - trial) / abs(base) * 100.0)


# --------------------------------------------------------------------------
# 4. Monte Carlo
# --------------------------------------------------------------------------
def run_monte_carlo_simulation(trade_returns: Iterable[float] | pd.Series | np.ndarray,
                               n_iterations: int = 1000,
                               confidence_pct: float = 0.95,
                               max_loss_pct: float = 8.0,
                               initial_capital: float = 100_000.0,
                               returns_are_dollars: bool = False,
                               seed: int | None = 42,
                               chunk: int | None = None,
                               max_bytes: int = 256 * 1024 ** 2) -> dict[str, Any]:
    """
    Bootstrap the trade sequence to get a drawdown distribution.

    One backtest yields exactly one drawdown, produced by one ordering of the
    trades. Reshuffling with replacement asks how bad the drawdown could have
    been had the same edge arrived in a different order - which is the question
    a prop account actually poses, since the account dies on the path, not on
    the total.

    Parameters
    ----------
    trade_returns
        Per-trade returns as fractions of equity, or dollar P&L with
        `returns_are_dollars=True`.
    max_loss_pct
        Drawdown threshold counted as a breach, e.g. the 8% trailing limit in
        compliance_rules/fundednext_rapid.json.

    Assumption worth stating
    ------------------------
    Resampling with replacement assumes trades are independent and identically
    distributed. They are not: real trades cluster, and a losing regime tends
    to produce consecutive losers. Destroying that autocorrelation makes this
    an OPTIMISTIC estimate of drawdown. Treat the reported figure as a floor on
    the risk, not a ceiling.
    """
    arr = np.asarray(pd.Series(list(trade_returns)
                               if not isinstance(trade_returns, (pd.Series, np.ndarray))
                               else trade_returns).dropna(), dtype=float)
    n = arr.size
    if n == 0:
        return {"ok": False, "error": "no trades to resample", "n_trades": 0}
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")
    if not 0 < confidence_pct < 1:
        raise ValueError(f"confidence_pct must be in (0, 1), got {confidence_pct}")

    if returns_are_dollars:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive to convert dollars")
        arr = arr / initial_capital

    rng = np.random.default_rng(seed)
    max_dds = np.empty(n_iterations, dtype=float)
    finals = np.empty(n_iterations, dtype=float)

    # Chunk size is derived from a byte budget rather than fixed, because the
    # cost per iteration scales with the trade count. A fixed 200 was sized for
    # an ~86k-trade run; on a 27-symbol 1-minute run with 535,563 trades the
    # same setting asks for 200 x 535,563 x 8 bytes per array, and this holds
    # two of them - roughly 1.7 GiB, on top of whatever the caller is holding.
    if chunk is None:
        per_row = n * 8 * 2          # two live float64 arrays of (chunk, n)
        chunk = max(1, min(n_iterations, int(max_bytes // max(per_row, 1))))

    done = 0
    while done < n_iterations:
        size = min(chunk, n_iterations - done)
        # In-place throughout: the resample becomes the equity curve, which
        # then becomes the drawdown series, so only `path` and `peak` are ever
        # resident rather than four separate matrices.
        path = rng.choice(arr, size=(size, n), replace=True)
        np.add(path, 1.0, out=path)
        np.cumprod(path, axis=1, out=path)
        finals[done:done + size] = path[:, -1]
        peak = np.maximum.accumulate(path, axis=1)
        np.divide(path, peak, out=path)
        np.subtract(path, 1.0, out=path)
        max_dds[done:done + size] = path.min(axis=1)
        done += size
        del path, peak

    max_dds_pct = max_dds * 100.0
    # Drawdowns are negative; the "95th percentile worst" is the 5th percentile
    # of the signed series.
    tail = float(np.percentile(max_dds_pct, (1.0 - confidence_pct) * 100.0))
    breach_prob = float(np.mean(max_dds_pct <= -abs(max_loss_pct)))

    return {
        "ok": True,
        "n_trades": int(n),
        "n_iterations": int(n_iterations),
        "confidence_pct": confidence_pct,
        "max_loss_pct": max_loss_pct,
        "max_drawdown_pct_at_confidence": tail,
        "prob_max_loss_breach": breach_prob,
        "median_max_drawdown_pct": float(np.median(max_dds_pct)),
        "worst_max_drawdown_pct": float(np.min(max_dds_pct)),
        "median_final_return_pct": float((np.median(finals) - 1.0) * 100.0),
        "prob_profit": float(np.mean(finals > 1.0)),
        "assumption": (
            "i.i.d. bootstrap - trade autocorrelation is destroyed, so this "
            "understates clustered drawdowns. Treat as a floor on risk."
        ),
    }


def trade_returns_from_result(result: dict[str, Any] | BacktestResult,
                              initial_capital: float | None = None) -> np.ndarray:
    """Per-trade returns as fractions of starting equity, for the bootstrap."""
    if isinstance(result, BacktestResult):
        trades, capital = result.trades, result.config.initial_capital
    else:
        trades = result.get("trades")
        capital = initial_capital or result.get("meta", {}).get(
            "initial_capital", 100_000.0)
    if trades is None or len(trades) == 0:
        return np.array([], dtype=float)
    return (trades["pnl"].to_numpy(dtype=float) / float(capital))


# --------------------------------------------------------------------------
# 5. Causal ML signal filter (the Dual-Version Mandate's Version B)
# --------------------------------------------------------------------------
ML_FEATURES = ["atr_norm", "volume_z", "rsi_14", "hour", "minute",
               "ret_1", "ret_5"]

# Below this many completed trades the classifier has nothing to learn from, so
# the signal passes through unfiltered. Suppressing entries during the warm-up
# instead would silently truncate the start of the sample and flatter Version B
# by removing trades it never actually judged.
MIN_TRAIN_TRADES = 30


def _bar_timestamps(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """
    The bar timestamps, from the `ts` column or a DatetimeIndex.

    The engine hands strategies a long-format frame with `ts` as a COLUMN and a
    positional index, so `bars.index.hour` raises there. Accepting both shapes
    keeps this usable from the engine path and from a caller holding a
    time-indexed frame, without either one silently producing garbage.
    """
    if "ts" in bars.columns:
        return pd.DatetimeIndex(pd.to_datetime(bars["ts"], utc=True))
    if isinstance(bars.index, pd.DatetimeIndex):
        idx = bars.index
        return idx if idx.tz is not None else idx.tz_localize("UTC")
    raise ValueError(
        "bars needs a `ts` column or a DatetimeIndex; got an index of type "
        f"{type(bars.index).__name__} and columns {list(bars.columns)}"
    )


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI on rolling means of gains and losses.

    Simple rolling means rather than Wilder's smoothing: a rolling window has a
    hard cutoff, so the value at bar i provably depends on exactly the last
    `window` bars. An EWM tail is also causal but never fully forgets, which
    makes "does this feature see the future" harder to prove by truncation -
    and that proof is the point of this whole module.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def causal_features(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Feature matrix for the ML filter. Every column is strictly causal.

    A value at row i is a function of bars 0..i only. No `shift(-k)`, no
    centred window, no reversed slice, nothing computed off a full-sample
    statistic such as a global mean or a fitted scaler - a StandardScaler fit
    on the whole frame leaks the test period's distribution into the training
    rows, which is lookahead that no shift-based audit would catch.

    Using bar i's close to decide a signal on bar i is legitimate here: the
    engine fills at bar i+1's open, never on the signal bar. That one-bar gap
    is what makes these features tradeable rather than clairvoyant.

    Columns: atr_norm, volume_z, rsi_14, hour, minute, ret_1, ret_5.
    NaN warm-up rows are left as NaN - HistGradientBoostingClassifier consumes
    them natively, and filling them with a column mean would import a
    full-sample statistic into the early rows.
    """
    ts = _bar_timestamps(bars)
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    volume = bars["volume"].astype(float)

    prev_close = close.shift(1)
    true_range = pd.concat([high - low,
                            (high - prev_close).abs(),
                            (low - prev_close).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()

    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std()

    out = pd.DataFrame({
        "atr_norm": (atr / close.where(close != 0)).to_numpy(dtype=float),
        "volume_z": ((volume - vol_mean) / (vol_std + 1e-8)).to_numpy(dtype=float),
        "rsi_14": _rsi(close, 14).to_numpy(dtype=float),
        "hour": ts.hour.to_numpy(dtype=float),
        "minute": ts.minute.to_numpy(dtype=float),
        "ret_1": close.pct_change(1).to_numpy(dtype=float),
        "ret_5": close.pct_change(5).to_numpy(dtype=float),
    }, index=bars.index)
    return out[ML_FEATURES]


def _label_baseline_trades(bars: pd.DataFrame,
                           entries: pd.Series,
                           exits: pd.Series,
                           symbol: str | None,
                           cfg: BacktestConfig | None,
                           direction: str = "long") -> dict[str, np.ndarray]:
    """
    Resolve the baseline's signals into labelled trades.

    Reproduces the engine's execution rules rather than approximating them: a
    signal on bar i fills on bar i+1's OPEN, and `_pair_trades` applies the
    same first-exit-strictly-after-entry pairing `_simulate` uses. A label
    computed off close-to-close would be scoring a trade the engine never took.

    Returns arrays of equal length:
        signal_idx  bar the entry was signalled on - where features are read
        exit_idx    bar the position was closed on - when the label becomes known
        label       1 if the trade made money net of costs, else 0

    Costs enter the label when `symbol` is supplied. They change the sign of
    marginal trades, and a filter trained on gross outcomes learns to keep
    trades that lose money after commission.

    `direction` is "long" or "short" and sets the SIGN of the gross P&L. It is
    a required distinction rather than a convenience: a short's gross is
    `entry - exit`, so labelling a short trade list with the long formula marks
    every winner a loser and every loser a winner. The classifier then learns
    the edge exactly inverted, suppresses the trades that would have made money
    and keeps the ones that lost - and Version B comes back with a smooth,
    plausible, precisely backwards equity curve. Nothing raises.
    """
    if direction not in ("long", "short"):
        raise ValueError(
            f"direction must be 'long' or 'short'; got {direction!r}")
    ent = np.roll(entries.to_numpy(dtype=bool), 1)
    exi = np.roll(exits.to_numpy(dtype=bool), 1)
    ent[0] = False
    exi[0] = False

    e_idx, x_idx = _pair_trades(ent, exi)
    empty = {"signal_idx": np.empty(0, dtype=np.int64),
             "exit_idx": np.empty(0, dtype=np.int64),
             "label": np.empty(0, dtype=np.int64)}
    if e_idx.size == 0:
        return empty

    px = bars["open"].to_numpy(dtype=float)
    entry_px = px[e_idx]
    exit_px = px[x_idx]

    # Signed by the side, matching the engine's own trade P&L.
    gross = (exit_px - entry_px) if direction == "long" else (entry_px - exit_px)
    if symbol is not None:
        spec = get_spec(symbol)
        config = cfg or BacktestConfig()
        # round_turn_cost is dollars per contract for the whole round trip;
        # dividing by the multiplier puts it back into price points so it can
        # be compared against a price difference.
        cost_points = (round_turn_cost(symbol, config)
                       / (float(spec.multiplier) * config.contracts))
        net = gross - cost_points
    else:
        net = gross

    return {"signal_idx": (e_idx - 1).astype(np.int64),
            "exit_idx": x_idx.astype(np.int64),
            "label": (net > 0).astype(np.int64)}


def apply_ml_signal_filter(bars: pd.DataFrame,
                           entries: pd.Series,
                           exits: pd.Series,
                           symbol: str | None = None,
                           cfg: BacktestConfig | None = None,
                           threshold: float = 0.50,
                           min_train_trades: int = MIN_TRAIN_TRADES,
                           random_state: int = 0,
                           direction: str = "long") -> tuple[pd.Series, pd.Series]:
    """
    Version B of the Dual-Version Mandate: suppress the baseline's entries the
    classifier expects to lose.

    Takes the rule-based signals and returns the same pair with some entries
    turned off. Exits are returned untouched - an exit with no open position is
    dropped by `clean_signals` downstream, so suppressing an entry cleanly
    removes the whole trade.

    Causality
    ---------
    This is an expanding-window walk-forward, not a fitted model applied to its
    own training data, and the distinction is the entire value of the function.
    For a candidate entry signalled on bar `s`, the model is fitted ONLY on
    trades that had already CLOSED before `s`:

        exit_idx < s

    Not "entered before s". A trade that opened last week and closes tomorrow
    has no label today, and training on it leaks the future outcome into a
    decision made before that outcome existed. That distinction is invisible in
    the equity curve - it just makes Version B look brilliant - which is why it
    is enforced here rather than left to the caller.

    The classifier is refitted whenever the pool of completed trades grows, so
    every decision uses the largest strictly-historical sample available. Fits
    cost one per completed trade, not one per bar.

    Warm-up
    -------
    Until `min_train_trades` trades have closed, or while only one class has
    been seen, entries pass through unchanged. Version B therefore begins life
    identical to Version A and diverges as evidence accumulates. Any comparison
    between the two must be read with that in mind: the early sample is shared,
    so the versions are not independent.

    Parameters
    ----------
    symbol, cfg
        Supply both to include commission and slippage in the training labels.
        Without them the model learns from gross outcomes and will keep trades
        that lose money after costs.
    threshold
        Keep the entry when P(win) >= this. 0.50 is "more likely than not".

    Returns (entries, exits), boolean Series on the input index.

    Direction
    ---------
    This filters ONE side. `direction` says which, and it reaches
    `_label_baseline_trades`, where it sets the sign of the gross P&L a label
    is computed from - see there for why a mislabelled side produces a
    confidently inverted filter rather than an error.

    A bidirectional strategy therefore calls this twice, once per side, and
    gets TWO classifiers rather than one trained across both. That is
    deliberate: each side is fitted only on its own completed trades, so a
    short is never scored by a model whose entire training set is longs, and a
    long-only run is bit-for-bit the run it was before shorts existed. The cost
    is that neither model can learn from the other side's evidence, which
    roughly halves each one's training sample on a strategy that trades both
    ways evenly - so `min_train_trades` bites later on each side, and Version B
    stays identical to Version A for longer.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "apply_ml_signal_filter needs scikit-learn. "
            "Run: uv pip install -r requirements.txt"
        ) from e

    entries = pd.Series(entries).fillna(False).astype(bool)
    exits = pd.Series(exits).fillna(False).astype(bool)
    if len(entries) != len(bars) or len(exits) != len(bars):
        raise ValueError(
            f"signals and bars disagree on length: {len(entries)}/{len(exits)} "
            f"signals for {len(bars)} bars")

    trades = _label_baseline_trades(bars, entries, exits, symbol, cfg,
                                    direction=direction)
    kept = entries.to_numpy(dtype=bool).copy()
    signal_bars = np.flatnonzero(kept)

    if trades["signal_idx"].size == 0 or signal_bars.size == 0:
        return entries, exits

    features = causal_features(bars).to_numpy(dtype=float)
    train_rows = features[trades["signal_idx"]]
    labels = trades["label"]
    # _pair_trades emits trades in order, so exits are non-decreasing and the
    # count of completed trades before bar s is a searchsorted, not a scan.
    exit_idx = trades["exit_idx"]

    model = None
    fitted_n = -1
    for s in signal_bars:
        n_available = int(np.searchsorted(exit_idx, s, side="left"))
        if n_available < min_train_trades:
            continue                       # warm-up: pass through unfiltered

        y = labels[:n_available]
        if np.unique(y).size < 2:
            continue                       # one class so far; nothing to learn

        if n_available != fitted_n:
            model = HistGradientBoostingClassifier(
                max_iter=100, max_depth=3, learning_rate=0.1,
                min_samples_leaf=5, early_stopping=False,
                random_state=random_state)
            model.fit(train_rows[:n_available], y)
            fitted_n = n_available

        p_win = float(model.predict_proba(features[s:s + 1])[0, 1])
        if p_win < threshold:
            kept[s] = False

    return pd.Series(kept, index=entries.index), exits


# --------------------------------------------------------------------------
# 6. Strategy boilerplate
# --------------------------------------------------------------------------
BOILERPLATE = '''"""
{name} - {description}

Generated by agents.tier3_workers.generate_strategy_boilerplate on {date}.

THIS IS A TEMPLATE. The signal logic below is a placeholder and must be
replaced before the result of any backtest over it means anything.

The streaming interface
-----------------------
`signal_fn` receives ONE symbol's bars and returns two boolean Series aligned
to them. It never sees more than one instrument, which is deliberate: the
engine used to accept a multi-symbol frame, and a `close.rolling(200).mean()`
over it averaged across 27 unrelated contracts without raising.

Rules for anything written here:

  - No data access. The engine reads the bars and hands them over.
  - No cost handling. Slippage and commission live in BacktestConfig.
  - No session logic. flat_by_close and the Sunday merge are handled upstream.
  - No lookahead. Anything derived from bar i must use data up to i only;
    the engine fills at the NEXT bar's open.
"""

from __future__ import annotations

import pandas as pd

TIMEFRAME = "{timeframe}"
SYMBOLS = {symbols}
DEFAULT_PARAMS = {params}


def make_signal_fn({signature}):
    """Bind parameters and return the signal function the engine calls."""
{validation}

    def signal_fn(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = bars["close"]

        # ------------------------------------------------------------------
        # PLACEHOLDER LOGIC - replace this.
        # Written as a trivially inspectable crossover so a first run
        # exercises the plumbing without pretending to be a strategy.
        #
        # Note the _ma suffixes: assigning to the parameter name here would
        # make it a local of signal_fn and shadow the closure, raising
        # UnboundLocalError on the first call.
        # ------------------------------------------------------------------
        fast_ma = close.rolling({fast_ref}, min_periods={fast_ref}).mean()
        slow_ma = close.rolling({slow_ref}, min_periods={slow_ref}).mean()

        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        return entries.fillna(False), exits.fillna(False)

    return signal_fn
'''


# --------------------------------------------------------------------------
# Validating generated strategy code
# --------------------------------------------------------------------------
# Importing a module executes it, so anything written here runs. Model-authored
# code gets an allowlist rather than a denylist: a denylist is a guess about
# what is dangerous, and a strategy legitimately needs nothing beyond arrays.
# The five libraries a signal function has any business touching, plus
# `__future__`, which is a compiler directive rather than a runtime import and
# is what the boilerplate opens with. Deliberately NOT here: `vectorbt`. The
# open-source package is a different library with the same-looking API, and
# importing it silently changes simulation semantics.
ALLOWED_IMPORTS = {
    "__future__", "numpy", "pandas", "math", "vectorbtpro", "numba",
}

# Names that give a strategy module reach it has no reason to have.
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "breakpoint", "memoryview",
}

# Lookahead written as an index shift. Cannot catch every form, but a negative
# shift is the one that actually shows up and it is invisible in the equity
# curve - the backtest simply becomes prescient and the Sharpe looks superb.
_LOOKAHEAD_CALLS = {"shift", "diff", "pct_change"}


class GeneratedCodeError(Exception):
    """Generated strategy code failed validation before it could be run."""


def _strip_code_fences(code: str) -> str:
    """Remove markdown fences a model added despite being told not to."""
    text = (code or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip() + "\n"


def _audit_ast(tree: ast.AST) -> list[str]:
    """Structural objections to generated code. Empty list means it may run."""
    problems: list[str] = []

    for node in ast.walk(tree):
        # Imports -------------------------------------------------------
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    problems.append(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                problems.append(f"import from '{node.module}' is not allowed")

        # Dangerous builtins --------------------------------------------
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"use of '{node.id}' is not allowed")

        # Dunder access, the usual sandbox escape ------------------------
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                problems.append(f"access to '{node.attr}' is not allowed")

    # Negative shifts, checked where the call node is in hand.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None)
        if name not in _LOOKAHEAD_CALLS:
            continue
        for arg in list(node.args) + [k.value for k in node.keywords]:
            # Any unary minus is rejected, whatever it wraps. Restricting this
            # to literals let `shift(-k)` and `shift(-(n))` through, which is
            # the same lookahead written one character differently.
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                shown = getattr(arg.operand, "value",
                                getattr(arg.operand, "id", "…"))
                problems.append(
                    f"lookahead: {name}(-{shown}) shifts future data into "
                    f"the present"
                )
            elif (isinstance(arg, ast.Constant)
                  and isinstance(arg.value, (int, float)) and arg.value < 0):
                problems.append(
                    f"lookahead: {name}({arg.value}) shifts future data into "
                    f"the present"
                )

    # Reversed slices, the other common way to see the future.
    for node in ast.walk(tree):
        if isinstance(node, ast.Slice) and isinstance(node.step, ast.UnaryOp):
            if isinstance(node.step.op, ast.USub):
                problems.append(
                    "lookahead: a reversed slice ([::-1]) on a price series "
                    "lets a bar depend on later bars"
                )
    return problems


ENGINE_ADAPTER = '''

# ---------------------------------------------------------------------------
# Engine adapter - written by agents.tier3_workers, NOT by the model.
#
# The generated `signal_fn(bars, **params)` already matches what
# backtest.engine calls, so this no longer reshapes arguments. It exists for
# the two things the model still cannot be trusted to get right:
#
#   1. Parameter binding. The engine calls `signal_fn(bars)` with no params,
#      so a parameterised module needs a `make_signal_fn(**params)` factory
#      for load_strategy to bind against.
#   2. Boolean coercion. A model that returns a price series instead of a
#      comparison must fail loudly rather than trade every bar.
#
# It is generated deterministically so correctness does not depend on the
# model getting the glue right.
# ---------------------------------------------------------------------------
import numpy as _np_adapter
import pandas as _pd_adapter


def _as_bool_signal(values, index, label):
    """
    Coerce a returned signal to boolean, refusing anything that is not one.

    Blanket .astype(bool) is what makes this dangerous: a strategy that returns
    the close series instead of a comparison becomes True on every nonzero bar,
    which is a position opened every bar and an equity curve that looks like
    leverage rather than a bug. Only genuine booleans, or numerics that are
    exactly 0/1, are accepted.
    """
    arr = _np_adapter.asarray(values)
    if arr.dtype != bool:
        finite = arr[_np_adapter.isfinite(arr)] if arr.dtype.kind == "f" else arr
        if finite.size and not _np_adapter.isin(finite, (0, 1)).all():
            raise ValueError(
                f"{label} must be boolean; got dtype {arr.dtype} with values "
                f"outside {{0, 1}} (min {finite.min()}, max {finite.max()}). "
                f"Return a comparison, not a price series."
            )
    return (_pd_adapter.Series(arr, index=index)
            .fillna(False).astype(bool))


def make_signal_fn(**params):
    def _engine_signal_fn(bars):
        entries, exits = signal_fn(bars, **params)
        idx = bars.index
        return (_as_bool_signal(entries, idx, "entries"),
                _as_bool_signal(exits, idx, "exits"))

    return _engine_signal_fn
'''


def _smoke_test(fn: Callable, n: int = 240) -> None:
    """
    Call the strategy once on synthetic bars.

    Callable-but-broken is the normal failure for generated code, and finding
    it here costs a millisecond where finding it inside the engine costs a
    full lake read first.
    """
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.standard_normal(n))
    bars = pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": rng.integers(1, 1000, n).astype("uint64"),
    }, index=pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"))

    try:
        result = fn(bars)
    except Exception as e:
        raise GeneratedCodeError(
            f"strategy raised on a synthetic frame: {type(e).__name__}: {e}"
        ) from e

    if not isinstance(result, tuple) or len(result) != 2:
        raise GeneratedCodeError(
            f"strategy must return (entries, exits); got {type(result).__name__}"
        )
    entries, exits = result
    for label, series in (("entries", entries), ("exits", exits)):
        if len(series) != len(bars):
            raise GeneratedCodeError(
                f"{label} has length {len(series)}, expected {len(bars)}"
            )
        if pd.Series(series).dtype != bool:
            raise GeneratedCodeError(
                f"{label} must be boolean, got {pd.Series(series).dtype}"
            )


def write_and_validate_strategy(name: str, code_str: str,
                                out_dir: str | Path | None = None,
                                overwrite: bool = True) -> Path:
    """
    Validate generated strategy code, write it, and prove it runs.

    Order matters: everything that can be checked without executing the code is
    checked first, because importing a module runs it.

      1. `ast.parse` - raises SyntaxError with the offending line
      2. structural audit - import allowlist, forbidden builtins, lookahead
      3. a `signal_fn` definition must exist
      4. write, appending the deterministic engine adapter
      5. import and smoke-test on a synthetic frame

    Raises `SyntaxError` when the code will not compile and
    `GeneratedCodeError` for everything else, both with the reason attached so
    Tier 1 can report it rather than retrying blind.
    """
    code = _strip_code_fences(code_str)
    if not code.strip():
        raise GeneratedCodeError("generated code is empty")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(
            f"generated strategy does not compile at line {e.lineno}: {e.msg}"
        ) from e

    problems = _audit_ast(tree)
    if problems:
        raise GeneratedCodeError(
            "generated strategy failed the safety audit: "
            + "; ".join(sorted(set(problems)))
        )

    defs = {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "signal_fn" not in defs:
        raise GeneratedCodeError(
            f"generated strategy defines no signal_fn (found: "
            f"{sorted(defs) or 'nothing'})"
        )

    slug = "".join(c if c.isalnum() or c == "_" else "_" for c in name.strip().lower())
    slug = "_".join(filter(None, slug.split("_"))) or "generated"
    if slug[0].isdigit():
        slug = f"s_{slug}"

    directory = Path(out_dir) if out_dir else EXPERIMENTAL
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.py"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")

    body = code
    if "make_signal_fn" not in defs:
        body = code.rstrip() + "\n" + ENGINE_ADAPTER
    path.write_text(body)

    # Only now is anything executed.
    try:
        fn, _info = load_strategy(path)
    except StrategyLoadError as e:
        raise GeneratedCodeError(f"generated strategy would not import: {e}") from e
    if not callable(fn):
        raise GeneratedCodeError("resolved strategy is not callable")

    _smoke_test(fn)
    return path


def generate_strategy_boilerplate(name: str,
                                  description: str = "candidate strategy",
                                  params: dict[str, Any] | None = None,
                                  symbols: list[str] | None = None,
                                  timeframe: str = DEFAULT_TIMEFRAME,
                                  out_dir: str | Path | None = None,
                                  overwrite: bool = False) -> Path:
    """
    Write a new candidate strategy into strategies/experimental/.

    The generated module follows the streaming `signal_fn` contract and carries
    the constraints in its docstring, so a strategy written from it starts out
    unable to express the multi-symbol bleed the engine was redesigned to
    prevent.

    The placeholder logic is labelled as a placeholder. A generator that emits
    a plausible-looking strategy invites someone to backtest the template and
    read the number.
    """
    from datetime import date

    slug = "".join(c if c.isalnum() or c == "_" else "_" for c in name.strip().lower())
    slug = "_".join(filter(None, slug.split("_")))
    if not slug:
        raise ValueError(f"cannot derive a module name from {name!r}")
    if slug[0].isdigit():
        slug = f"s_{slug}"

    directory = Path(out_dir) if out_dir else EXPERIMENTAL
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.py"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass overwrite=True to replace it."
        )

    p = dict(params or {"fast": 20, "slow": 50})
    fast_ref = "fast" if "fast" in p else repr(next(iter(p.values()), 20))
    slow_ref = "slow" if "slow" in p else repr(50)

    signature = ", ".join(f"{k}: {type(v).__name__} = {v!r}" for k, v in p.items())
    validation = "\n".join(
        f"    if not isinstance({k}, (int, float)) or {k} <= 0:\n"
        f"        raise ValueError(f\"{k} must be positive, got {{{k}!r}}\")"
        for k, v in p.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ) or "    pass"

    path.write_text(BOILERPLATE.format(
        name=name, description=description,
        date=date.today().isoformat(),
        timeframe=timeframe,
        symbols=repr(symbols or ["ES", "NQ"]),
        params=repr(p),
        signature=signature,
        validation=validation,
        fast_ref=fast_ref, slow_ref=slow_ref,
    ))
    return path


# --------------------------------------------------------------------------
# Agent orchestration - still scaffold
# --------------------------------------------------------------------------
def run_variant(order: WorkOrder, signal_fn: SignalFn) -> WorkReport:
    """Run one strategy variant and save its artifacts."""
    raise NotImplementedError("tier3_workers: not implemented yet")


def run_dual_version(order: WorkOrder,
                     version_a: SignalFn,
                     version_b: SignalFn) -> tuple[WorkReport, WorkReport]:
    """
    Run the rule-based baseline and the ML-filtered variant on identical terms.

    Same symbols, same window, same costs, same constraints. The only thing
    that may differ between the two is the signal logic.
    """
    raise NotImplementedError("tier3_workers: not implemented yet")


def generate_ml_filter(order: WorkOrder, base: SignalFn) -> SignalFn:
    """Fit a Version B filter over a Version A signal. Training data must not
    include the out-of-sample window."""
    raise NotImplementedError("tier3_workers: not implemented yet")


def main() -> int:
    raise NotImplementedError("tier3_workers: not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
