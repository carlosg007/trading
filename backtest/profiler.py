import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import os

from backtest.pipeline import pipeline_dir
from mdlib import regimes as _regime_cache

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

# The bar column `mdlib.lake` joins on from the pre-computed regime cache, and
# the integer -> label map. Built FROM `mdlib.regimes` rather than spelled out
# again here, and checked against REGIMES at import: two orderings of the same
# four quadrants would put a trade in "High Volatility / Trending" when the
# cache said "Low Volatility / Ranging", and every number downstream would
# still add up.
PRECOMPUTED_COLUMN = "regime_quadrant"
UNKNOWN_REGIME = "UNKNOWN"

QUADRANT_TO_REGIME = {q: _regime_cache.QUADRANT_LABELS[q] for q in (1, 2, 3, 4)}
if tuple(QUADRANT_TO_REGIME[q] for q in (1, 2, 3, 4)) != REGIMES:
    raise ImportError(
        f"quadrant encoding disagreement: mdlib.regimes numbers the quadrants "
        f"{[QUADRANT_TO_REGIME[q] for q in (1, 2, 3, 4)]} but "
        f"backtest.profiler.REGIMES orders them {list(REGIMES)}. A cached "
        f"quadrant and a profiled label would name different environments.")

# "Q1" and "High Volatility / Trending" as the same statement, inverted from
# the integer map checked above rather than spelled out again.
# `backtest.baseline.QUADRANT_ID` is an alias for this dict, not a second copy:
# a transposed literal there would move every trade between quadrants with
# every count in every table still adding up.
REGIME_TO_QUADRANT = {label: f"Q{q}" for q, label in QUADRANT_TO_REGIME.items()}


# --------------------------------------------------------------------------
# TRUE HOME REGIME DISCOVERY - scoring, the sample floor, and designation
#
# One implementation, used by the profiler here, by Stage 1's screen
# (`backtest.baseline.best_quadrant`) and, through the handoff, by Gate R.
# Before 2026-08-21 the profiler ranked on profit factor alone while Stage 1
# ranked on profit factor with a different trade floor and a different
# tie-break, so a profile artifact and the handoff beside it could name
# DIFFERENT home quadrants for the same run with nothing raising.
# --------------------------------------------------------------------------

# The sample floor for a DESIGNATION. Two bars, and the LARGER binds: 50
# trades absolute, or 10% of everything the run placed in a quadrant.
#
# The absolute bar is there because a quadrant is picked as the best of four,
# and a profit factor over a few dozen trades clears any bar by accident often
# enough to matter across a 108-configuration screen. The fraction is there
# because 50 stops being a meaningful floor once a run places 5,000 trades -
# a quadrant holding 1% of the sample is a corner of the window, not the
# environment the strategy lives in.
#
# This is DELIBERATELY stricter than Gate R's holdout floor
# (`baseline.MIN_REGIME_TRADES`, 30). They answer different questions: this
# one asks whether there is enough in-sample evidence to NAME a home regime,
# and Gate R asks whether the named one still traded out of sample. A holdout
# is shorter than the window that chose it, so holding it to the designation
# floor would fail configurations for the length of the holdout.
DESIGNATION_MIN_TRADES = 50
DESIGNATION_MIN_TRADE_FRACTION = 0.10

# A quadrant is only a candidate to be someone's HOME if it made money there.
# `Net_PnL x PF` is monotone in both terms only over positive net P&L: at a
# profit factor of 0.00 - a quadrant that never had a winning trade - the
# product is exactly 0.0 and would rank ABOVE a quadrant that lost $5,000 at
# a 0.50 factor (-2,500). Requiring positive expectancy removes that inversion
# from the selection path entirely rather than patching the formula, and every
# quadrant is still SCORED and reported so the ranking can be checked.
DESIGNATION_MIN_PROFIT_FACTOR = 1.00

# The profiler writes 999 when a quadrant never had a losing trade. That is a
# SENTINEL, not a measured factor, and `net_pnl * 999` ranks on it rather than
# on evidence - one unbeaten 51-trade quadrant would outscore a 4,000-trade
# engine by two orders of magnitude. Capped for SCORING only; the reported
# `profit_factor` is left exactly as the profiler computed it, and a row whose
# factor was capped says so.
SCORE_PF_CEILING = 10.0

DESIGNATION_RULE = ("primary = max(Net_PnL x Profit_Factor) among quadrants "
                    "with positive net P&L, profit factor >= "
                    f"{DESIGNATION_MIN_PROFIT_FACTOR:.2f} and at least "
                    f"max({DESIGNATION_MIN_TRADES}, "
                    f"{DESIGNATION_MIN_TRADE_FRACTION:.0%} of profiled trades)")


def _quadrant_risk(pnl) -> dict:
    """
    One quadrant's own Sharpe and drawdown, and what they are NOT.

    Both are QUADRANT-LOCAL. A quadrant's trades are scattered through the run
    - the tape moves in and out of High-Vol/Trending all year - so this is the
    risk of the STREAM OF TRADES THIS QUADRANT CONTRIBUTED, in entry order,
    and not the risk the account carried. The account's drawdown is the whole
    equity path and lives on the run metrics; mixing the two would put a
    quadrant-sized figure under an account-sized heading with both individually
    true, which is the mistake this repository keeps making room to avoid.

    `sharpe_trade` IS NOT ANNUALISED, and the name says so. Annualising needs a
    time basis, and a non-contiguous subset of the calendar has none that is
    not invented: 40 trades drawn from four separate High-Vol weeks do not
    describe a year. It is mean(pnl) / stdev(pnl) over the quadrant's trades -
    a per-trade signal-to-noise ratio, comparable across quadrants of the SAME
    run, which is exactly the comparison the all-quadrant gate makes. Reading
    it against an annualised Sharpe from anywhere else is a category error.

    `max_drawdown_pnl` is the deepest peak-to-trough fall of this quadrant's
    CUMULATIVE P&L, in dollars, and is <= 0 by construction (0 when the
    quadrant only ever made new highs). Dollars rather than a percent: a
    percent needs a capital base, the base here is the whole account, and the
    quotient would then describe neither the quadrant nor the account.

    A single trade has no dispersion, so `sharpe_trade` is None rather than
    infinite - one observation is not a measurement, and a None the gate
    refuses is safer than a large number it passes.
    """
    series = pd.Series(pnl, dtype="float64").dropna()
    n = int(len(series))
    if n == 0:
        return {"sharpe_trade": None, "max_drawdown_pnl": None}
    sd = float(series.std(ddof=1)) if n > 1 else 0.0
    sharpe = (round(float(series.mean()) / sd, 3)
              if n > 1 and sd > 0 else None)
    curve = series.cumsum()
    drawdown = float((curve - curve.cummax()).min())
    return {"sharpe_trade": sharpe,
            "max_drawdown_pnl": round(min(drawdown, 0.0), 2)}


def _score_num(value):
    """A float, or None - never a NaN masquerading as a measurement."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


#: `Q1`..`Q4` -> the regime label, inverted from the integer map rather than
#: spelled out again.
QUADRANT_TO_REGIME_CODE = {f"Q{q}": label
                           for q, label in QUADRANT_TO_REGIME.items()}


def normalize_target_quadrants(declared) -> tuple[str, ...]:
    """
    A module's `TARGET_QUADRANTS` as canonical `Q1`..`Q4` codes.

    BOTH SPELLINGS ARE ACCEPTED, because both are already in the tree: modules
    declare `("Q3",)` beside a `TARGET_REGIMES = ("Low Volatility / Trending",)`
    and either is the same statement. They resolve through
    `REGIME_TO_QUADRANT`/`QUADRANT_TO_REGIME`, so no spelling of a regime name
    is written down a second time here.

    AN UNKNOWN NAME RAISES. Left alone it would silently reduce to an empty set
    and restore the unrestricted best-of-four - a module that declared `Q5` or
    `"Low Vol Trend"` would be screened exactly as if it had declared nothing,
    with its designation quietly decided by dollar alpha and every log line
    reading correctly. That is the failure this whole mechanism exists to
    remove, so a typo has to be loud.

    An empty or absent declaration returns `()`, which every caller reads as
    "no restriction".
    """
    if not declared:
        return ()
    if isinstance(declared, str):
        declared = (declared,)
    out: list[str] = []
    for item in declared:
        name = str(item).strip()
        if name in REGIME_TO_QUADRANT:                 # a full regime label
            code = REGIME_TO_QUADRANT[name]
        elif name.upper() in QUADRANT_TO_REGIME_CODE:  # a Q1..Q4 code
            code = name.upper()
        else:
            raise ValueError(
                f"TARGET_QUADRANTS names {item!r}, which is neither a quadrant "
                f"code {sorted(QUADRANT_TO_REGIME_CODE)} nor a regime label "
                f"{sorted(REGIME_TO_QUADRANT)}. Refusing to reduce it to 'no "
                f"declaration' - that would screen this module on dollar alpha "
                f"exactly as if it had declared nothing.")
        if code not in out:
            out.append(code)
    return tuple(out)



def _would_designate(rows: list[dict], score_mode: str) -> dict | None:
    """
    What the OTHER ranking rule would have designated from the same table.

    Reported, never applied. A disagreement between the two is the finding
    worth surfacing - it says this configuration's home quadrant is an artefact
    of which scale the score is measured on, which is exactly the question the
    Q3 audit asked - and a reader who had to re-derive it downstream would be
    free to apply a different floor than the one that made the decision.

    `None` when the two agree, so a disagreement is never buried in a field
    that is always populated.
    """
    other = "vol_normalized" if score_mode == "alpha" else "alpha"
    key = "vol_normalized_score" if other == "vol_normalized" else "alpha_score"
    eligible = [r for r in rows if r["eligible"] and r.get(key) is not None]
    if not eligible:
        return None
    winner = max(eligible, key=lambda r: (r[key], r["trade_count"]))
    current = next((r for r in rows if r["eligible"]), None)
    if current is not None and winner["regime"] == current["regime"]:
        return None
    return {"score_mode": other, "regime": winner["regime"],
            "quadrant": winner["quadrant"], "score": winner[key],
            "note": (f"under {other} scoring the home regime would be "
                     f"{winner['regime']} rather than "
                     f"{(current or {}).get('regime')}")}


def designation_floor(total_profiled: int,
                      min_trades: int = DESIGNATION_MIN_TRADES,
                      fraction: float = DESIGNATION_MIN_TRADE_FRACTION) -> int:
    """
    How many trades a quadrant needs before it may be NAMED the home regime.

    `max(min_trades, ceil(fraction * total_profiled))`. The larger of the two
    binds, so the floor is an absolute minimum on a short run and a share of
    the sample on a long one. `fraction=0` reduces it to the flat count, which
    is how a caller asks for the pre-2026-08-21 behaviour.

    `total_profiled` is the trades PLACED in a quadrant, not the trade list:
    an unplaced trade (entry outside the frame, or inside the 14-bar indicator
    warm-up) is in no quadrant, so counting it would raise every quadrant's
    bar on the strength of trades no quadrant could ever claim.
    """
    total = max(int(total_profiled or 0), 0)
    share = -((-total * float(fraction or 0.0)) // 1)      # ceil, no math import
    return max(int(min_trades or 0), int(share))


def quadrant_score(stats: dict | None) -> float | None:
    """
    `Net_PnL x Profit_Factor` for one quadrant - the alpha contribution, not
    the per-trade edge.

    Ranking on profit factor alone answers "where is this strategy sharpest",
    which is not the same question as "where does this strategy make its
    money". A 1.55 factor over 45 trades and a 1.28 over 4,000 are both real,
    and the second is the engine. Multiplying by net P&L is what makes the
    score prefer it; a Sharpe computed inside the quadrant would express the
    same preference, and is not used here because the profiler is handed a
    trade list rather than an equity curve, and a Sharpe over per-trade P&L is
    not the daily-close Sharpe every other number in this repo means.

    `None` when either term is missing - never 0.0, which is a score a
    quadrant can legitimately have.
    """
    if not stats:
        return None
    pf = _score_num(stats.get("profit_factor"))
    net = _score_num(stats.get("net_pnl"))
    if pf is None or net is None:
        return None
    return net * min(pf, SCORE_PF_CEILING)


def vol_normalized_score(stats: dict | None) -> float | None:
    """
    The same alpha contribution with the DOLLAR SIZE of the regime divided out:
    `(net_pnl / avg_trade_abs_pnl) x min(PF, ceiling)`.

    WHY THE DEFAULT SCORE HAS A DIRECTION, measured 2026-09-02 across
    6E/6J/ES/NQ/GC/CL. The engine is fixed-size (`BacktestConfig.contracts = 1`,
    `size_type="amount"`), so per-trade P&L moves with the size of the move.
    Q3's mean ATR is 0.28x Q1's - 0.18x on NQ - and Q3 holds 0.67x the bars, so
    at an EQUAL profit factor a Q3 quadrant scores about 0.19x a Q1 one. To
    outscore a Q1 quadrant running PF 1.20 a Q3 quadrant has to reach PF 1.74,
    before Gate R has looked at anything. That is why 20 of 20 instances of the
    three trend-drift archetypes were designated into a high-volatility
    quadrant, `keltner_trend_drift_20260901` included - a module that declares
    `TARGET_QUADRANTS = ("Q3",)`.

    `net_pnl / avg_trade_abs_pnl` is the quadrant's expectancy in units of ITS
    OWN average trade - R-multiples - so a regime whose trades are a third the
    size no longer scores a third as well for the same edge. The divisor is
    measured from the same trades the profit factor is measured from rather
    than from an indicator, so it needs no new plumbing and cannot disagree
    with the rest of the row.

    NOT theta_vol, which the request suggested. `theta_vol` is ONE scalar per
    (symbol, TIMEFRAME) - the boundary between high and low volatility, not a
    per-quadrant statistic - so dividing all four quadrants by it is dividing
    by a constant and leaves the ranking exactly as it was.

    THIS IS NOT THE DEFAULT. `designate(score_mode=...)` selects, both scores
    are computed and reported on every row either way, and `alpha` remains what
    sorts unless a caller asks otherwise. Every strategy in
    `config/portfolios.json` was designated under `alpha`, and a rule that
    silently re-designated them would leave the live registry's quadrants and
    the rule that produced them disagreeing with nothing raising.

    `None` when either term is missing, or when the average trade is 0.0 - a
    quadrant whose trades all closed exactly flat has no scale to divide by,
    and 0/0 is not a ranking.
    """
    if not stats:
        return None
    pf = _score_num(stats.get("profit_factor"))
    net = _score_num(stats.get("net_pnl"))
    scale = _score_num(stats.get("avg_trade_abs_pnl"))
    if pf is None or net is None or scale is None or scale <= 0.0:
        return None
    return (net / scale) * min(pf, SCORE_PF_CEILING)


#: The two ranking rules, and the only place their names are written down.
#: `alpha` is the shipped default; see `vol_normalized_score`.
SCORE_MODES = ("alpha", "vol_normalized")


def score_for(stats: dict | None, mode: str = "alpha") -> float | None:
    """The score `mode` ranks on. An unknown mode RAISES rather than falling
    back to the default - a run that silently ranked on something other than
    what was asked for is a designation nobody can check."""
    if mode not in SCORE_MODES:
        raise ValueError(f"score_mode must be one of {SCORE_MODES}; got "
                         f"{mode!r}")
    return (quadrant_score(stats) if mode == "alpha"
            else vol_normalized_score(stats))


def rank_quadrants(breakdown: dict | None, floor: int,
                   min_profit_factor: float = DESIGNATION_MIN_PROFIT_FACTOR,
                   score_mode: str = "alpha") -> list[dict]:
    """
    Every quadrant in `breakdown`, scored and sorted best first, each carrying
    whether it is ELIGIBLE to be designated and - when it is not - which bar
    it missed.

    Every quadrant is returned, including the ones that lost money. A ranking
    that dropped them would make "this quadrant was disqualified on sample
    size" and "this quadrant was never traded" the same absent row, and they
    are fixed by different work.

    Sorted on score, ties broken on the LARGER trade count and then on the
    declared regime order. Ties are real: a quadrant no bar reaches and one
    the strategy never traded in round to the same numbers, and between two
    equal scores the better-evidenced one is the honest winner rather than
    whichever the quadrant order happened to put first.
    """
    if score_mode not in SCORE_MODES:
        raise ValueError(f"score_mode must be one of {SCORE_MODES}; got "
                         f"{score_mode!r}")
    rows = []
    for regime in REGIMES:
        stats = (breakdown or {}).get(regime)
        if not stats:
            continue
        pf = _score_num(stats.get("profit_factor"))
        net = _score_num(stats.get("net_pnl"))
        n = int(stats.get("trade_count", 0) or 0)
        # BOTH ARE ALWAYS COMPUTED, whichever one sorts. A reader comparing the
        # two columns can see what the other rule would have designated without
        # re-running anything, which is the whole point of shipping this as a
        # reported alternative rather than as a switched default.
        alpha = quadrant_score(stats)
        vol_norm = vol_normalized_score(stats)
        score = score_for(stats, score_mode)

        reasons = []
        if score is None:
            reasons.append("no profit factor or net P&L was recorded")
        else:
            if net <= 0:
                reasons.append(f"net P&L {net:,.2f} is not positive")
            if pf is not None and pf < float(min_profit_factor):
                reasons.append(f"profit factor {pf:.2f} is below "
                               f"{float(min_profit_factor):.2f}")
            if n < int(floor):
                reasons.append(f"{n} trades is below the sample floor "
                               f"of {int(floor)}")
        rows.append({
            "regime": regime,
            "quadrant": REGIME_TO_QUADRANT[regime],
            "trade_count": n,
            "profit_factor": pf,
            "net_pnl": net,
            "win_rate": _score_num(stats.get("win_rate")),
            "score": score,
            # The score that SORTED, named, plus both candidates. Without the
            # name a reader cannot tell which column produced the order.
            "score_mode": score_mode,
            "alpha_score": alpha,
            "vol_normalized_score": vol_norm,
            "avg_trade_abs_pnl": _score_num(stats.get("avg_trade_abs_pnl")),
            "pf_capped": bool(pf is not None and pf > SCORE_PF_CEILING),
            "eligible": not reasons,
            "reason": "; ".join(reasons) or "clears every designation bar",
        })
    rows.sort(key=lambda r: (r["score"] is not None,
                             r["score"] if r["score"] is not None else 0.0,
                             r["trade_count"],
                             -REGIMES.index(r["regime"])), reverse=True)
    return rows


def designate(breakdown: dict | None, total_profiled: int,
              min_trades: int = DESIGNATION_MIN_TRADES,
              fraction: float = DESIGNATION_MIN_TRADE_FRACTION,
              min_profit_factor: float = DESIGNATION_MIN_PROFIT_FACTOR,
              score_mode: str = "alpha",
              target_quadrants=None) -> dict:
    """
    The TRUE HOME REGIME: one primary quadrant, its positive-expectancy
    runners-up, and the whole scored table that produced them.

    `primary` is None when nothing clears the bars - which is a finding, not a
    missing value, and is why `reason` is populated either way. A strategy with
    no environment must not be handed one by falling back to the best of a bad
    set: the quadrant becomes Gate R's certification target and a live
    supervisor's permission to trade, and neither may be derived from a
    quadrant that lost money or was measured over 20 trades.

    `secondaries` are the OTHER quadrants with positive expectancy - eligible
    ones that lost on score, and profitable ones disqualified only on sample
    size, each carrying why.

    They are NOT a second certification target in the general case: naming two
    quadrants a strategy may trade doubles Gate R's chances of clearing 1.00
    out of sample, which is the best-of-four selection Gate R exists to avoid.

    THE ONE EXCEPTION, 2026-09-08: `audit_gates.regime_gate` may certify on the
    top `eligible` entry here when the PRIMARY quadrant placed fewer than
    `MIN_REGIME_TRADES` holdout trades - i.e. when the primary was never
    measured. That is not a second chance at a bar the primary missed; it is a
    first measurement in a pre-declared alternative, and the gate refuses the
    fallback outright when the primary traded enough and lost. The declaration
    is made HERE, from in-sample bars, which is what keeps the choice off the
    holdout - so the order and the `eligible` flag on these rows are load
    bearing rather than descriptive.
    """
    floor = designation_floor(total_profiled, min_trades, fraction)
    rows = rank_quadrants(breakdown, floor, min_profit_factor, score_mode)
    eligible = [r for r in rows if r["eligible"]]

    # THE MODULE'S DECLARATION RESTRICTS THE CANDIDATES, AND NOTHING ELSE.
    # Every bar above still binds on the declared quadrant exactly as it binds
    # on any other - positive net P&L, profit factor, and the same sample floor
    # of max(50, 10% of placed). The declaration decides which environment the
    # strategy is JUDGED in; it can never decide that it passed. A restriction
    # that also relaxed a bar would be a way to certify on thinner evidence by
    # writing a constant in a module.
    declared = normalize_target_quadrants(target_quadrants)
    unrestricted = eligible[0] if eligible else None
    if declared:
        eligible = [r for r in eligible if r["quadrant"] in declared]

    primary = eligible[0] if eligible else None

    secondaries = [
        r for r in rows
        if r is not primary
        and r["net_pnl"] is not None and r["net_pnl"] > 0
        and (r["eligible"] or r["trade_count"] < floor)
    ]

    if primary:
        reason = (f"{primary['regime']} scores "
                  f"{primary['score']:,.2f} (net P&L {primary['net_pnl']:,.2f} "
                  f"x PF {primary['profit_factor']:.2f}) over "
                  f"{primary['trade_count']} trades")
        if declared:
            reason += f" — declared target {'/'.join(declared)}"
    elif declared:
        # DROPPED, and the reason says so IN FULL. "no quadrant clears the
        # bars" would be false here: one may well have, and the strategy was
        # dropped because it was not the one the module said it was for.
        # Silently re-homing it is the behaviour this replaces - a Q3 module
        # certified into Q1 because higher volatility swung larger dollars,
        # with every log line reading correctly.
        missed = [r for r in rows if r["quadrant"] in declared]
        detail = ("; ".join(f"{r['quadrant']} {r['reason']}" for r in missed)
                  if missed
                  else f"{'/'.join(declared)} holds no trades at all")
        reason = (f"declared target {'/'.join(declared)} did not clear the "
                  f"designation bars: {detail}")
        if unrestricted is not None:
            reason += (f". {unrestricted['quadrant']} DID clear them and was "
                       f"NOT substituted — the module declares which "
                       f"environment it is for")
    elif rows:
        near = rows[0]
        reason = (f"no quadrant clears the designation bars; the closest is "
                  f"{near['regime']} - {near['reason']}")
    else:
        reason = "no quadrant holds a single trade"

    return {
        "primary": primary,
        "secondaries": secondaries,
        "scores": rows,
        "sample_floor": floor,
        "sample_floor_basis": (
            f"max({int(min_trades)} trades, {float(fraction):.0%} of the "
            f"{int(total_profiled or 0)} trades placed in a quadrant)"),
        "min_profit_factor": float(min_profit_factor),
        "score_mode": score_mode,
        # The declaration as it was APPLIED, and what an unrestricted screen
        # would have designated instead. Both travel onto the handoff: a pair
        # dropped for missing its own target while another quadrant qualified
        # is a specific, actionable finding, and it is invisible if only the
        # verdict is recorded.
        "declared_quadrants": list(declared),
        "designation_restricted": bool(declared),
        "unrestricted_primary": (
            {"regime": unrestricted["regime"],
             "quadrant": unrestricted["quadrant"],
             "score": unrestricted["score"]}
            if declared and unrestricted is not None
            and (primary is None
                 or unrestricted["quadrant"] != primary["quadrant"])
            else None),
        # WHAT THE OTHER RULE WOULD HAVE DESIGNATED, on the same table. A
        # disagreement is the finding - it says this configuration's home
        # quadrant is an artefact of which scale the score is measured on -
        # and re-deriving it downstream would let a reader apply a different
        # floor than the one that made the decision.
        "would_designate": _would_designate(rows, score_mode),
        "score_formula": (
            "net_pnl * min(profit_factor, "
            f"{SCORE_PF_CEILING:.1f})" if score_mode == "alpha" else
            "(net_pnl / avg_trade_abs_pnl) * min(profit_factor, "
            f"{SCORE_PF_CEILING:.1f})"),
        "rule": DESIGNATION_RULE,
        "reason": reason,
    }


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
                 version: str = "", quiet: bool = False,
                 min_trades: int = DESIGNATION_MIN_TRADES,
                 min_trade_fraction: float = DESIGNATION_MIN_TRADE_FRACTION,
                 min_profit_factor: float = DESIGNATION_MIN_PROFIT_FACTOR,
                 target_quadrants=()):
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

        `min_trades`, `min_trade_fraction` and `min_profit_factor` are the
        DESIGNATION bars - what a quadrant must clear before it may be named
        the home regime. They are not Gate R's bars and not Stage 1's survival
        bars; see `designate`. They are arguments rather than constants because
        Stage 1 exposes them on the CLI, and a screen re-run at a different
        floor must not be silently comparable to one run at the default.
        """
        self.df = df.copy()
        self.df.index = _entry_timestamps(self.df)
        self.portfolio = portfolio
        self.strat_name = strat_name
        self.symbol = symbol
        self.tf = tf
        self.version = str(version or "")
        self.quiet = bool(quiet)
        self.min_trades = int(min_trades)
        self.min_trade_fraction = float(min_trade_fraction)
        self.min_profit_factor = float(min_profit_factor)
        # The module's TARGET_QUADRANTS, or () when it declares none.
        #
        # PASSED IN RATHER THAN LEFT OFF (2026-09-08). `generate_profile` calls
        # `designate` and writes the answer into
        # `regime_profile_<SYM>_<TF>_version_<v>.json`, and Stage 1 calls
        # `designate` AGAIN with the declaration to build the handoff. Omitted
        # here, the two disagreed for every configuration whose best-scoring
        # eligible quadrant was outside the declared set: on
        # compressed_bollinger_reversion_20260901, six of fourteen artifacts
        # named a home quadrant the handoff did not, and two designated
        # nothing where the handoff carried Q4. The artifact is the file a
        # human opens; the handoff is the one that certifies.
        self.target_quadrants = normalize_target_quadrants(target_quadrants)
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

    def _classify_bars(self) -> dict:
        """
        Write `self.df['Regime']`, and return how it was decided.

        **The pre-computed cache wins when the bars carry it.** `mdlib.lake`
        left-joins `regime_quadrant` onto every frame it returns, so a bar that
        arrived through the reader already has a quadrant computed once, from
        Wilder's ADX(14)/ATR(14), against a volatility threshold pinned to the
        in-sample window. Recomputing here would be the same indicator pass
        repeated per stage AND per version, against a DIFFERENT threshold - see
        below - so the two would disagree about which environment a trade was
        in, and both would look right.

        Quadrant 0 is the indicator warm-up (and any bar outside the cache's
        span). It maps to UNKNOWN, exactly where the live pass falls through to
        UNKNOWN for a NaN indicator, so those trades are counted as unplaced
        rather than filed under a regime nobody measured.

        The fallback is the original live pass, unchanged, and it is NOT
        equivalent: it takes the median ATR of whatever frame it was handed, so
        its threshold moves with the requested date range. It stays because a
        caller holding a hand-built frame - a test, a notebook, a strategy the
        cache has not been built for - must still get a profile rather than an
        exception. Which one ran is recorded on the artifact; a quadrant read
        without knowing which threshold drew it is not a measurement.
        """
        if PRECOMPUTED_COLUMN in self.df.columns:
            quad = self.df[PRECOMPUTED_COLUMN]
            self.df["Regime"] = (quad.map(QUADRANT_TO_REGIME)
                                     .fillna(UNKNOWN_REGIME)
                                     .astype(object))
            prov = _regime_cache.provenance(self.symbol, self.tf) or {}
            n_undefined = int((quad == _regime_cache.QUADRANT_UNDEFINED).sum())
            return {
                "regime_source": "precomputed_cache",
                "volatility_threshold": prov.get("theta_vol"),
                "volatility_threshold_basis": (
                    f"median ATR({prov.get('atr_length', 14)}) over the "
                    f"in-sample window {prov.get('is_start')} .. "
                    f"{prov.get('is_end')}"
                    if prov.get("theta_vol") is not None else
                    "not recorded - the bars carried a quadrant column but no "
                    "cache file was found for this (symbol, timeframe)"),
                "adx_trend_threshold": prov.get(
                    "adx_trend_threshold", _regime_cache.ADX_TREND_THRESHOLD),
                "cache_file": str(
                    _regime_cache.cache_path(self.symbol, self.tf)),
                "bars_without_regime": n_undefined,
            }

        # Fallback: the original live pass.
        self.df.ta.adx(length=14, append=True)
        self.df.ta.atr(length=14, append=True)

        adx_col = [c for c in self.df.columns if c.startswith('ADX')][0]
        atr_col = [c for c in self.df.columns
                   if c.startswith('ATRe') or c.startswith('ATR')][0]

        atr_median = self.df[atr_col].median()

        conditions = [
            (self.df[atr_col] > atr_median) & (self.df[adx_col] > 25),
            (self.df[atr_col] > atr_median) & (self.df[adx_col] <= 25),
            (self.df[atr_col] <= atr_median) & (self.df[adx_col] > 25),
            (self.df[atr_col] <= atr_median) & (self.df[adx_col] <= 25),
        ]
        self.df['Regime'] = np.select(conditions, list(REGIMES),
                                      default=UNKNOWN_REGIME)
        return {
            "regime_source": "recomputed_live",
            "volatility_threshold": (None if pd.isna(atr_median)
                                     else float(atr_median)),
            "volatility_threshold_basis": (
                "median ATR(14) over the frame handed to the profiler - moves "
                "with the requested date range"),
            "adx_trend_threshold": 25.0,
            "cache_file": None,
            "bars_without_regime": int(
                (self.df['Regime'] == UNKNOWN_REGIME).sum()),
        }

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

        # 1-3. Label every bar with its quadrant, from the pre-computed cache
        # when the bars carry one and from a live ADX/ATR pass when they do not.
        choices = list(REGIMES)
        provenance = self._classify_bars()
        self._say(f"[PROFILER] regime source: {provenance['regime_source']}"
                  + (f" (theta_vol={provenance['volatility_threshold']:.6f})"
                     if provenance.get("volatility_threshold") is not None
                     else ""))
        
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
                "optimal_quadrant": None,
                "optimal_profit_factor": None,
                "optimal_trade_count": 0,
                "optimal_net_pnl": None,
                "optimal_score": None,
                "kill_switch_conditions": [],
                "regime_breakdown": {},
                # The same keys a profiled run carries, so a caller reading
                # `regime_scores` never has to branch on whether anything
                # traded. An ABSENT key and an empty table are the same
                # `.get()` and mean different things.
                "regime_scores": {},
                "secondary_regimes": [],
                "designation": designate({}, 0, min_trades=self.min_trades,
                                         fraction=self.min_trade_fraction,
                                         min_profit_factor=self
                                         .min_profit_factor),
                "trades_profiled": 0,
                "trades_unplaced": 0,
                "artifact": None,
                **provenance,
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
                "net_pnl": round(net_pnl, 2),
                # THE DOLLAR SCALE OF THIS QUADRANT'S TRADES, and the divisor
                # `vol_normalized_score` uses. The engine is fixed-size, so an
                # average trade in Low-Vol/Trending is about a third the size
                # of one in High-Vol/Trending on the same contract - which is
                # why the default alpha score prefers the high-volatility
                # quadrants whatever the edge. Measured from the SAME trades
                # the profit factor is measured from, so the two cannot
                # disagree about which population they describe.
                "avg_trade_abs_pnl": round(
                    float(regime_trades['pnl'].abs().mean()), 2),
                # PER-QUADRANT RISK, for the all-quadrant robustness gate.
                # Both are QUADRANT-LOCAL and neither is an account figure -
                # see `_quadrant_risk`. Written here so the gate reads one
                # recorded number rather than re-deriving it from the trade
                # list, which is how two modules come to disagree about what
                # they measured.
                **_quadrant_risk(regime_trades['pnl']),
            }
            
            self._say(f" {regime:<30} | {count:<8} | {win_rate:>5.1f}%  | {pf:>13.2f} | ${net_pnl:,.2f}")
        self._say("="*80)

        # 5b. TRUE HOME REGIME DISCOVERY. The primary quadrant is the one that
        # CONTRIBUTES the alpha (net P&L x profit factor), not the one with the
        # sharpest per-trade edge, and it has to clear a sample floor that
        # scales with the run. `designate` is shared with Stage 1's screen, so
        # this artifact and the handoff written beside it can never name
        # different home quadrants for the same result.
        placed = int(len(trades) - unplaced)
        decision = designate(profile, placed,
                             min_trades=self.min_trades,
                             fraction=self.min_trade_fraction,
                             min_profit_factor=self.min_profit_factor,
                             target_quadrants=self.target_quadrants)
        primary = decision["primary"]
        best_regime = primary["regime"] if primary else "None"

        self._say(f" {'ALPHA SCORE (net P&L x PF)':<30} | {'SCORE':>14} | "
                  f"{'ELIGIBLE':<9}| WHY NOT")
        self._say("-" * 80)
        for row in decision["scores"]:
            score = (f"{row['score']:>14,.0f}" if row["score"] is not None
                     else f"{'not scored':>14}")
            self._say(f" {row['regime']:<30} | {score} | "
                      f"{'yes' if row['eligible'] else 'no':<9}| "
                      f"{'' if row['eligible'] else row['reason']}")
        self._say("=" * 80)
        if primary:
            self._say(f"✅ TRUE HOME REGIME: {best_regime} "
                      f"[{primary['quadrant']}]  score "
                      f"{primary['score']:,.0f} = net "
                      f"${primary['net_pnl']:,.2f} x PF "
                      f"{primary['profit_factor']:.2f} over "
                      f"{primary['trade_count']} trades "
                      f"(sample floor {decision['sample_floor']})")
        else:
            self._say(f"⚠  NO HOME REGIME DESIGNATED — "
                      f"{decision['reason']}")
        for extra in decision["secondaries"]:
            note = ("" if extra["eligible"]
                    else f"  — not designatable: {extra['reason']}")
            self._say(f"   secondary alpha: {extra['regime']} "
                      f"[{extra['quadrant']}] n={extra['trade_count']}"
                      + (f" score {extra['score']:,.0f}"
                         if extra["score"] is not None else "")
                      + note)
        self._say("")

        # 6. Save Artifact for the Live Supervisor
        file_path = self.artifact_path
        out_data = {
            "strategy": self.strat_name,
            "symbol": self.symbol,
            "timeframe": self.tf,
            "version": self.version,
            "optimal_regime": best_regime,
            # `Q1`..`Q4` beside the name, so a reader (and the CrossTrade
            # supervisor) never has to re-derive the code from the spelling.
            "optimal_quadrant": REGIME_TO_QUADRANT.get(best_regime),
            # The designated quadrant's own numbers, beside its name. Stage 1
            # screens on the PAIR (profit factor at a trade count), and a name
            # with no numbers under it forces every reader to re-derive them
            # from the breakdown - where a reader is free to apply a different
            # trade floor than the one that chose the name.
            "optimal_profit_factor": (profile.get(best_regime) or {}
                                      ).get("profit_factor"),
            "optimal_trade_count": (profile.get(best_regime) or {}
                                    ).get("trade_count", 0),
            "optimal_net_pnl": (profile.get(best_regime) or {}).get("net_pnl"),
            "optimal_score": (primary or {}).get("score"),
            "kill_switch_conditions": ([r for r in choices if r != best_regime]
                                       if best_regime in profile else []),
            "regime_breakdown": profile,
            # The whole scored table, keyed by regime: in-sample profit factor,
            # net P&L, trade count, the alpha score, and - for anything that
            # cannot be designated - which bar it missed. This is what Stage 2
            # embeds in `best_params_<SYMBOL>_<TF>.json`, so a Gate R target
            # can always be checked against the four quadrants it beat rather
            # than read as a name somebody chose.
            "regime_scores": {r["regime"]: r for r in decision["scores"]},
            # Quadrants with POSITIVE expectancy that are not the primary -
            # metadata for the live supervisor, never a second certification
            # target. Two permitted quadrants would give Gate R two chances at
            # a 1.00 holdout profit factor, which is the best-of-N selection
            # the single-quadrant rule exists to prevent.
            "secondary_regimes": decision["secondaries"],
            "designation": {k: v for k, v in decision.items()
                            if k not in ("primary", "secondaries", "scores")},
            "trades_profiled": int(len(trades) - unplaced),
            "trades_unplaced": int(unplaced),
            "artifact": file_path,
            # How the quadrants were drawn, beside the numbers they produced.
            **provenance,
        }

        with open(file_path, "w") as f:
            json.dump(out_data, f, indent=4)
        return out_data
