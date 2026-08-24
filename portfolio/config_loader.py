"""
The portfolio-to-account configuration: load it, validate it, and answer the
four questions a router asks of it.

Location:  ~/src/trading/portfolio/config_loader.py
Config:    ~/src/trading/config/portfolios.json

WHAT THIS FILE IS, AND WHAT IT IS NOT
=====================================
`config/portfolios.json` is a DECLARATION, in the same sense as
`compliance_rules/*.json`: it states which strategies trade which basket, on
which account, and inside what risk envelope. It is not an enforcement
mechanism and nothing in this repository turns it into one.

That split is the 2026-08-15 separation of concerns, and it is the reason the
numbers in `risk_profile` can sit in this repository at all:

    THE PYTHON RESEARCH BACKEND      generates signals, manages net basket
                                     positions, and sizes them from ATR.
    CROSSTRADE NAM (Windows/cloud)   enforces drawdown lockouts, loss caps and
                                     the prop-firm rulebook, against a LIVE
                                     ACCOUNT BALANCE it can actually see.

So `max_trailing_drawdown_usd` and `max_forward_incubation_dd_pct` below are
**specifications handed to CrossTrade**, not gates any Python code applies.
Nothing here reads an account balance, and `backtest/` must never import this
module — a research run that depended on which live account a strategy would be
routed to would be tuned to a funding program rather than to a market, which is
exactly what the separation exists to prevent.

`fixed_risk_budget_usd` and `clamping` ARE for this side of the line: they are
inputs to position sizing, which is the backend's job. They are still not
enforcement — a clamp of 5 contracts is how many the sizer may ask for, not a
limit anything checks after the fact.

TWO CONFLICTS BETWEEN THIS SCHEMA AND THE REST OF THE REPOSITORY
================================================================
Both are resolved here rather than left for a reader to trip over, because both
are the kind that produce a plausible number rather than an error.

1. **THE QUADRANT NUMBERING WAS NOT THIS REPOSITORY'S, AND IS NOW.** Schema
   1.0.0 named its regimes `Q1_LOW_VOL_TREND`, `Q2_HIGH_VOL_TREND`,
   `Q3_LOW_VOL_MEAN_REVERSION` and `Q4_HIGH_VOL_CHOP`. `mdlib/regimes.py` — the
   single place the quadrant encoding is written down, and the one Stage 1's
   designation, Gate R and the Discord cards all read — numbers them:

       Q1  High Volatility / Trending          Q3  Low Volatility / Trending
       Q2  High Volatility / Ranging           Q4  Low Volatility / Ranging

   Every one of the four digits disagreed, and its `Q2` was this repository's
   High-Volatility RANGING quadrant — the chop the strategy premise says to
   filter out. A live supervisor reading that file's `Q1` while Stage 1 had
   designated the repository's `Q1` would have permitted the strategy in the
   one environment nobody certified it for, with every count in every table
   still adding up.

   **Schema 1.1.0 relabels them to agree**, which is the right fix: one
   encoding, not two spellings and a translator. `CANONICAL_QUADRANT` stays,
   and `_reconcile_quadrants` now checks it against `backtest.profiler` on
   every load and REFUSES the config on a disagreement — because an agreement
   nothing verifies is an agreement that lasts until someone edits one side.
   `canonical_quadrant` remains the only supported way to resolve a label.

2. **`asset_metadata` DUPLICATES `backtest/specs.py`.** `point_value` here is
   `ContractSpec.multiplier` there, and `tick_size` is `tick_size`. A wrong
   multiplier silently scales every P&L figure for that symbol and the backtest
   still looks plausible — which is why this repository keeps exactly one copy
   of it. Two copies cannot be prevented once the schema declares them, so they
   are RECONCILED instead: `load_portfolio_config` compares every entry against
   `backtest.specs` and RAISES on a disagreement. They agree today, so the
   check costs nothing now and is the only thing that will catch the drift
   later. `strict_specs=False` turns it into a warning for a bench that has no
   spec table.

A THIRD THING, WHICH IS NOT A CONFLICT BUT WILL STOP THE NEXT TASK
==================================================================
**None of the four basket assets — MNQ, MES, MCL, MGC — exist in the data
lake.** `/mnt/backtest/lake/futures/bars/` holds 27 symbols and every one is a
full-size contract; the micros are absent. Their specs are present in
`backtest/specs.py` and CLAUDE.md lists them among the 17 symbols whose
definitions were never downloaded, so they are UNVERIFIED there as well.

Nothing in this file needs the lake and nothing here fails because of it. But
no strategy can be backtested, screened, swept or certified on these baskets
until the data is pulled, so a portfolio configured here cannot yet be filled
by the pipeline that is supposed to fill it. `assets_without_market_data` is on
the loaded config so the gap is visible to whatever runs next rather than being
discovered when a run returns no bars.

MUTATION
========
Every accessor returns a DEEP COPY. The config is cached per resolved path —
re-reading a small JSON file on every lookup would be silly — and a cached
mutable dict handed out by reference is a trap: one caller appending to
`active_strategies` would change what every other caller sees, including the
routing decision for a live account, with nothing raising and nothing written
to disk. Copying is cheap at this size and removes the whole class of bug.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# The same two lines `backtest/scan.py` carries, for the same reason: this
# module is importable as `portfolio.config_loader` from the repository root
# AND runnable as `python3 portfolio/config_loader.py`, and in the second case
# Python puts `portfolio/` on the path rather than the root — so the lazy
# `backtest.specs` import inside the reconciliation would fail with
# ModuleNotFoundError and the check that exists to catch a wrong contract
# multiplier would be skipped by the one entry point a human uses by hand.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG_PATH = "config/portfolios.json"

# The four accounts the architecture partitions into, and the tracks they
# belong to. Written down here rather than inferred from whatever the file
# happens to contain: a config missing `Prop-Even` is a routing table with a
# hole in it, and the hole would only surface when a strategy needed that
# account.
REQUIRED_PORTFOLIOS = ("Incubator-Odd", "Incubator-Even",
                       "Prop-Odd", "Prop-Even")

INCUBATOR_ACCOUNT_TYPE = "incubator_sim"
PROP_ACCOUNT_TYPE = "prop_eval"
ACCOUNT_TYPES = (INCUBATOR_ACCOUNT_TYPE, PROP_ACCOUNT_TYPE)

# The schema's regime labels mapped onto `mdlib.regimes` / `backtest.profiler`
# ids. FROM SCHEMA VERSION 1.1.0 THE DIGITS AGREE, and this table is what keeps
# them agreeing rather than a restatement of an agreement.
#
# They did not agree at 1.0.0: that schema numbered Q1 as Low-Vol/Trending and
# Q2 as High-Vol/Trending, so every one of the four digits named a different
# environment from the one `mdlib/regimes.py` gives it, and its Q2 was this
# repository's High-Volatility RANGING quadrant — the chop the strategy premise
# says to avoid. The labels were corrected in 1.1.0 rather than the mapping
# being left to translate them forever.
#
# The table is kept, and `_reconcile_quadrants` checks it against
# `backtest.profiler` on every load, BECAUSE the digits agree. An agreement
# nothing verifies is an agreement that lasts until someone edits one side:
# `canonical_quadrant` stays the only supported way to resolve a label, so a
# future relabelling is caught here instead of in a live account trading the
# one environment nobody certified.
#
# "MEAN_REVERSION" and "CHOP" both describe a RANGING market; the schema uses
# two words for one regime and the difference is one of tone, not of state.
CANONICAL_QUADRANT: dict[str, str] = {
    "Q1_HIGH_VOL_TREND": "Q1",             # High Volatility / Trending
    "Q2_HIGH_VOL_CHOP": "Q2",              # High Volatility / Ranging
    "Q3_LOW_VOL_TREND": "Q3",              # Low Volatility / Trending
    "Q4_LOW_VOL_MEAN_REVERSION": "Q4",     # Low Volatility / Ranging
}

# The regime NAME each schema label refers to, spelled exactly as
# `backtest.profiler.REGIMES` spells it. Carried beside the id because a
# quadrant code with no name beside it is the thing that made the numbering
# conflict possible in the first place.
CANONICAL_REGIME: dict[str, str] = {
    "Q1_HIGH_VOL_TREND": "High Volatility / Trending",
    "Q2_HIGH_VOL_CHOP": "High Volatility / Ranging",
    "Q3_LOW_VOL_TREND": "Low Volatility / Trending",
    "Q4_LOW_VOL_MEAN_REVERSION": "Low Volatility / Ranging",
}

# THE STRATEGY ALLOCATION RECORDS, written by `backtest/promote.py`.
#
# A portfolio grants PERMISSION through `active_strategies`, a flat list of
# strategy-id strings, and that is what `get_portfolio_for_strategy`,
# `realtime/live_dispatcher.py` and the Stage 5 card all read. The rich record
# behind each id - the certified contract, timeframe, version, quadrant and
# contract count - lives here instead, keyed by the same id, because putting a
# dict into `active_strategies` would break those three readers quietly rather
# than loudly.
#
# THIS BLOCK IS OPTIONAL AND DESCRIPTIVE. Nothing routes, sizes or gates an
# order from it: a strategy is permitted because `active_strategies` names it,
# and it is permitted in a quadrant because `basket.regime_quadrants` declares
# that quadrant for the ACCOUNT. An allocation record cannot widen either, and
# `_reconcile_allocations` reports where the two halves disagree rather than
# raising, for the same reason `assets_without_market_data` is reported: a
# config written before this key existed is a legitimate state, and so is one
# whose records lag a hand edit.
#
# `backtest/promote.py` carries this same literal. It cannot import it - the
# dependency runs one way and nothing in `backtest/` may import from
# `portfolio/` - so `tests/test_portfolio_config.py` asserts the two agree,
# which is what turns an unenforceable convention into a caught drift.
ALLOCATIONS_KEY = "strategy_allocations"

_CACHE: dict[str, dict] = {}


class PortfolioConfigError(RuntimeError):
    """The portfolio configuration cannot be loaded, or does not describe the
    four-account architecture it is supposed to."""


# --------------------------------------------------------------------------
# Quadrants
# --------------------------------------------------------------------------
def canonical_quadrant(label: str) -> str:
    """
    A schema regime label as this repository's `Q1`..`Q4` id.

    THE ONLY SUPPORTED WAY TO COMPARE A PORTFOLIO'S DECLARED REGIMES AGAINST A
    STAGE 1 DESIGNATION. Reading the digit off the label instead is wrong for
    all four values — see conflict 1 in the module docstring.

    Raises on an unknown label rather than passing it through. A label this
    mapping does not recognise is a regime nobody has decided the meaning of,
    and defaulting it to anything at all would hand a live supervisor a
    permission that was never granted.
    """
    try:
        return CANONICAL_QUADRANT[label]
    except KeyError:
        raise PortfolioConfigError(
            f"unknown regime label {label!r}. Known labels are "
            f"{sorted(CANONICAL_QUADRANT)} — and note that their digits do NOT "
            f"match this repository's quadrant ids, so a new label must be "
            f"added to CANONICAL_QUADRANT by MEANING rather than by number."
        ) from None


def canonical_regime(label: str) -> str:
    """The regime NAME a schema label refers to, as `profiler.REGIMES` spells it."""
    try:
        return CANONICAL_REGIME[label]
    except KeyError:
        raise PortfolioConfigError(
            f"unknown regime label {label!r}. Known labels are "
            f"{sorted(CANONICAL_REGIME)}.") from None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PortfolioConfigError(message)


def _validate_risk_profile(pid: str, profile: Any) -> None:
    """
    The risk envelope. Every value is a SPECIFICATION for CrossTrade except the
    budget and the clamps, which feed position sizing here — see the module
    docstring.
    """
    _require(isinstance(profile, dict) and profile,
             f"{pid}: risk_profile is missing or empty. A portfolio with no "
             f"risk profile would size positions from nothing.")
    for key in ("fixed_risk_budget_usd", "max_trailing_drawdown_usd",
                "max_forward_incubation_dd_pct"):
        _require(key in profile, f"{pid}: risk_profile has no {key!r}")
        value = profile[key]
        _require(isinstance(value, (int, float)) and not isinstance(value, bool),
                 f"{pid}: risk_profile.{key} must be a number; got {value!r}")
        _require(float(value) > 0,
                 f"{pid}: risk_profile.{key} must be > 0; got {value!r}. A "
                 f"zero risk budget sizes every position at zero contracts, "
                 f"which reads as a strategy that never traded.")
    pct = float(profile["max_forward_incubation_dd_pct"])
    _require(0.0 < pct <= 1.0,
             f"{pid}: max_forward_incubation_dd_pct is a FRACTION, not a "
             f"percent — 0.40 means 40%. Got {pct!r}, which would make the "
             f"allowable forward drawdown {pct:.0f}x the trailing limit.")

    clamp = profile.get("clamping")
    _require(isinstance(clamp, dict),
             f"{pid}: risk_profile.clamping is missing")
    for key in ("min_contracts", "max_contracts"):
        _require(isinstance(clamp.get(key), int)
                 and not isinstance(clamp.get(key), bool),
                 f"{pid}: clamping.{key} must be a whole number of contracts; "
                 f"got {clamp.get(key)!r}")
    _require(clamp["min_contracts"] >= 1,
             f"{pid}: clamping.min_contracts must be >= 1; got "
             f"{clamp['min_contracts']}. A floor of zero is not a clamp, it is "
             f"a strategy that is allowed to place no order and report no "
             f"error.")
    _require(clamp["min_contracts"] <= clamp["max_contracts"],
             f"{pid}: clamping.min_contracts ({clamp['min_contracts']}) is "
             f"above max_contracts ({clamp['max_contracts']}) — no contract "
             f"count satisfies both, so every order would be rejected or "
             f"silently coerced.")


def _validate_basket(pid: str, basket: Any, known_assets: set[str]) -> None:
    _require(isinstance(basket, dict) and basket,
             f"{pid}: basket is missing or empty")
    assets = basket.get("assets")
    _require(isinstance(assets, list) and assets,
             f"{pid}: basket.assets must be a non-empty list; got {assets!r}")
    _require(len(set(assets)) == len(assets),
             f"{pid}: basket.assets repeats a symbol ({assets!r}). A duplicate "
             f"would be sized twice and netted once.")
    for symbol in assets:
        _require(symbol in known_assets,
                 f"{pid}: basket asset {symbol!r} has no entry in "
                 f"asset_metadata, so nothing can size a position in it. "
                 f"Known: {sorted(known_assets)}")
    _require(isinstance(basket.get("correlation_group"), str)
             and basket["correlation_group"],
             f"{pid}: basket.correlation_group is missing")

    quadrants = basket.get("regime_quadrants")
    _require(isinstance(quadrants, list) and quadrants,
             f"{pid}: basket.regime_quadrants must be a non-empty list")
    for label in quadrants:
        canonical_quadrant(label)          # raises on an unknown label
    canon = [canonical_quadrant(q) for q in quadrants]
    _require(len(set(canon)) == len(canon),
             f"{pid}: two of {quadrants!r} resolve to the same canonical "
             f"quadrant ({canon!r}). The schema's digits are not this "
             f"repository's — see CANONICAL_QUADRANT.")

    structures = basket.get("structures")
    _require(isinstance(structures, list) and structures,
             f"{pid}: basket.structures must be a non-empty list")


def _validate_allocations(pid: str, allocations: Any) -> None:
    """
    Shape-check the optional `strategy_allocations` block.

    ABSENT IS FINE and is the state of every config written before
    `backtest/promote.py` started writing one. What is not fine is a block that
    is present and is not a mapping of strategy id to record: `active_strategies`
    would still grant the permission, so a malformed block changes no routing
    and would be discovered by whoever next tried to read a contract count out
    of it.

    The record's CONTENTS are deliberately not pinned here. They describe what
    was certified - a contract, a timeframe, a quadrant - and this module is
    not the authority on any of those; pinning them would make an older
    promotion's record unloadable the first time a field was added to it.
    """
    if allocations is None:
        return
    _require(isinstance(allocations, dict),
             f"{pid}: {ALLOCATIONS_KEY} must be an object keyed by strategy "
             f"id; got {type(allocations).__name__}. Permission still comes "
             f"from active_strategies, so this block being wrong changes no "
             f"routing - which is exactly why it has to be caught here.")
    for key, record in allocations.items():
        _require(isinstance(key, str) and key.strip(),
                 f"{pid}: {ALLOCATIONS_KEY} has an empty strategy id")
        _require(isinstance(record, dict),
                 f"{pid}: {ALLOCATIONS_KEY}[{key!r}] must be an object; got "
                 f"{type(record).__name__}")


def _reconcile_allocations(portfolios: dict) -> dict[str, list]:
    """
    Where the permission and the record disagree. REPORTED, NEVER RAISED.

    Three findings, kept apart because they are fixed by completely different
    work:

    `orphan_records`      an allocation record for a strategy the portfolio's
                          `active_strategies` does not name. The record
                          describes an allocation that grants nothing. This is
                          what a graduation that moved only half the pair
                          leaves behind - see `portfolio/promotion_daemon.py`,
                          which moves both.
    `unallocated`         an id in `active_strategies` with no record beside
                          it. NOT a defect: it is every strategy assigned by
                          hand, and every one assigned before this key existed.
                          Reported so "registered by promote.py" and "added by
                          an operator" are distinguishable.
    `regime_conflicts`    a record whose `regime_filter` is not among the
                          ACCOUNT's canonical quadrants. The live gate reads
                          `basket.regime_quadrants` and nothing else, so this
                          strategy is stood down in the one quadrant it was
                          certified for and will never trade. It is the finding
                          most worth seeing and the one least likely to
                          announce itself: every count in every table still
                          adds up, and the symptom is silence.

    None of the three is raised. An allocation record cannot route an order,
    size one, or grant a permission, so refusing to load the config over one
    would take the live loop down for a descriptive field - and the config the
    loader must keep loading is precisely the one somebody is midway through
    fixing.
    """
    out: dict[str, list] = {"orphan_records": [], "unallocated": [],
                            "regime_conflicts": []}
    for pid in sorted(portfolios):
        portfolio = portfolios[pid]
        active = [str(s) for s in (portfolio.get("active_strategies") or [])]
        records = portfolio.get(ALLOCATIONS_KEY) or {}
        permitted = list(portfolio.get("derived", {})
                         .get("canonical_quadrants") or [])
        for name in active:
            if name not in records:
                out["unallocated"].append({"portfolio_id": pid,
                                           "strategy_id": name})
        for name, record in sorted(records.items()):
            if name not in active:
                out["orphan_records"].append({
                    "portfolio_id": pid, "strategy_id": name,
                    "note": (f"{ALLOCATIONS_KEY} describes {name} but "
                             f"active_strategies does not name it, so the "
                             f"record grants nothing")})
            quadrant = str((record or {}).get("regime_filter") or "").strip()
            if quadrant and permitted and quadrant not in permitted:
                out["regime_conflicts"].append({
                    "portfolio_id": pid, "strategy_id": name,
                    "regime_filter": quadrant,
                    "portfolio_quadrants": permitted,
                    "note": (f"{name} is certified in {quadrant} and {pid} "
                             f"trades {permitted}. The live regime gate reads "
                             f"the portfolio's quadrants, so this strategy is "
                             f"stood down in the one quadrant it was "
                             f"certified for and will never trade.")})
    return out


def _validate_portfolio(pid: str, portfolio: Any,
                        known_assets: set[str]) -> None:
    _require(isinstance(portfolio, dict),
             f"{pid}: portfolio entry must be an object; got "
             f"{type(portfolio).__name__}")
    _require(portfolio.get("portfolio_id") == pid,
             f"{pid}: portfolio_id is {portfolio.get('portfolio_id')!r} but "
             f"the key is {pid!r}. The two are looked up separately and a "
             f"mismatch routes an order to the wrong account.")
    _require(portfolio.get("account_type") in ACCOUNT_TYPES,
             f"{pid}: account_type must be one of {list(ACCOUNT_TYPES)}; got "
             f"{portfolio.get('account_type')!r}")
    _require(isinstance(portfolio.get("target_account"), str)
             and portfolio["target_account"],
             f"{pid}: target_account is missing — nothing would know where to "
             f"send the order")
    size = portfolio.get("default_account_size")
    _require(isinstance(size, (int, float)) and not isinstance(size, bool)
             and size > 0,
             f"{pid}: default_account_size must be a positive number; got "
             f"{size!r}")
    _require(isinstance(portfolio.get("active_strategies"), list),
             f"{pid}: active_strategies must be a list (an empty one is "
             f"fine — it means nothing has been assigned yet)")
    _validate_allocations(pid, portfolio.get(ALLOCATIONS_KEY))
    _validate_risk_profile(pid, portfolio.get("risk_profile"))
    _validate_basket(pid, portfolio.get("basket"), known_assets)


def _validate_asset_metadata(metadata: Any) -> set[str]:
    _require(isinstance(metadata, dict) and metadata,
             "asset_metadata is missing or empty")
    for symbol, spec in metadata.items():
        _require(isinstance(spec, dict), f"asset_metadata.{symbol} must be an "
                                         f"object")
        for key in ("point_value", "tick_size"):
            value = spec.get(key)
            _require(isinstance(value, (int, float))
                     and not isinstance(value, bool) and value > 0,
                     f"asset_metadata.{symbol}.{key} must be a positive "
                     f"number; got {value!r}")
        _require(isinstance(spec.get("sector"), str) and spec["sector"],
                 f"asset_metadata.{symbol}.sector is missing")
    return set(metadata)


def _reconcile_with_specs(metadata: dict, strict: bool) -> list[dict]:
    """
    Check every `asset_metadata` entry against `backtest/specs.py`, the
    repository's single source of truth for a contract's multiplier and tick.

    THIS IS THE DUPLICATE-SOURCE GUARD, and it is worth more than it looks. A
    wrong multiplier does not raise anywhere: it silently scales every P&L
    figure for that symbol and the equity curve still looks plausible. The
    schema declares its own copy of both numbers, so the only defence available
    is to compare them and refuse the config when they disagree.

    Returns the reconciliation record. A symbol with no spec at all is reported
    rather than skipped — a basket asset the backtest engine cannot price is a
    portfolio that cannot be filled.
    """
    try:
        from backtest.specs import SPECS
    except Exception as exc:                                    # noqa: BLE001
        if strict:
            raise PortfolioConfigError(
                f"cannot import backtest.specs to reconcile asset_metadata "
                f"against it: {type(exc).__name__}: {exc}. Pass "
                f"strict_specs=False to load without the check, and read what "
                f"that costs in portfolio/config_loader.py.") from exc
        return [{"symbol": s, "status": "NOT CHECKED",
                 "detail": "backtest.specs could not be imported"}
                for s in sorted(metadata)]

    rows: list[dict] = []
    problems: list[str] = []
    for symbol in sorted(metadata):
        entry = metadata[symbol]
        spec = SPECS.get(symbol)
        if spec is None:
            rows.append({"symbol": symbol, "status": "NO SPEC",
                         "detail": "absent from backtest/specs.py"})
            problems.append(
                f"{symbol}: no ContractSpec in backtest/specs.py, so a "
                f"backtest cannot compute P&L for it")
            continue
        mismatches = []
        if float(entry["point_value"]) != float(spec.multiplier):
            mismatches.append(
                f"point_value {entry['point_value']} vs multiplier "
                f"{spec.multiplier}")
        if float(entry["tick_size"]) != float(spec.tick_size):
            mismatches.append(
                f"tick_size {entry['tick_size']} vs {spec.tick_size}")
        if mismatches:
            rows.append({"symbol": symbol, "status": "MISMATCH",
                         "detail": "; ".join(mismatches)})
            problems.append(f"{symbol}: {'; '.join(mismatches)}")
        else:
            rows.append({"symbol": symbol, "status": "OK",
                         "detail": f"multiplier {spec.multiplier}, tick "
                                   f"{spec.tick_size}"})

    if problems and strict:
        raise PortfolioConfigError(
            "config/portfolios.json disagrees with backtest/specs.py, which is "
            "this repository's single source of truth for a contract's "
            "multiplier and tick size:\n  "
            + "\n  ".join(problems)
            + "\nA wrong multiplier scales every P&L figure for that symbol "
              "and nothing raises. Fix the file that is wrong; do not silence "
              "the check.")
    return rows


def _reconcile_quadrants(strict: bool) -> list[dict]:
    """
    Check `CANONICAL_QUADRANT` and `CANONICAL_REGIME` against
    `backtest.profiler`, the repository's own encoding.

    THREE THINGS ARE CHECKED, and the third is the one that matters:

      * every regime NAME this module maps to is one of the four
        `profiler.REGIMES` spells — a typo here would make a portfolio's
        declared environment unmatchable against a Stage 1 designation, and it
        would read as a strategy that simply never traded its own quadrant;
      * every label resolves to the id `profiler` gives that regime;
      * the label's own DIGIT matches that id. From schema 1.1.0 the two
        encodings agree, so this holds — and it is checked precisely because it
        holds. The whole class of bug here is silent: a relabelled quadrant
        moves every routing decision between environments and nothing raises,
        because a quadrant id is a well-formed string whichever regime it names.

    `strict=False` records the disagreement instead of refusing, for a bench
    with no profiler. It is not for getting past a MISMATCH.
    """
    try:
        from backtest.profiler import REGIMES
    except Exception as exc:                                    # noqa: BLE001
        if strict:
            raise PortfolioConfigError(
                f"cannot import backtest.profiler to reconcile the regime "
                f"labels against it: {type(exc).__name__}: {exc}") from exc
        return [{"label": lab, "status": "NOT CHECKED",
                 "detail": "backtest.profiler could not be imported"}
                for lab in sorted(CANONICAL_QUADRANT)]

    # `REGIMES` is a tuple in quadrant order, so `Q{i+1}` IS the id. Derived
    # from the ordering rather than from a second constant, which is how
    # `backtest/profiler.py` builds its own inverted map.
    regime_to_quadrant = {name: f"Q{i + 1}" for i, name in enumerate(REGIMES)}

    rows: list[dict] = []
    problems: list[str] = []
    for label in sorted(CANONICAL_QUADRANT):
        quadrant = CANONICAL_QUADRANT[label]
        regime = CANONICAL_REGIME.get(label)
        detail = f"{quadrant} · {regime}"
        if regime not in regime_to_quadrant:
            rows.append({"label": label, "status": "UNKNOWN REGIME",
                         "detail": f"{regime!r} is not one of {list(REGIMES)}"})
            problems.append(f"{label}: {regime!r} is not a regime this "
                            f"repository defines")
            continue
        expected = regime_to_quadrant[regime]
        if quadrant != expected:
            rows.append({"label": label, "status": "MISMATCH",
                         "detail": f"maps to {quadrant}, {regime} is "
                                   f"{expected}"})
            problems.append(f"{label}: maps to {quadrant} but {regime} is "
                            f"{expected} in backtest.profiler")
            continue
        if not label.startswith(expected + "_"):
            rows.append({"label": label, "status": "DIGIT MISMATCH",
                         "detail": f"label digit vs {expected}"})
            problems.append(
                f"{label}: the label's own digit disagrees with {expected}. "
                f"Schema 1.1.0 relabelled these to agree with "
                f"mdlib/regimes.py; a label that drifts back is the bug that "
                f"moves every routing decision between environments silently.")
            continue
        rows.append({"label": label, "status": "OK", "detail": detail})

    if problems and strict:
        raise PortfolioConfigError(
            "the portfolio regime labels disagree with backtest.profiler, "
            "which is this repository's quadrant encoding:\n  "
            + "\n  ".join(problems)
            + "\nA quadrant id is a well-formed string whichever regime it "
              "names, so this cannot be caught downstream. Fix the label or "
              "the mapping; do not silence the check.")
    return rows


def _assets_without_market_data(symbols: set[str]) -> list[str]:
    """
    Which basket assets the lake holds no bars for.

    REPORTED, NEVER RAISED. Nothing in this module needs the lake, and a
    configuration written before a data pull is a legitimate state — but a
    portfolio whose assets cannot be backtested cannot be filled by the
    pipeline that is supposed to fill it, and that is worth knowing before a
    strategy search is started rather than after it returns no bars. All four
    micros were absent when this was written.

    An unmounted lake yields `[]` rather than "everything is missing": absence
    of the mount is not evidence about the data.
    """
    try:
        from mdlib.lake import LAKE
    except Exception:                                           # noqa: BLE001
        return []
    if not LAKE.exists():
        return []
    return sorted(s for s in symbols if not (LAKE / f"symbol={s}").exists())


# --------------------------------------------------------------------------
# The public API
# --------------------------------------------------------------------------
def load_portfolio_config(config_path: str = DEFAULT_CONFIG_PATH,
                          strict_specs: bool = True,
                          use_cache: bool = True) -> dict:
    """
    Read, validate and enrich `config/portfolios.json`.

    A relative `config_path` resolves against the REPOSITORY ROOT, not the
    working directory. Every command in this repository is documented as being
    run from the root, but a cron job, a Streamlit process and a test runner
    all have different ideas about `cwd`, and a routing table that silently
    fails to load in one of them is worse than one that fails everywhere.

    WHAT IS ADDED TO WHAT WAS READ. The file is returned unchanged except for a
    `derived` block per portfolio and a `reconciliation` block at the top
    level. Derived values are kept in their own block rather than merged into
    `risk_profile` so that "this number is in the file" and "this number was
    computed from the file" stay distinguishable — an operator editing the JSON
    to change a computed value would otherwise have no way to know it will be
    overwritten on the next load.

        derived.allowable_forward_dd_usd
            max_trailing_drawdown_usd x max_forward_incubation_dd_pct.
            $2,500 x 0.40 = $1,000 on every profile as shipped, which is 2% of
            a $50,000 account. IT IS A FRACTION OF THE TRAILING LIMIT, NOT OF
            THE ACCOUNT — those differ by more than a factor of two here, and
            reading it as an account percentage would put the forward
            incubation bar at $20,000.
        derived.canonical_quadrants
            The basket's regimes as this repository's ids. See conflict 1 in
            the module docstring: the schema's digits are not these digits.
        derived.canonical_regimes
            The same, by name, as `backtest.profiler.REGIMES` spells them.
        derived.risk_budget_pct_of_account
            fixed_risk_budget_usd / default_account_size, for a reader
            comparing two accounts of different sizes.

    `strict_specs=False` downgrades the `backtest/specs.py` reconciliation from
    a refusal to a record. Use it on a bench with no spec table; do not use it
    to get past a MISMATCH, which means one of the two files is wrong about a
    contract multiplier.

    Raises `PortfolioConfigError` for a missing file, malformed JSON, a missing
    portfolio, or any validation failure. It never returns a partial config: a
    routing table that loaded with three of its four accounts would send orders
    for the fourth nowhere, and the failure would surface as a strategy that
    quietly never traded.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    key = str(path.resolve()) + f"|strict={bool(strict_specs)}"
    if use_cache and key in _CACHE:
        return copy.deepcopy(_CACHE[key])

    if not path.exists():
        raise PortfolioConfigError(
            f"no portfolio configuration at {path}. Expected "
            f"{DEFAULT_CONFIG_PATH} relative to {REPO_ROOT}.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PortfolioConfigError(
            f"{path} is not valid JSON: {exc.msg} at line {exc.lineno} "
            f"column {exc.colno}") from exc
    _require(isinstance(raw, dict), f"{path}: the top level must be an object")

    config = copy.deepcopy(raw)
    _require(isinstance(config.get("version"), str) and config["version"],
             f"{path}: no top-level `version`. A routing table with no schema "
             f"version cannot be migrated safely.")
    _require(isinstance(config.get("base_currency"), str)
             and config["base_currency"],
             f"{path}: no top-level `base_currency`")

    known_assets = _validate_asset_metadata(config.get("asset_metadata"))

    portfolios = config.get("portfolios")
    _require(isinstance(portfolios, dict) and portfolios,
             f"{path}: `portfolios` is missing or empty")
    missing = [p for p in REQUIRED_PORTFOLIOS if p not in portfolios]
    _require(not missing,
             f"{path}: the four-account architecture requires "
             f"{list(REQUIRED_PORTFOLIOS)}; missing {missing}. A partition "
             f"with a hole in it routes some strategy nowhere.")

    accounts: dict[str, str] = {}
    for pid, portfolio in portfolios.items():
        _validate_portfolio(pid, portfolio, known_assets)
        account = portfolio["target_account"]
        _require(account not in accounts,
                 f"{pid} and {accounts.get(account)} both target the account "
                 f"{account!r}. `get_portfolio_by_account` would have to pick "
                 f"one, and the two baskets would net against each other on a "
                 f"live account.")
        accounts[account] = pid

        risk = portfolio["risk_profile"]
        basket = portfolio["basket"]
        portfolio["derived"] = {
            "allowable_forward_dd_usd": (
                float(risk["max_trailing_drawdown_usd"])
                * float(risk["max_forward_incubation_dd_pct"])),
            "allowable_forward_dd_basis": (
                "max_trailing_drawdown_usd x max_forward_incubation_dd_pct — a "
                "fraction of the TRAILING LIMIT, not of the account balance"),
            "canonical_quadrants": [canonical_quadrant(q)
                                    for q in basket["regime_quadrants"]],
            "canonical_regimes": [canonical_regime(q)
                                  for q in basket["regime_quadrants"]],
            "risk_budget_pct_of_account": (
                float(risk["fixed_risk_budget_usd"])
                / float(portfolio["default_account_size"])),
            "enforced_by": (
                "CrossTrade NAM — the drawdown figures here are a "
                "specification, not a gate this repository applies"),
        }

    # Two contracts with the same assets on different tracks is the intended
    # design (Incubator-Odd mirrors Prop-Odd), so overlapping baskets are NOT
    # an error. Overlapping WITHIN a track would be: two accounts on the same
    # track holding the same asset are one position split in two, and the
    # correlation diversification the partition exists for would be fictional.
    for account_type in ACCOUNT_TYPES:
        seen: dict[str, str] = {}
        for pid, portfolio in portfolios.items():
            if portfolio["account_type"] != account_type:
                continue
            for symbol in portfolio["basket"]["assets"]:
                if symbol in seen:
                    raise PortfolioConfigError(
                        f"{pid} and {seen[symbol]} are both {account_type} and "
                        f"both trade {symbol}. The tracks are meant to be "
                        f"orthogonal: the same asset on two accounts of one "
                        f"track is a single position split in two, and the "
                        f"diversification the partition exists for would be "
                        f"fictional.")
                seen[symbol] = pid

    config["reconciliation"] = {
        "specs": _reconcile_with_specs(config["asset_metadata"], strict_specs),
        "quadrants": _reconcile_quadrants(strict_specs),
        "strict": bool(strict_specs),
    }
    # Reported, never raised — see `_assets_without_market_data`.
    config["assets_without_market_data"] = _assets_without_market_data(
        known_assets)
    # Runs AFTER the `derived` blocks above, because the regime comparison is
    # against `derived.canonical_quadrants` — the schema labels resolved
    # through CANONICAL_QUADRANT — and never against the digit in the label,
    # which is wrong for all four values at schema 1.0.0.
    config["allocation_reconciliation"] = _reconcile_allocations(portfolios)
    config["config_path"] = str(path)

    if use_cache:
        _CACHE[key] = copy.deepcopy(config)
    return config


def clear_cache() -> None:
    """Drop the cached configs. For tests, and for a process that edited the file."""
    _CACHE.clear()


def _config(config: dict | None, config_path: str) -> dict:
    """The caller's config, or the cached one."""
    return config if config is not None else load_portfolio_config(config_path)


def get_portfolio_by_account(account_name: str,
                             config: dict | None = None,
                             config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    The portfolio routed to `account_name`, as a deep copy.

    Keyed on `target_account` rather than on the dict key. The two are equal in
    the file as shipped and `load_portfolio_config` requires the key to match
    `portfolio_id` — but `target_account` is the field that names the ACCOUNT,
    and it is the one that will change first when a broker account is renamed.
    Looking up by the wrong one would send orders to an account that no longer
    exists, or worse, to one that does.

    Raises on an unknown account. There is no default portfolio, deliberately:
    a fallback here would route an unrecognised account's orders to whichever
    portfolio happened to be first.
    """
    cfg = _config(config, config_path)
    for portfolio in cfg["portfolios"].values():
        if portfolio["target_account"] == account_name:
            return copy.deepcopy(portfolio)
    known = sorted(p["target_account"] for p in cfg["portfolios"].values())
    raise PortfolioConfigError(
        f"no portfolio targets the account {account_name!r}. Known accounts: "
        f"{known}")


def get_portfolio_for_strategy(strategy_id: str,
                               is_incubating: bool = True,
                               config: dict | None = None,
                               config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """
    The `portfolio_id` a strategy trades under, on the incubator track by
    default and the prop track when `is_incubating=False`.

    ASSIGNMENT IS EXPLICIT AND THERE IS NO FALLBACK. A strategy is routed to
    the portfolio whose `active_strategies` names it; if no portfolio on the
    requested track does, this RAISES. The alternative — deriving a portfolio
    from the strategy's name, its assets, or the odd/even parity of a hash —
    would be a live account chosen by a rule nobody wrote down, and the first
    time anyone noticed would be an order arriving on the wrong account.
    `active_strategies` is empty in the file as shipped, so this raises for
    every strategy until a human assigns one, and that is the intended state.

    A strategy named on TWO portfolios of one track raises as well. Both would
    size it independently against the same signal, and the resulting position
    would be double what either account's risk profile describes.

    The same strategy appearing on both an incubator and a prop portfolio is
    NORMAL and is what the two tracks are for — that is why the track is a
    parameter rather than a search across all four.
    """
    cfg = _config(config, config_path)
    wanted = INCUBATOR_ACCOUNT_TYPE if is_incubating else PROP_ACCOUNT_TYPE
    matches = [pid for pid, p in cfg["portfolios"].items()
               if p["account_type"] == wanted
               and strategy_id in (p.get("active_strategies") or [])]
    if len(matches) == 1:
        return matches[0]
    track = "incubator" if is_incubating else "prop"
    if not matches:
        candidates = sorted(pid for pid, p in cfg["portfolios"].items()
                            if p["account_type"] == wanted)
        raise PortfolioConfigError(
            f"{strategy_id!r} is not assigned to any {track} portfolio. Add it "
            f"to `active_strategies` on one of {candidates} in "
            f"{cfg.get('config_path', DEFAULT_CONFIG_PATH)}. There is no "
            f"default: a portfolio inferred from a strategy's name would be a "
            f"live account chosen by a rule nobody wrote down.")
    raise PortfolioConfigError(
        f"{strategy_id!r} is assigned to more than one {track} portfolio "
        f"({sorted(matches)}). Both would size it against the same signal and "
        f"the net position would be double what either risk profile describes.")


def get_asset_spec(symbol: str,
                   config: dict | None = None,
                   config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    `{"symbol", "point_value", "tick_size", "sector", "tick_value"}` for one
    contract.

    `tick_value` — dollars per tick, `point_value x tick_size` — is derived
    rather than declared, and derived HERE rather than left to each caller.
    It is the number a sizer actually multiplies a stop distance by, and a
    second implementation of one multiplication is still a second place for it
    to be wrong.

    Reads `asset_metadata`, which `load_portfolio_config` has already
    reconciled against `backtest/specs.py` — so these values are the engine's
    values, or the config did not load.
    """
    cfg = _config(config, config_path)
    metadata = cfg["asset_metadata"]
    if symbol not in metadata:
        raise PortfolioConfigError(
            f"no asset_metadata for {symbol!r}. Known: {sorted(metadata)}. A "
            f"contract with no point value cannot be sized.")
    entry = metadata[symbol]
    point_value = float(entry["point_value"])
    tick_size = float(entry["tick_size"])
    return {
        "symbol": symbol,
        "point_value": point_value,
        "tick_size": tick_size,
        "sector": entry["sector"],
        "tick_value": point_value * tick_size,
    }


def describe(config: dict | None = None,
             config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """A console summary: the four accounts, their baskets and their envelopes."""
    cfg = _config(config, config_path)
    lines = [f"portfolio configuration v{cfg['version']} "
             f"({cfg['base_currency']}) — {cfg.get('config_path', '')}"]
    for pid in REQUIRED_PORTFOLIOS:
        p = cfg["portfolios"][pid]
        d = p["derived"]
        lines.append(
            f"  {pid:<16} {p['account_type']:<14} "
            f"${p['default_account_size']:,} | "
            f"{'/'.join(p['basket']['assets']):<9} | "
            f"risk ${p['risk_profile']['fixed_risk_budget_usd']:,.0f} | "
            f"fwd DD ${d['allowable_forward_dd_usd']:,.0f} | "
            f"{'+'.join(d['canonical_quadrants'])} "
            f"({'+'.join(p['basket']['regime_quadrants'])})")
    missing = cfg.get("assets_without_market_data") or []
    if missing:
        lines.append(f"  NO MARKET DATA in the lake for: {', '.join(missing)} "
                     f"— these baskets cannot be backtested yet")
    return "\n".join(lines)


if __name__ == "__main__":                                  # pragma: no cover
    try:
        print(describe())
    except PortfolioConfigError as exc:
        print(f"portfolio configuration ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
