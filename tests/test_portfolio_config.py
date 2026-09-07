#!/usr/bin/env python3
"""
test_portfolio_config.py — the portfolio-to-account routing table: that all
four accounts are there and well formed, that a contract's point value agrees
with the engine's, that the derived drawdown arithmetic is what it claims, and
that the schema's regime digits are translated rather than trusted.

Location:  ~/src/trading/tests/test_portfolio_config.py

Run EITHER way — both report the same answer:

    OMP_NUM_THREADS=1 python tests/test_portfolio_config.py
    /home/cgrullon/src/trading/.venv/bin/pytest tests/test_portfolio_config.py

EVERY CASE FAILS THROUGH `assert`, DELIBERATELY. The older convention in this
directory (a `check(name, ok)` helper and `sys.exit(1)` in `main`) is invisible
to pytest, which collects those suites, watches their checks fail and reports
all green.

Nothing here needs the lake or a network.

WHAT THIS COVERS, and why each one is here rather than assumed:

  * **THE FOUR-ACCOUNT PARTITION IS COMPLETE AND WELL FORMED.** All four
    portfolios, each with a non-null risk profile, a non-empty asset list whose
    every symbol has metadata, and a target account no other portfolio claims.
    A routing table with a hole in it sends some strategy nowhere, and the hole
    only surfaces when that strategy needs the account.
  * **THE POINT VALUES ARE THE ENGINE'S.** `config/portfolios.json` declares
    its own copy of every contract's multiplier and tick, which
    `backtest/specs.py` also holds. A wrong multiplier does not raise: it
    silently scales every P&L figure for that symbol and the equity curve still
    looks plausible. The loader reconciles the two and the case checks both the
    agreement and the refusal.
  * **THE DERIVED DRAWDOWN IS A FRACTION OF THE TRAILING LIMIT, NOT OF THE
    ACCOUNT.** $2,500 x 0.40 = $1,000, which is 2% of the $50,000 account. The
    two readings differ by a factor of twenty and both produce a plausible
    number, so the case pins the basis as well as the value.
  * **THE SCHEMA'S QUADRANT DIGITS ARE NOT THIS REPOSITORY'S.** All four
    disagree with `mdlib/regimes.py`. The mapping is checked against
    `backtest.profiler` by MEANING, so a change to either encoding fails here
    rather than in a live account trading the one regime nobody certified.
  * **ROUTING HAS NO FALLBACK.** An unknown account, an unassigned strategy and
    a doubly-assigned strategy all raise. Every alternative is a live account
    picked by a rule nobody wrote down.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.profiler import REGIMES                             # noqa: E402
from mdlib.regimes import QUADRANT_LABELS                        # noqa: E402
from backtest.specs import SPECS                                 # noqa: E402
from portfolio.config_loader import (                            # noqa: E402
    ACCOUNT_TYPES, CANONICAL_QUADRANT, CANONICAL_REGIME,
    DEFAULT_CONFIG_PATH, EVAL_ACCOUNT_TYPE, INCUBATOR_ACCOUNT_TYPE,
    PROP_ACCOUNT_TYPE,
    PortfolioConfigError, REQUIRED_PORTFOLIOS, canonical_quadrant,
    canonical_regime, clear_cache, describe, get_asset_spec,
    get_portfolio_by_account, get_portfolio_for_strategy,
    load_portfolio_config)

CONFIG_PATH = REPO / DEFAULT_CONFIG_PATH

# The point values the request specifies, retyped rather than read from the
# file. This is the one place the suite compares the CONFIG against the
# SPECIFICATION instead of against itself; a case that read them out of the
# config would pass whatever they were changed to.
# The NinjaTrader account each portfolio EXECUTES on, retyped from the
# specification for the same reason. `target_account` is not the portfolio id:
# NT8 prefixes a simulation account with `Sim`, and the ids keep the Odd/Even
# spelling that encodes the basket split (Odd = MNQ/6E/6J, Even = MES/MGC).
# Read
# off the file instead, this case would pass whatever the accounts were
# renamed to — and an order sent to an account NinjaTrader does not have is
# rejected on a config that loads perfectly.
# The NT8 account each rung executes on. Two streams of three since
# 2026-09-03: the evaluation accounts sit between incubation and a funded
# book, and `promotion_daemon.PROMOTION_ROUTES` walks the same ladder.
EXECUTION_ACCOUNTS = {"Incubator-Odd": "SimIncubator1",
                      "Eval-Odd": "SimPropSim",
                      "Prop-Odd": "SimProp1",
                      "Incubator-Even": "SimIncubator2",
                      "Eval-Even": "Sim101",
                      "Prop-Even": "SimProp2"}

# The strategies the shipped routing table is EXPECTED to hold, named one by
# one. This list used to be "nothing, anywhere", which was true until
# `backtest/promote.py` registered its first promotion — but the check was
# never really about emptiness. It is about a strategy appearing in
# `active_strategies` that nobody put there ON PURPOSE, because that list is
# what routes an order to an account. Naming the expected ones keeps the guard
# and lets a deliberate promotion through: promote a strategy, add it here in
# the same commit, and an assignment that arrives any other way still fails.
# DECLARED IN THE SAME COMMIT THAT REGISTERS THEM, which is the rule this
# constant exists to enforce and which earned its keep on 2026-08-27: an
# `--auto-promote` pipeline run wrote NINE strategies into this table
# unattended, and this assertion is what caught them before a restart could
# arm any of them.
#
# EVERY ENTRY HERE MUST BE ABLE TO TRADE WHERE IT SITS. The dispatcher walks
# `basket.assets` and asks `trades_symbol` for each, so a strategy certified on
# a contract the basket cannot reach is refused on every asset and declines
# forever — visibly running, permanently inert. Incubator-Odd holds
# MNQ/6E/6J (NQ, 6E, 6J) and Incubator-Even holds MES/MGC (ES, GC); an ES
# strategy on the Odd account is the mistake `backtest/promote.py`'s own
# routing comment records having made before.
#
# NG and RB promotions are deliberately absent: `contract_alias.MICRO_TO_PARENT`
# has no micro for either, so NO incubator basket can reach them and moving
# them between accounts cannot help. They need full-size contracts, which is a
# different risk profile from these sim accounts.
#
# CL promotions are absent for a THIRD reason now, and the history matters if
# anyone is tempted to put them back. NG and RB cannot be reached by any
# basket. CL once could - Incubator-Odd held MCL and the routing worked - but
# the DATA was missing: NinjaTrader on this box streams no CL series, so
# /mnt/backtest/artifacts/nt8_bars carries 27 symbols and none of them is CL,
# and the regime daemon failed every cycle with "the feed returned no 15m
# bars". Five CL strategies sat here reading `no_regime_published` -
# allocated, certified, permanently inert.
#
# `724a92b portfolios: route the certified FX and Q3/Q4 packages; drop MCL`
# then took MCL out of the basket and its `asset_metadata` entry with it, and
# put 6E/6J in its place. So restoring CL is now TWO changes, not one: the
# feed, and then the basket. Confirm the feed with the spool rather than with
# this comment:
#
#     ls /mnt/backtest/artifacts/nt8_bars/ | grep '^CL'
EXPECTED_ASSIGNMENTS = {
    "Incubator-Odd":  [
        "sma_momentum_crossover_20260818_NQ_5m_VA",           #  5m, NQ
        "t3_braid_scalp_20260823_NQ_1h_VA",                   #  1h, NQ
        "t3_braid_scalp_20260823_NQ_1h_VB",                   #  1h, NQ
        "ema_crossover_20260821_NQ_1h_VA",                    #  1h, NQ
        "ema_crossover_20260821_NQ_1h_VB",                    #  1h, NQ
        "ema_crossover_20260821_NQ_15m_VA",                   # 15m, NQ
        "ma_anchoring_spread_20260820_NQ_15m_VA",             # 15m, NQ
    ],
    "Incubator-Even": [
        "sma_momentum_crossover_20260818_GC_1h_VA",           #  1h, GC
        "sma_momentum_crossover_20260818_ES_15m_VA",          # 15m, ES
        "sma_momentum_crossover_20260818_GC_5m_VA",           #  5m, GC
        "sma_momentum_crossover_20260818_GC_1h_VB",           #  1h, GC
        "sma_momentum_crossover_20260818_ES_15m_VB",          # 15m, ES
        "t3_braid_scalp_20260823_GC_30m_VA",                  # 30m, GC
        "t3_braid_scalp_20260823_GC_30m_VB",                  # 30m, GC
        "ema_crossover_20260821_ES_30m_VA",                   # 30m, ES
        "ema_crossover_20260821_ES_15m_VA",                   # 15m, ES
        "ma_anchoring_spread_20260820_GC_1h_VA",              #  1h, GC
        "ma_anchoring_spread_20260820_GC_1h_VB",              #  1h, GC
    ],
    "Eval-Odd":       [],
    "Eval-Even":      [],
    "Prop-Odd":       [],
    "Prop-Even":      [],
}

REQUESTED_POINT_VALUES = {"MNQ": 2.0, "MES": 5.0, "MGC": 10.0,
                          "6E": 125_000.0, "6J": 12_500_000.0}
REQUESTED_TICK_SIZES = {"MNQ": 0.25, "MES": 0.25, "MGC": 0.10,
                        "6E": 0.00005, "6J": 0.0000005}

# The repository's regime -> quadrant id map, built HERE from
# `backtest.profiler.REGIMES` rather than imported. `REGIMES` is a tuple in
# quadrant order, so `Q{i+1}` IS the id — which is exactly how
# `backtest/profiler.py` builds its own inverted map, and deriving it from the
# ordering means this suite depends on the one constant rather than on a second
# one that could be renamed.
REGIME_TO_QUADRANT = {name: f"Q{i + 1}" for i, name in enumerate(REGIMES)}


def config() -> dict:
    """The real configuration, freshly loaded."""
    return load_portfolio_config()


def raises(fn, *args, **kwargs) -> str:
    """Run `fn` and return the PortfolioConfigError message; assert if it did not."""
    try:
        fn(*args, **kwargs)
    except PortfolioConfigError as exc:
        return str(exc)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise")


def write_temp_config(mutate) -> Path:
    """
    The real config with one thing changed, written to a temp file.

    A real file rather than an injected dict, because these cases exercise
    `load_portfolio_config`'s own validation — the half that runs before any
    caller has a config object to inject.
    """
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = Path(tempfile.mkdtemp(prefix="portfolio_cfg_")) / "portfolios.json"
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return path


# ==========================================================================
# 1. Schema validation — the request's first testing clause
# ==========================================================================
def test_all_six_target_accounts_exist_and_are_well_formed() -> None:
    """
    THE REQUEST'S FIRST CLAUSE: every target account, valid asset lists,
    non-null risk profiles.

    Four accounts until `48d1091 config: the three-stage ladder, and an
    evaluation tier to carry it` put an evaluation rung between incubation and
    the funded book; six since.

    The names are retyped from the specification rather than read off the
    file, so a portfolio quietly renamed fails here instead of at the first
    order that cannot be routed.
    """
    cfg = config()
    assert set(REQUIRED_PORTFOLIOS) == {"Incubator-Odd", "Incubator-Even",
                                        "Eval-Odd", "Eval-Even",
                                        "Prop-Odd", "Prop-Even"}
    assert set(cfg["portfolios"]) == set(REQUIRED_PORTFOLIOS), (
        sorted(cfg["portfolios"]))
    assert cfg["version"] == "1.1.0"
    assert cfg["base_currency"] == "USD"

    for pid in REQUIRED_PORTFOLIOS:
        p = cfg["portfolios"][pid]
        assert p["portfolio_id"] == pid, p["portfolio_id"]
        assert p["target_account"] == EXECUTION_ACCOUNTS[pid], (
            p["target_account"])
        assert p["account_type"] in ACCOUNT_TYPES, p["account_type"]
        assert p["default_account_size"] == 50000, p["default_account_size"]

        risk = p["risk_profile"]
        assert risk, f"{pid}: risk_profile is empty"
        assert risk["fixed_risk_budget_usd"] == 250.0
        assert risk["max_trailing_drawdown_usd"] == 2500.0
        assert risk["max_forward_incubation_dd_pct"] == 0.40
        # The evaluation rung is capped at ONE contract. Retyped per tier
        # rather than relaxed to a range: a clamp that silently widened from
        # 1 to 5 on an evaluation account is a funded-challenge breach, and a
        # test asserting `<= 5` would pass through it.
        want_max = 1 if p["account_type"] == EVAL_ACCOUNT_TYPE else 5
        assert risk["clamping"] == {"min_contracts": 1,
                                    "max_contracts": want_max}, (pid, risk)

        assets = p["basket"]["assets"]
        assert isinstance(assets, list) and assets, f"{pid}: {assets!r}"
        assert len(set(assets)) == len(assets), f"{pid}: duplicate asset"
        for symbol in assets:
            assert symbol in cfg["asset_metadata"], (
                f"{pid}: {symbol} has no metadata, so nothing can size it")
        assert p["basket"]["correlation_group"]
        assert p["basket"]["structures"]
        assert p["active_strategies"] == EXPECTED_ASSIGNMENTS[pid], (
            f"{pid} holds {p['active_strategies']}, and the deliberate "
            f"assignments are {EXPECTED_ASSIGNMENTS[pid]}. A strategy "
            f"appearing here unannounced is routed to a live account; one "
            f"promoted on purpose belongs in EXPECTED_ASSIGNMENTS in the same "
            f"commit that registers it.")


def test_the_two_tracks_hold_the_same_baskets_and_are_orthogonal_within() -> None:
    """
    The partition's actual shape: Odd trades MNQ/6E/6J and Even trades
    MES/MGC, and the incubation and evaluation rungs carry the SAME basket —
    that is what makes paper validation transferable up the ladder. The funded
    rung may carry less (Prop-Odd is MNQ alone), never more.

    Within one track the baskets must not overlap. Two accounts of the same
    track holding the same asset are one position split in two, and the
    correlation diversification the partition exists for would be fictional.
    """
    cfg = config()
    for track in ACCOUNT_TYPES:
        holders: dict[str, str] = {}
        for pid, p in cfg["portfolios"].items():
            if p["account_type"] != track:
                continue
            for symbol in p["basket"]["assets"]:
                assert symbol not in holders, (
                    f"{pid} and {holders.get(symbol)} are both {track} and "
                    f"both trade {symbol}")
                holders[symbol] = pid

    odd = {pid: cfg["portfolios"][pid]["basket"]["assets"]
           for pid in ("Incubator-Odd", "Eval-Odd", "Prop-Odd")}
    assert odd["Incubator-Odd"] == odd["Eval-Odd"] == ["MNQ", "6E", "6J"]
    # Prop-Odd is deliberately NARROWER than the two rungs below it: the FX
    # pair is carried through incubation and evaluation but is not on the
    # funded book. Asserted as a subset as well as by value, so widening the
    # funded basket past what was paper-validated fails here.
    assert odd["Prop-Odd"] == ["MNQ"]
    assert set(odd["Prop-Odd"]) <= set(odd["Incubator-Odd"])
    even = {pid: cfg["portfolios"][pid]["basket"]["assets"]
            for pid in ("Incubator-Even", "Eval-Even", "Prop-Even")}
    assert even["Incubator-Even"] == even["Eval-Even"] == even["Prop-Even"] \
        == ["MES", "MGC"]
    assert cfg["portfolios"]["Incubator-Odd"]["account_type"] \
        == INCUBATOR_ACCOUNT_TYPE
    assert cfg["portfolios"]["Prop-Odd"]["account_type"] == PROP_ACCOUNT_TYPE


def test_a_config_missing_a_portfolio_is_refused() -> None:
    """
    A routing table with a hole in it routes some strategy nowhere, and the
    failure surfaces as a strategy that quietly never traded rather than as an
    error. Refused at load, naming what is missing.
    """
    def drop(raw):
        del raw["portfolios"]["Prop-Even"]

    msg = raises(load_portfolio_config, str(write_temp_config(drop)),
                 use_cache=False)
    assert "Prop-Even" in msg and "four-account" in msg, msg


def test_a_malformed_risk_profile_is_refused_with_the_reason() -> None:
    """
    Each of these produces a plausible number rather than an error if it is let
    through: a zero budget sizes every position at zero contracts; a clamp
    floor of zero permits an order of no contracts; an inverted clamp is
    satisfied by no contract count at all; and a percentage written as `40`
    instead of `0.40` makes the allowable forward drawdown forty times the
    trailing limit.
    """
    def zero_budget(raw):
        raw["portfolios"]["Prop-Odd"]["risk_profile"][
            "fixed_risk_budget_usd"] = 0.0
    assert "must be > 0" in raises(
        load_portfolio_config, str(write_temp_config(zero_budget)),
        use_cache=False)

    def zero_floor(raw):
        raw["portfolios"]["Prop-Odd"]["risk_profile"]["clamping"][
            "min_contracts"] = 0
    assert "must be >= 1" in raises(
        load_portfolio_config, str(write_temp_config(zero_floor)),
        use_cache=False)

    def inverted(raw):
        raw["portfolios"]["Prop-Odd"]["risk_profile"]["clamping"][
            "max_contracts"] = 0
    assert "above max_contracts" in raises(
        load_portfolio_config, str(write_temp_config(inverted)),
        use_cache=False)

    def percent_not_fraction(raw):
        raw["portfolios"]["Prop-Odd"]["risk_profile"][
            "max_forward_incubation_dd_pct"] = 40.0
    msg = raises(load_portfolio_config,
                 str(write_temp_config(percent_not_fraction)), use_cache=False)
    assert "FRACTION" in msg, msg

    def no_profile(raw):
        raw["portfolios"]["Prop-Odd"]["risk_profile"] = {}
    assert "risk_profile is missing or empty" in raises(
        load_portfolio_config, str(write_temp_config(no_profile)),
        use_cache=False)


def test_an_asset_with_no_metadata_is_refused() -> None:
    """A basket symbol with no point value cannot be sized, so it cannot ship."""
    def unknown_asset(raw):
        raw["portfolios"]["Prop-Odd"]["basket"]["assets"] = ["MNQ", "M6E"]
    msg = raises(load_portfolio_config, str(write_temp_config(unknown_asset)),
                 use_cache=False)
    assert "M6E" in msg and "asset_metadata" in msg, msg


def test_two_portfolios_cannot_claim_one_account() -> None:
    """
    `get_portfolio_by_account` would have to pick one, and on a live account
    the two baskets would net against each other.
    """
    def collide(raw):
        raw["portfolios"]["Prop-Even"]["target_account"] = \
            EXECUTION_ACCOUNTS["Prop-Odd"]
    msg = raises(load_portfolio_config, str(write_temp_config(collide)),
                 use_cache=False)
    assert "target the account" in msg, msg


def test_a_missing_or_malformed_file_is_refused_with_a_usable_message() -> None:
    """
    A routing table that fails to load has to say which file and why. `cwd` is
    the usual cause — a relative path resolves against the REPOSITORY ROOT
    here, so a cron job and a test runner read the same file.
    """
    msg = raises(load_portfolio_config, "config/does_not_exist.json",
                 use_cache=False)
    assert "no portfolio configuration at" in msg and str(REPO) in msg, msg

    broken = Path(tempfile.mkdtemp(prefix="portfolio_bad_")) / "p.json"
    broken.write_text('{"version": "1.0.0",,}', encoding="utf-8")
    msg = raises(load_portfolio_config, str(broken), use_cache=False)
    assert "not valid JSON" in msg and "line" in msg, msg

    # A relative path is resolved against the repo root, not the cwd.
    assert load_portfolio_config(DEFAULT_CONFIG_PATH)["config_path"] \
        == str(CONFIG_PATH)


# ==========================================================================
# 2. Asset metadata — the request's second testing clause
# ==========================================================================
def test_the_point_values_are_the_ones_specified() -> None:
    """
    THE REQUEST'S SECOND CLAUSE: MNQ $2.00, MES $5.00, MGC $10.00, and the
    FX pair 6E $125,000 / 6J $12,500,000 that replaced MCL in `724a92b` —
    checked through `get_asset_spec` and against values retyped from the
    specification.
    """
    cfg = config()
    for symbol, want in REQUESTED_POINT_VALUES.items():
        spec = get_asset_spec(symbol, config=cfg)
        assert spec["point_value"] == want, (symbol, spec["point_value"])
        assert spec["tick_size"] == REQUESTED_TICK_SIZES[symbol], symbol
        assert spec["symbol"] == symbol
        assert spec["sector"], symbol

    assert get_asset_spec("MNQ", config=cfg)["sector"] == "Equity_Index"
    assert get_asset_spec("MES", config=cfg)["sector"] == "Equity_Index"
    assert get_asset_spec("MGC", config=cfg)["sector"] == "Metals"
    assert get_asset_spec("6E", config=cfg)["sector"] == "FX"
    assert get_asset_spec("6J", config=cfg)["sector"] == "FX"
    assert "no asset_metadata" in raises(get_asset_spec, "MCL", config=cfg), (
        "MCL left the baskets in 724a92b and its metadata went with it")

    assert "no asset_metadata" in raises(get_asset_spec, "NQ", config=cfg), (
        "the full-size contracts are not in this basket and must not resolve")


def test_the_tick_value_is_derived_once_and_is_right() -> None:
    """
    Dollars per tick is `point_value x tick_size` and is the number a sizer
    multiplies a stop distance by. Derived in the loader rather than by each
    caller: one multiplication is still one place to get it wrong, and two
    copies would be two.

    The five values are hand-worked here rather than recomputed from the same
    two fields, which would make the case a restatement of the formula. 6E and
    6J both land on $6.25 a tick from very different multipliers, which is the
    pair most worth writing out by hand.
    """
    cfg = config()
    hand_worked = {"MNQ": 0.50, "MES": 1.25, "MGC": 1.00,
                   "6E": 6.25, "6J": 6.25}
    for symbol, want in hand_worked.items():
        got = get_asset_spec(symbol, config=cfg)["tick_value"]
        assert abs(got - want) < 1e-9, (symbol, got, want)


def test_the_point_values_agree_with_the_engines_own_specs() -> None:
    """
    `config/portfolios.json` declares a SECOND copy of every contract's
    multiplier and tick; `backtest/specs.py` holds the first. A wrong
    multiplier silently scales every P&L figure for that symbol and the
    backtest still looks plausible — which is why the repository keeps one
    copy, and why the loader reconciles the two rather than trusting them.
    """
    cfg = config()
    for symbol, entry in cfg["asset_metadata"].items():
        spec = SPECS[symbol]
        assert float(entry["point_value"]) == float(spec.multiplier), (
            f"{symbol}: config says {entry['point_value']}, "
            f"backtest/specs.py says {spec.multiplier}")
        assert float(entry["tick_size"]) == float(spec.tick_size), symbol
        assert abs(get_asset_spec(symbol, config=cfg)["tick_value"]
                   - spec.tick_value) < 1e-9, symbol

    rows = {r["symbol"]: r["status"] for r in cfg["reconciliation"]["specs"]}
    assert set(rows) == set(REQUESTED_POINT_VALUES), sorted(rows)
    assert all(status == "OK" for status in rows.values()), rows
    assert cfg["reconciliation"]["strict"] is True


def test_a_point_value_that_disagrees_with_the_engine_is_refused() -> None:
    """
    The guard, exercised. Without it the two files drift and the first symptom
    is a position sized on one multiplier against P&L computed on another.
    """
    def wrong_multiplier(raw):
        raw["asset_metadata"]["MNQ"]["point_value"] = 20.0

    path = str(write_temp_config(wrong_multiplier))
    msg = raises(load_portfolio_config, path, use_cache=False)
    assert "backtest/specs.py" in msg and "MNQ" in msg, msg
    assert "20.0" in msg and "multiplier 2" in msg, msg

    # `strict_specs=False` records the disagreement instead of refusing. It is
    # for a bench with no spec table, not for getting past a MISMATCH.
    loose = load_portfolio_config(path, strict_specs=False, use_cache=False)
    rows = {r["symbol"]: r["status"] for r in loose["reconciliation"]["specs"]}
    assert rows["MNQ"] == "MISMATCH", rows
    assert loose["reconciliation"]["strict"] is False


# ==========================================================================
# 3. The derived drawdown — the request's third testing clause
# ==========================================================================
def test_the_dynamic_forward_drawdown_is_one_thousand_dollars() -> None:
    """
    THE REQUEST'S THIRD CLAUSE: a $50,000 account with a $2,500 trailing
    drawdown and a 0.40 forward fraction gives an allowable forward drawdown of
    $1,000.

    THE BASIS IS PINNED ALONGSIDE THE VALUE, because the wrong reading also
    produces a plausible number. $1,000 is 40% of the TRAILING LIMIT; read as
    40% of the ACCOUNT it would be $20,000, twenty times larger and still a
    perfectly ordinary-looking figure on a report.
    """
    cfg = config()
    for pid in REQUIRED_PORTFOLIOS:
        p = cfg["portfolios"][pid]
        risk, derived = p["risk_profile"], p["derived"]

        assert p["default_account_size"] == 50000
        assert risk["max_trailing_drawdown_usd"] == 2500.0
        assert risk["max_forward_incubation_dd_pct"] == 0.40

        got = derived["allowable_forward_dd_usd"]
        assert got == 1000.0, f"{pid}: allowable forward DD is {got}"
        assert got == (risk["max_trailing_drawdown_usd"]
                       * risk["max_forward_incubation_dd_pct"])
        # The wrong basis, named so the case fails if the formula is switched.
        assert got != p["default_account_size"] * risk[
            "max_forward_incubation_dd_pct"], (
            "the forward drawdown was computed off the ACCOUNT SIZE, not the "
            "trailing limit — $20,000 instead of $1,000")
        assert "TRAILING LIMIT" in derived["allowable_forward_dd_basis"]

        assert derived["risk_budget_pct_of_account"] == 250.0 / 50000
        assert "CrossTrade" in derived["enforced_by"], (
            "the drawdown figures are a specification for the execution "
            "bridge, not a gate this repository applies")


def test_the_derived_block_tracks_the_inputs_it_was_computed_from() -> None:
    """
    Derived values live in their own block rather than merged into
    `risk_profile`, so "this number is in the file" and "this number was
    computed from the file" stay distinguishable — an operator editing a
    computed value in the JSON would otherwise have no way to know it is
    overwritten on every load.

    Checked by moving the inputs and requiring the output to follow.
    """
    def halve(raw):
        for pid in REQUIRED_PORTFOLIOS:
            raw["portfolios"][pid]["risk_profile"][
                "max_forward_incubation_dd_pct"] = 0.20

    cfg = load_portfolio_config(str(write_temp_config(halve)), use_cache=False)
    for pid in REQUIRED_PORTFOLIOS:
        assert cfg["portfolios"][pid]["derived"][
            "allowable_forward_dd_usd"] == 500.0

    def bigger(raw):
        raw["portfolios"]["Prop-Odd"]["risk_profile"][
            "max_trailing_drawdown_usd"] = 5000.0
    cfg = load_portfolio_config(str(write_temp_config(bigger)),
                                use_cache=False)
    assert cfg["portfolios"]["Prop-Odd"]["derived"][
        "allowable_forward_dd_usd"] == 2000.0


# ==========================================================================
# 4. The quadrant conflict — not in the request, and the reason it is here
# ==========================================================================
def test_the_schema_quadrant_digits_are_not_this_repositorys() -> None:
    """
    THE CONFLICT THIS CONFIG CARRIED AT 1.0.0, AND THE GUARD THAT KEEPS IT
    FIXED.

    Schema 1.0.0 labelled its regimes `Q1_LOW_VOL_TREND`, `Q2_HIGH_VOL_TREND`,
    `Q3_LOW_VOL_MEAN_REVERSION`, `Q4_HIGH_VOL_CHOP`, while `mdlib/regimes.py` —
    the single place the encoding is written down, and the one Stage 1's
    designation and Gate R both read — numbers them High/Trending,
    High/Ranging, Low/Trending, Low/Ranging. EVERY DIGIT DISAGREED, and its
    `Q2` was this repository's High-Volatility RANGING quadrant, the chop the
    premise says to avoid.

    1.1.0 relabels them to agree. This case asserts the agreement AND the
    digit, because the failure mode is silent either way: a quadrant id is a
    well-formed string whichever regime it names, so a label that drifts back
    moves every routing decision between environments with nothing raising.
    """
    expected = {
        "Q1_HIGH_VOL_TREND": "High Volatility / Trending",
        "Q2_HIGH_VOL_CHOP": "High Volatility / Ranging",
        "Q3_LOW_VOL_TREND": "Low Volatility / Trending",
        "Q4_LOW_VOL_MEAN_REVERSION": "Low Volatility / Ranging",
    }
    assert set(CANONICAL_QUADRANT) == set(expected), sorted(CANONICAL_QUADRANT)
    for label, regime in expected.items():
        assert regime in REGIMES, regime
        assert canonical_regime(label) == regime, label
        assert CANONICAL_REGIME[label] == regime, label
        assert canonical_quadrant(label) == REGIME_TO_QUADRANT[regime], (
            f"{label} maps to {canonical_quadrant(label)} but {regime} is "
            f"{REGIME_TO_QUADRANT[regime]} in this repository")

    # From 1.1.0 the digits agree, and the check runs precisely BECAUSE they
    # do: an agreement nothing verifies lasts until someone edits one side.
    for label in expected:
        assert canonical_quadrant(label) == f"Q{label[1]}", (
            f"{label} resolves to {canonical_quadrant(label)}, which is not "
            f"its own digit — the schema and mdlib/regimes.py have drifted "
            f"apart again")
    assert canonical_quadrant("Q1_HIGH_VOL_TREND") == "Q1"
    assert canonical_quadrant("Q3_LOW_VOL_TREND") == "Q3"

    # And the loader refuses a config whose labels disagree with the profiler,
    # rather than translating them forever.
    rows = {r["label"]: r["status"]
            for r in config()["reconciliation"]["quadrants"]}
    assert set(rows) == set(expected), sorted(rows)
    assert all(v == "OK" for v in rows.values()), rows

    assert "unknown regime label" in raises(canonical_quadrant, "Q5_SIDEWAYS")
    assert "unknown regime label" in raises(canonical_regime, "Q5_SIDEWAYS")


def test_each_portfolio_carries_its_regimes_in_both_encodings() -> None:
    """
    Both spellings travel on the loaded config so a consumer never has to
    choose which one a bare `Q1` meant.

    The Odd basket declares ALL FOUR from 2026-08-24 and the Even basket from
    2026-09-02: the partition originally split assets and quadrants on the same
    axis, so MNQ - which sits only in the Odd basket - could never be traded by
    a strategy certified in a high-volatility quadrant, and three promotions of
    `t3_braid_scalp_20260823` were routable to no account at all. Widening the
    Even track closed the mirror of that hole for Q3 on ES and GC. The regime
    diversification the partition once claimed is therefore gone, and the live
    gate reads this list and nothing else.

    WHAT IS ASSERTED IS THE AGREEMENT BETWEEN THE TWO ENCODINGS, not which
    quadrants a portfolio happens to declare. A literal list here is a config
    state, and it failed on an operator's config edit rather than on a code
    change - which is the whole reason the count below is not pinned either.
    """
    cfg = config()
    for pid in ("Incubator-Odd", "Incubator-Even"):
        portfolio = cfg["portfolios"][pid]
        derived = portfolio["derived"]
        declared = portfolio["basket"]["regime_quadrants"]
        # The declared labels and the canonical ids are the same statement.
        assert derived["canonical_quadrants"] == [q[:2] for q in declared], pid
        assert len(derived["canonical_regimes"]) == len(declared), pid
        for code, regime in zip(derived["canonical_quadrants"],
                                derived["canonical_regimes"]):
            assert QUADRANT_LABELS[int(code[1:])] == regime, (pid, code, regime)
    for pid in REQUIRED_PORTFOLIOS:
        p = cfg["portfolios"][pid]
        assert len(p["derived"]["canonical_quadrants"]) == \
            len(p["basket"]["regime_quadrants"])
        # No duplicates, and never empty. The COUNT is deliberately not pinned:
        # the Odd track declares all four since 2026-08-24 and the Even track
        # two, so a fixed number here would be asserting the deadlock back.
        canon = p["derived"]["canonical_quadrants"]
        assert canon and len(set(canon)) == len(canon), pid

    def unknown(raw):
        raw["portfolios"]["Prop-Odd"]["basket"]["regime_quadrants"] = [
            "Q9_UNKNOWN"]
    assert "unknown regime label" in raises(
        load_portfolio_config, str(write_temp_config(unknown)),
        use_cache=False)


# ==========================================================================
# 5. Routing — the helpers, and the fallbacks they deliberately lack
# ==========================================================================
def test_an_account_resolves_to_its_portfolio_and_nothing_else_does() -> None:
    """
    Looked up by `target_account`, which is the field that NAMES the account
    and the one that changes first when a broker account is renamed. There is
    no default portfolio: a fallback would route an unrecognised account's
    orders to whichever portfolio happened to be first.
    """
    cfg = config()
    for pid in REQUIRED_PORTFOLIOS:
        got = get_portfolio_by_account(EXECUTION_ACCOUNTS[pid], config=cfg)
        assert got["portfolio_id"] == pid
        assert got["target_account"] == EXECUTION_ACCOUNTS[pid]
    # The PORTFOLIO ID is not an account and must not resolve as one. The two
    # were the same string until the NT8 accounts were renamed, which is
    # exactly when a lookup that quietly accepted either would stop being
    # tested and start being a guess.
    assert raises(get_portfolio_by_account, "Incubator-Odd", config=cfg)
    assert get_portfolio_by_account("SimProp2", config=cfg)["basket"][
        "assets"] == ["MES", "MGC"]

    msg = raises(get_portfolio_by_account, "Prop-Sideways", config=cfg)
    assert "no portfolio targets" in msg and "Known accounts" in msg, msg


def test_an_unassigned_strategy_raises_rather_than_being_routed() -> None:
    """
    `active_strategies` is empty in the file as shipped, so every strategy
    raises until a human assigns one — and that is the intended state.

    THE ALTERNATIVE IS THE POINT. Deriving a portfolio from the strategy's
    name, its assets, or the odd/even parity of a hash would be a live account
    chosen by a rule nobody wrote down, and the first time anyone noticed would
    be an order arriving on the wrong account.
    """
    cfg = config()
    msg = raises(get_portfolio_for_strategy, "double_rsi_macd_scalp",
                 config=cfg)
    assert "not assigned to any incubator portfolio" in msg, msg
    assert "active_strategies" in msg and "no default" in msg, msg

    msg = raises(get_portfolio_for_strategy, "double_rsi_macd_scalp",
                 is_incubating=False, config=cfg)
    assert "not assigned to any evaluation or prop portfolio" in msg, msg
    assert "Prop-Even" in msg and "Prop-Odd" in msg, msg


def test_an_assigned_strategy_routes_by_track() -> None:
    """
    The same strategy on both an incubator and a prop portfolio is NORMAL and
    is what the two tracks are for — which is why the track is a parameter
    rather than a search across all four accounts.
    """
    cfg = config()
    cfg["portfolios"]["Incubator-Odd"]["active_strategies"].append("strat_a")
    cfg["portfolios"]["Prop-Odd"]["active_strategies"].append("strat_a")
    cfg["portfolios"]["Incubator-Even"]["active_strategies"].append("strat_b")

    assert get_portfolio_for_strategy("strat_a", config=cfg) == "Incubator-Odd"
    assert get_portfolio_for_strategy("strat_a", is_incubating=False,
                                      config=cfg) == "Prop-Odd"
    assert get_portfolio_for_strategy("strat_b", config=cfg) == "Incubator-Even"
    # `strat_b` was never promoted to the prop track.
    assert "not assigned to any evaluation or prop portfolio" in raises(
        get_portfolio_for_strategy, "strat_b", is_incubating=False, config=cfg)


def test_a_strategy_on_two_portfolios_of_one_track_raises() -> None:
    """
    Both would size it against the same signal and the net position would be
    double what either risk profile describes — on two accounts whose baskets
    are supposed to be orthogonal.
    """
    cfg = config()
    cfg["portfolios"]["Incubator-Odd"]["active_strategies"].append("strat_c")
    cfg["portfolios"]["Incubator-Even"]["active_strategies"].append("strat_c")
    msg = raises(get_portfolio_for_strategy, "strat_c", config=cfg)
    assert "more than one incubator portfolio" in msg, msg
    assert "double" in msg, msg


def test_the_accessors_hand_out_copies_and_not_the_cache() -> None:
    """
    The config is cached per path, and a cached mutable dict handed out by
    reference is a trap: one caller appending to `active_strategies` would
    change what every other caller sees — including the routing decision for a
    live account — with nothing raising and nothing written to disk.
    """
    clear_cache()
    first = load_portfolio_config()
    first["portfolios"]["Prop-Odd"]["active_strategies"].append("ghost")
    first["asset_metadata"]["MNQ"]["point_value"] = 999.0

    second = load_portfolio_config()
    assert second["portfolios"]["Prop-Odd"]["active_strategies"] == [], (
        "a mutation leaked into the cached config")
    assert second["asset_metadata"]["MNQ"]["point_value"] == 2.0

    p = get_portfolio_by_account(EXECUTION_ACCOUNTS["Prop-Odd"], config=second)
    p["basket"]["assets"].append("SPY")
    assert second["portfolios"]["Prop-Odd"]["basket"]["assets"] == \
        ["MNQ"], "get_portfolio_by_account returned a live reference"


# ==========================================================================
# 6. The gap that will stop the next task
# ==========================================================================
def test_the_basket_assets_have_no_market_data_and_the_config_says_so() -> None:
    """
    THE MICROS ARE ABSENT FROM THE LAKE; 6E AND 6J ARE NOT. MNQ, MES and MGC
    have no `symbol=` partition — CLAUDE.md lists the micros among the symbols
    whose definitions were never downloaded, so they are UNVERIFIED in
    `backtest/specs.py` as well — while the FX pair `724a92b` added trades on
    a full-size series the lake has carried all along.

    THIS DOES NOT GATE THE BACKTEST PIPELINE, and the summary line's wording
    ("these baskets cannot be backtested yet") should not be read as saying it
    does. Nothing under `backtest/` imports this module: a strategy is swept
    and certified on the FULL-SIZE contract (NQ, ES, GC), and the micro is
    only what the live dispatcher sends the order in. What the gap actually
    blocks is a forward test of a BASKET on its own execution symbol.

    This case asserts the REPORTING, not the absence: if the micros are pulled
    tomorrow the list empties and the case still passes.
    """
    cfg = config()
    missing = cfg["assets_without_market_data"]
    assert isinstance(missing, list)
    assert set(missing) <= set(cfg["asset_metadata"]), missing

    from mdlib.lake import LAKE
    if not LAKE.exists():
        print("        SKIPPED the lake half: /mnt/backtest is not mounted")
        return
    for symbol in cfg["asset_metadata"]:
        on_disk = (LAKE / f"symbol={symbol}").exists()
        assert (symbol in missing) != on_disk, (
            f"{symbol}: reported as missing={symbol in missing} but the lake "
            f"says {on_disk}")
    if missing:
        print(f"        no market data for {', '.join(missing)} — these "
              f"baskets cannot be backtested yet")


def test_the_console_summary_names_every_account() -> None:
    """
    `python3 portfolio/config_loader.py` is how an operator checks the routing
    table by eye. It has to show both quadrant encodings, because a summary
    printing a bare `Q1` is the ambiguity this module exists to remove.
    """
    text = describe(config=config())
    for pid in REQUIRED_PORTFOLIOS:
        assert pid in text, pid
    for symbol in REQUESTED_POINT_VALUES:
        assert symbol in text, symbol
    assert "Q3_LOW_VOL_TREND" in text and "Q3" in text
    assert "$1,000" in text, "the derived forward drawdown is not on the summary"


# ==========================================================================
# The script runner. `assert` is the failure mechanism, so pytest and this
# report the same thing — see the module docstring.
# ==========================================================================
def main() -> int:
    cases = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    print(f"portfolio configuration — {len(cases)} cases\n")
    for name, fn in cases:
        clear_cache()
        try:
            fn()
        except Exception as e:                   # noqa: BLE001 - reported below
            failures.append((name, e))
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
        else:
            print(f"  PASS  {name}")
    print()
    if failures:
        print(f"{len(failures)} of {len(cases)} FAILED: "
              + ", ".join(n for n, _ in failures))
        return 1
    print(f"all {len(cases)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
