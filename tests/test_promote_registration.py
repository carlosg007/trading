#!/usr/bin/env python3
"""
test_promote_registration.py - Stage 5 registers a promotion onto a portfolio.

Location:  ~/src/trading/tests/test_promote_registration.py
Run:       python3 tests/test_promote_registration.py

WHAT IS UNDER TEST, AND WHY EACH CASE EXISTS
============================================
`backtest/promote.py` writes a promoted strategy into `config/portfolios.json`
in TWO places at once - the id into `active_strategies`, which is the
permission three separate modules read, and the record into
`strategy_allocations`, which is the description. Every case below is a way
those two halves, or the certified scope they describe, can go wrong QUIETLY:

  * a dict written into `active_strategies` would break
    `get_portfolio_for_strategy`, the live dispatcher and the Stage 5 card
    without any of them saying the schema changed, so the id stays a string;
  * a promotion registered twice would size one signal twice on one account;
  * a strategy left on two incubator portfolios is the config
    `portfolio.config_loader` refuses, so a re-route is a MOVE;
  * the module's `SYMBOLS`/`TIMEFRAME` are its declarations and NOT the
    certified pair, and mixing the halves records an allocation for a run
    nobody made with both halves individually true;
  * `regime_filter` disagreeing with the account's `basket.regime_quadrants`
    means the live gate stands the strategy down in the one quadrant it was
    certified for - it never trades, every count still adds up, and the symptom
    is silence.

NOTHING HERE TOUCHES THE REAL ROUTING TABLE. Every case copies
`config/portfolios.json` into a temp directory first: a test that edited the
live file would allocate a fixture strategy to a live account, and it would
still pass.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest.pipeline import (                                   # noqa: E402
    base_strategy,
    split_strategy_id,
    strategy_id,
)
from backtest.promote import (                                    # noqa: E402
    ALLOCATIONS_KEY,
    DEFAULT_ALLOCATION,
    DEFAULT_VERSION,
    NOT_RESOLVED,
    SOURCE_CANDIDATES,
    certified_scope,
    unrouted_packages,
    ensure_portfolio_groups,
    incubator_portfolios,
    merge_configuration,
    register_portfolio,
    resolve_audit_file,
    resolve_portfolio,
    resolve_source,
    resolve_version,
)

REAL_CONFIG = REPO_ROOT / "config" / "portfolios.json"
STRAT = "fixture_registration_strategy"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def temp_config(mutate=None, *, clear=True) -> Path:
    """
    The real routing table, copied somewhere a test may write to.

    `active_strategies` and `strategy_allocations` are CLEARED on every
    portfolio first. The cases below assert what registration does to an
    account — which one the default rule picks, whether an id was appended
    once, whether a re-route left the old entry behind — and every one of those
    is a statement about a known starting point. Inheriting whatever the live
    table happens to hold makes them pass or fail on how many strategies are
    allocated today, which is a fact about the operator's week rather than
    about the code.
    """
    blob = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    if clear:
        for portfolio in blob.get("portfolios", {}).values():
            portfolio["active_strategies"] = []
            portfolio.pop(ALLOCATIONS_KEY, None)
    if mutate:
        mutate(blob)
    path = Path(tempfile.mkdtemp(prefix="promote_reg_")) / "portfolios.json"
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    return path


def narrow(pid: str, quadrants: list[str]):
    """
    A `temp_config` mutation that RESTRICTS one portfolio's quadrants.

    The cases below test what registration does when an account does not trade
    the certified quadrant. They used the LIVE table's Even track as that
    fixture, which held Q1/Q2 - so widening the real config to admit Q3
    (2026-09-02) left them asserting a config state instead of the guard, and
    they failed on an operator's config edit rather than on a code change. The
    helper's own docstring already names that class of defect. The guard is
    real and is still tested; only the fixture moved out of the live file.
    """
    def _mutate(blob):
        blob["portfolios"][pid]["basket"]["regime_quadrants"] = list(quadrants)
    return _mutate


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scope(symbol="NQ", timeframe="1h", quadrant="Q2") -> dict:
    return {"symbol": symbol, "timeframe": timeframe,
            "regime_filter": quadrant, "regime": "High Volatility / Ranging",
            "resolved_from": {"symbol": "gate_audit_NQ_1h.json"}}


def audit_cert(**over) -> dict:
    cert = {"audit_file": "/x/pipeline/s/gate_audit_NQ_1h.json",
            "audit_symbol": "NQ", "audit_timeframe": "1h",
            "target_quadrant": "Q2",
            "target_regime": "High Volatility / Ranging"}
    cert.update(over)
    return cert


# ==========================================================================
# 1. the payload, and where each half of it lands
# ==========================================================================
def test_permission_is_a_string_and_the_record_is_a_sibling() -> None:
    """
    The id goes into `active_strategies` as a STRING; the payload goes into
    `strategy_allocations` beside it.

    This is the case that pins the whole design. Three modules read
    `active_strategies` as a list of strings — `get_portfolio_for_strategy`
    with `id in list`, `live_dispatcher` by building
    `approved_incubator/<id>/`, and the Stage 5 card by lowercasing each
    element — and every one of them fails silently on a dict.
    """
    path = temp_config()
    out = register_portfolio(STRAT, version="A", scope=scope(),
                             config_path=path)
    block = read(path)["portfolios"][out["portfolio_id"]]

    assert block["active_strategies"] == [STRAT], block["active_strategies"]
    assert all(isinstance(s, str) for s in block["active_strategies"])

    record = block[ALLOCATIONS_KEY][STRAT]
    assert record["strat"] == STRAT
    assert record["symbol"] == "NQ"
    assert record["timeframe"] == "1h"
    assert record["version"] == "A"
    assert record["allocation"] == DEFAULT_ALLOCATION == 1
    assert record["regime_filter"] == "Q2"
    assert record["status"] == "incubating"
    assert record["path"] == (f"strategies/approved_incubator/{STRAT}/strat.py")


def test_the_record_is_readable_by_the_stage5_card() -> None:
    """
    The Stage 5 card resolves membership from `active_strategies` alone, and
    the registration has to satisfy the reader that already exists rather than
    a new one written alongside it.
    """
    from backtest.discord_reporter import portfolio_membership

    path = temp_config()
    out = register_portfolio(STRAT, version="A", scope=scope(),
                             config_path=path)
    member = portfolio_membership(STRAT, staged=True, config_path=path)
    assert member["membership"] == f"Active {out['portfolio_id']} (Allocated)", (
        member["membership"])
    assert "active_strategies" in member["basis"]


def test_the_written_table_still_loads_through_the_real_loader() -> None:
    """
    `backtest/promote.py` cannot import `portfolio.config_loader` — the
    dependency runs one way — so nothing in the writer proves the file it
    produced is one the loader accepts. This case is that proof.
    """
    from portfolio.config_loader import clear_cache, load_portfolio_config

    path = temp_config()
    register_portfolio(STRAT, version="A", scope=scope(), config_path=path)
    clear_cache()
    cfg = load_portfolio_config(str(path))
    assert STRAT in cfg["portfolios"]["Incubator-Even"]["active_strategies"] \
        or STRAT in cfg["portfolios"]["Incubator-Odd"]["active_strategies"]


def test_the_two_modules_spell_the_key_the_same_way() -> None:
    """
    `ALLOCATIONS_KEY` is duplicated in `backtest/promote.py` because the
    dependency runs one way and it cannot import the authoritative copy. A
    convention nothing enforces is a convention until somebody renames one
    side, at which point the writer writes a block the loader never looks at —
    silently, because the permission still works.
    """
    from portfolio.config_loader import ALLOCATIONS_KEY as LOADER_KEY
    assert ALLOCATIONS_KEY == LOADER_KEY == "strategy_allocations"


# ==========================================================================
# 2. routing
# ==========================================================================
def test_default_routing_takes_the_emptier_incubator_and_alternates() -> None:
    """
    Fewest active strategies wins, ties alphabetically — which is round-robin
    across successive promotions, because the account that just took one has
    more next time.

    Scoped to promotions the CERTIFIED QUADRANT cannot route, which since
    2026-08-24 is what leaves the headcount rule deciding alone. A quadrant
    only one account declares pins every such promotion to that account, and
    correctly so — see
    `test_the_certified_quadrant_routes_to_an_account_that_permits_it`.
    """
    path = temp_config()
    # NEITHER axis may route, or the rule under test is not the one deciding.
    # The certified SYMBOL filters before the quadrant from 2026-08-24, and NQ
    # is carried by exactly one incubator basket - so a scope naming it pins
    # every promotion to that account and there is no alternation to observe.
    unrouted = dict(scope(), regime_filter=NOT_RESOLVED, symbol=NOT_RESOLVED)
    first = register_portfolio("strat_one", version="A", scope=unrouted,
                               config_path=path)
    second = register_portfolio("strat_two", version="A", scope=unrouted,
                                config_path=path)
    third = register_portfolio("strat_three", version="A", scope=unrouted,
                               config_path=path)
    assert first["portfolio_id"] == "Incubator-Even", first["portfolio_id"]
    assert second["portfolio_id"] == "Incubator-Odd", second["portfolio_id"]
    assert third["portfolio_id"] == "Incubator-Even", third["portfolio_id"]
    assert "fewest active strategies" in third["basis"]


def test_explicit_portfolio_matches_ignoring_case_and_punctuation() -> None:
    """`incubator-odd` is what an operator types; `Incubator-Odd` is the key."""
    for spelling in ("incubator-odd", "Incubator-Odd", "INCUBATOR_ODD",
                     "incubatorodd"):
        path = temp_config()
        out = register_portfolio(STRAT, version="A", scope=scope(),
                                 portfolio=spelling, config_path=path)
        assert out["portfolio_id"] == "Incubator-Odd", (spelling, out)
        assert out["basis"] == f"--portfolio {spelling}"


def test_the_prop_track_is_refused() -> None:
    """
    A promotion registers onto the incubator track only. Graduating to prop is
    decided on FORWARD paper trades by `portfolio/promotion_daemon.py`, not on
    a certification, and this is the one step the forward incubation exists to
    sit between.
    """
    path = temp_config()
    try:
        register_portfolio(STRAT, version="A", scope=scope(),
                           portfolio="prop-odd", config_path=path)
    except ValueError as exc:
        assert "incubator" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("a prop portfolio was accepted")
    # And nothing was written on the way to refusing.
    for pid in incubator_portfolios(read(path)["portfolios"]):
        assert read(path)["portfolios"][pid]["active_strategies"] == []


def test_an_unknown_portfolio_is_refused_by_name() -> None:
    path = temp_config()
    try:
        register_portfolio(STRAT, version="A", scope=scope(),
                           portfolio="incubator-middle", config_path=path)
    except ValueError as exc:
        assert "no portfolio named" in str(exc), str(exc)
    else:
        raise AssertionError("an unknown portfolio was accepted")


# ==========================================================================
# 3. idempotence, and the double-assignment the loader refuses
# ==========================================================================
def test_registering_twice_does_not_append_twice() -> None:
    """
    Two entries on one account would size one signal twice, and the position
    would be double what the risk profile describes.
    """
    path = temp_config()
    first = register_portfolio(STRAT, version="A", scope=scope(),
                               portfolio="incubator-odd", config_path=path)
    second = register_portfolio(STRAT, version="A",
                                scope=scope(quadrant="Q2"), allocation=3,
                                portfolio="incubator-odd", config_path=path)
    block = read(path)["portfolios"]["Incubator-Odd"]
    assert block["active_strategies"] == [STRAT], block["active_strategies"]
    assert block[ALLOCATIONS_KEY][STRAT]["allocation"] == 3
    assert first["was_registered"] is False
    assert second["was_registered"] is True


def test_re_routing_moves_rather_than_adds() -> None:
    """
    One strategy on two portfolios of one track is the config
    `portfolio.config_loader` refuses to load. A re-route removes the old
    assignment AND its allocation record.
    """
    path = temp_config()
    register_portfolio(STRAT, version="A", scope=scope(),
                       portfolio="incubator-odd", config_path=path)
    out = register_portfolio(STRAT, version="A", scope=scope(),
                             portfolio="incubator-even", config_path=path)
    blob = read(path)["portfolios"]
    assert out["moved_from"] == ["Incubator-Odd"], out["moved_from"]
    assert blob["Incubator-Odd"]["active_strategies"] == []
    assert STRAT not in (blob["Incubator-Odd"].get(ALLOCATIONS_KEY) or {})
    assert blob["Incubator-Even"]["active_strategies"] == [STRAT]
    assert STRAT in blob["Incubator-Even"][ALLOCATIONS_KEY]


def test_a_moved_strategy_still_routes_through_the_loader() -> None:
    """The move has to leave a config `get_portfolio_for_strategy` accepts."""
    from portfolio.config_loader import clear_cache, get_portfolio_for_strategy

    path = temp_config()
    # ES, which Incubator-Even's basket carries - the move has to leave a
    # config the loader ACCEPTS, and from 2026-08-24 it refuses one routing a
    # strategy to a basket that cannot trade its certified contract.
    register_portfolio(STRAT, version="A", scope=scope(symbol="ES"),
                       portfolio="incubator-odd", config_path=path)
    register_portfolio(STRAT, version="A", scope=scope(symbol="ES"),
                       portfolio="incubator-even", config_path=path)
    clear_cache()
    assert get_portfolio_for_strategy(STRAT, config_path=str(path)) \
        == "Incubator-Even"


# ==========================================================================
# 4. the certified scope
# ==========================================================================
def test_the_pair_comes_from_the_certification_not_the_module() -> None:
    """
    A promotion is ONE contract at ONE timeframe.
    `t3_braid_scalp_20260823` declares NQ,ES,CL,GC at 5m and was certified on
    NQ at 1h; taking the pair from the module records an allocation for a run
    nobody made, with both halves individually true.
    """
    meta = {"symbols": ["NQ", "ES", "CL", "GC"], "timeframe": "5m"}
    out = certified_scope(audit_cert(), meta)
    assert out["symbol"] == "NQ"
    assert out["timeframe"] == "1h"
    assert out["regime_filter"] == "Q2"
    assert "certification" in out["resolved_from"]["timeframe"]


def test_the_timeframe_falls_back_to_the_audit_filename() -> None:
    """
    Stage 3 writes the pair into `gate_audit_<SYMBOL>_<TF>.json` whether or not
    it writes a `timeframe` into the body.
    """
    cert = audit_cert()
    cert.pop("audit_timeframe")
    out = certified_scope(cert, {"symbols": ["NQ"], "timeframe": "5m"})
    assert out["timeframe"] == "1h", out
    assert "filename" in out["resolved_from"]["timeframe"]


def test_the_module_timeframe_is_used_only_last_and_says_so() -> None:
    out = certified_scope(None, {"symbols": ["NQ"], "timeframe": "5m"})
    assert out["timeframe"] == "5m"
    assert "MODULE" in out["resolved_from"]["timeframe"], out["resolved_from"]
    assert out["symbol"] == "NQ"          # exactly one declared contract
    assert out["regime_filter"] == NOT_RESOLVED


def test_a_multi_symbol_module_with_no_certification_resolves_no_symbol() -> None:
    """
    Four declared contracts is not a certified contract, and picking one would
    be inventing the promotion's subject.
    """
    out = certified_scope(None, {"symbols": ["NQ", "ES"], "timeframe": "5m"})
    assert out["symbol"] == NOT_RESOLVED
    assert "4 contracts" in out["resolved_from"]["symbol"] \
        or "2 contracts" in out["resolved_from"]["symbol"]


def test_the_quadrant_is_derived_from_the_regime_name_when_absent() -> None:
    """
    The `Q1`..`Q4` code comes from `backtest.profiler.REGIME_TO_QUADRANT`, so
    no spelling of a regime name lives in `promote.py`.
    """
    cert = audit_cert()
    cert.pop("target_quadrant")
    out = certified_scope(cert, {})
    assert out["regime_filter"] == "Q2", out


def test_an_uncertified_promotion_records_not_resolved() -> None:
    """Written rather than omitted: an absent key reads as a field nobody
    filled in, and this one is the difference between a scope that was
    resolved and one that was never available."""
    path = temp_config()
    out = register_portfolio(STRAT, version="A",
                             scope=certified_scope("NOT CERTIFIED", {}),
                             config_path=path)
    record = out["record"]
    assert record["symbol"] == NOT_RESOLVED
    assert record["regime_filter"] == NOT_RESOLVED
    assert any("NOT RESOLVED" in n for n in out["notes"]), out["notes"]


# ==========================================================================
# 5. the regime conflict — the finding whose symptom is silence
# ==========================================================================
def test_a_quadrant_the_account_does_not_trade_is_flagged_not_widened() -> None:
    """
    `basket.regime_quadrants` is the ACCOUNT's permission, shared by every
    strategy on it. Widening it to admit this promotion would hand every other
    strategy on that account a quadrant nobody certified it for.

    A Q3 certification registered onto an account that trades only Q1/Q2 must
    be reported and must leave the basket alone.

    THE NARROWED ACCOUNT IS A FIXTURE, not the live table. Every portfolio in
    `config/portfolios.json` was widened to all four quadrants on 2026-09-02,
    so reading the real Even track here would assert a config state rather
    than the guard - and would fail on an operator's config edit rather than
    on a code change. `narrow()` pins the starting point.
    """
    path = temp_config(narrow("Incubator-Even",
                              ["Q1_HIGH_VOL_TREND", "Q2_HIGH_VOL_CHOP"]))
    out = register_portfolio(STRAT, version="A",
                             scope=scope(symbol="ES", quadrant="Q3"),
                             portfolio="incubator-even", config_path=path)
    basket = read(path)["portfolios"]["Incubator-Even"]["basket"]
    assert basket["regime_quadrants"] == ["Q1_HIGH_VOL_TREND",
                                          "Q2_HIGH_VOL_CHOP"], basket
    assert any("Q3" in n and "never trade" in n for n in out["notes"]), out["notes"]


def test_a_matching_quadrant_raises_no_note() -> None:
    """
    Incubator-Even trades Q1/Q2, so a Q2 certification is silent.

    On ES, because Incubator-Even's basket is MES/MGC: a NQ certification here
    would be silent on the QUADRANT and raise a note about the SYMBOL, which
    is a different subject and is covered by its own case.
    """
    path = temp_config()
    out = register_portfolio(STRAT, version="A",
                             scope=scope(symbol="ES", quadrant="Q2"),
                             portfolio="incubator-even", config_path=path)
    assert not out["notes"], out["notes"]


def test_the_loader_reports_the_conflict_too() -> None:
    """
    The authoritative comparison lives in `portfolio/config_loader.py`, where
    the schema-label -> Q1..Q4 mapping lives. It is REPORTED, never raised: an
    allocation record routes nothing, and refusing the config would take the
    live loop down over a descriptive field.
    """
    from portfolio.config_loader import clear_cache, load_portfolio_config

    # Narrowed as a FIXTURE — the live table permits all four quadrants
    # everywhere since 2026-09-02, so there is no conflict to report there.
    path = temp_config(narrow("Incubator-Even",
                              ["Q1_HIGH_VOL_TREND", "Q2_HIGH_VOL_CHOP"]))
    register_portfolio(STRAT, version="A",
                       scope=scope(symbol="ES", quadrant="Q3"),
                       portfolio="incubator-even", config_path=path)
    clear_cache()
    cfg = load_portfolio_config(str(path))
    conflicts = cfg["allocation_reconciliation"]["regime_conflicts"]
    assert len(conflicts) == 1, conflicts
    assert conflicts[0]["strategy_id"] == STRAT
    assert conflicts[0]["regime_filter"] == "Q3"
    assert conflicts[0]["portfolio_quadrants"] == ["Q1", "Q2"]


def test_an_orphan_record_is_reported_and_does_not_refuse_the_config() -> None:
    from portfolio.config_loader import clear_cache, load_portfolio_config

    def orphan(blob):
        blob["portfolios"]["Incubator-Odd"][ALLOCATIONS_KEY] = {
            "ghost": {"strat": "ghost", "regime_filter": "Q3"}}
    path = temp_config(orphan)
    clear_cache()
    cfg = load_portfolio_config(str(path))
    orphans = cfg["allocation_reconciliation"]["orphan_records"]
    assert [o["strategy_id"] for o in orphans] == ["ghost"], orphans


def test_a_hand_added_strategy_is_reported_as_unallocated_not_as_broken() -> None:
    """
    An id with no record beside it is every strategy assigned by hand and every
    one assigned before this key existed. Reported so "registered by
    promote.py" and "added by an operator" stay distinguishable.
    """
    from portfolio.config_loader import clear_cache, load_portfolio_config

    def by_hand(blob):
        blob["portfolios"]["Incubator-Odd"]["active_strategies"] = ["typed_in"]
    path = temp_config(by_hand)
    clear_cache()
    cfg = load_portfolio_config(str(path))
    un = cfg["allocation_reconciliation"]["unallocated"]
    assert [u["strategy_id"] for u in un] == ["typed_in"], un


# ==========================================================================
# 6. writing
# ==========================================================================
def test_allocation_below_one_contract_is_refused() -> None:
    path = temp_config()
    for bad in (0, -1):
        try:
            register_portfolio(STRAT, version="A", scope=scope(),
                               allocation=bad, config_path=path)
        except ValueError as exc:
            assert "at least 1 contract" in str(exc)
        else:
            raise AssertionError(f"allocation={bad} was accepted")


def test_a_malformed_table_is_refused_before_anything_is_written() -> None:
    def wreck(blob):
        blob["portfolios"]["Incubator-Odd"]["active_strategies"] = "not a list"
    path = temp_config(wreck)
    before = path.read_text(encoding="utf-8")
    try:
        register_portfolio(STRAT, version="A", scope=scope(),
                           portfolio="incubator-odd", config_path=path)
    except ValueError as exc:
        assert "active_strategies" in str(exc)
    else:
        raise AssertionError("a malformed table was written into")
    assert path.read_text(encoding="utf-8") == before


def test_no_temp_file_is_left_behind() -> None:
    path = temp_config()
    register_portfolio(STRAT, version="A", scope=scope(), config_path=path)
    leftovers = [p.name for p in path.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == [], leftovers


def test_untouched_portfolios_are_byte_identical() -> None:
    """
    Registration edits ONE portfolio. A rewrite that reordered or reformatted
    the other three would make every future diff of this file unreadable, which
    is how a change nobody intended gets reviewed as noise.
    """
    path = temp_config()
    before = read(path)["portfolios"]
    out = register_portfolio(STRAT, version="A", scope=scope(),
                             config_path=path)
    after = read(path)["portfolios"]
    for pid in before:
        if pid == out["portfolio_id"]:
            continue
        assert json.dumps(before[pid]) == json.dumps(after[pid]), pid


# ==========================================================================
# 7. graduation carries the record
# ==========================================================================
def test_graduation_moves_the_allocation_record_with_the_permission() -> None:
    """
    `portfolio/promotion_daemon.py` moves ONE HOP of the ladder. Moving only
    `active_strategies` would leave the record on an account that no longer
    holds the strategy, beside one holding it with nothing describing it.

    THE FIRST HOP IS INCUBATOR -> EVAL since 2026-09-03, not Incubator ->
    Prop, and the status says so: a strategy that cleared incubation has
    reached an EVALUATION account and stamping it `GRADUATED_PROP` would
    report it as trading a funded book.
    """
    from portfolio.promotion_daemon import promote_strategy

    path = temp_config()
    register_portfolio(STRAT, version="A", scope=scope(quadrant="Q4"),
                       portfolio="incubator-odd", config_path=path)

    ledger_path = path.parent / "incubator_ledger.json"
    ledger_path.write_text(json.dumps({"strategies": {STRAT: {
        "status": "INCUBATING", "portfolio": "Incubator-Odd"}}}), "utf-8")

    promote_strategy(STRAT, "Incubator-Odd", config_path=str(path),
                     ledger_path=str(ledger_path))
    blob = read(path)["portfolios"]
    assert STRAT not in (blob["Incubator-Odd"].get(ALLOCATIONS_KEY) or {})
    assert STRAT not in (blob["Prop-Odd"].get(ALLOCATIONS_KEY) or {}), (
        "the first hop reached the funded book, skipping evaluation")
    moved = blob["Eval-Odd"][ALLOCATIONS_KEY][STRAT]
    assert moved["status"] == "GRADUATED_EVAL", moved["status"]
    assert moved["source_portfolio"] == "Incubator-Odd"
    assert moved["symbol"] == "NQ"


def test_the_second_hop_reaches_the_funded_book() -> None:
    """
    Eval -> Prop, graded on the forward trades the EVALUATION account
    produced. A strategy that cleared incubation has said nothing yet about
    whether it clears a prop firm's own rules, which is the whole reason the
    rung exists.
    """
    from portfolio.promotion_daemon import promote_strategy

    path = temp_config()
    register_portfolio(STRAT, version="A", scope=scope(quadrant="Q4"),
                       portfolio="incubator-odd", config_path=path)
    ledger_path = path.parent / "incubator_ledger.json"
    ledger_path.write_text(json.dumps({"strategies": {STRAT: {
        "status": "INCUBATING", "portfolio": "Incubator-Odd"}}}), "utf-8")

    promote_strategy(STRAT, "Incubator-Odd", config_path=str(path),
                     ledger_path=str(ledger_path))
    promote_strategy(STRAT, "Eval-Odd", config_path=str(path),
                     ledger_path=str(ledger_path))

    blob = read(path)["portfolios"]
    assert STRAT not in (blob["Eval-Odd"].get(ALLOCATIONS_KEY) or {})
    moved = blob["Prop-Odd"][ALLOCATIONS_KEY][STRAT]
    assert moved["status"] == "GRADUATED_PROP", moved["status"]
    assert moved["source_portfolio"] == "Eval-Odd"


# ==========================================================================
# 6. the CLI resolves what it was not told, and the record holds every
#    certified pair rather than only the last one promoted
# ==========================================================================
def test_a_missing_routing_table_is_created_rather_than_failing_a_promotion() -> None:
    """
    A promotion is written and committed BEFORE registration runs, so a
    routing table that is not there must not turn a completed promotion into
    an error and an unallocated strategy. The bootstrap creates the whole
    four-account partition - `portfolio.config_loader` refuses a table missing
    any of the four - with both incubator groups empty.
    """
    path = Path(tempfile.mkdtemp(prefix="promote_boot_")) / "portfolios.json"
    assert not path.exists()
    out = register_portfolio(STRAT, version="A", scope=scope(),
                             config_path=path)
    blob = read(path)
    from portfolio.config_loader import REQUIRED_PORTFOLIOS
    assert set(blob["portfolios"]) == set(REQUIRED_PORTFOLIOS)
    assert out["portfolio_id"] in incubator_portfolios(blob["portfolios"])
    # The one account that did NOT take the strategy is empty, which is what
    # "initialised to []" has to mean for it to be worth anything.
    other = [p for p in incubator_portfolios(blob["portfolios"])
             if p != out["portfolio_id"]][0]
    assert blob["portfolios"][other]["active_strategies"] == []


def test_a_bootstrapped_table_announces_the_risk_envelope_it_invented() -> None:
    """
    `default_account_size` and the risk budget are what a live position is
    sized against. A bootstrap cannot know them, so it writes the shipped
    defaults and SAYS SO - a placeholder nobody was told about is the "rule
    nobody wrote down" the loader refuses to apply elsewhere.
    """
    path = Path(tempfile.mkdtemp(prefix="promote_boot_")) / "portfolios.json"
    out = register_portfolio(STRAT, version="A", scope=scope(),
                             config_path=path)
    assert any("SHIPPED DEFAULT" in n for n in out["notes"]), out["notes"]


def test_a_bootstrapped_table_still_loads_through_the_real_loader() -> None:
    """
    The one guarantee that matters about a generated config: the loader that
    the live loop, the incubator tracker and the Stage 5 card all go through
    accepts it. A skeleton it refuses takes all three down together, on the
    next run rather than on this one.
    """
    from portfolio.config_loader import load_portfolio_config
    path = Path(tempfile.mkdtemp(prefix="promote_boot_")) / "portfolios.json"
    register_portfolio(STRAT, version="A", scope=scope(), config_path=path)
    cfg = load_portfolio_config(path)
    # Compared against the LOADER's own requirement rather than a literal set:
    # the ladder gained an evaluation rung on 2026-09-03 and a hardcoded four
    # here would fail on the config change instead of on a bootstrap defect.
    from portfolio.config_loader import REQUIRED_PORTFOLIOS
    assert set(cfg["portfolios"]) == set(REQUIRED_PORTFOLIOS)


def test_a_group_missing_from_an_existing_table_is_created_alone() -> None:
    """One missing group is added; the three that are there are not rewritten."""
    def drop(blob):
        blob["portfolios"].pop("Incubator-Odd")
    path = temp_config(drop)
    before = read(path)["portfolios"]["Prop-Even"]
    out = register_portfolio(STRAT, version="A", scope=scope(),
                             config_path=path)
    after = read(path)["portfolios"]
    assert "Incubator-Odd" in after
    assert after["Prop-Even"] == before
    assert any("created Incubator-Odd" in n for n in out["notes"]), out["notes"]


def test_active_strategies_of_a_wrong_shape_is_refused_not_repaired() -> None:
    """
    An empty list and a string are the same to `if name in active`, so
    replacing one would throw away a permission that is currently granted.
    The bootstrap adds the key when it is ABSENT and never rewrites it.
    """
    def wreck(blob):
        blob["portfolios"]["Incubator-Odd"]["active_strategies"] = "alpha"
        blob["portfolios"]["Incubator-Even"]["active_strategies"] = "alpha"
    path = temp_config(wreck)
    try:
        register_portfolio(STRAT, version="A", scope=scope(), config_path=path)
    except ValueError as e:
        assert "active_strategies" in str(e)
    else:
        raise AssertionError("a string active_strategies was accepted")


def test_three_certified_timeframes_all_survive_in_the_record() -> None:
    """
    THE BUG THIS EXISTS FOR. A campaign certifies one strategy at several
    timeframes and promotes it once per pair. Keyed by strategy id alone the
    third promotion silently REPLACED the first two, and nothing on the
    console said which two had been dropped.
    """
    path = temp_config()
    pid = None
    for tf, quad in (("15m", "Q1"), ("30m", "Q2"), ("1h", "Q2")):
        out = register_portfolio(STRAT, version="A",
                                 scope=scope(timeframe=tf, quadrant=quad),
                                 config_path=path)
        pid = out["portfolio_id"]
    record = read(path)["portfolios"][pid][ALLOCATIONS_KEY][STRAT]
    pairs = [(c["symbol"], c["timeframe"]) for c in record["configurations"]]
    assert pairs == [("NQ", "15m"), ("NQ", "30m"), ("NQ", "1h")], pairs
    # Each carries the whole payload, not just the pair.
    for c in record["configurations"]:
        for key in ("strat", "symbol", "timeframe", "version", "allocation",
                    "regime_filter", "status"):
            assert key in c, f"{key} missing from {c}"
        assert c["status"] == "incubating"
    # The top-level fields still describe ONE promotion - the last - because
    # that is where config_loader and promotion_daemon read them.
    assert record["timeframe"] == "1h"
    assert record["symbol"] == "NQ"


def test_re_promoting_one_pair_updates_it_and_does_not_duplicate_it() -> None:
    """
    Two entries for one (strat, symbol, timeframe) would size the same signal
    twice on one account. The entry is updated IN PLACE and keeps its
    position, so a re-run that changed nothing does not churn the table.
    """
    path = temp_config()
    for tf in ("15m", "30m", "1h"):
        out = register_portfolio(STRAT, version="A", scope=scope(timeframe=tf),
                                 config_path=path)
    pid = out["portfolio_id"]
    again = register_portfolio(STRAT, version="B", scope=scope(timeframe="30m"),
                               allocation=3, config_path=path)
    record = read(path)["portfolios"][pid][ALLOCATIONS_KEY][STRAT]
    configs = record["configurations"]
    assert len(configs) == 3, [c["timeframe"] for c in configs]
    assert [c["timeframe"] for c in configs] == ["15m", "30m", "1h"]
    assert configs[1]["version"] == "B" and configs[1]["allocation"] == 3
    assert any("UPDATED in place" in n for n in again["notes"]), again["notes"]


def test_a_record_written_before_configurations_existed_is_not_lost() -> None:
    """
    Every allocation already in a live routing table predates this list. The
    first re-promotion must carry it into `configurations` rather than
    starting an empty one, which would drop a promotion somebody made.
    """
    def seed(blob):
        block = blob["portfolios"]["Incubator-Even"]
        block["active_strategies"] = [STRAT]
        block[ALLOCATIONS_KEY] = {STRAT: {
            "strat": STRAT, "symbol": "ES", "timeframe": "5m",
            "version": "A", "allocation": 1, "regime_filter": "Q1",
            "status": "incubating"}}
    path = temp_config(seed)
    # Seeded on Incubator-Even, so the contract is ES - that basket is MES/MGC
    # and a NQ record there is the routing conflict a different case covers.
    register_portfolio(STRAT, version="A",
                       scope=scope(symbol="ES", timeframe="1h"),
                       config_path=path)
    record = read(path)["portfolios"]["Incubator-Even"][ALLOCATIONS_KEY][STRAT]
    pairs = [(c["symbol"], c["timeframe"]) for c in record["configurations"]]
    assert pairs == [("ES", "5m"), ("ES", "1h")], pairs


def test_a_re_promotion_does_not_move_the_account() -> None:
    """
    The count rule balances NEW strategies. Applied to one that already has an
    account, each successive promotion of the same strategy found it on the
    fuller side and moved it - bouncing Even -> Odd -> Even across three
    certified timeframes, flipping which quadrants the account permits every
    time, with each move announced as a routing decision.
    """
    path = temp_config()
    first = register_portfolio(STRAT, version="A", scope=scope(timeframe="15m"),
                               config_path=path)
    for tf in ("30m", "1h"):
        again = register_portfolio(STRAT, version="A",
                                   scope=scope(timeframe=tf),
                                   config_path=path)
        assert again["portfolio_id"] == first["portfolio_id"], (
            f"{tf} moved {STRAT} from {first['portfolio_id']} to "
            f"{again['portfolio_id']}")
        assert again["moved_from"] == []
        assert "already registered" in again["basis"]


def test_an_explicit_portfolio_still_beats_staying_put() -> None:
    """An operator correcting the route outranks the strategy's own history."""
    path = temp_config()
    first = register_portfolio(STRAT, version="A", scope=scope(),
                               config_path=path)
    other = [p for p in incubator_portfolios(read(path)["portfolios"])
             if p != first["portfolio_id"]][0]
    moved = register_portfolio(STRAT, version="A", scope=scope(),
                               portfolio=other, config_path=path)
    assert moved["portfolio_id"] == other
    assert moved["moved_from"] == [first["portfolio_id"]]


def test_merge_configuration_is_keyed_on_the_triple_and_nothing_else() -> None:
    """
    Same strategy, same symbol, different timeframe is a DIFFERENT allocation;
    the same triple in another spelling is the SAME one. Case folding is the
    point - `nq`/`NQ` and `1H`/`1h` name one pair.
    """
    base = {"strat": "s", "symbol": "NQ", "timeframe": "1h", "version": "A"}
    out, replaced = merge_configuration([], base)
    assert not replaced and len(out) == 1
    out, replaced = merge_configuration(out, dict(base, timeframe="30m"))
    assert not replaced and len(out) == 2
    out, replaced = merge_configuration(out, dict(base, symbol="nq",
                                                 timeframe="1H", version="B"))
    assert replaced and len(out) == 2, out
    assert out[0]["version"] == "B"


def test_ensure_portfolio_groups_leaves_a_complete_table_untouched() -> None:
    """A table that already has both groups is returned byte-identical."""
    blob = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    before = json.dumps(blob, sort_keys=True)
    out, notes = ensure_portfolio_groups(blob)
    assert notes == [], notes
    assert json.dumps(out, sort_keys=True) == before


# ==========================================================================
# 7. what the command line no longer has to say
# ==========================================================================
def test_the_source_module_resolves_from_the_strategy_name() -> None:
    """
    `--source` was `required=True`, which made the orchestrator the only
    practical caller. The candidates are tried in order and the one that
    exists wins; this repository keeps its modules in `experimental/`.
    """
    path, basis = resolve_source("t3_braid_scalp_20260823")
    assert path == REPO_ROOT / "strategies/experimental/t3_braid_scalp_20260823.py"
    assert "experimental" in basis
    assert SOURCE_CANDIDATES[0] == "strategies/{strat}.py"


def test_an_unresolvable_source_raises_rather_than_guessing() -> None:
    try:
        resolve_source("no_such_strategy_at_all")
    except FileNotFoundError as e:
        assert "Tried" in str(e)
    else:
        raise AssertionError("a missing module resolved to something")


def test_an_explicit_source_is_honoured_verbatim() -> None:
    path, basis = resolve_source("anything", "strategies/x.py")
    assert basis == "--source" and path.name == "x.py"


def test_the_version_comes_from_the_gate_audits_own_per_version_verdict() -> None:
    """
    A gate audit carries `passed` KEYED BY VERSION, not a scalar `version`,
    because Stage 3 audits both twins in one pass. The version promoted is the
    one whose Gate R passed - promoting the other attaches a generated Version
    B wrapper to an audit of the baseline.
    """
    directory = Path(tempfile.mkdtemp(prefix="promote_ver_"))
    audit = directory / "gate_audit_NQ_1h.json"
    audit.write_text(json.dumps({"passed": {"A": False, "B": True},
                                 "versions": {"A": {}, "B": {}}}))
    version, basis = resolve_version(audit)
    assert version == "B", basis
    assert "gate_audit_NQ_1h.json" in basis


def test_two_passing_versions_are_not_chosen_between() -> None:
    """
    Which twin to trade is the Dual-Version Mandate's decision, made on
    whether B beat A out of sample. It is the operator's, so both passing
    falls through to the default and says why.
    """
    directory = Path(tempfile.mkdtemp(prefix="promote_ver_"))
    audit = directory / "gate_audit_NQ_1h.json"
    audit.write_text(json.dumps({"passed": {"A": True, "B": True}}))
    version, basis = resolve_version(audit)
    assert version == DEFAULT_VERSION
    assert "--version" in basis


def test_an_explicit_version_wins_and_is_upper_cased() -> None:
    assert resolve_version(None, "b") == ("B", "--version")


def test_the_audit_file_resolves_from_the_symbol_and_timeframe() -> None:
    """`gate_audit_<SYMBOL>_<TF>.json` is where Stage 3 writes it."""
    directory = Path(tempfile.mkdtemp(prefix="promote_aud_"))
    (directory / "gate_audit_NQ_1h.json").write_text("{}")
    path, basis = resolve_audit_file("s", "NQ", "1h", out_dir=str(directory))
    assert path == directory / "gate_audit_NQ_1h.json"
    assert "--symbol/--timeframe" in basis


def test_several_certified_configurations_refuse_to_resolve_to_one() -> None:
    """
    Promoting one of several would leave the rest on disk with nothing saying
    they were not chosen, and every field on the resulting card would still
    read correctly.
    """
    directory = Path(tempfile.mkdtemp(prefix="promote_aud_"))
    for tf in ("15m", "1h"):
        (directory / f"gate_audit_NQ_{tf}.json").write_text("{}")
    (directory / "stage3_audit_summary.json").write_text(json.dumps({
        "results": [{"symbol": "NQ", "timeframe": "15m", "certified": True},
                    {"symbol": "NQ", "timeframe": "1h", "certified": True}]}))
    try:
        resolve_audit_file("s", out_dir=str(directory))
    except ValueError as e:
        assert "2 certified" in str(e) and "--promote-only" in str(e)
    else:
        raise AssertionError("two certifications resolved to one")


def test_one_certified_configuration_resolves_without_being_named() -> None:
    directory = Path(tempfile.mkdtemp(prefix="promote_aud_"))
    (directory / "gate_audit_NQ_1h.json").write_text("{}")
    (directory / "stage3_audit_summary.json").write_text(json.dumps({
        "results": [{"symbol": "NQ", "timeframe": "1h", "certified": True},
                    {"symbol": "NQ", "timeframe": "5m", "certified": False}]}))
    path, basis = resolve_audit_file("s", out_dir=str(directory))
    assert path == directory / "gate_audit_NQ_1h.json"
    assert "the one certified configuration" in basis


def test_the_unsuffixed_gate_audit_is_never_counted_as_a_certification() -> None:
    """
    `gate_audit_<SYM>.json` duplicates whichever timeframe ran last. Counting
    it would make one certification look like two and refuse the directory.
    """
    directory = Path(tempfile.mkdtemp(prefix="promote_aud_"))
    (directory / "gate_audit_NQ.json").write_text("{}")
    (directory / "gate_audit_NQ_1h.json").write_text("{}")
    path, _ = resolve_audit_file("s", out_dir=str(directory))
    assert path == directory / "gate_audit_NQ_1h.json"


def test_an_audit_file_named_by_hand_and_missing_raises() -> None:
    try:
        resolve_audit_file("s", explicit="/nowhere/gate_audit_NQ_1h.json")
    except FileNotFoundError as e:
        assert "refused" in str(e)
    else:
        raise AssertionError("a missing --audit-file was resolved to another")


def test_an_uncertified_pair_reports_why_rather_than_raising() -> None:
    """
    Nothing to cite is NOT CERTIFIED, which `promote()` already reports and
    `--require-certification` already refuses. Raising here would fail a
    promotion the operator may have meant to make without one.
    """
    directory = Path(tempfile.mkdtemp(prefix="promote_aud_"))
    path, basis = resolve_audit_file("s", "NQ", "1h", out_dir=str(directory))
    assert path is None
    assert "has not certified" in basis


# ==========================================================================
# 8. one strategy id per certified pair
# ==========================================================================
def test_version_qualified_ids_round_trip():
    """
    Version A and Version B of one pair are two directories, not one.

    Both can certify. On 2026-08-29 NQ 1h passed Gate R on A at OOS PF 1.25
    and on B at 1.22; without a version segment both resolved to
    `t3_braid_scalp_20260823_NQ_1h`, promote.py wrote A, found the path taken
    when it reached B and exited 1 - `promoted: 1, failed: 1`. A certified
    package was lost with no gate having refused it.
    """
    from backtest.pipeline import (split_strategy_id, strategy_id,
                                   version_of_strategy_id)

    a = strategy_id("t3_braid_scalp_20260823", "NQ", "1h", "A")
    b = strategy_id("t3_braid_scalp_20260823", "NQ", "1h", "B")
    assert (a, b) == ("t3_braid_scalp_20260823_NQ_1h_VA",
                      "t3_braid_scalp_20260823_NQ_1h_VB")
    assert a != b, "the collision this exists to prevent"

    # The parser must still find the timeframe. LiveDispatcher._resolve_
    # timeframe falls back to split_strategy_id when a meta.json declares
    # none, and an unparsed id returns (whole, None, None) - which would admit
    # a promoted Version B with its timeframe unknown rather than checked.
    for sid in (a, b):
        assert split_strategy_id(sid) == ("t3_braid_scalp_20260823", "NQ", "1h")
    assert (version_of_strategy_id(a), version_of_strategy_id(b)) == ("A", "B")

    # FORWARD ONLY. Every id already on disk and in config/portfolios.json was
    # written without a version and must keep its exact spelling.
    plain = strategy_id("t3_braid_scalp_20260823", "NQ", "1h")
    assert plain == "t3_braid_scalp_20260823_NQ_1h"
    assert split_strategy_id(plain) == ("t3_braid_scalp_20260823", "NQ", "1h")
    assert version_of_strategy_id(plain) is None, (
        "an unqualified id is NOT Version A - the name never recorded it, and "
        "meta.json is where that fact lives")

    # "VB" and "B" are the same request; anything else is refused rather than
    # silently producing a directory nobody can parse.
    assert strategy_id("s", "NQ", "1h", "VB") == "s_NQ_1h_VB"
    try:
        strategy_id("s", "NQ", "1h", "C")
    except ValueError:
        pass
    else:                                                    # pragma: no cover
        raise AssertionError("an unknown version must be refused, not "
                             "silently turned into a directory nobody parses")


def test_a_strategy_id_names_the_pair_and_splits_back_apart() -> None:
    """
    The id is the directory under `approved_incubator/`, the id in
    `active_strategies` and `meta.json`'s `name` - one spelling in three
    places, because the live dispatcher builds the second from the first.
    """
    sid = strategy_id("t3_braid_scalp_20260823", "nq", "1H")
    assert sid == "t3_braid_scalp_20260823_NQ_1h"
    assert split_strategy_id(sid) == ("t3_braid_scalp_20260823", "NQ", "1h")


def test_the_split_is_anchored_on_the_timeframe_not_the_underscores() -> None:
    """
    Strategy names here carry underscores AND a date suffix. Counting a fixed
    number of segments from the right turns `ma_anchoring_spread_20260820`
    into a symbol of `spread` at a timeframe of `20260820`.
    """
    assert split_strategy_id("ma_anchoring_spread_20260820") == (
        "ma_anchoring_spread_20260820", None, None)
    assert base_strategy("ma_anchoring_spread_20260820") == \
        "ma_anchoring_spread_20260820"
    # A name that merely contains an underscore is returned whole, so it can
    # never pass as the base of an unrelated strategy.
    assert split_strategy_id("foo_bar") == ("foo_bar", None, None)


def test_a_pair_without_both_halves_stays_a_bare_id() -> None:
    """
    Half an id would split back to a timeframe of None and read as a pair
    whose timeframe nobody recorded. The bare name is the `bt-run` workflow's
    id and is left exactly as it was.
    """
    assert strategy_id("s", "NQ", None) == "s"
    assert strategy_id("s", None, "1h") == "s"
    assert strategy_id("s") == "s"


def test_each_certified_pair_registers_under_its_own_isolated_id() -> None:
    """
    THE BUG THIS EXISTS FOR. Three certified timeframes shared one directory
    and one `active_strategies` entry, so the live loop would have traded
    whichever pair was promoted LAST for all three.
    """
    path = temp_config()
    ids = []
    for tf, quad in (("1h", "Q2"), ("30m", "Q2"), ("15m", "Q1")):
        sid = strategy_id(STRAT, "NQ", tf)
        ids.append(sid)
        register_portfolio(sid, version="A",
                           scope=scope(timeframe=tf, quadrant=quad),
                           config_path=path)
    blob = read(path)["portfolios"]
    holders = {pid: block["active_strategies"]
               for pid, block in blob.items()
               if any(i in (block.get("active_strategies") or []) for i in ids)}
    assert len(holders) == 1, holders
    pid, active = next(iter(holders.items()))
    assert sorted(active) == sorted(ids), active
    records = blob[pid][ALLOCATIONS_KEY]
    # Each id carries ITS OWN pair and quadrant, and its own module path.
    assert records[ids[0]]["timeframe"] == "1h"
    assert records[ids[2]]["timeframe"] == "15m"
    assert records[ids[2]]["regime_filter"] == "Q1"
    assert records[ids[0]]["regime_filter"] == "Q2"
    for sid in ids:
        assert records[sid]["path"].endswith(f"{sid}/strat.py"), records[sid]


def test_the_bare_registration_is_retired_by_the_first_pair_id() -> None:
    """
    The bare entry grants permission to `approved_incubator/<strategy>/`,
    whose meta.json describes ONE of the pairs. Left beside the isolated ids
    it arms a fourth allocation nobody certified and double-sizes that pair.
    """
    def seed(blob):
        block = blob["portfolios"]["Incubator-Even"]
        block["active_strategies"] = [STRAT]
        block[ALLOCATIONS_KEY] = {STRAT: {"strat": STRAT, "symbol": "NQ",
                                          "timeframe": "15m"}}
    path = temp_config(seed)
    sid = strategy_id(STRAT, "NQ", "1h")
    out = register_portfolio(sid, version="A", scope=scope(), config_path=path)
    block = read(path)["portfolios"][out["portfolio_id"]]
    assert STRAT not in block["active_strategies"]
    assert sid in block["active_strategies"]
    assert STRAT not in block[ALLOCATIONS_KEY]
    assert out["retired"] == [STRAT], out["retired"]


def test_retiring_the_bare_id_does_not_swallow_the_new_one() -> None:
    """
    The retirement loop used to REBIND `active_strategies` to a fresh list,
    while `active` still aliased the old object - so the id being registered
    was appended to a list nothing read. The promotion printed a successful
    registration and the routing table granted NOTHING, which is the one
    failure mode where the console and the config disagree.
    """
    def seed(blob):
        blob["portfolios"]["Incubator-Even"]["active_strategies"] = [STRAT]
    path = temp_config(seed)
    sid = strategy_id(STRAT, "NQ", "1h")
    out = register_portfolio(sid, version="A", scope=scope(), config_path=path)
    active = read(path)["portfolios"][out["portfolio_id"]]["active_strategies"]
    assert active == [sid], active


def test_the_certified_quadrant_routes_to_an_account_that_permits_it() -> None:
    """
    An id is ONE pair with ONE quadrant. When the incubator accounts hold
    DISJOINT permissions, balancing purely by headcount sends about half of
    every campaign to an account that forbids the quadrant it was certified
    in, and the live gate stands those down forever - every count still adding
    up, the symptom being silence.

    THE DISJOINT SPLIT IS A FIXTURE. Every portfolio in the live table permits
    all four quadrants since 2026-09-02, so the quadrant no longer narrows the
    choice there and routing correctly falls back to the headcount rule (see
    `test_an_unroutable_quadrant_falls_back_to_the_headcount_rule`). The
    preference itself still has to work the moment any account is narrowed
    again.
    """
    path = temp_config(lambda blob: (
        narrow("Incubator-Even", ["Q1_HIGH_VOL_TREND",
                                  "Q2_HIGH_VOL_CHOP"])(blob),
        narrow("Incubator-Odd", ["Q3_LOW_VOL_TREND",
                                 "Q4_LOW_VOL_MEAN_REVERSION"])(blob)))
    portfolios = read(path)["portfolios"]
    for quad, expected in (("Q1", "Incubator-Even"), ("Q3", "Incubator-Odd")):
        pid, basis = resolve_portfolio(portfolios, None, None, quad)
        assert pid == expected, f"{quad} -> {pid} ({basis})"
        assert f"declares {quad}" in basis


def test_an_unroutable_quadrant_falls_back_to_the_headcount_rule() -> None:
    """
    No account declaring the quadrant is reported, not resolved by widening a
    basket: `regime_quadrants` is the ACCOUNT's permission and every strategy
    on it inherits anything added there.
    """
    portfolios = read(temp_config())["portfolios"]
    pid, basis = resolve_portfolio(portfolios, None, None, "Q9")
    assert pid in incubator_portfolios(portfolios)
    assert "declares Q9" in basis and "could not narrow" in basis, basis


def test_the_certified_symbol_routes_before_the_quadrant() -> None:
    """
    The certified CONTRACT filters the candidate accounts, and it filters
    before the quadrant does.

    The two failures are not equally recoverable. A quadrant mismatch stands
    the strategy down in its own environment and widening
    `basket.regime_quadrants` fixes it in place. A SYMBOL mismatch is terminal:
    `realtime.live_dispatcher.trades_symbol` refuses a strategy on any asset
    its certification does not cover, so an account holding none of its
    contract refuses it on everything it reaches, forever.

    Ranking the quadrant first is what sent three NQ promotions of
    `t3_braid_scalp_20260823` to Incubator-Even - which declares their Q1/Q2
    and trades MES and MGC - correct on quadrant and unable to place an order.
    """
    portfolios = read(temp_config())["portfolios"]

    # Q1 is Incubator-Even's, and NQ is Incubator-Odd's. The contract wins.
    pid, basis = resolve_portfolio(portfolios, None, None, "Q1", "NQ")
    assert pid == "Incubator-Odd", pid
    assert "trades NQ" in basis, basis

    # ...and with no contention the quadrant still narrows within that pool.
    pid, _ = resolve_portfolio(portfolios, None, None, "Q1", "ES")
    assert pid == "Incubator-Even", pid

    # A micro is the same price series as its parent, so either spelling routes.
    assert resolve_portfolio(portfolios, None, None, None, "MNQ")[0] \
        == "Incubator-Odd"


def test_an_unroutable_symbol_falls_back_rather_than_guessing() -> None:
    """
    A contract NO incubator basket carries is REPORTED in the basis and left
    to the headcount rule, not resolved by widening a basket here. The refusal
    belongs to `register_portfolio`, which knows the account it would write to.
    """
    portfolios = read(temp_config())["portfolios"]
    pid, basis = resolve_portfolio(portfolios, None, None, None, "ZS")
    assert pid in incubator_portfolios(portfolios)
    assert "no incubator account trades ZS" in basis, basis


def test_an_automatic_route_into_a_basket_that_cannot_trade_it_is_refused() -> None:
    """
    The blocker itself: a promotion routed AUTOMATICALLY onto an account whose
    basket cannot carry the certified contract is refused, rather than written
    and discovered as silence on a live console.
    """
    path = temp_config()
    try:
        register_portfolio(STRAT, version="A", scope=scope(symbol="NQ"),
                           portfolio="incubator-even", config_path=path)
    except ValueError as exc:                       # explicit: a NOTE, not a refusal
        raise AssertionError(f"an explicit --portfolio must stand: {exc}")

    # Explicit stands, and says so loudly.
    out = register_portfolio(STRAT, version="A", scope=scope(symbol="NQ"),
                             portfolio="incubator-even", config_path=temp_config())
    assert any("certified on NQ" in n for n in out["notes"]), out["notes"]

    # Automatic routing lands on the account that CAN trade it.
    auto = register_portfolio("auto_nq_strategy", version="A",
                              scope=scope(symbol="NQ"),
                              config_path=temp_config())
    assert auto["portfolio_id"] == "Incubator-Odd", auto["portfolio_id"]


def test_an_explicit_portfolio_still_beats_the_quadrant() -> None:
    portfolios = read(temp_config())["portfolios"]
    pid, basis = resolve_portfolio(portfolios, "incubator-odd", None, "Q1")
    assert pid == "Incubator-Odd" and basis == "--portfolio incubator-odd"


# ==========================================================================
def main() -> int:
    cases = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    print(f"stage 5 portfolio registration - {len(cases)} cases\n")
    for name, fn in cases:
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


# ---------------------------------------------------------------------------
# unrouted_packages / the end-of-run card
# ---------------------------------------------------------------------------
def _incubator(tmp_path, packages):
    """`packages` is {package_id: symbol}. Writes a minimal meta.json each."""
    inc = tmp_path / "approved_incubator"
    inc.mkdir(parents=True, exist_ok=True)
    for pid, symbol in packages.items():
        d = inc / pid
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"symbol": symbol}))
    return inc


def _config(tmp_path, active, assets):
    cfg = tmp_path / "portfolios.json"
    cfg.write_text(json.dumps({"portfolios": {"Incubator-Odd": {
        "portfolio_id": "Incubator-Odd",
        "account_type": "incubator_sim",
        "active_strategies": list(active),
        "basket": {"assets": list(assets)}}}}))
    return cfg


def test_unrouted_packages_groups_by_certified_contract(tmp_path):
    inc = _incubator(tmp_path, {"a_PL_1h_VA": "PL", "b_PL_30m_VA": "PL",
                                "c_NQ_1h_VA": "NQ", "d_NQ_1h_VB": "NQ"})
    cfg = _config(tmp_path, active=["d_NQ_1h_VB"], assets=["MNQ"])
    groups = unrouted_packages(cfg, inc)
    assert groups["PL"] == ["a_PL_1h_VA", "b_PL_30m_VA"]
    # d_ is routed, so only c_ is left - and NQ IS carried via MNQ, so it is
    # marked as the re-promotion case rather than pooled with PL.
    assert groups["NQ (carried; re-promote to route)"] == ["c_NQ_1h_VA"]
    assert "d_NQ_1h_VB" not in {p for v in groups.values() for p in v}


def test_the_two_reasons_are_not_pooled(tmp_path):
    """A contract the basket carries needs a re-promotion; one it does not
    needs a decision about accounts. Saying "no basket carries their contract"
    over the first would send the reader to a basket edit already done."""
    inc = _incubator(tmp_path, {"x_NQ_1h_VA": "NQ", "y_PL_1h_VA": "PL"})
    cfg = _config(tmp_path, active=[], assets=["MNQ"])
    groups = unrouted_packages(cfg, inc)
    assert "NQ (carried; re-promote to route)" in groups
    assert "PL" in groups and "PL (carried; re-promote to route)" not in groups


def test_a_directory_without_meta_is_not_an_unrouted_strategy(tmp_path):
    inc = _incubator(tmp_path, {"a_PL_1h_VA": "PL"})
    (inc / "__pycache__").mkdir()
    cfg = _config(tmp_path, active=[], assets=[])
    assert unrouted_packages(cfg, inc) == {"PL": ["a_PL_1h_VA"]}


def test_an_unreadable_config_reports_nothing_rather_than_everything(tmp_path):
    """A missing routing table would make every package look unrouted. An
    empty tally is the honest answer to "I could not read the table"."""
    inc = _incubator(tmp_path, {"a_PL_1h_VA": "PL"})
    assert unrouted_packages(tmp_path / "nope.json", inc) == {}


def test_render_unrouted_is_empty_when_everything_is_routed():
    from backtest.run_pipeline import render_unrouted
    assert render_unrouted({}) == ""


def test_render_unrouted_leads_with_the_biggest_contract():
    from backtest.run_pipeline import render_unrouted
    card = render_unrouted({"NG": ["a", "b"], "PL": ["c", "d", "e"],
                            "SI": ["f"]})
    body = [ln.strip() for ln in card.splitlines() if ln.strip()]
    order = [ln.split()[0] for ln in body
             if ln.split()[0] in ("PL", "NG", "SI")]
    assert order == ["PL", "NG", "SI"], order
    assert "6 promoted package(s) are NOT routed" in card
    assert "--include-unallocated" in card


# --------------------------------------------------------------------------
# The canonical promotion id
#
# TWO WAYS ONE CERTIFIED PAIR ENDED UP WITH TWO IDS, both silent, both fixed
# in `promote()` rather than in `strategy_id` - the bare fallback there is
# correct for the `bt-run` dual-version workflow, which is not scoped to a
# pair and has nothing to name.
#
#   * A promotion carrying an AUDIT FILE whose pair could not be resolved fell
#     back to the bare strategy name. On 2026-09-02 that registered
#     `dual_ema_slope_scalp_20260831` beside the
#     `dual_ema_slope_scalp_20260831_6J_1h_VA` already holding the same
#     certification, with meta.json's top-level symbol/timeframe filled from
#     the MODULE (NQ/ES at 5m) while `certification.*` read 6J at 1h. The live
#     registry then refused BOTH for disagreeing with each other.
#   * Passing an already-qualified `--strat` concatenated a second suffix:
#     `..._6J_1h_VA_6J_1h_VA`.
# --------------------------------------------------------------------------
def test_an_already_qualified_strat_does_not_gain_a_second_suffix() -> None:
    """`--strat` is accepted in either spelling and resolves to one id."""
    bare = "keltner_trend_drift_20260901"
    canonical = strategy_id(bare, "6J", "1h", "A")
    assert canonical == "keltner_trend_drift_20260901_6J_1h_VA"

    # What `promote()` now does to whatever it was handed.
    for spelling in (bare, canonical):
        assert strategy_id(base_strategy(spelling), "6J", "1h", "A") == \
            canonical, f"{spelling} did not normalise to the canonical id"


def test_normalising_a_bare_name_is_a_no_op() -> None:
    """The `bt-run` id has to survive untouched — nothing is renamed."""
    for bare in ("keltner_trend_drift_20260901", "sma_momentum_crossover_20260818",
                 "ma_anchoring_spread_20260820"):
        assert base_strategy(bare) == bare


def test_strategy_id_still_falls_back_to_bare_without_a_pair() -> None:
    """
    The fallback itself is NOT removed. It is the documented `bt-run` id, and
    the fix is that `promote()` refuses to USE it once an audit file is
    supplied — not that the composer stops producing it.
    """
    assert strategy_id("some_strategy", "", "", "A") == "some_strategy"
    assert strategy_id("some_strategy", "NQ", "", "A") == "some_strategy"
    assert strategy_id("some_strategy", "", "1h", "A") == "some_strategy"


def test_a_certified_promotion_resolves_its_pair_through_the_certification():
    """
    THE ORPHAN FIX. `scope_symbol`/`scope_tf` read the audit BODY's top-level
    keys, which `dual_ema_slope_scalp_20260831`'s audit did not carry - so the
    id fell back to the bare name beside the canonical one already holding the
    same certification. `certified_scope` is the authority on that pair (it
    reads the certification block and falls back to the audit FILENAME) and is
    now consulted before giving up.

    IT WARNS RATHER THAN RAISING when the certification genuinely names no
    pair. An unsuffixed `gate_audit_<SYMBOL>.json` from a dual-version
    `bt-run` is a real artifact and promoting from it is documented, so a
    raise there would break a path that predates the bug - it broke
    `tests/test_pipeline_filters.py::test_promote_certification` exactly that
    way. The id stays bare, as it always did, and the note is the visible half
    the duplicate registration never had.
    """
    import inspect
    from backtest import promote as P

    src = inspect.getsource(P.promote)
    assert "certified_scope(certification, None, audit_path)" in src, (
        "the certification is no longer consulted for the pair")
    assert "pair_notes.append" in src, (
        "a bare certified id is registered with no note at all")
    assert "base_strategy(strat)" in src, (
        "the --strat normalisation is gone; an already-qualified id would "
        "gain a second suffix")
