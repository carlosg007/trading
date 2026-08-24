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

from backtest.promote import (                                    # noqa: E402
    ALLOCATIONS_KEY,
    DEFAULT_ALLOCATION,
    NOT_RESOLVED,
    certified_scope,
    incubator_portfolios,
    register_portfolio,
    resolve_portfolio,
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
    """
    path = temp_config()
    first = register_portfolio("strat_one", version="A", scope=scope(),
                               config_path=path)
    second = register_portfolio("strat_two", version="A", scope=scope(),
                                config_path=path)
    third = register_portfolio("strat_three", version="A", scope=scope(),
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
    register_portfolio(STRAT, version="A", scope=scope(),
                       portfolio="incubator-odd", config_path=path)
    register_portfolio(STRAT, version="A", scope=scope(),
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

    Incubator-Odd trades Q3/Q4. A Q2 certification registered there must be
    reported and must leave the basket alone.
    """
    path = temp_config()
    out = register_portfolio(STRAT, version="A", scope=scope(quadrant="Q2"),
                             portfolio="incubator-odd", config_path=path)
    basket = read(path)["portfolios"]["Incubator-Odd"]["basket"]
    assert basket["regime_quadrants"] == ["Q3_LOW_VOL_TREND",
                                          "Q4_LOW_VOL_MEAN_REVERSION"], basket
    assert any("Q2" in n and "never trade" in n for n in out["notes"]), out["notes"]


def test_a_matching_quadrant_raises_no_note() -> None:
    """Incubator-Even trades Q1/Q2, so a Q2 certification is silent."""
    path = temp_config()
    out = register_portfolio(STRAT, version="A", scope=scope(quadrant="Q2"),
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

    path = temp_config()
    register_portfolio(STRAT, version="A", scope=scope(quadrant="Q2"),
                       portfolio="incubator-odd", config_path=path)
    clear_cache()
    cfg = load_portfolio_config(str(path))
    conflicts = cfg["allocation_reconciliation"]["regime_conflicts"]
    assert len(conflicts) == 1, conflicts
    assert conflicts[0]["strategy_id"] == STRAT
    assert conflicts[0]["regime_filter"] == "Q2"
    assert conflicts[0]["portfolio_quadrants"] == ["Q3", "Q4"]


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
    `portfolio/promotion_daemon.py` moves Incubator -> Prop. Moving only
    `active_strategies` would leave the record on an account that no longer
    holds the strategy, beside one holding it with nothing describing it.
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
    moved = blob["Prop-Odd"][ALLOCATIONS_KEY][STRAT]
    assert moved["status"] == "GRADUATED_PROP", moved["status"]
    assert moved["source_portfolio"] == "Incubator-Odd"
    assert moved["symbol"] == "NQ"


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
