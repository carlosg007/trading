"""
tests/test_strategy_tag_manifest.py - `scripts/strategy_tag_manifest.py`.

ASSERT-BASED, so `tests/conftest.py` collects it normally. No `def check(`
marker: that is what routes a suite to the subprocess runner.

WHAT THIS IS GUARDING
=====================
The manifest exists to pre-register CrossTrade locks in a journal. Its ONE
guarantee is that a tag it names is a tag the dispatcher would actually emit -
a manifest that composed tags its own way would have a journal matching on
strings no order ever carries, and the locks would look unregistered forever.

So the cases below pin it to `live_dispatcher.compose_strategy_tag` rather
than to expected strings, and pin the two resolution rules a second
implementation would get wrong: the micro alias (a strategy certified on NQ
contributes to an MNQ position) and the certification check (one certified on
ES does not).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime.live_dispatcher import compose_strategy_tag          # noqa: E402
from scripts.strategy_tag_manifest import (                        # noqa: E402
    build,
    build_parser,
    covers,
    main,
    render,
)


def _manifest() -> dict:
    return build()


# --------------------------------------------------------------------------
# 1. The tags are the dispatcher's, not this script's
# --------------------------------------------------------------------------
def test_every_tag_is_composed_by_the_dispatchers_own_function():
    """
    THE ONE GUARANTEE. A manifest that spelled tags its own way would have the
    journal matching strings no order carries, and every lock would read as
    unregistered.
    """
    for row in _manifest()["rows"]:
        for sid, tag in zip(row["eligible_strategies"],
                            row["singleton_tags"]):
            assert tag == compose_strategy_tag(row["portfolio_id"], [sid])
        if row["eligible_strategies"]:
            assert row["full_set_tag"] == compose_strategy_tag(
                row["portfolio_id"], row["eligible_strategies"])


def test_a_pair_with_no_eligible_strategy_has_no_tag_rather_than_an_empty_one():
    """
    `portfolio:` with nothing after it is a lock on the empty string, shared by
    every untagged order on the account. `None` says "this pair emits nothing",
    which is a real state - an asset in a basket nothing is certified on.
    """
    for row in _manifest()["rows"]:
        if not row["eligible_strategies"]:
            assert row["full_set_tag"] is None
            assert row["singleton_tags"] == []


def test_the_tag_carries_the_portfolio_not_the_account():
    """
    `Incubator-Odd:...`, never `SimIncubator1:...`. The account is where the
    order goes; the portfolio is what the lock is named for, and the two were
    the same string until the NT8 accounts were renamed.
    """
    for row in _manifest()["rows"]:
        for tag in row["singleton_tags"]:
            assert tag.startswith(f"{row['portfolio_id']}:")
            assert not tag.startswith(row["account"])


# --------------------------------------------------------------------------
# 2. Eligibility is the live loop's rule
# --------------------------------------------------------------------------
def test_a_micro_is_covered_by_its_full_size_certification():
    """A strategy certified on NQ contributes to an MNQ position: same price
    series, same tick size, only the multiplier differs."""
    assert covers(["NQ"], "MNQ")
    assert covers(["ES"], "MES")
    assert covers(["MNQ"], "NQ"), "the reverse resolves too"


def test_an_unrelated_certification_does_not_cover():
    """A strategy certified on ES cannot contribute to an MNQ position however
    it is routed, and the live loop refuses exactly that."""
    assert not covers(["ES"], "MNQ")
    assert not covers(["GC"], "6J")
    assert not covers([], "MNQ")


def test_no_strategy_is_listed_for_a_symbol_it_is_not_certified_on():
    """
    Cross-checked against each strategy's own meta.json, because a manifest
    naming a tag the dispatcher would refuse is worse than one that is short.
    """
    from scripts.strategy_tag_manifest import certified_symbols

    for row in _manifest()["rows"]:
        for sid in row["eligible_strategies"]:
            assert covers(certified_symbols(sid), row["symbol"]), (
                f"{sid} listed on {row['symbol']} but certified on "
                f"{certified_symbols(sid)}")


# --------------------------------------------------------------------------
# 3. Shape and coverage
# --------------------------------------------------------------------------
def test_the_manifest_covers_every_account_in_the_routing_table():
    from portfolio.config_loader import clear_cache, load_portfolio_config

    clear_cache()
    cfg = load_portfolio_config()
    expected = {p["target_account"] for p in cfg["portfolios"].values()}
    assert {r["account"] for r in _manifest()["rows"]} == expected


def test_one_row_per_portfolio_and_symbol():
    from portfolio.config_loader import clear_cache, load_portfolio_config

    clear_cache()
    cfg = load_portfolio_config()
    expected = {(pid, asset)
                for pid, p in cfg["portfolios"].items()
                for asset in p["basket"]["assets"]}
    assert {(r["portfolio_id"], r["symbol"])
            for r in _manifest()["rows"]} == expected


def test_the_pattern_is_stated_so_a_journal_matches_on_it():
    """
    Subsets between the singleton and the full set are possible and are not
    enumerated - 16 eligible strategies on one symbol is 65,535 tags. The
    pattern is what a journal should match.
    """
    for row in _manifest()["rows"]:
        assert row["tag_pattern"] == f"{row['portfolio_id']}:<strategy>" \
                                     f"[+<strategy>...]"


def test_the_note_says_a_tag_is_composed_at_dispatch():
    """The manifest is a pre-registration aid, not a claim about what has
    traded. A reader who took it for the latter would think an account with 27
    listed tags had placed 27 kinds of order."""
    manifest = _manifest()
    assert "composed at dispatch" in manifest["note"]
    assert manifest["composer"].endswith("compose_strategy_tag")


# --------------------------------------------------------------------------
# 4. The CLI
# --------------------------------------------------------------------------
def test_the_account_filter_narrows_the_manifest(capsys):
    assert main(["--account", "SimIncubator1"]) == 0
    out = capsys.readouterr().out
    assert "SimIncubator1" in out and "SimIncubator2" not in out


def test_json_output_parses_and_carries_the_rows(capsys):
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] and "generated_utc" in payload


def test_it_writes_a_manifest_when_asked(tmp_path, capsys):
    out = tmp_path / "nested" / "tags.json"
    assert main(["--out", str(out)]) == 0
    capsys.readouterr()
    written = json.loads(out.read_text())
    assert written["rows"] == build()["rows"]


def test_render_survives_a_manifest_with_no_eligible_strategies():
    """Four of the six accounts hold nothing today - the ladder's downstream
    rungs are empty until something is promoted."""
    manifest = {"generated_utc": "t", "composer": "c", "note": "n",
                "rows": [{"portfolio_id": "Eval-Odd", "account": "SimPropSim",
                          "symbol": "MNQ", "eligible_strategies": [],
                          "singleton_tags": [], "full_set_tag": None,
                          "tag_pattern": "Eval-Odd:<strategy>[+<strategy>...]"}]}
    card = render(manifest)
    assert "NO eligible strategy" in card
    assert "SimPropSim/MNQ" in card


def test_the_parser_accepts_the_documented_flags():
    args = build_parser().parse_args(
        ["--account", "Sim101", "--json", "--out", "/tmp/x.json"])
    assert args.account == "Sim101" and args.json is True


# --------------------------------------------------------------------------
# 5. The automated export, and the two callers that must not die on it
# --------------------------------------------------------------------------
def test_export_writes_the_manifest_and_reports_what_it_wrote(tmp_path):
    from scripts.strategy_tag_manifest import export

    out = tmp_path / "nested" / "strategy_tags.json"
    result = export(out)
    assert result["ok"] and result["path"] == str(out)
    written = json.loads(out.read_text())
    assert written["rows"] == build()["rows"]
    assert result["rows"] == len(written["rows"])
    assert result["singleton_tags"] == sum(
        len(r["singleton_tags"]) for r in written["rows"])


def test_export_never_raises_when_the_destination_is_unwritable(tmp_path):
    """
    THE WHOLE POINT OF THE RETURN VALUE. This runs after a promotion has
    already been written and at `master_live` startup. A promotion that
    succeeded and then raised would leave an operator unsure whether the
    strategy was registered; a live loop that refused to start because the NFS
    mount was busy would be down for a journal convenience.
    """
    from scripts.strategy_tag_manifest import export

    blocked = tmp_path / "afile"
    blocked.write_text("not a directory")
    result = export(blocked / "under" / "tags.json")
    assert result["ok"] is False
    assert result["error"], "the reason has to survive, not just the failure"


def test_export_is_atomic_so_a_polling_journal_never_reads_half_a_manifest(
        tmp_path):
    """
    A reader gets the previous complete document or the new one. A truncated
    write parses as a SHORTER strategy list, which reads as strategies having
    been retired rather than as a partial file.
    """
    from scripts.strategy_tag_manifest import export

    out = tmp_path / "tags.json"
    export(out)
    first = out.read_text()
    export(out)
    assert json.loads(out.read_text())["rows"] == json.loads(first)["rows"]
    assert not list(tmp_path.glob("*.tmp")), "the temp file is not left behind"


def test_the_manifest_path_is_read_at_call_time_not_import(monkeypatch,
                                                           tmp_path):
    """Every artifact path in this repository follows this rule: a module
    imported before `.env` loaded would otherwise pin the default forever."""
    from scripts.strategy_tag_manifest import DEFAULT_MANIFEST, manifest_path

    assert str(manifest_path()) == DEFAULT_MANIFEST
    monkeypatch.setenv("BT_STRATEGY_TAGS", str(tmp_path / "elsewhere.json"))
    assert str(manifest_path()) == str(tmp_path / "elsewhere.json")
    assert str(manifest_path(tmp_path / "explicit.json")) == str(
        tmp_path / "explicit.json"), "an explicit path outranks the env"


def test_master_live_refreshes_the_manifest_at_startup():
    """
    The routing table is read ONCE at startup, so that is when the manifest
    can be audited against the config this process actually loaded. A manifest
    generated from a different config pre-registers locks the loop will never
    take out.
    """
    source = (REPO_ROOT / "master_live.py").read_text()
    assert "from scripts.strategy_tag_manifest import export" in source
    assert "--no-tag-manifest" in source, "an operator can opt out"


def test_promote_refreshes_the_manifest_after_a_registration():
    """
    A registration changes which strategies can contribute to a netted
    position and therefore which locks the account can take out — the manifest
    is stale from that moment. It is hooked AFTER the registration and inside
    the `not args.no_register` branch: a staged promotion changes no routing.
    """
    source = (REPO_ROOT / "backtest" / "promote.py").read_text()
    assert "from scripts.strategy_tag_manifest import export" in source
    hook = source.index("from scripts.strategy_tag_manifest import export")
    assert source.index("registration = register_portfolio(") < hook, (
        "the manifest is regenerated from a routing table that must already "
        "carry the new registration")
