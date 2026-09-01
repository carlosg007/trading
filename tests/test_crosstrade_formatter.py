"""
tests.test_crosstrade_formatter - the strategy tag, on both wire forms.

WHY THIS SUITE EXISTS SEPARATELY FROM `test_regime_daemon.py`
=============================================================
That suite already covers the formatter's validation - the sides it refuses,
the quantities, the tif, the key fallback. This one covers the CrossTrade
STRATEGY LOCK, which is a different kind of property: the tag is not validated
against a list, it is matched by CrossTrade against the tag a previous order
carried. What has to be true is an EQUALITY between two payloads formatted at
different times by different call sites, and the failure it guards against is
not a refused order - it is an accepted flatten that releases nothing and
leaves a live position open with every log line reading correctly.

Assert-based and pytest-shaped: every case fails through `assert`, so
`pytest tests/` and running this file directly report the same thing. There is
no `check()` helper here on purpose - see `tests/conftest.py` for what that
marker does to a suite's results.

EVERY INSTRUMENT HERE IS SPELLED WITH ITS CONTRACT MONTH. `resolve_contract`
returns a qualified instrument before it looks at `config/contracts.json`'s
expiry date, so this suite does not start failing on the roll date for a
reason that has nothing to do with what it tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest                                                     # noqa: E402

from realtime.crosstrade_formatter import (                       # noqa: E402
    CrossTradeFormatError,
    REDACTED,
    format_crosstrade_command,
    format_crosstrade_json,
    format_flatten_command,
    format_flatten_json,
    redact,
    sanitize_strategy_tag,
)

INSTRUMENT = "MES SEP26"
ACCOUNT = "SimIncubator1"
TAG = "keltner_trend_drift"


def _fields(command: str) -> dict[str, str]:
    """The plain-text command as `{name: value}`, in the receiver's own terms."""
    out: dict[str, str] = {}
    for part in command.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition("=")
        out[name.strip()] = value.strip()
    return out


# --------------------------------------------------------------------------
# the tag is appended when given
# --------------------------------------------------------------------------
def test_place_command_carries_the_tag():
    command = format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 1,
                                        key="k", strategy_tag=TAG)
    assert f"strategy_tag={TAG};" in command
    assert _fields(command)["strategy_tag"] == TAG


def test_flatten_command_carries_the_tag():
    command = format_flatten_command(ACCOUNT, INSTRUMENT, key="k",
                                     strategy_tag=TAG)
    assert f"strategy_tag={TAG};" in command
    assert _fields(command)["strategy_tag"] == TAG


def test_the_tag_is_appended_after_the_fields_that_predate_it():
    """
    Field ORDER is part of the plain-text contract - some NinjaTrader add-on
    builds parse positionally when a field is absent - so the new field goes
    at the END and every existing field keeps its place.
    """
    tagged = format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 1,
                                       key="k", strategy_tag=TAG)
    untagged = format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 1,
                                         key="k")
    assert tagged.startswith(untagged)
    assert list(_fields(tagged)) == list(_fields(untagged)) + ["strategy_tag"]


# --------------------------------------------------------------------------
# and omitted, entirely, when it is not
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tag", [None, "", "   "])
def test_no_tag_appends_no_field_at_all(tag):
    """
    NOT `strategy_tag=;`. A field with an empty value is a value: a receiver
    holding a lock on `""` is not a receiver holding no lock, and every
    untagged order on the account would then share one.
    """
    command = format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 1,
                                        key="k", strategy_tag=tag)
    assert "strategy_tag" not in command
    assert command.endswith("tif=DAY;")

    flatten = format_flatten_command(ACCOUNT, INSTRUMENT, key="k",
                                     strategy_tag=tag)
    assert "strategy_tag" not in flatten
    assert flatten.endswith(f"instrument={INSTRUMENT};")


def test_the_default_is_untagged():
    """The parameter is optional, and omitting it changes nothing on the wire."""
    assert "strategy_tag" not in format_crosstrade_command(
        ACCOUNT, INSTRUMENT, "buy", 1, key="k")
    assert "strategy_tag" not in format_flatten_command(ACCOUNT, INSTRUMENT,
                                                        key="k")


# --------------------------------------------------------------------------
# sanitising: the separators cannot survive into the payload
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw, clean", [
    ("keltner trend drift", "keltnertrenddrift"),
    ("a;b", "ab"),
    ("a=b", "ab"),
    ("  padded  ", "padded"),
    ("multi\tline\nid", "multilineid"),
    ("Prop-Even:kt_drift+vwap_fade", "Prop-Even:kt_drift+vwap_fade"),
])
def test_sanitize_strips_the_field_separators(raw, clean):
    assert sanitize_strategy_tag(raw) == clean


@pytest.mark.parametrize("raw", ["a;b", "a=b", "a b"])
def test_a_separator_in_the_tag_never_reaches_the_payload(raw):
    """
    The whole reason to sanitise: an unescaped `;` or `=` is not an odd
    character in a value, it is EXTRA FIELDS the receiver will parse.
    """
    command = format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 1,
                                        key="k", strategy_tag=raw)
    fields = _fields(command)
    assert fields["strategy_tag"] == "ab"
    # ...and nothing new appeared alongside it.
    assert set(fields) == {"key", "command", "account", "instrument", "action",
                           "qty", "order_type", "tif", "strategy_tag"}


@pytest.mark.parametrize("raw", ["; ;", "==", "   ;", "\t="])
def test_a_tag_that_is_nothing_but_separators_is_refused(raw):
    """
    Refused rather than reduced to `""`. A caller that asked for a lock and
    silently got an untagged order is exactly the failure the field exists to
    prevent: the order is placed against no lock, and the exit that expects one
    has nothing to clear.
    """
    with pytest.raises(CrossTradeFormatError):
        sanitize_strategy_tag(raw)
    with pytest.raises(CrossTradeFormatError):
        format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 1,
                                  key="k", strategy_tag=raw)
    with pytest.raises(CrossTradeFormatError):
        format_flatten_command(ACCOUNT, INSTRUMENT, key="k", strategy_tag=raw)


# --------------------------------------------------------------------------
# the lock property: one tag, four payloads, one string
# --------------------------------------------------------------------------
def test_entry_and_exit_carry_the_same_tag():
    """
    THE PROPERTY THE LOCK IS. CrossTrade matches a lock by string equality, so
    an entry and the flatten meant to release it must carry byte-identical
    tags. A flatten that spells the tag differently is accepted, logged, and
    releases nothing.
    """
    entry = format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 1,
                                      key="k", strategy_tag=TAG)
    exit_ = format_flatten_command(ACCOUNT, INSTRUMENT, key="k",
                                   strategy_tag=TAG)
    assert _fields(entry)["strategy_tag"] == _fields(exit_)["strategy_tag"]


def test_both_wire_forms_agree_on_the_tag():
    """
    The text command and the JSON object are different payloads on purpose -
    different casing, different fields - but the tag is a lock key and is the
    SAME string in both. It reached the JSON form through `.strip()` and the
    text form through the sanitiser once, which made `a b` two different locks.
    """
    raw = " keltner trend drift "
    text = _fields(format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 1,
                                             key="k", strategy_tag=raw))
    place = format_crosstrade_json(ACCOUNT, INSTRUMENT, "buy", 1,
                                   strategy_tag=raw)
    flat = format_flatten_json(ACCOUNT, INSTRUMENT, strategy_tag=raw)
    assert text["strategy_tag"] == place["strategy_tag"] == flat["strategy_tag"]
    assert place["strategy_tag"] == "keltnertrenddrift"


def test_the_json_forms_keep_their_empty_tag_field():
    """
    Unlike the text form, the JSON object always DECLARES `strategy_tag` -
    the endpoint's schema has the key - so an absent tag is an empty value
    there and an absent field in the command. That asymmetry is the two
    schemas', not a disagreement about the tag.
    """
    assert format_crosstrade_json(ACCOUNT, INSTRUMENT, "buy", 1
                                  )["strategy_tag"] == ""
    assert format_flatten_json(ACCOUNT, INSTRUMENT)["strategy_tag"] == ""


# --------------------------------------------------------------------------
# the tag must not disturb what was already true
# --------------------------------------------------------------------------
def test_redact_still_removes_the_key_from_a_tagged_command():
    """
    `redact` matches `key=...;` and the tag is appended after it. A tagged
    command is logged like any other, and a key in a log file is a replayable
    credential.
    """
    command = format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 1,
                                        key="SECRET-KEY", strategy_tag=TAG)
    safe = redact(command)
    assert "SECRET-KEY" not in safe
    assert f"key={REDACTED};" in safe
    assert f"strategy_tag={TAG};" in safe


def test_a_tag_does_not_excuse_any_other_validation():
    """The tag is appended last; every field before it is still checked."""
    with pytest.raises(CrossTradeFormatError):
        format_crosstrade_command(ACCOUNT, INSTRUMENT, "flatten", 1,
                                  key="k", strategy_tag=TAG)
    with pytest.raises(CrossTradeFormatError):
        format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 0,
                                  key="k", strategy_tag=TAG)
    with pytest.raises(CrossTradeFormatError):
        format_crosstrade_command(ACCOUNT, INSTRUMENT, "buy", 1,
                                  order_type="limit", strategy_tag=TAG)
    with pytest.raises(CrossTradeFormatError):
        format_flatten_command("", INSTRUMENT, key="k", strategy_tag=TAG)


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
