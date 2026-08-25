"""
The watchdog's Discord alert path.

WHY THIS SUITE EXISTS
=====================
`scripts/watchdog.py` called the notifier as `post_embed(payload, url)`. The
signature is `post_embed(webhook, payload)`. `requests.post` was therefore
handed the payload DICT as its URL and raised `AttributeError: 'dict' object
has no attribute 'decode'` — inside `alert()`'s own bare `except`, which by
design never re-raises, because losing an alert must not lose the check that
produced it.

The result was a watchdog that ran every two minutes, correctly found the feed
stale, printed the verdict, exited 1 — and delivered nothing, ever. Every
symptom pointed at a healthy system: the timer was active, the check was
running, the log had a line in it.

Two more defects rode along and are pinned here too:

  * `build_payload` takes ONE embed, not a list. `build_payload([embed])`
    produces `{"embeds": [[embed]]}`, which Discord rejects with a 400.
  * `post_embed` returns a DICT. `bool(getattr(result, "ok", result))` looks
    for an ATTRIBUTE named `ok`, does not find one, and falls back to the dict
    itself — truthy for every non-empty dict, so a REJECTED post reported
    success and the failure had no second chance to be noticed.

Nothing here touches the network: `post_embed` is replaced with a recorder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backtest.discord_reporter as reporter                       # noqa: E402
import mdlib.env as env                                            # noqa: E402
from scripts import watchdog                                       # noqa: E402

FAKE_URL = "https://discord.example/api/webhooks/1/synthetic-token"


@pytest.fixture
def recorder(monkeypatch):
    """Replace the transport and hand back what it was called with."""
    calls: list[tuple] = []

    def fake_post_embed(webhook, payload):
        calls.append((webhook, payload))
        return {"ok": True, "http_status": 204, "error": None}

    monkeypatch.setattr(reporter, "post_embed", fake_post_embed)
    monkeypatch.setattr(env, "discord_webhook", lambda *a, **k: FAKE_URL)
    return calls


def test_webhook_is_the_first_argument(recorder):
    """The regression itself: URL first, payload second."""
    assert watchdog.alert("regime: stale") is True
    assert len(recorder) == 1
    webhook, payload = recorder[0]
    assert webhook == FAKE_URL, (
        "post_embed(webhook, payload) — reversed, requests.post receives a "
        "dict as its URL and every alert is lost inside alert()'s except")
    assert isinstance(payload, dict)


def test_payload_holds_exactly_one_flat_embed(recorder):
    """`build_payload` takes one embed; a list produces a nested 400."""
    watchdog.alert("feed: no bars")
    _, payload = recorder[0]
    embeds = payload["embeds"]
    assert len(embeds) == 1
    assert isinstance(embeds[0], dict), (
        "build_payload([embed]) nests a list inside embeds[] and Discord "
        "rejects it with a 400")
    assert "feed: no bars" in embeds[0]["description"]


def test_a_rejected_post_is_reported_as_failure(monkeypatch):
    """The truthy-dict bug: a 400 must not read as a delivered alert."""
    monkeypatch.setattr(
        reporter, "post_embed",
        lambda webhook, payload: {"ok": False, "http_status": 400,
                                  "error": "Invalid Form Body"})
    monkeypatch.setattr(env, "discord_webhook", lambda *a, **k: FAKE_URL)
    assert watchdog.alert("regime: stale") is False, (
        "post_embed returns a dict — `getattr(result, 'ok', result)` finds no "
        "attribute and falls back to the truthy dict itself")


def test_no_webhook_configured_is_a_quiet_false(monkeypatch):
    monkeypatch.setattr(env, "discord_webhook", lambda *a, **k: None)
    assert watchdog.alert("regime: stale") is False


def test_a_raising_transport_never_escapes(monkeypatch):
    """Losing the alert must not lose the check that produced it."""
    def boom(webhook, payload):
        raise RuntimeError("socket closed")

    monkeypatch.setattr(reporter, "post_embed", boom)
    monkeypatch.setattr(env, "discord_webhook", lambda *a, **k: FAKE_URL)
    assert watchdog.alert("regime: stale") is False


def test_description_is_truncated_under_the_embed_limit(recorder):
    """Discord caps a description at 4096; the fence adds 8 characters."""
    watchdog.alert("x" * 10_000)
    _, payload = recorder[0]
    assert len(payload["embeds"][0]["description"]) <= 4096
