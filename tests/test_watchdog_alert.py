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


# --------------------------------------------------------------------------
# Market hours and alert debouncing
#
# The second failure this module has had: the watchdog fires every two minutes
# under its timer, and `alert()` was called on EVERY cycle a fault was present.
# One stale feed therefore produced 30 identical webhooks an hour until somebody
# fixed it - and across a 49-hour weekend, when the feed is stale because the
# exchange is shut, roughly 1,470 of them. A channel that is paged that often is
# a channel nobody reads, which is the same outcome as the alert never arriving.
# --------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone                 # noqa: E402
from zoneinfo import ZoneInfo                                      # noqa: E402

import scripts.watchdog as wd                                      # noqa: E402

_ET = ZoneInfo("America/New_York")


def _et(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=_ET)


@pytest.mark.parametrize("when,is_open,why", [
    ("2026-08-28 16:59", True,  "Friday before the 17:00 close"),
    ("2026-08-28 17:00", False, "the weekend starts AT 17:00, not after it"),
    ("2026-08-29 09:00", False, "Saturday"),
    ("2026-08-30 17:59", False, "Sunday, one minute before the reopen"),
    ("2026-08-30 18:00", True,  "the reopen is inclusive"),
    ("2026-08-31 17:30", False, "Monday maintenance break"),
    ("2026-08-31 18:00", True,  "maintenance ends AT 18:00"),
    ("2026-09-02 02:00", True,  "Wednesday overnight IS a trading session"),
])
def test_cme_schedule_boundaries(when, is_open, why):
    assert wd.market_status(_et(when))[0] is is_open, why


def test_a_stale_feed_on_saturday_is_not_a_fault(tmp_path, monkeypatch):
    """The whole point: closed-market staleness raises nothing."""
    sent = []
    monkeypatch.setattr(wd, "alert",
                        lambda *a, **k: sent.append(k.get("kind")) or True)
    monkeypatch.setattr(wd, "kill_switch_engaged", lambda: (False, "clear"))
    monkeypatch.setattr(wd, "check_feed", lambda *a, **k: [])
    monkeypatch.setattr(wd, "check_state",
                        lambda p: [{"check": "state", "ok": True, "detail": ""}])
    # A NEW dict per call: main() marks these in place.
    monkeypatch.setattr(wd, "check_regime", lambda *a, **k: [
        {"check": "regime", "ok": False, "detail": "no reading for 3h"}])

    guard = tmp_path / "state.json"
    for _ in range(5):
        rc = wd.main(["--now", "2026-08-29T13:00:00+00:00",
                      "--alert-state-path", str(guard)])
        assert rc == wd.HEALTHY
    assert sent == [], "the weekend must not page anybody"


def test_kill_switch_is_not_silenced_by_the_weekend(tmp_path, monkeypatch):
    """Only STALENESS is suppressed. A real fault is real on a Saturday."""
    monkeypatch.setattr(wd, "alert", lambda *a, **k: True)
    monkeypatch.setattr(wd, "kill_switch_engaged", lambda: (False, "clear"))
    monkeypatch.setattr(wd, "check_feed", lambda *a, **k: [])
    monkeypatch.setattr(wd, "check_regime", lambda *a, **k: [])
    monkeypatch.setattr(wd, "check_state", lambda p: [
        {"check": "state", "ok": False, "detail": "1 UNVERIFIED claim"}])
    rc = wd.main(["--now", "2026-08-29T13:00:00+00:00",
                  "--alert-state-path", str(tmp_path / "s.json")])
    assert rc == wd.DEGRADED


def test_one_alert_on_the_transition_then_silence():
    now = datetime(2026, 9, 2, 14, 0, tzinfo=timezone.utc)
    kind, state = wd.decide_alert({}, True, now)
    assert kind == "transition"
    kind, state = wd.decide_alert(state, True, now + timedelta(minutes=2))
    assert kind is None, "a STATE is not an EVENT; only changes are news"
    kind, state = wd.decide_alert(state, True, now + timedelta(hours=2))
    assert kind is None


def test_a_persistent_fault_reminds_after_the_throttle():
    now = datetime(2026, 9, 2, 14, 0, tzinfo=timezone.utc)
    _, state = wd.decide_alert({}, True, now)
    kind, state = wd.decide_alert(
        state, True, now + timedelta(hours=wd.ALERT_REMINDER_HOURS, minutes=1))
    assert kind == "reminder"
    kind, _ = wd.decide_alert(state, True,
                              now + timedelta(hours=wd.ALERT_REMINDER_HOURS,
                                              minutes=5))
    assert kind is None, "the reminder resets the clock, it does not open a gate"


def test_recovery_is_announced_once():
    now = datetime(2026, 9, 2, 14, 0, tzinfo=timezone.utc)
    _, state = wd.decide_alert({}, True, now)
    kind, state = wd.decide_alert(state, False, now + timedelta(minutes=10))
    assert kind == "recovery", "a fix must be as audible as the fault"
    kind, _ = wd.decide_alert(state, False, now + timedelta(minutes=12))
    assert kind is None


def test_unreadable_state_alerts_rather_than_staying_quiet(tmp_path):
    """Safe direction: one duplicate beats a fault nobody is told about."""
    broken = tmp_path / "s.json"
    broken.write_text("{truncated")
    assert wd.read_alert_state(broken) == {}
    kind, _ = wd.decide_alert(wd.read_alert_state(broken), True,
                              datetime.now(timezone.utc))
    assert kind == "transition"


def test_a_missing_last_alert_time_reminds(tmp_path):
    """An unknown must not be read as 'recently told'."""
    kind, _ = wd.decide_alert({"degraded": True, "last_alert_at": None}, True,
                              datetime.now(timezone.utc))
    assert kind == "reminder"
