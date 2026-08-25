#!/usr/bin/env python3
"""
test_nt8_feed.py — live bars from the broker: that an NT8 bar lands on the
timestamp this repository means, that a bar with no timezone is refused rather
than guessed at, and that a missing publisher is loud instead of quiet.

Location:  ~/src/trading/tests/test_nt8_feed.py

Run EITHER way — both report the same answer:

    OMP_NUM_THREADS=1 python tests/test_nt8_feed.py
    /home/cgrullon/src/trading/.venv/bin/pytest tests/test_nt8_feed.py

EVERY CASE FAILS THROUGH `assert`. Nothing here touches NinjaTrader or a
network: each case writes a spool file into `tmp_path` in the documented
format, which is also the only way to test this today — the NinjaScript
publisher runs on a Windows box and
`/mnt/backtest/artifacts/nt8_bars/` has never held a file.

WHAT THIS COVERS, and why each case is here rather than assumed:

  * **THE STAMP CONVENTION, WHICH IS THE ONE THAT SILENTLY RUINS EVERYTHING.**
    NinjaTrader stamps a bar with its CLOSE; the lake, every certification and
    every indicator in this repository stamp the OPEN. Ingested as-is, an
    hourly bar stamped 17:00 reads as the bar STARTING at 17:00 — every bar
    shifted a period, every indicator computed on misaligned data, and not one
    error anywhere. The conversion is tested directly, and so is the header
    that lets a script declare it writes the other way.
  * **A NAIVE TIMESTAMP IS REFUSED.** NT8 writes in the instrument's or the
    workstation's timezone unless the script converts. Assumed to be UTC and
    wrong, the whole series shifts by hours and still looks like a market:
    bars in order, prices sane, sessions the wrong length.
  * **A MISSING PUBLISHER IS NOT A QUIET MARKET.** No spool directory raises;
    a directory with no file for one symbol leaves that symbol absent, which
    the loop reports differently.
  * **A WRITTEN BAR IS NOT TRUSTED TO BE A CLOSED BAR.** The script is meant to
    append on close, but the frame still goes through the one forming-bar rule
    — a script switched to `Calculate.OnEachTick` would otherwise start
    appending live bars and nothing would notice.
  * **DATABENTO IS OUT OF THE LIVE PATH.** Asserted on the import graph, not
    on a promise: live bars come from the broker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime.feed import BAR_COLUMNS, FeedError, resolve_feed     # noqa: E402
from realtime.nt8_feed import (DEFAULT_STAMP,                      # noqa: E402
                               NT8BarFeed,
                               NT8FeedError,
                               availability,
                               read_spool,
                               spool_path)

HEADER = "ts,open,high,low,close,volume\n"


def write_spool(directory: Path, symbol: str, tf: str, rows: list[str],
                header: str | None = None) -> Path:
    """One spool file in the documented format."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{symbol}_{tf}.csv"
    path.write_text((f"# {header}\n" if header else "") + HEADER + "".join(rows),
                    encoding="utf-8")
    return path


def bar(ts: str, close: float = 100.0, volume: int = 10) -> str:
    return f"{ts},{close - 0.5},{close + 1.0},{close - 1.0},{close},{volume}\n"


# --------------------------------------------------------------------------
# 1. the stamp convention
# --------------------------------------------------------------------------

def test_an_nt8_close_stamped_bar_lands_on_its_open(tmp_path: Path) -> None:
    """
    THE CASE THIS MODULE EXISTS FOR.

    NinjaTrader stamps the bar that ran 16:00–17:00 as `17:00`. This
    repository stamps it `16:00` — the lake resamples `label="left"` and every
    certification was computed that way. One subtraction on ingest, or every
    bar in the system is off by one period with nothing raising.
    """
    path = write_spool(tmp_path, "MNQ", "1h", [
        bar("2026-08-25T16:00:00Z", 100.0),      # ran 15:00-16:00
        bar("2026-08-25T17:00:00Z", 101.0),      # ran 16:00-17:00
    ])
    bars, convention = read_spool(path, "1h", "close")

    assert convention == "close"
    assert list(bars["ts"].astype(str)) == ["2026-08-25 15:00:00+00:00",
                                            "2026-08-25 16:00:00+00:00"]
    # The prices ride with their own bar, not with the timestamp they moved to.
    assert bars.loc[bars["ts"] == pd.Timestamp("2026-08-25 16:00", tz="UTC"),
                    "close"].iloc[0] == 101.0


def test_a_file_may_declare_that_it_stamps_the_open(tmp_path: Path) -> None:
    """The NinjaScript is the thing that knows which end it writes, and an
    operator who changes it should not have to remember a flag on another
    machine."""
    path = write_spool(tmp_path, "MNQ", "1h",
                       [bar("2026-08-25T16:00:00Z")], header="stamp=open")
    bars, convention = read_spool(path, "1h", "close")

    assert convention == "open"
    assert bars["ts"].iloc[0] == pd.Timestamp("2026-08-25 16:00", tz="UTC")


def test_the_subtraction_follows_the_timeframe(tmp_path: Path) -> None:
    """A 15m close-stamped bar moves back 15 minutes, not an hour. Hard-coding
    the shift would be right on one timeframe and wrong on the rest."""
    path = write_spool(tmp_path, "MNQ", "15m", [bar("2026-08-25T16:15:00Z")])
    bars, _ = read_spool(path, "15m", "close")
    assert bars["ts"].iloc[0] == pd.Timestamp("2026-08-25 16:00", tz="UTC")


def test_an_unknown_stamp_convention_is_refused(tmp_path: Path) -> None:
    path = write_spool(tmp_path, "MNQ", "1h", [bar("2026-08-25T16:00:00Z")])
    with pytest.raises(NT8FeedError, match="stamp must be one of"):
        read_spool(path, "1h", "middle")

    declared = write_spool(tmp_path, "MES", "1h",
                           [bar("2026-08-25T16:00:00Z")], header="stamp=eod")
    with pytest.raises(NT8FeedError, match="stamp="):
        read_spool(declared, "1h", "close")


# --------------------------------------------------------------------------
# 2. timezones
# --------------------------------------------------------------------------

def test_a_timestamp_with_no_offset_is_refused_not_assumed(
        tmp_path: Path) -> None:
    """
    Assumed to be UTC and wrong, the series shifts by hours and still looks
    like a market — bars in order, prices sane, sessions the wrong length in a
    way nobody reads off a chart.
    """
    path = write_spool(tmp_path, "MNQ", "1h", [bar("2026-08-25 16:00:00")])
    with pytest.raises(NT8FeedError, match="no UTC offset"):
        read_spool(path, "1h", "close")


def test_an_explicit_offset_is_converted_not_relabelled(
        tmp_path: Path) -> None:
    """A bar stamped 12:00-04:00 closed at 16:00 UTC and therefore opened at
    15:00 UTC. Relabelling instead of converting would move it four hours."""
    path = write_spool(tmp_path, "MNQ", "1h", [bar("2026-08-25T12:00:00-04:00")])
    bars, _ = read_spool(path, "1h", "close")
    assert bars["ts"].iloc[0] == pd.Timestamp("2026-08-25 15:00", tz="UTC")


# --------------------------------------------------------------------------
# 3. the feed
# --------------------------------------------------------------------------

def test_a_missing_spool_directory_raises_rather_than_reading_empty(
        tmp_path: Path) -> None:
    """No publisher is not a quiet market, and a loop told the difference can
    say so on the console instead of printing "no signal" forever."""
    feed = NT8BarFeed(spool_dir=tmp_path / "never_created")
    with pytest.raises(NT8FeedError, match="no NT8 bar spool"):
        feed.closed_bars(["MNQ"], "1h", 10)

    ok, why = availability(tmp_path / "never_created")
    assert ok is False and "not installed" in why


def test_a_symbol_with_no_spool_is_absent_not_empty(tmp_path: Path) -> None:
    spool = tmp_path / "nt8_bars"
    write_spool(spool, "MNQ", "1h", [bar("2026-08-25T16:00:00Z")])
    feed = NT8BarFeed(spool_dir=spool)

    bars, sources = feed.closed_bars(["MNQ", "MGC"], "1h", 10)
    assert "MGC" not in bars and "MGC" not in sources


def test_the_feed_drops_a_bar_the_script_wrote_before_it_closed(
        tmp_path: Path, monkeypatch) -> None:
    """
    The publisher is MEANT to append on bar close. It is not trusted to.

    A NinjaScript switched to `Calculate.OnEachTick` starts appending the bar
    the market is printing into, and every frame goes through the one
    forming-bar rule precisely so that nothing notices it downstream — the bar
    is dropped here instead.
    """
    spool = tmp_path / "nt8_bars"
    write_spool(spool, "MNQ", "1h", [
        bar("2026-08-25T16:00:00Z", 100.0),      # opened 15:00, closed
        bar("2026-08-25T18:00:00Z", 102.0),      # opened 17:00, STILL OPEN
    ])
    import realtime.feed as feed_module
    monkeypatch.setattr(
        feed_module, "datetime",
        type("D", (), {"now": staticmethod(
            lambda tz=None: pd.Timestamp("2026-08-25 17:30", tz="UTC"))}))

    feed = NT8BarFeed(spool_dir=spool)
    bars, _ = feed.closed_bars(["MNQ"], "1h", 10)

    assert bars["MNQ"]["ts"].iloc[-1] == pd.Timestamp("2026-08-25 15:00",
                                                      tz="UTC")
    assert feed.last_forming["MNQ"] == pd.Timestamp("2026-08-25 17:00",
                                                    tz="UTC")


def test_a_minute_spool_serves_a_higher_timeframe(tmp_path: Path) -> None:
    """A publisher writing only minutes still serves a 15m strategy, and the
    aggregation is the lake's own — so the bar is the one the certification
    was computed on."""
    spool = tmp_path / "nt8_bars"
    rows = [bar(f"2026-08-25T{15 + (i + 1) // 60:02d}:{(i + 1) % 60:02d}:00Z",
                100.0 + i) for i in range(30)]
    write_spool(spool, "MNQ", "1m", rows)

    feed = NT8BarFeed(spool_dir=spool)
    bars, sources = feed.closed_bars(["MNQ"], "15m", 10)

    frame = bars["MNQ"]
    assert sources["MNQ"] == "MNQ 1m", "the substitution is reported"
    assert frame["ts"].iloc[0] == pd.Timestamp("2026-08-25 15:00", tz="UTC")
    assert frame["open"].iloc[0] == 99.5          # first minute's open
    assert frame["close"].iloc[0] == 114.0        # fifteenth minute's close
    assert frame["volume"].iloc[0] == 150         # 15 bars x 10


def test_a_spool_of_the_full_size_contract_answers_for_its_micro(
        tmp_path: Path) -> None:
    """NT8 may publish NQ while the basket trades MNQ. Same price series, the
    same alias every other tier resolves through."""
    spool = tmp_path / "nt8_bars"
    write_spool(spool, "NQ", "1h", [bar("2026-08-25T16:00:00Z")])
    assert spool_path("MNQ", "1h", spool).name == "NQ_1h.csv"

    feed = NT8BarFeed(spool_dir=spool)
    bars, _ = feed.closed_bars(["MNQ"], "1h", 10)
    # The order is still for the micro; only the tape is the parent's.
    assert set(bars["MNQ"]["symbol"]) == {"MNQ"}


def test_the_frames_are_the_canonical_shape(tmp_path: Path) -> None:
    spool = tmp_path / "nt8_bars"
    write_spool(spool, "MNQ", "1h", [bar(f"2026-08-25T{h:02d}:00:00Z")
                                     for h in range(10, 17)])
    bars, _ = NT8BarFeed(spool_dir=spool).closed_bars(["MNQ"], "1h", 10)
    frame = bars["MNQ"]

    assert list(frame.columns) == list(BAR_COLUMNS)
    assert str(frame["ts"].dt.tz) == "UTC"
    assert frame["ts"].is_monotonic_increasing


# --------------------------------------------------------------------------
# 4. the separation itself
# --------------------------------------------------------------------------

def test_the_live_path_does_not_import_databento() -> None:
    """
    Asserted on the source, not on a promise.

    Live bars come from the broker so that the bars a strategy decides on are
    the bars its orders execute against. A vendor import creeping back into
    `realtime/` is how that quietly stops being true.
    """
    for module in ("feed.py", "nt8_feed.py", "live_dispatcher.py",
                   "regime_daemon.py"):
        text = (REPO_ROOT / "realtime" / module).read_text(encoding="utf-8")
        code = "\n".join(line for line in text.splitlines()
                         if not line.lstrip().startswith("#"))
        assert "import databento" not in code, module
        assert "databento_live" not in code, module

    # The word may legitimately appear in prose that explains the split; what
    # must not appear anywhere in the live path is a way to CALL it.
    master = (REPO_ROOT / "master_live.py").read_text(encoding="utf-8")
    assert "import databento" not in master
    assert "databento_live" not in master
    assert "data_pull" not in master


def test_live_resolves_to_the_broker_feed_and_refuses_without_it(
        monkeypatch, tmp_path: Path) -> None:
    """`live` is a spelling of `nt8`; there is one live feed and it is the
    broker's. Neither falls back — an operator who asked for live bars and got
    history would read a rehearsal as a session."""
    import realtime.feed as feed_module
    monkeypatch.setattr(feed_module, "live_feed_available",
                        lambda: (False, "no spool"))
    for mode in ("nt8", "live"):
        with pytest.raises(FeedError, match="Refusing to fall back"):
            resolve_feed(mode)

    monkeypatch.setattr(feed_module, "live_feed_available",
                        lambda: (True, "spooling"))
    feed = resolve_feed("live", spool_dir=tmp_path)
    assert isinstance(feed, NT8BarFeed)
    assert DEFAULT_STAMP == "close", "NT8's own convention is the default"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
