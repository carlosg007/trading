"""
tests/test_regime_cli.py - `scripts/check_market_regime.py`.

ASSERT-BASED, so `tests/conftest.py` collects it normally and every case keeps
its own granularity. There is deliberately no `def check(` helper in this file:
that marker is what routes a suite to the subprocess runner, and a suite that
recorded results in a list instead of asserting would report green while
failing.

No case here touches the lake, a mount, a broker or the live state file. Bars
come from `FakeFeed` and every theta anchor is one the test wrote, so a case
that should fail for a missing anchor cannot pass because the box happens to
have NQ cached.

WHAT THIS SUITE IS ACTUALLY GUARDING
====================================
The script is a READER of the regime engine, so the interesting failures are
not "is the ADX right" - `mdlib.regimes` owns that and
`tests/test_regime_daemon.py` tests it. They are:

  * the card inventing a quadrant, or renaming the four that exist
  * Q0 being reported as though it were a fifth quadrant
  * the (symbol, TIMEFRAME) join pooling two timeframes into one bucket
  * one symbol's missing anchor taking the whole card down
  * the card WRITING to `data/live_regime_state.json` and racing the daemon
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime.regime_daemon import (                               # noqa: E402
    MasterRegimeDaemon,
    STATUS_ACTIVE,
    STATUS_MUTED,
    UNDEFINED_LABEL,
)
from scripts.check_market_regime import (                          # noqa: E402
    EMA_SLOW,
    TREND_BIAS_DEADBAND_PCT,
    VOLUME_Z_PERIOD,
    build_parser,
    collect,
    main,
    render,
    strategies_for,
    supporting_metrics,
)

CONFIG = str(REPO_ROOT / "config" / "portfolios.json")

# Sits between the two fixtures' ATRs (0.75 and 8.00), so "high volatility" is
# decided by the anchor rather than by luck.
FIXTURE_THETA = 2.0


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def make_bars(n: int, kind: str, tf_minutes: int = 15,
              volume: str = "flat") -> pd.DataFrame:
    """
    Deterministic bars in the shape `realtime.feed` yields.

    Same two shapes `tests/test_regime_daemon.py` uses, so a quadrant asserted
    there and a quadrant asserted here mean the same thing. No randomness: a
    regime test whose fixture moves is a test that will one day fail for a
    reason nobody can reproduce.
    """
    ts = pd.date_range("2026-08-01", periods=n, freq=f"{tf_minutes}min",
                       tz="UTC")
    if kind == "trend":
        close = 15000.0 + np.arange(n) * 5.0
        high, low = close + 3.0, close - 2.0
    elif kind == "chop":
        close = 15000.0 + np.where(np.arange(n) % 2 == 0, 0.25, -0.25)
        high, low = close + 0.25, close - 0.25
    else:
        raise ValueError(kind)

    if volume == "flat":
        vol = np.full(n, 100.0)
    elif volume == "spike":
        vol = np.full(n, 100.0)
        vol[-1] = 1000.0            # a clean, large final-bar z-score
    else:
        raise ValueError(volume)

    return pd.DataFrame({"ts": ts, "open": close, "high": high, "low": low,
                         "close": close, "volume": vol})


class FakeFeed:
    """
    A `realtime.feed.BarFeed` stand-in.

    `frames` is keyed `(symbol, tf)`. A pair that is absent is ABSENT from the
    result rather than present and empty, exactly as the real feed behaves -
    "no bars for this symbol" and "this symbol has no signal" are different
    statements and the card reports them differently.
    """

    def __init__(self, frames: dict, raise_on_tf: str | None = None):
        self.frames = frames
        self.raise_on_tf = raise_on_tf
        self.calls: list[tuple] = []

    def describe(self) -> str:
        return "fake feed"

    def closed_bars(self, symbols, tf, lookback_bars=500):
        self.calls.append((tuple(symbols), tf, lookback_bars))
        if self.raise_on_tf == tf:
            raise RuntimeError(f"synthetic feed failure at {tf}")
        bars, sources = {}, {}
        for symbol in symbols:
            frame = self.frames.get((symbol, tf))
            if frame is None:
                continue
            bars[symbol] = frame.tail(lookback_bars).reset_index(drop=True)
            sources[symbol] = symbol
        return bars, sources


def make_daemon(tmp_path: Path, anchors: dict | None = None,
                state_file: str | None = None) -> MasterRegimeDaemon:
    """
    A daemon wired entirely to `tmp_path`.

    `cache_root` is an empty directory so the REAL regime caches on
    /mnt/backtest cannot supply an anchor behind the test's back.
    """
    if anchors is None:
        anchors = {"NQ": {tf: {"theta_vol": FIXTURE_THETA,
                               "is_start": "2013-01-01",
                               "is_end": "2022-12-31"}
                          for tf in ("5m", "15m", "30m", "1h")}}
    anchors_path = tmp_path / "theta_vol_anchors.json"
    anchors_path.write_text(json.dumps(anchors, indent=2))
    empty_cache = tmp_path / "no_cache"
    empty_cache.mkdir(parents=True, exist_ok=True)
    return MasterRegimeDaemon(
        config_path=CONFIG,
        state_file=str(state_file or tmp_path / "state" / "live.json"),
        ml_model_dir=str(tmp_path / "models"),
        anchors_path=str(anchors_path),
        cache_root=str(empty_cache),
        strict_config=False)


# --------------------------------------------------------------------------
# 1. It classifies, and it classifies through the engine
# --------------------------------------------------------------------------
def test_a_trending_high_vol_tape_is_reported_as_q1(tmp_path):
    """ATR 8.00 against a theta of 2.00, with ADX pinned high, is Q1."""
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "trend")})
    snap = collect(symbols=["NQ"], timeframes=["15m"], feed=feed,
                   daemon=make_daemon(tmp_path))

    assert len(snap["rows"]) == 1
    row = snap["rows"][0]
    assert row["error"] is None
    assert row["quadrant"] == "Q1"
    assert row["is_high_vol"] is True
    assert row["is_trending"] is True
    assert row["regime_name"] == "High Volatility / Trending"


def test_a_quiet_choppy_tape_is_reported_as_q4(tmp_path):
    """ATR 0.75 against the same theta, ADX collapsed, is Q4."""
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "chop")})
    snap = collect(symbols=["NQ"], timeframes=["15m"], feed=feed,
                   daemon=make_daemon(tmp_path))

    row = snap["rows"][0]
    assert row["quadrant"] == "Q4"
    assert row["is_high_vol"] is False
    assert row["is_trending"] is False


def test_the_card_reports_only_the_four_canonical_quadrants(tmp_path):
    """
    THE WHOLE POINT OF THE FILE. `config/portfolios.json` 1.0.0 shipped a
    second quadrant numbering once, where every digit named a different
    environment, and a strategy stood down in the regime it was certified for
    while every log line read correctly. A card that renamed the axes - "Bull
    Trend / High Vol" - would be that same second vocabulary.
    """
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "trend"),
                     ("NQ", "1h"): make_bars(120, "chop", tf_minutes=60)})
    snap = collect(symbols=["NQ"], timeframes=["15m", "1h"], feed=feed,
                   daemon=make_daemon(tmp_path))

    for row in snap["rows"]:
        assert row["quadrant"] in {"Q0", "Q1", "Q2", "Q3", "Q4"}
        assert "Bull" not in str(row.get("regime_name"))
        assert "Bear" not in str(row.get("regime_name"))

    card = render(snap)
    assert "VOLATILITY x TREND, not direction" in card
    assert "Bull" not in card and "Bear" not in card


def test_the_warmup_is_q0_and_q0_is_not_offered_as_a_quadrant(tmp_path):
    """
    Fewer than 2*ADX_LENGTH+1 bars is a warm-up. `NaN > 25.0` is False, so a
    naive encoding files every warm-up bar under Low-Vol/Ranging - a populated
    label on bars where no indicator exists.
    """
    feed = FakeFeed({("NQ", "15m"): make_bars(10, "trend")})
    snap = collect(symbols=["NQ"], timeframes=["15m"], feed=feed,
                   daemon=make_daemon(tmp_path))

    row = snap["rows"][0]
    assert row["quadrant"] == "Q0"
    assert row["regime"] == UNDEFINED_LABEL
    assert row["adx_14"] is None, "no indicator exists inside the warm-up"
    assert row["atr_14"] is None
    assert row["quadrant"] != "Q4", "Q0 must not collapse into Low-Vol/Ranging"


# --------------------------------------------------------------------------
# 2. It stays read-only
# --------------------------------------------------------------------------
def test_the_card_never_writes_the_live_state_file(tmp_path):
    """
    A diagnostic that can move the file every other tier reads is not a
    diagnostic. `update_state` would also race the live daemon: both would be
    doing a read-modify-write of one JSON document.
    """
    state_file = tmp_path / "state" / "live_regime_state.json"
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "trend")})
    snap = collect(symbols=["NQ"], timeframes=["15m"], feed=feed,
                   daemon=make_daemon(tmp_path, state_file=str(state_file)))

    assert snap["rows"][0]["quadrant"] == "Q1", "it did classify"
    assert not state_file.exists(), "the card published to the live state file"


def test_the_synthetic_state_is_shaped_the_way_the_daemon_publishes(tmp_path):
    """
    The switchboard resolves a reading through `_published_record`, which keys
    on `by_timeframe[symbol][tf]`. A snapshot shaped any other way would
    evaluate every strategy against "no regime published" and mute all of
    them - which on a card looks exactly like a market nobody is certified for.
    """
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "trend")})
    snap = collect(symbols=["NQ"], timeframes=["15m"], feed=feed,
                   daemon=make_daemon(tmp_path))

    state = snap["state"]
    assert state["by_timeframe"]["NQ"]["15m"]["quadrant"] == "Q1"
    assert state["symbols"]["NQ"]["quadrant"] == "Q1"
    assert state["by_timeframe"]["NQ"]["15m"]["tf"] == "15m"


# --------------------------------------------------------------------------
# 3. One broken pair does not take the card down
# --------------------------------------------------------------------------
def test_a_symbol_without_an_anchor_is_a_row_not_an_exception(tmp_path):
    """
    The card is most wanted when something is already broken, so a missing
    anchor for ES must not cost the operator NQ's reading.
    """
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "trend"),
                     ("ES", "15m"): make_bars(120, "trend")})
    snap = collect(symbols=["NQ", "ES"], timeframes=["15m"], feed=feed,
                   daemon=make_daemon(tmp_path))   # anchors: NQ only

    rows = {r["symbol"]: r for r in snap["rows"]}
    assert rows["NQ"]["quadrant"] == "Q1"
    assert rows["ES"]["error"] is not None
    assert "theta_vol" in rows["ES"]["error"]
    assert "quadrant" not in rows["ES"], "a failed row must not carry a label"


def test_a_symbol_the_feed_cannot_answer_for_is_reported_as_such(tmp_path):
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "trend")})
    snap = collect(symbols=["NQ", "ES"], timeframes=["15m"], feed=feed,
                   daemon=make_daemon(tmp_path))

    rows = {r["symbol"]: r for r in snap["rows"]}
    assert rows["ES"]["error"] == "no closed bars"


def test_a_feed_failure_on_one_timeframe_leaves_the_others_standing(tmp_path):
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "trend"),
                     ("NQ", "1h"): make_bars(120, "trend", tf_minutes=60)},
                    raise_on_tf="1h")
    snap = collect(symbols=["NQ"], timeframes=["15m", "1h"], feed=feed,
                   daemon=make_daemon(tmp_path))

    assert [r["tf"] for r in snap["rows"]] == ["15m"]
    assert any("1h" in e and "synthetic feed failure" in e
               for e in snap["errors"])


def test_render_survives_a_snapshot_in_which_everything_failed(tmp_path):
    """The card has to print when nothing worked; that is when it is read."""
    feed = FakeFeed({})
    snap = collect(symbols=["NQ"], timeframes=["15m"], feed=feed,
                   daemon=make_daemon(tmp_path))
    card = render(snap, show_strategies=True)
    assert "no closed bars" in card


# --------------------------------------------------------------------------
# 4. The (symbol, TIMEFRAME) join
# --------------------------------------------------------------------------
def _snap_with_strategies() -> dict:
    """Two strategies on NQ that differ ONLY in the timeframe."""
    return {
        "strategies": {
            "nq_15m_strat": {"symbol": "NQ", "timeframe": "15m",
                             "status": STATUS_ACTIVE},
            "nq_1h_strat": {"symbol": "NQ", "timeframe": "1h",
                            "status": STATUS_MUTED},
            "es_15m_strat": {"symbol": "ES", "timeframe": "15m",
                             "status": STATUS_ACTIVE},
        }
    }


def test_a_reading_answers_only_its_own_timeframe():
    """
    theta_vol is per (symbol, TIMEFRAME) - NQ's is 7.90 at 15m and 11.33 at
    30m, a 43% different boundary on the same tape. Pooling both into one "NQ"
    bucket would credit a strategy with a quadrant match drawn against a
    boundary it was never certified on.
    """
    snap = _snap_with_strategies()

    active, muted = strategies_for(snap, "NQ", "15m")
    assert active == ["nq_15m_strat"]
    assert muted == []

    active, muted = strategies_for(snap, "NQ", "1h")
    assert active == []
    assert muted == ["nq_1h_strat"], "the 1h strategy, not the 15m one"


def test_the_join_does_not_leak_across_symbols():
    snap = _snap_with_strategies()
    active, muted = strategies_for(snap, "NQ", "15m")
    assert "es_15m_strat" not in active + muted


def test_a_micro_is_answered_by_its_full_size_parents_certification():
    """
    A strategy certified on NQ and routed to a basket of MNQ must be found:
    same price series, same tick size, only the multiplier differs - and a
    multiplier appears nowhere in an ADX or an ATR.
    """
    snap = _snap_with_strategies()
    active, _ = strategies_for(snap, "MNQ", "15m")
    assert active == ["nq_15m_strat"]


def test_every_registered_strategy_is_attributed_to_exactly_one_row(tmp_path):
    """
    The per-row Active/Muted counts must reconcile with the switchboard total.
    A strategy counted twice, or dropped between the join and the summary,
    makes the card's headline number a different claim from its table.
    """
    feed = FakeFeed({("NQ", tf): make_bars(300, "trend", tf_minutes=m)
                     for tf, m in (("5m", 5), ("15m", 15),
                                   ("30m", 30), ("1h", 60))})
    snap = collect(symbols=["NQ"], timeframes=["5m", "15m", "30m", "1h"],
                   feed=feed, daemon=make_daemon(tmp_path))

    seen: list[str] = []
    for row in snap["rows"]:
        if row.get("error"):
            continue
        active, muted = strategies_for(snap, row["symbol"], row["tf"])
        seen.extend(active + muted)

    assert len(seen) == len(set(seen)), "a strategy was counted on two rows"
    nq = [sid for sid, s in snap["strategies"].items()
          if str(s.get("symbol")) == "NQ"]
    assert set(seen) == set(nq)


# --------------------------------------------------------------------------
# 5. Supporting metrics are diagnostics
# --------------------------------------------------------------------------
def test_norm_atr_rescales_the_engines_atr_and_never_recomputes_it():
    """
    `norm_atr` must be the ENGINE's ATR over the close, not a second ATR. A
    second implementation is how a cached number and a live one come to
    disagree with nothing raising.
    """
    bars = make_bars(120, "trend")
    metrics = supporting_metrics(bars, atr_14=8.0)
    assert metrics["norm_atr"] == pytest.approx(8.0 / metrics["close"])


def test_a_metric_whose_window_is_not_filled_is_none_rather_than_a_number():
    """
    A ribbon slope computed over 40 bars in a column headed EMA(200) is a
    number that looks like a measurement and is not one.
    """
    short = supporting_metrics(make_bars(40, "trend"), atr_14=8.0)
    assert short["ribbon_ratio"] is None
    assert short["ribbon_slope"] is None

    long = supporting_metrics(make_bars(EMA_SLOW + 50, "trend"), atr_14=8.0)
    assert long["ribbon_ratio"] is not None
    assert long["ribbon_slope"] is not None


def test_volume_z_is_none_inside_its_own_window_and_a_number_after_it():
    assert supporting_metrics(make_bars(VOLUME_Z_PERIOD - 1, "trend"),
                              atr_14=8.0)["volume_z"] is None
    spike = supporting_metrics(make_bars(60, "trend", volume="spike"),
                               atr_14=8.0)
    assert spike["volume_z"] > 1.0, "a 10x final bar is a high z-score"


def test_a_dead_flat_volume_window_does_not_divide_by_zero():
    """The 1e-8 guard, spelled as the promoted feature matrices spell it."""
    flat = supporting_metrics(make_bars(60, "trend", volume="flat"),
                              atr_14=8.0)
    assert flat["volume_z"] == pytest.approx(0.0, abs=1e-6)


def test_trend_bias_is_directional_and_is_not_the_quadrant(tmp_path):
    """
    Direction is a fair thing to want and it is NOT a quadrant. It is reported
    in its own column, from ROC, and gates nothing.
    """
    up = supporting_metrics(make_bars(120, "trend"), atr_14=8.0)
    assert up["trend_bias"] == "UP"

    flat = supporting_metrics(make_bars(120, "chop"), atr_14=0.75)
    assert flat["trend_bias"] == "FLAT", "a 0.25-point alternation is not a trend"

    # And the two axes are independent: a Q1 tape is high-vol AND trending,
    # which says nothing about the sign.
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "trend")})
    snap = collect(symbols=["NQ"], timeframes=["15m"], feed=feed,
                   daemon=make_daemon(tmp_path))
    row = snap["rows"][0]
    assert row["quadrant"] == "Q1" and row["trend_bias"] == "UP"
    assert "trend_bias" not in row["regime"], "bias is not part of the label"


def test_a_tape_inside_the_dead_band_reads_flat_rather_than_borrowing_a_sign():
    n = 120
    ts = pd.date_range("2026-08-01", periods=n, freq="15min", tz="UTC")
    # A drift far below TREND_BIAS_DEADBAND_PCT over the ROC window.
    close = 15000.0 + np.arange(n) * 0.001
    bars = pd.DataFrame({"ts": ts, "open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close, "volume": 100.0})
    metrics = supporting_metrics(bars, atr_14=1.0)
    assert abs(metrics["roc_15"]) < TREND_BIAS_DEADBAND_PCT
    assert metrics["trend_bias"] == "FLAT"


# --------------------------------------------------------------------------
# 6. The CLI itself
# --------------------------------------------------------------------------
def test_the_defaults_come_from_the_registry_not_from_a_list(tmp_path):
    """
    A hardcoded symbol list is one promotion away from omitting a contract, and
    a symbol missing from this card is a symbol whose strategies silently
    report "no regime published" - which reads exactly like a market that never
    entered its quadrant.
    """
    feed = FakeFeed({})
    snap = collect(feed=feed, daemon=make_daemon(tmp_path))

    daemon_symbols = {e["symbol"] for e in make_daemon(tmp_path).registry.values()
                      if e.get("symbol")}
    daemon_tfs = {e["timeframe"] for e in make_daemon(tmp_path).registry.values()
                  if e.get("timeframe")}
    assert set(snap["symbols_requested"]) == daemon_symbols
    assert set(snap["timeframes_requested"]) == daemon_tfs


def test_every_certified_timeframe_is_scanned(tmp_path):
    """
    A timeframe left out of the scan leaves its strategies reporting no
    published regime, which is indistinguishable on the card from a strategy
    the market has moved away from.
    """
    feed = FakeFeed({})
    daemon = make_daemon(tmp_path)
    collect(feed=feed, daemon=daemon)

    scanned = {tf for _, tf, _ in feed.calls}
    certified = {e["timeframe"] for e in daemon.registry.values()
                 if e.get("timeframe")}
    assert certified <= scanned, f"not scanned: {certified - scanned}"


def test_the_parser_accepts_the_documented_flags():
    args = build_parser().parse_args(
        ["--symbols", "NQ", "ES", "--tf", "15m", "1h",
         "--feed", "lake", "--json", "--strategies", "--lookback-bars", "250"])
    assert args.symbols == ["NQ", "ES"]
    assert args.timeframes == ["15m", "1h"]
    assert args.feed == "lake"
    assert args.json is True
    assert args.strategies is True
    assert args.lookback_bars == 250


def test_json_output_is_parseable_and_carries_the_switchboard(tmp_path,
                                                              capsys,
                                                              monkeypatch):
    """
    The `--json` payload is what the Hermes orchestrator consumes, so it has to
    parse and it has to carry the per-row verdict RESOLVED - a consumer
    reimplementing the (symbol, timeframe) join is a second chance to get it
    wrong.
    """
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "trend")})
    daemon = make_daemon(tmp_path)
    monkeypatch.setattr("scripts.check_market_regime.resolve_feed",
                        lambda *a, **k: feed)
    monkeypatch.setattr("scripts.check_market_regime.MasterRegimeDaemon",
                        lambda **k: daemon)

    rc = main(["--symbols", "NQ", "--tf", "15m", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    row = payload["rows"][0]
    assert row["quadrant"] == "Q1"
    assert "active_strategies" in row and "suppressed_strategies" in row
    assert isinstance(row["active_strategies"], list)


def test_the_table_prints_a_row_per_symbol_timeframe(tmp_path, capsys,
                                                     monkeypatch):
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "trend"),
                     ("NQ", "1h"): make_bars(120, "chop", tf_minutes=60)})
    daemon = make_daemon(tmp_path)
    monkeypatch.setattr("scripts.check_market_regime.resolve_feed",
                        lambda *a, **k: feed)
    monkeypatch.setattr("scripts.check_market_regime.MasterRegimeDaemon",
                        lambda **k: daemon)

    assert main(["--symbols", "NQ", "--tf", "15m", "1h"]) == 0
    out = capsys.readouterr().out
    assert "Quadrant" in out and "NormATR%" in out
    assert "Q1" in out and "Q4" in out


def test_nothing_classified_exits_non_zero(tmp_path, capsys, monkeypatch):
    """
    Non-zero means "this tool could not answer", never "the market is quiet".
    A quadrant that suits no strategy is a real state of the market.
    """
    monkeypatch.setattr("scripts.check_market_regime.resolve_feed",
                        lambda *a, **k: FakeFeed({}))
    monkeypatch.setattr("scripts.check_market_regime.MasterRegimeDaemon",
                        lambda **k: make_daemon(tmp_path))

    assert main(["--symbols", "NQ", "--tf", "15m"]) == 1
    assert "no closed bars" in capsys.readouterr().out


def test_a_classified_card_with_no_active_strategies_still_exits_zero(
        tmp_path, capsys, monkeypatch):
    feed = FakeFeed({("NQ", "15m"): make_bars(120, "chop")})
    monkeypatch.setattr("scripts.check_market_regime.resolve_feed",
                        lambda *a, **k: feed)
    monkeypatch.setattr("scripts.check_market_regime.MasterRegimeDaemon",
                        lambda **k: make_daemon(tmp_path))

    assert main(["--symbols", "NQ", "--tf", "15m"]) == 0
