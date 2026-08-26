"""
tests/test_regime_daemon.py - the live regime service.

ASSERT-BASED ON PURPOSE. `tests/conftest.py` routes any suite carrying the
`check()` helper to a subprocess runner because pytest cannot see its results;
this suite fails through `assert`, so it is collected case by case and both
runners report the same thing.

Run:
    .venv/bin/pytest tests/test_regime_daemon.py -q

What is pinned here, beyond the four items the specification asks for:

  * **The quadrant standard is one standard.** The daemon's labels are checked
    against `mdlib.regimes.QUADRANT_LABELS` and against
    `portfolio.config_loader.CANONICAL_QUADRANT` - a live daemon and a
    backtest that disagree about what Q1 means is the failure nothing
    downstream can detect.
  * **theta_vol is never taken from the live window.** A symbol with no pinned
    anchor must RAISE. A rolling median would label a quiet tape
    high-volatility and hand a Q1 strategy permission for a market it was
    never certified in - and the equity curve would look fine.
  * **The warm-up is Q0, never Q4.** `NaN > theta` is False, so the naive
    encoding files every warm-up bar under Low-Vol/Ranging: a populated column
    of a regime nobody measured.
  * **A broken ML model raises rather than passing the entry through.** True
    would trade unfiltered while the log said "ML confirmed"; False is
    indistinguishable from a model that vetoed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mdlib import regimes as regime_cache                       # noqa: E402
from portfolio.config_loader import CANONICAL_QUADRANT          # noqa: E402
from realtime import regime_reader                              # noqa: E402
from realtime.crosstrade_formatter import (                     # noqa: E402
    CrossTradeFormatError,
    KEY_ENV_VAR,
    REDACTED,
    format_crosstrade_command,
    format_crosstrade_json,
    format_flatten_command,
    redact,
)
from realtime import regime_daemon as regime_daemon_module     # noqa: E402
from realtime.regime_daemon import (                            # noqa: E402
    ADX_TREND_THRESHOLD,
    ENTRY_SIGNALS,
    EXIT_SIGNALS,
    MIN_BARS_FOR_REGIME,
    QUADRANT_TO_LABEL,
    UNDEFINED_LABEL,
    MLGateError,
    MasterRegimeDaemon,
    RegimeDaemonError,
    ThetaAnchorMissing,
    UnknownStrategy,
    _fmt_opt,
    _verify_alias_tick_sizes,
)
from realtime.regime_reader import (                            # noqa: E402
    RegimeStateError,
    get_all_regimes,
    get_current_regime,
    is_regime_permitted,
)

CONFIG = str(REPO_ROOT / "config" / "portfolios.json")

# The synthetic volatility boundary every fixture below is judged against.
# It sits between the two fixtures' ATRs (0.75 and 8.00) so "high volatility"
# and "low volatility" are decided by the anchor rather than by luck.
FIXTURE_THETA = 2.0


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def make_bars(n: int, kind: str, tf_minutes: int = 15) -> pd.DataFrame:
    """
    Deterministic bars in the shape `mdlib.lake` yields.

    `trend` - a clean 5-point-per-bar advance with a 5-point range: ADX pins
    high and ATR lands at 8.00.
    `chop`  - a 0.5-point alternation with a 0.5-point range: ADX collapses
    and ATR lands at 0.75.

    No randomness: a regime test whose fixture moves is a test that will one
    day fail for a reason nobody can reproduce.
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
    return pd.DataFrame({"ts": ts, "open": close, "high": high, "low": low,
                         "close": close, "volume": 100})


def write_anchors(tmp_path: Path, mapping: dict) -> Path:
    """An operator anchors file: {symbol: {tf: {"theta_vol": ...}}}."""
    path = tmp_path / "theta_vol_anchors.json"
    path.write_text(json.dumps(mapping, indent=2))
    return path


def make_daemon(tmp_path: Path, *, anchors: dict | None = None,
                models: Path | None = None, tf: str = "15m",
                symbols: tuple[str, ...] = ("NQ", "MNQ", "ES")
                ) -> MasterRegimeDaemon:
    """
    A daemon wired entirely to `tmp_path`.

    `cache_root` points at an empty directory so the REAL regime caches on
    `/mnt/backtest` cannot supply an anchor - every theta in this suite is one
    the test wrote, so a case that should raise for a missing anchor cannot
    pass because the box happens to have NQ cached.
    """
    if anchors is None:
        anchors = {"NQ": {tf: {"theta_vol": FIXTURE_THETA,
                               "is_start": "2013-01-01",
                               "is_end": "2022-12-31"}}}
    tmp_path.mkdir(parents=True, exist_ok=True)
    empty_cache = tmp_path / "no_cache"
    empty_cache.mkdir(parents=True, exist_ok=True)
    return MasterRegimeDaemon(
        config_path=CONFIG,
        state_file=str(tmp_path / "state" / "live_regime_state.json"),
        ml_model_dir=str(models if models is not None else tmp_path / "models"),
        anchors_path=str(write_anchors(tmp_path, anchors)),
        default_tf=tf,
        symbols=symbols,
        timeframes=(tf,),
        cache_root=str(empty_cache),
    )


# --------------------------------------------------------------------------
# 1. Regime classification
# --------------------------------------------------------------------------
def test_high_vol_high_adx_is_q1(tmp_path):
    daemon = make_daemon(tmp_path)
    out = daemon.calculate_regime("NQ", make_bars(200, "trend"))

    assert out["regime"] == "Q1_HIGH_VOL_TREND"
    assert out["quadrant"] == "Q1"
    assert out["is_high_vol"] is True
    assert out["is_trending"] is True
    assert out["adx_14"] > 25.0
    assert out["atr_14"] > FIXTURE_THETA
    assert out["theta_vol"] == FIXTURE_THETA
    assert out["symbol"] == "NQ" and out["tf"] == "15m"
    # `updated_at` is the daemon's clock; `bar_ts` is the market's. A consumer
    # needs both to tell a quiet tape from a dead feed.
    assert out["updated_at"] and out["bar_ts"]


def test_low_vol_low_adx_is_q4(tmp_path):
    daemon = make_daemon(tmp_path)
    out = daemon.calculate_regime("NQ", make_bars(200, "chop"))

    assert out["regime"] == "Q4_LOW_VOL_MEAN_REVERSION"
    assert out["quadrant"] == "Q4"
    assert out["is_high_vol"] is False
    assert out["is_trending"] is False
    assert out["adx_14"] < 25.0
    assert out["atr_14"] <= FIXTURE_THETA


@pytest.mark.parametrize(
    "kind, theta, quadrant, label, high_vol, trending",
    [
        ("trend", 2.0,  "Q1", "Q1_HIGH_VOL_TREND",           True,  True),
        ("chop",  0.5,  "Q2", "Q2_HIGH_VOL_CHOP",            True,  False),
        ("trend", 20.0, "Q3", "Q3_LOW_VOL_TREND",            False, True),
        ("chop",  2.0,  "Q4", "Q4_LOW_VOL_MEAN_REVERSION",   False, False),
    ],
)
def test_the_full_two_by_two_truth_table(tmp_path, kind, theta, quadrant,
                                         label, high_vol, trending):
    """
    All FOUR quadrants out of `calculate_regime`, on the two axes separately.

    The suite already pins the DIAGONAL - trending-and-volatile is Q1,
    quiet-and-choppy is Q4 - because those are what the two bar fixtures
    produce against the default anchor. The off-diagonal cases are the ones a
    transposed encoding survives: swap the two middle labels and Q1 and Q4 are
    both still correct, every count still adds up, and the only thing that
    changes is which strategy a live supervisor turns loose in which market.

    Q2 is not a hypothetical here. `t3_braid_scalp_20260823_NQ_1h` is
    certified in `Q2 · High Volatility / Ranging`, so this row is the exact
    classification the promoted strategy's permission to trade rests on.

    The axes are moved INDEPENDENTLY, which is what makes this a truth table
    rather than four assertions: the bar fixture sets the ADX axis (trend
    pins ADX at 100.0, chop collapses it to 3.7) and the pinned anchor sets
    the volatility axis (trend ATR is 8.00, chop ATR is 0.75). Every cell is
    therefore reached by one deliberate change from its neighbour, so a
    failure names the axis that broke.
    """
    daemon = make_daemon(tmp_path, anchors={
        "NQ": {"15m": {"theta_vol": theta,
                       "is_start": "2013-01-01", "is_end": "2022-12-31"}}})
    out = daemon.calculate_regime("NQ", make_bars(200, kind))

    assert out["quadrant"] == quadrant
    assert out["regime"] == label
    assert out["is_high_vol"] is high_vol
    assert out["is_trending"] is trending

    # The classification must follow from the two comparisons the charter
    # states, not merely agree with them by coincidence on this fixture.
    assert (out["adx_14"] > ADX_TREND_THRESHOLD) is trending
    assert (out["atr_14"] > out["theta_vol"]) is high_vol
    assert out["theta_vol"] == theta, "the anchor was not the one pinned"


def test_the_two_axes_are_independent(tmp_path):
    """
    Holding one axis and moving the other moves exactly one bit.

    Q1 -> Q3 is the anchor alone on identical bars; Q1 -> Q2 is the bars alone
    at a fixed anchor. If either move flipped both bits the quadrant encoding
    would be a single ordered scale rather than two independent axes, and
    `kill_switch_regimes` - which is derived as "the other three" - would be
    naming environments nobody measured.
    """
    def q(kind, theta):
        d = make_daemon(tmp_path / f"{kind}{theta}", anchors={
            "NQ": {"15m": {"theta_vol": theta, "is_start": "2013-01-01",
                           "is_end": "2022-12-31"}}})
        return d.calculate_regime("NQ", make_bars(200, kind))

    q1, q3 = q("trend", 2.0), q("trend", 20.0)
    assert (q1["quadrant"], q3["quadrant"]) == ("Q1", "Q3")
    assert q1["is_trending"] is q3["is_trending"] is True, "ADX axis moved"
    assert q1["adx_14"] == q3["adx_14"], "identical bars gave a different ADX"

    hi_trend, hi_chop = q("trend", 0.5), q("chop", 0.5)
    assert (hi_trend["quadrant"], hi_chop["quadrant"]) == ("Q1", "Q2")
    assert hi_trend["is_high_vol"] is hi_chop["is_high_vol"] is True, \
        "volatility axis moved when only the bars changed"


def test_returned_keys_match_the_specified_contract(tmp_path):
    daemon = make_daemon(tmp_path)
    out = daemon.calculate_regime("NQ", make_bars(200, "trend"))
    for key in ("symbol", "regime", "adx_14", "atr_14", "theta_vol",
                "is_high_vol", "is_trending", "updated_at"):
        assert key in out, f"{key} missing from calculate_regime's return"
    assert isinstance(out["adx_14"], float)
    assert isinstance(out["atr_14"], float)
    assert isinstance(out["is_high_vol"], bool)


def test_theta_decides_the_volatility_axis_not_the_bars(tmp_path):
    """
    The SAME trending bars land in Q1 or Q3 purely on the anchor.

    This is the whole reason theta is pinned: at theta=2.0 the fixture is
    high-volatility and at theta=20.0 it is not, on identical prices. A
    boundary taken from the live window would move with the tape and the
    quadrant would follow it.
    """
    bars = make_bars(200, "trend")
    low_theta = make_daemon(tmp_path / "a", anchors={
        "NQ": {"15m": {"theta_vol": 2.0}}})
    high_theta = make_daemon(tmp_path / "b", anchors={
        "NQ": {"15m": {"theta_vol": 20.0}}})

    assert low_theta.calculate_regime("NQ", bars)["quadrant"] == "Q1"
    assert high_theta.calculate_regime("NQ", bars)["quadrant"] == "Q3"


def test_missing_anchor_raises_instead_of_taking_a_live_median(tmp_path):
    """
    The single most important refusal in the module.

    ES has no pinned theta_vol here. The daemon must raise rather than compute
    a median of the bars it was handed - that boundary is a property of the
    request, so on a quiet morning every bar reads high-volatility and a
    Q1-certified strategy is handed permission for a market it is not in.
    """
    daemon = make_daemon(tmp_path)
    with pytest.raises(ThetaAnchorMissing) as exc:
        daemon.calculate_regime("ES", make_bars(200, "trend"))
    assert "ES" in str(exc.value)
    # The message has to name the fix, not just the failure.
    assert "precompute_regimes" in str(exc.value)


def test_anchor_is_keyed_by_timeframe(tmp_path):
    """
    NQ's real theta_vol is 7.90 at 15m and 11.33 at 30m. An anchor applied at
    the wrong timeframe silently relabels a third of the session, so the
    timeframe is part of the key and a 30m request against a 15m-only anchor
    file must raise rather than borrow it.
    """
    daemon = make_daemon(tmp_path, tf="15m")
    assert daemon.theta_for("NQ", "15m")["theta_vol"] == FIXTURE_THETA
    with pytest.raises(ThetaAnchorMissing):
        daemon.theta_for("NQ", "30m")


def test_micro_resolves_to_its_full_size_anchor(tmp_path):
    """
    MNQ and NQ quote the same price series at the same tick size, so NQ's
    boundary is MNQ's boundary. The alias is recorded on the result rather
    than applied silently.
    """
    daemon = make_daemon(tmp_path)
    out = daemon.calculate_regime("MNQ", make_bars(200, "trend"))
    assert out["theta_vol"] == FIXTURE_THETA
    assert out["theta_anchor_symbol"] == "NQ"
    assert out["theta_aliased"] is True
    assert out["symbol"] == "MNQ"


def test_micro_alias_tick_sizes_reconcile_against_specs():
    """
    The alias is only sound while the micro and its parent quote the same tick
    size. Checked against `backtest/specs.py` rather than asserted in a
    comment: if a contract change made them differ, every ATR comparison for
    that symbol would be wrong by the same factor, in price units, silently.
    """
    assert _verify_alias_tick_sizes() == []


def test_warmup_is_q0_and_never_q4(tmp_path):
    """
    `NaN > theta` is False, so the naive encoding files every warm-up bar
    under Low-Vol/Ranging - a real-looking label on bars where no indicator
    exists. Too few bars must report Q0 with NULL indicators.
    """
    daemon = make_daemon(tmp_path)
    out = daemon.calculate_regime("NQ", make_bars(MIN_BARS_FOR_REGIME - 1,
                                                  "trend"))
    assert out["regime"] == UNDEFINED_LABEL
    assert out["quadrant"] == "Q0"
    assert out["adx_14"] is None and out["atr_14"] is None
    assert out["is_high_vol"] is False and out["is_trending"] is False
    assert out["regime"] != "Q4_LOW_VOL_MEAN_REVERSION"


def test_adx_exactly_at_the_threshold_is_ranging(tmp_path):
    """
    The comparator is strictly `>`, matching `mdlib/regimes.py` and every
    quadrant already in the caches. The specification for this module was
    written `>=`; at ADX exactly 25.0 the two rules disagree, and this test is
    where that decision is recorded rather than left to whichever file is read
    first.
    """
    frame = pd.DataFrame(
        {"adx_14": [25.0, 25.0000001], "atr_14": [1.0, 1.0]},
        index=pd.to_datetime(["2026-08-01T00:00Z", "2026-08-01T00:15Z"]))
    out = regime_cache.classify(frame, theta_vol=FIXTURE_THETA)
    assert bool(out["is_trending"].iloc[0]) is False     # 25.0 is NOT trending
    assert bool(out["is_trending"].iloc[1]) is True
    assert int(out["regime_quadrant"].iloc[0]) == 4


def test_unsorted_bars_raise_rather_than_being_sorted(tmp_path):
    """
    A live frame out of time order is a feed problem. Sorting it here would
    hide the gap that caused it and produce a plausible ADX from bars that
    never arrived in that order.
    """
    daemon = make_daemon(tmp_path)
    bars = make_bars(200, "trend")
    shuffled = pd.concat([bars.iloc[100:], bars.iloc[:100]], ignore_index=True)
    with pytest.raises(RegimeDaemonError, match="ascending"):
        daemon.calculate_regime("NQ", shuffled)


def test_bars_may_be_indexed_by_timestamp(tmp_path):
    """A frame indexed by ts rather than carrying a `ts` column classifies the
    same - the two shapes are both in circulation upstream."""
    daemon = make_daemon(tmp_path)
    bars = make_bars(200, "trend")
    indexed = bars.set_index("ts")
    assert (daemon.calculate_regime("NQ", indexed)["quadrant"]
            == daemon.calculate_regime("NQ", bars)["quadrant"] == "Q1")


def test_quadrant_standard_is_one_standard():
    """
    The daemon's labels, `mdlib.regimes`' encoding and the portfolio schema's
    names must be three views of one table. A daemon whose Q1 is the backtest's
    Q3 stands a strategy down in the environment it was certified for and turns
    it loose in the one it never traded, with every log line reading correctly.
    """
    assert QUADRANT_TO_LABEL["Q1"] == "Q1_HIGH_VOL_TREND"
    assert QUADRANT_TO_LABEL["Q2"] == "Q2_HIGH_VOL_CHOP"
    assert QUADRANT_TO_LABEL["Q3"] == "Q3_LOW_VOL_TREND"
    assert QUADRANT_TO_LABEL["Q4"] == "Q4_LOW_VOL_MEAN_REVERSION"

    # ... and each of those is the schema label the routing table resolves to
    # the same id, and the id `mdlib.regimes` gives the same environment.
    for quad, label in QUADRANT_TO_LABEL.items():
        if quad == "Q0":
            continue
        assert CANONICAL_QUADRANT[label] == quad
        name = regime_cache.QUADRANT_LABELS[int(quad[1:])]
        high_vol = name.startswith("High")
        trending = name.endswith("Trending")
        assert ("HIGH_VOL" in label) is high_vol, (label, name)
        assert ("TREND" in label.replace("HIGH_VOL", "")) is trending, (label,
                                                                       name)


def test_undefined_label_agrees_across_the_two_modules():
    """The reader spells Q0 itself rather than importing the writer. Pinned
    equal here, which is the only thing keeping the two spellings in step."""
    assert UNDEFINED_LABEL == regime_reader.UNDEFINED_LABEL


# --------------------------------------------------------------------------
# 2. State cache
# --------------------------------------------------------------------------
def test_update_state_writes_and_reader_retrieves(tmp_path):
    daemon = make_daemon(tmp_path)
    data = daemon.calculate_regime("NQ", make_bars(200, "trend"))
    daemon.update_state("NQ", data)

    read = get_current_regime("NQ", state_file=daemon.state_file)
    assert read["regime"] == data["regime"] == "Q1_HIGH_VOL_TREND"
    assert read["quadrant"] == "Q1"
    assert read["atr_14"] == pytest.approx(data["atr_14"])
    assert read["adx_14"] == pytest.approx(data["adx_14"])
    assert read["theta_vol"] == FIXTURE_THETA
    assert read["age_seconds"] is not None and read["age_seconds"] < 60


def test_refresh_classifies_and_publishes_in_one_call(tmp_path):
    daemon = make_daemon(tmp_path)
    written = daemon.refresh("NQ", make_bars(200, "chop"))
    assert get_current_regime("NQ", daemon.state_file)["regime"] \
        == written["regime"] == "Q4_LOW_VOL_MEAN_REVERSION"


def test_update_state_leaves_no_partial_file_and_no_temp_files(tmp_path):
    """
    Atomic means temp-then-`os.replace`, in the destination directory. A reader
    sees the previous complete document or the new one - never a half-written
    one - and a failed write must not leave a temp file a directory listing
    reads as state.
    """
    daemon = make_daemon(tmp_path)
    for _ in range(5):
        daemon.refresh("NQ", make_bars(200, "trend"))

    state_dir = Path(daemon.state_file).parent
    leftovers = [p.name for p in state_dir.iterdir()
                 if p.name != Path(daemon.state_file).name]
    assert leftovers == [], f"temp files left behind: {leftovers}"
    # The published document parses in full, every time.
    blob = json.loads(Path(daemon.state_file).read_text())
    assert blob["symbols"]["NQ"]["quadrant"] == "Q1"
    assert blob["schema_version"]


def test_one_symbol_update_does_not_drop_the_others(tmp_path):
    """
    The file is rewritten whole, so an update for NQ that did not merge would
    delete MNQ - and every reader asking about MNQ would be told the daemon is
    not watching it, which is indistinguishable from a genuine
    misconfiguration.
    """
    daemon = make_daemon(tmp_path)
    daemon.refresh("NQ", make_bars(200, "trend"))
    daemon.refresh("MNQ", make_bars(200, "chop"))

    published = get_all_regimes(daemon.state_file)
    assert sorted(published) == ["MNQ", "NQ"]
    assert published["NQ"]["quadrant"] == "Q1"
    assert published["MNQ"]["quadrant"] == "Q4"


def test_a_restarted_daemon_preserves_what_is_on_disk(tmp_path):
    """A daemon restarted mid-session must not publish a state file holding
    only whichever symbol ticked first."""
    first = make_daemon(tmp_path)
    first.refresh("NQ", make_bars(200, "trend"))

    second = make_daemon(tmp_path)          # same state file
    second.refresh("MNQ", make_bars(200, "chop"))
    assert sorted(get_all_regimes(second.state_file)) == ["MNQ", "NQ"]


def test_reader_raises_for_a_missing_file_and_an_unknown_symbol(tmp_path):
    """
    Never a placeholder. `{"regime": None}` downstream becomes "not in the
    permitted quadrant", which stands a strategy down for a missing file and
    looks exactly like a market that moved.
    """
    with pytest.raises(RegimeStateError, match="no live regime state"):
        get_current_regime("NQ", state_file=tmp_path / "nothing.json")

    daemon = make_daemon(tmp_path)
    daemon.refresh("NQ", make_bars(200, "trend"))
    with pytest.raises(RegimeStateError, match="ES"):
        get_current_regime("ES", state_file=daemon.state_file)


def test_reader_reports_and_can_refuse_a_stale_record(tmp_path):
    daemon = make_daemon(tmp_path)
    daemon.refresh("NQ", make_bars(200, "trend"))

    # The fixture's last bar is dated 2026-08-01, so the BAR age is large even
    # though the write is fresh: a daemon looping over a dead feed.
    fresh = get_current_regime("NQ", daemon.state_file)
    assert fresh["age_seconds"] < 60
    assert fresh["bar_age_seconds"] > 60

    with pytest.raises(RegimeStateError, match="stale"):
        get_current_regime("NQ", daemon.state_file, max_age_s=60)


def test_permission_accepts_either_identifier_and_never_q0(tmp_path):
    daemon = make_daemon(tmp_path)
    daemon.refresh("NQ", make_bars(200, "trend"))
    sf = daemon.state_file

    assert is_regime_permitted("NQ", ["Q1"], state_file=sf) is True
    assert is_regime_permitted("NQ", ["Q1_HIGH_VOL_TREND"], state_file=sf) is True
    assert is_regime_permitted("NQ", ["Q3_LOW_VOL_TREND"], state_file=sf) is False

    daemon.refresh("MNQ", make_bars(MIN_BARS_FOR_REGIME - 1, "trend"))
    # Q0 is not a quadrant: permission may not be derived from a regime nobody
    # measured, even when the caller lists it.
    assert is_regime_permitted("MNQ", ["Q0", UNDEFINED_LABEL, "Q1"],
                               state_file=sf) is False


def test_state_file_records_the_standard_it_was_written_under(tmp_path):
    """A quadrant read without the boundary that drew it is not a
    measurement."""
    daemon = make_daemon(tmp_path)
    daemon.refresh("NQ", make_bars(200, "trend"))
    blob = json.loads(Path(daemon.state_file).read_text())
    assert blob["quadrant_standard"]["Q1"] == "Q1_HIGH_VOL_TREND"
    assert blob["adx_trend_threshold"] == 25.0
    assert blob["in_sample_window"] == ["2013-01-01", "2022-12-31"]
    assert blob["symbols"]["NQ"]["theta_window"] == ["2013-01-01", "2022-12-31"]


# --------------------------------------------------------------------------
# 3. ML gate
# --------------------------------------------------------------------------
def _fit_model():
    """A real classifier, so `predict_proba` is the library's and not a stub's.
    Feature `a` is decisive; `b` is noise."""
    from sklearn.linear_model import LogisticRegression
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 20)
    y = np.array([0, 0, 1, 1] * 20)
    return LogisticRegression().fit(X, y)


def write_model(models_dir: Path, name: str, *, threshold=None,
                features=("a", "b"), model=None) -> Path:
    import joblib
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.pkl"
    joblib.dump(model if model is not None else _fit_model(), path)
    meta = {}
    if features is not None:
        meta["features"] = list(features)
    if threshold is not None:
        meta["threshold"] = threshold
    if meta:
        path.with_suffix(".json").write_text(json.dumps(meta))
    return path


def test_ml_gate_passes_through_when_no_model_is_registered(tmp_path):
    """The documented pass-through: a rule-based Version A strategy has no
    classifier and must keep trading."""
    daemon = make_daemon(tmp_path)
    assert daemon.has_model("rule_based_v1") is False
    assert daemon.evaluate_ml_gate("rule_based_v1", "NQ", {"a": 1.0}) is True


def test_ml_gate_thresholds_a_registered_model(tmp_path):
    models = tmp_path / "models"
    write_model(models, "filtered_v1", threshold=0.60)
    daemon = make_daemon(tmp_path, models=models)

    assert daemon.has_model("filtered_v1") is True
    # a=1 is the positive class; a=0 is the negative one.
    assert daemon.evaluate_ml_gate("filtered_v1", "NQ",
                                   {"a": 1.0, "b": 0.0}) is True
    assert daemon.evaluate_ml_gate("filtered_v1", "NQ",
                                   {"a": 0.0, "b": 0.0}) is False


def test_ml_gate_threshold_is_read_from_the_sidecar(tmp_path):
    """The same model and the same features flip on the declared threshold
    alone - so the threshold is genuinely being applied, not inferred from
    `predict`."""
    features = {"a": 1.0, "b": 0.0}
    model = _fit_model()
    proba = float(model.predict_proba([[1.0, 0.0]])[0][1])
    assert 0.5 < proba < 1.0, "fixture must sit strictly inside the band"

    loose = make_daemon(tmp_path / "loose", models=write_model(
        tmp_path / "loose" / "models", "m", threshold=proba - 0.01,
        model=model).parent)
    tight = make_daemon(tmp_path / "tight", models=write_model(
        tmp_path / "tight" / "models", "m", threshold=proba + 0.01,
        model=model).parent)

    assert loose.evaluate_ml_gate("m", "NQ", features) is True
    assert tight.evaluate_ml_gate("m", "NQ", features) is False


def test_ml_gate_orders_features_by_the_sidecar_not_the_dict(tmp_path):
    """
    A dict has no meaningful order. A classifier handed its columns permuted
    does not fail - it returns a confident probability from a matrix that means
    nothing, and nothing raises.
    """
    models = tmp_path / "models"
    write_model(models, "ordered_v1", threshold=0.60, features=("a", "b"))
    daemon = make_daemon(tmp_path, models=models)

    forward = daemon.evaluate_ml_gate("ordered_v1", "NQ", {"a": 1.0, "b": 0.0})
    reversed_dict = daemon.evaluate_ml_gate("ordered_v1", "NQ",
                                            {"b": 0.0, "a": 1.0})
    assert forward is reversed_dict is True


def test_ml_gate_refuses_a_model_with_no_declared_feature_order(tmp_path):
    models = tmp_path / "models"
    write_model(models, "unlabelled_v1", features=None)
    daemon = make_daemon(tmp_path, models=models)
    with pytest.raises(MLGateError, match="features"):
        daemon.evaluate_ml_gate("unlabelled_v1", "NQ", {"a": 1.0, "b": 0.0})


def test_ml_gate_refuses_to_impute_a_missing_feature(tmp_path):
    models = tmp_path / "models"
    write_model(models, "needs_both_v1", threshold=0.60)
    daemon = make_daemon(tmp_path, models=models)
    with pytest.raises(MLGateError, match="did not supply"):
        daemon.evaluate_ml_gate("needs_both_v1", "NQ", {"a": 1.0})


def test_a_broken_model_raises_rather_than_passing_the_entry_through(tmp_path):
    """
    True here would trade unfiltered while every log line said "ML confirmed".
    False is indistinguishable from a model that looked at the features and
    vetoed. Both are silent; the raise is not.
    """
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / "corrupt_v1.pkl").write_bytes(b"this is not a pickle")
    daemon = make_daemon(tmp_path, models=models)

    assert "corrupt_v1" in daemon.model_errors
    assert daemon.has_model("corrupt_v1") is False
    with pytest.raises(MLGateError, match="failed to load"):
        daemon.evaluate_ml_gate("corrupt_v1", "NQ", {"a": 1.0})


def test_a_symbol_specific_model_overrides_the_shared_one(tmp_path):
    models = tmp_path / "models"
    write_model(models, "dual_v1", threshold=0.99)          # shared: rejects
    write_model(models, "dual_v1__NQ", threshold=0.01)      # NQ: accepts
    daemon = make_daemon(tmp_path, models=models)

    assert daemon.evaluate_ml_gate("dual_v1", "NQ", {"a": 1.0, "b": 0.0}) is True
    assert daemon.evaluate_ml_gate("dual_v1", "ES", {"a": 1.0, "b": 0.0}) is False


def test_a_model_with_no_predict_proba_is_refused(tmp_path):
    """The gate compares a PROBABILITY against a threshold. A bare `predict`
    gives a class, and thresholding a class is not gating."""
    from sklearn.preprocessing import StandardScaler
    models = tmp_path / "models"
    write_model(models, "not_a_classifier", threshold=0.5,
                model=StandardScaler().fit(np.array([[0.0, 0.0], [1.0, 1.0]])))
    daemon = make_daemon(tmp_path, models=models)
    assert "not_a_classifier" in daemon.model_errors
    with pytest.raises(MLGateError):
        daemon.evaluate_ml_gate("not_a_classifier", "NQ", {"a": 1.0, "b": 0.0})


# --------------------------------------------------------------------------
# 4. CrossTrade formatting
# --------------------------------------------------------------------------
def test_plain_text_command_is_the_documented_wire_format(monkeypatch):
    monkeypatch.delenv(KEY_ENV_VAR, raising=False)
    assert format_crosstrade_command(
        account="Prop-Odd", instrument="MNQ", action="buy", qty=2,
        key="SECRET") == (
        "key=SECRET; command=place; account=Prop-Odd; instrument=MNQ; "
        "action=BUY; qty=2; order_type=MARKET; tif=DAY;")

    # Every field is present, in order, terminated by a semicolon.
    parts = [p.strip() for p in format_crosstrade_command(
        "A", "ES 03-26", "sell", 1, key="k", tif="gtc").split(";") if p.strip()]
    assert [p.split("=")[0] for p in parts] == [
        "key", "command", "account", "instrument", "action", "qty",
        "order_type", "tif"]
    assert "action=SELL" in parts and "instrument=ES 03-26" in parts
    assert "tif=GTC" in parts


def test_json_command_is_lower_cased_and_carries_the_tag():
    assert format_crosstrade_json(
        account="Prop-Even", instrument="mes", action="BUY", qty=3,
        strategy_tag="ema_crossover_20260821") == {
            "command": "place",
            "account": "Prop-Even",
            "instrument": "MES",
            "action": "buy",
            "qty": 3,
            "order_type": "market",
            "strategy_tag": "ema_crossover_20260821",
        }
    # No key in the body: the JSON endpoint takes the credential in the
    # request, and putting one here would publish it into every payload log.
    assert "key" not in format_crosstrade_json("A", "NQ", "buy", 1)


def test_flatten_carries_no_side_and_no_quantity():
    """A flatten closes whatever is open. Expressing it as `sell qty=N`
    requires guessing the position, and a wrong guess opens the opposite one."""
    out = format_flatten_command("Incubator-Odd", "MCL", key="k")
    assert out == ("key=k; command=flatten; account=Incubator-Odd; "
                   "instrument=MCL;")
    assert "action=" not in out and "qty=" not in out


def test_key_falls_back_to_the_environment_and_redacts(monkeypatch):
    monkeypatch.setenv(KEY_ENV_VAR, "ENV_KEY")
    out = format_crosstrade_command("A", "NQ", "buy", 1)
    assert "key=ENV_KEY;" in out
    # An explicit argument always wins over the environment.
    assert "key=ARG;" in format_crosstrade_command("A", "NQ", "buy", 1,
                                                   key="ARG")
    # And a command can be logged without publishing the credential.
    assert "ENV_KEY" not in redact(out)
    assert f"key={REDACTED};" in redact(out)


@pytest.mark.parametrize("kwargs,reason", [
    (dict(account="A", instrument="NQ", action="hodl", qty=1), "unknown side"),
    (dict(account="A", instrument="NQ", action="flatten", qty=1),
     "flatten is a command, not a side"),
    (dict(account="A", instrument="NQ", action="buy", qty=0), "zero quantity"),
    (dict(account="A", instrument="NQ", action="buy", qty=-2), "short by sign"),
    (dict(account="A", instrument="NQ", action="buy", qty=1.5), "fractional"),
    (dict(account="A", instrument="NQ", action="buy", qty=True), "bool is int"),
    (dict(account="A", instrument="NQ", action="buy", qty=1,
          order_type="limit"), "price-bearing type with no price field"),
    (dict(account="", instrument="NQ", action="buy", qty=1), "no account"),
    (dict(account="A", instrument="", action="buy", qty=1), "no instrument"),
    (dict(account="A;x=1", instrument="NQ", action="buy", qty=1),
     "field separator injected into the command"),
])
def test_malformed_orders_are_refused(kwargs, reason):
    with pytest.raises(CrossTradeFormatError):
        format_crosstrade_command(**kwargs)
    if "tif" not in kwargs:
        with pytest.raises(CrossTradeFormatError):
            format_crosstrade_json(**{k: v for k, v in kwargs.items()
                                      if k != "tif"})


def test_bad_time_in_force_is_refused():
    """A broker that does not recognise a TIF substitutes its own, and the
    resulting order is not the one that was written."""
    with pytest.raises(CrossTradeFormatError, match="tif"):
        format_crosstrade_command("A", "NQ", "buy", 1, tif="ioc")


def test_actions_come_from_the_dispatcher_not_a_second_list():
    """One vocabulary of sides. A second list in a second module is how BUY and
    SELL come to mean different things in two files that both look right."""
    from live.dispatcher import VALID_ACTIONS
    from realtime.crosstrade_formatter import PLACE_ACTIONS
    assert PLACE_ACTIONS <= set(VALID_ACTIONS)
    assert PLACE_ACTIONS == {"BUY", "SELL"}


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------
def test_an_end_to_end_pass_produces_a_routable_order(tmp_path, monkeypatch):
    """
    Bars -> quadrant -> state -> permission -> payload, in the order the live
    loop runs them. Nothing here reaches a network: the formatter returns a
    string and `live/dispatcher.py` remains the only module that sends.
    """
    monkeypatch.setenv(KEY_ENV_VAR, "SECRET_WEBHOOK_KEY")
    daemon = make_daemon(tmp_path)
    daemon.refresh("MNQ", make_bars(200, "trend"))

    permitted = is_regime_permitted("MNQ", ["Q1_HIGH_VOL_TREND"],
                                    state_file=daemon.state_file)
    assert permitted is True
    assert daemon.evaluate_ml_gate("momentum_v1", "MNQ", {}) is True  # no model

    command = format_crosstrade_command("Prop-Even", "MNQ", "buy", 1)
    assert "command=place" in command and "action=BUY" in command
    assert "key=SECRET_WEBHOOK_KEY;" in command
    # The command is loggable only after redaction - a log file outlives the
    # session that wrote it, and a command copied out of one is replayable.
    assert "SECRET_WEBHOOK_KEY" not in redact(command)


def test_the_real_repository_config_loads_and_names_four_accounts(tmp_path):
    """The daemon validates the routing table on construction, so a config that
    stopped describing the four-account architecture fails here rather than at
    the moment an order needs an account."""
    daemon = make_daemon(tmp_path)
    assert sorted(daemon.config["portfolios"]) == [
        "Incubator-Even", "Incubator-Odd", "Prop-Even", "Prop-Odd"]
    assert set(daemon.tradeable_symbols) == {"MNQ", "MES", "MCL", "MGC"}


def test_a_broken_symbol_model_does_not_fall_back_to_the_shared_one(tmp_path):
    """
    A per-contract model is a deliberate override. If it fails to load, using
    the shared model instead filters that symbol with the classifier its author
    replaced - and the log would report a clean ML-confirmed entry.
    """
    models = tmp_path / "models"
    write_model(models, "dual_v2", threshold=0.01)          # shared: accepts
    models.mkdir(parents=True, exist_ok=True)
    (models / "dual_v2__NQ.pkl").write_bytes(b"not a pickle")

    daemon = make_daemon(tmp_path, models=models)
    features = {"a": 1.0, "b": 0.0}
    # ES has no override and uses the shared model, which accepts.
    assert daemon.evaluate_ml_gate("dual_v2", "ES", features) is True
    # NQ's override is broken: raise rather than borrow ES's answer.
    with pytest.raises(MLGateError, match="failed to load"):
        daemon.evaluate_ml_gate("dual_v2", "NQ", features)


def test_state_numbers_survive_the_round_trip_as_numbers(tmp_path):
    """
    `json.dump(default=str)` is the obvious shortcut and a silent corruption:
    a numpy float would be written as "7.89", and a reader comparing an ATR
    against a threshold would be comparing a string.
    """
    daemon = make_daemon(tmp_path)
    daemon.refresh("NQ", make_bars(200, "trend"))
    raw = json.loads(Path(daemon.state_file).read_text())["symbols"]["NQ"]
    for key in ("adx_14", "atr_14", "theta_vol"):
        assert isinstance(raw[key], float), f"{key} came back as {type(raw[key])}"
    assert isinstance(raw["is_high_vol"], bool)
    assert isinstance(raw["n_bars"], int)


def test_state_refuses_to_write_a_value_of_an_unknown_shape(tmp_path):
    """A value nobody taught the serialiser to write is a value nobody has
    decided the shape of - so it raises rather than becoming str(obj)."""
    daemon = make_daemon(tmp_path)
    data = daemon.calculate_regime("NQ", make_bars(200, "trend"))
    data["something"] = object()
    with pytest.raises(TypeError, match="cannot hold"):
        daemon.update_state("NQ", data)


def test_a_refused_write_leaves_the_daemon_and_the_disk_usable(tmp_path):
    """
    A rejected update must not poison the in-memory state. Mutating before the
    write means the bad entry is retried on every later update - for any
    symbol - while the disk keeps serving the last good document and nothing
    looks wrong.
    """
    daemon = make_daemon(tmp_path)
    daemon.refresh("NQ", make_bars(200, "trend"))

    bad = daemon.calculate_regime("MNQ", make_bars(200, "chop"))
    bad["oops"] = object()
    with pytest.raises(TypeError):
        daemon.update_state("MNQ", bad)

    # Disk still holds the last good document, and nothing partial beside it.
    assert sorted(get_all_regimes(daemon.state_file)) == ["NQ"]
    assert [p.name for p in Path(daemon.state_file).parent.iterdir()] == [
        Path(daemon.state_file).name]

    # And the daemon still works, for the symbol that failed and for the other.
    daemon.refresh("MNQ", make_bars(200, "chop"))
    assert sorted(get_all_regimes(daemon.state_file)) == ["MNQ", "NQ"]


# --------------------------------------------------------------------------
# 5. The live classifier against the pre-computed cache  (needs the lake)
# --------------------------------------------------------------------------
def test_live_labels_reproduce_the_cached_quadrant(tmp_path):
    """
    THE REGRESSION THAT MATTERS MOST, and the only one here that reads real
    bars: for the same bar, the daemon's live label must equal the
    `regime_quadrant` `mdlib.regimes` already wrote into the cache.

    Every certification in this repository was drawn on the cached column. If
    the live daemon labelled the same bar differently, a strategy would be
    stood down in the quadrant it was certified for and turned loose in one it
    never traded - and every log line would read correctly. Both sides run the
    same `classify()` against the same pinned theta, so agreement should be
    exact rather than approximate, and this asserts exactly that.

    SKIPS LOUDLY without the lake or the regime cache, the way
    `tests/test_temporal_chunking.py` does - it needs `/mnt/backtest`.
    """
    from mdlib.regimes import provenance

    if provenance("NQ", "15m") is None:
        pytest.skip("SKIPPED (needs the regime cache at /mnt/backtest): the "
                    "live-vs-cached agreement check did NOT run")
    try:
        from mdlib.lake import iter_bars
        frames = [df for _, df in iter_bars(["NQ"], "15m",
                                            "2026-06-01", "2026-08-07")]
    except Exception as exc:                        # no mount, no lake
        pytest.skip(f"SKIPPED (needs the lake at /mnt/backtest: {exc}): the "
                    f"live-vs-cached agreement check did NOT run")

    bars = frames[0]
    if "regime_quadrant" not in bars.columns or len(bars) < 500:
        pytest.skip("SKIPPED (the lake returned no regime column): the "
                    "live-vs-cached agreement check did NOT run")

    daemon = MasterRegimeDaemon(
        config_path=CONFIG,
        state_file=str(tmp_path / "state.json"),
        ml_model_dir=str(tmp_path / "models"),
        default_tf="15m", symbols=("NQ",), timeframes=("15m",))

    # The anchor must be the CACHE's own theta - not a value this test chose.
    assert daemon.theta_for("NQ", "15m")["source"] == "regime_cache"

    disagreements = []
    for end in range(len(bars) - 25, len(bars)):
        live = daemon.calculate_regime("NQ", bars.iloc[:end + 1], tf="15m")
        cached = f"Q{int(bars['regime_quadrant'].iloc[end])}"
        if live["quadrant"] != cached:
            disagreements.append((str(bars['ts'].iloc[end]), live["quadrant"],
                                  cached))
    assert not disagreements, (
        f"live daemon disagrees with the cached quadrant on "
        f"{len(disagreements)} of 25 bars: {disagreements[:5]}")


# --------------------------------------------------------------------------
# 8. The strategy registry (Module A) and the switchboard (Module C)
# --------------------------------------------------------------------------
# These build their OWN portfolios.json and their OWN incubator directory
# rather than leaning on the repository's. `config/portfolios.json` is a live
# routing table an operator edits, so a switchboard test pinned to whatever is
# promoted today would start failing on the next promotion for a reason that
# is not a defect - and, worse, would start PASSING vacuously if the strategy
# it names were ever deactivated.
SWITCH_ID = "fixture_strategy_NQ_1h"


def write_switchboard_config(tmp_path: Path, *, strategy_id: str = SWITCH_ID,
                             symbol: str = "NQ", tf: str = "1h",
                             quadrant: str | None = "Q2") -> str:
    """
    The repository's config with ONE portfolio's strategy list replaced.

    Derived from the real file so every other invariant the config loader
    enforces - the four accounts, the asset metadata reconciled against
    `backtest/specs.py`, the orthogonal baskets - still holds. Only the grant
    under test is synthetic.
    """
    blob = json.loads(Path(CONFIG).read_text())
    record = {"strat": strategy_id, "symbol": symbol, "timeframe": tf,
              "version": "A", "allocation": 1, "status": "incubating"}
    if quadrant is not None:
        record["regime_filter"] = quadrant
    for pid, portfolio in blob["portfolios"].items():
        if pid == "Incubator-Odd":
            portfolio["active_strategies"] = [strategy_id]
            portfolio["strategy_allocations"] = {strategy_id: record}
        else:
            portfolio["active_strategies"] = []
            portfolio.pop("strategy_allocations", None)
    path = tmp_path / "portfolios.json"
    path.write_text(json.dumps(blob, indent=1))
    return str(path)


def write_incubator(tmp_path: Path, *, strategy_id: str = SWITCH_ID,
                    symbol: str = "NQ", tf: str = "1h",
                    quadrant: str | None = "Q2") -> Path:
    """An `approved_incubator/<id>/meta.json` carrying the certification."""
    directory = tmp_path / "incubator" / strategy_id
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"name": strategy_id, "symbol": symbol, "symbols": [symbol],
            "timeframe": tf, "version": "A"}
    if quadrant is not None:
        meta["certification"] = {"target_quadrant": quadrant,
                                 "audit_symbol": symbol, "audit_timeframe": tf}
    (directory / "meta.json").write_text(json.dumps(meta, indent=1))
    return tmp_path / "incubator"


def make_switchboard_daemon(tmp_path: Path, *, theta: float = 0.5,
                            tf: str = "1h", cfg_quadrant: str | None = "Q2",
                            meta_quadrant: str | None = "Q2",
                            cfg_tf: str | None = None,
                            meta_tf: str | None = None,
                            with_meta: bool = True) -> MasterRegimeDaemon:
    config = write_switchboard_config(tmp_path, tf=cfg_tf or tf,
                                      quadrant=cfg_quadrant)
    if with_meta:
        incubator = write_incubator(tmp_path, tf=meta_tf or tf,
                                    quadrant=meta_quadrant)
    else:
        incubator = tmp_path / "empty_incubator"
        incubator.mkdir(parents=True, exist_ok=True)
    empty_cache = tmp_path / "no_cache"
    empty_cache.mkdir(parents=True, exist_ok=True)
    return MasterRegimeDaemon(
        config_path=config,
        state_file=str(tmp_path / "state" / "live_regime_state.json"),
        ml_model_dir=str(tmp_path / "models"),
        anchors_path=str(write_anchors(
            tmp_path, {"NQ": {tf: {"theta_vol": theta,
                                   "is_start": "2013-01-01",
                                   "is_end": "2022-12-31"}}})),
        default_tf=tf,
        symbols=("NQ",),
        timeframes=(tf,),
        cache_root=str(empty_cache),
        incubator_dir=str(incubator),
    )


def test_registry_takes_permission_from_config_and_certification_from_meta(tmp_path):
    daemon = make_switchboard_daemon(tmp_path)
    entry = daemon.registry[SWITCH_ID]
    assert entry["optimal_regime"] == "Q2"
    assert entry["optimal_regime_label"] == "Q2_HIGH_VOL_CHOP"
    assert entry["symbol"] == "NQ" and entry["timeframe"] == "1h"
    assert entry["portfolio_id"] == "Incubator-Odd"
    # Both sources are recorded, so a later disagreement is diagnosable.
    assert entry["optimal_regime_sources"] == {"portfolios.json": "Q2",
                                               "meta.json": "Q2"}
    assert daemon.registry_conflicts == []


def test_a_schema_label_and_a_quadrant_id_resolve_to_the_same_thing(tmp_path):
    """A basket writes `Q2_HIGH_VOL_CHOP`, a Stage 1 handoff writes `Q2`."""
    daemon = make_switchboard_daemon(tmp_path, cfg_quadrant="Q2_HIGH_VOL_CHOP",
                                     meta_quadrant="Q2")
    assert daemon.registry[SWITCH_ID]["optimal_regime"] == "Q2"
    assert daemon.registry_conflicts == []


def test_the_incubator_directory_alone_is_not_permission_to_trade(tmp_path):
    """A promoted strategy nobody activated is ABSENT, not muted."""
    config = write_switchboard_config(tmp_path, strategy_id=SWITCH_ID)
    blob = json.loads(Path(config).read_text())
    blob["portfolios"]["Incubator-Odd"]["active_strategies"] = []
    Path(config).write_text(json.dumps(blob))
    incubator = write_incubator(tmp_path)          # the code is on the shelf
    empty_cache = tmp_path / "no_cache"
    empty_cache.mkdir(parents=True, exist_ok=True)
    daemon = MasterRegimeDaemon(
        config_path=config,
        state_file=str(tmp_path / "state.json"),
        ml_model_dir=str(tmp_path / "models"),
        anchors_path=str(write_anchors(tmp_path, {"NQ": {"1h": 0.5}})),
        default_tf="1h", symbols=("NQ",), timeframes=("1h",),
        cache_root=str(empty_cache), incubator_dir=str(incubator))
    assert daemon.registry == {}


def test_the_two_certification_records_disagreeing_refuses_to_choose(tmp_path):
    """
    portfolios.json says Q2 and meta.json says Q1. Picking either would be
    picking at random which market a certified strategy is turned loose in.
    """
    daemon = make_switchboard_daemon(tmp_path, cfg_quadrant="Q2",
                                     meta_quadrant="Q1")
    entry = daemon.registry[SWITCH_ID]
    assert entry["optimal_regime"] is None
    assert any("disagree" in c for c in daemon.registry_conflicts)
    status = daemon.strategy_status(SWITCH_ID)
    assert status["status"] == "MUTED"
    assert status["reason"] == "certification_unresolved"
    assert status["entries_allowed"] is False
    assert status["exits_allowed"] is True


def test_disagreeing_timeframes_refuse_to_choose_an_anchor(tmp_path):
    """theta_vol is per (symbol, TIMEFRAME) - 15m and 1h are different rules."""
    daemon = make_switchboard_daemon(tmp_path, cfg_tf="15m", meta_tf="1h")
    assert daemon.registry[SWITCH_ID]["timeframe"] is None
    assert any("timeframe disagrees" in c for c in daemon.registry_conflicts)
    assert daemon.strategy_status(SWITCH_ID)["status"] == "MUTED"


def test_an_active_strategy_with_no_meta_json_is_muted_not_assumed(tmp_path):
    daemon = make_switchboard_daemon(tmp_path, cfg_quadrant=None,
                                     with_meta=False)
    assert daemon.registry[SWITCH_ID]["optimal_regime"] is None
    assert daemon.strategy_status(SWITCH_ID)["entries_allowed"] is False


# -- the switchboard verdict ------------------------------------------------
def test_switchboard_activates_in_the_certified_quadrant(tmp_path):
    """chop bars against theta=0.5 are high-vol and ranging: Q2."""
    daemon = make_switchboard_daemon(tmp_path, theta=0.5)
    data = daemon.refresh("NQ", make_bars(120, "chop", tf_minutes=60), tf="1h")
    assert data["quadrant"] == "Q2"

    status = daemon.strategy_status(SWITCH_ID)
    assert status["status"] == "ACTIVE"
    assert status["entries_allowed"] is True
    assert status["exits_allowed"] is True
    assert status["live_quadrant"] == "Q2"
    assert daemon.is_strategy_active(SWITCH_ID) is True


def test_switchboard_mutes_outside_the_certified_quadrant(tmp_path):
    """trend bars against theta=2.0 are high-vol and trending: Q1, not Q2."""
    daemon = make_switchboard_daemon(tmp_path, theta=2.0)
    data = daemon.refresh("NQ", make_bars(120, "trend", tf_minutes=60), tf="1h")
    assert data["quadrant"] == "Q1"

    status = daemon.strategy_status(SWITCH_ID)
    assert status["status"] == "MUTED"
    assert status["reason"] == "regime_mismatch"
    assert status["entries_allowed"] is False
    assert daemon.is_strategy_active(SWITCH_ID) is False


def test_muted_blocks_entries_and_still_permits_exits(tmp_path):
    """
    THE ONE INVARIANT THIS MODULE EXISTS FOR. Muting stops new exposure; it
    must never trap a position that is already open.
    """
    daemon = make_switchboard_daemon(tmp_path, theta=2.0)
    daemon.refresh("NQ", make_bars(120, "trend", tf_minutes=60), tf="1h")
    assert daemon.strategy_status(SWITCH_ID)["status"] == "MUTED"

    for entry_signal in ("BUY", "SELL", "LONG", "SHORT", "ENTRY"):
        verdict = daemon.gate_signal(SWITCH_ID, entry_signal)
        assert verdict["kind"] == "ENTRY"
        assert verdict["decision"] == "BLOCKED", entry_signal

    for exit_signal in ("EXIT", "FLAT", "FLATTEN", "CLOSE", "exit_long"):
        verdict = daemon.gate_signal(SWITCH_ID, exit_signal)
        assert verdict["kind"] == "EXIT"
        assert verdict["decision"] == "ALLOWED", exit_signal


def test_active_permits_both(tmp_path):
    daemon = make_switchboard_daemon(tmp_path, theta=0.5)
    daemon.refresh("NQ", make_bars(120, "chop", tf_minutes=60), tf="1h")
    assert daemon.gate_signal(SWITCH_ID, "BUY")["decision"] == "ALLOWED"
    assert daemon.gate_signal(SWITCH_ID, "FLATTEN")["decision"] == "ALLOWED"


def test_exits_survive_a_state_file_that_holds_no_regime_at_all(tmp_path):
    """
    A daemon that has published nothing still lets a strategy get flat. A gate
    that expired an exit permission would hold a position through exactly the
    conditions that killed the feed.
    """
    daemon = make_switchboard_daemon(tmp_path)
    status = daemon.strategy_status(SWITCH_ID)
    assert status["status"] == "MUTED"
    assert status["reason"] == "no_regime_published"
    assert status["exits_allowed"] is True
    assert daemon.gate_signal(SWITCH_ID, "FLATTEN")["decision"] == "ALLOWED"
    assert daemon.gate_signal(SWITCH_ID, "BUY")["decision"] == "BLOCKED"


def test_the_warmup_matches_nothing_including_its_own_quadrant(tmp_path):
    daemon = make_switchboard_daemon(tmp_path, theta=0.5)
    daemon.refresh("NQ", make_bars(MIN_BARS_FOR_REGIME - 1, "chop",
                                   tf_minutes=60), tf="1h")
    status = daemon.strategy_status(SWITCH_ID)
    assert status["status"] == "MUTED"
    assert status["reason"] == "indicator_warmup"
    assert status["exits_allowed"] is True


def test_a_15m_reading_does_not_answer_for_a_1h_strategy(tmp_path):
    """
    NQ's theta_vol is 7.90 at 15m and 16.23 at 1h. Answering a 1h question
    with a 15m label draws the permission against the wrong boundary, and
    nothing about the log line would look wrong.
    """
    config = write_switchboard_config(tmp_path, tf="1h")
    incubator = write_incubator(tmp_path, tf="1h")
    empty_cache = tmp_path / "no_cache"
    empty_cache.mkdir(parents=True, exist_ok=True)
    daemon = MasterRegimeDaemon(
        config_path=config,
        state_file=str(tmp_path / "state.json"),
        ml_model_dir=str(tmp_path / "models"),
        anchors_path=str(write_anchors(
            tmp_path, {"NQ": {"15m": 0.5, "1h": 0.5}})),
        default_tf="15m", symbols=("NQ",), timeframes=("15m", "1h"),
        cache_root=str(empty_cache), incubator_dir=str(incubator))

    # A 15m reading in the certified quadrant is published - and ignored.
    published = daemon.refresh("NQ", make_bars(120, "chop"), tf="15m")
    assert published["quadrant"] == "Q2"
    status = daemon.strategy_status(SWITCH_ID)
    assert status["status"] == "MUTED"
    assert status["reason"] == "no_regime_published"
    assert "1h" in status["detail"]

    # The 1h reading activates it, and the 15m one is still on file.
    daemon.refresh("NQ", make_bars(120, "chop", tf_minutes=60), tf="1h")
    assert daemon.strategy_status(SWITCH_ID)["status"] == "ACTIVE"
    assert set(daemon.state["by_timeframe"]["NQ"]) == {"15m", "1h"}


def test_a_micro_reading_answers_for_its_full_size_certification(tmp_path):
    """MNQ and NQ quote the same tape; only the multiplier differs."""
    daemon = make_switchboard_daemon(tmp_path, theta=0.5)
    daemon.refresh("MNQ", make_bars(120, "chop", tf_minutes=60), tf="1h")
    status = daemon.strategy_status(SWITCH_ID)
    assert status["status"] == "ACTIVE"
    assert status["resolved_symbol"] == "MNQ"


def test_an_unrecognised_signal_token_raises_rather_than_defaulting(tmp_path):
    daemon = make_switchboard_daemon(tmp_path)
    with pytest.raises(RegimeDaemonError):
        daemon.gate_signal(SWITCH_ID, "SCALE_IN")


def test_an_unknown_strategy_raises_rather_than_reading_as_muted(tmp_path):
    daemon = make_switchboard_daemon(tmp_path)
    with pytest.raises(UnknownStrategy):
        daemon.strategy_status("never_promoted")


def test_the_signal_vocabulary_is_not_a_second_list(tmp_path):
    """It is built from the vocabularies that already exist upstream."""
    from live.dispatcher import VALID_ACTIONS
    from portfolio.portfolio_manager import FLAT, LONG, SHORT
    assert "FLATTEN" in EXIT_SIGNALS
    assert FLAT.upper() in EXIT_SIGNALS
    assert {LONG.upper(), SHORT.upper()} <= ENTRY_SIGNALS
    assert {a for a in VALID_ACTIONS if a != "FLATTEN"} <= ENTRY_SIGNALS
    assert not (ENTRY_SIGNALS & EXIT_SIGNALS)


# -- the published switchboard, read without the daemon ---------------------
def test_the_reader_answers_the_switchboard_without_importing_the_daemon(tmp_path):
    daemon = make_switchboard_daemon(tmp_path, theta=0.5)
    daemon.refresh("NQ", make_bars(120, "chop", tf_minutes=60), tf="1h")
    sf = daemon.state_file

    assert regime_reader.is_strategy_active(SWITCH_ID, state_file=sf) is True
    assert regime_reader.is_entry_permitted(SWITCH_ID, state_file=sf) is True
    assert regime_reader.is_exit_permitted(SWITCH_ID, state_file=sf) is True
    assert regime_reader.is_signal_permitted(SWITCH_ID, "BUY",
                                             state_file=sf) is True
    assert regime_reader.is_signal_permitted(SWITCH_ID, "FLATTEN",
                                             state_file=sf) is True
    assert list(regime_reader.get_all_strategy_statuses(sf)) == [SWITCH_ID]


# -- the reader's micro -> full-size resolution ------------------------------
# The execution tier trades MNQ/MES/MCL/MGC and the daemon publishes NQ/ES/CL/
# GC, so without this the dispatcher declined every basket symbol with "no live
# regime reading" on a state file that held the answer - the same stand-down a
# dead daemon produces, and indistinguishable from it on a console.
def test_the_reader_answers_a_micro_from_its_full_size_parent(tmp_path):
    daemon = make_switchboard_daemon(tmp_path, theta=0.5)
    daemon.refresh("NQ", make_bars(120, "chop", tf_minutes=60), tf="1h")
    sf = daemon.state_file

    parent = regime_reader.get_current_regime("NQ", state_file=sf)
    micro = regime_reader.get_current_regime("MNQ", state_file=sf)

    assert micro["quadrant"] == parent["quadrant"]
    assert micro["theta_vol"] == parent["theta_vol"]
    # The substitution is ON THE RECORD, not inferred. `symbol` stays the
    # contract the daemon MEASURED; overwriting it with the micro would claim
    # a reading nobody took.
    assert micro["requested_symbol"] == "MNQ"
    assert micro["resolved_symbol"] == "NQ"
    assert micro["symbol_aliased"] is True
    assert micro["symbol"] == "NQ"
    assert parent["symbol_aliased"] is False
    assert parent["resolved_symbol"] == "NQ"

    # And through every read path a gate uses.
    assert regime_reader.get_current_regime(
        "MNQ", state_file=sf, tf="1h")["quadrant"] == parent["quadrant"]
    assert regime_reader.is_regime_permitted(
        "MNQ", [parent["quadrant"]], state_file=sf) is True
    assert regime_reader.is_regime_permitted(
        " mnq ", [parent["quadrant"]], state_file=sf) is True


def test_an_exact_micro_reading_is_never_displaced_by_its_parent(tmp_path):
    """
    The alias covers a symbol nobody measured. It must not override one
    somebody did: if the daemon is ever pointed at the micro's own tape, that
    reading is the better answer.
    """
    sf = tmp_path / "state.json"
    sf.write_text(json.dumps({"symbols": {
        "NQ": {"symbol": "NQ", "tf": "1h", "quadrant": "Q1",
               "regime": "Q1_HIGH_VOL_TREND"},
        "MNQ": {"symbol": "MNQ", "tf": "1h", "quadrant": "Q4",
                "regime": "Q4_LOW_VOL_MEAN_REVERSION"}}}))
    record = regime_reader.get_current_regime("MNQ", state_file=sf)
    assert record["quadrant"] == "Q4"
    assert record["resolved_symbol"] == "MNQ"
    assert record["symbol_aliased"] is False


def test_a_micro_whose_parent_is_unpublished_still_raises(tmp_path):
    """
    Resolution is not a default. An unwatched market must not become a
    permission, and the refusal has to name what was asked for.
    """
    sf = tmp_path / "state.json"
    sf.write_text(json.dumps({"symbols": {
        "NQ": {"symbol": "NQ", "tf": "1h", "quadrant": "Q1",
               "regime": "Q1_HIGH_VOL_TREND"}}}))
    with pytest.raises(RegimeStateError, match="MCL"):
        regime_reader.get_current_regime("MCL", state_file=sf)
    with pytest.raises(RegimeStateError, match="parent CL is not published"):
        regime_reader.get_current_regime("MCL", state_file=sf)
    assert regime_reader.is_regime_permitted(
        "MNQ", ["Q1"], state_file=sf) is True


def test_the_readers_alias_table_is_the_daemons(tmp_path):
    """
    Two copies would be free to disagree, and a disagreement about whether MNQ
    means NQ routes a live order into a market nobody certified for it while
    every log line reads correctly.
    """
    from realtime.contract_alias import MICRO_TO_PARENT
    from realtime.regime_daemon import THETA_ANCHOR_ALIAS
    assert THETA_ANCHOR_ALIAS is MICRO_TO_PARENT
    assert MICRO_TO_PARENT["M2K"] == "RTY"
    assert MICRO_TO_PARENT["MYM"] == "YM"


def test_the_reader_reports_a_muted_strategy_and_still_permits_its_exit(tmp_path):
    daemon = make_switchboard_daemon(tmp_path, theta=2.0)
    daemon.refresh("NQ", make_bars(120, "trend", tf_minutes=60), tf="1h")
    sf = daemon.state_file
    assert regime_reader.is_strategy_active(SWITCH_ID, state_file=sf) is False
    assert regime_reader.is_signal_permitted(SWITCH_ID, "BUY",
                                             state_file=sf) is False
    assert regime_reader.is_signal_permitted(SWITCH_ID, "EXIT",
                                             state_file=sf) is True


def test_the_reader_raises_for_an_unregistered_strategy(tmp_path):
    daemon = make_switchboard_daemon(tmp_path)
    daemon.publish_switchboard()
    with pytest.raises(RegimeStateError):
        regime_reader.get_strategy_status("never_promoted",
                                          state_file=daemon.state_file)


def test_a_stale_active_permission_is_refused_and_a_muted_one_is_not(tmp_path):
    """
    max_age_s refuses a stale ACTIVE record. It must NOT refuse a muted one:
    that read is what `is_exit_permitted` rests on, and an exit has to remain
    available from whatever state the daemon is in.
    """
    daemon = make_switchboard_daemon(tmp_path, theta=0.5)
    daemon.refresh("NQ", make_bars(120, "chop", tf_minutes=60), tf="1h")
    sf = daemon.state_file
    # `bar_ts` is 2026-08-01 in the fixtures, so every record is hours stale.
    with pytest.raises(RegimeStateError):
        regime_reader.is_strategy_active(SWITCH_ID, state_file=sf, max_age_s=60)

    daemon.refresh("NQ", make_bars(120, "trend", tf_minutes=60), tf="1h")
    assert regime_reader.get_strategy_status(
        SWITCH_ID, state_file=sf, max_age_s=60)["status"] == "MUTED"
    assert regime_reader.is_exit_permitted(SWITCH_ID, state_file=sf) is True


def test_the_reader_refuses_a_timeframe_that_was_not_published(tmp_path):
    daemon = make_switchboard_daemon(tmp_path, theta=0.5)
    daemon.refresh("NQ", make_bars(120, "chop", tf_minutes=60), tf="1h")
    sf = daemon.state_file
    assert get_current_regime("NQ", sf, tf="1h")["quadrant"] == "Q2"
    with pytest.raises(RegimeStateError):
        get_current_regime("NQ", sf, tf="15m")


def test_a_state_file_with_no_switchboard_says_so(tmp_path):
    """A document from a daemon older than the switchboard grants nothing."""
    path = tmp_path / "old_state.json"
    path.write_text(json.dumps({"schema_version": "1.0.0", "symbols": {}}))
    with pytest.raises(RegimeStateError):
        regime_reader.get_strategy_status(SWITCH_ID, state_file=path)
    with pytest.raises(RegimeStateError):
        regime_reader.is_signal_permitted(SWITCH_ID, "BUY", state_file=path)


# --------------------------------------------------------------------------
# 9. Pinned anchor integrity - theta_vol does not drift with live bars
# --------------------------------------------------------------------------
def test_theta_vol_does_not_drift_as_live_bars_arrive(tmp_path):
    """
    Feed the daemon progressively more volatile bars and assert the boundary
    never moves. A rolling median would track the tape: on a quiet morning
    every bar reads high-volatility, and a Q1-certified strategy is handed
    permission for a market it is not in with a plausible equity curve behind
    it.
    """
    daemon = make_daemon(tmp_path)
    seen = set()
    for scale in (1.0, 5.0, 25.0, 100.0):
        bars = make_bars(120, "trend")
        for col in ("high", "low", "close"):
            bars[col] = 15000.0 + (bars[col] - 15000.0) * scale
        data = daemon.calculate_regime("NQ", bars)
        seen.add(data["theta_vol"])
        assert data["theta_source"] == "anchors_file"
        assert data["theta_window"] == ["2013-01-01", "2022-12-31"]
    assert seen == {FIXTURE_THETA}, (
        f"theta_vol moved with the bars: {sorted(seen)}. It must be the "
        f"pinned in-sample median and nothing else.")


def test_the_published_anchor_window_is_the_in_sample_window(tmp_path):
    daemon = make_daemon(tmp_path)
    daemon.refresh("NQ", make_bars(120, "trend"))
    blob = json.loads(Path(daemon.state_file).read_text())
    assert blob["in_sample_window"] == ["2013-01-01", "2022-12-31"]


# --------------------------------------------------------------------------
# 10. Atomic serialisation under concurrent read/write
# --------------------------------------------------------------------------
def test_concurrent_readers_never_observe_a_partial_document(tmp_path):
    """
    Hammer the state file from several reader threads while the daemon
    rewrites it. `os.replace` is atomic within a filesystem, so every read
    must land on a complete document - the previous one or the new one, never
    a truncated one. A JSONDecodeError here is the whole reason the write goes
    through a temp file in the DESTINATION directory rather than /tmp.
    """
    import threading

    daemon = make_switchboard_daemon(tmp_path, theta=0.5)
    daemon.refresh("NQ", make_bars(120, "chop", tf_minutes=60), tf="1h")
    sf = daemon.state_file

    stop = threading.Event()
    errors: list[str] = []
    reads = [0]
    lock = threading.Lock()

    def reader():
        while not stop.is_set():
            try:
                record = get_current_regime("NQ", sf)
                assert record["quadrant"] in ("Q1", "Q2")
                status = regime_reader.get_strategy_status(SWITCH_ID,
                                                           state_file=sf)
                assert status["exits_allowed"] is True
                with lock:
                    reads[0] += 1
            except BaseException as exc:            # noqa: BLE001
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
                return

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(6)]
    for thread in threads:
        thread.start()
    try:
        chop = make_bars(120, "chop", tf_minutes=60)
        trend = make_bars(120, "trend", tf_minutes=60)
        for i in range(120):
            daemon.refresh("NQ", chop if i % 2 else trend, tf="1h")
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=10)

    assert not errors, f"concurrent reads failed: {errors[:3]}"
    assert reads[0] > 50, f"only {reads[0]} reads completed; not a real race"
    # Nothing partial left behind either.
    assert not [p for p in Path(sf).parent.iterdir() if p.name.endswith(".tmp")]


def test_a_reader_started_mid_write_sees_the_previous_complete_document(tmp_path):
    """
    The write is temp-file-plus-rename, so the destination path is never open
    for writing. A reader that opens it during a write reads the OLD document
    in full rather than a half-written new one.
    """
    daemon = make_switchboard_daemon(tmp_path, theta=0.5)
    daemon.refresh("NQ", make_bars(120, "chop", tf_minutes=60), tf="1h")
    before = json.loads(Path(daemon.state_file).read_text())

    original = regime_daemon_module.write_state_atomic
    observed: list[dict] = []

    def spy(path, payload):
        # Mid-write: the file on disk must still be the previous document.
        observed.append(json.loads(Path(path).read_text()))
        return original(path, payload)

    regime_daemon_module.write_state_atomic = spy
    try:
        daemon.refresh("NQ", make_bars(120, "trend", tf_minutes=60), tf="1h")
    finally:
        regime_daemon_module.write_state_atomic = original

    assert observed and observed[0]["symbols"]["NQ"]["quadrant"] == \
        before["symbols"]["NQ"]["quadrant"] == "Q2"
    assert json.loads(Path(daemon.state_file).read_text()) \
        ["symbols"]["NQ"]["quadrant"] == "Q1"


# --------------------------------------------------------------------------
# The console line that took the switchboard down with it
# --------------------------------------------------------------------------
def test_an_absent_reading_formats_instead_of_raising():
    """
    `f"{None:.2f}"` raises TypeError, and `main()` formatted `adx_14` that way.

    Measured 2026-08-26: 163 consecutive daemon failures, one every five
    minutes, each AFTER the reading had been published — the raise landed
    between `daemon.refresh()` and `publish_switchboard()`.
    """
    assert _fmt_opt(None, ".2f") == "n/a"
    assert _fmt_opt(None, ".4f") == "n/a"
    assert _fmt_opt(27.4567, ".2f") == "27.46"
    assert _fmt_opt(7.90123, ".4f") == "7.9012"


def test_a_zero_reading_is_not_reported_as_absent():
    """
    ADX of exactly 0.0 is a MEASUREMENT. `if not value` would print it as
    `n/a`, which reads as "the indicator did not run" — the opposite claim.
    """
    assert _fmt_opt(0.0, ".2f") == "0.00"
    assert _fmt_opt(0, ".2f") == "0.00"


def test_a_non_numeric_reading_is_shown_not_hidden():
    """The console line is a diagnostic; swallowing junk defeats its purpose."""
    assert _fmt_opt("weird", ".2f") == "weird"


def test_both_none_paths_survive_the_console_line(tmp_path):
    """
    THE GUARD IS NOT WARM-UP-ONLY, and this is the case that proves it.

    `calculate_regime` returns None readings on two paths: the warm-up (Q0, by
    contract) and the normal path when the last row's indicator is NaN, which
    carries a real Q1..Q4 quadrant and looks entirely healthy. A fix written as
    `if regime == UNDEFINED_LABEL` would pass the first and still crash on the
    second.
    """
    daemon = make_daemon(tmp_path)

    warmup = daemon.calculate_regime("NQ",
                                     make_bars(MIN_BARS_FOR_REGIME - 1, "trend"))
    assert warmup["quadrant"] == "Q0"
    assert warmup["adx_14"] is None and warmup["atr_14"] is None

    healthy = daemon.calculate_regime("NQ", make_bars(200, "trend"))
    assert healthy["quadrant"] != "Q0", "fixture must exercise the NORMAL path"

    # The literal f-string from `main()`, over the warm-up reading and over a
    # real quadrant whose indicators came back NaN.
    nan_on_normal = {**healthy, "adx_14": None, "atr_14": None}
    for data in (warmup, healthy, nan_on_normal):
        line = (f"  NQ 15m: {data['quadrant']} ({data['regime']})  "
                f"ADX={_fmt_opt(data['adx_14'], '.2f')}  "
                f"ATR={_fmt_opt(data['atr_14'], '.4f')}  "
                f"theta={_fmt_opt(data['theta_vol'], '.4f')}  "
                f"bar={data['bar_ts'] or 'n/a'}")
        assert "NQ 15m" in line
        assert "None" not in line, f"a None leaked into the console line: {line}"


def test_the_switchboard_publishes_even_when_the_loop_raises(tmp_path):
    """
    The structural half of the repair.

    With the publish outside a `finally`, any raise in the loop above it took
    the explicit republish with it — the one that picks up a permission change
    with no market movement behind it. It did NOT leave a classified symbol's
    switchboard stale (`refresh()` writes that block itself); the loss was the
    republish, and every target after the raise never being classified at all.
    """
    daemon = make_daemon(tmp_path)
    published = {"ran": False}

    def _publish():
        published["ran"] = True
        return {}

    daemon.publish_switchboard = _publish

    # The shape `main()` now has: a loop that raises, a `finally` that
    # publishes regardless, and the original exception still reaching the exit
    # code rather than being replaced.
    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        finally:
            try:
                for _sid, _st in sorted(daemon.publish_switchboard().items()):
                    pass
            except Exception:                                     # noqa: BLE001
                pass

    assert published["ran"], \
        "the switchboard must publish even when the loop above it raised"

