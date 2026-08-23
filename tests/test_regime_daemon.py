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
from realtime.regime_daemon import (                            # noqa: E402
    MIN_BARS_FOR_REGIME,
    QUADRANT_TO_LABEL,
    UNDEFINED_LABEL,
    MLGateError,
    MasterRegimeDaemon,
    RegimeDaemonError,
    ThetaAnchorMissing,
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
