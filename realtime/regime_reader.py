"""
realtime.regime_reader - the read side of the live regime cache.

Deliberately tiny, and deliberately dependency-light: `json`, `pathlib` and
`datetime`. It does NOT import pandas, pandas_ta, the portfolio config loader
or `realtime.regime_daemon`. An execution script asking "what quadrant is NQ
in" should pay for a 4 KB file read, not for a config validation pass and a
numeric stack - and a reader that imported the writer would fail to answer
whenever the writer's dependencies were the thing that was broken.

Non-blocking means no locks
---------------------------
There are none, and none are needed. The daemon publishes with
`os.replace`, which is atomic within a filesystem: a reader opening the path
gets either the previous complete document or the new complete one. A partial
read is not a state this reader has to defend against, so it never retries and
never waits.

Three things this reader will not do
------------------------------------
* **Return a default.** An unknown symbol, or a state file that does not exist,
  RAISES. The alternative is a dict that reads like an answer -
  `{"regime": None}` or an empty mapping - and downstream that becomes "not in
  the permitted quadrant", which stands a strategy down for a missing file and
  looks identical to a market that moved.
* **Hide staleness.** Every result carries `age_seconds` (against the daemon's
  write clock) and `bar_age_seconds` (against the timestamp of the BAR the
  regime describes). They answer different questions: a daemon looping over a
  dead feed keeps `age_seconds` at zero forever while `bar_age_seconds` grows.
  `max_age_s` turns either into a refusal.
* **Interpret the quadrant.** `is_regime_permitted` compares ids and nothing
  else. Whether Q2 is close enough to Q1 to trade is a decision for whoever
  wrote the portfolio's `regime_quadrants` list, not for a file reader.

`Q0_UNDEFINED_WARMUP` is not a quadrant. It is what the daemon publishes when
the indicators are inside their 14-bar warm-up, and it is never permitted by
`is_regime_permitted` - a permission granted on a regime nobody measured is the
exact failure the quadrant-0 sentinel exists to prevent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_STATE_FILE = "data/live_regime_state.json"

# Published by the daemon when the ADX/ATR warm-up has not completed. Spelled
# here as well as in the daemon because this module refuses to import it - the
# two are pinned equal by `tests/test_regime_daemon.py`.
UNDEFINED_LABEL = "Q0_UNDEFINED_WARMUP"


class RegimeStateError(RuntimeError):
    """The live regime state cannot be read, or holds no answer for a symbol."""


def resolve_state_path(state_file: str | Path = DEFAULT_STATE_FILE) -> Path:
    """
    Resolve the state path the same way for readers and for the writer.

    An ABSOLUTE path is used as given. A RELATIVE one resolves against the
    current directory when a file is already there, and against the repository
    root otherwise - so `data/live_regime_state.json` means the same file
    whether a script runs from the repo root, from `realtime/`, or from a cron
    job with no working directory to speak of. Without this rule a daemon
    started from the wrong directory publishes to a second `data/` tree and
    every reader reports the market as unknown while the daemon logs success.
    """
    path = Path(state_file)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return REPO_ROOT / path


def _load(state_file: str | Path) -> tuple[dict[str, Any], Path]:
    path = resolve_state_path(state_file)
    if not path.is_file():
        raise RegimeStateError(
            f"no live regime state at {path}. The Master Regime Daemon has not "
            f"published yet. This is not 'no regime' - nothing has been "
            f"measured, so no strategy may be permitted or stood down on it.")
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RegimeStateError(f"{path} could not be read: {exc}") from None
    if not isinstance(blob, dict) or "symbols" not in blob:
        raise RegimeStateError(
            f"{path} is not a live regime state document (no `symbols` key).")
    return blob, path


def _age_seconds(stamp: Any) -> float | None:
    """Seconds between `stamp` and now, or None when it is unparseable."""
    if not stamp:
        return None
    try:
        when = datetime.fromisoformat(str(stamp))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds()


def get_current_regime(symbol: str,
                       state_file: str | Path = DEFAULT_STATE_FILE,
                       max_age_s: float | None = None) -> dict[str, Any]:
    """
    The latest published regime for one symbol.

    Returns the daemon's record verbatim, plus:

        `age_seconds`      since the daemon last WROTE this symbol
        `bar_age_seconds`  since the timestamp of the BAR it describes
        `state_file`       which file answered, fully resolved

    `max_age_s` refuses a record older than that many seconds, measured on
    whichever of the two ages is LARGER - a fresh write over a frozen feed is
    stale, and so is a live feed the daemon stopped reading. Left None, the
    ages are reported and nothing is refused, because how old is too old
    depends on the timeframe: 90 seconds is stale on 1m bars and current on
    30m ones, and this module does not know which is being traded.

    Raises `RegimeStateError` for a missing file, an unknown symbol, or a
    record refused by `max_age_s`. It never returns a placeholder.
    """
    blob, path = _load(state_file)
    sym = str(symbol).strip().upper()
    symbols = blob.get("symbols") or {}
    if sym not in symbols:
        raise RegimeStateError(
            f"no regime published for {sym!r} in {path}. Published: "
            f"{sorted(symbols)}. An unpublished symbol is not a symbol in an "
            f"unknown regime - it is one the daemon is not watching.")

    record = dict(symbols[sym])
    record["age_seconds"] = _age_seconds(record.get("written_at")
                                         or record.get("updated_at"))
    record["bar_age_seconds"] = _age_seconds(record.get("bar_ts"))
    record["state_file"] = str(path)

    if max_age_s is not None:
        ages = [a for a in (record["age_seconds"], record["bar_age_seconds"])
                if a is not None]
        worst = max(ages) if ages else None
        if worst is None:
            raise RegimeStateError(
                f"{sym} carries no readable timestamp, so its age cannot be "
                f"checked against max_age_s={max_age_s}.")
        if worst > max_age_s:
            raise RegimeStateError(
                f"{sym} regime is {worst:.0f}s old (write "
                f"{record['age_seconds']}, bar {record['bar_age_seconds']}), "
                f"past max_age_s={max_age_s}. Refusing to hand back a stale "
                f"quadrant: a permission granted on it is a permission for a "
                f"market that has since moved.")
    return record


def get_all_regimes(state_file: str | Path = DEFAULT_STATE_FILE
                    ) -> dict[str, dict[str, Any]]:
    """Every published symbol, each enriched exactly as `get_current_regime`."""
    blob, path = _load(state_file)
    return {sym: get_current_regime(sym, state_file=path)
            for sym in sorted(blob.get("symbols") or {})}


def is_regime_permitted(symbol: str,
                        allowed_quadrants,
                        state_file: str | Path = DEFAULT_STATE_FILE,
                        max_age_s: float | None = None) -> bool:
    """
    True when the symbol's live quadrant is in `allowed_quadrants`.

    Accepts either form of identifier in the list - the `Q1`..`Q4` id or the
    schema label `Q1_HIGH_VOL_TREND` - because a portfolio's
    `regime_quadrants` is written in labels and a Stage 1 handoff carries ids,
    and a comparison that accepted only one of them would silently never match
    for the other.

    `Q0_UNDEFINED_WARMUP` is never permitted, even if it appears in the list:
    it means the indicators had not warmed up, and permission to trade must not
    be derived from a regime nobody measured.
    """
    record = get_current_regime(symbol, state_file=state_file,
                                max_age_s=max_age_s)
    quadrant = str(record.get("quadrant") or "")
    label = str(record.get("regime") or "")
    if label == UNDEFINED_LABEL or quadrant in ("Q0", ""):
        return False
    wanted = {str(q).strip() for q in (allowed_quadrants or [])}
    return quadrant in wanted or label in wanted


def main(argv: list[str] | None = None) -> int:
    """`python3 realtime/regime_reader.py [--symbol NQ]` - print the cache."""
    import argparse

    ap = argparse.ArgumentParser(description="Read the live regime state.")
    ap.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    ap.add_argument("--symbol", default=None)
    args = ap.parse_args(argv)

    try:
        if args.symbol:
            records = {args.symbol.upper():
                       get_current_regime(args.symbol, args.state_file)}
        else:
            records = get_all_regimes(args.state_file)
    except RegimeStateError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not records:
        print("no symbols published")
        return 1
    print(f"{'SYMBOL':<8}{'TF':<6}{'QUADRANT':<28}{'ADX':>8}{'ATR':>10}"
          f"{'THETA':>10}{'BAR AGE':>10}")
    for sym, rec in records.items():
        adx = rec.get("adx_14")
        atr = rec.get("atr_14")
        age = rec.get("bar_age_seconds")
        print(f"{sym:<8}{str(rec.get('tf')):<6}{str(rec.get('regime')):<28}"
              f"{(f'{adx:.2f}' if adx is not None else '--'):>8}"
              f"{(f'{atr:.4f}' if atr is not None else '--'):>10}"
              f"{rec.get('theta_vol', float('nan')):>10.4f}"
              f"{(f'{age:.0f}s' if age is not None else '--'):>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
