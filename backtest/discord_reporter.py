"""
backtest.discord_reporter - post a pipeline card to a Discord webhook.

Location:  ~/src/trading/backtest/discord_reporter.py

Four cards, one transport
-------------------------
- **`--mode promotion`** (the default, and `--stage 5`): the handful of numbers
  a promotion decision rests on, passed in on the command line by whatever
  produced them (Stage 3's `gate_audit_<SYMBOL>.json`, Stage 5's `meta.json`).
- **`--mode baseline`** (equivalently `--stage 1`): Stage 1's REGIME FIREWALL
  leaderboard, read straight out of `surviving_assets.json` - every
  (symbol, timeframe) configuration screened, the quadrant it cleared, that
  quadrant's profit factor and trade count, and whether it was PROMOTED to
  Stage 2 or DROPPED.
- **`--mode scan`** (equivalently `--stage 2`): Stage 2's PARAMETER
  OPTIMIZATION summary, read straight out of `stage2_summary.json` - the
  in-sample window, and per configuration the symbol, timeframe, target regime
  quadrant, the selected best parameters, the in-sample profit factor and the
  max drawdown. Its parameter sets are printed TWICE and deliberately: once in
  the table with the keys abbreviated (`f=5 s=50 tp=1.5 sl=1.0`) so the
  fixed-width columns stay aligned, and once below it under
  `Optimized Parameters (Full)` with every key spelled as the strategy
  declared it and nothing clipped. The table is what a reader scans; the block
  is what they retype into `--param`, and a clipped parameter set is the one
  thing on this card that would be acted on while wrong.
- **`--mode audit`** (equivalently `--stage 3`): Stage 3's GATE AUDIT AND
  CERTIFICATION, read straight out of `stage3_audit_summary.json` - both
  windows, and per configuration the symbol, timeframe, target regime
  quadrant, Gate R's verdict, the quadrant profit factor and trade count it
  was measured on, the in-sample and out-of-sample blended profit factors side
  by side, and the SHA-256 seal. Three profit factors per row, each labelled,
  because only ONE of them decided anything: Gate R scores the quadrant, and
  the other two are the blended-sample pair that says whether the edge
  collapsed. The seals are printed twice for the same reason the Stage 2
  parameter sets are - a 12-character prefix in the table to keep the columns
  aligned, and all three hashes in full below it, which is the copy a reader
  checks against a promoted `meta.json`.

Four cards now, and the reason the count keeps growing is that each one
announces a DIFFERENT decision. A Stage 2 card is not a promotion and not a
screen: every configuration on it advanced, because Stage 2 prunes nothing, and
the card says so rather than letting a reader infer a survival rate from a
leaderboard's length. A Stage 3 card is the first that carries a PASS - and
its `Certified` count is Gate R's, not a roll-up of the three advisory gates
printed beside it.

It reads no bars, opens no lake file, and computes nothing. The Stage 1 card
reads a handoff, which is not the same thing: every number on it is one Stage 1
wrote, transcribed. That is deliberate - **LLMs never do math and neither does
a notifier**. A reporter that recomputed a profit factor would be free to
disagree with the stage it is announcing, and the two would be compared by
nobody. In particular the card does not re-apply the survival hurdle: it prints
the `status` Stage 1 recorded, so a card can never promote a configuration the
stage dropped.

What it will not do
-------------------
- **It does not decide anything.** Nothing here checks a gate or a screening
  hurdle, so a card can be posted for a strategy that was never certified. The
  card announces what it was told; `backtest/promote.py` is what refuses an
  uncertified version and `backtest/baseline.py` is what decides PROMOTED from
  DROPPED. The Stage 2 card in particular re-ranks nothing: `backtest/scan.py`
  chose the winning parameter set off the Sharpe plateau, and this transcribes
  the row it wrote. The Stage 3 card re-scores nothing: it prints the
  `gate_regime` status and the `certified` flag `backtest/audit_gates.py`
  recorded, so it can never announce a certification the audit refused.
- **It does not invent a missing number.** `--pf` and `--dd` are taken as text,
  not floats. A numeric value is formatted (`1.42`, `-8.30 %`) and anything
  else - `NOT EVALUATED`, `n/a` - is printed verbatim. Coercing those to 0.0
  would put a zero drawdown on a card for a run whose drawdown nobody measured,
  which is the one failure mode a status notifier can cause on its own.
- **It does not raise on a transport failure.** A dead webhook must not take
  down whatever called it; the outcome is printed and returned in the exit
  code. Same reasoning as `live/dispatcher.send_execution_signal`.

The webhook URL is a credential
-------------------------------
It is never printed, never echoed into a log line, and never included in the
failure message - only its host is. Anyone holding the full URL can post to the
channel. `$BT_DISCORD_WEBHOOK` supplies it when `--webhook` is omitted, so it
does not have to sit in shell history.

Discord specifics that are easy to get wrong
--------------------------------------------
- A successful webhook POST returns **204 No Content**, not 200. Treating only
  200 as success reports every successful post as a failure.
- `color` is a decimal integer, not a CSS string. Emerald green is 0x2ECC71.
- Embed field values cap at 1024 characters and the whole embed at 6000; a
  local artifact path is well inside that, but the value is truncated rather
  than sent to be rejected with a 400 nobody reads.
- Only an http(s) URL renders as a link. A `/mnt/backtest/...` path is not
  clickable in any client, so it is rendered as inline code instead of as a
  markdown link that would look broken.
- An embed **description** caps at 4096 characters and an embed carries at most
  25 fields. A full screen is 27 contracts x 4 timeframes = 108 rows, which
  fits in neither as fields; the Stage 1 leaderboard is therefore one
  fixed-width block in the description, truncated to `STAGE1_MAX_ROWS` with the
  count of what was left off printed on the card. A silently shortened
  leaderboard reads as a complete one, which is the same failure the HTML trade
  log's row cap exists to avoid.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.pipeline import (STAGE2_SUMMARY_FILE,               # noqa: E402
                               STAGE3_SUMMARY_FILE, SURVIVORS_FILE,
                               pipeline_dir, read_stage)

# Emerald green. Discord wants a decimal int; 0x2ECC71 == 3066993.
EMERALD_GREEN = 0x2ECC71
# The Stage 1 card is a different colour from the promotion card on purpose: a
# screening leaderboard and a promotion are read very differently, and in a
# channel that carries both, colour is the only thing distinguishing them at a
# glance. Slate blue - deliberately not green, because nothing on the Stage 1
# card is an approval to trade.
SLATE_BLUE = 0x3498DB
# Nothing survived. Amber rather than red: an empty screen is a result about
# the idea on these contracts, not an error.
AMBER = 0xE67E22

# Discord's documented limits. Exceeding any of them is a 400.
MAX_FIELD_VALUE = 1024
MAX_EMBED_TOTAL = 6000
MAX_EMBED_DESCRIPTION = 4096

# Leaderboard rows that fit the description block with room for the header, the
# legend and the truncation note. A full screen is 108 configurations; what is
# left off is COUNTED on the card and the handoff path is printed beside it.
STAGE1_MAX_ROWS = 40

# The Stage 2 card carries a parameter set per row, which is far wider than a
# Stage 1 row, so fewer of them fit the 4096-character description. Whatever
# does not fit is COUNTED on the card, exactly as it is on the Stage 1 card:
# a silently shortened optimisation summary reads as the whole stage, and
# Stage 2 optimises EVERY survivor, so its table's length is the one number a
# reader is entitled to trust.
STAGE2_MAX_ROWS = 20

# Parameter sets get long once risk axes are swept, and a fixed-width table
# column that reflows is harder to read than a clipped one. So the TABLE cell
# carries the parameter set with its keys abbreviated (`f=5 s=50 tp=1.5`) and
# is clipped past this many characters - and every row's UNABBREVIATED,
# UNCLIPPED parameter set is printed below the table in its own field. The
# table is for scanning; that block is the copy a reader retypes.
STAGE2_MAX_PARAM_CHARS = 46
ELLIPSIS = "\u2026"

# The full-parameter block below the table. One field per chunk, each inside
# Discord's 1024-character field cap, and at most this many chunks - the embed
# already carries six other fields and Discord caps an embed at 25 fields and
# 6000 characters.
STAGE2_PARAM_FIELD_NAME = "Optimized Parameters (Full)"
STAGE2_PARAM_MAX_FIELDS = 6
FENCE_OPEN = "```text\n"
FENCE_CLOSE = "\n```"
# Characters held back from the embed budget for the "N further ..." note, so
# a block that had to leave rows out can always say so.
NOTE_RESERVE = 96
# Continuation indent for a parameter set too wide for one field.
INDENT_WIDTH = 4

# The line under the table that says its cells are abbreviated, in its two
# forms: the full sets are below, or - when the embed had no room for them -
# they are in the handoff. The second is deliberately the SHORTER string, so
# swapping it in after the budget has been measured can only shrink the embed.
ABBREV_NOTE = ("_Table parameter keys are abbreviated \u2014 the full "
               "key=value sets are below._")
ABBREV_NOTE_NO_BLOCK = ("_Table parameter keys are abbreviated \u2014 the "
                        "full sets are in the handoff._")

# Tokens that say what KIND of parameter something is rather than which one it
# is. Every period is a period and every stop multiple is quoted in ATRs, so
# inside one parameter set they distinguish nothing while costing most of the
# column width. Dropped only to build the TABLE's abbreviation; the full block
# prints the key exactly as the strategy declared it.
PARAM_NOISE_TOKENS = frozenset({
    "window", "windows", "period", "periods", "length", "len", "lookback",
    "mult", "multiple", "multiplier", "factor", "atr", "bars", "num",
})

# Stage 2's own colour, distinct from the Stage 1 slate and the promotion
# green. Violet, and deliberately not green: an optimised parameter set is not
# an approval to trade, and in a channel carrying all three cards the colour is
# what separates them at a glance.
VIOLET = 0x9B59B6

# Stage 2 optimised the configuration; the sweep raised and it did not.
OPTIMIZED = "OPTIMIZED"
ERRORED = "ERROR"

# Stage 3's own colour. Teal, and once again deliberately not the promotion
# green: a certification is a verdict about a holdout, not a decision to trade.
# The four cards in a channel are slate (screen), violet (sweep), teal
# (certification) and green (promotion), which is the only thing separating
# them at a glance in a scrollback.
TEAL = 0x1ABC9C

# A Stage 3 row carries a quadrant, two profit factors, two trade counts and a
# hash prefix, so it is wider than a Stage 1 row and narrower than a Stage 2
# one. Whatever does not fit is COUNTED on the card, as everywhere else here.
STAGE3_MAX_ROWS = 24

# How much of each SHA-256 goes on the card. Twelve hex characters is 48 bits -
# enough to tell two builds of the same strategy apart at a glance, which is
# what a reader uses it for. The full 64 are in the seal, and the card says so:
# a truncated hash presented as the hash is a checksum nobody can verify.
SEAL_PREFIX_CHARS = 12

# The seal block below the Stage 3 table: at most this many fields, each inside
# Discord's 1024-character cap. The embed already carries six other fields and
# Discord caps an embed at 25 fields and 6000 characters.
STAGE3_SEAL_MAX_FIELDS = 6

# The tokens Stage 3 records per configuration.
CERTIFIED = "PASS"
NOT_AUDITED = "NOT AUDITED"

# The Stage 1 handoff, and the two words it records per configuration.
PROMOTED = "PROMOTED"
DROPPED = "DROPPED"

# A webhook POST succeeds with 204 No Content. With ?wait=true it is 200 and the
# body is the created message, so both are accepted.
SUCCESS_STATUS = frozenset({200, 204})

POST_TIMEOUT_SECONDS = 10.0

ENV_WEBHOOK = "BT_DISCORD_WEBHOOK"


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def _fmt_number(raw: str, suffix: str = "", decimals: int = 2) -> str:
    """
    Format a numeric argument, or pass a non-numeric token through untouched.

    `NOT EVALUATED` and `1.42` are different statements about a backtest and
    must not collapse into one. Anything float() rejects is returned verbatim.
    """
    text = (raw or "").strip()
    if not text:
        return "NOT REPORTED"
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text
    return f"{value:.{decimals}f}{suffix}"


def _fmt_report(raw: str) -> str:
    """
    Render the artifact reference. An http(s) URL becomes a markdown link; a
    filesystem path becomes inline code, because a path is not clickable and a
    link that never opens reads as a broken report rather than a local file.
    """
    text = (raw or "").strip()
    if not text:
        return "NOT RECORDED"
    scheme = urlparse(text).scheme.lower()
    if scheme in ("http", "https"):
        rendered = f"[Open report]({text})"
    else:
        rendered = f"`{text}`"
    if len(rendered) > MAX_FIELD_VALUE:
        rendered = rendered[: MAX_FIELD_VALUE - 3] + "..."
    return rendered


def build_embed(
    strat: str,
    symbol: str,
    tf: str,
    pf: str,
    dd: str,
    regime: str,
    report: str,
) -> dict[str, Any]:
    """Build the Discord embed dict. Pure - sends nothing, reads nothing."""
    return {
        "title": f"\U0001F680 Incubation Promotion: {strat}",
        "color": EMERALD_GREEN,
        "fields": [
            {
                "name": "Asset / Timeframe",
                "value": f"**{symbol}** · `{tf}`",
                "inline": True,
            },
            {
                "name": "Out-of-Sample PF",
                "value": _fmt_number(pf),
                "inline": True,
            },
            {
                "name": "Max Drawdown",
                "value": _fmt_number(dd, suffix=" %"),
                "inline": True,
            },
            {
                "name": "Certified Regime Firewall",
                "value": (regime or "").strip() or "NOT DECLARED",
                "inline": False,
            },
            {
                "name": "Artifacts / Report",
                "value": _fmt_report(report),
                "inline": False,
            },
        ],
        "footer": {"text": "backtest/discord_reporter.py · values as supplied, not recomputed"},
    }




# --------------------------------------------------------------------------
# Stage 1 · the regime-firewall leaderboard
# --------------------------------------------------------------------------

def _fmt_metric(value: Any, decimals: int = 2) -> str:
    """
    One leaderboard cell. A float is formatted; a token is passed through; an
    absent value renders as `--`.

    `--` rather than `0.00`: a dropped configuration has no winning quadrant,
    so it has no quadrant profit factor, and a zero in that column reads as a
    quadrant that was measured and found worthless. Those are different
    findings and the table has to keep them apart.
    """
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_count(value: Any) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def default_survivors_path(strat: str, out_dir: str | None = None) -> Path:
    """`<BT_ARTIFACTS>/pipeline/<strategy>/surviving_assets.json`."""
    return pipeline_dir(strat, out_dir) / SURVIVORS_FILE


def load_stage1(path: str | Path, strat: str | None = None) -> dict[str, Any]:
    """
    Read Stage 1's handoff, and refuse the wrong one.

    Delegated to `pipeline.read_stage` rather than parsed here, because that is
    where "this file was written by stage 3, not stage 1" and "this belongs to
    another strategy" are already refusals. A notifier with its own laxer
    reader would happily post one strategy's screen under another's name, and
    a Discord card is exactly the artifact nobody cross-checks.
    """
    return read_stage(Path(path), 1, strat)


def stage1_rows(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Every configuration the screen EVALUATED, in one shape.

    `screen_results` is what Stage 1 writes for this purpose and is used when
    present. A handoff written before that field existed is reassembled from
    `surviving_pairs` + `dropped`, and the reassembly is deliberately lossy in
    a visible way: those older dropped entries carry no version and no
    quadrant, so the cells render `--` rather than being back-filled with a
    plausible value. A leaderboard that silently invented a version column is
    worse than one that says it does not have it.
    """
    results = blob.get("screen_results")
    if isinstance(results, list) and results:
        return [dict(r) for r in results if isinstance(r, dict)]

    rows: list[dict[str, Any]] = []
    for pair in blob.get("surviving_pairs") or []:
        if not isinstance(pair, dict):
            continue
        rows.append({**pair, "tf": pair.get("tf"),
                     "status": pair.get("status") or PROMOTED})
    for drop in blob.get("dropped") or []:
        if not isinstance(drop, dict):
            continue
        rows.append({"symbol": drop.get("symbol"),
                     # Stage 1's `dropped` list keys the timeframe as
                     # `timeframe`; `surviving_pairs` keys it as `tf`. Both are
                     # read, neither is renamed in the handoff, because the two
                     # lists are read by different consumers.
                     "tf": drop.get("tf") or drop.get("timeframe"),
                     "status": drop.get("status") or DROPPED,
                     "optimal_regime": drop.get("optimal_regime"),
                     "reason": drop.get("reason")})
    return rows


def _sort_key(row: dict[str, Any]) -> tuple:
    """
    Promoted first, by the quadrant profit factor the screen decided on,
    descending; then the drops, alphabetically.

    Sorted on the REGIME profit factor rather than on either version's blended
    one, for the same reason Stage 1's own leaderboard is: the blend is not
    what advanced the configuration, and ranking on it would put a broad
    mediocre edge above a sharp one in a single environment.
    """
    promoted = str(row.get("status") or "").upper() == PROMOTED
    try:
        pf = float(row.get("regime_pf"))
    except (TypeError, ValueError):
        pf = float("-inf")
    return (0 if promoted else 1, -pf if promoted else 0.0,
            str(row.get("symbol") or ""), str(row.get("tf") or ""))


def format_stage1_table(rows: list[dict[str, Any]],
                        max_rows: int = STAGE1_MAX_ROWS
                        ) -> tuple[str, int, dict[str, str]]:
    """
    The leaderboard as one fixed-width block, plus the quadrant legend.

    Returns `(text, hidden, legend)`. `hidden` is how many rows did not fit and
    is printed on the card by the caller - a leaderboard truncated in silence
    reads as the whole screen.

    The QUAD column carries the `Q1`..`Q4` id Stage 1 recorded and the legend
    underneath maps only the ids that actually appear, built FROM the rows. No
    short-name table lives in this module: a second spelling of "High
    Volatility / Trending" here would be free to disagree with the one
    `mdlib.regimes` numbers, and a card naming the wrong environment is the
    kind of error that is only ever caught in live trading.
    """
    header = ["SYMBOL", "TF", "VER", "QUAD", "REGIME PF", "N", "STATUS"]
    body: list[list[str]] = []
    legend: dict[str, str] = {}

    ordered = sorted(rows, key=_sort_key)
    shown = ordered[: max(0, int(max_rows))]
    for row in shown:
        quad = row.get("quadrant")
        regime = row.get("optimal_regime")
        if quad and regime:
            legend[str(quad)] = str(regime)
        body.append([
            str(row.get("symbol") or "?"),
            str(row.get("tf") or "?"),
            # `V` + the version letter, or `--`. A blank cell here would read
            # as Version A, which is a claim about which twin carried the
            # configuration.
            f"V{row['version']}" if row.get("version") else "--",
            str(quad) if quad else "--",
            _fmt_metric(row.get("regime_pf")),
            _fmt_count(row.get("regime_trade_count")),
            str(row.get("status") or "?").upper(),
        ])

    widths = [max(len(header[i]), *(len(r[i]) for r in body)) if body
              else len(header[i]) for i in range(len(header))]
    align = ["<", "<", "<", "<", ">", ">", "<"]

    def line(cells: list[str]) -> str:
        return "  ".join(format(c, f"{align[i]}{widths[i]}")
                         for i, c in enumerate(cells)).rstrip()

    out = [line(header), line(["-" * w for w in widths])]
    out.extend(line(r) for r in body)
    return "\n".join(out), len(ordered) - len(shown), legend


def build_stage1_embed(strat: str, blob: dict[str, Any],
                       source: str | Path | None = None,
                       max_rows: int = STAGE1_MAX_ROWS) -> dict[str, Any]:
    """
    Stage 1's card. Pure - sends nothing, and every number on it is read off
    the handoff rather than derived from it.
    """
    rows = stage1_rows(blob)
    promoted = [r for r in rows
                if str(r.get("status") or "").upper() == PROMOTED]
    table, hidden, legend = format_stage1_table(rows, max_rows)

    window = blob.get("in_sample_window") or {}
    start = window.get("start") or blob.get("start") or "lake start"
    end = window.get("end") or blob.get("end") or "lake end"
    timeframes = blob.get("timeframes") or (
        [blob["timeframe"]] if blob.get("timeframe") else [])
    criterion = blob.get("criterion") or "not recorded"
    ml = blob.get("ml_evaluated")

    description = [
        f"**In-sample window** `{start} → {end}`",
        f"**Screen** {criterion}",
        "```text",
        table if table.strip() else "no configuration was evaluated",
        "```",
    ]
    if legend:
        description.append("**Quadrants** " + " · ".join(
            f"`{q}` {legend[q]}" for q in sorted(legend)))
    if hidden:
        description.append(
            f"_{hidden} further configuration(s) are not shown — the full "
            f"screen is in the handoff._")

    text = "\n".join(description)
    if len(text) > MAX_EMBED_DESCRIPTION:
        # Trim the TABLE, never the header lines: the window and the screening
        # rule are what make the numbers readable at all, and a card that lost
        # them to a truncation is a leaderboard of unlabelled figures.
        keep = MAX_EMBED_DESCRIPTION - 64
        text = text[:keep] + "\n```\n_truncated — see the handoff._"

    fields = [
        {"name": "Evaluated", "value": str(len(rows)), "inline": True},
        {"name": "Promoted → Stage 2", "value": str(len(promoted)),
         "inline": True},
        {"name": "Dropped", "value": str(len(rows) - len(promoted)),
         "inline": True},
        {"name": "Timeframes",
         "value": ", ".join(f"`{t}`" for t in timeframes) or "not recorded",
         "inline": True},
        # Three states, never two. `--no-ml` means survival was decided on
        # Version A alone, which is a narrower claim than one both versions
        # were given a chance at - and an older handoff that recorded nothing
        # must not be reported as either.
        {"name": "Version B",
         "value": ("evaluated" if ml is True
                   else "NOT RUN" if ml is False else "not recorded"),
         "inline": True},
        {"name": "Handoff", "value": _fmt_report(str(source or "")),
         "inline": False},
    ]

    return {
        "title": f"\U0001F9ED Stage 1 · Regime Firewall: {strat}",
        "description": text,
        "color": SLATE_BLUE if promoted else AMBER,
        "fields": fields,
        "footer": {"text": "backtest/discord_reporter.py · Stage 1 screen · "
                           "values as recorded by baseline.py, not recomputed"},
    }


# --------------------------------------------------------------------------
# Stage 2 · parameter optimization
# --------------------------------------------------------------------------

def default_scan_summary_path(strat: str, out_dir: str | None = None) -> Path:
    """`<BT_ARTIFACTS>/pipeline/<strategy>/stage2_summary.json`."""
    return pipeline_dir(strat, out_dir) / STAGE2_SUMMARY_FILE


def load_stage2(path: str | Path, strat: str | None = None) -> dict[str, Any]:
    """
    Read Stage 2's summary handoff, and refuse the wrong one.

    Through `pipeline.read_stage` for the same reason Stage 1's card is: it is
    where "written by another stage" and "belongs to another strategy" are
    already refusals. A card announcing one strategy's optimised parameters
    under another's name would be believed - nobody re-derives a Discord post.
    """
    return read_stage(Path(path), 2, strat)


def stage2_rows(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Every configuration Stage 2 was asked to optimise, in one shape.

    `results` is the summary matrix Stage 2 writes for exactly this purpose.
    Errored configurations are included and keep their `ERROR` status: Stage 2
    prunes nothing, so a row missing from the card is a configuration whose
    absence has to be explained, not one that quietly failed a hurdle.
    """
    results = blob.get("results")
    if isinstance(results, list):
        return [dict(r) for r in results if isinstance(r, dict)]
    return []


# `fast_window=5, slow_window=50` -> [("fast_window", "5"), ...]. Split only
# where a comma is followed by an identifier and an `=`, so a value that
# itself contains a comma - `windows=(5, 10)` - is not torn in half.
_PARAM_SPLIT = re.compile(r",\s*(?=[A-Za-z_][A-Za-z0-9_]*\s*=)")


def parse_param_pairs(value: Any) -> list[tuple[str, str]]:
    """
    Split Stage 2's `k=v, k=v` parameter string into pairs, TEXTUALLY.

    No types are restored and no value is reformatted: `False`, `None` and
    `1.5` reach the card as the characters Stage 2 wrote. This module is a
    transcription, and a reporter that parsed `None` into a float would be
    free to print a take-profit that was never modelled.

    Returns `[]` for anything that is not a parameter list - `NOT OPTIMIZED`,
    `(no winner)`, an empty cell - so the caller passes those through verbatim
    instead of rendering them as a malformed pair.
    """
    text = str(value or "").strip()
    if not text or "=" not in text:
        return []
    pairs: list[tuple[str, str]] = []
    for chunk in _PARAM_SPLIT.split(text):
        key, sep, val = chunk.partition("=")
        if not sep or not key.strip():
            return []
        pairs.append((key.strip(), val.strip()))
    return pairs


def _abbrev_key(key: str) -> str:
    """One key, shortened: `fast_window` -> `f`, `tp_atr_mult` -> `tp`."""
    tokens = [t for t in key.split("_") if t]
    if not tokens:
        return key
    significant = [t for t in tokens if t.lower() not in PARAM_NOISE_TOKENS]
    if not significant:
        # Every token was noise (`atr_mult`); keep them rather than return an
        # empty name, and let the collision rule below decide if it is unique.
        significant = tokens
    if len(significant) == 1:
        token = significant[0]
        return token if len(token) <= 3 else token[0]
    return "".join(t[0] for t in significant)


def abbreviate_param_keys(keys: list[str]) -> dict[str, str]:
    """
    Map each key to its table abbreviation, refusing to collapse two keys into
    one.

    A collision gives EVERY key that collided its full name back rather than
    numbering them: `sl_atr_mult` and `slow_window` both shortening to `s` and
    being told apart by a trailing `1` is exactly how a stop distance gets read
    as a moving-average length. A wider column is the cheap failure.
    """
    proposed = {k: _abbrev_key(k) for k in keys}
    taken: dict[str, list[str]] = {}
    for key, short in proposed.items():
        taken.setdefault(short, []).append(key)
    return {k: (short if len(taken[short]) == 1 else k)
            for k, short in proposed.items()}


def compact_params(value: Any) -> str:
    """
    The parameter set as a table cell: keys abbreviated, values verbatim,
    space separated (`f=5 s=50 tp=1.5 sl=1.0 t=False`).

    Nothing is dropped - every parameter Stage 2 recorded is on the cell, only
    its NAME is shortened, and the full names are printed below the table. A
    cell that omitted a parameter would describe a run nobody performed.
    """
    pairs = parse_param_pairs(value)
    if not pairs:
        return str(value or "").strip()
    short = abbreviate_param_keys([k for k, _ in pairs])
    return " ".join(f"{short[k]}={v}" for k, v in pairs)


def _fmt_params(value: Any, limit: int = STAGE2_MAX_PARAM_CHARS) -> str:
    """
    A parameter set for one table cell, abbreviated and then clipped.

    Never `--`: a row that reached the card without parameters is either an
    errored sweep (which says so) or a grid that produced no measurable Sharpe,
    and both are findings. `NOT OPTIMIZED` and `(no winner)` are written by
    Stage 2 and passed through verbatim.

    The ellipsis is a backstop now rather than the card's answer to a wide
    parameter set: whatever is clipped here is printed in full, under its real
    key names, in the block below the table.
    """
    text = compact_params(value)
    if not text:
        return "not recorded"
    return text if len(text) <= limit else text[: limit - 1] + ELLIPSIS


def _stage2_sort_key(row: dict[str, Any]) -> tuple:
    """
    The card's row order: optimised configurations by in-sample profit factor
    descending, errored ones last, ties by symbol and timeframe.

    One function because the table and the full-parameter block below it are
    read as the same list - row three of one has to be row three of the other,
    and two sorts would be free to disagree about which contract that is.
    """
    ok = str(row.get("status") or "").upper() == OPTIMIZED
    try:
        pf = float(row.get("profit_factor"))
    except (TypeError, ValueError):
        pf = float("-inf")
    return (0 if ok else 1, -pf if ok else 0.0,
            str(row.get("symbol") or ""), str(row.get("timeframe") or ""))


def format_stage2_table(rows: list[dict[str, Any]],
                        max_rows: int = STAGE2_MAX_ROWS
                        ) -> tuple[str, int, dict[str, str]]:
    """
    The optimisation summary as one fixed-width block, plus the quadrant legend.

    Ordered by in-sample profit factor descending, which is the metric on the
    card - not by the plateau score the sweep selected on, because that number
    is not a column here and sorting a table on something it does not show is
    how a reader concludes the order is arbitrary. Errored rows sort last and
    keep their place in the count.

    Returns `(text, hidden, legend)`; `hidden` is printed by the caller.
    """
    header = ["SYMBOL", "TF", "QUAD", "IS PF", "MAX DD", "BEST PARAMS"]
    body: list[list[str]] = []
    legend: dict[str, str] = {}

    ordered = sorted(rows, key=_stage2_sort_key)
    shown = ordered[: max(0, int(max_rows))]
    for row in shown:
        quad = row.get("quadrant")
        regime = row.get("optimal_regime")
        if quad and regime:
            legend[str(quad)] = str(regime)
        dd = _fmt_metric(row.get("max_drawdown_pct"))
        body.append([
            str(row.get("symbol") or "?"),
            str(row.get("timeframe") or "?"),
            # `--` where Stage 1 attached no scope, which happens for a pair
            # swept because it was named rather than because it survived. A
            # blank cell would read as a quadrant nobody wrote down.
            str(quad) if quad else "--",
            _fmt_metric(row.get("profit_factor")),
            dd if dd == "--" else f"{dd} %",
            _fmt_params(row.get("params")),
        ])

    if not body:
        # An empty string, not a bare header row. A header with nothing under
        # it reads as a table whose rows were lost; the caller replaces this
        # with a sentence saying the stage optimised nothing, which is a
        # result and needs to be legible as one.
        return "", len(ordered), legend

    widths = [max(len(header[i]), *(len(r[i]) for r in body))
              for i in range(len(header))]
    align = ["<", "<", "<", ">", ">", "<"]

    def line(cells: list[str]) -> str:
        return "  ".join(format(c, f"{align[i]}{widths[i]}")
                         for i, c in enumerate(cells)).rstrip()

    out = [line(header), line(["-" * w for w in widths])]
    out.extend(line(r) for r in body)
    return "\n".join(out), len(ordered) - len(shown), legend


def stage2_param_lines(rows: list[dict[str, Any]]) -> list[str]:
    """
    One line per configuration: `SYMBOL  TF  <parameter set, verbatim>`.

    The parameter set is the string Stage 2 wrote, with its real key names and
    nothing clipped - this block exists so the card carries a copy that can be
    retyped into `--param` without opening the handoff. The symbol and
    timeframe columns are padded so the sets line up under each other; a
    reader comparing two contracts is comparing the values, and ragged left
    edges are what makes that hard.

    Rows arrive in the table's order, so line 1 here is row 1 there.
    """
    if not rows:
        return []
    cells = [(str(r.get("symbol") or "?"), str(r.get("timeframe") or "?"),
              str(r.get("params") or "").strip() or "not recorded")
             for r in rows]
    sym_w = max(len(c[0]) for c in cells)
    tf_w = max(len(c[1]) for c in cells)
    return [f"{sym:<{sym_w}}  {tf:<{tf_w}}  {params}"
            for sym, tf, params in cells]


def format_stage2_param_fields(rows: list[dict[str, Any]],
                               budget: int = MAX_EMBED_TOTAL,
                               max_fields: int = STAGE2_PARAM_MAX_FIELDS,
                               name: str = STAGE2_PARAM_FIELD_NAME
                               ) -> tuple[list[dict[str, Any]], int]:
    """
    The full parameter sets as embed fields, packed to Discord's limits.

    The table above them abbreviates and clips, because a fixed-width column
    has to align; this is the unabbreviated record, and it covers EVERY
    configuration in the matrix - including the rows the table's own row cap
    left off, and including a failed sweep's `NOT OPTIMIZED`, which is a
    finding rather than a missing parameter set.

    `budget` is what is left of the 6000-character embed after the description
    and the other fields, and a line that does not fit is COUNTED in the
    returned `hidden` - the same rule as every other cap on these cards, and
    for the same reason: a silently shortened list of winning parameters reads
    as the whole stage. A line too wide for one field is WRAPPED, never
    clipped; the whole point of the block is that nothing in it is cut.

    Returns `(fields, hidden)`.
    """
    lines = stage2_param_lines(rows)
    if not lines:
        return [], 0

    fence_cost = len(FENCE_OPEN) + len(FENCE_CLOSE)
    # Room for the "N further ..." note, which is appended to the last field.
    room = int(budget) - NOTE_RESERVE
    width = min(MAX_FIELD_VALUE, room) - fence_cost - len(name) - len(" \u00b7 cont.")
    if width <= 0:
        return [], len(lines)

    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_param_line(line, width))

    fields: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(wrapped) and len(fields) < max_fields:
        label = name if not fields else f"{name} \u00b7 cont."
        # Recomputed per field: `room` shrinks as fields are appended, and a
        # capacity fixed at the first field's size is how an embed clears a
        # local check and is rejected with a 400 nobody reads.
        capacity = min(MAX_FIELD_VALUE, room - len(label)) - fence_cost
        if capacity <= 0:
            break
        chunk: list[str] = []
        used = 0
        while cursor < len(wrapped):
            addition = len(wrapped[cursor]) + (1 if chunk else 0)
            if used + addition > capacity:
                break
            chunk.append(wrapped[cursor])
            used += addition
            cursor += 1
        if not chunk:
            break
        field = {"name": label,
                 "value": FENCE_OPEN + "\n".join(chunk) + FENCE_CLOSE,
                 "inline": False}
        fields.append(field)
        room -= len(field["name"]) + len(field["value"])

    # `hidden` counts CONFIGURATIONS, not wrapped physical lines: a reader
    # chasing "3 further configurations" into the handoff is looking for three
    # contracts, and a count of line fragments would send them looking for a
    # number of rows that is not in the file. A configuration whose line was
    # cut mid-wrap counts as hidden, because a half-printed parameter set is
    # not a printed one.
    cut_mid_row = (cursor < len(wrapped)
                   and wrapped[cursor].startswith(" " * INDENT_WIDTH))
    hidden = len(lines) - _count_configurations(wrapped[:cursor], cut_mid_row)
    if hidden > 0:
        note = (f"\n_{hidden} further configuration(s) not shown \u2014 see "
                f"`best_params_<SYMBOL>_<TF>.json`._")
        if fields:
            fields[-1]["value"] += note
    return fields, max(0, hidden)


def _param_field(name: str, lines: list[str], index: int) -> dict[str, Any]:
    """One packed field. Continuations say so rather than repeating the name
    unqualified, which would read as a second, different list."""
    return {
        "name": name if index == 0 else f"{name} \u00b7 cont.",
        "value": FENCE_OPEN + "\n".join(lines) + FENCE_CLOSE,
        "inline": False,
    }


def _wrap_param_line(line: str, width: int) -> list[str]:
    """
    Wrap one configuration's line to `width`, continuations indented.

    Wrapped rather than clipped, and split on the parameter separator rather
    than mid-token: half of `sl_atr_mult=1.0` on one line and half on the next
    is a value a reader can misread as a whole one.
    """
    if len(line) <= width or width <= INDENT_WIDTH + 1:
        return [line]
    out: list[str] = []
    remaining = line
    indent = ""
    while len(remaining) > width:
        cut = remaining.rfind(" ", 0, width + 1)
        if cut <= len(indent):
            cut = width
        out.append(remaining[:cut].rstrip())
        indent = " " * INDENT_WIDTH
        remaining = indent + remaining[cut:].lstrip()
    out.append(remaining)
    return out


def _count_configurations(lines: list[str], cut_mid_row: bool = False) -> int:
    """
    How many WHOLE configurations a slice of wrapped lines covers.

    A continuation is indented, so it is not a row of its own. `cut_mid_row`
    says the slice ended with a configuration's continuation still to come, and
    that row is not counted: a half-printed parameter set is not a printed one,
    and counting it would leave a reader one contract short with nothing on the
    card saying so.
    """
    whole = sum(1 for line in lines if not line.startswith(" " * INDENT_WIDTH))
    return max(0, whole - 1) if cut_mid_row else whole


def build_stage2_embed(strat: str, blob: dict[str, Any],
                       source: str | Path | None = None,
                       max_rows: int = STAGE2_MAX_ROWS) -> dict[str, Any]:
    """
    Stage 2's card. Pure - sends nothing, computes nothing, and every value on
    it is transcribed from the summary Stage 2 wrote.

    The window is on the card because an optimised parameter set is only
    meaningful with the bars it was fitted to, and because it is the one field
    that says the holdout was not touched. The coverage line is there because
    Stage 2 prunes nothing: `12/12 optimised` is the claim, and anything less
    is a run failure rather than a screening result.
    """
    rows = stage2_rows(blob)
    ordered = sorted(rows, key=_stage2_sort_key)
    optimized = [r for r in rows
                 if str(r.get("status") or "").upper() == OPTIMIZED]
    table, hidden, legend = format_stage2_table(rows, max_rows)

    window = blob.get("in_sample_window") or {}
    start = window.get("start") or blob.get("start") or "not recorded"
    end = window.get("end") or blob.get("end") or "not recorded"
    holdout = window.get("holdout_starts")
    timeframes = blob.get("timeframes") or []
    coverage = blob.get("coverage") or {}
    rank = blob.get("rank") or "not recorded"

    description = [
        f"**In-sample window** `{start} → {end}`"
        + (f" · holdout from `{holdout}` untouched" if holdout else ""),
        # `rank` is what was APPLIED, which Stage 2 resolves - a rebuild of a
        # table with no plateau columns is ranked on Sharpe however the sweep
        # was invoked, and the card must not claim otherwise.
        f"**Selection** best parameters by `{rank}` rank, per configuration",
        "```text",
        table if table.strip() else "no configuration was optimised",
        "```",
    ]
    if legend:
        description.append("**Target regimes** " + " · ".join(
            f"`{q}` {legend[q]}" for q in sorted(legend)))
    if hidden:
        description.append(
            f"_{hidden} further configuration(s) are not shown — the full "
            f"matrix is in the handoff._")
    if any(str(r.get("params") or "").strip() for r in rows):
        # The table's cells are abbreviated and clipped, so the card has to say
        # where the copy that is neither lives. Without this line an `f=5` cell
        # reads as the parameter name the strategy declared.
        description.append(ABBREV_NOTE)

    text = "\n".join(description)
    if len(text) > MAX_EMBED_DESCRIPTION:
        # Trim the TABLE and never the header lines, for the same reason the
        # Stage 1 card does: without the window and the selection rule the
        # numbers underneath are unlabelled.
        keep = MAX_EMBED_DESCRIPTION - 64
        text = text[:keep] + "\n```\n_truncated — see the handoff._"

    errors = len(rows) - len(optimized)
    fields = [
        {"name": "Configurations", "value": str(len(rows)), "inline": True},
        # "Optimised" rather than "Promoted": Stage 2 promotes nothing and
        # drops nothing. Every row advances to Stage 3, and a heading borrowed
        # from the Stage 1 card would import a survival rate that does not
        # exist here.
        {"name": "Optimised → Stage 3", "value": str(len(optimized)),
         "inline": True},
        {"name": "Failed to sweep", "value": str(errors), "inline": True},
        {"name": "Timeframes",
         "value": ", ".join(f"`{t}`" for t in timeframes) or "not recorded",
         "inline": True},
        {"name": "Pruning",
         "value": ("none — every Stage 1 survivor advances"
                   if coverage.get("complete") is True
                   else f"none by design; {errors} configuration(s) failed to "
                        f"sweep and carry no parameters"),
         "inline": True},
    ]

    embed = {
        "title": f"\U0001F39B\uFE0F Stage 2 · Parameter Optimization: {strat}",
        "description": text,
        # Violet when something was optimised, amber when nothing was. Amber
        # rather than red for the same reason as Stage 1: a stage that produced
        # no rows is a result to look at, not a crash.
        "color": VIOLET if optimized else AMBER,
        "fields": fields,
        "footer": {"text": "backtest/discord_reporter.py · Stage 2 parameter "
                           "optimization · values as recorded by scan.py, not "
                           "recomputed"},
    }

    # The full parameter sets go in LAST, on whatever the rest of the card
    # left of Discord's 6000 characters. Sized against the FINISHED embed
    # rather than against a constant, because the description holding the
    # table is most of it: a fixed reservation would either starve this block
    # under a wide table or overflow the embed under a narrow one.
    handoff = {"name": "Handoff", "value": _fmt_report(str(source or "")),
               "inline": False}
    budget = (MAX_EMBED_TOTAL - _embed_size(embed)
              - len(handoff["name"]) - len(handoff["value"]))
    param_fields, _hidden = format_stage2_param_fields(ordered, budget=budget)
    if not param_fields:
        # The block did not fit at all. The note must not keep pointing at it:
        # a line saying the full parameters are below, with nothing below, is
        # worse than the clipped cells it was added to explain. The
        # replacement is SHORTER, so the budget just measured still holds.
        embed["description"] = embed["description"].replace(
            ABBREV_NOTE, ABBREV_NOTE_NO_BLOCK)
    embed["fields"] = fields + param_fields + [handoff]
    return embed


# --------------------------------------------------------------------------
# Stage 3 · gate audit & certification
# --------------------------------------------------------------------------

def default_audit_summary_path(strat: str, out_dir: str | None = None) -> Path:
    """`<BT_ARTIFACTS>/pipeline/<strategy>/stage3_audit_summary.json`."""
    return pipeline_dir(strat, out_dir) / STAGE3_SUMMARY_FILE


def load_stage3(path: str | Path, strat: str | None = None) -> dict[str, Any]:
    """
    Read Stage 3's handoff, and refuse the wrong one.

    Through `pipeline.read_stage` for the same reason the other two cards are:
    a file written by another stage or belonging to another strategy is a
    refusal there already. It matters most here - this card carries a
    PASS/FAIL and a hash seal, and a certification announced under the wrong
    strategy's name is the artifact nobody cross-checks.
    """
    return read_stage(Path(path), 3, strat)


def stage3_rows(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Every configuration the certification run covered, in one shape.

    Straight off `results`, which Stage 3 wrote with errors and skips already
    in it as `NOT AUDITED` rows. Nothing is filtered here: a card shorter than
    the stage's input reads as a complete certification, and "the audit raised"
    is a different statement from "the edge did not hold out of sample".
    """
    return list(blob.get("results") or [])


def _seal_prefix(row: dict[str, Any]) -> str:
    """
    The strategy-code hash, shortened for the table.

    The CODE hash rather than the audit hash, because it is what a reader
    compares against a promoted `meta.json`. `--` when nothing was staged: an
    uncertified configuration has no seal, and a blank cell in a hash column
    reads as a hash of nothing.
    """
    seal = row.get("seal") or {}
    digest = ((seal.get("strategy_code") or {}).get("sha256") or "")
    if not digest or digest == "NOT AVAILABLE":
        return "--"
    return digest[:SEAL_PREFIX_CHARS]


def _stage3_sort_key(row: dict[str, Any]) -> tuple:
    """
    Certified first, then the strongest out-of-sample quadrant, then by name.

    Sorted on the QUADRANT profit factor rather than the blended one, because
    that is what Gate R decided on: ranking on the blend would put a
    configuration with a broad mediocre result above one with a sharp edge in
    its own environment, which inverts the question the stage asks. A row with
    no measurable factor sorts last rather than being dropped.
    """
    certified = bool(row.get("certified"))
    pf = _fmt_float(row.get("oos_profit_factor"))
    return (not certified, -(pf if pf is not None else float("-inf")),
            str(row.get("symbol") or ""), str(row.get("timeframe") or ""),
            str(row.get("version") or ""))


def _fmt_float(value: Any) -> float | None:
    """A float, or None. NaN is None - it is not a number and must not sort."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def format_stage3_table(rows: list[dict[str, Any]],
                        max_rows: int = STAGE3_MAX_ROWS
                        ) -> tuple[str, int, dict[str, str]]:
    """
    The certification table as one fixed-width block, plus the quadrant legend.

    Returns `(text, hidden, legend)`. `hidden` is counted on the card by the
    caller.

    **IS PF and OOS PF sit next to each other on purpose.** "the holdout profit
    factor is 1.10" and "1.10, down from 2.40" are different findings, and a
    card printing only the second number would let a collapsing edge look like
    a healthy one. `REG PF` beside them is the quadrant factor Gate R actually
    scored - three profit factors on one row, each labelled, because two of
    them are blended-sample numbers that decided nothing.

    The QUAD column carries the `Q1`..`Q4` id the handoff recorded, with the
    legend built FROM the rows. No short spelling of a regime name lives in
    this module, for the same reason it does not on the Stage 1 card: a second
    one would be free to disagree with `mdlib.regimes`.
    """
    header = ["SYMBOL", "TF", "VER", "QUAD", "GATE R", "REG PF", "REG N",
              "IS PF", "OOS PF", "SEAL"]
    body: list[list[str]] = []
    legend: dict[str, str] = {}

    ordered = sorted(rows, key=_stage3_sort_key)
    shown = ordered[: max(0, int(max_rows))]
    for row in shown:
        quad = row.get("quadrant")
        regime = row.get("target_regime")
        if quad and regime:
            legend[str(quad)] = str(regime)
        body.append([
            str(row.get("symbol") or "?"),
            str(row.get("timeframe") or "?"),
            f"V{row['version']}" if row.get("version") else "--",
            str(quad) if quad else "--",
            # The GATE R status verbatim from the handoff. Never re-derived
            # from the numbers beside it: this module computes nothing, and a
            # card that recomputed a verdict would be free to disagree with the
            # audit it is announcing.
            str(row.get("gate_regime") or NOT_AUDITED).upper(),
            _fmt_metric(row.get("oos_profit_factor")),
            _fmt_count(row.get("oos_trade_count")),
            _fmt_metric(row.get("is_profit_factor")),
            _fmt_metric(row.get("holdout_profit_factor")),
            _seal_prefix(row),
        ])

    widths = [max(len(header[i]), *(len(r[i]) for r in body)) if body
              else len(header[i]) for i in range(len(header))]
    align = ["<", "<", "<", "<", "<", ">", ">", ">", ">", "<"]

    def line(cells: list[str]) -> str:
        return "  ".join(format(c, f"{align[i]}{widths[i]}")
                         for i, c in enumerate(cells)).rstrip()

    out = [line(header), line(["-" * w for w in widths])]
    out.extend(line(r) for r in body)
    return "\n".join(out), len(ordered) - len(shown), legend


def format_stage3_seals(rows: list[dict[str, Any]],
                        max_rows: int = STAGE3_MAX_ROWS) -> list[str]:
    """
    The full 64-character seals, one block per staged configuration.

    The table carries a 12-character prefix, which is enough to tell two builds
    apart and not enough to verify one. This is the copy a reader checks
    against a promoted `meta.json`, so all three hashes are here in full and
    labelled by what they cover - the code, the winning parameter file, and the
    gate audit. Only configurations that were actually STAGED appear: a seal
    for something nobody promoted is a checksum of a file the reader cannot
    find.
    """
    lines: list[str] = []
    for row in sorted(rows, key=_stage3_sort_key)[: max(0, int(max_rows))]:
        seal = row.get("seal") or {}
        if not seal:
            continue
        head = (f"{row.get('symbol') or '?'} {row.get('timeframe') or '?'} "
                f"V{row.get('version') or '?'}")
        for key, label in (("strategy_code", "code"),
                           ("winning_parameters", "params"),
                           ("gate_audit", "audit")):
            digest = ((seal.get(key) or {}).get("sha256") or "NOT AVAILABLE")
            lines.append(f"{head} {label:<7}{digest}")
    return lines


def build_stage3_embed(strat: str, blob: dict[str, Any],
                       source: str | Path | None = None,
                       max_rows: int = STAGE3_MAX_ROWS) -> dict[str, Any]:
    """
    Stage 3's card. Pure - sends nothing, computes nothing, and every value on
    it is transcribed from the summary Stage 3 wrote.

    Both windows are on the card because a certification is a claim about two
    date ranges and is meaningless with either one missing: the in-sample
    window says what the parameters were fitted to, and the holdout says what
    they were then measured on. `Verdict` states in words that Gates 1-3 are
    advisory, because a reader who has seen this pipeline before the charter
    would otherwise read a `CERTIFIED` beside a failing Gate 1 as a bug.
    """
    rows = stage3_rows(blob)
    certified = [r for r in rows if r.get("certified")]
    audited = [r for r in rows
               if str(r.get("status") or "").upper() != NOT_AUDITED]
    table, hidden, legend = format_stage3_table(rows, max_rows)

    is_window = blob.get("in_sample") or {}
    ho_window = blob.get("holdout") or {}
    rule = blob.get("certification_rule") or {}
    coverage = blob.get("coverage") or {}
    tf = blob.get("timeframe")

    description = [
        f"**In-sample** `{is_window.get('start') or 'not recorded'} → "
        f"{is_window.get('end') or 'not recorded'}` "
        f"· fitted at Stage 2, evidence only",
        f"**Out-of-sample holdout** `{ho_window.get('start') or 'not recorded'}"
        f" → {ho_window.get('end') or 'present'}` · the verdict",
        f"**Verdict** Gate R — profit factor `>= "
        f"{_fmt_metric(rule.get('min_profit_factor'))}` over "
        f"`{_fmt_count(rule.get('min_trades'))}`+ trades INSIDE the target "
        f"quadrant. Gates 1–3 are reported as evidence and cannot fail a "
        f"certification.",
        "```text",
        table if table.strip() else "no configuration was audited",
        "```",
    ]
    if legend:
        description.append("**Target regimes** " + " · ".join(
            f"`{q}` {legend[q]}" for q in sorted(legend)))
    if hidden:
        description.append(
            f"_{hidden} further configuration(s) are not shown — the full "
            f"summary is in the handoff._")
    description.append(
        f"_`SEAL` is the first {SEAL_PREFIX_CHARS} characters of the strategy "
        f"code's SHA-256; the full seals are below._")

    text = "\n".join(description)
    if len(text) > MAX_EMBED_DESCRIPTION:
        # Trim the TABLE and never the header lines: without the two windows
        # and the verdict rule the numbers underneath are unlabelled.
        keep = MAX_EMBED_DESCRIPTION - 64
        text = text[:keep] + "\n```\n_truncated — see the handoff._"

    fields = [
        {"name": "Strategy", "value": f"`{strat}`", "inline": True},
        {"name": "Timeframe", "value": f"`{tf}`" if tf else "not recorded",
         "inline": True},
        {"name": "Configurations", "value": str(len(rows)), "inline": True},
        # "Certified" is Gate R and nothing else. Counted from the handoff's
        # own `certified` flag rather than re-derived from the numbers on the
        # card, so this can never announce a pass Stage 3 did not record.
        {"name": "Certified → Incubator", "value": str(len(certified)),
         "inline": True},
        {"name": "Audited", "value": f"{len(audited)}/{len(rows)}",
         "inline": True},
        {"name": "Pruning",
         "value": ("none on an aggregate metric; no prop-firm rule applied"
                   if coverage.get("complete") is not False
                   else f"none by design; "
                        f"{coverage.get('errors', 0)} error(s), "
                        f"{coverage.get('skipped', 0)} skipped"),
         "inline": False},
    ]

    embed = {
        "title": f"\U0001F510 Stage 3 · Gate Audit & Certification: {strat}",
        "description": text,
        # Teal when something was certified, amber when nothing was. Amber
        # rather than red for the same reason as the other cards: a holdout
        # that certified nothing is a result to read, not a crash.
        "color": TEAL if certified else AMBER,
        "fields": fields,
        "footer": {"text": "backtest/discord_reporter.py · Stage 3 gate audit "
                           "· values as recorded by audit_gates.py, not "
                           "recomputed"},
    }

    # The full seals go in LAST, on whatever the rest of the card left of
    # Discord's 6000 characters - sized against the FINISHED embed rather than
    # a constant, exactly as the Stage 2 parameter block is, because the
    # description holding the table is most of the budget.
    handoff = {"name": "Handoff", "value": _fmt_report(str(source or "")),
               "inline": False}
    budget = (MAX_EMBED_TOTAL - _embed_size(embed)
              - len(handoff["name"]) - len(handoff["value"]))
    seal_fields, dropped = _seal_fields(
        format_stage3_seals(rows, max_rows), budget)
    if not seal_fields:
        # The block did not fit. The note pointing at it must go with it: a
        # line saying the full seals are below, with nothing below, is worse
        # than the prefix it was added to explain.
        embed["description"] = embed["description"].replace(
            f"the full seals are below._",
            f"the full seals are in the handoff._")
    elif dropped:
        seal_fields[-1]["value"] = seal_fields[-1]["value"].rstrip("`\n") + (
            f"\n… {dropped} further seal line(s) — see the handoff." + FENCE_CLOSE)
    embed["fields"] = fields + seal_fields + [handoff]
    return embed


def _seal_fields(lines: list[str], budget: int) -> tuple[list[dict], int]:
    """
    The seal block as Discord fields, and how many lines did not fit.

    One field per chunk under `MAX_FIELD_VALUE`, at most `STAGE3_SEAL_MAX_FIELDS` of
    them, and never past `budget`. A hash line is 80-odd characters and does
    not wrap usefully, so a line that does not fit is COUNTED rather than
    truncated: half a SHA-256 is not a shorter checksum, it is a different
    string that looks like one.
    """
    if not lines or budget <= len(FENCE_OPEN) + len(FENCE_CLOSE) + NOTE_RESERVE:
        return [], len(lines)

    fields: list[dict] = []
    spent = 0
    i = 0
    while i < len(lines) and len(fields) < STAGE3_SEAL_MAX_FIELDS:
        chunk: list[str] = []
        size = len(FENCE_OPEN) + len(FENCE_CLOSE)
        while i < len(lines):
            need = len(lines[i]) + (1 if chunk else 0)
            if size + need > MAX_FIELD_VALUE:
                break
            name = "Seals (SHA-256)" if not fields else "Seals (cont.)"
            if spent + size + need + len(name) + NOTE_RESERVE > budget:
                break
            chunk.append(lines[i])
            size += need
            i += 1
        if not chunk:
            break
        name = "Seals (SHA-256)" if not fields else "Seals (cont.)"
        fields.append({"name": name,
                       "value": FENCE_OPEN + "\n".join(chunk) + FENCE_CLOSE,
                       "inline": False})
        spent += size + len(name)
    return fields, len(lines) - i


def build_payload(embed: dict[str, Any]) -> dict[str, Any]:
    return {"embeds": [embed]}


def _embed_size(embed: dict[str, Any]) -> int:
    """
    Total characters Discord counts against the 6000 embed limit.

    The DESCRIPTION counts too, and on the Stage 1 card it is most of the
    embed. Omitting it here would let a 40-row leaderboard sail past a check
    that reported 300 characters and be rejected with a 400 nobody reads.
    """
    total = (len(embed.get("title", ""))
             + len(embed.get("description", ""))
             + len(embed.get("footer", {}).get("text", "")))
    for field in embed.get("fields", []):
        total += len(field["name"]) + len(field["value"])
    return total


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

def post_embed(webhook: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    POST the payload. Never raises: the outcome is RETURNED, because a notifier
    that throws loses the record of what it attempted.

    Returns {ok, http_status, error}. `http_status` is None when the request
    never reached a server.
    """
    try:
        response = requests.post(
            webhook,
            json=payload,
            timeout=POST_TIMEOUT_SECONDS,
            headers={"Content-Type": "application/json"},
        )
    except requests.exceptions.RequestException as exc:
        # Never echo the URL: it is a credential, and requests' own exception
        # text embeds the FULL url including the token - so the exception
        # message is deliberately not forwarded. The type and the host are
        # enough to tell a typo'd domain from a dead network from a timeout.
        host = urlparse(webhook).netloc or "<unparseable url>"
        return {
            "ok": False,
            "http_status": None,
            "error": f"{type(exc).__name__} contacting {host}",
        }

    if response.status_code in SUCCESS_STATUS:
        return {"ok": True, "http_status": response.status_code, "error": None}

    body = (response.text or "").strip()
    if len(body) > 500:
        body = body[:500] + "..."
    return {
        "ok": False,
        "http_status": response.status_code,
        "error": body or response.reason or "no response body",
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discord_reporter.py",
        description="Post a pipeline card to a Discord webhook: a promotion "
                    "scorecard (--mode promotion), Stage 1's regime-firewall "
                    "leaderboard (--stage 1), Stage 2's parameter "
                    "optimization summary (--stage 2), or Stage 3's gate "
                    "audit and certification (--stage 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Values are printed as supplied - nothing here recomputes a metric.\n"
            f"--webhook may be omitted when ${ENV_WEBHOOK} is set.\n"
            "\n"
            "  --mode promotion   --strat X --symbol NQ --tf 15m --pf 1.42 ...\n"
            "  --stage 1          --strat X [--survivors <surviving_assets.json>]\n"
            "  --stage 2          --strat X [--summary <stage2_summary.json>]\n"
            "  --stage 3          --strat X [--audit <stage3_audit_summary.json>]"
        ),
    )
    parser.add_argument("--webhook", default=os.environ.get(ENV_WEBHOOK),
                        help=f"Discord webhook URL (default: ${ENV_WEBHOOK})")
    # `--mode` and `--stage` are two spellings of one choice, and they share a
    # dest so they cannot disagree. A card labelled Stage 1 that was built by
    # the promotion path would announce a screen as a promotion.
    parser.add_argument("--mode", dest="mode", default=None,
                        choices=["promotion", "baseline", "scan", "audit"],
                        help="promotion (default): the Stage 5 scorecard. "
                             "baseline: Stage 1's regime-firewall leaderboard. "
                             "scan: Stage 2's parameter optimization summary. "
                             "audit: Stage 3's gate audit and certification.")
    parser.add_argument("--stage", dest="stage", default=None,
                        choices=["1", "2", "3", "5"],
                        help="1 == --mode baseline, 2 == --mode scan, "
                             "3 == --mode audit, 5 == --mode promotion")
    parser.add_argument("--strat", required=True, help="strategy name, e.g. sma_momentum_crossover")
    parser.add_argument("--symbol", default="", help="promotion mode: the contract the decision rests on, e.g. NQ")
    parser.add_argument("--tf", default="", help="promotion mode: timeframe, e.g. 15m")
    parser.add_argument("--survivors", default=None,
                        help="baseline mode: path to surviving_assets.json "
                             "(default: <BT_ARTIFACTS>/pipeline/<strat>/"
                             f"{SURVIVORS_FILE})")
    parser.add_argument("--summary", default=None,
                        help="scan mode: path to stage2_summary.json "
                             "(default: <BT_ARTIFACTS>/pipeline/<strat>/"
                             f"{STAGE2_SUMMARY_FILE})")
    parser.add_argument("--audit", default=None,
                        help="audit mode: path to stage3_audit_summary.json "
                             "(default: <BT_ARTIFACTS>/pipeline/<strat>/"
                             f"{STAGE3_SUMMARY_FILE})")
    parser.add_argument("--out-dir", default=None,
                        help="baseline, scan and audit modes: override the "
                             "pipeline directory the handoff is looked up in")
    parser.add_argument("--max-rows", type=int, default=None,
                        help=f"leaderboard rows on the card. Each mode keeps "
                             f"its OWN default, because the rows are different "
                             f"widths: baseline {STAGE1_MAX_ROWS}, scan "
                             f"{STAGE2_MAX_ROWS} (each row carries a parameter "
                             f"set), audit {STAGE3_MAX_ROWS}. Whatever does "
                             f"not fit is COUNTED on the card, never dropped "
                             f"in silence.")
    parser.add_argument("--pf", default="", help="out-of-sample profit factor, or a token like 'NOT EVALUATED'")
    parser.add_argument("--dd", default="", help="max drawdown in percent, or a token like 'NOT EVALUATED'")
    parser.add_argument("--regime", default="", help="certified regime, e.g. 'High-Vol/Trending'")
    parser.add_argument("--report", default="", help="artifact URL or path to the tear sheet")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the payload and send nothing")
    return parser


def resolve_mode(mode: str | None, stage: str | None) -> str:
    """
    One mode from the two flags, or a refusal.

    `--mode baseline --stage 5` is not a typo worth guessing at: one of the two
    is what the operator meant and picking either silently posts the wrong
    card. Neither flag given is `promotion`, which is what this script was
    before the Stage 1 mode existed.
    """
    from_stage = {"1": "baseline", "2": "scan", "3": "audit",
                  "5": "promotion"}.get(stage or "")
    if mode and from_stage and mode != from_stage:
        raise ValueError(f"--mode {mode} and --stage {stage} disagree "
                         f"(--stage {stage} means --mode {from_stage}).")
    return mode or from_stage or "promotion"


def _build_card(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """The embed and the one-line summary its success message prints."""
    mode = resolve_mode(args.mode, args.stage)

    if mode == "baseline":
        path = Path(args.survivors) if args.survivors else \
            default_survivors_path(args.strat, args.out_dir)
        blob = load_stage1(path, args.strat)
        # `--max-rows` defaults to None so each card keeps its OWN row cap: a
        # Stage 2 row carries a parameter set and is roughly twice as wide, and
        # one shared default would either waste the Stage 1 card's description
        # or overflow the Stage 2 one.
        embed = build_stage1_embed(args.strat, blob, source=path,
                                   max_rows=(args.max_rows
                                             if args.max_rows is not None
                                             else STAGE1_MAX_ROWS))
        rows = stage1_rows(blob)
        kept = sum(1 for r in rows
                   if str(r.get("status") or "").upper() == PROMOTED)
        return embed, (f"Stage 1 screen '{args.strat}' "
                       f"({kept}/{len(rows)} promoted)")

    if mode == "scan":
        path = Path(args.summary) if args.summary else \
            default_scan_summary_path(args.strat, args.out_dir)
        blob = load_stage2(path, args.strat)
        embed = build_stage2_embed(args.strat, blob, source=path,
                                   max_rows=(args.max_rows
                                             if args.max_rows is not None
                                             else STAGE2_MAX_ROWS))
        rows = stage2_rows(blob)
        done = sum(1 for r in rows
                   if str(r.get("status") or "").upper() == OPTIMIZED)
        return embed, (f"Stage 2 optimization '{args.strat}' "
                       f"({done}/{len(rows)} optimised)")

    if mode == "audit":
        path = Path(args.audit) if args.audit else \
            default_audit_summary_path(args.strat, args.out_dir)
        blob = load_stage3(path, args.strat)
        embed = build_stage3_embed(args.strat, blob, source=path,
                                   max_rows=(args.max_rows
                                             if args.max_rows is not None
                                             else STAGE3_MAX_ROWS))
        rows = stage3_rows(blob)
        # `certified` is Stage 3's own flag, not a count re-derived from the
        # numbers on the card. A reporter that recomputed a verdict would be
        # free to announce a pass the audit did not record.
        passed = sum(1 for r in rows if r.get("certified"))
        return embed, (f"Stage 3 certification '{args.strat}' "
                       f"({passed}/{len(rows)} certified)")

    # The promotion card names one contract, so those two are required here
    # and only here. Checked rather than defaulted: a card headed `?` · `?` is
    # a promotion announcement for a strategy on no instrument.
    missing = [f for f, v in (("--symbol", args.symbol), ("--tf", args.tf))
               if not (v or "").strip()]
    if missing:
        raise ValueError(f"--mode promotion needs {' and '.join(missing)}.")

    embed = build_embed(
        strat=args.strat,
        symbol=args.symbol,
        tf=args.tf,
        pf=args.pf,
        dd=args.dd,
        regime=args.regime,
        report=args.report,
    )
    return embed, f"'{args.strat}' ({args.symbol} {args.tf})"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        embed, summary = _build_card(args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        # Nothing is posted. A card built from a handoff that could not be read
        # would have to invent the numbers on it, which is the one thing this
        # module refuses to do.
        print(f"FAILED  {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    size = _embed_size(embed)
    if size > MAX_EMBED_TOTAL:
        print(f"FAILED  embed is {size} characters, over Discord's {MAX_EMBED_TOTAL} limit; nothing sent.",
              file=sys.stderr)
        return 1

    payload = build_payload(embed)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        print("DRY RUN  nothing was sent.")
        return 0

    if not args.webhook or not args.webhook.strip():
        print(f"FAILED  no webhook: pass --webhook or set ${ENV_WEBHOOK}.", file=sys.stderr)
        return 1

    result = post_embed(args.webhook.strip(), payload)

    if result["ok"]:
        print(f"SUCCESS  posted {summary} to Discord "
              f"[HTTP {result['http_status']}]")
        return 0

    status = result["http_status"]
    where = f"HTTP {status}" if status is not None else "no response"
    print(f"FAILED  Discord rejected the post [{where}]: {result['error']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
