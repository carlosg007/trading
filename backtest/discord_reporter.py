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
  quadrant, Gate R's verdict, and the quadrant profit factor and trade count
  it was measured on. **The table is built to a WIDTH** (45 characters,
  `STAGE3_TABLE_WIDTH`): Discord wraps a code block that overruns the
  viewport, and a wrapped fixed-width table is worse than none - every row
  becomes two, the second one unlabelled, and the columns a reader is
  comparing stop lining up under each other. The ten-column row this replaced
  ran to 68 characters and wrapped on every phone.

  So `PF` and `N` on the row are Gate R's OWN quadrant numbers and nothing
  else, and the description says which they are, because an unlabelled profit
  factor under a regime-gated verdict is the one value here a reader must not
  have to guess at. The blended IS/OOS pair moved to the certified rows in the
  promotion block, where the collapse it exposes ("1.10, down from 2.40")
  changes a decision somebody is about to make; on the other rows they were
  two more numbers that decided nothing. A `FAIL` carries WHY - `FAIL·N` for a
  quadrant that starved, `FAIL·PF` for an edge that died - because those read
  identically as `FAIL` and are fixed by completely different work. The
  PASS/FAIL token itself is still transcribed; only the reason is derived, and
  only from the thresholds the handoff recorded (see `gate_r_reason`). The 999
  profit-factor sentinel renders as `--`: a quadrant with one winning holdout
  trade has no measured factor, and 999.00 beside a FAIL reads as the
  strongest configuration on the card.

  Its table spans EVERY timeframe the summary indexes, because Stage 3 now
  merges its per-timeframe invocations into one file; a card headed `15m`
  above a table carrying 5m rows described neither. Below the table it carries
  the one section on this card somebody ACTS on: a bullet per certified
  configuration - target quadrant, the factor Gate R scored, the winning
  parameter plateau, the blended pair behind it, and 12-character prefixes of
  its code and parameter seals - and then ONE command, in a field of its own,
  that promotes all of them. `run_pipeline.py --promote-only`, never
  `--auto-promote`: the second re-runs Stages 1-4 first and OVERWRITES the
  handoff this card was built from, so the winners it then promotes are a
  fresh sweep's rather than the ones the reader is looking at. The per-pair
  `promote.py` command survives for the one case that needs it - a promotion
  that FAILED, where an operator finishes a single pair and must cite that
  pair's own `gate_audit_<SYMBOL>_<TF>.json` rather than the unsuffixed file,
  which holds whichever timeframe ran last. The full 64-character seals are
  not on the card: the 30-line dump they used to close it was unreadable on a
  phone and verified by nobody from one, and a reader checking a seal has the
  promoted `meta.json` open.

  Headed READY FOR PROMOTION / STAGED until a promotion has actually happened,
  and AUTOMATICALLY PROMOTED TO INCUBATOR with the commit once
  `run_pipeline.py --auto-promote` has written its outcome back onto the
  handoff. The heading is that RECORD's and is never inferred from a seal: a
  sealed configuration was staged by Stage 3 and committed by nobody, and
  announcing it as promoted is how a strategy nobody promoted comes to be
  believed to be in the incubator.

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

# A Stage 3 row carries a quadrant, Gate R's verdict and the factor and sample
# it was measured on. Whatever does not fit is COUNTED on the card, as
# everywhere else here.
STAGE3_MAX_ROWS = 24

# The width the Stage 3 table is built to. Discord wraps a code block that
# overruns the viewport, and a wrapped fixed-width table is worse than no
# table: every row becomes two, the second one unlabelled, and the columns a
# reader is comparing stop lining up. 45 characters clears the narrowest phone
# viewport this card is read on. It is a design TARGET rather than a hard cut -
# the columns still size to their widest cell, because clipping a symbol or a
# verdict to hit a number is how a table starts lying - so the way to keep it
# is to keep the tokens in it short.
STAGE3_TABLE_WIDTH = 45

# The profiler writes 999 as a profit factor when a quadrant never had a losing
# trade (backtest/profiler.py). It is a SENTINEL, not a measured factor, and it
# is exactly the number a STARVED quadrant prints: one winning holdout trade
# shows 999.00 next to a Gate R that failed on the sample count, which reads as
# the strongest configuration on the card. It renders as `--`.
REGIME_PF_SENTINEL = 999.0

# The long verdict tokens, shortened for the table and nowhere else. Both keep
# their NOT: a gate that was not evaluated and a run that was not audited are
# statements about what has not happened yet, and "EVAL"/"AUDIT" alone would
# read as the opposite.
GATE_R_TOKENS = {"NOT EVALUATED": "NO EVAL", "NOT AUDITED": "NO AUDIT"}

# How much of each SHA-256 goes on the card. Twelve hex characters is 48 bits -
# enough to tell two builds of the same strategy apart at a glance, which is
# what a reader uses it for. The full 64 live in the promoted `meta.json` and
# on the handoff, and the card says so: a truncated hash presented as THE hash
# is a checksum nobody can verify. The 30-line dump of full digests that used
# to close this card was unreadable on a phone and verified by nobody from
# one - a reader checking a seal has the file open.
SEAL_PREFIX_CHARS = 12

# The promotion block. It is the one part of this card somebody ACTS on, so it
# is budgeted BEFORE the seals: a reader who cannot find the promote command
# goes looking through the handoff, while a reader missing a seal has lost a
# checksum they were not going to verify from a chat client anyway.
STAGE3_PROMO_MAX_ROWS = 10
STAGE3_PROMO_MAX_FIELDS = 4

# The two headings the section takes, and they are not interchangeable. The
# first says a human still has to run something; the second says a commit
# already happened. Labelling a staged-but-uncommitted configuration as
# promoted is how a strategy nobody promoted ends up believed to be in the
# incubator.
PROMO_READY_TITLE = "\U0001F3C6 READY FOR PROMOTION / STAGED"
PROMO_DONE_TITLE = "\U0001F680 AUTOMATICALLY PROMOTED TO INCUBATOR"

# The one command that promotes the certified set, in its own field. Its own
# field and not the tail of the promotion block, because the chunker splits a
# long block across fields and half a command is a command that runs and does
# something else. A field is never split.
PROMO_COMMAND_FIELD = "\u25B6 Promote all certified · one command"

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


def _seal_prefix(row: dict[str, Any], key: str = "strategy_code") -> str:
    """
    One of a configuration's seals, shortened.

    `--` when nothing was staged: an uncertified configuration has no seal,
    and a blank cell in a hash column reads as a hash of nothing. The CODE
    hash and the WINNING PARAMETERS hash are both used - the same module under
    a different grid cell is a different strategy with the same code checksum,
    so a code prefix alone identifies a build only if the parameters travelled
    with it.
    """
    seal = row.get("seal") or {}
    digest = ((seal.get(key) or {}).get("sha256") or "")
    if not digest or digest == "NOT AVAILABLE":
        return "--"
    return digest[:SEAL_PREFIX_CHARS]


def _regime_pf_cell(row: dict[str, Any]) -> str:
    """
    Gate R's quadrant profit factor as a table cell, with the sentinel
    rendered as `--`.

    `999.00` is what the profiler writes when a quadrant never had a losing
    trade. It is not a measured factor, and it is exactly the number a starved
    quadrant prints: one winning holdout trade shows 999.00 beside a Gate R
    that failed on the sample count, which reads as the strongest row on the
    card. `--` is what every other column here uses for "nothing was
    measured", and that is the true statement about a factor with no losing
    trade under it.
    """
    pf = _fmt_float(row.get("oos_profit_factor"))
    if pf is None or pf >= REGIME_PF_SENTINEL:
        return "--"
    return f"{pf:.2f}"


def gate_r_reason(row: dict[str, Any], rule: dict[str, Any] | None) -> str:
    """
    WHY a Gate R failed: `N` (too few trades in the quadrant), `PF` (the
    factor missed), or `""` when it did not fail or cannot be told.

    A FAIL on the count and a FAIL on the factor read identically as `FAIL`
    and are fixed by completely different work - a quadrant the strategy never
    entered again is a designation problem, a factor below 1.00 is a dead
    edge. Stage 3 records the first as `regime_starvation` and that record is
    preferred whenever it is there.

    **The PASS/FAIL token itself is never re-derived** - it is transcribed
    from `gate_regime`, as everything else on this card is. Only the
    parenthetical reason is worked out here, and only from the thresholds the
    handoff itself recorded in `certification_rule`, so the card cannot hold a
    configuration to a bar the audit did not use. With either number missing
    it returns `""` and the cell stays a bare `FAIL`, which is the honest
    answer when the reason is not on the file.
    """
    if str(row.get("gate_regime") or "").upper() != "FAIL":
        return ""
    if row.get("regime_starvation"):
        return "N"
    rule = rule or {}
    n, floor = _fmt_float(row.get("oos_trade_count")), \
        _fmt_float(rule.get("min_trades"))
    if n is not None and floor is not None and n < floor:
        return "N"
    pf, bar = _fmt_float(row.get("oos_profit_factor")), \
        _fmt_float(rule.get("min_profit_factor"))
    if pf is not None and bar is not None and pf < bar:
        return "PF"
    return ""


def _gate_r_cell(row: dict[str, Any], rule: dict[str, Any] | None) -> str:
    """`PASS`, `FAIL`, `FAIL·N`, `FAIL·PF`, `NO EVAL` or `NO AUDIT`."""
    status = str(row.get("gate_regime") or NOT_AUDITED).upper()
    cell = GATE_R_TOKENS.get(status, status)
    reason = gate_r_reason(row, rule)
    return f"{cell}·{reason}" if reason else cell


def _status_cell(row: dict[str, Any], rule: dict[str, Any] | None) -> str:
    """
    The outcome column: `CERTIFIED`, `STARVED`, `REJECTED`, `NOT CERT` or
    `NO AUDIT`.

    `CERTIFIED` is the handoff's own `certified` flag and never a re-reading
    of the numbers beside it. The other four are what that flag being false
    MEANS, which is not one thing: a run that broke, a quadrant nobody
    designated, a quadrant that starved and an edge that died are four
    findings fixed by four different pieces of work, and one `NOT CERTIFIED`
    token hides all of them.
    """
    if row.get("certified"):
        return "CERTIFIED"
    if str(row.get("status") or "").upper() == NOT_AUDITED:
        return "NO AUDIT"
    return {"N": "STARVED", "PF": "REJECTED"}.get(gate_r_reason(row, rule),
                                                  "NOT CERT")


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
                        max_rows: int = STAGE3_MAX_ROWS,
                        rule: dict[str, Any] | None = None
                        ) -> tuple[str, int, dict[str, str]]:
    """
    The certification table as one fixed-width block, plus the quadrant legend.

    Returns `(text, hidden, legend)`. `hidden` is counted on the card by the
    caller.

    **It is built to a WIDTH, not to a column list** (`STAGE3_TABLE_WIDTH`,
    45 characters). Discord wraps a code block that overruns the viewport, and
    a wrapped fixed-width table is worse than no table: every row becomes two,
    the second one unlabelled, and the columns a reader is comparing no longer
    line up under each other. The old ten-column row ran to 68 characters and
    wrapped on every phone. So the row carries the six things a certification
    IS - which contract, at which timeframe, in which quadrant, what Gate R
    said, and the factor and sample it said it on - and nothing else.

    What was dropped, and where it went. `VER` is on the promotion bullets,
    which is the only place a reader acts on it. The blended IS and OOS
    factors are on the promotion bullets too, for the certified rows: the
    collapse they exist to expose ("1.10, down from 2.40") only changes a
    decision for a configuration somebody is about to promote, and for the
    rest they are two more numbers that decided nothing. The `SEAL` prefix is
    likewise on the bullets, beside the parameters it seals.

    `PF` and `N` are Gate R's OWN quadrant numbers - the factor and the trade
    count inside the designated quadrant, on the holdout - never the blended
    sample. The header is short and the description says which they are,
    because an unlabelled profit factor under a regime-gated verdict is the
    one number on this card that must not be guessed at.

    The QUAD column carries the `Q1`..`Q4` id the handoff recorded, with the
    legend built FROM the rows. No short spelling of a regime name lives in
    this module, for the same reason it does not on the Stage 1 card: a second
    one would be free to disagree with `mdlib.regimes`.
    """
    header = ["SYM", "TF", "QD", "GATE R", "PF", "N", "STATUS"]
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
            str(quad) if quad else "--",
            # The GATE R status verbatim from the handoff. Never re-derived
            # from the numbers beside it: this module computes nothing, and a
            # card that recomputed a verdict would be free to disagree with the
            # audit it is announcing. Only the ·N / ·PF suffix is worked out
            # here - see gate_r_reason.
            _gate_r_cell(row, rule),
            _regime_pf_cell(row),
            _fmt_count(row.get("oos_trade_count")),
            _status_cell(row, rule),
        ])

    widths = [max(len(header[i]), *(len(r[i]) for r in body)) if body
              else len(header[i]) for i in range(len(header))]
    align = ["<", ">", "<", "<", ">", ">", "<"]

    def line(cells: list[str]) -> str:
        return "  ".join(format(c, f"{align[i]}{widths[i]}")
                         for i, c in enumerate(cells)).rstrip()

    out = [line(header), line(["-" * w for w in widths])]
    out.extend(line(r) for r in body)
    return "\n".join(out), len(ordered) - len(shown), legend


def stage3_timeframes(blob: dict[str, Any]) -> list[str]:
    """
    Every timeframe this summary indexes, in order.

    Read off the handoff's own `timeframes` when it has one and off the rows
    otherwise, so a summary written before Stage 3 merged across timeframes
    still names what it holds. The single `timeframe` field is the LAST run's
    and is deliberately not used here: a card headed `15m` above a table
    carrying 5m rows describes neither.
    """
    declared = [str(t) for t in (blob.get("timeframes") or []) if t]
    if declared:
        return declared
    seen: list[str] = []
    for row in stage3_rows(blob):
        tf = row.get("timeframe")
        if tf and str(tf) not in seen:
            seen.append(str(tf))
    return seen


def stage3_param_text(value: Any) -> str:
    """
    A winning parameter set as `k=v, k=v`, from a dict or from a string.

    Stage 3 records `params` as a DICT (Stage 2 records a string), and both
    shapes reach this module. Values are stringified and never reformatted -
    `None` stays `None`, which for `tp_atr_mult` means no take-profit was
    modelled at all, and a reporter that turned it into a number would print a
    target that never existed.
    """
    if isinstance(value, dict):
        if not value:
            return "not recorded"
        return ", ".join(f"{k}={value[k]}" for k in value)
    text = str(value or "").strip()
    return text or "not recorded"


def stage3_auto_promotion(blob: dict[str, Any]) -> dict[str, Any]:
    """
    The auto-promotion record `run_pipeline.py --auto-promote` writes back onto
    Stage 3's summary, or `{}` when nothing promoted.

    Written by the orchestrator rather than by Stage 3, because Stage 3 stages
    into the incubator and never commits - the commit is Stage 5's, and the
    hash only exists once it has run. The card reads it and changes its
    HEADING; it never infers a promotion from a seal, because a sealed
    configuration is one Stage 3 staged and nobody committed.
    """
    blk = blob.get("auto_promotion")
    return blk if isinstance(blk, dict) else {}


def promotion_command(strat: str, row: dict[str, Any],
                      source: str | None) -> str:
    """
    The exact Stage 5 command for ONE configuration, on one line.

    This is the RECOVERY path and not the card's normal instruction: it is
    printed only for a configuration whose automatic promotion failed, where
    an operator needs the one pair rather than the whole certified set. It is
    spelled out in full - strategy, version, source module and that pair's own
    `gate_audit_<SYMBOL>_<TF>.json` - because the alternative is reconstructing
    it against a directory holding one audit per timeframe, and picking the
    wrong file promotes a verdict about different bars. `--source` comes from
    the handoff; when Stage 3 did not record one the placeholder is left
    visible rather than the flag dropped, so the command fails loudly instead
    of promoting the module's defaults.
    """
    version = str(row.get("version") or "A")
    audit = str(row.get("audit_file") or "NOT RECORDED")
    src = str(source or "<path to the strategy module>")
    return (f"python3 backtest/promote.py --strat {strat} "
            f"--version {version} --source {src} --audit-file {audit}")


def unified_promote_command(strat: str) -> list[str]:
    """
    The one command that promotes everything Stage 3 certified, as the two
    lines it is printed on.

    `--promote-only`, never `--auto-promote`: the second re-runs Stages 1-4
    first and OVERWRITES the handoff this card was built from, so the winners
    it then promotes are a fresh sweep's and not the ones the reader is
    looking at. One flag is the difference between promoting what was
    certified and silently re-certifying from scratch.

    Split on a backslash continuation rather than run out to 90 characters:
    a code block that overruns the viewport wraps, and a wrapped command is
    one a reader copies half of.
    """
    return ["python3 backtest/run_pipeline.py \\",
            f"    --strat {strat} --promote-only"]


def format_stage3_promotions(strat: str, blob: dict[str, Any],
                             max_rows: int = STAGE3_PROMO_MAX_ROWS
                             ) -> tuple[str, list[str], list[str], int]:
    """
    The promotion block: `(title, lines, pairs, hidden)`.

    `pairs` is the one-line answer the description carries (`` `CL 15m` ``);
    `lines` are MARKDOWN, not a code fence - a bullet per certified
    configuration with its quadrant, the factor Gate R scored, the winning
    parameter plateau, the blended pair behind it and its seals, and ONE
    command underneath that promotes all of them.

    **One command, not one per pair.** Three certified configurations used to
    print three three-line bash blocks carrying four absolute filesystem
    paths each, which is 12 lines of wrapped path on a phone and the reason
    this section was unreadable. `run_pipeline.py --promote-only` iterates
    every certified row against that row's OWN
    `gate_audit_<SYMBOL>_<TF>.json`, which is exactly what the per-pair
    commands spelled out by hand. The per-pair command survives for the one
    case that needs it: a promotion that FAILED, where an operator is
    finishing a single pair rather than running the set.

    Only rows Stage 3 recorded as `certified` appear. Never a row whose
    numbers look like a pass: this module transcribes, and a section headed
    READY FOR PROMOTION is exactly where a re-derived verdict would do the
    most damage.

    The heading is the auto-promotion record's, not a guess. A configuration
    Stage 3 SEALED has been staged into the incubator and not committed, and
    presenting that as promoted is the failure this whole card is written to
    avoid.
    """
    rows = [r for r in stage3_rows(blob) if r.get("certified")]
    rows.sort(key=_stage3_sort_key)
    shown = rows[: max(0, int(max_rows))]

    auto = stage3_auto_promotion(blob)
    ran = bool(auto.get("ran"))
    commit = str(auto.get("commit") or "").strip()
    title = (f"{PROMO_DONE_TITLE} (Commit {commit})" if ran and commit
             else PROMO_DONE_TITLE if ran else PROMO_READY_TITLE)
    # Keyed by the same (symbol, timeframe, version) the orchestrator promoted
    # under, so a partially-failed auto-promotion labels each row by what
    # actually happened to it rather than by what happened to the run.
    done = {(str(d.get("symbol")), str(d.get("timeframe")),
             str(d.get("version") or "A")): d
            for d in (auto.get("promotions") or [])
            if isinstance(d, dict)}
    source = blob.get("strategy_source")

    pairs: list[str] = []
    lines: list[str] = []
    if any(not (done.get((str(r.get("symbol")), str(r.get("timeframe")),
                          str(r.get("version") or "A"))) or {}).get("promoted")
           for r in shown):
        # Said ONCE, above the rows, rather than on every line: staged and
        # promoted are different states and the difference is the whole point
        # of this section - a configuration Stage 3 sealed is in the incubator
        # directory and in nobody's git history.
        lines.append("_Sealed and STAGED by Stage 3 — nothing is committed "
                     "until the command below runs._")
        lines.append("")
    for row in shown:
        sym = str(row.get("symbol") or "?")
        tf = str(row.get("timeframe") or "?")
        ver = str(row.get("version") or "A")
        quad = str(row.get("quadrant") or row.get("target_regime")
                   or "quadrant not recorded")
        pairs.append(f"`{sym} {tf}`")
        record = done.get((sym, tf, ver))
        if record and record.get("promoted"):
            sha = str(record.get("commit") or commit or "").strip()
            state = f"promoted `{sha}`" if sha else "promoted"
        elif record:
            state = "**NOT PROMOTED**"
        else:
            state = "staged" if row.get("incubator_dir") else "not staged"
        lines.append(f"• **{sym} {tf}** V{ver} `{quad}` · PF "
                     f"**{_regime_pf_cell(row)}** (n="
                     f"{_fmt_count(row.get('oos_trade_count'))}) · {state}")
        lines.append(f"  params `{compact_params(stage3_param_text(row.get('params')))}`")
        # The blended pair, for the rows somebody is about to act on. A
        # holdout factor read alone lets an edge that collapsed from 2.40 to
        # 1.10 look like a healthy 1.10, and this is the section where that
        # reading costs something.
        lines.append(f"  blended IS {_fmt_metric(row.get('is_profit_factor'))}"
                     f" → OOS {_fmt_metric(row.get('holdout_profit_factor'))}")
        code, params = _seal_prefix(row), _seal_prefix(row,
                                                       "winning_parameters")
        if code != "--" or params != "--":
            lines.append(f"  seals code `{code}` · params `{params}`")
        if record and not record.get("promoted"):
            lines.append(f"  ⚠ {record.get('error') or 'promote.py failed'}")
            lines.append(f"  `{promotion_command(strat, row, source)}`")
        lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    return title, lines, pairs, len(rows) - len(shown)


def outstanding_promotions(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """
    The certified configurations that have NOT been committed yet.

    A configuration Stage 3 sealed is STAGED and not promoted, and one whose
    automatic promotion failed is neither - both still need the command. One
    the orchestrator committed does not, and telling a reader to run a command
    that has already run is how the same strategy gets promoted twice under
    two commits.
    """
    done = {(str(d.get("symbol")), str(d.get("timeframe")),
             str(d.get("version") or "A"))
            for d in (stage3_auto_promotion(blob).get("promotions") or [])
            if isinstance(d, dict) and d.get("promoted")}
    return [r for r in stage3_rows(blob) if r.get("certified")
            and (str(r.get("symbol")), str(r.get("timeframe")),
                 str(r.get("version") or "A")) not in done]


def promotion_footer(strat: str, blob: dict[str, Any]) -> str:
    """
    The single promotion command, as one field value, or `""` when there is
    nothing left to run.

    Its own field rather than the last lines of the bullet block, because the
    field chunker splits a long block across fields on a blank line and half a
    command is a command that runs and does something else. A field cannot be
    split.
    """
    pending = outstanding_promotions(blob)
    if not pending:
        return ("_Every certified configuration is promoted and committed; "
                "nothing is left to run._" if stage3_rows(blob)
                and any(r.get("certified") for r in stage3_rows(blob)) else "")
    return (f"Promotes the {len(pending)} configuration(s) above and nothing "
            f"else — it reads this handoff and promotes only what Gate R "
            f"certified.\n" + FENCE_OPEN
            + "\n".join(unified_promote_command(strat)) + FENCE_CLOSE)


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
    is_window = blob.get("in_sample") or {}
    ho_window = blob.get("holdout") or {}
    rule = blob.get("certification_rule") or {}
    table, hidden, legend = format_stage3_table(rows, max_rows, rule)
    coverage = blob.get("coverage") or {}
    # Every timeframe the summary indexes, not only the one the last Stage 3
    # invocation certified. A card headed `15m` above a table carrying 5m rows
    # describes neither, and that is exactly what a multi-timeframe campaign
    # produced before the summary merged.
    tfs = stage3_timeframes(blob)
    tf = " · ".join(f"`{t}`" for t in tfs) if tfs else None

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
        # What the two number columns ARE. They are Gate R's own quadrant
        # numbers and not the blended sample, and an unlabelled profit factor
        # under a regime-gated verdict is the one value on this card a reader
        # must not have to guess at.
        "`PF` `N` — Gate R's factor and trades INSIDE the target quadrant, "
        "on the holdout. `FAIL·N` starved there · `FAIL·PF` factor missed. "
        "Blended IS/OOS: on the certified rows below.",
    ]
    promo_title, promo_lines, promo_pairs, promo_hidden = \
        format_stage3_promotions(strat, blob)
    if promo_pairs:
        # The passing pairs in one line, above the fold. The block below
        # carries the parameters and the command; this is the answer to "did
        # anything certify", which is the question the card is opened with.
        description.append(f"**{promo_title}** " + " · ".join(promo_pairs)
                           + (f" _(+{promo_hidden} more)_" if promo_hidden
                              else ""))
    if legend:
        description.append("**Target regimes** " + " · ".join(
            f"`{q}` {legend[q]}" for q in sorted(legend)))
    if hidden:
        description.append(
            f"_{hidden} further configuration(s) are not shown — the full "
            f"summary is in the handoff._")

    text = "\n".join(description)
    if len(text) > MAX_EMBED_DESCRIPTION:
        # Trim the TABLE and never the header lines: without the two windows
        # and the verdict rule the numbers underneath are unlabelled.
        keep = MAX_EMBED_DESCRIPTION - 64
        text = text[:keep] + "\n```\n_truncated — see the handoff._"

    fields = [
        {"name": "Strategy", "value": f"`{strat}`", "inline": True},
        # Named in the singular because a single-timeframe campaign is still
        # the common case and the value reads as one; it carries every
        # timeframe the summary indexes, separated.
        {"name": "Timeframe", "value": tf or "not recorded",
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

    # The promotion block goes in LAST, on whatever the rest of the card left
    # of Discord's 6000 characters - sized against the FINISHED embed rather
    # than a constant, exactly as the Stage 2 parameter block is, because the
    # description holding the table is most of the budget.
    handoff = {"name": "Handoff", "value": _fmt_report(str(source or "")),
               "inline": False}
    footer = promotion_footer(strat, blob)
    budget = (MAX_EMBED_TOTAL - _embed_size(embed)
              - len(handoff["name"]) - len(handoff["value"])
              - (len(PROMO_COMMAND_FIELD) + len(footer) if footer else 0))
    # It is the one part of this card somebody ACTS on, which is why it is
    # budgeted ahead of everything optional and why the command that promotes
    # the set is reserved out of the budget before the bullets are chunked: a
    # card that listed three certified configurations and dropped the command
    # sends a reader into the handoff to reconstruct one against a directory
    # holding an audit per timeframe, and picking the wrong file promotes a
    # verdict about different bars.
    promo_fields, promo_dropped = _fenced_fields(
        promo_lines, budget, promo_title, f"{promo_title} (cont.)",
        STAGE3_PROMO_MAX_FIELDS, fence=False)
    if promo_fields and promo_dropped:
        promo_fields[-1]["value"] += (
            f"\n… {promo_dropped} further line(s) — see the handoff.")
    if footer:
        promo_fields.append({"name": PROMO_COMMAND_FIELD, "value": footer,
                             "inline": False})
    embed["fields"] = fields + promo_fields + [handoff]
    return embed


def _fenced_fields(lines: list[str], budget: int, name: str, cont: str,
                   max_fields: int, fence: bool = True
                   ) -> tuple[list[dict], int]:
    """
    A block of lines as Discord fields, and how many did not fit.

    One field per chunk under `MAX_FIELD_VALUE`, at most `max_fields` of them,
    and never past `budget`. A line that does not fit is COUNTED rather than
    truncated - half a SHA-256 is not a shorter checksum but a different string
    that looks like one, and half a promote command is a command that runs and
    does something else.

    `fence=False` renders the lines as MARKDOWN instead of wrapping them in a
    code fence. Stage 3's promotion block needs it: bold, backticks and
    bullets render in a field value and do not inside a fence, and the block is
    a list of configurations rather than a fixed-width table. The chunking is
    otherwise identical, including the preference for cutting on a blank line.
    """
    wrap = (len(FENCE_OPEN) + len(FENCE_CLOSE)) if fence else 0
    if not lines or budget <= wrap + NOTE_RESERVE:
        return [], len(lines)

    fields: list[dict] = []
    spent = 0
    i = 0
    while i < len(lines) and len(fields) < max_fields:
        chunk: list[str] = []
        size = wrap
        while i < len(lines):
            need = len(lines[i]) + (1 if chunk else 0)
            if size + need > MAX_FIELD_VALUE:
                break
            label = name if not fields else cont
            if spent + size + need + len(label) + NOTE_RESERVE > budget:
                break
            chunk.append(lines[i])
            size += need
            i += 1
        if not chunk:
            break
        # Prefer to end a field on a BLANK line when there is more to come.
        # The promotion block is a stack of records separated by blank lines,
        # and a field boundary through the middle of one splits a promote
        # command across two Discord fields - where the half a reader copies
        # is a command that runs and does something else. A block with no
        # blank line in it (a fixed-width table) is chunked straight through.
        if i < len(lines) and "" in chunk[:-1]:
            cut = len(chunk) - 1 - chunk[::-1].index("")
            if cut > 0:
                i -= len(chunk) - cut
                chunk = chunk[:cut]
        label = name if not fields else cont
        # The blank line a chunk was cut on belongs to the break, not to the
        # next field: rendered, it is an empty first row above the record.
        while chunk and not chunk[0]:
            chunk.pop(0)
        if not chunk:
            continue
        body = "\n".join(chunk)
        fields.append({"name": label,
                       "value": (FENCE_OPEN + body + FENCE_CLOSE) if fence
                       else body,
                       "inline": False})
        spent += size + len(label)
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
