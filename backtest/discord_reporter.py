"""
backtest.discord_reporter - post a pipeline card to a Discord webhook.

Location:  ~/src/trading/backtest/discord_reporter.py

Two cards, one transport
------------------------
- **`--mode promotion`** (the default, and `--stage 5`): the handful of numbers
  a promotion decision rests on, passed in on the command line by whatever
  produced them (Stage 3's `gate_audit_<SYMBOL>.json`, Stage 5's `meta.json`).
- **`--mode baseline`** (equivalently `--stage 1`): Stage 1's REGIME FIREWALL
  leaderboard, read straight out of `surviving_assets.json` - every
  (symbol, timeframe) configuration screened, the quadrant it cleared, that
  quadrant's profit factor and trade count, and whether it was PROMOTED to
  Stage 2 or DROPPED.

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
  DROPPED.
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
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.pipeline import (SURVIVORS_FILE, pipeline_dir,        # noqa: E402
                               read_stage)

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
                    "scorecard (--mode promotion) or Stage 1's regime-firewall "
                    "leaderboard (--stage 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Values are printed as supplied - nothing here recomputes a metric.\n"
            f"--webhook may be omitted when ${ENV_WEBHOOK} is set.\n"
            "\n"
            "  --mode promotion   --strat X --symbol NQ --tf 15m --pf 1.42 ...\n"
            "  --stage 1          --strat X [--survivors <surviving_assets.json>]"
        ),
    )
    parser.add_argument("--webhook", default=os.environ.get(ENV_WEBHOOK),
                        help=f"Discord webhook URL (default: ${ENV_WEBHOOK})")
    # `--mode` and `--stage` are two spellings of one choice, and they share a
    # dest so they cannot disagree. A card labelled Stage 1 that was built by
    # the promotion path would announce a screen as a promotion.
    parser.add_argument("--mode", dest="mode", default=None,
                        choices=["promotion", "baseline"],
                        help="promotion (default): the Stage 5 scorecard. "
                             "baseline: Stage 1's regime-firewall leaderboard.")
    parser.add_argument("--stage", dest="stage", default=None,
                        choices=["1", "5"],
                        help="1 == --mode baseline, 5 == --mode promotion")
    parser.add_argument("--strat", required=True, help="strategy name, e.g. sma_momentum_crossover")
    parser.add_argument("--symbol", default="", help="promotion mode: the contract the decision rests on, e.g. NQ")
    parser.add_argument("--tf", default="", help="promotion mode: timeframe, e.g. 15m")
    parser.add_argument("--survivors", default=None,
                        help="baseline mode: path to surviving_assets.json "
                             "(default: <BT_ARTIFACTS>/pipeline/<strat>/"
                             f"{SURVIVORS_FILE})")
    parser.add_argument("--out-dir", default=None,
                        help="baseline mode: override the pipeline directory "
                             "the handoff is looked up in")
    parser.add_argument("--max-rows", type=int, default=STAGE1_MAX_ROWS,
                        help=f"baseline mode: leaderboard rows on the card "
                             f"(default {STAGE1_MAX_ROWS}). Whatever does not "
                             f"fit is COUNTED on the card, never dropped in "
                             f"silence.")
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
    from_stage = {"1": "baseline", "5": "promotion"}.get(stage or "")
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
        embed = build_stage1_embed(args.strat, blob, source=path,
                                   max_rows=args.max_rows)
        rows = stage1_rows(blob)
        kept = sum(1 for r in rows
                   if str(r.get("status") or "").upper() == PROMOTED)
        return embed, (f"Stage 1 screen '{args.strat}' "
                       f"({kept}/{len(rows)} promoted)")

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
