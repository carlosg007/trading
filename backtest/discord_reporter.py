"""
backtest.discord_reporter - post a promotion scorecard to a Discord webhook.

Location:  ~/src/trading/backtest/discord_reporter.py

Formats the handful of numbers a promotion decision actually rests on into a
Discord embed and POSTs it. It reads no bars, opens no lake file, and computes
nothing: every value on the card is passed in on the command line by whatever
produced it (Stage 3's `gate_audit_<SYMBOL>.json`, Stage 5's `meta.json`). That
is deliberate - **LLMs never do math and neither does a notifier**. A reporter
that recomputed a profit factor would be free to disagree with the audit it is
announcing, and the two would be compared by nobody.

What it will not do
-------------------
- **It does not decide anything.** Nothing here checks a gate, so a card can be
  posted for a strategy that was never certified. The card announces what it
  was told; `backtest/promote.py` is what refuses an uncertified version.
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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.parse import urlparse

import requests

# Emerald green. Discord wants a decimal int; 0x2ECC71 == 3066993.
EMERALD_GREEN = 0x2ECC71

# Discord's documented limits. Exceeding either is a 400.
MAX_FIELD_VALUE = 1024
MAX_EMBED_TOTAL = 6000

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


def build_payload(embed: dict[str, Any]) -> dict[str, Any]:
    return {"embeds": [embed]}


def _embed_size(embed: dict[str, Any]) -> int:
    """Total characters Discord counts against the 6000 embed limit."""
    total = len(embed.get("title", "")) + len(embed.get("footer", {}).get("text", ""))
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
        description="Post a promotion scorecard to a Discord webhook.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Values are printed as supplied - nothing here recomputes a metric.\n"
            f"--webhook may be omitted when ${ENV_WEBHOOK} is set."
        ),
    )
    parser.add_argument("--webhook", default=os.environ.get(ENV_WEBHOOK),
                        help=f"Discord webhook URL (default: ${ENV_WEBHOOK})")
    parser.add_argument("--strat", required=True, help="strategy name, e.g. sma_momentum_crossover")
    parser.add_argument("--symbol", required=True, help="contract the decision rests on, e.g. NQ")
    parser.add_argument("--tf", required=True, help="timeframe, e.g. 15m")
    parser.add_argument("--pf", default="", help="out-of-sample profit factor, or a token like 'NOT EVALUATED'")
    parser.add_argument("--dd", default="", help="max drawdown in percent, or a token like 'NOT EVALUATED'")
    parser.add_argument("--regime", default="", help="certified regime, e.g. 'High-Vol/Trending'")
    parser.add_argument("--report", default="", help="artifact URL or path to the tear sheet")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the payload and send nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    embed = build_embed(
        strat=args.strat,
        symbol=args.symbol,
        tf=args.tf,
        pf=args.pf,
        dd=args.dd,
        regime=args.regime,
        report=args.report,
    )
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
        print(f"SUCCESS  posted '{args.strat}' ({args.symbol} {args.tf}) to Discord "
              f"[HTTP {result['http_status']}]")
        return 0

    status = result["http_status"]
    where = f"HTTP {status}" if status is not None else "no response"
    print(f"FAILED  Discord rejected the post [{where}]: {result['error']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
