#!/usr/bin/env python3
"""
realtime.send_test_probe - one hand-built order at the CrossTrade webhook.

    python3 realtime/send_test_probe.py                       # show, send nothing
    python3 realtime/send_test_probe.py --send                # actually transmit
    python3 realtime/send_test_probe.py --action CLOSE --send

WHY THIS EXISTS
===============
`check_crosstrade_connection.py` proves DNS, TLS and the certificate and
deliberately makes ZERO requests to the webhook path - it cannot tell you
whether the endpoint accepts a payload, because finding that out means placing
an order. On 2026-08-31 the live loop was armed and every order it sent came
back HTTP 400, 125 of them across five symbols, with nothing in the response
readable. This is the smallest thing that can answer "what does the endpoint
actually want", one order at a time, run by a person who meant it.

IT SENDS NOTHING WITHOUT `--send`, and that is the same convention
`master_live.py` uses for `--live`. Without the flag it formats the command,
prints it REDACTED, and exits - so the default way to run this file is safe,
and arming it is a word an operator types rather than a side effect of
curiosity. There is no `--dry-run`: a double negative is how somebody ends up
believing they sent nothing when they did.

WHAT IT SENDS
=============
The semicolon plain-text command, as `text/plain`, which is what the
CrossTrade webhook and the NinjaTrader add-on parse:

    key=...; command=place; account=...; instrument=...; action=BUY; qty=1;
    order_type=MARKET; tif=DAY;

`--action CLOSE` builds the FLATTEN form instead, which carries no side and no
quantity - a flatten closes whatever is open, and a guessed quantity is how a
close becomes a reversal. `--qty` is ignored there and says so.

NOTHING IS REIMPLEMENTED HERE. The command comes from
`realtime.crosstrade_formatter`, the credentials from the same
`resolve_credentials` the live loop uses, and the redaction from the same
`redact`. A probe that formatted its own payload would prove the endpoint
accepts THE PROBE's payload, which is not the question.

THE KEY AND THE PATH ARE BOTH CREDENTIALS. The command carries `key=`, and
CrossTrade's `/v1/send/<token>/<token>` route authorises orders on the account
by itself. Every line this prints is scrubbed of both.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime.crosstrade_formatter import (  # noqa: E402
    CrossTradeFormatError,
    format_crosstrade_command,
    format_flatten_command,
    redact,
    sanitize_strategy_tag,
)
from realtime.live_dispatcher import (  # noqa: E402
    _host_only,
    _url_path,
    resolve_credentials,
)

DEFAULT_TIMEOUT_S = 5.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Send ONE order to the CrossTrade webhook and print what "
                     "it says back. Formats and sends nothing unless --send "
                     "is passed."))
    p.add_argument("--account", default="SimIncubator1",
                   help="target account (default: SimIncubator1)")
    p.add_argument("--symbol", default="MES SEP26",
                   help=("instrument, as NinjaTrader names it - 'MES SEP26', "
                         "'MES 09-26' or a bare root (default: 'MES SEP26'). "
                         "Not translated: this script has no roll calendar "
                         "and will not guess an expiry."))
    p.add_argument("--action", default="BUY",
                   choices=("BUY", "SELL", "CLOSE"),
                   help="BUY/SELL place an order; CLOSE sends a flatten")
    p.add_argument("--qty", type=int, default=1,
                   help="contracts (default: 1). Ignored for CLOSE.")
    p.add_argument("--order-type", default="MARKET",
                   help="MARKET only - nothing here carries a price")
    p.add_argument("--tif", default="DAY", help="DAY or GTC (default: DAY)")
    p.add_argument("--strategy-tag", default=None,
                   help=("CrossTrade strategy tag to lock the order to "
                         "(default: none, an untagged order). Pass the SAME "
                         "tag to the BUY/SELL and to the CLOSE - the lock is "
                         "matched by string equality, so a differently-tagged "
                         "flatten does not release it. `;`, `=` and spaces "
                         "are stripped: they are the wire format's field "
                         "separators."))
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   help=f"seconds (default: {DEFAULT_TIMEOUT_S})")
    p.add_argument("--send", action="store_true",
                   help=("ACTUALLY TRANSMIT. Without this the command is "
                         "formatted, printed redacted, and nothing is sent."))
    return p


def _scrub(text: str, webhook_url: str) -> str:
    """Both halves of the credential out of anything printed.

    `redact` replaces the `key=` field and passes text without one through
    unchanged; the webhook PATH is scrubbed separately because `redact` knows
    nothing about it and an endpoint that echoes the request back hands it
    straight to the terminal.
    """
    out = redact(str(text))
    path = _url_path(webhook_url)
    if path:
        out = out.replace(path, "/<redacted>")
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        # `(url, key, source)` - the third element names WHERE they came from
        # (explicit argument, .env, or the process environment), which is
        # worth printing: "the key is set" and "the key you think is set is
        # the one in use" are different facts.
        webhook_url, api_key, cred_source = resolve_credentials()
    except Exception as exc:                                  # noqa: BLE001
        print(f"FAILED  could not resolve credentials: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not webhook_url:
        print("FAILED  CROSSTRADE_WEBHOOK_URL is not set in .env",
              file=sys.stderr)
        return 2

    try:
        if args.action == "CLOSE":
            if args.qty != 1:
                print("  note: --qty is ignored for CLOSE; a flatten carries "
                      "no quantity")
            command = format_flatten_command(account=args.account,
                                             instrument=args.symbol,
                                             key=api_key,
                                             strategy_tag=args.strategy_tag)
        else:
            command = format_crosstrade_command(account=args.account,
                                                instrument=args.symbol,
                                                action=args.action,
                                                qty=args.qty,
                                                order_type=args.order_type,
                                                key=api_key,
                                                tif=args.tif,
                                                strategy_tag=args.strategy_tag)
    except CrossTradeFormatError as exc:
        # Refused before the socket. The formatter's message never contains
        # the key - see `_reject` - so it is safe to print as-is.
        print(f"FAILED  the formatter refused this order: {exc}",
              file=sys.stderr)
        return 2

    W = 78
    print("=" * W)
    print("CROSSTRADE TEST PROBE")
    print("=" * W)
    print(f"  endpoint   : {_host_only(webhook_url)}  "
          f"(path withheld - it is the credential)")
    print(f"  creds from : {cred_source or 'unknown'}"
          + (f"   key: set ({len(api_key)} chars)" if api_key
             else "   key: NOT SET"))
    print(f"  account    : {args.account}")
    print(f"  instrument : {args.symbol}")
    print(f"  action     : {args.action}"
          + ("" if args.action == "CLOSE" else f"   qty: {args.qty}"))
    # THE SANITISED TAG, not what was typed. `--strategy-tag "my strat"`
    # goes on the wire as `mystrat`, and that is the string CrossTrade
    # matches - an operator pairing a BUY with a CLOSE has to be looking at
    # the tag that is actually locked, not the one they typed.
    wire_tag = sanitize_strategy_tag(args.strategy_tag)
    print(f"  strategy   : "
          + (f"{wire_tag}   (locked)"
             + (f"   [typed: {args.strategy_tag!r}]"
                if wire_tag != (args.strategy_tag or "") else "")
             if wire_tag
             else "untagged - acts on whatever the account holds"))
    print(f"  wire form  : semicolon plain text, sent as text/plain")
    print(f"  command    : {_scrub(command, webhook_url)}")
    print("-" * W)

    if not args.send:
        print("  NOT SENT.  Nothing left this process.")
        print("  Re-run with --send to transmit this exact command.")
        print("=" * W)
        return 0

    body = command.encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=body, method="POST",
        headers={"Content-Type": "text/plain", "Accept": "*/*"})

    started = time.perf_counter()
    status: int | None = None
    text = ""
    error = None
    try:
        with urllib.request.urlopen(req, timeout=float(args.timeout)) as resp:
            status = int(getattr(resp, "status", None) or resp.getcode())
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:                                     # noqa: BLE001
            text = ""
        error = f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        error = f"{type(exc).__name__}: {exc.reason}"
    except Exception as exc:                                  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    ok = status is not None and 200 <= status < 300
    print(f"  SENT       : {elapsed_ms} ms")
    print(f"  status     : {status if status is not None else 'no response'}")
    if error:
        print(f"  error      : {error}")
    # THE BODY IS THE POINT OF THIS SCRIPT. Printed whole rather than
    # truncated - a probe run by hand is exactly where the long form belongs -
    # and scrubbed of both halves of the credential first.
    print(f"  response   : {_scrub(text, webhook_url) if text else '(empty)'}")
    print("-" * W)
    print(f"  {'ACCEPTED' if ok else 'REFUSED'}"
          + ("  - check the CrossTrade Alert History and the NT8 account."
             if ok else "  - the response above is what to fix."))
    print("=" * W)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
