#!/usr/bin/env python3
"""
tools/tool_log_watcher.py - what went wrong in the live loop, with the evidence.

Location:  ~/src/trading/tools/tool_log_watcher.py

    python3 tools/tool_log_watcher.py
    python3 tools/tool_log_watcher.py --since-minutes 60 --markdown
    python3 tools/tool_log_watcher.py --file /tmp/probe.log --json

The Watchdog & Ops agent's eyes on `logs/master_live.log` and
`logs/master_live.err`. It scans the TAIL of each file for three classes of
failure - HTTP 4xx/5xx responses, Python exceptions, and dropped
socket/webhook connections - and emits each finding with the surrounding lines
so an operator reads the failure rather than a category name.

WHY THIS TAILS RATHER THAN READS
--------------------------------
`master_live.log` is 22 MB after two days and rotates daily. Reading it whole
to find the last error costs more than the finding is worth, and a tool that
gets slower every day is one that stops being run. The default window is the
last 2 MiB, seeked from the end; `--tail-bytes 0` reads the whole file when
that is genuinely wanted. A window that starts mid-line drops that partial
first line rather than reporting half a message as a finding.

WHY FINDINGS ARE GROUPED AND COUNTED
------------------------------------
A webhook that is refusing connections writes the same line every 60-second
cycle. Six hundred identical alerts is not six hundred findings, and a card
that lists them is one nobody reads to the end. Findings are grouped by
(category, message with digits and timestamps masked out) and reported once
with an occurrence count and the FIRST and LAST time each was seen. The count
is the signal: 1 is a blip, 600 is an outage.

WHAT THIS DOES NOT DO
---------------------
**It does not decide the loop is healthy.** A clean scan means no matching
line appeared in the window - not that the loop is running, not that orders
are going out, and not that the feed is alive. A process that died silently
writes nothing at all, and this reports zero findings for it. `bt-status`,
`signal-check` and `systemctl status` answer those questions; this one only
reads what was written down.

**It does not follow rotation.** A finding older than the current
`master_live.log` lives in `master_live.log.1` or a `.gz`, and is only scanned
when named with `--file`.

Exit codes:  0 nothing matched · 1 findings present · 2 could not run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LOGS = ("logs/master_live.log", "logs/master_live.err")

#: How much of the end of each file to scan. Two MiB is ~6 hours of cycle
#: lines at the current rate and costs nothing to read.
DEFAULT_TAIL_BYTES = 2 * 1024 * 1024

#: Lines of context kept either side of a matching line.
DEFAULT_CONTEXT = 2

# --------------------------------------------------------------------------
# what counts as a finding
# --------------------------------------------------------------------------
#
# Each pattern is anchored on something a failing subsystem actually writes,
# not on the word "error". A bare /error/ matches "no errors" and the phrase
# "error budget", and a watchdog that cries on those gets muted.

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"

CATEGORY_HTTP = "HTTP"
CATEGORY_EXCEPTION = "EXCEPTION"
CATEGORY_DISCONNECT = "DISCONNECT"

_HTTP_PHRASES = (
    r"Bad Request|Unauthorized|Payment Required|Forbidden|Not Found|"
    r"Method Not Allowed|Request Timeout|Conflict|Gone|Payload Too Large|"
    r"Unprocessable|Too Many Requests|Internal Server Error|Not Implemented|"
    r"Bad Gateway|Service Unavailable|Gateway Time-?out"
)

PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    # -- HTTP 4xx/5xx ----------------------------------------------------
    (CATEGORY_HTTP, SEVERITY_ERROR,
     re.compile(r"\bHTTP[/ ]?(?:\d\.\d\s+)?([45]\d{2})\b", re.I)),
    (CATEGORY_HTTP, SEVERITY_ERROR,
     re.compile(r"\bstatus(?:[ _-]?code)?\s*[=:]\s*([45]\d{2})\b", re.I)),
    (CATEGORY_HTTP, SEVERITY_ERROR,
     re.compile(r"\bresponse\s*[=:]\s*([45]\d{2})\b", re.I)),
    (CATEGORY_HTTP, SEVERITY_ERROR,
     re.compile(rf"\b([45]\d{{2}})\s+(?:{_HTTP_PHRASES})", re.I)),

    # -- Python exceptions ------------------------------------------------
    (CATEGORY_EXCEPTION, SEVERITY_CRITICAL,
     re.compile(r"Traceback \(most recent call last\)")),
    (CATEGORY_EXCEPTION, SEVERITY_ERROR,
     re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b\s*:")),
    (CATEGORY_EXCEPTION, SEVERITY_CRITICAL,
     re.compile(r"\b(?:CRITICAL|FATAL|UNHANDLED)\b")),

    # -- dropped connections ----------------------------------------------
    (CATEGORY_DISCONNECT, SEVERITY_ERROR,
     re.compile(r"\b(?:connection\s+(?:reset|refused|aborted|closed|lost)|"
                r"broken\s+pipe|connection\s*error)\b", re.I)),
    (CATEGORY_DISCONNECT, SEVERITY_ERROR,
     re.compile(r"\bdisconnect(?:ed|ion)?\b", re.I)),
    (CATEGORY_DISCONNECT, SEVERITY_ERROR,
     re.compile(r"\b(?:socket|webhook|websocket)\b[^\n]{0,60}?"
                r"\b(?:fail(?:ed|ure)?|clos(?:ed|ing)|drop(?:ped)?|"
                r"timed?\s*out|unreachable)\b", re.I)),
    (CATEGORY_DISCONNECT, SEVERITY_WARNING,
     re.compile(r"\b(?:read|request|connect)\s+timed?\s*out\b", re.I)),
)

#: A line matching one of these is never a finding, whatever else it matches.
#: These are the loop's own prose about conditions it HANDLED - the log line
#: that says a webhook is absent is not the same event as one failing.
SUPPRESS = (
    re.compile(r"no webhook configured", re.I),
    re.compile(r"\bno errors?\b", re.I),
    re.compile(r"^\s*#", re.I),
)

#: Leading `[2026-09-06T00:45:55+00:00]` on the loop's cycle lines.
TS_RE = re.compile(r"^\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
                   r"(?:[+-]\d{2}:?\d{2}|Z)?)\]?")

#: Digits, hex blobs and quoted paths are masked before grouping, so the same
#: failure at a different timestamp or order id collapses to one finding.
_MASK_NUM = re.compile(r"\d+")
_MASK_HEX = re.compile(r"\b[0-9a-f]{8,}\b", re.I)


def parse_timestamp(line: str) -> datetime | None:
    """The line's own timestamp, or None when it carries no parseable one."""
    match = TS_RE.match(line.strip())
    if not match:
        return None
    text = match.group(1).replace(",", ".").replace("Z", "+00:00")
    text = text.replace(" ", "T", 1)
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def fingerprint(category: str, line: str) -> str:
    """Group key: the line with every number and hex blob masked away."""
    body = TS_RE.sub("", line.strip())
    body = _MASK_HEX.sub("<hex>", body)
    body = _MASK_NUM.sub("#", body)
    return f"{category}|{' '.join(body.split())[:240]}"


def read_tail(path: Path, tail_bytes: int) -> list[str]:
    """
    The last `tail_bytes` of `path` as lines, dropping a partial first line.

    `tail_bytes <= 0` reads the whole file. Decoding is `errors="replace"`:
    a log with one bad byte in it is still the log, and refusing to scan it
    would hide every finding behind an encoding complaint.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        truncated = False
        if tail_bytes > 0 and size > tail_bytes:
            handle.seek(size - tail_bytes)
            truncated = True
        raw = handle.read()
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if truncated and lines:
        # The window opened mid-line; that fragment is not a finding.
        lines = lines[1:]
    return lines


def classify(line: str) -> tuple[str, str, str] | None:
    """`(category, severity, detail)` for the first pattern that matches."""
    if any(rule.search(line) for rule in SUPPRESS):
        return None
    for category, severity, pattern in PATTERNS:
        match = pattern.search(line)
        if match:
            detail = match.group(1) if match.groups() else match.group(0)
            return category, severity, detail
    return None


def scan_lines(lines: list[str], source: str,
               context: int = DEFAULT_CONTEXT,
               since: datetime | None = None,
               first_line_no: int = 1) -> dict[str, dict[str, Any]]:
    """
    Group the findings in `lines`.

    `since` filters on the line's OWN timestamp. A line without one is kept:
    a traceback body carries no timestamp, and dropping it would discard the
    exception the operator is looking for while keeping the header.
    """
    findings: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        verdict = classify(line)
        if verdict is None:
            continue
        category, severity, detail = verdict

        stamp = parse_timestamp(line)
        if since is not None and stamp is not None and stamp < since:
            continue

        key = fingerprint(category, line)
        entry = findings.get(key)
        if entry is None:
            low = max(0, index - context)
            high = min(len(lines), index + context + 1)
            entry = findings[key] = {
                "category": category,
                "severity": severity,
                "detail": detail,
                "source": source,
                "line_no": first_line_no + index,
                "sample": line.strip()[:400],
                # Which row of `snippet` is the line that actually matched.
                # Without it the context reads as the finding, and adjacent
                # ERROR lines from OTHER strategies look like this one's.
                "match_index": index - low,
                "snippet": [ln.rstrip()[:400] for ln in lines[low:high]],
                "count": 0,
                "first_seen": None,
                "last_seen": None,
            }
        entry["count"] += 1
        if stamp is not None:
            iso = stamp.isoformat()
            if entry["first_seen"] is None:
                entry["first_seen"] = iso
            entry["last_seen"] = iso
    return findings


SEVERITY_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_ERROR: 1, SEVERITY_WARNING: 2}


def collect(paths: list[Path], tail_bytes: int, context: int,
            since: datetime | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Findings across every readable path, and a note per path that was not."""
    findings: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for path in paths:
        if not path.exists():
            problems.append(f"{path}: no such file — nothing was scanned for it")
            continue
        try:
            lines = read_tail(path, tail_bytes)
        except OSError as exc:
            problems.append(f"{path}: unreadable ({exc})")
            continue
        for key, entry in scan_lines(lines, str(path), context, since).items():
            existing = findings.get(key)
            if existing is None:
                findings[key] = entry
                continue
            existing["count"] += entry["count"]
            if entry["first_seen"] and (
                    not existing["first_seen"]
                    or entry["first_seen"] < existing["first_seen"]):
                existing["first_seen"] = entry["first_seen"]
            if entry["last_seen"] and (
                    not existing["last_seen"]
                    or entry["last_seen"] > existing["last_seen"]):
                existing["last_seen"] = entry["last_seen"]

    ordered = sorted(
        findings.values(),
        key=lambda f: (SEVERITY_ORDER.get(f["severity"], 9), -f["count"]))
    return ordered, problems


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------

def format_text(findings: list[dict[str, Any]], problems: list[str],
                scanned: list[Path], window: str, limit: int) -> str:
    lines = ["LOG WATCHER — master_live",
             f"  window   {window}",
             f"  scanned  {', '.join(str(p) for p in scanned) or '(nothing)'}",
             f"  findings {len(findings)}"]
    for note in problems:
        lines.append(f"  NOTE     {note}")
    if not findings:
        lines.append("")
        lines.append("  No matching line in the window. That is not a health "
                     "check — a loop that died silently writes nothing at all.")
        return "\n".join(lines)

    for finding in findings[:limit]:
        lines.append("")
        span = ""
        if finding["first_seen"]:
            span = f"  {finding['first_seen']}"
            if finding["last_seen"] != finding["first_seen"]:
                span += f" .. {finding['last_seen']}"
        lines.append(f"  [{finding['severity']}] {finding['category']} "
                     f"x{finding['count']}{span}")
        lines.append(f"    {finding['source']}:{finding['line_no']}  "
                     f"({finding['detail']})")
        for offset, snippet in enumerate(finding["snippet"]):
            marker = ">>" if offset == finding.get("match_index") else " |"
            lines.append(f"     {marker} {snippet}")
    if len(findings) > limit:
        lines.append("")
        lines.append(f"  … {len(findings) - limit} further finding(s) not "
                     f"shown; raise --max-findings to see them.")
    return "\n".join(lines)


def format_markdown(findings: list[dict[str, Any]], problems: list[str],
                    window: str, limit: int) -> str:
    """A card for the `#system-health` topic. Fenced snippets, no tables."""
    head = "*Log watcher — master\\_live*"
    out = [head, f"Window: `{window}`"]
    for note in problems:
        out.append(f"- note: {note}")
    if not findings:
        out.append("")
        out.append("No matching line in the window. Not a health check — a "
                   "loop that died silently writes nothing at all.")
        return "\n".join(out)

    out.append(f"{len(findings)} distinct finding(s).")
    for finding in findings[:limit]:
        out.append("")
        span = finding["first_seen"] or "no timestamp"
        if finding["last_seen"] and finding["last_seen"] != finding["first_seen"]:
            span += f" .. {finding['last_seen']}"
        out.append(f"*{finding['severity']} · {finding['category']} · "
                   f"x{finding['count']}*  \n`{span}`  \n"
                   f"`{finding['source']}:{finding['line_no']}`")
        body = "\n".join(
            (">> " if offset == finding.get("match_index") else "   ") + row
            for offset, row in enumerate(finding["snippet"]))
        out.append(f"```\n{body}\n```")
    if len(findings) > limit:
        out.append(f"\n… {len(findings) - limit} further finding(s) not shown.")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan the live-loop logs for HTTP 4xx/5xx, exceptions "
                    "and dropped connections.")
    parser.add_argument("--file", action="append", default=None, metavar="PATH",
                        help="log to scan; repeatable. Default: "
                             + ", ".join(DEFAULT_LOGS))
    parser.add_argument("--since-minutes", type=float, default=None,
                        help="only report lines stamped within this many "
                             "minutes. Lines without a timestamp are kept.")
    parser.add_argument("--tail-bytes", type=int, default=DEFAULT_TAIL_BYTES,
                        help=f"bytes read from the end of each file "
                             f"(default {DEFAULT_TAIL_BYTES}; 0 = whole file)")
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                        help="lines of context either side of a match")
    parser.add_argument("--max-findings", type=int, default=15,
                        help="findings shown (default 15)")
    parser.add_argument("--markdown", action="store_true",
                        help="emit a Markdown card for #system-health")
    parser.add_argument("--json", action="store_true",
                        help="emit the findings as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    names = args.file if args.file else list(DEFAULT_LOGS)
    paths = [Path(n) if Path(n).is_absolute() else REPO_ROOT / n
             for n in names]

    since = None
    window = "whole tail"
    if args.since_minutes is not None:
        since = datetime.now(timezone.utc) - timedelta(minutes=args.since_minutes)
        window = f"last {args.since_minutes:g} min (since {since.isoformat()})"

    try:
        findings, problems = collect(paths, args.tail_bytes, args.context, since)
    except Exception as exc:                                    # noqa: BLE001
        print(f"log watcher could not run: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"window": window,
                          "scanned": [str(p) for p in paths],
                          "problems": problems,
                          "findings": findings}, indent=2))
    elif args.markdown:
        print(format_markdown(findings, problems, window, args.max_findings))
    else:
        print(format_text(findings, problems, paths, window, args.max_findings))

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
