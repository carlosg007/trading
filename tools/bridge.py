#!/usr/bin/env python3
"""
tools/bridge.py - read bridge_config.yaml, resolve its secrets, address a topic.

Location:  ~/src/trading/tools/bridge.py

    python3 tools/bridge.py check                 # config vs reality
    python3 tools/bridge.py topics                # the routing table
    python3 tools/bridge.py target portfolio_mgmt # telegram:<chat>:<thread>
    tools/tool_contract_guard.py --markdown | python3 tools/bridge.py send roll_alerts

The one place that knows how a report reaches a person. The five tool wrappers
produce text; this decides which Telegram topic it goes to and hands it to
`hermes send`, which already holds the gateway's platform credentials.

WHY THE SECRETS ARE RESOLVED HERE AND NOWHERE ELSE
--------------------------------------------------
`bridge_config.yaml` carries `${VAR}` references, never values. They resolve
against the process environment first and `~/.hermes/.env` second, so a token
lives in exactly one file on disk (mode 0600, git-ignored) and every consumer
reads it through this module. A resolved value is never printed: `check` and
`topics` report whether a variable IS SET, which is the only thing an operator
needs to see and the only thing safe to put in a chat message.

An unresolved reference is an ERROR, never an empty string. `telegram::8` is a
perfectly well-formed target that silently posts nowhere, and a report that
vanishes into a malformed address is worse than one that was never sent.

WHY `check` COMPARES TWO FILES
------------------------------
The Honcho block in `bridge_config.yaml` is documentation; `~/.hermes/honcho.json`
is what the agent actually connects with. Two files describing one connection
drift, and the drift is invisible - both read correctly on their own. `check`
compares them field by field and fails on a disagreement rather than trusting
the prettier one.

Exit codes:  0 fine · 1 a check failed · 2 could not run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from mdlib.env import load_env                                     # noqa: E402

# The repo's `.env`, loaded the way every other entrypoint here loads it -
# filling only names this process has not already set, so an explicit
# `TELEGRAM_FORUM_CHAT_ID=... bridge.py send` still wins. This module's own
# reader below handles `~/.hermes/.env`, which is a DIFFERENT file: Hermes owns
# that one and the gateway writes it. Both feed `environment()`; the process
# environment beats both.
load_env()

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
BRIDGE_CONFIG = HERMES_HOME / "bridge_config.yaml"
HERMES_ENV = HERMES_HOME / ".env"
HONCHO_JSON = HERMES_HOME / "honcho.json"

#: `${VAR}` — the only interpolation this file supports. Deliberately not
#: `$VAR`, so a literal dollar in prose is never mistaken for a reference.
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}")

#: A forum's General topic. Telegram numbers it 1 everywhere it REPORTS a
#: thread, but it refuses `message_thread_id=1` on the way in - sendMessage
#: answers "Bad Request: message thread not found". General is addressed by
#: omitting the thread entirely.
#:
#: This is not cosmetic. Hermes catches that exact error and retries WITHOUT
#: the thread id (plugins/platforms/telegram/adapter.py: "Thread %s not found
#: ... retrying without message_thread_id"), so a send to 1 still succeeds and
#: still reports "sent" - it just arrives through an error path. Emitting the
#: bare target instead keeps the General card on the success path, and leaves
#: that fallback meaning what it should: a topic that has gone missing.
TELEGRAM_GENERAL_THREAD_ID = 1


class BridgeError(RuntimeError):
    """The bridge configuration could not be read or does not agree with reality."""


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def read_env_file(path: Path = HERMES_ENV) -> dict[str, str]:
    """
    `KEY=value` pairs from a dotenv file. Never logged, never echoed.

    Quotes are stripped and `export ` prefixes tolerated, because both appear
    in files people hand-edit. A malformed line is skipped rather than raising:
    one bad line should not make every credential unreadable.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def environment() -> dict[str, str]:
    """The process environment, then `~/.hermes/.env`. Process wins."""
    merged = read_env_file()
    merged.update({k: v for k, v in os.environ.items() if v})
    return merged


def resolve(value: Any, env: dict[str, str], missing: list[str],
            path: str = "") -> Any:
    """Substitute `${VAR}` throughout a loaded document, recording gaps."""
    if isinstance(value, dict):
        return {k: resolve(v, env, missing, f"{path}.{k}" if path else str(k))
                for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, env, missing, f"{path}[{i}]")
                for i, v in enumerate(value)]
    if not isinstance(value, str):
        return value

    def swap(match: re.Match[str]) -> str:
        name = match.group(1)
        found = env.get(name, "")
        if not found:
            missing.append(f"{name} (at {path or '<root>'})")
            return ""
        return found

    return VAR_RE.sub(swap, value)


def referenced_vars(value: Any) -> set[str]:
    """Every `${VAR}` named anywhere in the document."""
    if isinstance(value, dict):
        return set().union(*(referenced_vars(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(referenced_vars(v) for v in value)) if value else set()
    if isinstance(value, str):
        return set(VAR_RE.findall(value))
    return set()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_raw(path: Path = BRIDGE_CONFIG) -> dict[str, Any]:
    try:
        blob = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BridgeError(f"{path}: no such file") from None
    except (OSError, yaml.YAMLError) as exc:
        raise BridgeError(f"{path}: unreadable ({exc})") from None
    if not isinstance(blob, dict):
        raise BridgeError(f"{path}: top level is not a mapping")
    return blob


def load(path: Path = BRIDGE_CONFIG,
         strict: bool = True) -> tuple[dict[str, Any], list[str]]:
    """
    The config with `${VAR}` resolved, plus the list of references that were not.

    `strict` raises on a missing reference. Callers that only want to LOOK at
    the routing table pass `strict=False`; anything that is about to send
    leaves it on.
    """
    raw = load_raw(path)
    missing: list[str] = []
    resolved = resolve(raw, environment(), missing)
    if missing and strict:
        raise BridgeError(
            "unresolved reference(s): " + "; ".join(sorted(set(missing)))
            + f". Set them in {HERMES_ENV} (mode 0600) — an empty value would "
              f"produce a well-formed address that posts nowhere.")
    return resolved, missing


# --------------------------------------------------------------------------
# addressing
# --------------------------------------------------------------------------

def topic_target(config: dict[str, Any], topic: str) -> str:
    """`telegram:<chat_id>:<thread_id>` for a named topic."""
    telegram = config.get("telegram") or {}
    topics = telegram.get("topics") or {}
    if topic not in topics:
        raise BridgeError(
            f"unknown topic {topic!r}. Known: {', '.join(sorted(topics))}")
    chat_id = str(telegram.get("chat_id") or "").strip()
    thread_id = topics[topic].get("thread_id")
    if not chat_id:
        raise BridgeError(
            "the supergroup chat id is not set. `hermes send` addresses a "
            "forum topic as telegram:<chat_id>:<thread_id>, and a thread id "
            "alone cannot be addressed. Set TELEGRAM_FORUM_CHAT_ID in "
            f"{HERMES_ENV}; `python3 tools/bridge.py discover` finds it once "
            "somebody has posted in the group.")
    if thread_id is None:
        raise BridgeError(f"topic {topic!r} declares no thread_id")
    if int(thread_id) == TELEGRAM_GENERAL_THREAD_ID:
        # General is the chat itself; see TELEGRAM_GENERAL_THREAD_ID.
        return f"telegram:{chat_id}"
    return f"telegram:{chat_id}:{thread_id}"


def send(config: dict[str, Any], topic: str, text: str,
         subject: str = "", dry_run: bool = False) -> tuple[int, str]:
    """Hand `text` to `hermes send`. Returns `(exit_code, detail)`."""
    target = topic_target(config, topic)
    argv = ["hermes", "send", "--to", target]
    if subject:
        argv += ["--subject", subject]
    if dry_run:
        return 0, f"would send to {target} ({len(text)} chars)"
    try:
        done = subprocess.run(argv, input=text, text=True, timeout=90,
                              capture_output=True, check=False)
    except FileNotFoundError:
        raise BridgeError("hermes is not on PATH") from None
    except subprocess.TimeoutExpired:
        raise BridgeError(f"`hermes send --to {target}` timed out") from None
    detail = (done.stdout or done.stderr).strip()
    return done.returncode, detail or f"sent to {target}"


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check(path: Path = BRIDGE_CONFIG) -> tuple[bool, list[str]]:
    """Every problem this file can have, named. Returns `(ok, lines)`."""
    lines: list[str] = []
    ok = True

    raw = load_raw(path)
    env = environment()
    config, missing = load(path, strict=False)

    lines.append(f"  config      {path}")

    # -- credentials: reported as set/unset, never as values ----------------
    for name in sorted(referenced_vars(raw)):
        present = bool(env.get(name))
        lines.append(f"  {'SET  ' if present else 'UNSET'}       ${{{name}}}")
        if not present:
            ok = False

    # -- topics -------------------------------------------------------------
    telegram = config.get("telegram") or {}
    topics = telegram.get("topics") or {}
    chat_id = str(telegram.get("chat_id") or "").strip()
    for name, topic in sorted(topics.items()):
        try:
            target = topic_target(config, name) if chat_id else "(no chat id)"
        except BridgeError as exc:
            target = f"UNADDRESSABLE — {exc}"
            ok = False
        lines.append(f"  topic       {name:<16} thread "
                     f"{topic.get('thread_id')}  -> {target}")

    shared: dict[Any, list[str]] = {}
    for name, topic in topics.items():
        shared.setdefault(topic.get("thread_id"), []).append(name)
    for thread, names in sorted(shared.items(), key=lambda kv: str(kv[0])):
        if len(names) > 1:
            lines.append(f"  NOTE        thread {thread} is shared by "
                         f"{', '.join(sorted(names))} — by design where one "
                         f"declares alias_of")

    # -- tools exist --------------------------------------------------------
    for name, tool in sorted((config.get("tools") or {}).items()):
        script = REPO_ROOT / str(tool.get("script", ""))
        if script.exists():
            lines.append(f"  tool        {name:<16} {tool['script']}")
        else:
            lines.append(f"  MISSING     {name:<16} {tool.get('script')}")
            ok = False

    # -- honcho: the mirror must agree with the live config -----------------
    declared = config.get("honcho") or {}
    if not HONCHO_JSON.exists():
        lines.append(f"  honcho      {HONCHO_JSON} does not exist — the "
                     f"agent has no Honcho connection configured")
        ok = False
    else:
        try:
            live = json.loads(HONCHO_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            lines.append(f"  honcho      {HONCHO_JSON} unreadable ({exc})")
            ok = False
            live = {}
        host = os.environ.get("HERMES_HONCHO_HOST", "hermes")
        block = (live.get("hosts") or {}).get(host, {})
        pairs = [("base_url", live.get("baseUrl")),
                 ("workspace", block.get("workspace")),
                 ("peer_name", block.get("peerName")),
                 ("ai_peer", block.get("aiPeer")),
                 ("session_strategy", block.get("sessionStrategy"))]
        for key, actual in pairs:
            want = declared.get(key)
            if want is None:
                continue
            if str(want) == str(actual):
                lines.append(f"  honcho      {key:<16} {actual}")
            else:
                lines.append(f"  DRIFT       {key:<16} bridge_config says "
                             f"{want!r}, honcho.json says {actual!r}")
                ok = False

    # -- is the chat actually a forum? ------------------------------------
    forum_ok, forum_lines = telegram_chat_check(config)
    lines.extend(forum_lines)
    ok = ok and forum_ok

    if missing:
        lines.append("  Unresolved references make the affected targets "
                     "unaddressable; nothing is sent with an empty id.")
    return ok, lines


def telegram_chat_check(config: dict[str, Any],
                        timeout: int = 15) -> tuple[bool, list[str]]:
    """
    Ask Telegram whether the chat is really a forum, because a lie here is silent.

    `sendMessage` to a NON-forum supergroup with a `message_thread_id` returns
    `ok: true` and drops the thread id on the floor - the reply carries
    `message_thread_id: null` and the message lands in the main chat. Measured
    on this account 2026-09-06.

    Nothing downstream can see that. `hermes send` reports success, this
    module's `send()` returns 0, the timer logs a delivered card, and all four
    topics quietly collapse into one undifferentiated stream. That is the exact
    shape of failure this repo exists to catch: every line reads correctly and
    the routing is not happening.

    So the forum flag is checked against the API rather than assumed from the
    fact that a send succeeded.
    """
    import urllib.error                                          # noqa: PLC0415
    import urllib.parse                                          # noqa: PLC0415
    import urllib.request                                        # noqa: PLC0415

    lines: list[str] = []
    telegram = config.get("telegram") or {}
    token = str(telegram.get("bot_token") or "").strip()
    chat_id = str(telegram.get("chat_id") or "").strip()
    if not token or not chat_id:
        lines.append("  telegram    not checked — bot token or chat id unset")
        return True, lines

    query = urllib.parse.urlencode({"chat_id": chat_id})
    url = f"https://api.telegram.org/bot{token}/getChat?{query}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        # Not reachable is not the same as not a forum. Reported, not failed:
        # a check that goes red on a flaky network teaches people to ignore it.
        lines.append(f"  telegram    could not reach the API ({exc}); the "
                     f"forum flag is UNKNOWN, not confirmed")
        return True, lines

    if not payload.get("ok"):
        lines.append(f"  telegram    getChat refused: "
                     f"{payload.get('description', 'no reason given')}")
        return False, lines

    chat = payload.get("result") or {}
    is_forum = bool(chat.get("is_forum"))
    lines.append(f"  telegram    {chat.get('title')!r} "
                 f"({chat.get('type')}, id {chat.get('id')})")

    topics = telegram.get("topics") or {}
    threads = {t.get("thread_id") for t in topics.values()
               if t.get("thread_id") is not None}
    if is_forum:
        lines.append(f"  telegram    topics ENABLED — {len(threads)} distinct "
                     f"thread id(s) in the registry are addressable")
        return True, lines

    if not threads:
        lines.append("  telegram    topics are not enabled, and the registry "
                     "declares no thread ids — consistent")
        return True, lines

    lines.append(
        f"  PROBLEM     topics are NOT enabled on this supergroup, but the "
        f"registry routes to thread id(s) {sorted(threads)}. Telegram accepts "
        f"those sends with ok:true and DISCARDS the thread id, so every topic "
        f"lands in the main chat and nothing reports an error. Enable Topics "
        f"on the group, then re-read the real thread ids — the ones here have "
        f"never been confirmed against this chat.")
    return False, lines


def discover() -> tuple[bool, list[str]]:
    """
    Find the supergroup chat id from what the gateway has already recorded.

    The gateway owns Telegram's long poll, so `getUpdates` from here returns
    nothing while it runs. What it HAS seen lands in its session store, and
    that is what this reads.
    """
    lines: list[str] = []
    found = False

    directory = HERMES_HOME / "channel_directory.json"
    if directory.exists():
        try:
            blob = json.loads(directory.read_text(encoding="utf-8"))
            channels = (blob.get("platforms") or {}).get("telegram") or []
            if channels:
                found = True
                lines.append("  channel_directory.json:")
                for channel in channels:
                    lines.append(f"    {json.dumps(channel)}")
            else:
                lines.append("  channel_directory.json: no telegram channels "
                             "recorded yet")
        except (OSError, json.JSONDecodeError) as exc:
            lines.append(f"  channel_directory.json unreadable ({exc})")

    state = HERMES_HOME / "state.db"
    if state.exists():
        import shutil                                            # noqa: PLC0415
        import sqlite3                                           # noqa: PLC0415
        import tempfile                                          # noqa: PLC0415
        # Copied first: the gateway holds this open in WAL mode and a reader
        # competing with it is not worth the risk for a diagnostic.
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "state.db"
            shutil.copy2(state, copy)
            try:
                conn = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
                rows = list(conn.execute(
                    "select distinct chat_id, chat_type, thread_id, "
                    "display_name, user_id from sessions "
                    "where source = 'telegram' and chat_id is not null"))
            except sqlite3.Error as exc:
                rows = []
                lines.append(f"  state.db unreadable ({exc})")
            if rows:
                found = True
                lines.append("  state.db sessions:")
                for chat_id, chat_type, thread_id, name, user_id in rows:
                    lines.append(f"    chat_id={chat_id} type={chat_type} "
                                 f"thread={thread_id} user={user_id} {name or ''}")
            else:
                lines.append("  state.db: no telegram sessions recorded yet")

    if not found:
        lines.append("")
        lines.append("  Nothing recorded. Post one message in each forum "
                     "topic (the bot must be a member of the group), then run "
                     "this again. Put the chat id in "
                     f"{HERMES_ENV} as TELEGRAM_FORUM_CHAT_ID and your own "
                     "numeric id as TELEGRAM_ADMIN_USER_ID.")
    return found, lines


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read bridge_config.yaml and address the Telegram topics.")
    parser.add_argument("--config", default=str(BRIDGE_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="config vs reality; secrets reported set/unset")
    sub.add_parser("topics", help="the routing table")
    sub.add_parser("discover", help="find the supergroup chat id")

    target_cmd = sub.add_parser("target", help="print a topic's send target")
    target_cmd.add_argument("topic")

    send_cmd = sub.add_parser("send", help="send stdin to a topic")
    send_cmd.add_argument("topic")
    send_cmd.add_argument("--subject", default="")
    send_cmd.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.config)

    try:
        if args.command == "check":
            ok, lines = check(path)
            print("BRIDGE CHECK")
            print("\n".join(lines))
            print(f"\n  {'OK' if ok else 'PROBLEMS ABOVE'}")
            return 0 if ok else 1

        if args.command == "discover":
            found, lines = discover()
            print("TELEGRAM DISCOVERY")
            print("\n".join(lines))
            return 0 if found else 1

        config, _ = load(path, strict=False)

        if args.command == "topics":
            telegram = config.get("telegram") or {}
            print("BRIDGE TOPICS")
            for name, topic in sorted((telegram.get("topics") or {}).items()):
                alias = topic.get("alias_of")
                print(f"  {name:<16} thread {topic.get('thread_id'):<4} "
                      f"agent={topic.get('agent')}"
                      f"{'  alias_of=' + alias if alias else ''}")
            return 0

        if args.command == "target":
            print(topic_target(config, args.topic))
            return 0

        if args.command == "send":
            text = sys.stdin.read()
            if not text.strip():
                print("nothing on stdin to send", file=sys.stderr)
                return 2
            code, detail = send(config, args.topic, text, args.subject,
                                args.dry_run)
            print(detail)
            return 0 if code == 0 else 1

    except BridgeError as exc:
        print(f"bridge: {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
