"""
mdlib/env.py - the ONE place this repository loads `.env` and resolves the
Discord webhook.

Location:  ~/src/trading/mdlib/env.py

Two jobs, both of which were previously done in several places at once:

1. **Load `~/src/trading/.env` into `os.environ`** before anything reads a
   `BT_*` variable, so an operator opening a fresh terminal never has to
   `source .env` first. Twelve modules used to carry a byte-identical
   fourteen-line bootstrap block; they now call `load_env()`.
2. **Resolve the Discord webhook** from a single, documented alias chain.
   `backtest/discord_reporter.py` read `$BT_DISCORD_WEBHOOK` alone and
   `scripts/incubator_tracker.py` read `$DISCORD_WEBHOOK_URL` first, so the
   same `.env` could configure one card and not the other - the failure being
   a report that is simply never posted, which looks exactly like a quiet
   pipeline.

It lives in `mdlib/` because that is the BOTTOM of the one-way dependency chain
(`agents -> strategies -> backtest -> mdlib -> lake`): `backtest/`, `scripts/`,
`portfolio/`, `realtime/`, `data_pull/` and `master_live.py` may all import it
without inverting anything. It imports `os`, `pathlib` and `dotenv` and nothing
else - no pandas, no numpy - because it runs at the top of every entrypoint,
above the thread-count variables `backtest/run.py` sets before numpy is
imported.

The rules, and the failure each one prevents
--------------------------------------------
- **The repository root is derived from `__file__`, never from the working
  directory.** `find_dotenv()` walks up from the CALLER's cwd, and the runs
  that matter here start from `/mnt/backtest`, from a `--bg` daemon and from
  cron. From any of those it finds nothing, silently, and the script then uses
  every default path as though the file did not exist. `$BT_ENV_FILE` is the
  supported override for a non-standard checkout; it is honoured over the
  derived path so a second checkout can be pointed at its own file.

- **An existing environment variable WINS.** `load_env` only ever fills a name
  that is unset, so `BT_ARTIFACTS=/tmp/x bt-run` still beats the file. The
  reverse - a file that overrides the command - is the kind of difference
  nobody notices until a run writes its evidence somewhere else.

- **It parses the file once.** Twelve importers reach this on a single
  `bt-run`; re-reading and re-deciding per import would let two of them
  disagree if the file changed mid-run.

- **`NO_EXPORT` names are read from the file and NOT put into `os.environ`.**
  `realtime/live_dispatcher.load_env_file` reads the CrossTrade credentials
  out of `.env` directly and documents why they must not enter the process
  environment: everything in `os.environ` is inherited by every subprocess,
  which is how a broker key reaches an unrelated tool's debug output. Loading
  them here would quietly undo that, and nothing downstream would report it.
  Withholding them changes no behaviour - `resolve_credentials` reads the file
  before it reads the environment - and a credential the operator exported
  themselves is untouched, because this module never removes a name.

- **An empty or whitespace value is NOT a webhook.** `DISCORD_WEBHOOK=` left
  in a file is a name somebody meant to fill in, and treating it as set would
  shadow the alias that actually carries the URL and fail with the one message
  ("no webhook") that sends the operator to look at the wrong variable.

- **The webhook is a credential.** `describe_webhook` reports which VARIABLE
  supplied it and never the value; a URL in a log outlives the session that
  wrote it and is directly replayable by anyone who reads it.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

__all__ = [
    "PROJECT_ROOT",
    "ENV_FILE_VAR",
    "DEFAULT_ENV_FILE",
    "NO_EXPORT",
    "DISCORD_WEBHOOK_VARS",
    "WEBHOOK_HINT",
    "env_file_path",
    "load_env",
    "loaded_record",
    "discord_webhook",
    "describe_webhook",
]

# `mdlib/env.py` -> `mdlib/` -> the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE_VAR = "BT_ENV_FILE"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# Read out of the file by `realtime/live_dispatcher.load_env_file`, which
# deliberately keeps them out of `os.environ`. See the module docstring.
NO_EXPORT = frozenset({"CROSSTRADE_WEBHOOK_URL", "CROSSTRADE_API_KEY"})

# Precedence, highest first. `BT_DISCORD_WEBHOOK` leads because it is this
# repository's own name and the four pipeline cards have always read it; the
# other two are the spellings an operator's `.env` is likely to already carry.
DISCORD_WEBHOOK_VARS = ("BT_DISCORD_WEBHOOK", "DISCORD_WEBHOOK_URL",
                        "DISCORD_WEBHOOK")

WEBHOOK_HINT = ("pass --webhook, or set one of "
                + ", ".join("$" + name for name in DISCORD_WEBHOOK_VARS)
                + f" (in the environment or in {DEFAULT_ENV_FILE})")

# The record of what the one load did. Keyed by the resolved path, so a test
# pointing at its own file is not answered from the production file's cache.
_LOADED: dict[str, dict[str, object]] = {}


def env_file_path(path: str | Path | None = None) -> Path:
    """
    Which `.env` this process loads: an explicit argument, then `$BT_ENV_FILE`,
    then `<repo root>/.env`. Never the working directory - see the docstring.
    """
    if path is not None:
        return Path(path).expanduser()
    override = (os.environ.get(ENV_FILE_VAR) or "").strip()
    if override:
        return Path(override).expanduser()
    return DEFAULT_ENV_FILE


def load_env(path: str | Path | None = None, *,
             force: bool = False) -> dict[str, object]:
    """
    Load the `.env` into `os.environ` and return the record of what happened:
    `{"path", "exists", "set", "already_set", "withheld"}` - variable NAMES
    only, never values.

    Filling only unset names (rather than `load_dotenv(override=True)`) is what
    keeps an explicit `BT_ARTIFACTS=... bt-run` authoritative over the file.
    A missing file is not an error: `.env` is optional and every consumer has
    a default, so `exists: False` is a fact to report rather than a failure to
    raise. `force=True` re-reads a file that has already been loaded, which is
    for tests - a second read cannot change a name this process already set.
    """
    resolved = env_file_path(path)
    key = str(resolved)
    if not force and key in _LOADED:
        return _LOADED[key]

    record: dict[str, object] = {"path": key, "exists": resolved.is_file(),
                                 "set": [], "already_set": [], "withheld": []}
    if record["exists"]:
        was_set: list[str] = []
        already: list[str] = []
        withheld: list[str] = []
        for name, value in dotenv_values(resolved).items():
            if value is None:            # a bare `KEY` line declares nothing
                continue
            if name in NO_EXPORT:
                withheld.append(name)
                continue
            if name in os.environ:
                already.append(name)
                continue
            os.environ[name] = value
            was_set.append(name)
        record["set"] = sorted(was_set)
        record["already_set"] = sorted(already)
        record["withheld"] = sorted(withheld)

    _LOADED[key] = record
    return record


def loaded_record(path: str | Path | None = None) -> dict[str, object] | None:
    """What `load_env` recorded for this file, or None if it has not run."""
    return _LOADED.get(str(env_file_path(path)))


def discord_webhook(explicit: str | None = None,
                    env: dict[str, str] | None = None) -> str | None:
    """
    The webhook URL, stripped, or None when none is configured.

    Precedence is `explicit` (a `--webhook` flag), then `DISCORD_WEBHOOK_VARS`
    in order. An empty or whitespace-only value is treated as UNSET at every
    step, so a half-filled name never shadows the one carrying the URL.

    This does not call `load_env`: an entrypoint bootstraps once, at import,
    and a resolver that loaded a file as a side effect of being asked a
    question would make the load order depend on which card was being built.
    """
    source = os.environ if env is None else env
    for value in (explicit, *(source.get(name) for name in DISCORD_WEBHOOK_VARS)):
        cleaned = (value or "").strip()
        if cleaned:
            return cleaned
    return None


def describe_webhook(explicit: str | None = None,
                     env: dict[str, str] | None = None) -> tuple[str | None, str]:
    """
    `(url, source)` - the URL and the NAME that supplied it (`--webhook`, a
    variable name, or `"unset"`). The name is safe to print; the URL is a
    credential and is not.
    """
    source = os.environ if env is None else env
    if (explicit or "").strip():
        return explicit.strip(), "--webhook"
    for name in DISCORD_WEBHOOK_VARS:
        cleaned = (source.get(name) or "").strip()
        if cleaned:
            return cleaned, "$" + name
    return None, "unset"
