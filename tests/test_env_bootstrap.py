#!/usr/bin/env python3
"""
test_env_bootstrap.py — the centralized `.env` loader and the ONE Discord
webhook alias chain (`mdlib/env.py`), and that every operator entrypoint
actually bootstraps before it reads a variable.

Location:  ~/src/trading/tests/test_env_bootstrap.py

Run EITHER way — both report the same answer:

    python tests/test_env_bootstrap.py
    /home/cgrullon/src/trading/.venv/bin/pytest tests/test_env_bootstrap.py

EVERY CASE FAILS THROUGH `assert`, DELIBERATELY — the older `check(name, ok)`
convention in this directory is invisible to pytest, which collects those
suites, watches their checks fail and reports all green.

Nothing here needs the lake. One case starts a loopback HTTP server on port 0
and posts to it; nothing contacts Discord, and no case opens an outbound
socket.

WHAT THIS COVERS, and why each one is here rather than assumed:

  * **A FRESH SHELL NEEDS NO `source .env`.** The failure this whole module
    exists for is a standalone script run from a terminal that never sourced
    the file: it read an unset variable, took a default, and said nothing.
    `test_subprocess_reads_env_without_sourcing` runs a real subprocess with a
    scrubbed environment and a scratch `.env`, from a FOREIGN working
    directory, and asserts the value arrived.

  * **THE WORKING DIRECTORY IS NOT THE ROOT.** The runs that matter start from
    `/mnt/backtest`, from a `--bg` daemon and from cron. A loader anchored on
    cwd (`find_dotenv()`) finds nothing from any of them, silently. Two cases
    pin the `__file__` anchoring: the resolved path, and a subprocess that
    chdir's away and still loads.

  * **AN EXPLICIT VARIABLE STILL WINS.** `BT_ARTIFACTS=/tmp/x bt-run` has to
    beat the file, or a command's own flag is overridden by a file the operator
    is not looking at, and every artifact lands somewhere else.

  * **THE CROSSTRADE CREDENTIALS ARE WITHHELD FROM `os.environ`.**
    `realtime/live_dispatcher.load_env_file` reads them from the file directly
    and documents why they must not enter the process environment — anything in
    `os.environ` is inherited by every subprocess. A blanket `load_dotenv()` in
    `master_live.py` would have quietly undone that, so it is pinned here
    rather than left to a comment.

  * **ONE ALIAS CHAIN, ONE PRECEDENCE.** `backtest/discord_reporter.py` read
    `$BT_DISCORD_WEBHOOK` alone and `scripts/incubator_tracker.py` tried
    `$DISCORD_WEBHOOK_URL` first. With both set they posted to two different
    channels; with only one set, one card posted and the other silently did
    not — indistinguishable from a quiet pipeline. Both now resolve through
    `mdlib.env.discord_webhook`, and a case asserts they return the SAME URL
    for every combination of the three names.

  * **AN EMPTY VALUE IS UNSET, NOT A WEBHOOK.** `DISCORD_WEBHOOK=` left in a
    file is a name somebody meant to fill in. Treating it as set shadows the
    alias that carries the URL and fails with the one message ("no webhook")
    that sends the operator to look at the wrong variable.

  * **THE URL IS NEVER PRINTED; THE VARIABLE NAME IS.** A webhook in a log
    outlives the session that wrote it and is directly replayable.

  * **EVERY ENTRYPOINT BOOTSTRAPS.** A grep-level case over the repository, so
    a new runner that reads `os.environ` without loading the file is caught
    here rather than by an operator wondering why one script writes to
    `/mnt/backtest/artifacts` and another to a default.

  * **THE CARD IS BUILT AND POSTED END TO END** from a `.env` carrying only
    `DISCORD_WEBHOOK_URL`, against a loopback server — the operator's exact
    Stage 1 command, minus Discord.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mdlib import env as mdenv  # noqa: E402

PY = str(REPO / ".venv" / "bin" / "python3")
if not Path(PY).is_file():          # a checkout without the production venv
    PY = sys.executable


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _scratch_env(text: str) -> Path:
    """A `.env` in its own directory, so the path is unambiguous."""
    d = Path(tempfile.mkdtemp(prefix="envtest_"))
    p = d / ".env"
    p.write_text(text)
    return p


def _clean_environ() -> dict[str, str]:
    """
    A subprocess environment with every name these cases care about REMOVED.
    Inheriting the developer's own shell would let a case pass because the
    variable was already exported — which is the exact condition the fix is
    supposed to make unnecessary.
    """
    e = dict(os.environ)
    for name in (*mdenv.DISCORD_WEBHOOK_VARS, *mdenv.NO_EXPORT,
                 mdenv.ENV_FILE_VAR, "BT_ARTIFACTS"):
        e.pop(name, None)
    e["PYTHONPATH"] = str(REPO)
    return e


def _run(code: str, *, cwd: Path | str, env: dict[str, str]) -> str:
    out = subprocess.run([PY, "-c", code], cwd=str(cwd), env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    return out.stdout.strip()


def _fresh(path: Path):
    """`load_env` caches per path; a case that rewrites a file must re-read."""
    return mdenv.load_env(path, force=True)


# --------------------------------------------------------------------------
# where the file is found
# --------------------------------------------------------------------------
def test_default_env_file_is_the_repository_root():
    assert mdenv.DEFAULT_ENV_FILE == REPO / ".env", (
        f"the default .env must be the repository root's, got "
        f"{mdenv.DEFAULT_ENV_FILE}")
    assert mdenv.PROJECT_ROOT == REPO


def test_env_file_path_precedence():
    """explicit argument > $BT_ENV_FILE > the repository root."""
    saved = os.environ.pop(mdenv.ENV_FILE_VAR, None)
    try:
        assert mdenv.env_file_path() == mdenv.DEFAULT_ENV_FILE
        os.environ[mdenv.ENV_FILE_VAR] = "/tmp/from_var/.env"
        assert mdenv.env_file_path() == Path("/tmp/from_var/.env")
        assert mdenv.env_file_path("/tmp/explicit/.env") == Path("/tmp/explicit/.env")
        # An empty override is a name somebody meant to fill in, not a path.
        os.environ[mdenv.ENV_FILE_VAR] = "   "
        assert mdenv.env_file_path() == mdenv.DEFAULT_ENV_FILE
    finally:
        os.environ.pop(mdenv.ENV_FILE_VAR, None)
        if saved is not None:
            os.environ[mdenv.ENV_FILE_VAR] = saved


def test_a_foreign_working_directory_still_loads():
    """
    The cwd-anchored failure, run for real. `find_dotenv()` walks up from the
    working directory, so from `/tmp` it finds nothing and the script proceeds
    on defaults with no message.
    """
    envfile = _scratch_env("BT_UNIT_TEST_MARKER=found-from-elsewhere\n")
    code = ("import os, sys; sys.path.insert(0, %r);"
            "from mdlib.env import load_env; load_env(%r);"
            "print(os.environ.get('BT_UNIT_TEST_MARKER', 'MISSING'))"
            % (str(REPO), str(envfile)))
    got = _run(code, cwd="/tmp", env=_clean_environ())
    assert got == "found-from-elsewhere", got


def test_a_missing_file_is_reported_not_raised():
    rec = mdenv.load_env("/nonexistent/dir/.env", force=True)
    assert rec["exists"] is False
    assert rec["set"] == []


# --------------------------------------------------------------------------
# what is loaded, and what is not
# --------------------------------------------------------------------------
def test_an_existing_variable_wins_over_the_file():
    envfile = _scratch_env("BT_UNIT_TEST_WINS=from-file\n")
    os.environ["BT_UNIT_TEST_WINS"] = "from-shell"
    try:
        rec = _fresh(envfile)
        assert os.environ["BT_UNIT_TEST_WINS"] == "from-shell", (
            "the file overrode an explicit variable; an explicit "
            "BT_ARTIFACTS=... on the command line must beat .env")
        assert "BT_UNIT_TEST_WINS" in rec["already_set"]
        assert "BT_UNIT_TEST_WINS" not in rec["set"]
    finally:
        os.environ.pop("BT_UNIT_TEST_WINS", None)


def test_crosstrade_credentials_are_withheld_from_the_environment():
    """
    `realtime/live_dispatcher.load_env_file` reads these from the FILE and
    documents that they must not enter `os.environ`, where every subprocess
    inherits them. Loading them centrally would undo that silently.
    """
    envfile = _scratch_env("CROSSTRADE_API_KEY=secret\n"
                           "CROSSTRADE_WEBHOOK_URL=https://example.invalid/hook\n"
                           "BT_UNIT_TEST_OK=yes\n")
    for name in mdenv.NO_EXPORT:
        os.environ.pop(name, None)
    rec = _fresh(envfile)
    try:
        for name in mdenv.NO_EXPORT:
            assert name not in os.environ, f"{name} was exported into os.environ"
            assert name in rec["withheld"], f"{name} was not recorded as withheld"
        assert os.environ.get("BT_UNIT_TEST_OK") == "yes", (
            "withholding a credential must not stop the rest of the file loading")
    finally:
        os.environ.pop("BT_UNIT_TEST_OK", None)


def test_the_live_dispatcher_still_reads_the_withheld_credentials():
    """
    Withholding must cost nothing: the live loop's own reader takes them
    straight out of the file, so the credentials still resolve.
    """
    from realtime.live_dispatcher import load_env_file

    envfile = _scratch_env("CROSSTRADE_API_KEY=secret-key\n"
                           "CROSSTRADE_WEBHOOK_URL=https://example.invalid/hook\n")
    values = load_env_file(envfile)
    assert values["CROSSTRADE_API_KEY"] == "secret-key"
    assert values["CROSSTRADE_WEBHOOK_URL"] == "https://example.invalid/hook"


def test_load_is_idempotent_across_importers():
    """Twelve modules reach this on one `bt-run`; the file is parsed once."""
    envfile = _scratch_env("BT_UNIT_TEST_ONCE=first\n")
    _fresh(envfile)
    envfile.write_text("BT_UNIT_TEST_ONCE=second\n")
    try:
        rec = mdenv.load_env(envfile)                 # cached, not re-read
        assert rec["set"] == ["BT_UNIT_TEST_ONCE"]
        assert os.environ["BT_UNIT_TEST_ONCE"] == "first"
    finally:
        os.environ.pop("BT_UNIT_TEST_ONCE", None)


def test_subprocess_reads_env_without_sourcing():
    """The whole point: a fresh shell, no `source .env`, no export."""
    envfile = _scratch_env("BT_ARTIFACTS=/tmp/artifacts_from_dotenv\n")
    env = _clean_environ()
    env[mdenv.ENV_FILE_VAR] = str(envfile)
    code = ("import os, sys; sys.path.insert(0, %r);"
            "from mdlib.env import load_env; load_env();"
            "print(os.environ.get('BT_ARTIFACTS', 'MISSING'))" % str(REPO))
    assert _run(code, cwd="/tmp", env=env) == "/tmp/artifacts_from_dotenv"


# --------------------------------------------------------------------------
# the webhook alias chain
# --------------------------------------------------------------------------
def test_precedence_is_flag_then_bt_then_url_then_bare():
    env = {"BT_DISCORD_WEBHOOK": "bt", "DISCORD_WEBHOOK_URL": "url",
           "DISCORD_WEBHOOK": "bare"}
    assert mdenv.discord_webhook("flag", env) == "flag"
    assert mdenv.discord_webhook(None, env) == "bt"
    assert mdenv.discord_webhook(None, {k: v for k, v in env.items()
                                        if k != "BT_DISCORD_WEBHOOK"}) == "url"
    assert mdenv.discord_webhook(None, {"DISCORD_WEBHOOK": "bare"}) == "bare"
    assert mdenv.discord_webhook(None, {}) is None


def test_an_empty_or_blank_value_is_unset():
    """
    A half-filled name must not shadow the alias carrying the URL — that fails
    with "no webhook" and points the operator at the wrong variable.
    """
    env = {"BT_DISCORD_WEBHOOK": "", "DISCORD_WEBHOOK_URL": "   ",
           "DISCORD_WEBHOOK": "https://discord.com/api/webhooks/real"}
    assert mdenv.discord_webhook(None, env) == "https://discord.com/api/webhooks/real"
    assert mdenv.discord_webhook("  ", env) == "https://discord.com/api/webhooks/real"
    assert mdenv.discord_webhook(None, {"BT_DISCORD_WEBHOOK": " "}) is None


def test_the_value_is_stripped():
    assert mdenv.discord_webhook(None, {"BT_DISCORD_WEBHOOK": " https://x/y \n"}) \
        == "https://x/y"


def test_describe_names_the_variable_and_not_the_url():
    url = "https://discord.com/api/webhooks/123/tok"
    for name in mdenv.DISCORD_WEBHOOK_VARS:
        got, source = mdenv.describe_webhook(None, {name: url})
        assert got == url
        assert source == "$" + name
        assert url not in source, "the credential leaked into the source label"
    assert mdenv.describe_webhook(url, {})[1] == "--webhook"
    assert mdenv.describe_webhook(None, {}) == (None, "unset")


def test_the_hint_names_every_alias():
    """The failure message has to name the variable the operator should set."""
    for name in mdenv.DISCORD_WEBHOOK_VARS:
        assert "$" + name in mdenv.WEBHOOK_HINT, name


def test_both_cards_resolve_identically():
    """
    The regression this exists for: two resolvers, two precedences, and an
    operator whose `.env` configured one card and not the other.
    """
    from backtest import discord_reporter
    from scripts import incubator_tracker

    combos = [
        {"DISCORD_WEBHOOK_URL": "u"},
        {"BT_DISCORD_WEBHOOK": "b"},
        {"DISCORD_WEBHOOK": "d"},
        {"BT_DISCORD_WEBHOOK": "b", "DISCORD_WEBHOOK_URL": "u"},
        {"BT_DISCORD_WEBHOOK": "b", "DISCORD_WEBHOOK_URL": "u",
         "DISCORD_WEBHOOK": "d"},
        {"BT_DISCORD_WEBHOOK": "", "DISCORD_WEBHOOK_URL": "u"},
        {},
    ]
    for env in combos:
        tracker = incubator_tracker.webhook_from_env(env)
        shared = mdenv.discord_webhook(None, env)
        assert tracker == shared, (
            f"the tracker and the shared resolver disagree on {env}: "
            f"{tracker!r} vs {shared!r}")
    # The reporter has no resolver of its own left to disagree with.
    assert not hasattr(discord_reporter, "ENV_WEBHOOK"), (
        "discord_reporter re-declared a webhook variable name; the chain lives "
        "in mdlib.env so the two cannot fall out of step")


# --------------------------------------------------------------------------
# every entrypoint bootstraps
# --------------------------------------------------------------------------
def test_every_env_reading_entrypoint_bootstraps():
    """
    A new runner that reads `os.environ` without loading the file takes a
    default and says nothing. Caught here rather than by an operator wondering
    why two scripts disagree about where the artifacts are.
    """
    skip_dirs = {"tests", ".venv", "docs"}
    offenders = []
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO)
        if rel.parts[0] in skip_dirs or path == REPO / "mdlib" / "env.py":
            continue
        text = path.read_text(errors="ignore")
        if '__name__ == "__main__"' not in text:
            continue
        if "os.environ" not in text and "os.getenv" not in text:
            continue
        if "mdlib.env" not in text:
            offenders.append(str(rel))
    assert not offenders, (
        "these entrypoints read the environment without loading .env: "
        + ", ".join(offenders))


def test_the_documented_runners_bootstrap():
    """The commands CLAUDE.md tells an operator to type, named explicitly."""
    required = ["backtest/discord_reporter.py", "backtest/baseline.py",
                "backtest/scan.py", "backtest/run.py", "backtest/run_pipeline.py",
                "backtest/audit_gates.py", "backtest/verify_full.py",
                "backtest/promote.py", "backtest/status.py",
                "scripts/incubator_tracker.py", "realtime/regime_daemon.py",
                "master_live.py"]
    missing = [r for r in required
               if "mdlib.env" not in (REPO / r).read_text(errors="ignore")]
    assert not missing, f"no .env bootstrap in: {missing}"


# --------------------------------------------------------------------------
# end to end: the operator's Stage 1 command, against a loopback server
# --------------------------------------------------------------------------
class _Collector(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):                                   # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        _Collector.received.append(json.loads(self.rfile.read(n) or b"{}"))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):                           # keep the suite quiet
        pass


def test_stage1_card_posts_from_a_dotenv_carrying_only_discord_webhook_url():
    """
    `.env` -> discovery -> card -> POST, with `DISCORD_WEBHOOK_URL` as the ONLY
    name set and nothing exported in the shell. This is the operator's reported
    failure, end to end; the server is loopback, so no case here contacts
    Discord.
    """
    _Collector.received = []
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        work = Path(tempfile.mkdtemp(prefix="stage1card_"))
        (work / ".env").write_text(
            f"DISCORD_WEBHOOK_URL=http://127.0.0.1:{port}/api/webhooks/1/tok\n")

        survivors = work / "surviving_assets.json"
        survivors.write_text(json.dumps({
            # `stage` is the integer the handoff carries; `pipeline.read_stage`
            # refuses a file written by another stage or another strategy.
            "stage": 1, "strategy": "unit_test_strat",
            "evaluated": 1, "promoted": 1, "surviving": ["NQ"],
            "surviving_pairs": [{
                "symbol": "NQ", "tf": "15m", "version": "A",
                "status": "PROMOTED", "optimal_regime": "High Volatility Trending",
                "quadrant": "Q1", "regime_pf": 1.42, "regime_trade_count": 88,
                "regime_win_rate": 0.51, "regime_net_pnl": 12000.0,
                "kill_switch_regimes": ["Q2", "Q3", "Q4"]}],
            "screen_results": [{
                "symbol": "NQ", "tf": "15m", "version": "A",
                "status": "PROMOTED", "quadrant": "Q1",
                "optimal_regime": "High Volatility Trending",
                "regime_pf": 1.42, "regime_trade_count": 88,
                "pf_a": 1.10, "pf_b": None, "reason": "Version A"}],
            "dropped": [],
        }))

        env = _clean_environ()
        env[mdenv.ENV_FILE_VAR] = str(work / ".env")
        out = subprocess.run(
            [PY, str(REPO / "backtest" / "discord_reporter.py"),
             "--stage", "1", "--strat", "unit_test_strat",
             "--survivors", str(survivors)],
            cwd="/tmp", env=env, capture_output=True, text=True, timeout=180)

        assert out.returncode == 0, (
            f"the Stage 1 card failed with only $DISCORD_WEBHOOK_URL in .env:\n"
            f"{out.stdout}\n{out.stderr}")
        assert "SUCCESS" in out.stdout, out.stdout
        assert "$DISCORD_WEBHOOK_URL" in out.stdout, (
            "the card did not report WHICH variable supplied the webhook: "
            + out.stdout)
        assert f"127.0.0.1:{port}" not in out.stdout + out.stderr, (
            "the webhook URL was printed; it is a credential")
        assert len(_Collector.received) == 1, _Collector.received
        assert "embeds" in _Collector.received[0]
    finally:
        server.shutdown()
        server.server_close()


def test_no_webhook_names_all_three_variables():
    """
    With nothing configured the failure has to say what to set — this is the
    message the operator hit, and it used to name one variable out of three.
    """
    work = Path(tempfile.mkdtemp(prefix="nohook_"))
    (work / ".env").write_text("BT_UNIT_TEST_NOISE=1\n")
    survivors = work / "surviving_assets.json"
    survivors.write_text(json.dumps({
        "stage": 1, "strategy": "unit_test_strat",
        "evaluated": 0, "promoted": 0, "surviving": [],
        "surviving_pairs": [], "screen_results": [], "dropped": []}))

    env = _clean_environ()
    env[mdenv.ENV_FILE_VAR] = str(work / ".env")
    out = subprocess.run(
        [PY, str(REPO / "backtest" / "discord_reporter.py"),
         "--stage", "1", "--strat", "unit_test_strat",
         "--survivors", str(survivors)],
        cwd="/tmp", env=env, capture_output=True, text=True, timeout=180)

    assert out.returncode == 1, out.stdout + out.stderr
    for name in mdenv.DISCORD_WEBHOOK_VARS:
        assert "$" + name in out.stderr, (
            f"the failure message never names ${name}:\n{out.stderr}")


# ==========================================================================
# The script runner. `assert` is the failure mechanism, so pytest and this
# report the same thing — see the module docstring.
# ==========================================================================
def main() -> int:
    cases = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    print(f".env bootstrap and webhook resolution — {len(cases)} cases\n")
    for name, fn in cases:
        try:
            fn()
        except Exception as e:                   # noqa: BLE001 - reported below
            failures.append((name, e))
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
        else:
            print(f"  PASS  {name}")
    print()
    if failures:
        print(f"{len(failures)} of {len(cases)} FAILED: "
              + ", ".join(n for n, _ in failures))
        return 1
    print(f"all {len(cases)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
