#!/usr/bin/env python3
"""
tools/tool_daemon_control.py - look at the live loop's units, and restart them on request.

Location:  ~/src/trading/tools/tool_daemon_control.py

    python3 tools/tool_daemon_control.py status
    python3 tools/tool_daemon_control.py inspect
    python3 tools/tool_daemon_control.py logs --lines 60
    python3 tools/tool_daemon_control.py restart --confirm

The Supervisor agent's hands on `trading-master-live.service` and
`trading-master-live.timer`. `status`, `inspect` and `logs` are read-only and
need no privilege. `start`, `stop` and `restart` change what is running on a
box that places live orders, so they need BOTH `--confirm` and a sudo rule
that already permits them.

WHAT IT REFUSES TO DO, PERMANENTLY
----------------------------------
**It never edits a unit file, and it never changes the loop's flags.** Arming
this system is ADDING `--live`; it is never REMOVING `--dry-run`, and
`deploy/redeploy.sh` refuses to disarm the live loop for the same reason. A
tool that could rewrite `ExecStart` from a chat message is one message away
from silently disarming the execution interlock while every log line still
reads correctly. Changing what the loop runs stays a human editing a file.

**`enable`/`disable`/`mask` are not implemented either.** They change what
happens after the next reboot, which is a change nobody is watching when it
takes effect.

WHY THE INTERLOCK IS REPORTED FROM THE RUNNING PROCESS
------------------------------------------------------
`status` reads the flags off the process that is ACTUALLY running, not off
the repo's copy of the unit. Those two disagree more often than they should:
`systemctl edit --full` writes an override under /etc/systemd/system that
shadows the tracked unit and survives a redeploy, and a unit edited but not
restarted leaves the old flags on the live process. The repo file, the
installed file and reality are three different answers, and this reports the
one that is placing orders. `firewall-check` does the full three-way
comparison; this is the short version.

A STOPPED LOOP IS NOT NECESSARILY A FAULT
-----------------------------------------
`trading-master-live` is timer-driven and gated on market hours. Inactive
outside a session is the design, not an incident, and the status output says
so rather than colouring every stop red. `market-check` says whether the
market is open.

Exit codes:  0 the unit is active (or the action succeeded) ·
             1 inactive/failed, or a mutating action was refused ·
             2 could not run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any

SERVICE = "trading-master-live.service"
TIMER = "trading-master-live.timer"

#: Everything this tool will ever touch. A unit outside this list is refused,
#: so a mistyped or injected name cannot reach systemctl.
MANAGED_UNITS = (SERVICE, TIMER)

#: Actions that change what is running. Each needs --confirm and sudo.
MUTATING = ("start", "stop", "restart")

#: Properties worth having in every report.
PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "UnitFileState",
              "Result", "ExecMainPID", "ExecMainStatus", "MainPID",
              "ActiveEnterTimestamp", "InactiveEnterTimestamp",
              "NRestarts", "FragmentPath")


class DaemonControlError(RuntimeError):
    """The request could not be carried out."""


def _run(argv: list[str], timeout: int = 20) -> tuple[int, str, str]:
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except FileNotFoundError:
        raise DaemonControlError(f"{argv[0]} is not on PATH") from None
    except subprocess.TimeoutExpired:
        raise DaemonControlError(f"{' '.join(argv)} timed out") from None
    return done.returncode, done.stdout, done.stderr


def show(unit: str) -> dict[str, str]:
    """`systemctl show` as a dict. Read-only and unprivileged."""
    code, out, err = _run(["systemctl", "show", unit,
                           "--property=" + ",".join(PROPERTIES)])
    if code != 0 and not out.strip():
        raise DaemonControlError(f"systemctl show {unit}: {err.strip() or code}")
    values: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key] = value
    return values


def running_command(properties: dict[str, str]) -> str | None:
    """
    The command line of the process that is running now, from /proc.

    Read from the PID rather than the unit file on purpose — see the module
    docstring. None when the unit is not running.
    """
    pid = properties.get("MainPID") or properties.get("ExecMainPID") or "0"
    if not pid.isdigit() or int(pid) == 0:
        return None
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    return " ".join(raw.decode("utf-8", "replace").split("\0")).strip() or None


def interlock(command: str | None) -> str:
    """What the RUNNING process's flags say about live execution."""
    if command is None:
        return "not running — no flags to read"
    live = "--live" in command.split()
    dry = "--dry-run" in command.split()
    if live and dry:
        return ("BOTH --live and --dry-run are on the running process; which "
                "wins is master_live.py's business, not a guess to make here")
    if live:
        return "ARMED — the running process carries --live"
    if dry:
        return "DRY RUN — orders are formatted, not sent"
    return "neither --live nor --dry-run on the running process"


def sudo_ready() -> tuple[bool, str]:
    """Whether a password-less sudo rule already covers us."""
    if shutil.which("sudo") is None:
        return False, "sudo is not on PATH"
    code, _, _ = _run(["sudo", "-n", "true"], timeout=8)
    if code == 0:
        return True, "sudo -n succeeds"
    return False, ("sudo would prompt for a password; this tool never supplies "
                   "one. Grant a NOPASSWD rule for the systemctl verbs, or run "
                   "the action by hand.")


def collect(units: tuple[str, ...] = MANAGED_UNITS) -> dict[str, Any]:
    report: dict[str, Any] = {"units": {}}
    for unit in units:
        properties = show(unit)
        command = running_command(properties)
        report["units"][unit] = {
            "load": properties.get("LoadState", "?"),
            "active": properties.get("ActiveState", "?"),
            "sub": properties.get("SubState", "?"),
            "enabled": properties.get("UnitFileState", "?"),
            "result": properties.get("Result", "?"),
            "since": (properties.get("ActiveEnterTimestamp")
                      or properties.get("InactiveEnterTimestamp") or "?"),
            "restarts": properties.get("NRestarts", "?"),
            "exit_status": properties.get("ExecMainStatus", "?"),
            "fragment": properties.get("FragmentPath", "?"),
            "pid": properties.get("MainPID", "0"),
            "command": command,
            "interlock": interlock(command) if unit == SERVICE else None,
        }
    return report


def format_status(report: dict[str, Any]) -> str:
    lines = ["DAEMON CONTROL — trading-master-live"]
    for unit, info in report["units"].items():
        healthy = info["active"] == "active"
        lines.append("")
        lines.append(f"  {unit}")
        lines.append(f"    state      {info['active']}/{info['sub']}"
                     f"{'' if healthy else '   <-- not active'}")
        lines.append(f"    enabled    {info['enabled']}")
        lines.append(f"    result     {info['result']}   "
                     f"(exit status {info['exit_status']}, "
                     f"{info['restarts']} restarts)")
        lines.append(f"    since      {info['since']}")
        lines.append(f"    unit file  {info['fragment']}")
        if info["command"]:
            lines.append(f"    running    [{info['pid']}] {info['command']}")
        if info["interlock"]:
            lines.append(f"    interlock  {info['interlock']}")
    lines.append("")
    lines.append("  A stopped loop outside a session is the design, not an "
                 "incident — the unit is timer-driven and gated on market "
                 "hours. `market-check` says whether the market is open.")
    return "\n".join(lines)


def act(action: str, unit: str, confirm: bool) -> tuple[int, list[str]]:
    """Carry out a mutating action, or explain precisely why it did not."""
    lines: list[str] = []
    if unit not in MANAGED_UNITS:
        lines.append(f"  refused: {unit!r} is not a unit this tool manages "
                     f"({', '.join(MANAGED_UNITS)})")
        return 1, lines

    command = ["sudo", "-n", "systemctl", action, unit]
    if not confirm:
        lines.append(f"  refused: `{action} {unit}` changes what is running on "
                     f"a box that places live orders.")
        lines.append(f"  Re-run with --confirm to carry it out:")
        lines.append(f"    {' '.join(command)}")
        return 1, lines

    ready, why = sudo_ready()
    if not ready:
        lines.append(f"  refused: {why}")
        lines.append(f"  The command a human would run:")
        lines.append(f"    sudo systemctl {action} {unit}")
        return 1, lines

    code, out, err = _run(command, timeout=90)
    lines.append(f"  ran: {' '.join(command)}  -> exit {code}")
    for stream in (out, err):
        for line in stream.splitlines():
            lines.append(f"    {line}")
    if code != 0:
        return 1, lines

    info = collect((unit,))["units"][unit]
    lines.append(f"  now: {info['active']}/{info['sub']}")
    return 0 if info["active"] == "active" else 1, lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and, on explicit confirmation, restart the "
                    "trading-master-live units. Never edits a unit file.")
    sub = parser.add_subparsers(dest="command", required=True)

    status_cmd = sub.add_parser("status", help="both units, and the interlock")
    status_cmd.add_argument("--json", action="store_true")

    sub.add_parser("inspect", help="status plus the unit file systemd loaded")

    logs_cmd = sub.add_parser("logs", help="recent journal for the service")
    logs_cmd.add_argument("--lines", type=int, default=40)

    for action in MUTATING:
        cmd = sub.add_parser(action, help=f"{action} a managed unit "
                                          f"(needs --confirm)")
        cmd.add_argument("--unit", default=SERVICE, choices=list(MANAGED_UNITS))
        cmd.add_argument("--confirm", action="store_true",
                         help="carry it out; without this the command is "
                              "printed and nothing changes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "status":
            report = collect()
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(format_status(report))
            return 0 if report["units"][SERVICE]["active"] == "active" else 1

        if args.command == "inspect":
            report = collect()
            print(format_status(report))
            for unit in MANAGED_UNITS:
                print("")
                print(f"  ---- systemctl cat {unit} ----")
                code, out, err = _run(["systemctl", "cat", unit])
                for line in (out or err).splitlines():
                    print(f"    {line}")
            return 0 if report["units"][SERVICE]["active"] == "active" else 1

        if args.command == "logs":
            code, out, err = _run(
                ["journalctl", "-u", SERVICE, "-n", str(args.lines),
                 "--no-pager"], timeout=30)
            print(out or err)
            return 0 if code == 0 else 1

        if args.command in MUTATING:
            code, lines = act(args.command, args.unit, args.confirm)
            print(f"DAEMON CONTROL — {args.command} {args.unit}")
            print("\n".join(lines))
            return code

    except DaemonControlError as exc:
        print(f"daemon control could not run: {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
