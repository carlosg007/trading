# Deployment and hardening

Every command is run on the trading box as `cgrullon` unless it says `sudo`.
Nothing here is idempotent by accident — re-running it is safe.

**Read `deploy/RUNBOOK.md` before the first `--live` start.** This file gets the
machine ready; the runbook is what you follow while money is on it.

---

## 0. Preconditions, checked not assumed

```bash
cd ~/src/trading

# The venv, the NFS mount, and the credential file.
.venv/bin/python3 --version                      # expect 3.13.x
mountpoint -q /mnt/backtest && echo "lake mounted" || echo "LAKE NOT MOUNTED"
test -f .env && stat -c '%a %n' .env             # expect 600

# The routing table must load, or nothing downstream starts.
.venv/bin/python3 -c "from portfolio.config_loader import load_portfolio_config; \
    load_portfolio_config(use_cache=False); print('config loads')"

# The gate. Three failures are known and documented in the runbook.
.venv/bin/python3 -m pytest tests/ -q
```

```bash
# Credentials live in .env and are NEVER exported into the environment:
# everything in the environment is inherited by every subprocess, which is how
# a webhook URL — which IS the credential for a live account — reaches an
# unrelated tool's debug output.
chmod 600 ~/src/trading/.env
chmod 600 ~/src/trading/deploy/systemd/trading.env
# `logs/` and `.cache/` are not optional and are not in git. Skipping this line
# does not produce a missing-directory error — systemd opens the append: target
# as PID 1, before the process exists, and reports:
#   Failed at step STDOUT spawning .../python3: No such file or directory
# which names the interpreter and sends you to audit a venv that is fine.
mkdir -p ~/src/trading/logs ~/src/trading/data ~/src/trading/.cache/numba
```

Then make it stick across reboots and fresh clones, so the step above is never
the one somebody skips:

```bash
sudo cp ~/src/trading/deploy/systemd/trading-dirs.conf /etc/tmpfiles.d/
sudo systemd-tmpfiles --create /etc/tmpfiles.d/trading-dirs.conf
```

## 1. Install the units

```bash
# .service and .timer ONLY. `cp deploy/systemd/*` also lands trading.env in
# /etc/systemd/system, where systemd ignores it — leaving a second copy of the
# env file that is not the one EnvironmentFile= reads, for somebody to edit.
sudo cp ~/src/trading/deploy/systemd/*.service /etc/systemd/system/
sudo cp ~/src/trading/deploy/systemd/*.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/trading-*.service \
                            /etc/systemd/system/trading-*.timer

# The feed FIRST: the regime daemon and the loop both read what it spools, and
# a daemon started against an empty spool publishes nothing and exits 1.
#
# It binds :8000. If a listener was started by hand in a terminal it still
# holds that port and this fails with EADDRINUSE — stop it first, and know
# that the gap is a LIVE FEED INTERRUPTION: bars posted while nothing is
# listening are refused, and nothing interpolates a missing bar.
ss -ltnp 'sport = :8000'                  # who holds it, if anyone
sudo systemctl enable --now trading-nt8-listener
curl -s localhost:8000/health | python3 -m json.tool

# Health checks and the regime publisher first. They place no orders.
sudo systemctl enable --now trading-regime-daemon.timer
sudo systemctl enable --now trading-watchdog.timer

# Watch one cycle of each before going further.
journalctl -u trading-regime-daemon -n 40 --no-pager
journalctl -u trading-watchdog -n 40 --no-pager

# The execution loop. THE SHIPPED UNIT IS A DRY RUN — see step 5.
sudo systemctl enable --now trading-master-live
systemctl status trading-master-live --no-pager
```

## 2. Log rotation

```bash
# The stanza is a FILE in the repo, not a heredoc here: a config that only
# exists inside install docs is one nobody notices was never installed.
sudo cp ~/src/trading/deploy/systemd/trading.logrotate /etc/logrotate.d/trading

sudo logrotate --debug /etc/logrotate.d/trading   # dry run, prints its plan
sudo logrotate --force /etc/logrotate.d/trading   # do it once now
```

The audit ledgers are NOT rotated: `data/engine_state.json` is the record of
what this process sent and `data/incubator_ledger.json` is forward-performance
evidence that drives promotion. Both are small and both are evidence.

## 3. Firewall

```bash
# The box makes OUTBOUND connections only — CrossTrade, Discord, the data
# vendor for research. Nothing needs to reach it, and the bar feed is a file
# on a mount rather than a listening port, which is why there is no inbound
# rule for it here.
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Keep your own way in. Change the port if sshd is not on 22, and do this
# BEFORE enabling, or you will lock yourself out of the machine that is
# trading.
sudo ufw limit 22/tcp comment 'ssh, rate limited'

sudo ufw enable
sudo ufw status verbose
```

If the NFS server is on a private network, allow it explicitly rather than
opening a range:

```bash
sudo ufw allow from <NFS_SERVER_IP> to any port 2049 proto tcp comment 'NFS lake'
```

## 4. Unattended security updates

```bash
sudo apt-get update && sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

sudo tee /etc/apt/apt.conf.d/52unattended-upgrades-trading >/dev/null <<'EOF'
// Security updates only. A feature upgrade that restarts Python or the NFS
// client mid-session is an outage during market hours.
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
};
// NEVER reboot automatically. A reboot with a position open is a position
// nobody is watching; see the runbook's reboot procedure.
Unattended-Upgrade::Automatic-Reboot "false";
// Outside the CME session: 17:00-18:00 ET is the daily break.
Unattended-Upgrade::Mail "";
EOF

sudo systemctl edit --force --full apt-daily-upgrade.timer   # set OnCalendar=*-*-* 22:30 UTC
sudo unattended-upgrade --dry-run --debug | tail -20
```

## 5. Arming live trading

This is the only step that risks money, and it is deliberately manual.

```bash
# 1. Read the runbook's pre-flight section. All of it.
# 2. Confirm the loop is healthy in dry run, on the feed you intend to trade:
journalctl -u trading-master-live -n 50 --no-pager | grep -E "bar feed|newest closed|RISK"

# 3. ADD `--live` to the unit's ExecStart. Removing `--dry-run` does NOTHING:
#    `master_live.resolve_dry_run` returns `not args.live`, so dry run is the
#    DEFAULT and a unit carrying neither flag still sends no order. On
#    2026-08-27 a deployment removed `--dry-run`, verified the ExecStart was
#    clean, restarted, and got a loop everybody believed was armed and was not.
#    Only its own `mode=DRY RUN` banner said otherwise.
#
#    Edit deploy/systemd/trading-master-live.service in the REPO, not with
#    `systemctl edit --full`: that writes an override under
#    /etc/systemd/system/ which SHADOWS the repo unit, survives
#    deploy/redeploy.sh, and is invisible to git — so the tracked unit can say
#    one thing while the running one does another.
sudo cp ~/src/trading/deploy/systemd/trading-master-live.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart trading-master-live

# 3a. CONFIRM IT ACTUALLY ARMED. The flag is not the evidence; the banner is.
grep -m1 "LiveExecutionDispatcher  mode=" ~/src/trading/logs/master_live.log | tail -1
#    Expect `mode=LIVE`. `mode=DRY RUN` means it is not armed, whatever the
#    unit says. `firewall-check` reads the same thing across all three places
#    the interlock can disagree.

# 4. Watch the first cycle to completion before walking away.
journalctl -u trading-master-live -f
```

## 6. Verify the kill switch BEFORE you need it

```bash
# Arm it, confirm the loop refuses, disarm it. Do this on the day you deploy,
# not on the day you need it.
.venv/bin/python3 -c "from realtime.risk_firewall import arm_kill_switch; \
    print(arm_kill_switch('deployment drill'))"
journalctl -u trading-master-live -f          # expect RISK BLOCKED [kill_switch]

.venv/bin/python3 -c "from realtime.risk_firewall import disarm_kill_switch; \
    print('cleared' if disarm_kill_switch() else 'was not armed')"
```
