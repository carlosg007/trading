#!/usr/bin/env bash
# Redeploy the trading units after the fixes. Run with sudo.
set -euo pipefail
R=/home/cgrullon/src/trading

echo "== 1. directories (the 209/STDOUT root cause), made durable =="
cp "$R/deploy/systemd/trading-dirs.conf" /etc/tmpfiles.d/
systemd-tmpfiles --create /etc/tmpfiles.d/trading-dirs.conf

echo "== 2. units — .service and .timer ONLY =="
cp "$R"/deploy/systemd/*.service /etc/systemd/system/
cp "$R"/deploy/systemd/*.timer   /etc/systemd/system/
# Stray copy of the env file from an earlier `cp deploy/systemd/*`. systemd
# ignores it, but it is a second copy of the file EnvironmentFile= does NOT
# read, sitting where somebody will edit it.
rm -f /etc/systemd/system/trading.env
systemctl daemon-reload

echo "== 3. log rotation (documented but never installed) =="
cp "$R/deploy/systemd/trading.logrotate" /etc/logrotate.d/trading
logrotate --debug /etc/logrotate.d/trading >/dev/null && echo "logrotate config OK"

echo "== 4. verify =="
systemd-analyze verify /etc/systemd/system/trading-*.service \
                       /etc/systemd/system/trading-*.timer

echo "== 5. one-shot runs =="
systemctl start trading-regime-daemon.service || true
systemctl start trading-watchdog.service      || true

echo "== 6. results =="
systemctl status trading-regime-daemon.service trading-watchdog.service \
    --no-pager -n 0 | grep -E '^●|Active:|Process:' || true
systemctl list-timers --all --no-pager | grep trading || true
