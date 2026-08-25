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

echo "== 5. the NT8 listener =="
# It binds :8000. A listener started by hand in a terminal still holds that
# port, and `systemctl start` would fail with EADDRINUSE — so say so plainly
# rather than enabling a unit that cannot start. Stopping the manual one is a
# LIVE FEED INTERRUPTION: bars posted during the gap get connection-refused,
# and nothing interpolates a missing bar.
# Ask the PORT who holds it, not `pgrep`. `pgrep -f` matches whole command
# lines, so any shell whose argv happens to mention the script — a wrapper, an
# editor, the command that greps for it — counts as a listener and blocks the
# start forever. The socket cannot be wrong about who is bound to it.
MANUAL=$(ss -ltnpH "sport = :8000" 2>/dev/null \
         | grep -oP 'pid=\K[0-9]+' | sort -u || true)
IN_UNIT=$(systemctl show trading-nt8-listener.service -p MainPID --value 2>/dev/null || echo 0)
for pid in $MANUAL; do
    if [ "$pid" != "$IN_UNIT" ]; then
        echo "  REFUSING to start: pid $pid holds :8000 outside systemd."
        echo "  Cut over deliberately, when a dropped bar is acceptable:"
        echo "      kill $pid && sudo systemctl enable --now trading-nt8-listener"
        MANUAL_HELD=1
    fi
done
if [ -z "${MANUAL_HELD:-}" ]; then
    systemctl enable --now trading-nt8-listener.service
    systemctl is-active trading-nt8-listener.service
fi

echo "== 6. one-shot runs =="
systemctl start trading-regime-daemon.service || true
systemctl start trading-watchdog.service      || true

echo "== 7. results =="
systemctl status trading-regime-daemon.service trading-watchdog.service \
    trading-nt8-listener.service --no-pager -n 0 | grep -E '^●|Active:|Process:' || true
systemctl list-timers --all --no-pager | grep trading || true
