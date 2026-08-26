# Operational runbook

For the person on the box while the market is open. Deployment and hardening
are in `deploy/DEPLOY.md`; this is what you do when something is happening.

---

## Read this first: what this system is, right now

Two facts that change what every procedure below means.

**1. The live path has no data feed.** Live bars come from NinjaTrader through
`realtime/nt8_feed.py`, and the NinjaScript publisher does not exist yet —
`/mnt/backtest/artifacts/nt8_bars/` has never held a file. `--feed nt8`
correctly refuses; `--feed auto` falls back to the lake, whose newest bar is
weeks old. **Until the publisher runs, this stack cannot trade, and the
watchdog will report DEGRADED continuously.** That is the truth being reported,
not a fault in the reporting.

**2. This loop opens positions and can close only what it opened.** There is no
broker reconciliation. `PositionBook` is deliberately not persisted, and
`EngineState` records what was SENT, not what the account HOLDS. Any procedure
below that says "flatten" means "flatten what this process opened"; anything
else is done by hand in NinjaTrader.

---

## Daily pre-flight (before the session)

```bash
cd ~/src/trading
systemctl status trading-master-live trading-regime-daemon.timer \
                 trading-watchdog.timer trading-nt8-listener --no-pager
# The receiver's own view. STARVED means it is up and nothing has arrived —
# which is NT8's end, not this one; STALE means bars stopped. `counters`
# rising with `rejected` is a publisher sending malformed bars, and each
# rejection carries its reason in the journal.
#
# /health answers 503 while STARVED and 200 while HEALTHY or STALE. The ring
# buffer is in memory and is re-seeded lazily, on the first post per symbol,
# so EVERY restart shows STARVED/503 until the next bar arrives — up to one
# bar width. That is not an outage: measured on a SIGKILL test, the process
# was back in 5s and the spool shows no missing bar. Point an external uptime
# monitor at `status`, not at the HTTP code, or it will page once a restart.
# trading-watchdog is unaffected — it reads the spool directory, not this
# endpoint.
curl -s localhost:8000/health | python3 -m json.tool
.venv/bin/python3 scripts/watchdog.py --tf 1h        # expect exit 0
.venv/bin/python3 realtime/regime_reader.py          # bar age < ~2 bars
.venv/bin/python3 -c "from realtime.lifecycle import EngineState, startup_report; \
    print(startup_report(EngineState()))"
```

Stop and investigate if any of these is true:

* the watchdog exits non-zero
* `startup_report` lists UNVERIFIED CLAIMS — a previous run sent orders whose
  outcome nobody has confirmed. Reconcile in NinjaTrader before trading.
* the kill switch is armed and you do not know who armed it
* the newest bar is older than about two bars
* the spool holds a MICRO's tape instead of the full-size contract's — see below

### The spool must carry the FULL-SIZE tape

```bash
ls /mnt/backtest/artifacts/nt8_bars/     # expect NQ_1m.csv, not MNQ_1m.csv
```

The NT8 push writes one file per instrument it is attached to, named for that
instrument. The alias table is one-directional on purpose (`realtime/
contract_alias.py`): a micro resolves to its parent, never the reverse. So a
spool holding `MNQ_1m.csv` answers the LOOP, which asks for the basket's micro
— and does not answer the DAEMON, which asks for the certification symbol NQ.

That split is close to invisible. Every component reports correctly:

    watchdog     ok   feed    newest MNQ 1h bar closed 73 min ago
    daemon       NQ 1h: FAILED — the feed returned no 1h bars
    watchdog     FAIL regime  daemon wrote 292 min ago

The watchdog's feed check passes, because there IS a fresh bar in the spool —
it is just not the one anything is gated on. The only line naming the cause is
the daemon's, and the failure reaches the operator as regime STALENESS two
checks downstream. Attach the NT8 indicator to **NQ**: the daemon then matches
it exactly and the loop's MNQ request aliases onto the same file, so the gate
and the trade read one tape rather than two that can drift apart.

## During the session

The console line to watch on every cycle:

```
[master_live] newest closed bar 2026-08-25 16:00:00+00:00 — closed 17.1 min ago (1h bar = 60 min)
```

`A WHOLE BAR BEHIND` means the feed is lagging, stale, or the market is shut.
The first two are actionable; check which before doing anything.

`RISK BLOCKED <symbol> [rule] detail` means the pre-trade gate refused an
order. It is not an error — the gate did its job. Read the rule:

| rule | what happened | what to do |
|---|---|---|
| `kill_switch` | trading is halted | find out who armed it and why before clearing |
| `session_cutoff` | past the entry cutoff | nothing; exits are unaffected |
| `duplicate_bar` | this signal was already sent for that bar | nothing — this is the restart guard working |
| `max_contracts_per_order` / `_per_symbol` | size beyond the cap | check the sizer's ATR input before raising a cap |
| `max_open_positions` | too many concurrent | expected under a broad signal; review exposure |
| `max_orders_per_session` | a loop is looping | investigate before raising |
| `max_session_loss_usd` | this loop's realised loss hit the local cap | **stop for the day.** Do not raise the cap to keep trading |

## EMERGENCY: stop everything now

```bash
# 1. HALT — blocks every new order this process or the next one would send.
#    Takes effect on the next order, not the next cycle.
cd ~/src/trading && .venv/bin/python3 -c \
  "from realtime.risk_firewall import arm_kill_switch; print(arm_kill_switch('MANUAL HALT'))"

# 2. Close what THIS PROCESS opened, and alert.
.venv/bin/python3 -c "
from realtime.lifecycle import emergency_halt
from realtime.live_dispatcher import LiveExecutionDispatcher
d = LiveExecutionDispatcher(dry_run=False)
print(emergency_halt('MANUAL HALT', dispatcher=d))"

# 3. Then look at the account in NinjaTrader. Step 2 cannot close a position
#    it did not open — placed by hand, by a previous run, or by another tool.
```

Stopping the service does **not** flatten. A stopped loop with a position open
is a position nobody is watching:

```bash
sudo systemctl stop trading-master-live     # halts trading, holds inventory
```

## Deploying an update

Never during the session. Between 21:00 and 22:00 UTC, or at a weekend.

```bash
cd ~/src/trading
git fetch && git log --oneline HEAD..@{u}      # read what you are about to run
.venv/bin/python3 -m pytest tests/ -q          # the gate, ~17 min

git pull
.venv/bin/python3 master_live.py --dry-run --once --tf 1h   # one cycle, no socket

sudo systemctl restart trading-master-live
journalctl -u trading-master-live -n 30 --no-pager
```

The three test failures that are expected on this box, and are not yours:

* `test_live_dispatcher.py::test_a_muted_strategy_still_flattens...`
* `test_live_dispatcher.py::test_entries_stay_blocked_while_muted...`
  — both flip with suite order depending on whether `.env` has been loaded
* `test_suite_runners.py::test_script_suite[test_streaming_lake]`
  — two frame-equality checks at 1m/1h, pre-existing and unexplained

Anything else failing is a reason not to deploy.

## Rolling restart without losing the session

The loop holds no sockets between cycles, so a restart loses nothing in flight
— but it does drop the in-memory `PositionBook`, so the new process starts
believing it holds nothing.

```bash
# 1. Check what is open BEFORE restarting. If anything is, decide first:
#    restart and lose the ability to auto-close it, or flatten and restart flat.
.venv/bin/python3 -c "from realtime.lifecycle import EngineState, startup_report; \
    print(startup_report(EngineState()))"

# 2. SIGTERM lets the current cycle finish; systemd waits up to 90s.
sudo systemctl restart trading-master-live
```

## Reboot with a position open

Do not let unattended-upgrades do this — `Automatic-Reboot "false"` is set for
this reason.

1. Arm the kill switch.
2. Flatten in NinjaTrader, by hand, and confirm the account is flat.
3. Reboot.
4. On boot, run the pre-flight. `EngineState` will list claims from before the
   reboot; they are records of what was sent, not positions. Clear them only
   after confirming the account is flat.
5. Disarm the kill switch.

## When the feed goes stale mid-session

The loop does not trade on stale bars — `drop_forming_bar` and the horizon cut
mean it produces fewer bars rather than wrong ones, and the watchdog alerts.
So the danger is not a bad trade, it is an open position nobody is updating.

1. `journalctl -u trading-master-live -n 20` — confirm bar age is the problem.
2. If a position is open: arm the kill switch and manage the position by hand.
   If flat: no urgency; fix the feed.
3. `systemctl restart trading-regime-daemon` republishes the regime; the loop
   picks up the new state file on its next cycle without a restart.

## Escalation

| symptom | first action |
|---|---|
| loop crash-looping (5 restarts / 5 min, then stopped) | `journalctl -u trading-master-live -n 100`; it stays stopped on purpose |
| `RISK BLOCKED [max_session_loss_usd]` | stop for the day; do not raise the cap |
| unverified claims after a crash | reconcile in NinjaTrader before any restart |
| kill switch armed, nobody knows why | `cat data/KILL_SWITCH` — it records the time and the reason |
| CrossTrade rejecting orders | check the account name in `config/portfolios.json` matches NT8 exactly |
