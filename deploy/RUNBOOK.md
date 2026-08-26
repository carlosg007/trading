# Operational runbook

For the person on the box while the market is open. Deployment and hardening
are in `deploy/DEPLOY.md`; this is what you do when something is happening.

---

## Read this first: what this system is, right now

Three facts that change what every procedure below means.

**1. The feed is live. The execution loop is not.** The NinjaScript publisher
(`LiveBarPublisher`, on the Windows box, outside this repo) POSTs closed bars
to `trading-nt8-listener` on :8000, which appends them to
`/mnt/backtest/artifacts/nt8_bars/`. As of 2026-08-26 that path carries four
streams — `ES_1m.csv`, `GC_1m.csv`, `NQ_1m.csv`, `MNQ_1m.csv` — `/health`
reports HEALTHY, and `scripts/watchdog.py` exits 0 on all four checks.

This REPLACES the standing note that the publisher did not exist and the
watchdog would report DEGRADED continuously. It did, until the listener was
deployed; it does not now, and a DEGRADED verdict today is a finding rather
than the expected background.

`trading-master-live` is nevertheless **inactive**. Bars arriving is not
trading: arming the loop is step 5 of `deploy/DEPLOY.md` and is a deliberate
act. Check which of the two you are looking at before reading anything below
as an outage — a healthy feed under a stopped loop is the normal resting state
of this box.

**2. This loop opens positions and can close only what it opened.** There is no
broker reconciliation. `PositionBook` is deliberately not persisted, and
`EngineState` records what was SENT, not what the account HOLDS. Any procedure
below that says "flatten" means "flatten what this process opened"; anything
else is done by hand in NinjaTrader.

**3. Research and the live stack share this box.** `pipeline-all` and
`precompute-all` are hours-long jobs and run at `nice 19 / ionice idle` so they
yield to the loop, the daemon and the listener. A backtest that wins a CPU
fight with the execution loop is a backtest that cost money. Never launch one
at normal priority; see the shell helpers section.

---

## Daily pre-flight (before the session)

```bash
cd ~/src/trading
systemctl status trading-master-live trading-regime-daemon.timer \
                 trading-watchdog.timer trading-nt8-listener --no-pager
# The receiver's own view. STARVED means it is up and nothing has arrived —
# which is NT8's end, not this one; STALE means bars stopped. A 503 for up to
# one bar width after a restart is expected, not an outage. A rising
# `counters.rejected` is a publisher sending malformed bars. Full behaviour,
# including where a rejection's REASON does and does not appear:
# "The listener, and port 8000" below.
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
* `symbols_active` names a code of 4+ characters, or one with a dash — the
  publisher was rolled onto a physical contract and has forked a new spool.
  See the rollover section

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

### The 24-asset universe, and what is not in it

`_TRADING_UNIVERSE` in `deploy/shell/trading_helpers.sh` is the list every
multi-asset job sweeps, and it is a CHOICE rather than an inventory:

```
equity index   ES NQ RTY YM
energy         CL NG RB HO
metals         GC SI PL
rates          ZB ZN ZF ZT
FX             6E 6J 6B 6A 6C 6S
grains         ZC ZS ZW
```

**HG, ZM and ZL are deliberately absent.** Neither the lake nor
`backtest/specs.py` has them — `mdlib.lake.available_symbols()` does not list
them and `SPECS` has no entry — so adding one buys a failed run, not more
coverage. Pull the definitions and the 1m history first.

BTC, ETH and LE *do* have both, and are still not in the 24. Their contract
definitions are in the UNVERIFIED set, so the list is not "everything the lake
holds" and must not be regenerated from the catalog. Change it by editing the
helper, in git, where the change is reviewable.

### Contract rollover: roll NT8, never the spool name

At the roll, the NT8 **Market Analyzer** rows move to the new front month —
`NQ 12-26` becomes `NQ 03-27`. Nothing on this box changes. The publisher must
go on sending the **generic master symbol**, so the backend keeps appending to
the same `{SYMBOL}_1m.csv` it has always written.

That is not a convention, it is load-bearing: the listener does not strip a
contract month. It upper-cases what it is sent and writes `<that>_<tf>.csv`.
Measured, on this listener:

| the publisher sends | what happens |
|---|---|
| `NQ` | appended to `NQ_1m.csv` — correct |
| `NQ 12-26` | **rejected, HTTP 400**, `counters.rejected` rises — the reason goes to the publisher, not the journal |
| `NQ12-26` | **accepted** — and silently opens a NEW spool `NQ12-26_1m.csv` |

The rejection is the good case: it is loud and it recovers the moment the
publisher is fixed. The third row is the one to fear. It reports `ok`, the
counters rise, `/health` goes HEALTHY on a symbol nobody reads — while
`NQ_1m.csv` stops growing, the regime daemon fails on a symbol with no fresh
bars, and the failure reaches you as regime staleness two checks downstream.
Same shape as the MNQ/NQ split above, same tell:

```bash
ls /mnt/backtest/artifacts/nt8_bars/
curl -s localhost:8000/health | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['symbols_active'])"
```

Every name must be a bare root or a micro. `NQ_1m.csv`, `MNQ_1m.csv`,
`ES_1m.csv`, `GC_1m.csv` is the current, correct set.

**The tell is LENGTH, not digits.** Every legal symbol here is 2 or 3
characters — the 24 roots plus the six micros `MES MNQ M2K MYM MGC MCL`. Six of
the roots contain a digit (`6A 6B 6C 6E 6J 6S`) and so does `M2K`, so "has a
digit" flags the whole FX complex and is the wrong test. Four characters or
more, or a dash anywhere, is a physical contract:

```bash
ls /mnt/backtest/artifacts/nt8_bars/ | grep -vE '^[A-Z0-9]{2,3}_[0-9]+[a-z]\.csv$'
# any output = a spool nothing reads
```

Fix it at the NT8 end; do not rename the spool under a running daemon.

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

## The services on this box

Unit files are in `deploy/systemd/`, versioned. What is installed under
`/etc/systemd/system/` is a COPY — edit the repo, then redeploy, or the next
redeploy silently reverts you.

| unit | type | cadence | what it does |
|---|---|---|---|
| `trading-nt8-listener.service` | simple, `Restart=always` | continuous, binds **:8000** | receives NT8's POSTs, validates, appends to the spool |
| `trading-regime-daemon.service` | oneshot | `.timer`: 5 min, `OnBootSec=2min`, `Persistent=true` | recomputes and publishes the live quadrant |
| `trading-watchdog.service` | oneshot | `.timer`: 2 min, `OnBootSec=3min`, `Persistent=false` | feed staleness, regime write age, state, kill switch |
| `trading-master-live.service` | simple | continuous when armed | the execution loop. **Ships as `--dry-run`** |

The TIMERS are what is enabled; the `.service` units are one-shots the timer
triggers. On a healthy box `systemctl status trading-watchdog.service` reads:

```
Loaded: loaded (/etc/systemd/system/trading-watchdog.service; disabled; ...)
Active: inactive (dead) since Wed 2026-08-26 13:04:51 UTC; 1min ago
```

Both of those alarm people and neither is a fault. `disabled` is correct — the
`.timer` carries the `[Install]` that matters. `inactive (dead)` is a oneshot
that FINISHED. Read `systemctl list-timers | grep trading` for the schedule and
`journalctl -u <unit>` for the last verdict; a failed run shows as
`Active: failed`, not as either of these.

The daemon and the watchdog append to `logs/*.log` and `logs/*.err`; the
listener goes to the **journal** instead, because a process that dies before
opening its log file writes nothing to it, and a bind failure on :8000 is
exactly that kind of death.

### The listener, and port 8000

One port, several streams. NT8 posts one closed bar per instrument it is
attached to, and the listener keys on `(symbol, timeframe)` — so a single
:8000 carries every Market Analyzer row at once, each landing in its own
`{SYMBOL}_{TF}.csv`. A POST body may also be a LIST, which is how a publisher
drains a backlog after a reconnect in one request; each bar in the batch is
judged on its own, so one bad bar neither discards the good ones around it nor
gets dropped quietly.

`/health` reports one of three, and the HTTP code is deliberately not the
signal:

| `status` | HTTP | meaning |
|---|---|---|
| `STARVED` | **503** | up, and no bar has arrived for any symbol yet |
| `HEALTHY` | 200 | every stream inside its staleness limit |
| `STALE` | 200 | a stream has had no bar for 3 bar-widths — also what a shut market looks like |

The ring buffer is in memory and re-seeds lazily on the first post per symbol,
so **every restart shows STARVED/503 until the next bar arrives** — up to one
bar width. That is not an outage, and it is not a lost bar: measured on a
SIGKILL test, the process was back in 5s and the spool showed no gap. Point an
external uptime monitor at the `status` field, never at the HTTP code, or it
pages once per restart. `trading-watchdog` is unaffected: it reads the spool
directory, not this endpoint.

`counters` is cumulative since process start — `accepted`, `duplicate`,
`rejected`. A rising `rejected` is a publisher sending malformed bars.

**The reason is NOT in the journal.** The listener has no log call on the
rejection path: the reason goes back to the PUBLISHER, in the HTTP response
body, and all the journal carries is uvicorn's access line —
`"POST /api/bars HTTP/1.1" 400 Bad Request`. So the journal tells you a bar was
refused and never why. To read the reason, look at the NT8 publisher's own log,
or reproduce the bar against the endpoint:

```bash
# A bar you construct yourself — nothing reaches the spool, but note it does
# increment counters.rejected by one.
curl -s -XPOST localhost:8000/api/bars -d '{"symbol":"NQ 12-26", ...}' | \
  python3 -m json.tool          # {"status": "rejected", "reason": "..."}
```

See the rollover section for the one malformation that is NOT rejected.

### The listener is currently UNAUTHENTICATED

Verified on 2026-08-26. It binds `0.0.0.0:8000` — every interface, not
loopback — and `BT_NT8_LISTEN_TOKEN` is set in neither
`deploy/systemd/trading.env` nor `.env`, so `authorized()` returns True for
every request. Its own startup banner says so:

```
[nt8_listener] auth  OPEN - anything that can route here can inject bars.
                     Set $BT_NT8_LISTEN_TOKEN.
```

`ufw` is active and NT8 reaches the port from the LAN, so the exposure is
whatever that ruleset allows — read it with `sudo ufw status numbered` before
assuming it is only the Windows box. What an injected bar buys an attacker is
not a nuisance: the spool is what the regime daemon draws the quadrant from and
what the execution loop trades on, and a fabricated price is indistinguishable
downstream from a real one.

To close it, set the variable in `deploy/systemd/trading.env` (mode 600, and
NOT in `/etc/systemd/system/` — see the redeploy note below), restart the unit,
and add the matching `X-NT8-Token` header at the publisher. Restarting drops
bars posted during the gap and nothing interpolates them, so do it out of
session.

Note that `deploy/DEPLOY.md` §3 still describes this box as outbound-only,
"the bar feed is a file on a mount rather than a listening port". That was true
before the listener was deployed and is not true now.

### Re-deploying the units

```bash
sudo /home/cgrullon/src/trading/deploy/redeploy.sh
```

It installs `/etc/tmpfiles.d/trading-dirs.conf`, copies `.service` and `.timer`
only, reloads, installs the logrotate stanza, runs `systemd-analyze verify`,
then starts the listener and fires one daemon and one watchdog run.

Two behaviours to know before you run it:

* It copies `*.service` and `*.timer` and **not** `trading.env`, then removes
  any stray copy of it from `/etc/systemd/system/`. systemd ignores that file
  there, so a copy is a second env file that is not the one `EnvironmentFile=`
  reads — sitting exactly where somebody will edit it.
* It asks the **socket** who holds :8000, via `ss -ltnp`, not `pgrep`. If a
  listener started by hand holds the port, it **refuses to start the unit** and
  prints the cut-over command rather than enabling a unit that cannot bind.
  Taking the manual one down is a LIVE FEED INTERRUPTION: bars posted into the
  gap are connection-refused and nothing interpolates a missing bar. Do it when
  a dropped bar is acceptable.

## Shell helpers and aliases

Two sources, and the difference matters when one shadows another.

**In the repo**, `deploy/shell/trading_helpers.sh`, sourced from `~/.bashrc` by
a single line at the end of it. Versioned, reviewable, and a fix reaches the
box through git rather than through somebody editing a dotfile from memory.

| helper | function | what it does |
|---|---|---|
| `precompute-all` | `precompute_all_regimes` | ATR/ADX regime parquets, 24 contracts x 10 timeframes (`1m,2m,3m,5m,15m,30m,1h,2h,4h,1d`), at `nice 19 / ionice idle`. Add `--force` to rebuild caches that exist |
| `pipeline-all` | `run_pipeline_all` | the multi-asset pipeline over the 24-asset universe at `1m,2m,3m,5m,15m,30m`. Takes the strategy as `$1`, or PROMPTS for it |
| `bt-check` / `bt-progress` | — | the progress card: which stage, which timeframe, which contract, how long it has been running |

`precompute-all` does **not** pass `--is-start`/`--is-end`. The script already
defaults to the pinned 2013-01-01..2022-12-31 anchor, and `theta_vol` is LOADED
from that anchor rather than recomputed per request — an override there would
silently rewrite the boundary every certified strategy was drawn against.

`pipeline-all` runs with `--auto-promote` ON: every configuration Stage 3
certifies registers unattended, up to 144 of them (24 contracts x 6
timeframes), without a human seeing a card first. It never overrides a gate. To
look before registering, print the plan first — **the strategy is `$1` and
every other flag comes after it**:

```bash
pipeline-all double_rsi_macd_scalp_20260823 --dry-run   # plan only, runs nothing
```

Reversed, `pipeline-all --dry-run <strategy>` takes `--dry-run` AS the strategy
name and passes the real one through as a stray flag. Or run without
`--auto-promote` and promote afterwards with `--promote-only`, which is what
the Stage 3 Discord card prints.

**In `~/.bashrc` only**, not versioned: `bt-run`, `bt-status`, `bt-ps`,
`bt-watch`, `bt-stage1`, `bt-stage1-check`, `bt-survivors`. These are dotfile
helpers; a change to one is not in git and does not survive a rebuild of the
box. Re-create them from a backup, or move the one you rely on into the repo
file.

### `bt-status` vs `bt-ps` vs `bt-check`

Three different questions, and two of them were the same command until
2026-08-26:

| | reads | answers |
|---|---|---|
| `bt-status` | `active_job.json`, written by `backtest/run.py` | what the multi-asset BATCH is doing — symbol N of M, per-symbol scorecards |
| `bt-ps` | `ps aux`, grepped for the four stage scripts | is a stage process alive at all |
| `bt-check` | the process table AND the artifact tree | where the PIPELINE is up to — stage, timeframe, contract, elapsed |

`bt-ps` was defined as a second `alias bt-status` until 2026-08-26. Sitting 77
lines below the real one in `~/.bashrc`, it silently won: `bt-status` ran a
`ps` grep and never reached `backtest/status.py`, which is what CLAUDE.md
documents it as. Nothing errored — the later definition simply replaced the
earlier one.

That is a live hazard, not history. `~/.bashrc` sources the repo helpers at its
END, so anything the repo defines wins over an earlier dotfile alias of the
same name — and anything added to `.bashrc` *after* that source line would win
over the repo. When adding an alias, grep the whole file first:

```bash
grep -n "alias bt-" ~/.bashrc deploy/shell/trading_helpers.sh
type -a bt-status bt-check          # what the shell will ACTUALLY run
```

`bt-status` and `bt-check` answer different questions and neither replaces the
other: `bt-status` is the batch runner's own progress file, `bt-check` infers a
pipeline's position from processes and artifacts because `run_pipeline.py`
writes no progress file at all. `bt-check` is also the only one of the three
that is safe to run from inside a Claude Code session — it starts nothing.

## Deploying an update

Never during the session. Between 21:00 and 22:00 UTC, or at a weekend.

```bash
cd ~/src/trading
git fetch && git log --oneline HEAD..@{u}      # read what you are about to run
.venv/bin/python3 -m pytest tests/ -q          # the gate, ~18 min, exits 1

git pull
.venv/bin/python3 master_live.py --dry-run --once --tf 1h   # one cycle, no socket

sudo systemctl restart trading-master-live
journalctl -u trading-master-live -n 30 --no-pager
```

The failures that are expected on this box, and are not yours. Measured
2026-08-26: **2 failed, 642 passed in 18:00**, exit 1.

* `test_live_dispatcher.py::test_a_muted_strategy_still_flattens...`
* `test_live_dispatcher.py::test_entries_stay_blocked_while_muted...`
  — both environmental: a live dispatcher is refused when no CrossTrade webhook
  URL is configured, and whether one is depends on suite order and on whether
  `.env` has been loaded. They flip in BOTH directions, so a green pair is not
  evidence either
* `test_suite_runners.py::test_script_suite[test_streaming_lake]`
  — two frame-equality checks at 1m/1h. **INTERMITTENT**, not constant: it
  failed consistently enough to be documented here, and it PASSED on the
  2026-08-26 run. Treat a failure as known and a pass as normal; neither tells
  you anything about your change

So the honest bar is **2 or 3 failures, all named above**. Anything else, or a
failure outside these three tests, is a reason not to deploy. A/B it against
`origin/main` before calling it a regression — these three have wasted more
time being re-diagnosed than they have ever cost in defects.

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

## Server rebuild: the verification checklist

After a rebuild, a restore, or any change to the units or the helpers. Run it
in order — each step's failure mode is invisible from the step after it.

**1. The environment.**

```bash
cd ~/src/trading
test -x .venv/bin/python3 && .venv/bin/python3 -V     # 3.13.x, one venv
mountpoint /mnt/backtest                              # hard mount, not soft
uv pip list | grep -i vectorbt                        # vectorbtpro ONLY
ls -d logs .cache                                     # both must exist
```

`logs/` and `.cache/` are not optional and are not in git. Missing, systemd
opens the `append:` target as PID 1 and reports `Failed at step STDOUT
spawning .../python3: No such file or directory` — which names the interpreter
and sends you to audit a venv that is fine.

Open-source `vectorbt` must NOT appear. `riskfolio-lib` declares it as a
dependency, so a reinstall can pull it back and the wrong package answers
`import vectorbt` silently.

**2. The units.**

```bash
systemd-analyze verify deploy/systemd/*.service       # exit 0
systemd-analyze verify deploy/systemd/*.timer         # exit 0
systemctl list-timers --all --no-pager | grep trading # both timers listed
systemctl is-active trading-nt8-listener              # active
```

**Judge it on the exit code, not on silence.** `systemd-analyze verify` also
loads each unit's dependencies, so it reports on OS-shipped units too. On this
box it prints two lines about `CPUAccounting=` in `xfs_scrub_all.service` and
`system-xfs_scrub.slice` — neither is ours, no `deploy/systemd/*` file mentions
that option, and it still exits 0. A line naming a `trading-*` unit is the one
to act on.

Verify the REPO copies, not the installed ones — the repo is what the next
redeploy will install.

**3. The feed, end to end.**

```bash
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/health   # 200, or 503 if restarted <1 bar ago
curl -s localhost:8000/health | python3 -m json.tool | \
  grep -E '"status"|symbols_active|rejected'
ls /mnt/backtest/artifacts/nt8_bars/                   # master symbols, no digits
.venv/bin/python3 scripts/watchdog.py --tf 1h          # exit 0
```

A 503 immediately after a restart is expected for up to one bar width. A 503
that persists past that is a publisher that is not posting.

**4. The aliases.** A fresh shell, because these exist only in one:

```bash
bash -ic 'type -a bt-run bt-status bt-ps bt-check bt-progress \
                  precompute-all pipeline-all'
bash -ic 'bt-check'          # renders a card, exit 0, from any directory
bash -ic 'bt-status'         # reads active_job.json — NOT a ps grep
```

**`-i`, not `-l`.** `~/.bashrc` opens with `case $- in *i*) ;; *) return;;
esac`, so a NON-interactive shell returns before defining anything — and
`bash -lc 'type -a bt-check'` prints `not found` for every name on a box where
all of them are correct. That reads as seven broken aliases and sends you to
debug a file that is fine. `bash -ic` prints a harmless
`cannot set terminal process group` line first; ignore it.

`type -a` rather than running each: it prints EVERY definition of a name, which
is how the `bt-ps`/`bt-status` collision above would have been caught the day
it was introduced. **Exactly one line per name.** Two means one is dead, and
the live one is whichever was defined last.

Do not "verify" `pipeline-all` or `precompute-all` by running them — they are
hours-long jobs. `type -a` proves they are defined;
`pipeline-all <strategy> --dry-run` prints the plan and runs nothing. Note the
order: the strategy is `$1`, flags follow it.

**5. The gate.**

```bash
.venv/bin/python3 -m pytest tests/ -q                  # ~18 min, exits 1
```

Two or three failures are known on this box and are named under *Deploying an
update* — 642 passed / 2 failed on 2026-08-26. The gate exits non-zero even on
a clean run, so read the summary line, not `$?`. Anything outside those three
tests is a reason to stop.

## Escalation

| symptom | first action |
|---|---|
| loop crash-looping (5 restarts / 5 min, then stopped) | `journalctl -u trading-master-live -n 100`; it stays stopped on purpose |
| `RISK BLOCKED [max_session_loss_usd]` | stop for the day; do not raise the cap |
| unverified claims after a crash | reconcile in NinjaTrader before any restart |
| kill switch armed, nobody knows why | `cat data/KILL_SWITCH` — it records the time and the reason |
| CrossTrade rejecting orders | check the account name in `config/portfolios.json` matches NT8 exactly |
| `/health` 503 for more than one bar width | the publisher stopped posting; check NT8, not this box |
| `/health` HEALTHY but the daemon says no bars | a forked spool — `ls nt8_bars/` for a symbol with digits, or a micro without its parent |
| `systemctl start trading-nt8-listener` fails EADDRINUSE | a hand-started listener holds :8000. `ss -ltnp 'sport = :8000'`; cutting over drops bars |
| a stage is running and nobody knows where it is up to | `bt-check` — stage, timeframe, contract, elapsed |
| an alias runs the wrong thing | `type -a <name>`; two definitions means one is dead — see the alias section |
