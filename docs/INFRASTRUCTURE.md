# Infrastructure

Drafted 2026-08-09. Living document.

---

## Three environments

| Environment | Machine | Role |
|---|---|---|
| **Research** | `backtest` (Proxmox, local) | Data lake, backtesting, strategy development |
| **Incubator** | Proxmox VM (local) | Paper trading — validates strategies against live market |
| **Production** | Cloud VM | Live trading on funded accounts |

Strategies move Research → Incubator → Production. Graduation criteria TBD.

**All three run the same code from the same git repo.** They differ only in
configuration: API keys, account IDs, and demo vs live. If the incubator and
production run different codebases, a strategy that graduates is not the same
strategy that was validated.

**Use the same provisioning script on every machine.** Hand-installing Python
on each box produces subtly different versions — a newer pandas, a different
patch release. Those differences surface as signal discrepancies that look
like strategy bugs and are very hard to trace. `provision-backtest.sh` plus
`uv pip install -r requirements.txt` pins both the interpreter and the
package set.

---

## Research server — `backtest`

| | |
|---|---|
| OS | Ubuntu 26.04 LTS, minimal install |
| Host | Proxmox |
| IP | 192.168.102.119 (VLAN 102) |
| Disk | 100 GB root |
| Python | 3.13 via `uv` (not system 3.14 — scientific stack wheels lag) |
| Venv | `~/src/trading/.venv` |

**Verified stack:** duckdb 1.5.5, pyarrow 25.0.0, pandas 3.0.5.

### Storage

NFSv3 mount at `/mnt/backtest` from UNAS Pro (192.168.100.10). 19 TB, 16 TB free.

    192.168.100.10:/volume/…/Backtest/.data  /mnt/backtest  nfs
      rw,hard,_netdev,nofail,noatime,vers=3,proto=tcp,
      rsize=1048576,wsize=1048576,timeo=600,retrans=2  0  0

Option choices are deliberate:

- `hard` not `soft` — a soft mount returns an I/O error on timeout, which can
  truncate a Parquet write into a corrupt file that still half-parses.
  A hung process is better than bad data.
- `timeo=600` — 60s, the standard pairing with `hard`.
- `vers=3` — UniFi Drive does not export NFSv4.

**Ownership shows a bogus numeric UID** (no v3 idmapping; it collides with the
local `polkitd` user). Permissions are enforced server-side.
**Do not chown or chmod the mount.**

**DuckDB catalog stays on local disk** (`~/.local/share/mdlib/`). NFSv3 with
`local_lock=none` sends file locking over NLM, which is not safe for DuckDB.
The catalog is only an index over Parquet — rebuildable at any time.

**Never put a venv on NFS.** Python imports thousands of small files and each
is a network round trip.

### Network

Measured throughput: 107 MB/s — exactly 1 GbE line rate.

Both endpoints negotiate 2.5 GbE on the switch (Proxmox host on Port 3, UNAS
Pro on Port 4), so the link speed is not the cap. The VM is on VLAN 102 and
the NAS on VLAN 100, so traffic is **routed through the UDM Pro** rather than
switched. That is the most likely bottleneck.

Not pursued — 107 MB/s is adequate for bar data. Revisit only if tick or L1
data is added later.

---

## Directory layout

Code on local disk, data on NFS.

    ~/src/trading/
      mdlib/         lake access; owns session and roll definitions
      data_pull/     vendor downloaders — the ONLY place that talks to a vendor API
      backtest/      engine: fills, costs, metrics
      strategies/    signal logic only
      scripts/       utilities
      courses/       study material
      .venv/

    /mnt/backtest/
      raw/{futures,equities,options,crypto,forex}   original vendor files, write-once
      lake/futures/bars/symbol=X/tf={1m,1d}/year=Y/month=M/
      reference/futures/                            roll calendars, degraded days
      artifacts/                                    backtest outputs

`raw/` is immutable so the lake can be rebuilt after a parser bug without
re-downloading. Vendor name lives in the filename, not a directory level.

---

## Deployment path

### The key fact

**CrossTrade routes everything through the cloud.** Every API request goes out
to `app.crosstrade.io` and is forwarded back down to the add-on running inside
NinjaTrader. There is no local IPC path between Python and NT8, even when both
run on the same machine.

Consequences:

- The internet is in the critical order path. An ISP outage means no orders,
  even with NT8 running normally in front of you.
- **Co-locating Python with NT8 buys nothing for order latency.** Same round
  trip either way.
- NT8 must be open and the add-on connected for any API call to reach the
  broker.

### Therefore

Python runs where the research code already lives, not on the Windows VM.
The Windows VM shrinks to a single job: run NT8 with the add-on connected.
No Python, no scheduling, no scripts there — easier to rebuild if it breaks.

    Python (Linux)  →  app.crosstrade.io  →  CrossTrade add-on  →  NT8  →  broker

### CrossTrade API

| | |
|---|---|
| Base URL | `https://app.crosstrade.io/v1/api` |
| Auth | Bearer token (CrossTrade Secret Key) |
| Endpoints | ~25 — accounts, positions, orders, strategies, executions, quotes, bars |
| Included with | Pro subscription, no usage fees |
| Also offers | WebSocket streaming, OpenAPI spec, MCP endpoint |

**The API is order-based, not position-based** — `PlaceOrder`,
`CancelAndBracket`, `FlattenEverything`, plus `GetPositions`.

So the reconciliation loop lives in Python: read current position, compare to
target, place the difference. This is arguably better than a vendor-side
reconciliation — the logic is testable and lives in code we control.

**Design principle: emit target positions, not entry/exit events.**
"Be long 2 ES right now" is stateless — a missed day, a restart, or a
disconnect self-corrects on the next cycle. "Buy" and "sell" events require
tracking whether each one was acted on.

### Useful endpoints

- `GetBars (historical)` — pull NT8 historical data programmatically instead
  of exporting CSVs by hand. This is the walk-forward data source.
- `FlattenEverything` — hard flat-by-close backstop for the intraday
  portfolio, callable on a schedule independent of strategy logic.
- `GetPositions` — the read side of the reconciliation loop.

### Account Manager

Server-side risk controls: per-account profit/loss thresholds, trading
windows, kill switches, trailing drawdown controls, EOD flattening.

Valuable because these fire even if the Python process has crashed, as long as
the add-on is connected. Risk limits enforced outside the strategy code are
worth more than the same limits inside it.

---

## Operational requirements

**Health check before every order cycle.** Confirm the add-on is connected
before computing signals, rather than discovering it after sending an order
into a dead socket.

**Offline policy — decide before it happens.** If the API is unreachable while
a position is open: wait, or flatten manually? This is a policy decision that
should not be made at 3pm on a bad day.

**Windows VM is a single point of failure.** If it is down, NT8 crashed, or
the add-on lost its WebSocket, orders do not reach the market.

---

## Open items

- Cloud VM provider and spec for production.
- CrossTrade failure semantics: behaviour when the WebSocket drops mid-session.

## Resolved

- **Production VM does not need lake access.** Live strategies use recent bars
  from NT8 via CrossTrade. The 16-year Databento history is research-only, so
  the cloud VM needs no NFS mount and no large disk.
- **FundedNext permits CrossTrade.**
