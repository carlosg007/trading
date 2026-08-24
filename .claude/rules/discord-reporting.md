---
name: discord-reporting
description: "The five Discord stage cards: what each reads, what it transcribes, what it refuses to derive, and the webhook alias chain."
paths:
  - "backtest/discord_reporter.py"
  - "tests/test_stage4_card.py"
  - "tests/test_stage5_card.py"
---

# The Discord stage cards

**`backtest/discord_reporter.py`** — the webhook notifier, and the only place
this repo posts anything to Discord. Five cards over one transport:
`--mode promotion` (the default, `--stage 5`) is the promotion scorecard, whose
values are passed in on the command line or resolved from what the promotion
wrote; `--stage 1` / `--mode baseline` is
Stage 1's regime-firewall leaderboard, read straight out of
`surviving_assets.json`; `--stage 2` / `--mode scan` is Stage 2's parameter
optimization summary, read straight out of `stage2_summary.json`; `--stage 3` /
`--mode audit` is Stage 3's gate audit and certification, read straight out of
`stage3_audit_summary.json` or out of a single `gate_audit_<SYMBOL>_<TF>.json`; `--stage 4` / `--mode verify` is Stage 4's
full-lifecycle summary, read straight out of the `dual_metrics_<SYMBOL>.json`
snapshots in one run's artifacts directory.

- **The Stage 5 card resolves itself from the promotion, and `--strat` alone
  is a complete command (2026-08-24).** It reads
  `strategies/approved_incubator/<strat>/meta.json`, the `dual_metrics.json`
  beside it, and the Stage 3 `gate_audit_<SYMBOL>_<TF>.json` the first cites,
  and fills the contract, the timeframe, Gate R's out-of-sample profit factor,
  its win rate, the max drawdown, the certified quadrant and the tear sheet
  path — plus the portfolio membership, read from `config/portfolios.json`.
  `--audit-file` and `--metrics` name those two files directly and are spelled
  the way `backtest/promote.py` spells them, so a Stage 5 command already in an
  operator's shell history runs here instead of dying on an unrecognised
  argument — which is what sends somebody back to retyping four numbers by
  hand. `--symbol`, `--tf`, `--pf`, `--win`, `--dd`, `--regime` and `--report`
  still override anything, `--incubator` moves the directory it all comes from
  and `--portfolios` moves the routing table.
  - **The WIN RATE is measured on the same sample as the profit factor beside
    it** — Gate R's own quadrant, on the holdout — and falls back to the
    blended holdout and then to the run snapshot, naming which of the three it
    read. A profit factor read without one is a ratio with no sense of how it
    was earned: 1.22 from a 53% hit rate and 1.22 from a 20% one are different
    strategies to sit in front of. Unlike the profit factor a snapshot value is
    NOT declined here, because the field is headed `Win Rate` and claims no
    window of its own. **The unit comes from the SOURCE, never from the
    magnitude**: `backtest/profiler.py` writes a percentage (53.75) and
    `report.summarize_result` a fraction (0.5233), and 0.52 and 52.0 are both
    plausible win rates — a magnitude test cannot tell them apart, it can only
    usually guess right.
  - **PORTFOLIO MEMBERSHIP comes from `active_strategies` in
    `config/portfolios.json` and is never typed.** Being in
    `approved_incubator/` is a record that a version was CHOSEN and explicitly
    not permission to trade it, so a promotion no portfolio names reads
    `Incubator Staging (Evaluation / Shadow)` and an allocated one reads
    `Active <portfolio> (Allocated)` — the same green embed otherwise. A
    strategy on an incubator AND a prop portfolio names both, which is what the
    two tracks are for; two portfolios of ONE track is the config
    `portfolio.config_loader` refuses to load and is flagged rather than
    resolved to one of them. A registry that cannot be READ reports
    `NOT RESOLVED` rather than the staging token, because "no portfolio names
    this" is a claim about a file nobody managed to open. The table is read as
    plain JSON rather than through `portfolio.config_loader`: nothing in
    `backtest/` may import from `portfolio/`, and that loader RAISES for an
    unassigned strategy — the ordinary state this field exists to report.
  - **The out-of-sample profit factor is Gate R's or nothing.**
    `dual_metrics.json` and meta.json's snapshot both carry a profit factor and
    both measured it over a window that CONTAINS the holdout. The field is
    headed `Out-of-Sample PF`, so taking it from either would put an in-sample
    number under an out-of-sample heading with every other field on the card
    still correct. With no readable certification the field keeps its
    `NOT REPORTED` token and the card says the number was DECLINED rather than
    absent — the two are fixed by different work. The drawdown beside it is the
    audit's HOLDOUT drawdown, from the same window as the factor above it, and
    is labelled `NOT the holdout` when it falls back to a snapshot.
  - **The contract and the timeframe resolve as a PAIR, from one file.**
    meta.json's top-level `symbols`/`timeframe` are the MODULE's declarations —
    every contract it targets, at the timeframe it prefers — while a promotion
    is one contract at one timeframe: `t3_braid_scalp_20260823` declares
    `NQ,ES,CL,GC` at 5m and was certified on NQ at 1h. Mixing the halves
    announces NQ at 5m for a run nobody made, with both halves individually
    true. The certification's own `audit_symbol` plus the timeframe in the
    audit's FILENAME is what supplies the pair when the audit itself is gone;
    the module's declarations are used only where it names exactly ONE symbol.
  - **Every resolved value names its file, and nothing is invented.** An
    `Auto-resolved` field lists what came from where and over which window; a
    card whose values were all typed does not carry it and is byte-identical to
    what it was before this existed. A value no file carries stays
    `NOT REPORTED` and the card names the files it looked in. A `--symbol` that
    disagrees with the certification is honoured and FLAGGED. A file named
    explicitly and missing RAISES; one this went looking for on its own is a
    note. Another strategy's meta.json, snapshot or gate audit is refused, and
    so is a gate audit written by another stage. `--audit-file`, `--metrics`
    and `--incubator` on any of the other four cards are REFUSED rather than
    parsed and ignored.
- **The Stage 4 card is the only one that carries no verdict, because Stage 4
  produces none.** That window CONTAINS the Stage 3 holdout, so every number on
  it is in-sample by construction: the card says so above the table, states it
  again in a `Certifies` field, and is drawn in graphite — never the promotion
  green, and never Stage 3's teal. A lifecycle run read as a certification is
  the one mistake this card could cause on its own.
- **One row per contract**: `SYM · TF · CAGR · NET P&L · TRD · FRIC · QD ·
  ALPHA`. Ordered by contract and NOT ranked — Stage 4 selects nothing, and
  sorting by CAGR would give a leaderboard's shape to a stage that produced no
  leaderboard. **Nothing is summed across contracts**: symbols are never
  blended here, so a total net P&L would be a portfolio number no backtest in
  this repository produced.
- **FRIC is the one derived value on any of the five cards**, and it is a
  division of two figures the run already recorded (`total_costs` over
  `gross_pnl`), not a metric re-scored from bars. It inherits
  `verify_full.cost_drag`'s rule exactly: **undefined, printed `--`, where
  gross P&L was not positive**, because a strategy that lost money gross has no
  profit for its costs to be a share of and `0%` there reads as a run that cost
  nothing. It is on the card because it is the number that decides whether an
  edge is real — an edge handing 85% of its gross to the broker dies on one
  extra tick of slippage while every ratio above it still reads fine.
- **ALPHA is the designated quadrant's `net P&L × profit factor`**, transcribed
  from the UNSUFFIXED `regime_profile_<SYM>_<TF>.json` Stage 4 wrote for the
  same run (looked up in the artifacts directory, then in the pipeline
  directory the profiler actually writes to). The suffixed `_version_a` files
  beside it are **Stage 1's**, profiled over the charter window alone, and are
  never read as a fallback — that would put an in-sample score under a
  lifecycle heading with every column still lining up. A missing profile and a
  run that designated no home quadrant both print `--` and are COUNTED
  separately under the table, because they are fixed by different work.
- **A snapshot that cannot be read is a ROW, not a dropped file.** It carries
  `--` in every metric column and is counted in `Unreadable`; a card shorter
  than the run it announces reads as a shorter run, and "the file is corrupt"
  and "this contract was never verified" must not share a shape. A directory
  holding NO snapshots is refused outright rather than posted as an empty card:
  that is the wrong directory, not an empty result.
- **Another strategy's snapshot is refused**, the way `pipeline.read_stage`
  refuses another strategy's handoff. `dual_metrics.json` is not written
  through `write_stage` so that check cannot be delegated, and the failure it
  prevents is worse here — the card would post one strategy's lifecycle under
  another's name. The one accepted mismatch is the module spelling `strat`,
  which is what `approved_incubator/<strat>/strat.py` records.
- **`--artifacts` defaults to the NEWEST `verify_<stamp>/`**, sorted by NAME
  (the stamp Stage 4 wrote it under) rather than by mtime, which moves when a
  directory is copied off the NFS mount. The directory it resolved to is named
  on the card and in the success line, so a defaulted choice is never a silent
  one. Stage 4 prints the exact command with `--artifacts` filled in.

- **The Stage 3 card reads EITHER of Stage 3's two on-disk shapes (2026-08-24),
  and the file named by hand always wins.** `stage3_audit_summary.json` is the
  campaign INDEX; `gate_audit_<SYMBOL>_<TF>.json` is the AUTHORITATIVE verdict
  for one pair and the file Stage 3 writes FIRST — the summary is transcribed
  from it. `--summary` takes the index and REFUSES a pair audit rather than
  adapting it; `--audit` takes either, discriminated on CONTENT (`results`
  versus `versions`) and never on the filename, because that is the flag every
  existing Stage 3 command already spells. With neither, the index is preferred
  and EVERY suffixed per-pair audit in the directory is the fallback — only the
  suffixed ones, since the unsuffixed `gate_audit_<SYMBOL>.json` duplicates
  whichever timeframe ran last, and never just one of them, because picking one
  announces a single certification while the rest sit on disk unread. Before
  this the card could only read the index, so a finished
  `audit_gates.py --strat X --tf 1h` whose summary was missing, stale or
  flattened by the pre-merge overwrite posted nothing at all. Whatever it
  resolved to is named on the card and in the success line, and nothing on disk
  names both files it looked for rather than posting an empty card.
  `stage3_rows_from_audit` is a pure TRANSCRIPTION into the same row shape,
  under the same field names, that `audit_gates.write_stage3_summary` writes —
  deliberately not imported from there, because `backtest.audit_gates` pulls in
  the engine and `vectorbtpro` and a notifier that cannot post because the
  simulation stack failed to import is a quiet pipeline.
  `tests/test_stage3_charter.py` hands both builders the same audit and
  requires an identical row, field for field, which is what stops the two
  transcriptions drifting. Windows that DISAGREE across assembled pairs (an
  explicit `--holdout-end` beside one that defaulted to the present) print
  `varies by pair` rather than the first file's, and `coverage` says its counts
  describe the audits READ and not the campaign Stage 3 was asked to certify —
  a pair whose audit raised wrote no file and cannot appear.
- **The Stage 3 card carries the whole claim per configuration**: the strategy,
  BOTH windows (in-sample, so a reader knows what the parameters were fitted
  to, and the holdout, which is the verdict — an open end prints as `present`),
  the symbol, timeframe, target regime quadrant, Gate R's status, and the
  quadrant profit factor and trade count Gate R was measured on.
- **The table is built to a WIDTH, not to a column list** (45 characters,
  `STAGE3_TABLE_WIDTH`). Discord wraps a code block that overruns the viewport,
  and a wrapped fixed-width table is worse than none: every row becomes two,
  the second one unlabelled, and the columns a reader is comparing stop lining
  up under each other. The ten-column row this replaced ran to 68 characters
  and wrapped on every phone. The row is now
  `SYM · TF · QD · GATE R · PF · N · STATUS`, and the columns still size to
  their widest CELL — the width is held by keeping the tokens short, never by
  clipping a symbol or a verdict, because a clipped table lies.
- **`PF` and `N` are Gate R's OWN quadrant numbers**, on the holdout, and the
  description says so under the table. An unlabelled profit factor under a
  regime-gated verdict is the one value on this card a reader must not have to
  guess at. **The blended IS/OOS pair moved to the promotion bullets**, where
  the collapse it exposes (`1.10`, down from `2.40`) changes a decision
  somebody is about to make; on the rows nobody is promoting they were two more
  numbers that decided nothing.
- **A `FAIL` says which bar it missed** — `FAIL·N` for a quadrant that starved,
  `FAIL·PF` for an edge that died, and `STARVED` / `REJECTED` in the STATUS
  column beside it. Both read identically as `FAIL` and are fixed by completely
  different work. **The PASS/FAIL token is still transcribed**; only the reason
  is derived, and only from the thresholds the handoff itself recorded in
  `certification_rule` (`gate_r_reason` prefers Stage 3's own
  `regime_starvation` record, and leaves the reason OFF rather than guessing
  when either number is missing).
- **The 999 profit-factor sentinel renders as `--`.** The profiler writes it
  when a quadrant never had a losing trade, so a starved quadrant with one
  winning holdout trade printed `999.00` beside a Gate R FAIL — the strongest
  number on the card, attached to the weakest row.
- **Seals are a 12-character prefix on the promotion bullets**, beside the
  parameters they seal, and the code hash and the parameter hash are both
  there: the same module under a different winning cell is a different strategy
  with the same code checksum. The 30-line dump of full 64-character digests
  that used to close this card is GONE — it was unreadable on a phone and
  verified by nobody from one, and a reader checking a seal has the promoted
  `meta.json` open. Only configurations that were actually STAGED carry one.
- **It re-scores nothing.** The card prints the `gate_regime` status and the
  `certified` flag Stage 3 recorded, so it can never announce a certification
  the audit refused.
- **The table lists CERTIFIED configurations ONLY, from 2026-08-21**
  (`STAGE3_CERTIFIED_ONLY`). STARVED, REJECTED, NOT CERT and NO AUDIT rows are
  off the code block entirely — the card is read to answer "what may be
  promoted", and that is the only row anybody acts on. Filtered on the STATUS
  cell rather than on the `certified` flag directly, so the table and the
  column cannot disagree about what the word means. **The exclusion is stated
  on the card and the counts beside it are NOT filtered**: `Configurations`,
  `Certified → Incubator` and `Audited` still describe the whole run, so a
  shorter table can never read as a shorter certification run, and every
  configuration's verdict stays on the handoff and in its own
  `gate_audit_<SYMBOL>_<TF>.json`. With nothing certified the block carries the
  header and `No certified configurations found.` rather than rendering empty —
  a holdout that certified nothing is a result to read, not a table that failed
  to draw. The `hidden` count is certified rows past `STAGE3_MAX_ROWS` and
  never rows the filter removed.
- **The table spans every timeframe the summary indexes**, now that Stage 3
  merges its per-timeframe invocations into one file. A card headed `15m` above
  a table carrying 5m rows described neither.
- **The promotion section is the one part of this card somebody ACTS on**, and
  it is budgeted ahead of everything optional for that reason. One bullet per
  certified configuration — target quadrant, the factor Gate R scored and the
  sample it scored on, the winning parameter plateau (abbreviated through the
  module's own collision-safe shortener), the blended IS→OOS pair behind it,
  and its two seal prefixes — and then **ONE command, in a field of its own**,
  that promotes all of them:
  `python3 backtest/run_pipeline.py --strat <STRAT> --promote-only`.
- **`--promote-only`, never `--auto-promote`.** The second re-runs Stages 1–4
  first, which OVERWRITES the handoff the card was built from: the winners it
  then promotes are a fresh sweep's, not the ones the reader is looking at, and
  nothing downstream could detect the substitution. `--promote-only` skips the
  four stages and runs Stage 5 alone over the certifications already on the
  handoff, iterating every certified row against that row's OWN
  `gate_audit_<SYMBOL>_<TF>.json` — which is exactly what the three-line
  per-pair `promote.py` blocks used to spell out by hand. Three certified
  configurations meant three of those blocks carrying four absolute filesystem
  paths each, which is what made this section unreadable on a phone.
- **The per-pair `promote.py` command survives for the one case that needs
  it**: a promotion that FAILED, where an operator is finishing a single pair
  and must cite that pair's own audit rather than the unsuffixed file, which
  holds whichever timeframe ran last. The unified command is its own field
  rather than the tail of the bullet block, because the field chunker splits a
  long block on a blank line and half a command is a command that runs and does
  something else; a field is never split.
- Headed `🏆 READY FOR PROMOTION / STAGED` until a promotion has happened and
  `🚀 AUTOMATICALLY PROMOTED TO INCUBATOR (Commit <hash>)` once
  `run_pipeline.py --auto-promote` (or `--promote-only`) has written its
  outcome back onto the handoff as `auto_promotion`. **The heading is that
  record's and is never inferred from a seal** — a sealed configuration was
  staged by Stage 3 and committed by nobody, and announcing it as promoted is
  how a strategy nobody promoted comes to be believed to be in the incubator.
  The bullets say `staged`, `promoted <commit>` or `NOT PROMOTED` per row, and
  the one-command footer counts only what is still OUTSTANDING, so it never
  tells a reader to re-run a promotion that already committed.

- **The Stage 2 card carries what the charter asks for and nothing derived**:
  the strategy, the in-sample window (with the holdout date it did not touch),
  and per configuration the symbol, timeframe, target regime quadrant, the
  selected best parameters, the in-sample profit factor and the max drawdown.
  Its counters say **Optimised → Stage 3**, never "promoted": Stage 2 promotes
  nothing and drops nothing, so a heading borrowed from the Stage 1 card would
  import a survival rate that does not exist, and the `Pruning` field states
  the guarantee outright. A configuration whose sweep FAILED is on the card as
  a row and in the `Failed to sweep` count — the one thing that can make the
  table shorter than the stage's input. `STAGE2_MAX_ROWS` is lower than Stage
  1's because each row carries a parameter set; what does not fit is COUNTED,
  as everywhere else here. The `Selection` line prints the rank Stage 2 says
  was APPLIED, which is not always the one requested.

- **It computes nothing and decides nothing.** The Stage 1 card prints the
  `status` Stage 1 recorded rather than re-applying the survival hurdle, so a
  card can never promote a configuration the stage dropped; the Stage 2 card
  re-ranks nothing, so it can never name a parameter set the sweep did not
  choose. A reporter that
  re-derived a profit factor would be free to disagree with the stage it is
  announcing, and the two would be compared by nobody.
- **The handoff is read through `pipeline.read_stage`**, so a file written by
  the wrong stage or belonging to another strategy is refused rather than
  posted. A Discord card is exactly the artifact nobody cross-checks.
- **A leaderboard is one fixed-width block in the embed DESCRIPTION**, not one
  field per row: Discord caps an embed at 25 fields and 6000 characters, and a
  full screen is 108 configurations. Rows past `STAGE1_MAX_ROWS` are COUNTED on
  the card — a silently shortened leaderboard reads as a complete one — while
  the Evaluated / Promoted / Dropped totals always describe the whole screen.
- **The QUAD column carries the `Q1`..`Q4` id the handoff recorded**, with a
  legend built FROM the rows. No short spelling of a regime name lives in this
  module: a second one would be free to disagree with `mdlib.regimes`, and a
  card naming the wrong environment is caught only in live trading.
- **The webhook URL is a credential** — never printed, never echoed into a
  failure message, only its host. With `--webhook` omitted it comes from
  `mdlib.env.discord_webhook`: the first of `$BT_DISCORD_WEBHOOK`,
  `$DISCORD_WEBHOOK_URL` and `$DISCORD_WEBHOOK` that carries a value, loaded
  from `.env` if it is not already exported. The success line names the
  VARIABLE that supplied it, never the URL.

## Commands

```bash
# Post Stage 1's leaderboard to $BT_DISCORD_WEBHOOK. Reads the handoff and
# recomputes nothing; --dry-run prints the payload and sends nothing.
python3 backtest/discord_reporter.py --stage 1 --strat X
# Stage 2's parameter-optimization card, read from stage2_summary.json.
python3 backtest/discord_reporter.py --stage 2 --strat X
# Stage 3's gate-audit & certification card, read from
# stage3_audit_summary.json: both windows, the target quadrant, Gate R, the
# IS/OOS profit factors and the SHA-256 seal. With no flag it takes that
# summary, and every gate_audit_<SYMBOL>_<TF>.json in the directory when there
# is none. --audit takes EITHER shape; --summary takes the index alone.
python3 backtest/discord_reporter.py --stage 3 --strat X
python3 backtest/discord_reporter.py --stage 3 --strat X \
    --audit /mnt/backtest/artifacts/pipeline/X/gate_audit_NQ_1h.json
python3 backtest/discord_reporter.py --stage 3 --strat X \
    --summary /mnt/backtest/artifacts/pipeline/X/stage3_audit_summary.json
# Stage 4's full-lifecycle card, read from the dual_metrics_<SYMBOL>.json
# snapshots in ONE run's artifacts directory: CAGR, net P&L, total trades, the
# friction share and the top regime's alpha score, per contract. It certifies
# nothing and carries no gate table. --artifacts defaults to the NEWEST
# verify_<stamp>/ and the card names whichever it read.
python3 backtest/discord_reporter.py --stage 4 --strat X
python3 backtest/discord_reporter.py --stage 4 --strat X \
    --artifacts /mnt/backtest/artifacts/pipeline/X/verify_20260824_101112
python3 backtest/discord_reporter.py --stage 1 --strat X \
    --survivors /mnt/backtest/artifacts/pipeline/X/surviving_assets.json
# Stage 5's promotion scorecard. With --strat alone every value is RESOLVED
# from what the promotion already wrote: the contract and timeframe, Gate R's
# out-of-sample profit factor, the holdout drawdown, the certified quadrant
# and the tear sheet path. Each one names the file it came from on the card.
python3 backtest/discord_reporter.py --stage 5 --strat X
# --audit-file and --metrics name those files explicitly (spelled the way
# promote.py spells them); --symbol/--tf/--pf/--dd/--regime/--report override
# any of the resolved values by hand.
python3 backtest/discord_reporter.py --stage 5 --strat X \
    --audit-file /mnt/backtest/artifacts/pipeline/X/gate_audit_NQ_15m.json
python3 backtest/discord_reporter.py --stage 5 --strat X --symbol NQ --tf 15m \
    --pf 1.42 --dd 8.30 --regime "Q1 · High Volatility / Trending"
# Stage 2 still HONOURS an exclude_days a handoff carries; nothing writes one.
python3 backtest/scan.py --strat X --tf 15m --ignore-stage1-exclude-days
python3 backtest/scan.py --strat X --tf 15m --exclude-days 3,4   # CLI wins
```
