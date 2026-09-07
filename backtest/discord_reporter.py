"""
backtest.discord_reporter - post a pipeline card to a Discord webhook.

Location:  ~/src/trading/backtest/discord_reporter.py

Five cards, one transport
-------------------------
- **`--mode promotion`** (the default, and `--stage 5`): the handful of numbers
  a promotion decision rests on. They may be passed in on the command line by
  whatever produced them, and with `--strat` alone they are RESOLVED from what
  the promotion already wrote - `approved_incubator/<strat>/meta.json`, the
  `dual_metrics.json` beside it, and the Stage 3 `gate_audit_<SYMBOL>_<TF>.json`
  the first cites. `--audit-file` and `--metrics` name those two files
  explicitly and are spelled the way `promote.py` spells them, so the command
  an operator already has from Stage 5 runs here rather than dying on an
  unrecognised argument and sending them back to retype four numbers by hand.

  Three rules make the resolution safe to trust, and each one is a way the
  card could otherwise mislead:

  **The out-of-sample profit factor is Gate R's or nothing.**
  `dual_metrics.json` and meta.json's snapshot both carry one, and both
  measured it over a window that CONTAINS the holdout - the field is headed
  `Out-of-Sample PF`, so filling it from either would print an in-sample
  number under an out-of-sample heading with every other field on the card
  still correct. Where no certification is readable the field keeps its
  `NOT REPORTED` token and the card says the number was DECLINED rather than
  absent. The drawdown beside it comes from the same holdout when the audit
  supplies it, and is labelled `NOT the holdout` when it falls back to the
  snapshot.

  **The contract and the timeframe resolve as a PAIR, from one file.**
  meta.json's top-level `symbols`/`timeframe` are the MODULE's declarations -
  every contract it targets, at the timeframe it prefers - while a promotion
  is one contract at one timeframe: `t3_braid_scalp_20260823` declares
  `NQ,ES,CL,GC` at 5m and was certified on NQ at 1h. Mixing the halves is how
  a card announces NQ at 5m for a run nobody made, with both halves
  individually true. The declarations are used only where the module names
  exactly ONE symbol and there is nothing to pick between.

  **Nothing is invented and every resolved value names its file.** An
  `Auto-resolved` field lists what came from where and over which window. A
  `--symbol` that disagrees with the certification is honoured and FLAGGED. A
  file named explicitly and missing RAISES; one this went looking for on its
  own is a note. Another strategy's meta.json, snapshot or audit is refused
  outright, the way `pipeline.read_stage` refuses another strategy's handoff.

  Two more fields sit beside those numbers, and neither is decoration:

  **The WIN RATE comes from the same sample as the profit factor above it** -
  Gate R's own quadrant, on the holdout - and falls back to the blended
  holdout and then to the run snapshot, saying on the card which of the three
  it read. 1.22 earned from a 53% hit rate and 1.22 earned from a 20% one are
  different strategies to sit in front of, and the ratio alone does not
  separate them. Unlike the profit factor a snapshot value is not DECLINED
  here: the field is headed `Win Rate` and claims no window of its own, while
  `Out-of-Sample PF` claims one. The unit is stated by the SOURCE, never
  sniffed from the magnitude - the profiler writes a percentage (53.75) and
  `summarize_result` writes a fraction (0.5233), and 0.52 and 52.0 are both
  plausible win rates, so a magnitude test cannot tell the two apart, it can
  only usually guess right.

  **PORTFOLIO MEMBERSHIP is read from `config/portfolios.json` and is never
  typed.** Being in `approved_incubator/` is a record that a version was
  CHOSEN and explicitly not permission to trade it; `active_strategies` on a
  portfolio is what grants that. So a promotion named by no portfolio reads
  `Incubator Staging (Evaluation / Shadow)` and one that is allocated reads
  `Active <portfolio> (Allocated)`, and the two are never the same green embed.
  A registry that cannot be READ reports `NOT RESOLVED` rather than the staging
  token, because "no portfolio names this strategy" is a claim about a file
  nobody managed to open. There is no fallback to a name, an asset or a parity
  rule - the same refusal `portfolio.config_loader` makes, since an allocation
  inferred that way is a live account chosen by a rule nobody wrote down.
- **`--mode baseline`** (equivalently `--stage 1`): Stage 1's REGIME FIREWALL
  leaderboard, read straight out of `surviving_assets.json` - per SURVIVING
  (symbol, timeframe) configuration both versions' blended profit factors, the
  quadrant it cleared, that quadrant's name, profit factor and trade count, and
  the version that carried it. **The table lists survivors only**
  (`STAGE1_SURVIVORS_ONLY`) - the card is read to answer "what goes to Stage
  2", and that is the only row anybody acts on. The exclusion is STATED on the
  card and the counts beside it are not filtered, so a shorter table can never
  read as a shorter screen; every drop keeps its row and the reason it fell
  short in the handoff and in `stage1_baseline_report.md`.

  The header carries the screen as the two bars a row is checked against -
  `Regime PF >= 1.00 · Regime Trades >= max(50, 10% placed)`, both
  TRANSCRIBED from the thresholds the run recorded, so a `--min-profit-factor`
  screen says what it actually applied. Stage 1's full `criterion` string (the
  net-P&L clause, the alpha score, the tie-break) stays on the handoff, which
  is where it is checked: 200 characters of mathematics above a two-number
  table is read by nobody, and what it displaces is the in-sample window.
- **`--mode scan`** (equivalently `--stage 2`): Stage 2's PARAMETER
  OPTIMIZATION summary, read straight out of `stage2_summary.json` - the
  in-sample window, and per configuration the symbol, timeframe, target regime
  quadrant, the selected best parameters, the in-sample profit factor and the
  max drawdown. Its parameter sets are printed TWICE and deliberately: once in
  the table with the keys abbreviated (`f=5 s=50 tp=1.5 sl=1.0`) so the
  fixed-width columns stay aligned, and once below it under
  `Optimized Parameters (Full)` with every key spelled as the strategy
  declared it and nothing clipped. The table is what a reader scans; the block
  is what they retype into `--param`, and a clipped parameter set is the one
  thing on this card that would be acted on while wrong.
- **`--mode audit`** (equivalently `--stage 3`): Stage 3's GATE AUDIT AND
  CERTIFICATION, read straight out of `stage3_audit_summary.json` - both
  windows, and per CERTIFIED configuration the symbol, timeframe, target
  regime quadrant, Gate R's verdict, and the quadrant profit factor and trade
  count it was measured on. **The table lists certified configurations only**
  (`STAGE3_CERTIFIED_ONLY`) - the card is read to answer "what may be
  promoted", and that is the only row anybody acts on. The exclusion is
  STATED on the card and the counts beside it are not filtered, so a shorter
  table can never read as a shorter certification run; every configuration's
  verdict stays on the handoff and in its own per-pair audit file, which is
  authoritative either way. **The table is built to a WIDTH** (45 characters,
  `STAGE3_TABLE_WIDTH`): Discord wraps a code block that overruns the
  viewport, and a wrapped fixed-width table is worse than none - every row
  becomes two, the second one unlabelled, and the columns a reader is
  comparing stop lining up under each other. The ten-column row this replaced
  ran to 68 characters and wrapped on every phone.

  So `PF` and `N` on the row are Gate R's OWN quadrant numbers and nothing
  else, and the description says which they are, because an unlabelled profit
  factor under a regime-gated verdict is the one value here a reader must not
  have to guess at. The blended IS/OOS pair moved to the certified rows in the
  promotion block, where the collapse it exposes ("1.10, down from 2.40")
  changes a decision somebody is about to make; on the other rows they were
  two more numbers that decided nothing. A `FAIL` carries WHY - `FAIL·N` for a
  quadrant that starved, `FAIL·PF` for an edge that died - because those read
  identically as `FAIL` and are fixed by completely different work. The
  PASS/FAIL token itself is still transcribed; only the reason is derived, and
  only from the thresholds the handoff recorded (see `gate_r_reason`). The 999
  profit-factor sentinel renders as `--`: a quadrant with one winning holdout
  trade has no measured factor, and 999.00 beside a FAIL reads as the
  strongest configuration on the card.

  Its table spans EVERY timeframe the summary indexes, because Stage 3 now
  merges its per-timeframe invocations into one file; a card headed `15m`
  above a table carrying 5m rows described neither. Below the table it carries
  the one section on this card somebody ACTS on: a bullet per certified
  configuration - target quadrant, the factor Gate R scored, the winning
  parameter plateau, the blended pair behind it, and 12-character prefixes of
  its code and parameter seals - and then ONE command, in a field of its own,
  that promotes all of them. `run_pipeline.py --promote-only`, never
  `--auto-promote`: the second re-runs Stages 1-4 first and OVERWRITES the
  handoff this card was built from, so the winners it then promotes are a
  fresh sweep's rather than the ones the reader is looking at. The per-pair
  `promote.py` command survives for the one case that needs it - a promotion
  that FAILED, where an operator finishes a single pair and must cite that
  pair's own `gate_audit_<SYMBOL>_<TF>.json` rather than the unsuffixed file,
  which holds whichever timeframe ran last. The full 64-character seals are
  not on the card: the 30-line dump they used to close it was unreadable on a
  phone and verified by nobody from one, and a reader checking a seal has the
  promoted `meta.json` open.

  Headed READY FOR PROMOTION / STAGED until a promotion has actually happened,
  and AUTOMATICALLY PROMOTED TO INCUBATOR with the commit once
  `run_pipeline.py --auto-promote` has written its outcome back onto the
  handoff. The heading is that RECORD's and is never inferred from a seal: a
  sealed configuration was staged by Stage 3 and committed by nobody, and
  announcing it as promoted is how a strategy nobody promoted comes to be
  believed to be in the incubator.
- **`--mode verify`** (equivalently `--stage 4`): Stage 4's FULL LIFECYCLE
  summary, read out of the `dual_metrics_<SYMBOL>.json` snapshots in the run's
  own artifacts directory (`--artifacts`, defaulting to the newest
  `verify_<stamp>/` and naming whichever it picked on the card). One row per
  contract: CAGR, net P&L, total trades, the friction share, and the top
  regime's alpha score.

  **It carries no gate table and no verdict, because Stage 4 produces
  neither.** That window CONTAINS the Stage 3 holdout, so every number on the
  card is in-sample by construction - the card says so above the table, in the
  colour it is drawn in (graphite, never the promotion green), and in a field
  that states outright that this stage certifies nothing. A lifecycle run read
  as a certification is the one mistake this card could cause on its own.

  Four of its five columns are transcribed straight from the snapshot. The
  fifth, FRIC, is a division of two figures the run recorded - total costs
  over GROSS profit - and it inherits `verify_full.cost_drag`'s rule exactly:
  undefined, printed `--`, where gross P&L was not positive, because a
  strategy that lost money gross has no profit for its costs to be a share of
  and `0%` there reads as a run that cost nothing. It is on the card because
  it is the number that decides whether an edge is real: an edge handing 85%
  of its gross to the broker dies on one extra tick of slippage while every
  ratio above it still reads fine.

  ALPHA is the designated quadrant's `net P&L x profit factor`, transcribed
  from the UNSUFFIXED `regime_profile_<SYM>_<TF>.json` Stage 4 wrote for the
  same run. The suffixed `_version_a` files beside it are STAGE 1's, profiled
  over the charter window alone; reading one as a fallback would put an
  in-sample score under a lifecycle heading with every column still lining up.
  A missing profile and a run that designated no home regime both print `--`
  and are COUNTED separately under the table, because they are fixed by
  different work.

  Rows are ordered by contract and NOT ranked. Stage 4 selects nothing, and
  sorting by CAGR would give a leaderboard's shape to a stage that produced no
  leaderboard. Nothing is summed across contracts either: symbols are never
  blended here, and a total net P&L over a run of independent simulations is a
  portfolio number no backtest in this repository produced.

Five cards now, and the reason the count keeps growing is that each one
announces a DIFFERENT decision. A Stage 2 card is not a promotion and not a
screen: every configuration on it advanced, because Stage 2 prunes nothing, and
the card says so rather than letting a reader infer a survival rate from a
leaderboard's length. A Stage 3 card is the first that carries a PASS - and
its `Certified` count is Gate R's, not a roll-up of the three advisory gates
printed beside it.

It reads no bars, opens no lake file, and computes nothing. The Stage 1 card
reads a handoff, which is not the same thing: every number on it is one Stage 1
wrote, transcribed. That is deliberate - **LLMs never do math and neither does
a notifier**. A reporter that recomputed a profit factor would be free to
disagree with the stage it is announcing, and the two would be compared by
nobody. In particular the card does not re-apply the survival hurdle: it prints
the `status` Stage 1 recorded, so a card can never promote a configuration the
stage dropped.

What it will not do
-------------------
- **It does not decide anything.** Nothing here checks a gate or a screening
  hurdle, so a card can be posted for a strategy that was never certified. The
  card announces what it was told; `backtest/promote.py` is what refuses an
  uncertified version and `backtest/baseline.py` is what decides PROMOTED from
  DROPPED. The Stage 2 card in particular re-ranks nothing: `backtest/scan.py`
  chose the winning parameter set off the Sharpe plateau, and this transcribes
  the row it wrote. The Stage 3 card re-scores nothing: it prints the
  `gate_regime` status and the `certified` flag `backtest/audit_gates.py`
  recorded, so it can never announce a certification the audit refused.
- **It does not invent a missing number.** `--pf` and `--dd` are taken as text,
  not floats. A numeric value is formatted (`1.42`, `-8.30 %`) and anything
  else - `NOT EVALUATED`, `n/a` - is printed verbatim. Coercing those to 0.0
  would put a zero drawdown on a card for a run whose drawdown nobody measured,
  which is the one failure mode a status notifier can cause on its own. The
  Stage 5 auto-resolution obeys the same rule: a value no file carries stays
  `NOT REPORTED`, and the card names the files it looked in - which is a
  different statement from a zero, and points at different work.
- **It does not raise on a transport failure.** A dead webhook must not take
  down whatever called it; the outcome is printed and returned in the exit
  code. Same reasoning as `live/dispatcher.send_execution_signal`.

The webhook URL is a credential
-------------------------------
It is never printed, never echoed into a log line, and never included in the
failure message - only its host is. Anyone holding the full URL can post to the
channel. It does not have to sit in shell history: with `--webhook` omitted it
comes from the first of `$BT_DISCORD_WEBHOOK`, `$DISCORD_WEBHOOK_URL` and
`$DISCORD_WEBHOOK` that carries a value, loaded from `~/src/trading/.env` if it
is not already in the environment. That chain is `mdlib.env.discord_webhook`
and is shared with `scripts/incubator_tracker.py`, because two resolvers would
be free to disagree about which variable configures Discord - and the symptom
of a disagreement is a card that is simply never posted, which is
indistinguishable from a quiet pipeline.

Discord specifics that are easy to get wrong
--------------------------------------------
- A successful webhook POST returns **204 No Content**, not 200. Treating only
  200 as success reports every successful post as a failure.
- `color` is a decimal integer, not a CSS string. Emerald green is 0x2ECC71.
- Embed field values cap at 1024 characters and the whole embed at 6000; a
  local artifact path is well inside that, but the value is truncated rather
  than sent to be rejected with a 400 nobody reads.
- Only an http(s) URL renders as a link. A `/mnt/backtest/...` path is not
  clickable in any client, so it is rendered as inline code instead of as a
  markdown link that would look broken.
- An embed **description** caps at 4096 characters and an embed carries at most
  25 fields. A full screen is 27 contracts x 4 timeframes = 108 rows, which
  fits in neither as fields; the Stage 1 leaderboard is therefore one
  fixed-width block in the description, truncated to `STAGE1_MAX_ROWS` with the
  count of what was left off printed on the card. A silently shortened
  leaderboard reads as a complete one, which is the same failure the HTML trade
  log's row cap exists to avoid.
"""

from __future__ import annotations

# --- .env bootstrap --------------------------------------------------------
# Load ~/src/trading/.env before ANYTHING reads os.environ. This runs at import
# time, above the local imports below, because several modules resolve their
# BT_* variables while being imported (backtest.run's ARTIFACTS_ROOT) - loading
# the file inside main() would be too late for those and would work here, which
# is the kind of difference nobody notices until one runner silently uses the
# default path. The rules - the repository root derived from __file__ rather
# than the working directory, existing variables winning over the file, the
# CrossTrade credentials withheld from os.environ - live in ONE module rather
# than in a block copied into every runner: see mdlib/env.py.
import sys                                                         # noqa: E402
from pathlib import Path                                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # `python3 backtest/x.py` puts backtest/ on sys.path, not the repository
    # root, so mdlib is not importable until this runs.
    sys.path.insert(0, str(PROJECT_ROOT))
from mdlib.env import (                                            # noqa: E402
    DISCORD_WEBHOOK_VARS, WEBHOOK_HINT, describe_webhook, load_env,
)

load_env()
# ---------------------------------------------------------------------------


import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.pipeline import (GATE_AUDIT_FILE,                    # noqa: E402
                               STAGE2_SUMMARY_FILE, STAGE3_SUMMARY_FILE,
                               STAGE_NAMES, SURVIVORS_FILE,
                               base_strategy, pipeline_dir, read_stage)
# The one place `strategies/approved_incubator/` is spelled out is
# `backtest/promote.py`, which creates it. A second copy of that path here
# would be free to point somewhere else after a move, and the symptom is a
# promotion card that silently resolves nothing.
# `sha256` comes from there too: the pair-audit adapter below records the
# digest of the audit it transcribed, exactly as Stage 3's own summary does,
# and a second implementation of a hash is a second thing that can disagree.
from backtest.promote import INCUBATOR, sha256                    # noqa: E402

# Emerald green. Discord wants a decimal int; 0x2ECC71 == 3066993.
EMERALD_GREEN = 0x2ECC71
# The Stage 1 card is a different colour from the promotion card on purpose: a
# screening leaderboard and a promotion are read very differently, and in a
# channel that carries both, colour is the only thing distinguishing them at a
# glance. Slate blue - deliberately not green, because nothing on the Stage 1
# card is an approval to trade.
SLATE_BLUE = 0x3498DB
# Nothing survived. Amber rather than red: an empty screen is a result about
# the idea on these contracts, not an error.
AMBER = 0xE67E22

# Discord's documented limits. Exceeding any of them is a 400.
MAX_FIELD_VALUE = 1024
MAX_EMBED_TOTAL = 6000
MAX_EMBED_DESCRIPTION = 4096

# Leaderboard rows that fit the description block with room for the header, the
# legend and the truncation note. A full screen is 108 configurations; what is
# left off is COUNTED on the card and the handoff path is printed beside it.
STAGE1_MAX_ROWS = 40

# The Stage 1 table lists SURVIVING configurations ONLY (2026-08-24), the same
# rule the Stage 3 card already applies to its certifications. The card is read
# to answer "what goes to Stage 2", and a survivor is the only row anybody acts
# on; three dropped ES rows above a promoted one is a reader scanning a STATUS
# column on a phone for the rows that matter. What is excluded is NOT hidden:
# `Evaluated`, `Promoted -> Stage 2` and `Dropped` still count the whole screen,
# the description says the table is filtered, and every dropped configuration
# keeps its row - with the reason it fell short - in `surviving_assets.json`
# and in `stage1_baseline_report.md`, which are authoritative either way.
STAGE1_SURVIVORS_ONLY = True

# What the code block says when nothing survived. A header over an empty block
# reads as a table that failed to render; the line states the RESULT, which is
# what a screen that promoted nothing is.
STAGE1_NO_ROWS_NOTE = "No surviving configurations found."

# The Stage 2 card carries a parameter set per row, which is far wider than a
# Stage 1 row, so fewer of them fit the 4096-character description. Whatever
# does not fit is COUNTED on the card, exactly as it is on the Stage 1 card:
# a silently shortened optimisation summary reads as the whole stage, and
# Stage 2 optimises EVERY survivor, so its table's length is the one number a
# reader is entitled to trust.
STAGE2_MAX_ROWS = 20

# Parameter sets get long once risk axes are swept, and a fixed-width table
# column that reflows is harder to read than a clipped one. So the TABLE cell
# carries the parameter set with its keys abbreviated (`f=5 s=50 tp=1.5`) and
# is clipped past this many characters - and every row's UNABBREVIATED,
# UNCLIPPED parameter set is printed below the table in its own field. The
# table is for scanning; that block is the copy a reader retypes.
STAGE2_MAX_PARAM_CHARS = 46
ELLIPSIS = "\u2026"

# The full-parameter block below the table. One field per chunk, each inside
# Discord's 1024-character field cap, and at most this many chunks - the embed
# already carries six other fields and Discord caps an embed at 25 fields and
# 6000 characters.
STAGE2_PARAM_FIELD_NAME = "Optimized Parameters (Full)"
STAGE2_PARAM_MAX_FIELDS = 6
FENCE_OPEN = "```text\n"
FENCE_CLOSE = "\n```"
# Characters held back from the embed budget for the "N further ..." note, so
# a block that had to leave rows out can always say so.
NOTE_RESERVE = 96
# Continuation indent for a parameter set too wide for one field.
INDENT_WIDTH = 4

# The line under the table that says its cells are abbreviated, in its two
# forms: the full sets are below, or - when the embed had no room for them -
# they are in the handoff. The second is deliberately the SHORTER string, so
# swapping it in after the budget has been measured can only shrink the embed.
ABBREV_NOTE = ("_Table parameter keys are abbreviated \u2014 the full "
               "key=value sets are below._")
ABBREV_NOTE_NO_BLOCK = ("_Table parameter keys are abbreviated \u2014 the "
                        "full sets are in the handoff._")

# Tokens that say what KIND of parameter something is rather than which one it
# is. Every period is a period and every stop multiple is quoted in ATRs, so
# inside one parameter set they distinguish nothing while costing most of the
# column width. Dropped only to build the TABLE's abbreviation; the full block
# prints the key exactly as the strategy declared it.
PARAM_NOISE_TOKENS = frozenset({
    "window", "windows", "period", "periods", "length", "len", "lookback",
    "mult", "multiple", "multiplier", "factor", "atr", "bars", "num",
})

# Stage 2's own colour, distinct from the Stage 1 slate and the promotion
# green. Violet, and deliberately not green: an optimised parameter set is not
# an approval to trade, and in a channel carrying all three cards the colour is
# what separates them at a glance.
VIOLET = 0x9B59B6

# Stage 2 optimised the configuration; the sweep raised and it did not.
OPTIMIZED = "OPTIMIZED"
ERRORED = "ERROR"

# Stage 3's own colour. Teal, and once again deliberately not the promotion
# green: a certification is a verdict about a holdout, not a decision to trade.
# The four cards in a channel are slate (screen), violet (sweep), teal
# (certification) and green (promotion), which is the only thing separating
# them at a glance in a scrollback.
TEAL = 0x1ABC9C

# A Stage 3 row carries a quadrant, Gate R's verdict and the factor and sample
# it was measured on. Whatever does not fit is COUNTED on the card, as
# everywhere else here.
STAGE3_MAX_ROWS = 24

# The width the Stage 3 table is built to. Discord wraps a code block that
# overruns the viewport, and a wrapped fixed-width table is worse than no
# table: every row becomes two, the second one unlabelled, and the columns a
# reader is comparing stop lining up. 45 characters clears the narrowest phone
# viewport this card is read on. It is a design TARGET rather than a hard cut -
# the columns still size to their widest cell, because clipping a symbol or a
# verdict to hit a number is how a table starts lying - so the way to keep it
# is to keep the tokens in it short.
STAGE3_TABLE_WIDTH = 45

# The Stage 3 table lists CERTIFIED configurations ONLY (2026-08-21). Everything
# else - STARVED, REJECTED, NOT CERT, NO AUDIT - is off the table entirely.
# The card is read to answer "what may be promoted", and a certification is the
# only row anybody acts on; five failing rows above three passing ones is the
# reader scanning a STATUS column on a phone for the three that matter. What is
# excluded is NOT hidden: `Configurations`, `Certified -> Incubator` and
# `Audited` still count the whole run, the description says the table is
# filtered, and the per-pair `gate_audit_<SYMBOL>_<TF>.json` files remain the
# authoritative verdict for every configuration audited.
STAGE3_CERTIFIED_ONLY = True

# What the code block says when nothing certified. A header over an empty block
# reads as a table that failed to render; the line states the RESULT, which is
# what a holdout that certified nothing is.
STAGE3_NO_ROWS_NOTE = "No certified configurations found."

# Minimum column widths, so the header and the empty-table note keep the shape
# a row has. Columns still size to their widest CELL above these - nothing is
# ever clipped to hit a number. The sum with its separators is 43, inside
# STAGE3_TABLE_WIDTH.
STAGE3_MIN_WIDTHS = (3, 3, 2, 7, 4, 3, 9)

# The profiler writes 999 as a profit factor when a quadrant never had a losing
# trade (backtest/profiler.py). It is a SENTINEL, not a measured factor, and it
# is exactly the number a STARVED quadrant prints: one winning holdout trade
# shows 999.00 next to a Gate R that failed on the sample count, which reads as
# the strongest configuration on the card. It renders as `--`.
REGIME_PF_SENTINEL = 999.0

# The long verdict tokens, shortened for the table and nowhere else. Both keep
# their NOT: a gate that was not evaluated and a run that was not audited are
# statements about what has not happened yet, and "EVAL"/"AUDIT" alone would
# read as the opposite.
GATE_R_TOKENS = {"NOT EVALUATED": "NO EVAL", "NOT AUDITED": "NO AUDIT"}

# How much of each SHA-256 goes on the card. Twelve hex characters is 48 bits -
# enough to tell two builds of the same strategy apart at a glance, which is
# what a reader uses it for. The full 64 live in the promoted `meta.json` and
# on the handoff, and the card says so: a truncated hash presented as THE hash
# is a checksum nobody can verify. The 30-line dump of full digests that used
# to close this card was unreadable on a phone and verified by nobody from
# one - a reader checking a seal has the file open.
SEAL_PREFIX_CHARS = 12

# The promotion block. It is the one part of this card somebody ACTS on, so it
# is budgeted BEFORE the seals: a reader who cannot find the promote command
# goes looking through the handoff, while a reader missing a seal has lost a
# checksum they were not going to verify from a chat client anyway.
STAGE3_PROMO_MAX_ROWS = 10
STAGE3_PROMO_MAX_FIELDS = 4

# The two headings the section takes, and they are not interchangeable. The
# first says a human still has to run something; the second says a commit
# already happened. Labelling a staged-but-uncommitted configuration as
# promoted is how a strategy nobody promoted ends up believed to be in the
# incubator.
PROMO_READY_TITLE = "\U0001F3C6 READY FOR PROMOTION / STAGED"
PROMO_DONE_TITLE = "\U0001F680 AUTOMATICALLY PROMOTED TO INCUBATOR"

# The one command that promotes the certified set, in its own field. Its own
# field and not the tail of the promotion block, because the chunker splits a
# long block across fields and half a command is a command that runs and does
# something else. A field is never split.
PROMO_COMMAND_FIELD = "\u25B6 Promote all certified · one command"

# The tokens Stage 3 records per configuration.
CERTIFIED = "PASS"
NOT_AUDITED = "NOT AUDITED"
# A gate the run never reached. NOT a pass, and deliberately a different token
# from NOT AUDITED: "the bootstrap was not run" and "the audit raised" are
# fixed by different work, and one token for both hides which happened.
NOT_EVALUATED = "NOT EVALUATED"
# What Gate R is called inside a `gate_audit_<SYMBOL>_<TF>.json`'s `gates`
# block, and what the summary row that transcribes it is called. One spelling,
# because the adapter reads the first and writes the second.
GATE_R_KEY = "gate_regime"

# The Stage 1 handoff, and the two words it records per configuration.
PROMOTED = "PROMOTED"
DROPPED = "DROPPED"

# --------------------------------------------------------------------------
# Stage 4 · the full-lifecycle card
# --------------------------------------------------------------------------

# Stage 4's own colour. Graphite, and deliberately neither the promotion green
# nor Stage 3's teal: this stage certifies NOTHING. Its window contains the
# holdout Stage 3 already spent, so every number on the card is in-sample by
# construction, and a card in a certification colour is exactly how a
# lifecycle run comes to be read as a verdict. The five cards in a channel are
# slate (screen), violet (sweep), teal (certification), graphite (lifecycle)
# and green (promotion).
GRAPHITE = 0x607D8B

# A Stage 4 row carries five metrics for one contract. Whatever does not fit is
# COUNTED on the card, as everywhere else here.
STAGE4_MAX_ROWS = 24

# The width the Stage 4 table is built to, for the reason the Stage 3 one is:
# Discord wraps a code block that overruns the viewport, and a wrapped
# fixed-width table is worse than no table. It is a design TARGET, not a clip -
# the columns still size to their widest CELL - so the way to hold it is to
# keep the TOKENS short, which is what `_fmt_money` is for (`123.5k`, not
# `123,456.78`).
STAGE4_TABLE_WIDTH = 56

# What Stage 4 writes per contract, and what this card reads. The metrics
# snapshot beside the tear sheets - `backtest/report_html.write_dual_reports` -
# rather than the `verify_<SYMBOL>.json` handoff, because the snapshot is the
# file a promotion cites and it lives in the run's OWN timestamped directory:
# a handoff at a stable path is rewritten by the next lifecycle run, and a card
# announcing one run's window over another run's numbers is the substitution
# nothing downstream could detect.
DUAL_METRICS_GLOB = "dual_metrics_*.json"
DUAL_METRICS_PREFIX = "dual_metrics_"

# The regime profile Stage 4 writes for the SAME run, and the only file this
# card takes a quadrant from. It is UNSUFFIXED because `verify_full.py`
# constructs `RegimeProfiler` with no `version`, and the suffixed
# `regime_profile_<SYM>_<TF>_version_<a|b>.json` files sitting beside it are
# STAGE 1's - profiled over the charter window alone. Reading one of those as a
# fallback would print an in-sample-window alpha score under a lifecycle
# heading, with every column still lining up. A missing profile is reported as
# missing.
REGIME_PROFILE_FILE = "regime_profile_{symbol}_{tf}.json"

# Stage 4's own words about what it is not, on the card rather than in a
# footnote. The stage prints this on the console and records
# `is_certification: false` in its JSON; a card that dropped it would be the
# one place these metrics appear with no such statement attached.
STAGE4_NOT_CERTIFICATION = (
    "**Not a certification** — this window CONTAINS the Stage 3 holdout, so "
    "every number below is in-sample by construction. The certified verdicts "
    "are Gate R's, in `gate_audit_<SYMBOL>_<TF>.json`.")

# The metrics on this card are VERSION A's. Stage 4 profiles Version A, drags
# its costs and writes its trade log, so pairing those with Version B's
# headline numbers would describe two different runs in one row.
STAGE4_VERSION_NOTE = "Version A · rule-based"

# The profiler's designation, when it did not make one. `primary` is None when
# no quadrant cleared the designation bars, which is a FINDING - the strategy
# has no home environment on these bars - and must not render as the same `--`
# a missing profile gets. Counted separately under the table.
NO_HOME_REGIME = "None"

# A webhook POST succeeds with 204 No Content. With ?wait=true it is 200 and the
# body is the created message, so both are accepted.
SUCCESS_STATUS = frozenset({200, 204})

POST_TIMEOUT_SECONDS = 10.0

# The webhook variable names, their precedence and the hint printed when none
# is set live in `mdlib/env.py` (DISCORD_WEBHOOK_VARS / WEBHOOK_HINT). The
# single name this module used to hardcode is gone rather than kept as an
# alias: a second spelling of the chain here would be free to fall out of step
# with the one `scripts/incubator_tracker.py` resolves on.


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def _fmt_number(raw: str, suffix: str = "", decimals: int = 2) -> str:
    """
    Format a numeric argument, or pass a non-numeric token through untouched.

    `NOT EVALUATED` and `1.42` are different statements about a backtest and
    must not collapse into one. Anything float() rejects is returned verbatim.
    """
    text = (raw or "").strip()
    if not text:
        return "NOT REPORTED"
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text
    return f"{value:.{decimals}f}{suffix}"


def _fmt_report(raw: str) -> str:
    """
    Render the artifact reference. An http(s) URL becomes a markdown link; a
    filesystem path becomes inline code, because a path is not clickable and a
    link that never opens reads as a broken report rather than a local file.
    """
    text = (raw or "").strip()
    if not text:
        return "NOT RECORDED"
    scheme = urlparse(text).scheme.lower()
    if scheme in ("http", "https"):
        rendered = f"[Open report]({text})"
    else:
        rendered = f"`{text}`"
    if len(rendered) > MAX_FIELD_VALUE:
        rendered = rendered[: MAX_FIELD_VALUE - 3] + "..."
    return rendered


# --------------------------------------------------------------------------
# Stage 5 · resolving the card from what the promotion already wrote
# --------------------------------------------------------------------------

# `backtest/promote.py` writes both of these into
# `strategies/approved_incubator/<strat>/`, and between them - plus the Stage 3
# certification `meta.json` cites - they already hold every value this card
# asks for on the command line. Reading them is not a convenience: the
# alternative is an operator copying a profit factor out of one file and a
# drawdown out of another into a card nobody cross-checks afterwards, and a
# transcription slip there is invisible in exactly the way a promotion
# announcement must not be.
PROMOTED_META_FILE = "meta.json"
PROMOTED_METRICS_FILE = "dual_metrics.json"

# `gate_audit_<SYMBOL>_<TF>.json`. The TIMEFRAME is in the filename and
# nowhere in meta.json's certification block, while meta.json's own
# `timeframe` is the MODULE's declared one - `t3_braid_scalp_20260823`
# declares 5m and was certified at 1h. Reading the declaration as the
# certified timeframe would head the card with a run nobody made.
GATE_AUDIT_NAME = re.compile(
    r"^gate_audit_(?P<symbol>[^_]+)_(?P<tf>[^_]+)\.json$", re.IGNORECASE)

# What each auto-resolved field is called on the card's provenance list. The
# labels are the card's own field names, so a reader can see at a glance which
# line above them a file supplied.
PROMOTION_FIELD_LABELS = {
    "symbol": "Asset",
    "tf": "Timeframe",
    "pf": "Out-of-Sample PF",
    "win": "Win Rate",
    "dd": "Max Drawdown",
    "regime": "Certified Regime Firewall",
    "membership": "Portfolio Membership",
    "report": "Artifacts / Report",
}

# The scalar fields, in the order the card carries them. `resolve_promotion_
# fields` walks this rather than a literal repeated at each of the three places
# it needs one, because a field added to the resolver and forgotten in the
# `missing` list is a value that silently stops being reported as absent.
PROMOTION_SCALARS = ("pf", "win", "dd", "regime", "report")

# A win rate is stored two ways in this repository and the difference is a
# factor of 100 that no reader would catch on a card: `backtest.profiler`
# writes a PERCENTAGE (53.75) into every regime breakdown and therefore into
# Gate R's `measured` block, while `backtest.report.summarize_result` writes a
# FRACTION (0.5233) into every metrics dict. Each source below states its own
# unit rather than the value being sniffed at from its magnitude - 0.52 and
# 52.0 are both perfectly plausible win rates, so a magnitude test cannot tell
# a fraction from a percentage, it can only usually guess right.
WIN_RATE_IS_PCT = "the profiler writes win_rate as a percentage"
WIN_RATE_IS_FRACTION = "summarize_result writes win_rate as a fraction"

# The one number on this card that may ONLY come from a Stage 3 gate audit.
# `dual_metrics.json` and the `metrics` block in `meta.json` both carry a
# profit factor, and both are measured over the whole run - which for a
# lifecycle snapshot CONTAINS the holdout. Filling "Out-of-Sample PF" from
# either would print an in-sample number under an out-of-sample heading with
# every other field on the card still correct. Gate R's is the only profit
# factor here that was measured out of sample, inside the one quadrant Stage 1
# designated, and it is the only one this module will resolve.
PF_IS_GATE_R_ONLY = (
    "profit factor is Gate R's or nothing: the one in dual_metrics.json is "
    "measured over the whole run, which contains the holdout")


def promoted_dir(strat: str, incubator: str | Path | None = None) -> Path:
    """`strategies/approved_incubator/<strat>/` - the directory Stage 5 wrote."""
    root = Path(incubator) if incubator else INCUBATOR
    return root / strat


def promotion_packages(strat: str,
                       incubator: str | Path | None = None) -> list[Path]:
    """
    The promoted package(s) `--strat` names, or [] when nothing was promoted.

    `--strat` may name a package OUTRIGHT (`..._NQ_15m_VA`, which is what
    `promote.py` passes when it posts its own card) or name the MODULE the
    pipeline swept (`..._20260901`, which is what `run_pipeline.discord_cmd`
    passes for every stage). Stage 5 splits a module into ONE PACKAGE PER
    CERTIFIED PAIR, so a module name is not a directory: `promoted_dir` finds
    nothing, every field resolves empty, and the card then refuses for a
    missing `--symbol` - which sends an operator looking for a contract when
    what is actually missing is the pair half of the id they meant to type.

    A package that names itself wins outright and no scan happens, so this
    cannot reinterpret an id that already resolves.

    Discovered by SCANNING for a `meta.json` rather than by rebuilding
    `promote.strategy_id`'s spelling here. `promote.py` imports this module to
    post its Stage 5 card, so an import back would be circular - and a second
    copy of the naming rule would be free to drift from the one that actually
    wrote the directories, which is how a card comes to look for a package
    that is on disk under a slightly different name.
    """
    root = Path(incubator) if incubator else INCUBATOR
    own = root / strat
    if (own / PROMOTED_META_FILE).exists():
        return [own]
    try:
        kids = sorted(root.glob(f"{strat}_*"))
    except OSError:
        return []
    return [d for d in kids
            if d.is_dir() and (d / PROMOTED_META_FILE).exists()]


def _read_promotion_json(path: Path, label: str) -> dict[str, Any]:
    """Read one of the promotion's files, or say which one could not be read."""
    path = Path(path)
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"{label} not readable: {path} ({exc})") from exc
    except ValueError as exc:
        raise ValueError(f"{label} is not valid JSON: {path} ({exc})") from exc
    if not isinstance(blob, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return blob


def _owns(recorded: str, wanted: str) -> bool:
    """
    Whether an artifact recording `recorded` belongs to the promotion `wanted`.

    Three spellings are accepted, and each one exists because two names for the
    same thing differ by construction here:

      * the same name.
      * the literal `strat` - a promoted module is
        `approved_incubator/<id>/strat.py`, so it records itself as `strat`.
      * the BASE strategy of a per-pair id. `--strat` now names one certified
        pair (`t3_braid_scalp_20260823_NQ_1h`) while the `dual_metrics_NQ.json`
        and the `gate_audit_NQ_1h.json` that promotion cites were written by
        the pipeline under the MODULE's name. Without this the Stage 5 card
        would refuse the very snapshot the promotion it is announcing points
        at.

    `base_strategy` splits only on a trailing known timeframe token, so
    `foo_bar` does not pass as the base of an unrelated `foo`.
    """
    recorded, wanted = str(recorded).strip(), str(wanted).strip()
    if recorded.lower() in (wanted.lower(), "strat"):
        return True
    return base_strategy(wanted).lower() == recorded.lower()


def _refuse_other_strategy(recorded: str, strat: str | None, path: Path,
                           what: str) -> None:
    """
    Refuse another strategy's promotion artifact.

    The same refusal `pipeline.read_stage` makes about a handoff and
    `_check_strategy` makes about a lifecycle snapshot, for the same reason and
    with more at stake: this card announces a PROMOTION, and one strategy's
    certified profit factor posted under another's name is a claim nobody
    downstream can contradict. `strat` is the name the operator typed and a
    promoted module records itself as `strat` (it is
    `approved_incubator/<strat>/strat.py`), so that one spelling is accepted.
    """
    recorded = str(recorded or "").strip()
    wanted = str(strat or "").strip()
    if not recorded or not wanted:
        return
    if _owns(recorded, wanted):
        return
    raise ValueError(
        f"{path.name} is {what} strategy {recorded!r}, not {wanted!r}. "
        f"Posting it under --strat {wanted} would announce one strategy's "
        f"promotion under another's name.")


def load_promoted_meta(path: str | Path,
                       strat: str | None = None) -> dict[str, Any]:
    """Read `approved_incubator/<strat>/meta.json`, and refuse another's."""
    path = Path(path)
    blob = _read_promotion_json(path, PROMOTED_META_FILE)
    _refuse_other_strategy(blob.get("name"), strat, path, "the promotion of")
    return blob


def load_promotion_metrics(path: str | Path,
                           strat: str | None = None) -> dict[str, Any]:
    """Read the `dual_metrics.json` a promotion locked, and refuse another's."""
    path = Path(path)
    blob = _read_promotion_json(path, PROMOTED_METRICS_FILE)
    _check_strategy(blob, path, strat)
    return blob


def load_promotion_audit(path: str | Path,
                         strat: str | None = None) -> dict[str, Any]:
    """
    Read one Stage 3 `gate_audit_<SYMBOL>_<TF>.json` - the certification.

    A file written by another stage is refused the way `promote.py` refuses it:
    only `backtest/audit_gates.py` produces a verdict a promotion may rest on,
    and the audit inside a `dual_metrics.json` can evaluate Gate 1 alone.
    """
    path = Path(path)
    blob = _read_promotion_json(path, "gate audit")
    stage = blob.get("stage")
    if stage is not None and str(stage) != "3":
        raise ValueError(
            f"{path.name} was written by stage {stage}, not stage 3. Only the "
            f"certification stage (backtest/audit_gates.py) produces the gate "
            f"verdict a promotion rests on.")
    _refuse_other_strategy(blob.get("strategy"), strat, path, "the audit of")
    return blob


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out                       # NaN is not a value


def audit_promotion_values(blob: dict[str, Any],
                           version: str = "A") -> dict[str, Any]:
    """
    The certified claim, transcribed out of Stage 3's own audit.

    Contract, timeframe, Gate R's HOLDOUT profit factor and the quadrant it was
    measured in, and the holdout max drawdown. Nothing is recomputed and no
    gate is re-scored - the same rule the Stage 3 card is held to.

    The version is not guessed at. An audit carrying only Version A while the
    promotion recorded B is reported as a gap rather than filled from the block
    that happens to be there: two versions' numbers under one heading is the
    substitution this card could not survive.
    """
    out: dict[str, Any] = {"notes": []}
    symbol = str(blob.get("symbol") or "").strip()
    tf = str(blob.get("timeframe") or "").strip()
    if symbol and tf:
        out["pair"] = (symbol, tf)

    want = str(version or "A").upper()
    block = (blob.get("versions") or {}).get(want)
    if not isinstance(block, dict):
        have = ", ".join(sorted(blob.get("versions") or {})) or "no version"
        out["notes"].append(
            f"the gate audit carries {have}, not Version {want} - its Gate R "
            f"numbers were left off rather than read off the wrong version")
        return out

    gate_r = ((block.get("gate_audit") or {}).get("gates") or {}).get("gate_regime") or {}
    regime = str(gate_r.get("target_regime") or blob.get("target_regime") or "").strip()
    quadrant = str(gate_r.get("quadrant") or blob.get("target_quadrant") or "").strip()
    where = " · ".join(t for t in (quadrant, regime) if t)

    measured = gate_r.get("measured") or {}
    pf = _as_float(measured.get("profit_factor"))
    trades = measured.get("trade_count")
    if pf is not None:
        if pf >= REGIME_PF_SENTINEL:
            # The profiler's sentinel: the quadrant never had a losing trade,
            # so there is no measured factor. 999.00 on a promotion card is
            # the strongest number anybody will ever read here, attached to a
            # quadrant that may hold one trade.
            out["pf"] = "NOT MEASURED"
            out["pf_basis"] = (f"Gate R · holdout · {where} · no losing trade "
                               f"in the quadrant" if where else
                               "Gate R · holdout · no losing trade in the quadrant")
        else:
            count = f" · {int(trades):,} trades" if _as_float(trades) is not None else ""
            out["pf"] = f"{pf:.2f}"
            out["pf_basis"] = f"Gate R · holdout{f' · {where}' if where else ''}{count}"

    # The win rate from the SAME sample as the profit factor above it: Gate R's
    # own quadrant, on the holdout. `measured` is written by the profiler and
    # is already a percentage (`WIN_RATE_IS_PCT`). It survives the 999 sentinel
    # branch deliberately - a quadrant with no losing trade has no measurable
    # profit factor and still has a perfectly real win rate.
    win = _as_float(measured.get("win_rate"))
    if win is not None:
        count = f" · {int(trades):,} trades" if _as_float(trades) is not None else ""
        out["win"] = f"{win:.2f}"
        out["win_basis"] = f"Gate R · holdout{f' · {where}' if where else ''}{count}"

    if where:
        out["regime"] = where
        out["regime_basis"] = "Stage 3's certification target"

    # The drawdown from the SAME window as the profit factor above it. Pairing
    # Gate R's holdout factor with a full-run drawdown would put two windows on
    # one card with nothing saying so.
    holdout = block.get("metrics_holdout") or {}
    dd = _as_float(holdout.get("max_drawdown_pct"))
    if dd is not None:
        out["dd"] = f"{abs(dd):.2f}"
        out["dd_basis"] = "holdout · blended across quadrants"

    # Only where Gate R recorded none. Still the holdout, so it is out of
    # sample - but blended across all four quadrants rather than scored in the
    # one that was certified, and the basis says which of the two a reader is
    # looking at. A fraction here (`WIN_RATE_IS_FRACTION`), unlike Gate R's.
    win = _as_float(holdout.get("win_rate"))
    if "win" not in out and win is not None:
        out["win"] = f"{win * 100:.2f}"
        out["win_basis"] = "holdout · blended across quadrants"
    return out


def metrics_promotion_values(blob: dict[str, Any],
                             version: str = "A") -> dict[str, Any]:
    """
    What the locked `dual_metrics.json` supplies: the contract, the timeframe,
    the run's max drawdown and the tear sheet.

    It supplies NO profit factor. Its window is the whole run, which for a
    lifecycle snapshot contains the Stage 3 holdout; the card's field is headed
    `Out-of-Sample PF` and that number is Gate R's alone (`PF_IS_GATE_R_ONLY`).
    """
    out: dict[str, Any] = {"notes": []}
    meta = blob.get("meta") or {}
    symbol = str(meta.get("symbol") or "").strip()
    tf = str(meta.get("timeframe") or "").strip()
    if symbol and tf:
        out["pair"] = (symbol, tf)

    key = "version_b" if str(version or "A").upper() == "B" else "version_a"
    block = blob.get(key)
    if not isinstance(block, dict):
        out["notes"].append(
            f"the metrics snapshot carries no {key} block - Version "
            f"{str(version).upper()} was not run in it")
        return out

    window = " → ".join(str(meta.get(k) or "")[:10] for k in ("start", "end")).strip(" →")
    if _as_float((block.get("metrics") or {}).get("profit_factor")) is not None:
        # Seen and DECLINED, and the card says so. Silence would read as "no
        # profit factor was recorded anywhere", which is a different fact.
        out["pf_declined"] = True
    dd = _as_float((block.get("metrics") or {}).get("max_drawdown_pct"))
    if dd is not None:
        out["dd"] = f"{abs(dd):.2f}"
        out["dd_basis"] = (f"whole run {window} · NOT the holdout" if window
                           else "whole run · NOT the holdout")

    # A fraction (`WIN_RATE_IS_FRACTION`), over the whole run. Unlike the
    # profit factor this is not DECLINED here: the card's field is headed
    # `Win Rate` and claims no window of its own, and the basis beside it says
    # which window produced it. A profit factor's field does claim one.
    win = _as_float((block.get("metrics") or {}).get("win_rate"))
    if win is not None:
        out["win"] = f"{win * 100:.2f}"
        out["win_basis"] = (f"whole run {window} · NOT the holdout" if window
                            else "whole run · NOT the holdout")

    report = (blob.get("reports") or {}).get(key)
    if report:
        out["report"] = str(report)
        out["report_basis"] = f"Version {str(version).upper()} tear sheet"
    return out


def meta_promotion_values(blob: dict[str, Any]) -> dict[str, Any]:
    """
    What `meta.json` supplies on its own, once the files it points at are gone.

    The contract comes from the CERTIFICATION block, never from the top-level
    `symbols`/`timeframe`: those are the MODULE's declarations - every symbol
    it targets and the timeframe it prefers - and a promotion is one contract
    at one timeframe. `t3_braid_scalp_20260823` declares `NQ,ES,CL,GC` at 5m
    and was certified on NQ at 1h. The declarations are used only when the
    module names exactly ONE symbol, where there is nothing to pick between,
    and the card says where they came from either way.
    """
    out: dict[str, Any] = {"notes": []}
    cert = blob.get("certification")
    cert = cert if isinstance(cert, dict) else {}

    symbol = str(cert.get("audit_symbol") or "").strip()
    audit_file = str(cert.get("audit_file") or "").strip()
    match = GATE_AUDIT_NAME.match(Path(audit_file).name) if audit_file else None
    if symbol and match and match.group("symbol").upper() == symbol.upper():
        out["pair"] = (symbol, match.group("tf"))
        out["pair_basis"] = f"certified on {Path(audit_file).name}"
    elif symbol and audit_file:
        out["notes"].append(
            f"meta.json certifies {symbol} but {Path(audit_file).name} names "
            f"no timeframe this can read - the pair was left unresolved rather "
            f"than paired with the module's declared timeframe")

    if _as_float((blob.get("metrics") or {}).get("profit_factor")) is not None:
        out["pf_declined"] = True
    dd = _as_float((blob.get("metrics") or {}).get("max_drawdown_pct"))
    if dd is not None:
        out["dd"] = f"{abs(dd):.2f}"
        out["dd_basis"] = "meta.json metrics snapshot · NOT the holdout"

    win = _as_float((blob.get("metrics") or {}).get("win_rate"))
    if win is not None:                          # a fraction, like the snapshot
        out["win"] = f"{win * 100:.2f}"
        out["win_basis"] = "meta.json metrics snapshot · NOT the holdout"

    declared = blob.get("symbols")
    declared = [str(s).strip() for s in declared] if isinstance(declared, list) else []
    tf = str(blob.get("timeframe") or "").strip()
    if len(declared) == 1 and declared[0] and tf:
        out["declared_pair"] = (declared[0], tf)
        out["declared_pair_basis"] = "the module's own SYMBOLS/TIMEFRAME"
    return out


# --------------------------------------------------------------------------
# Stage 5 · where the strategy actually sits
# --------------------------------------------------------------------------

# The live routing table. Read here as plain JSON rather than through
# `portfolio.config_loader`, for two reasons that are both about this staying a
# NOTIFIER. The dependency runs one way - `portfolio/` sits above `backtest/`
# and reads from it, and nothing in `backtest/` may import from there - and
# `get_portfolio_for_strategy` RAISES for a strategy no portfolio names, which
# is the ordinary state of a freshly staged promotion and the exact state this
# field exists to report. Loading the config through it would also reconcile
# every asset against `backtest/specs.py`, so an unrelated multiplier
# disagreement would stop a promotion card from being posted at all.
PORTFOLIO_CONFIG_FILE = PROJECT_ROOT / "config" / "portfolios.json"

# Staged under `approved_incubator/` and named by no portfolio.
# `approved_incubator/<strat>/` is a record that a version was CHOSEN and is
# explicitly not permission to trade it; `active_strategies` in the routing
# table is what grants that. The two states have to read differently on a card
# somebody acts on, because "certified and waiting" and "live on an account"
# are the same green embed otherwise.
INCUBATOR_STAGING = "Incubator Staging (Evaluation / Shadow)"
ALLOCATED_TEMPLATE = "Active {names} (Allocated)"

# Named by no portfolio and with no promotion on disk either. Distinct from
# the staging token: one says the promotion is waiting for an allocation, the
# other says there is no promotion here at all, and they are fixed by
# completely different work.
NOT_STAGED = "NOT STAGED"

# The registry could not be read. Deliberately NOT the staging token: "no
# portfolio names this strategy" is a fact about the config, and printing it
# when the config could not be opened is an allocation claim nobody checked.
MEMBERSHIP_UNRESOLVED = "NOT RESOLVED"


def portfolio_membership(strat: str, *, staged: bool = False,
                         config_path: str | Path | None = None
                         ) -> dict[str, Any]:
    """
    Where this strategy sits: allocated to a portfolio, or staged and waiting.

    Allocation is `active_strategies` on a portfolio in `config/portfolios.json`
    and nothing else. There is no fallback to the strategy's name, its assets
    or a parity rule - the same refusal `portfolio.config_loader` makes, for
    the same reason: an allocation inferred from a name is a live account
    chosen by a rule nobody wrote down.

    A strategy named on an incubator AND a prop portfolio is normal - that is
    what the two tracks are for - and both are named. Two portfolios of the
    SAME track is the config `portfolio.config_loader` refuses to load, and it
    is flagged as a note rather than resolved to one of them here.
    """
    path = Path(config_path) if config_path else PORTFOLIO_CONFIG_FILE
    out: dict[str, Any] = {"membership": MEMBERSHIP_UNRESOLVED, "basis": "",
                           "source": path.name, "notes": []}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        out["basis"] = f"{path.name} could not be read"
        out["notes"].append(f"the routing table {path} is not readable ({exc}) "
                            f"- the allocation was left unresolved rather than "
                            f"reported as none")
        return out
    except ValueError as exc:
        out["basis"] = f"{path.name} is not valid JSON"
        out["notes"].append(f"the routing table {path} is not valid JSON "
                            f"({exc}) - the allocation was left unresolved")
        return out

    portfolios = blob.get("portfolios")
    if not isinstance(portfolios, dict):
        out["basis"] = f"{path.name} carries no portfolios object"
        out["notes"].append(f"{path.name} carries no `portfolios` object - the "
                            f"allocation was left unresolved")
        return out

    # Matched case-insensitively on the stripped id: a strategy id differing
    # from the promoted directory's only in case is the same strategy, and
    # missing it would print the staging token for a strategy that is live.
    wanted = str(strat or "").strip().lower()
    holders: list[tuple[str, str]] = []
    for pid, portfolio in sorted(portfolios.items()):
        if not isinstance(portfolio, dict):
            continue
        active = portfolio.get("active_strategies")
        active = active if isinstance(active, list) else []
        if any(str(name).strip().lower() == wanted for name in active):
            holders.append((str(portfolio.get("portfolio_id") or pid),
                            str(portfolio.get("account_type") or "")))

    if holders:
        out["membership"] = ALLOCATED_TEMPLATE.format(
            names=" + ".join(pid for pid, _ in holders))
        out["basis"] = (f"named in active_strategies on "
                        f"{len(holders)} portfolio(s) in {path.name}")
        tracks: dict[str, list[str]] = {}
        for pid, account_type in holders:
            tracks.setdefault(account_type, []).append(pid)
        for account_type, pids in sorted(tracks.items()):
            if len(pids) > 1:
                out["notes"].append(
                    f"{strat} is named on {len(pids)} {account_type or 'same-track'} "
                    f"portfolios ({', '.join(pids)}); portfolio.config_loader "
                    f"refuses that config - both would size the same signal "
                    f"independently and the net position would be double")
        return out

    if staged:
        out["membership"] = INCUBATOR_STAGING
        out["basis"] = (f"staged under approved_incubator/ and named by no "
                        f"portfolio in {path.name}")
        return out

    out["membership"] = NOT_STAGED
    out["basis"] = (f"no promotion under approved_incubator/ and no portfolio "
                    f"in {path.name} names it")
    return out


def resolve_promotion_fields(
        strat: str,
        *,
        symbol: str = "",
        tf: str = "",
        pf: str = "",
        win: str = "",
        dd: str = "",
        regime: str = "",
        report: str = "",
        audit_file: str | Path | None = None,
        metrics_file: str | Path | None = None,
        incubator: str | Path | None = None,
        portfolio_config: str | Path | None = None) -> dict[str, Any]:
    """
    Fill the promotion card from what Stage 3 and Stage 5 already wrote.

    Precedence, strongest evidence first:

      1. the command line - an operator correcting the record outranks a file,
         exactly as `--params` outranks a handoff in `promote.py`;
      2. the Stage 3 certification (`--audit-file`, else the one `meta.json`
         cites) - the ONLY source of an out-of-sample profit factor and of the
         certified quadrant;
      3. the locked metrics snapshot (`--metrics`, else the promotion's own
         `dual_metrics.json`) - the drawdown and the tear sheet;
      4. `meta.json` itself, for the contract and a snapshot drawdown.

    The contract and the timeframe are resolved as a PAIR, from one source.
    They are two halves of one statement, and taking the symbol from a
    certification while taking the timeframe from a module declaration is how a
    card comes to announce NQ at 5m for a run certified on NQ at 1h - with
    every field on it individually true.

    Nothing is invented. A field no source carries is returned empty, the card
    prints its existing NOT REPORTED token, and the provenance list says where
    this looked - which is a different statement from a value of zero. A file
    named explicitly and missing RAISES; one this went looking for on its own
    is a note.
    """
    given = {"symbol": symbol, "tf": tf, "pf": pf, "win": win, "dd": dd,
             "regime": regime, "report": report}
    values = {k: str(v).strip() for k, v in given.items() if str(v or "").strip()}
    sources = {k: "--" + ("tf" if k == "tf" else k) for k in values}
    resolved: list[tuple[str, str, str]] = []
    notes: list[str] = []
    inspected: list[str] = []

    # One package, or the several a module name covers. `strat_id` is what the
    # rest of this resolution reads itself as: when a MODULE name resolved to a
    # single package, the ownership checks and the routing-table lookup must
    # both use the PACKAGE's id, because that is the name `promote.py` wrote
    # into meta.json and into `active_strategies`. Reading those under the
    # module name returns NOT STAGED for a strategy that is in fact routed.
    packages = promotion_packages(strat, incubator)
    home = promoted_dir(strat, incubator)
    strat_id = strat
    if len(packages) == 1 and packages[0].name != strat:
        home = packages[0]
        strat_id = home.name
        notes.append(f"'{strat}' is a module name; it has exactly one "
                     f"promoted package, {strat_id}, and this card describes "
                     f"that one")
    meta_path = home / PROMOTED_META_FILE
    meta: dict[str, Any] | None = None
    if meta_path.exists():
        meta = load_promoted_meta(meta_path, strat_id)
        inspected.append(meta_path.name)

    # Which twin was promoted, and therefore which block to read in both files.
    # Defaulted to A rather than guessed at from a file's contents: A is what
    # `promote.py` writes when nothing says otherwise.
    version = str((meta or {}).get("version") or "A").upper()

    # --- the certification -------------------------------------------------
    audit_path = Path(audit_file) if audit_file else None
    if audit_path is not None and not audit_path.exists():
        raise FileNotFoundError(f"--audit-file not found: {audit_path}")
    if audit_path is None and meta is not None:
        cited = str((meta.get("certification") or {}).get("audit_file") or "").strip()
        if cited:
            cited_path = Path(cited)
            if cited_path.exists():
                audit_path = cited_path
            else:
                notes.append(f"meta.json cites {cited_path.name}, which is not "
                             f"on disk - Gate R's numbers were not read")

    # --- the metrics snapshot ----------------------------------------------
    metrics_path = Path(metrics_file) if metrics_file else None
    if metrics_path is not None and not metrics_path.exists():
        raise FileNotFoundError(f"--metrics not found: {metrics_path}")
    if metrics_path is None:
        candidate = home / PROMOTED_METRICS_FILE
        if candidate.exists():
            metrics_path = candidate

    candidates: list[tuple[str, dict[str, Any]]] = []
    if audit_path is not None:
        blob = load_promotion_audit(audit_path, strat_id)
        inspected.append(audit_path.name)
        candidates.append((audit_path.name, audit_promotion_values(blob, version)))
    if metrics_path is not None:
        blob = load_promotion_metrics(metrics_path, strat_id)
        inspected.append(metrics_path.name)
        candidates.append((metrics_path.name,
                           metrics_promotion_values(blob, version)))
    if meta is not None:
        candidates.append((meta_path.name, meta_promotion_values(meta)))

    for name, cand in candidates:
        for note in cand.get("notes") or []:
            notes.append(f"{name}: {note}")

    # --- the scalar fields -------------------------------------------------
    for name, cand in candidates:
        for field in PROMOTION_SCALARS:
            if field in values or not cand.get(field):
                continue
            values[field] = str(cand[field])
            sources[field] = name
            resolved.append((PROMOTION_FIELD_LABELS[field],
                             str(cand.get(f"{field}_basis") or ""), name))

    # --- the contract, as a pair -------------------------------------------
    for key in ("pair", "declared_pair"):
        if values.get("symbol") and values.get("tf"):
            break
        for name, cand in candidates:
            pair = cand.get(key)
            if not pair:
                continue
            found_symbol, found_tf = (str(pair[0]).strip(), str(pair[1]).strip())
            basis = str(cand.get(f"{key}_basis") or "")
            if not values.get("symbol"):
                values["symbol"] = found_symbol
                sources["symbol"] = name
                resolved.append((PROMOTION_FIELD_LABELS["symbol"], basis, name))
            elif values["symbol"].upper() != found_symbol.upper():
                notes.append(f"--symbol {values['symbol']} names a different "
                             f"contract from {name}'s {found_symbol}")
            if not values.get("tf"):
                values["tf"] = found_tf
                sources["tf"] = name
                resolved.append((PROMOTION_FIELD_LABELS["tf"], basis, name))
            break

    # Said only when a file that WAS read carries a profit factor this
    # declined. Where nothing carried one, "not found" is the whole story and
    # this note would describe a decision nobody had to make.
    if "pf" not in values and any(c.get("pf_declined") for _, c in candidates):
        notes.append(PF_IS_GATE_R_ONLY)

    missing = [PROMOTION_FIELD_LABELS[f]
               for f in PROMOTION_SCALARS if f not in values]

    # Where the strategy sits. Resolved from the routing table and NEVER from
    # the command line: every other field on this card is a measurement an
    # operator may correct, while this one is a statement about which live
    # account holds the strategy right now, and a card is not the place that
    # gets decided.
    member = portfolio_membership(strat_id, staged=meta is not None,
                                  config_path=portfolio_config)
    sources["membership"] = member["source"]
    inspected.append(member["source"])
    resolved.append((PROMOTION_FIELD_LABELS["membership"],
                     member["basis"], member["source"]))
    notes.extend(member["notes"])

    return {
        "symbol": values.get("symbol", ""),
        "tf": values.get("tf", ""),
        "pf": values.get("pf", ""),
        "win": values.get("win", ""),
        "dd": values.get("dd", ""),
        "regime": values.get("regime", ""),
        "report": values.get("report", ""),
        "membership": member["membership"],
        "version": version,
        "sources": sources,
        "resolved": resolved,
        "missing": missing,
        "inspected": inspected,
        "notes": notes,
        "home": str(home),
        "strat_id": strat_id,
        "packages": [p.name for p in packages],
    }


def format_resolution(resolution: dict[str, Any]) -> str:
    """
    The provenance list, as one field value.

    Every auto-resolved value names the FILE it came from and the window it was
    measured over. A promotion card is read once and acted on, and "1.22"
    resolved out of a gate audit and "1.22" typed by an operator are the same
    six characters - the difference is whether anybody can check it later.
    """
    lines: list[str] = []
    for label, basis, where in resolution.get("resolved") or []:
        lines.append(f"`{label}` ← `{where}`" + (f" · {basis}" if basis else ""))
    for label in resolution.get("missing") or []:
        looked = ", ".join(f"`{n}`" for n in resolution.get("inspected") or [])
        lines.append(f"`{label}` · not found in {looked or 'any promoted artifact'}")
    for note in resolution.get("notes") or []:
        lines.append(f"⚠ {note}")
    value = "\n".join(lines)
    if len(value) > MAX_FIELD_VALUE:
        value = value[: MAX_FIELD_VALUE - 3] + "..."
    return value


def build_embed(
    strat: str,
    symbol: str,
    tf: str,
    pf: str,
    dd: str,
    regime: str,
    report: str,
    resolution: dict[str, Any] | None = None,
    win: str = "",
    membership: str = "",
) -> dict[str, Any]:
    """
    Build the Discord embed dict. Pure - sends nothing, reads nothing.

    `resolution` is what `resolve_promotion_fields` returned, when anything on
    the card was filled from a file rather than typed. It adds ONE field
    naming the file and the window behind each auto-resolved value, and is
    omitted entirely when every value came from the command line - a card an
    operator typed in full is unchanged, to the byte, by this parameter
    existing. (Through the CLI the provenance field is now always present,
    because `Portfolio Membership` can only ever come from the routing table.)

    `win` and `membership` are keyword arguments AFTER `resolution` rather than
    beside the metrics they are printed with: every existing caller passes the
    first seven positionally, and inserting a parameter in the middle would
    silently shift a drawdown into the profit factor's slot on any call this
    module does not own.
    """
    embed = {
        "title": f"\U0001F680 Incubation Promotion: {strat}",
        "color": EMERALD_GREEN,
        "fields": [
            {
                "name": "Asset / Timeframe",
                "value": f"**{symbol}** · `{tf}`",
                "inline": True,
            },
            {
                "name": "Out-of-Sample PF",
                "value": _fmt_number(pf),
                "inline": True,
            },
            # A fourth inline field wraps onto its own row in Discord, which
            # renders three per row. That is the intended layout: the win rate
            # belongs with the two numbers it qualifies, and a profit factor
            # read without one is a ratio with no sense of how it was earned -
            # 1.22 from a 53% hit rate and 1.22 from a 20% one are different
            # strategies to sit in front of.
            {
                "name": "Win Rate",
                "value": _fmt_number(win, suffix=" %"),
                "inline": True,
            },
            {
                "name": "Max Drawdown",
                "value": _fmt_number(dd, suffix=" %"),
                "inline": True,
            },
            # Above the certified quadrant on purpose: the quadrant is where
            # the strategy is PERMITTED to trade, and this is whether anything
            # is routing it there yet. Read the other way round, a certified
            # firewall reads as a live one.
            {
                "name": "Portfolio Membership",
                "value": (membership or "").strip() or MEMBERSHIP_UNRESOLVED,
                "inline": False,
            },
            {
                "name": "Certified Regime Firewall",
                "value": (regime or "").strip() or "NOT DECLARED",
                "inline": False,
            },
            {
                "name": "Artifacts / Report",
                "value": _fmt_report(report),
                "inline": False,
            },
        ],
        "footer": {"text": "backtest/discord_reporter.py · values as supplied, not recomputed"},
    }
    # Only when something was actually resolved from a file. A promotion card
    # whose values were all typed carries no provenance list, because there is
    # no provenance to state beyond the footer it already has.
    if resolution and resolution.get("resolved"):
        embed["fields"].append({
            "name": "Auto-resolved",
            "value": format_resolution(resolution),
            "inline": False,
        })
    return embed




# --------------------------------------------------------------------------
# Stage 1 · the regime-firewall leaderboard
# --------------------------------------------------------------------------

def _fmt_metric(value: Any, decimals: int = 2) -> str:
    """
    One leaderboard cell. A float is formatted; a token is passed through; an
    absent value renders as `--`.

    `--` rather than `0.00`: a dropped configuration has no winning quadrant,
    so it has no quadrant profit factor, and a zero in that column reads as a
    quadrant that was measured and found worthless. Those are different
    findings and the table has to keep them apart.
    """
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_count(value: Any) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def default_survivors_path(strat: str, out_dir: str | None = None) -> Path:
    """`<BT_ARTIFACTS>/pipeline/<strategy>/surviving_assets.json`."""
    return pipeline_dir(strat, out_dir) / SURVIVORS_FILE


def load_stage1(path: str | Path, strat: str | None = None) -> dict[str, Any]:
    """
    Read Stage 1's handoff, and refuse the wrong one.

    Delegated to `pipeline.read_stage` rather than parsed here, because that is
    where "this file was written by stage 3, not stage 1" and "this belongs to
    another strategy" are already refusals. A notifier with its own laxer
    reader would happily post one strategy's screen under another's name, and
    a Discord card is exactly the artifact nobody cross-checks.
    """
    return read_stage(Path(path), 1, strat)


def stage1_rows(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Every configuration the screen EVALUATED, in one shape.

    `screen_results` is what Stage 1 writes for this purpose and is used when
    present. A handoff written before that field existed is reassembled from
    `surviving_pairs` + `dropped`, and the reassembly is deliberately lossy in
    a visible way: those older dropped entries carry no version and no
    quadrant, so the cells render `--` rather than being back-filled with a
    plausible value. A leaderboard that silently invented a version column is
    worse than one that says it does not have it.
    """
    results = blob.get("screen_results")
    if isinstance(results, list) and results:
        return [dict(r) for r in results if isinstance(r, dict)]

    rows: list[dict[str, Any]] = []
    for pair in blob.get("surviving_pairs") or []:
        if not isinstance(pair, dict):
            continue
        rows.append({**pair, "tf": pair.get("tf"),
                     "status": pair.get("status") or PROMOTED})
    for drop in blob.get("dropped") or []:
        if not isinstance(drop, dict):
            continue
        rows.append({"symbol": drop.get("symbol"),
                     # Stage 1's `dropped` list keys the timeframe as
                     # `timeframe`; `surviving_pairs` keys it as `tf`. Both are
                     # read, neither is renamed in the handoff, because the two
                     # lists are read by different consumers.
                     "tf": drop.get("tf") or drop.get("timeframe"),
                     "status": drop.get("status") or DROPPED,
                     "optimal_regime": drop.get("optimal_regime"),
                     "reason": drop.get("reason")})
    return rows


def _sort_key(row: dict[str, Any]) -> tuple:
    """
    Promoted first, by the quadrant profit factor the screen decided on,
    descending; then the drops, alphabetically.

    Sorted on the REGIME profit factor rather than on either version's blended
    one, for the same reason Stage 1's own leaderboard is: the blend is not
    what advanced the configuration, and ranking on it would put a broad
    mediocre edge above a sharp one in a single environment.
    """
    promoted = str(row.get("status") or "").upper() == PROMOTED
    try:
        pf = float(row.get("regime_pf"))
    except (TypeError, ValueError):
        pf = float("-inf")
    return (0 if promoted else 1, -pf if promoted else 0.0,
            str(row.get("symbol") or ""), str(row.get("tf") or ""))


def _stage1_trade_floor(blob: dict[str, Any]) -> str:
    """
    The trade bar as the screen recorded it: `max(50, 10% placed)`.

    Both halves are READ from the handoff (`min_trades`, `min_trade_fraction`)
    rather than spelled out as literals, because both move: `--min-trades` and
    `--min-trade-fraction` raise them per run, and a header that kept printing
    the defaults would describe a screen nobody ran while every row under it
    stayed correct. A handoff carrying only one half prints that half alone,
    and one carrying neither says so - `not recorded` is a fact about the file,
    which is a different statement from a bar of zero.
    """
    floor = blob.get("min_trades")
    fraction = blob.get("min_trade_fraction")
    parts: list[str] = []
    try:
        parts.append(f"{int(floor):,}")
    except (TypeError, ValueError):
        pass
    try:
        parts.append(f"{float(fraction) * 100:g}% placed")
    except (TypeError, ValueError):
        pass
    if not parts:
        return "not recorded"
    return parts[0] if len(parts) == 1 else f"max({parts[0]}, {parts[1]})"


def format_stage1_table(rows: list[dict[str, Any]],
                        max_rows: int = STAGE1_MAX_ROWS
                        ) -> tuple[str, int, dict[str, str]]:
    """
    The survivors leaderboard as one fixed-width block, plus the quadrant
    legend.

    Returns `(text, hidden, legend)`. `hidden` is how many SURVIVING rows did
    not fit `max_rows` and is printed on the card by the caller - a leaderboard
    truncated in silence reads as the whole screen.

    **Only SURVIVING configurations are listed** (`STAGE1_SURVIVORS_ONLY`).
    DROPPED rows are off the table entirely: the card is read to answer "what
    goes to Stage 2", and that is the only row anybody acts on. The exclusion
    is stated on the card and the counts beside it still describe the WHOLE
    screen - `Evaluated`, `Promoted -> Stage 2` and `Dropped` are unfiltered -
    so a shorter table can never read as a shorter screen. Every dropped
    configuration keeps its row, and the reason it fell short, in
    `surviving_assets.json` and in `stage1_baseline_report.md`. Filtered on the
    STATUS the handoff RECORDED, never on a hurdle re-applied here: a notifier
    that re-derived survival would be free to promote a configuration Stage 1
    dropped.

    The row carries both versions' BLENDED profit factors beside the quadrant
    numbers the screen actually decided on, because they answer different
    questions and are read together: `PF (A)` / `PF (B)` say what the
    configuration did across every market state, and `REGIME PF` / `TRD` say
    what it did inside the one quadrant it is being promoted for. A survivor
    whose blend is 0.83 and whose quadrant is 1.10 is the ordinary shape of a
    regime-gated edge, and printing only one of the two hides which.

    `OPTIMAL REGIME` spells the designated quadrant's name in full, so the
    `QUAD` id beside it needs no legend under the table. The name is
    TRANSCRIBED from the handoff, never abbreviated here: a second spelling of
    "High Volatility / Trending" in this module would be free to disagree with
    the one `mdlib.regimes` numbers, and a card naming the wrong environment is
    the kind of error that is only ever caught in live trading.
    """
    header = ["SYM", "TF", "PF (A)", "PF (B)", "QUAD", "OPTIMAL REGIME",
              "REGIME PF", "TRD", "VER"]
    body: list[list[str]] = []
    legend: dict[str, str] = {}

    ordered = sorted((r for r in rows
                      if not STAGE1_SURVIVORS_ONLY
                      or str(r.get("status") or "").upper() == PROMOTED),
                     key=_sort_key)
    shown = ordered[: max(0, int(max_rows))]
    for row in shown:
        quad = row.get("quadrant")
        regime = row.get("optimal_regime")
        if quad and regime:
            legend[str(quad)] = str(regime)
        body.append([
            str(row.get("symbol") or "?"),
            str(row.get("tf") or "?"),
            # The BLENDED factors, as the screen recorded them. `--` where the
            # handoff carries none - an older one carries neither, and a 0.00
            # there would read as a version that ran and made nothing.
            _fmt_metric(row.get("profit_factor_a")),
            _fmt_metric(row.get("profit_factor_b")),
            str(quad) if quad else "--",
            str(regime) if regime else "--",
            _fmt_metric(row.get("regime_pf")),
            _fmt_count(row.get("regime_trade_count")),
            # `V` + the version letter, or `--`. A blank cell here would read
            # as Version A, which is a claim about which twin carried the
            # configuration.
            f"V{row['version']}" if row.get("version") else "--",
        ])

    widths = [max(len(header[i]), *(len(r[i]) for r in body)) if body
              else len(header[i]) for i in range(len(header))]
    align = ["<", "<", ">", ">", "<", "<", ">", ">", "<"]

    def line(cells: list[str]) -> str:
        return "  ".join(format(c, f"{align[i]}{widths[i]}")
                         for i, c in enumerate(cells)).rstrip()

    out = [line(header), line(["-" * w for w in widths])]
    # Nothing surviving is a RESULT, not an empty render. The header stays
    # above it so the block is recognisable as the same table.
    if body:
        out.extend(line(r) for r in body)
    else:
        out.append(STAGE1_NO_ROWS_NOTE)
    return "\n".join(out), len(ordered) - len(shown), legend


def build_stage1_embed(strat: str, blob: dict[str, Any],
                       source: str | Path | None = None,
                       max_rows: int = STAGE1_MAX_ROWS) -> dict[str, Any]:
    """
    Stage 1's card. Pure - sends nothing, and every number on it is read off
    the handoff rather than derived from it.

    NO LEADERBOARD TABLE. It used to carry a monospace table of the surviving
    configurations, which is a wide fixed-width block inside a container that
    reflows: on a narrow client every row wrapped and the columns stopped
    lining up, so the one thing the table existed to give - a scannable
    alignment - was the first thing lost. The card now answers "did the screen
    run, and what came out of it" in five numbers, and points at the two files
    that answer everything else.

    THE PER-PAIR DETAIL IS NOT ON THE CARD ANY MORE. `stage1_survivors.csv`
    carries every configuration with its designated quadrant and both versions'
    metrics; the card names it rather than reproducing a truncated version of
    it. A card that showed the first N rows had to also say how many it hid,
    which is a truncated leaderboard reading as a complete one.

    `max_rows` is accepted and IGNORED. It is still a live flag for Stage 2's
    card, and one shared `--max-rows` that silently did nothing here is better
    than a CLI that rejects a flag it used to take.
    """
    rows = stage1_rows(blob)
    promoted = [r for r in rows
                if str(r.get("status") or "").upper() == PROMOTED]

    window = blob.get("in_sample_window") or {}
    start = window.get("start") or blob.get("start") or "lake start"
    end = window.get("end") or blob.get("end") or "lake end"
    timeframes = blob.get("timeframes") or (
        [blob["timeframe"]] if blob.get("timeframe") else [])
    ml = blob.get("ml_evaluated")

    description = "\n".join([
        f"**Strategy** `{strat}`",
        f"**Window** `{start} \u2192 {end}`",
        # KEPT, though the table it qualified is gone. The counts below are
        # meaningless without the bar they were counted against, and both
        # numbers are TRANSCRIBED from the thresholds the screen recorded,
        # never restated as literals: a card that hardcoded 1.00 would keep
        # saying 1.00 after a `--min-profit-factor` run.
        f"**Screen** Regime PF `>= "
        f"{_fmt_metric(blob.get('min_profit_factor'))}` \u00b7 Regime Trades "
        f"`>= {_stage1_trade_floor(blob)}`",
        "",
        "_Detailed metrics and quadrant breakdowns saved to "
        "`stage1_survivors.csv` and `stage1_baseline_report.md`._",
    ])

    fields = [
        {"name": "Total Evaluated", "value": str(len(rows)), "inline": True},
        {"name": "Promoted \u2192 Stage 2", "value": str(len(promoted)),
         "inline": True},
        {"name": "Dropped", "value": str(len(rows) - len(promoted)),
         "inline": True},
        {"name": "Timeframes Screened",
         "value": ", ".join(f"`{t}`" for t in timeframes) or "not recorded",
         "inline": True},
        # Three states, never two. `--no-ml` means survival was decided on
        # Version A alone, which is a narrower claim than one both versions
        # were given a chance at - and an older handoff that recorded nothing
        # must not be reported as either.
        {"name": "Version B",
         "value": ("evaluated" if ml is True
                   else "NOT RUN" if ml is False else "not recorded"),
         "inline": True},
        {"name": "Handoff", "value": _fmt_report(str(source or "")),
         "inline": False},
    ]

    return {
        "title": "\U0001F50D Stage 1 Complete \u00b7 Baseline Screening",
        "description": description,
        # Green when something survived, amber when nothing did. Amber rather
        # than red because an empty screen is a RESULT - the idea does not work
        # on these contracts - and colouring it like a crash invites a re-run
        # with different parameters until something passes.
        "color": EMERALD_GREEN if promoted else AMBER,
        "fields": fields,
        "footer": {"text": "backtest/discord_reporter.py \u00b7 Stage 1 screen \u00b7 "
                           "values as recorded by baseline.py, not recomputed"},
    }


# --------------------------------------------------------------------------
# Stage 2 · parameter optimization
# --------------------------------------------------------------------------

def default_scan_summary_path(strat: str, out_dir: str | None = None) -> Path:
    """`<BT_ARTIFACTS>/pipeline/<strategy>/stage2_summary.json`."""
    return pipeline_dir(strat, out_dir) / STAGE2_SUMMARY_FILE


def load_stage2(path: str | Path, strat: str | None = None) -> dict[str, Any]:
    """
    Read Stage 2's summary handoff, and refuse the wrong one.

    Through `pipeline.read_stage` for the same reason Stage 1's card is: it is
    where "written by another stage" and "belongs to another strategy" are
    already refusals. A card announcing one strategy's optimised parameters
    under another's name would be believed - nobody re-derives a Discord post.
    """
    return read_stage(Path(path), 2, strat)


def stage2_rows(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Every configuration Stage 2 was asked to optimise, in one shape.

    `results` is the summary matrix Stage 2 writes for exactly this purpose.
    Errored configurations are included and keep their `ERROR` status: Stage 2
    prunes nothing, so a row missing from the card is a configuration whose
    absence has to be explained, not one that quietly failed a hurdle.
    """
    results = blob.get("results")
    if isinstance(results, list):
        return [dict(r) for r in results if isinstance(r, dict)]
    return []


# `fast_window=5, slow_window=50` -> [("fast_window", "5"), ...]. Split only
# where a comma is followed by an identifier and an `=`, so a value that
# itself contains a comma - `windows=(5, 10)` - is not torn in half.
_PARAM_SPLIT = re.compile(r",\s*(?=[A-Za-z_][A-Za-z0-9_]*\s*=)")


def parse_param_pairs(value: Any) -> list[tuple[str, str]]:
    """
    Split Stage 2's `k=v, k=v` parameter string into pairs, TEXTUALLY.

    No types are restored and no value is reformatted: `False`, `None` and
    `1.5` reach the card as the characters Stage 2 wrote. This module is a
    transcription, and a reporter that parsed `None` into a float would be
    free to print a take-profit that was never modelled.

    Returns `[]` for anything that is not a parameter list - `NOT OPTIMIZED`,
    `(no winner)`, an empty cell - so the caller passes those through verbatim
    instead of rendering them as a malformed pair.
    """
    text = str(value or "").strip()
    if not text or "=" not in text:
        return []
    pairs: list[tuple[str, str]] = []
    for chunk in _PARAM_SPLIT.split(text):
        key, sep, val = chunk.partition("=")
        if not sep or not key.strip():
            return []
        pairs.append((key.strip(), val.strip()))
    return pairs


def _abbrev_key(key: str) -> str:
    """One key, shortened: `fast_window` -> `f`, `tp_atr_mult` -> `tp`."""
    tokens = [t for t in key.split("_") if t]
    if not tokens:
        return key
    significant = [t for t in tokens if t.lower() not in PARAM_NOISE_TOKENS]
    if not significant:
        # Every token was noise (`atr_mult`); keep them rather than return an
        # empty name, and let the collision rule below decide if it is unique.
        significant = tokens
    if len(significant) == 1:
        token = significant[0]
        return token if len(token) <= 3 else token[0]
    return "".join(t[0] for t in significant)


def abbreviate_param_keys(keys: list[str]) -> dict[str, str]:
    """
    Map each key to its table abbreviation, refusing to collapse two keys into
    one.

    A collision gives EVERY key that collided its full name back rather than
    numbering them: `sl_atr_mult` and `slow_window` both shortening to `s` and
    being told apart by a trailing `1` is exactly how a stop distance gets read
    as a moving-average length. A wider column is the cheap failure.
    """
    proposed = {k: _abbrev_key(k) for k in keys}
    taken: dict[str, list[str]] = {}
    for key, short in proposed.items():
        taken.setdefault(short, []).append(key)
    return {k: (short if len(taken[short]) == 1 else k)
            for k, short in proposed.items()}


def compact_params(value: Any) -> str:
    """
    The parameter set as a table cell: keys abbreviated, values verbatim,
    space separated (`f=5 s=50 tp=1.5 sl=1.0 t=False`).

    Nothing is dropped - every parameter Stage 2 recorded is on the cell, only
    its NAME is shortened, and the full names are printed below the table. A
    cell that omitted a parameter would describe a run nobody performed.
    """
    pairs = parse_param_pairs(value)
    if not pairs:
        return str(value or "").strip()
    short = abbreviate_param_keys([k for k, _ in pairs])
    return " ".join(f"{short[k]}={v}" for k, v in pairs)


def _fmt_params(value: Any, limit: int = STAGE2_MAX_PARAM_CHARS) -> str:
    """
    A parameter set for one table cell, abbreviated and then clipped.

    Never `--`: a row that reached the card without parameters is either an
    errored sweep (which says so) or a grid that produced no measurable Sharpe,
    and both are findings. `NOT OPTIMIZED` and `(no winner)` are written by
    Stage 2 and passed through verbatim.

    The ellipsis is a backstop now rather than the card's answer to a wide
    parameter set: whatever is clipped here is printed in full, under its real
    key names, in the block below the table.
    """
    text = compact_params(value)
    if not text:
        return "not recorded"
    return text if len(text) <= limit else text[: limit - 1] + ELLIPSIS


def _stage2_sort_key(row: dict[str, Any]) -> tuple:
    """
    The card's row order: optimised configurations by in-sample profit factor
    descending, errored ones last, ties by symbol and timeframe.

    One function because the table and the full-parameter block below it are
    read as the same list - row three of one has to be row three of the other,
    and two sorts would be free to disagree about which contract that is.
    """
    ok = str(row.get("status") or "").upper() == OPTIMIZED
    try:
        pf = float(row.get("profit_factor"))
    except (TypeError, ValueError):
        pf = float("-inf")
    return (0 if ok else 1, -pf if ok else 0.0,
            str(row.get("symbol") or ""), str(row.get("timeframe") or ""))


def format_stage2_table(rows: list[dict[str, Any]],
                        max_rows: int = STAGE2_MAX_ROWS
                        ) -> tuple[str, int, dict[str, str]]:
    """
    The optimisation summary as one fixed-width block, plus the quadrant legend.

    Ordered by in-sample profit factor descending, which is the metric on the
    card - not by the plateau score the sweep selected on, because that number
    is not a column here and sorting a table on something it does not show is
    how a reader concludes the order is arbitrary. Errored rows sort last and
    keep their place in the count.

    Returns `(text, hidden, legend)`; `hidden` is printed by the caller.
    """
    header = ["SYMBOL", "TF", "QUAD", "IS PF", "MAX DD", "BEST PARAMS"]
    body: list[list[str]] = []
    legend: dict[str, str] = {}

    ordered = sorted(rows, key=_stage2_sort_key)
    shown = ordered[: max(0, int(max_rows))]
    for row in shown:
        quad = row.get("quadrant")
        regime = row.get("optimal_regime")
        if quad and regime:
            legend[str(quad)] = str(regime)
        dd = _fmt_metric(row.get("max_drawdown_pct"))
        body.append([
            str(row.get("symbol") or "?"),
            str(row.get("timeframe") or "?"),
            # `--` where Stage 1 attached no scope, which happens for a pair
            # swept because it was named rather than because it survived. A
            # blank cell would read as a quadrant nobody wrote down.
            str(quad) if quad else "--",
            _fmt_metric(row.get("profit_factor")),
            dd if dd == "--" else f"{dd} %",
            _fmt_params(row.get("params")),
        ])

    if not body:
        # An empty string, not a bare header row. A header with nothing under
        # it reads as a table whose rows were lost; the caller replaces this
        # with a sentence saying the stage optimised nothing, which is a
        # result and needs to be legible as one.
        return "", len(ordered), legend

    widths = [max(len(header[i]), *(len(r[i]) for r in body))
              for i in range(len(header))]
    align = ["<", "<", "<", ">", ">", "<"]

    def line(cells: list[str]) -> str:
        return "  ".join(format(c, f"{align[i]}{widths[i]}")
                         for i, c in enumerate(cells)).rstrip()

    out = [line(header), line(["-" * w for w in widths])]
    out.extend(line(r) for r in body)
    return "\n".join(out), len(ordered) - len(shown), legend


def stage2_param_lines(rows: list[dict[str, Any]]) -> list[str]:
    """
    One line per configuration: `SYMBOL  TF  <parameter set, verbatim>`.

    The parameter set is the string Stage 2 wrote, with its real key names and
    nothing clipped - this block exists so the card carries a copy that can be
    retyped into `--param` without opening the handoff. The symbol and
    timeframe columns are padded so the sets line up under each other; a
    reader comparing two contracts is comparing the values, and ragged left
    edges are what makes that hard.

    Rows arrive in the table's order, so line 1 here is row 1 there.
    """
    if not rows:
        return []
    cells = [(str(r.get("symbol") or "?"), str(r.get("timeframe") or "?"),
              str(r.get("params") or "").strip() or "not recorded")
             for r in rows]
    sym_w = max(len(c[0]) for c in cells)
    tf_w = max(len(c[1]) for c in cells)
    return [f"{sym:<{sym_w}}  {tf:<{tf_w}}  {params}"
            for sym, tf, params in cells]


def format_stage2_param_fields(rows: list[dict[str, Any]],
                               budget: int = MAX_EMBED_TOTAL,
                               max_fields: int = STAGE2_PARAM_MAX_FIELDS,
                               name: str = STAGE2_PARAM_FIELD_NAME
                               ) -> tuple[list[dict[str, Any]], int]:
    """
    The full parameter sets as embed fields, packed to Discord's limits.

    The table above them abbreviates and clips, because a fixed-width column
    has to align; this is the unabbreviated record, and it covers EVERY
    configuration in the matrix - including the rows the table's own row cap
    left off, and including a failed sweep's `NOT OPTIMIZED`, which is a
    finding rather than a missing parameter set.

    `budget` is what is left of the 6000-character embed after the description
    and the other fields, and a line that does not fit is COUNTED in the
    returned `hidden` - the same rule as every other cap on these cards, and
    for the same reason: a silently shortened list of winning parameters reads
    as the whole stage. A line too wide for one field is WRAPPED, never
    clipped; the whole point of the block is that nothing in it is cut.

    Returns `(fields, hidden)`.
    """
    lines = stage2_param_lines(rows)
    if not lines:
        return [], 0

    fence_cost = len(FENCE_OPEN) + len(FENCE_CLOSE)
    # Room for the "N further ..." note, which is appended to the last field.
    room = int(budget) - NOTE_RESERVE
    width = min(MAX_FIELD_VALUE, room) - fence_cost - len(name) - len(" \u00b7 cont.")
    if width <= 0:
        return [], len(lines)

    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_param_line(line, width))

    fields: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(wrapped) and len(fields) < max_fields:
        label = name if not fields else f"{name} \u00b7 cont."
        # Recomputed per field: `room` shrinks as fields are appended, and a
        # capacity fixed at the first field's size is how an embed clears a
        # local check and is rejected with a 400 nobody reads.
        capacity = min(MAX_FIELD_VALUE, room - len(label)) - fence_cost
        if capacity <= 0:
            break
        chunk: list[str] = []
        used = 0
        while cursor < len(wrapped):
            addition = len(wrapped[cursor]) + (1 if chunk else 0)
            if used + addition > capacity:
                break
            chunk.append(wrapped[cursor])
            used += addition
            cursor += 1
        if not chunk:
            break
        field = {"name": label,
                 "value": FENCE_OPEN + "\n".join(chunk) + FENCE_CLOSE,
                 "inline": False}
        fields.append(field)
        room -= len(field["name"]) + len(field["value"])

    # `hidden` counts CONFIGURATIONS, not wrapped physical lines: a reader
    # chasing "3 further configurations" into the handoff is looking for three
    # contracts, and a count of line fragments would send them looking for a
    # number of rows that is not in the file. A configuration whose line was
    # cut mid-wrap counts as hidden, because a half-printed parameter set is
    # not a printed one.
    cut_mid_row = (cursor < len(wrapped)
                   and wrapped[cursor].startswith(" " * INDENT_WIDTH))
    hidden = len(lines) - _count_configurations(wrapped[:cursor], cut_mid_row)
    if hidden > 0:
        note = (f"\n_{hidden} further configuration(s) not shown \u2014 see "
                f"`best_params_<SYMBOL>_<TF>.json`._")
        if fields:
            fields[-1]["value"] += note
    return fields, max(0, hidden)


def _param_field(name: str, lines: list[str], index: int) -> dict[str, Any]:
    """One packed field. Continuations say so rather than repeating the name
    unqualified, which would read as a second, different list."""
    return {
        "name": name if index == 0 else f"{name} \u00b7 cont.",
        "value": FENCE_OPEN + "\n".join(lines) + FENCE_CLOSE,
        "inline": False,
    }


def _wrap_param_line(line: str, width: int) -> list[str]:
    """
    Wrap one configuration's line to `width`, continuations indented.

    Wrapped rather than clipped, and split on the parameter separator rather
    than mid-token: half of `sl_atr_mult=1.0` on one line and half on the next
    is a value a reader can misread as a whole one.
    """
    if len(line) <= width or width <= INDENT_WIDTH + 1:
        return [line]
    out: list[str] = []
    remaining = line
    indent = ""
    while len(remaining) > width:
        cut = remaining.rfind(" ", 0, width + 1)
        if cut <= len(indent):
            cut = width
        out.append(remaining[:cut].rstrip())
        indent = " " * INDENT_WIDTH
        remaining = indent + remaining[cut:].lstrip()
    out.append(remaining)
    return out


def _count_configurations(lines: list[str], cut_mid_row: bool = False) -> int:
    """
    How many WHOLE configurations a slice of wrapped lines covers.

    A continuation is indented, so it is not a row of its own. `cut_mid_row`
    says the slice ended with a configuration's continuation still to come, and
    that row is not counted: a half-printed parameter set is not a printed one,
    and counting it would leave a reader one contract short with nothing on the
    card saying so.
    """
    whole = sum(1 for line in lines if not line.startswith(" " * INDENT_WIDTH))
    return max(0, whole - 1) if cut_mid_row else whole


def build_stage2_embed(strat: str, blob: dict[str, Any],
                       source: str | Path | None = None,
                       max_rows: int = STAGE2_MAX_ROWS) -> dict[str, Any]:
    """
    Stage 2's card. Pure - sends nothing, computes nothing, and every value on
    it is transcribed from the summary Stage 2 wrote.

    The window is on the card because an optimised parameter set is only
    meaningful with the bars it was fitted to, and because it is the one field
    that says the holdout was not touched. The coverage line is there because
    Stage 2 prunes nothing: `12/12 optimised` is the claim, and anything less
    is a run failure rather than a screening result.
    """
    rows = stage2_rows(blob)
    optimized = [r for r in rows
                 if str(r.get("status") or "").upper() == OPTIMIZED]
    # NO TABLE. Same reasoning as the Stage 1 card: a fixed-width block inside
    # a container that reflows loses its alignment on a narrow client, which is
    # the only thing a table was for. The per-configuration detail lives in
    # stage2_summary_matrix.csv, which this card links.
    window = blob.get("in_sample_window") or {}
    start = window.get("start") or blob.get("start") or "not recorded"
    end = window.get("end") or blob.get("end") or "not recorded"
    holdout = window.get("holdout_starts")
    timeframes = blob.get("timeframes") or []
    coverage = blob.get("coverage") or {}
    rank = blob.get("rank") or "not recorded"

    pfs = sorted(float(r["profit_factor"]) for r in rows
                 if isinstance(r.get("profit_factor"), (int, float)))

    description = [
        f"**Strategy** `{strat}`",
        f"**In-sample window** `{start} \u2192 {end}`"
        + (f" \u00b7 holdout from `{holdout}` untouched" if holdout else ""),
        # `rank` is what was APPLIED, which Stage 2 resolves - a rebuild of a
        # table with no plateau columns is ranked on Sharpe however the sweep
        # was invoked, and the card must not claim otherwise.
        f"**Selection** best parameters by `{rank}` rank, per configuration",
    ]
    if pfs:
        # The RANGE, not a mean. Averaging profit factors across contracts
        # blends separate simulations on different multipliers into one number
        # that describes no instrument - the same reason nothing else here is
        # summed across symbols.
        description.append(
            f"**Optimised PF** `{pfs[0]:.2f}` \u2013 `{pfs[-1]:.2f}` "
            f"(median `{pfs[len(pfs) // 2]:.2f}`)")
    description.append("")
    description.append(
        "_Per-configuration parameters, plateau scores and drawdowns saved to "
        "`stage2_summary_matrix.csv`._")

    text = "\n".join(description)

    errors = len(rows) - len(optimized)
    fields = [
        {"name": "Configurations", "value": str(len(rows)), "inline": True},
        # "Optimised" rather than "Promoted": Stage 2 promotes nothing and
        # drops nothing. Every row advances to Stage 3, and a heading borrowed
        # from the Stage 1 card would import a survival rate that does not
        # exist here.
        {"name": "Optimised → Stage 3", "value": str(len(optimized)),
         "inline": True},
        {"name": "Failed to sweep", "value": str(errors), "inline": True},
        {"name": "Timeframes",
         "value": ", ".join(f"`{t}`" for t in timeframes) or "not recorded",
         "inline": True},
        {"name": "Pruning",
         "value": ("none — every Stage 1 survivor advances"
                   if coverage.get("complete") is True
                   else f"none by design; {errors} configuration(s) failed to "
                        f"sweep and carry no parameters"),
         "inline": True},
    ]

    embed = {
        "title": f"\U0001F4CA Stage 2 · Parameter Scan & Plateau Ranking: {strat}",
        "description": text,
        # Violet when something was optimised, amber when nothing was. Amber
        # rather than red for the same reason as Stage 1: a stage that produced
        # no rows is a result to look at, not a crash.
        "color": VIOLET if optimized else AMBER,
        "fields": fields,
        "footer": {"text": "backtest/discord_reporter.py · Stage 2 parameter "
                           "optimization · values as recorded by scan.py, not "
                           "recomputed"},
    }

    # The full parameter sets are NOT fields any more. They were five
    # continuation blocks of monospace text - 5,212 of Discord's 6,000
    # characters on a 47-configuration run - and they wrapped on exactly the
    # clients the table did. Every one of them is the `params` column of
    # stage2_summary_matrix.csv and the `params` key of
    # best_params_<SYMBOL>_<TF>.json, both of which this card names.
    #
    # `format_stage2_param_fields` is KEPT, not deleted: it is covered by its
    # own tests and is the formatter to reach for if these ever return behind
    # a flag. Nothing calls it here.
    handoff = {"name": "Handoff", "value": _fmt_report(str(source or "")),
               "inline": False}
    embed["fields"] = fields + [handoff]
    return embed


# --------------------------------------------------------------------------
# Stage 3 · gate audit & certification
# --------------------------------------------------------------------------

def default_audit_summary_path(strat: str, out_dir: str | None = None) -> Path:
    """`<BT_ARTIFACTS>/pipeline/<strategy>/stage3_audit_summary.json`."""
    return pipeline_dir(strat, out_dir) / STAGE3_SUMMARY_FILE


def load_stage3(path: str | Path, strat: str | None = None) -> dict[str, Any]:
    """
    Read Stage 3's handoff, and refuse the wrong one.

    Through `pipeline.read_stage` for the same reason the other two cards are:
    a file written by another stage or belonging to another strategy is a
    refusal there already. It matters most here - this card carries a
    PASS/FAIL and a hash seal, and a certification announced under the wrong
    strategy's name is the artifact nobody cross-checks.
    """
    return read_stage(Path(path), 3, strat)


def stage3_rows(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Every configuration the certification run covered, in one shape.

    Straight off `results`, which Stage 3 wrote with errors and skips already
    in it as `NOT AUDITED` rows. Nothing is filtered here: a card shorter than
    the stage's input reads as a complete certification, and "the audit raised"
    is a different statement from "the edge did not hold out of sample".
    """
    return list(blob.get("results") or [])


# --------------------------------------------------------------------------
# Stage 3 has TWO inputs on disk, and they are different shapes.
#
# `stage3_audit_summary.json` is the campaign INDEX: one `results` row per
# (configuration, version) across every timeframe the campaign certified.
# `gate_audit_<SYMBOL>_<TF>.json` is the AUTHORITATIVE verdict for ONE pair,
# and it is the file Stage 3 writes first - the summary is transcribed from
# it. A pair audit therefore holds every value this card prints, and a card
# that could only read the index announced nothing at all whenever the index
# was missing, stale, or flattened by the pre-merge overwrite. That is the
# common case straight after a single `audit_gates.py --strat X --tf 1h` run.
#
# Both are read through `pipeline.read_stage`, which refuses another stage's
# file and another strategy's, and both are TRANSCRIBED - the row builder
# below re-derives no verdict, exactly as `audit_gates.write_stage3_summary`
# transcribes the same fields when it writes the index. It is deliberately not
# imported from there: `backtest.audit_gates` pulls in the engine and
# vectorbtpro, and a notifier that cannot post a card because the simulation
# stack failed to import is a quiet pipeline. `tests/test_stage3_charter.py`
# pins the two builders against each other field for field, which is what
# stops the transcription drifting.
# --------------------------------------------------------------------------

def is_pair_audit(blob: dict[str, Any]) -> bool:
    """
    True for a `gate_audit_<SYMBOL>_<TF>.json`, False for the summary index.

    Discriminated on CONTENT and never on the filename: an operator naming a
    file by hand can point `--audit` at either, and a renamed copy of one is
    still the shape it is. `results` is the summary's rows and `versions` is
    the pair audit's per-version blocks; neither file carries the other's key.
    """
    return "results" not in blob and bool(blob.get("versions"))


def _gate_status(gates: dict[str, Any], name: str) -> str:
    """One gate's recorded status, or `NOT EVALUATED` when it holds none."""
    return str((gates.get(name) or {}).get("status") or NOT_EVALUATED)


def stage3_rows_from_audit(blob: dict[str, Any],
                           path: str | Path | None = None
                           ) -> list[dict[str, Any]]:
    """
    One pair audit as the summary rows Stage 3 would have written for it.

    A pure transcription, field for field, in the SAME shape and under the
    same names `audit_gates.write_stage3_summary` uses - so everything
    downstream of `stage3_rows` (the table, the sort, the promotion block, the
    counters) reads a pair audit and the index identically, and a card built
    from one cannot say something different from a card built from the other.

    One row per VERSION, because a pair audit can carry Version A and Version
    B and they reach separate verdicts; `sorted` so A precedes B whatever
    order the file stored them in.

    The top-level `status` / `passed` maps are preferred and the per-version
    `gate_audit` block is the fallback, which is where `certify_symbol` lifted
    them from in the first place - reading it is the same field one level
    down, not a second opinion.
    """
    path = Path(path) if path else None
    versions = blob.get("versions") or {}
    status = blob.get("status") or {}
    passed = blob.get("passed") or {}
    incubator = blob.get("incubator") or {}
    exclude_days = list((blob.get("entry_filters") or {})
                        .get("exclude_days") or [])

    rows: list[dict[str, Any]] = []
    for ver in sorted(versions):
        block = versions[ver] or {}
        audit = block.get("gate_audit") or {}
        gates = audit.get("gates") or {}
        gate_r = gates.get(GATE_R_KEY) or {}
        measured = gate_r.get("measured") or {}
        retention = (block.get("retention") or {}).get("metrics") or {}
        staged = incubator.get(ver) or {}
        rows.append({
            "symbol": blob.get("symbol"),
            "timeframe": blob.get("timeframe"),
            "version": ver,
            # The version STAGE 1 qualified this pair on, beside the version
            # this row audits. `**prov` puts all three on the gate audit, so
            # the adapter reads them off the file rather than re-deriving
            # them - a B survivor certified as A only has to read the same way
            # here as it does in `stage3_audit_summary.json`.
            "stage1_version": blob.get("stage1_version"),
            "stage1_version_certified": (
                None if not blob.get("stage1_version")
                else str(blob.get("stage1_version")).upper() == ver),
            "version_b_certified": bool(blob.get("version_b_certified")),
            "version_b_source": blob.get("version_b_source"),
            "status": status.get(ver, audit.get("status", NOT_EVALUATED)),
            # Gate R's own flag, never re-read off the numbers beside it.
            "certified": bool(passed.get(ver, audit.get("passed"))),
            "target_regime": blob.get("target_regime"),
            "quadrant": blob.get("target_quadrant"),
            "gate_regime": _gate_status(gates, GATE_R_KEY),
            # Gate R's quadrant numbers, on the holdout. NOT the blended
            # sample - the two `profit_factor` fields below say which is which
            # rather than leaving one name to be read as either.
            "oos_profit_factor": measured.get("profit_factor"),
            "oos_trade_count": measured.get("trade_count"),
            "oos_win_rate": measured.get("win_rate"),
            "is_profit_factor": (retention.get("profit_factor")
                                 or {}).get("in_sample"),
            "holdout_profit_factor": (retention.get("profit_factor")
                                      or {}).get("holdout"),
            "retention": {k: (v or {}).get("retention")
                          for k, v in retention.items()},
            "gate1": _gate_status(gates, "gate1"),
            "gate2": _gate_status(gates, "gate2"),
            "gate3": _gate_status(gates, "gate3"),
            # Present only when Gate R failed on the TRADE COUNT. `None` says
            # the quadrant was not starved, which is a different statement
            # from a pass - `status` above is what says that.
            "regime_starvation": (gate_r.get("regime_starvation")
                                  or {}).get("message"),
            "params": blob.get("params") or {},
            "params_locked": bool(blob.get("params_locked")),
            "in_stage1": bool(blob.get("in_stage1", True)),
            "exclude_days": exclude_days,
            "audit_file": str(path) if path else None,
            "audit_sha256": (sha256(path) if path and path.exists()
                             else "NOT AVAILABLE"),
            "incubator_dir": (str(staged["dir"]) if staged.get("dir")
                              else None),
            "incubator_error": staged.get("error") or "",
            "seal": staged.get("seal") or None,
            "error": "",
        })
    return rows


def _agreed(blocks: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """
    One window (`in_sample` / `holdout`) shared by every audit read, or the
    disagreement stated in the value itself.

    Several pair audits assembled into one card can genuinely disagree here -
    a run given an explicit `--holdout-end` and one that defaulted to the
    present record different ends over the same bars. Printing the first
    file's window above rows measured on another's would misdescribe the
    verdict; `varies by pair` renders in the same slot and is true.
    """
    seen = [b.get(key) or {} for b in blocks]
    first = seen[0] if seen else {}
    if all(w == first for w in seen):
        return first
    return {"start": "varies by pair", "end": "varies by pair"}


def stage3_blob_from_audits(
        audits: list[tuple[dict[str, Any], Path]]) -> dict[str, Any]:
    """
    One or more pair audits assembled into the summary shape the card reads.

    It INDEXES; it does not certify. Every row is transcribed by
    `stage3_rows_from_audit` and every count below is a count of those rows,
    so this can never announce a verdict no `gate_audit_<SYMBOL>_<TF>.json`
    recorded.

    `coverage.targets` is the number of audits READ, which is not the number
    Stage 3 was asked to certify - a pair whose audit raised wrote no file at
    all, so it cannot appear here. `coverage.rule` says so on the card rather
    than leaving a complete-looking count to be read as the campaign's.
    """
    blobs = [b for b, _ in audits]
    rows = [row for blob, path in audits
            for row in stage3_rows_from_audit(blob, path)]
    tfs: list[str] = []
    for row in rows:
        tf = str(row.get("timeframe") or "")
        if tf and tf not in tfs:
            tfs.append(tf)
    audited = [r for r in rows
               if str(r.get("status") or "").upper() != NOT_AUDITED]
    certified = [r for r in rows if r.get("certified")]
    first = blobs[0] if blobs else {}
    return {
        "stage": 3,
        "stage_name": STAGE_NAMES.get(3, "GATE AUDIT · certification"),
        "strategy": first.get("strategy"),
        "generated_utc": first.get("generated_utc"),
        # Absent on a pair audit. Left absent rather than guessed at: the
        # promotion command prints a visible placeholder for a missing
        # `--source`, which fails loudly, where a path this module invented
        # would promote whatever happens to sit at it.
        "strategy_source": first.get("strategy_source"),
        "in_sample": _agreed(blobs, "in_sample"),
        "holdout": _agreed(blobs, "holdout"),
        # The thresholds Gate R was held to, from the audits themselves. They
        # are module constants, so the first file's block describes them all;
        # the card holds no configuration to a bar its own audit did not use.
        "certification_rule": first.get("certification_rule") or {},
        "prop_firm_rules": first.get("prop_firm_rules") or {},
        "timeframe": tfs[-1] if tfs else None,
        "timeframes": tfs,
        "coverage": {
            "targets": len(rows),
            "audited": len(audited),
            "certified": len(certified),
            "errors": 0,
            "skipped": 0,
            "complete": True,
            "timeframes": tfs,
            "rule": ("assembled from the per-pair gate audits on disk, which "
                     "are the authoritative verdicts. A configuration whose "
                     "audit RAISED wrote no file and cannot appear here, so "
                     "these counts describe the audits read and not the "
                     "campaign Stage 3 was asked to certify."),
        },
        "audits": [{"path": str(path), "timeframe": blob.get("timeframe"),
                    "symbol": blob.get("symbol"),
                    "status": blob.get("status") or {}}
                   for blob, path in audits],
        "results": rows,
        "source_kind": "per-pair gate audit(s)",
    }


def discover_pair_audits(strat: str, out_dir: str | None = None) -> list[Path]:
    """
    The `gate_audit_<SYMBOL>_<TF>.json` files in the strategy's pipeline
    directory, in a stable order.

    **Only the SUFFIXED files.** The unsuffixed `gate_audit_<SYMBOL>.json` is
    a duplicate of whichever timeframe ran last, and reading both would index
    one verdict twice under two names - the same rule
    `audit_gates.rebuild_stage3_summary` reads them by.

    **Every one of them, never the newest.** A campaign leaves one audit per
    (symbol, timeframe); picking one would announce a single certification
    while the others sat on disk unread, which is precisely the failure the
    summary's cross-timeframe merge exists to prevent.
    """
    glob = GATE_AUDIT_FILE.format(symbol="*")
    return sorted(p for p in pipeline_dir(strat, out_dir).glob(glob)
                  if GATE_AUDIT_NAME.match(p.name))


def resolve_stage3_input(strat: str, audit: str | Path | None = None,
                         summary: str | Path | None = None,
                         out_dir: str | None = None
                         ) -> tuple[dict[str, Any], Path, str]:
    """
    What the Stage 3 card is built from: `(blob, source, what)`.

    Three routes, and the file named by hand always wins:

    * `--summary` is the campaign index, and only the index. A pair audit
      handed to it is REFUSED rather than adapted - the flag names one shape
      and silently accepting the other makes the two words mean nothing.
    * `--audit` takes EITHER, discriminated on content by `is_pair_audit`,
      because that is the flag every existing Stage 3 command already spells
      and an operator pointing it at a single verdict means that verdict.
    * With neither, the index is preferred (it spans the whole campaign) and
      the per-pair audits are the fallback (they are the authoritative
      verdicts, and they exist whenever a Stage 3 run finished at all).

    A path named explicitly and missing RAISES; one this went looking for on
    its own is reported with what it looked for, because "no certification has
    run" and "I was pointed at the wrong directory" are fixed by different
    work.
    """
    if audit and summary:
        raise ValueError(
            "--audit and --summary name the same input twice. Pass --summary "
            f"for {STAGE3_SUMMARY_FILE}, or --audit for it or for a single "
            f"{GATE_AUDIT_FILE.format(symbol='<SYMBOL>_<TF>')}.")

    if summary:
        path = Path(summary)
        blob = load_stage3(path, strat)
        if is_pair_audit(blob):
            raise ValueError(
                f"--summary {path.name} is a per-pair gate audit, not "
                f"{STAGE3_SUMMARY_FILE}. Pass it with --audit.")
        return blob, path, "campaign summary"

    if audit:
        path = Path(audit)
        blob = load_stage3(path, strat)
        if is_pair_audit(blob):
            return (stage3_blob_from_audits([(blob, path)]), path,
                    "one per-pair gate audit")
        return blob, path, "campaign summary"

    path = default_audit_summary_path(strat, out_dir)
    if path.exists():
        return load_stage3(path, strat), path, "campaign summary"

    found = discover_pair_audits(strat, out_dir)
    if not found:
        raise FileNotFoundError(
            f"No Stage 3 handoff under {pipeline_dir(strat, out_dir)}: "
            f"neither {STAGE3_SUMMARY_FILE} nor any "
            f"{GATE_AUDIT_FILE.format(symbol='<SYMBOL>_<TF>')}. Run "
            f"`python3 backtest/audit_gates.py --strat {strat} --tf <TF>` "
            f"first, or name the file with --audit.")
    audits = [(load_stage3(p, strat), p) for p in found]
    return (stage3_blob_from_audits(audits), pipeline_dir(strat, out_dir),
            f"{len(audits)} per-pair gate audit(s)")

def _seal_prefix(row: dict[str, Any], key: str = "strategy_code") -> str:
    """
    One of a configuration's seals, shortened.

    `--` when nothing was staged: an uncertified configuration has no seal,
    and a blank cell in a hash column reads as a hash of nothing. The CODE
    hash and the WINNING PARAMETERS hash are both used - the same module under
    a different grid cell is a different strategy with the same code checksum,
    so a code prefix alone identifies a build only if the parameters travelled
    with it.
    """
    seal = row.get("seal") or {}
    digest = ((seal.get(key) or {}).get("sha256") or "")
    if not digest or digest == "NOT AVAILABLE":
        return "--"
    return digest[:SEAL_PREFIX_CHARS]


def _regime_pf_cell(row: dict[str, Any]) -> str:
    """
    Gate R's quadrant profit factor as a table cell, with the sentinel
    rendered as `--`.

    `999.00` is what the profiler writes when a quadrant never had a losing
    trade. It is not a measured factor, and it is exactly the number a starved
    quadrant prints: one winning holdout trade shows 999.00 beside a Gate R
    that failed on the sample count, which reads as the strongest row on the
    card. `--` is what every other column here uses for "nothing was
    measured", and that is the true statement about a factor with no losing
    trade under it.
    """
    pf = _fmt_float(row.get("oos_profit_factor"))
    if pf is None or pf >= REGIME_PF_SENTINEL:
        return "--"
    return f"{pf:.2f}"


def gate_r_reason(row: dict[str, Any], rule: dict[str, Any] | None) -> str:
    """
    WHY a Gate R failed: `N` (too few trades in the quadrant), `PF` (the
    factor missed), or `""` when it did not fail or cannot be told.

    A FAIL on the count and a FAIL on the factor read identically as `FAIL`
    and are fixed by completely different work - a quadrant the strategy never
    entered again is a designation problem, a factor below 1.00 is a dead
    edge. Stage 3 records the first as `regime_starvation` and that record is
    preferred whenever it is there.

    **The PASS/FAIL token itself is never re-derived** - it is transcribed
    from `gate_regime`, as everything else on this card is. Only the
    parenthetical reason is worked out here, and only from the thresholds the
    handoff itself recorded in `certification_rule`, so the card cannot hold a
    configuration to a bar the audit did not use. With either number missing
    it returns `""` and the cell stays a bare `FAIL`, which is the honest
    answer when the reason is not on the file.
    """
    if str(row.get("gate_regime") or "").upper() != "FAIL":
        return ""
    if row.get("regime_starvation"):
        return "N"
    rule = rule or {}
    n, floor = _fmt_float(row.get("oos_trade_count")), \
        _fmt_float(rule.get("min_trades"))
    if n is not None and floor is not None and n < floor:
        return "N"
    pf, bar = _fmt_float(row.get("oos_profit_factor")), \
        _fmt_float(rule.get("min_profit_factor"))
    if pf is not None and bar is not None and pf < bar:
        return "PF"
    return ""


def _gate_r_cell(row: dict[str, Any], rule: dict[str, Any] | None) -> str:
    """`PASS`, `FAIL`, `FAIL·N`, `FAIL·PF`, `NO EVAL` or `NO AUDIT`."""
    status = str(row.get("gate_regime") or NOT_AUDITED).upper()
    cell = GATE_R_TOKENS.get(status, status)
    reason = gate_r_reason(row, rule)
    return f"{cell}·{reason}" if reason else cell


def _status_cell(row: dict[str, Any], rule: dict[str, Any] | None) -> str:
    """
    The outcome column: `CERTIFIED`, `STARVED`, `REJECTED`, `NOT CERT` or
    `NO AUDIT`.

    `CERTIFIED` is the handoff's own `certified` flag and never a re-reading
    of the numbers beside it. The other four are what that flag being false
    MEANS, which is not one thing: a run that broke, a quadrant nobody
    designated, a quadrant that starved and an edge that died are four
    findings fixed by four different pieces of work, and one `NOT CERTIFIED`
    token hides all of them.
    """
    if row.get("certified"):
        return "CERTIFIED"
    if str(row.get("status") or "").upper() == NOT_AUDITED:
        return "NO AUDIT"
    return {"N": "STARVED", "PF": "REJECTED"}.get(gate_r_reason(row, rule),
                                                  "NOT CERT")


def _stage3_sort_key(row: dict[str, Any]) -> tuple:
    """
    Certified first, then the strongest out-of-sample quadrant, then by name.

    Sorted on the QUADRANT profit factor rather than the blended one, because
    that is what Gate R decided on: ranking on the blend would put a
    configuration with a broad mediocre result above one with a sharp edge in
    its own environment, which inverts the question the stage asks. A row with
    no measurable factor sorts last rather than being dropped.
    """
    certified = bool(row.get("certified"))
    pf = _fmt_float(row.get("oos_profit_factor"))
    return (not certified, -(pf if pf is not None else float("-inf")),
            str(row.get("symbol") or ""), str(row.get("timeframe") or ""),
            str(row.get("version") or ""))


def _fmt_float(value: Any) -> float | None:
    """A float, or None. NaN is None - it is not a number and must not sort."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def format_stage3_table(rows: list[dict[str, Any]],
                        max_rows: int = STAGE3_MAX_ROWS,
                        rule: dict[str, Any] | None = None
                        ) -> tuple[str, int, dict[str, str]]:
    """
    The certification table as one fixed-width block, plus the quadrant legend.

    Returns `(text, hidden, legend)`. `hidden` is the number of CERTIFIED rows
    that did not fit `max_rows`, and is counted on the card by the caller.

    **Only CERTIFIED configurations are listed** (`STAGE3_CERTIFIED_ONLY`).
    STARVED, REJECTED, NOT CERT and NO AUDIT rows are off the table entirely:
    this card is read to answer "what may be promoted", and that is the only
    row anybody acts on. The exclusion is stated on the card and the counts
    beside it still describe the WHOLE run - `Configurations`, `Certified ->
    Incubator` and `Audited` are unfiltered - so a shorter table can never read
    as a shorter certification run. Every configuration's verdict remains on
    the handoff and in its own `gate_audit_<SYMBOL>_<TF>.json`, which is the
    authoritative file either way. With nothing certified the block carries the
    header and `STAGE3_NO_ROWS_NOTE` rather than rendering empty.

    **It is built to a WIDTH, not to a column list** (`STAGE3_TABLE_WIDTH`,
    45 characters). Discord wraps a code block that overruns the viewport, and
    a wrapped fixed-width table is worse than no table: every row becomes two,
    the second one unlabelled, and the columns a reader is comparing no longer
    line up under each other. The old ten-column row ran to 68 characters and
    wrapped on every phone. So the row carries the six things a certification
    IS - which contract, at which timeframe, in which quadrant, what Gate R
    said, and the factor and sample it said it on - and nothing else.

    What was dropped, and where it went. `VER` is on the promotion bullets,
    which is the only place a reader acts on it. The blended IS and OOS
    factors are on the promotion bullets too, for the certified rows: the
    collapse they exist to expose ("1.10, down from 2.40") only changes a
    decision for a configuration somebody is about to promote, and for the
    rest they are two more numbers that decided nothing. The `SEAL` prefix is
    likewise on the bullets, beside the parameters it seals.

    `PF` and `N` are Gate R's OWN quadrant numbers - the factor and the trade
    count inside the designated quadrant, on the holdout - never the blended
    sample. The header is short and the description says which they are,
    because an unlabelled profit factor under a regime-gated verdict is the
    one number on this card that must not be guessed at.

    The QUAD column carries the `Q1`..`Q4` id the handoff recorded, with the
    legend built FROM the rows. No short spelling of a regime name lives in
    this module, for the same reason it does not on the Stage 1 card: a second
    one would be free to disagree with `mdlib.regimes`.
    """
    header = ["SYM", "TF", "QD", "GATE R", "PF", "N", "STATUS"]
    body: list[list[str]] = []
    legend: dict[str, str] = {}

    # CERTIFIED rows only. Filtered on the STATUS cell rather than on the
    # `certified` flag directly, so the table and the column can never
    # disagree about what the word means - `_status_cell` is the one place the
    # token is produced, and it transcribes the handoff's own flag.
    ordered = sorted((r for r in rows
                      if not STAGE3_CERTIFIED_ONLY
                      or _status_cell(r, rule) == "CERTIFIED"),
                     key=_stage3_sort_key)
    shown = ordered[: max(0, int(max_rows))]
    for row in shown:
        quad = row.get("quadrant")
        regime = row.get("target_regime")
        if quad and regime:
            legend[str(quad)] = str(regime)
        body.append([
            str(row.get("symbol") or "?"),
            str(row.get("timeframe") or "?"),
            str(quad) if quad else "--",
            # The GATE R status verbatim from the handoff. Never re-derived
            # from the numbers beside it: this module computes nothing, and a
            # card that recomputed a verdict would be free to disagree with the
            # audit it is announcing. Only the ·N / ·PF suffix is worked out
            # here - see gate_r_reason.
            _gate_r_cell(row, rule),
            _regime_pf_cell(row),
            _fmt_count(row.get("oos_trade_count")),
            _status_cell(row, rule),
        ])

    widths = [max(len(header[i]), STAGE3_MIN_WIDTHS[i],
                  *(len(r[i]) for r in body)) if body
              else max(len(header[i]), STAGE3_MIN_WIDTHS[i])
              for i in range(len(header))]
    align = ["<", ">", "<", "<", ">", ">", "<"]

    def line(cells: list[str]) -> str:
        return "  ".join(format(c, f"{align[i]}{widths[i]}")
                         for i, c in enumerate(cells)).rstrip()

    out = [line(header), line(["-" * w for w in widths])]
    # Nothing certified is a RESULT, not an empty render. The header stays
    # above it so the block is recognisable as the same table.
    if body:
        out.extend(line(r) for r in body)
    else:
        out.append(STAGE3_NO_ROWS_NOTE)
    return "\n".join(out), len(ordered) - len(shown), legend


def stage3_timeframes(blob: dict[str, Any]) -> list[str]:
    """
    Every timeframe this summary indexes, in order.

    Read off the handoff's own `timeframes` when it has one and off the rows
    otherwise, so a summary written before Stage 3 merged across timeframes
    still names what it holds. The single `timeframe` field is the LAST run's
    and is deliberately not used here: a card headed `15m` above a table
    carrying 5m rows describes neither.
    """
    declared = [str(t) for t in (blob.get("timeframes") or []) if t]
    if declared:
        return declared
    seen: list[str] = []
    for row in stage3_rows(blob):
        tf = row.get("timeframe")
        if tf and str(tf) not in seen:
            seen.append(str(tf))
    return seen


def stage3_param_text(value: Any) -> str:
    """
    A winning parameter set as `k=v, k=v`, from a dict or from a string.

    Stage 3 records `params` as a DICT (Stage 2 records a string), and both
    shapes reach this module. Values are stringified and never reformatted -
    `None` stays `None`, which for `tp_atr_mult` means no take-profit was
    modelled at all, and a reporter that turned it into a number would print a
    target that never existed.
    """
    if isinstance(value, dict):
        if not value:
            return "not recorded"
        return ", ".join(f"{k}={value[k]}" for k in value)
    text = str(value or "").strip()
    return text or "not recorded"


def stage3_auto_promotion(blob: dict[str, Any]) -> dict[str, Any]:
    """
    The auto-promotion record `run_pipeline.py --auto-promote` writes back onto
    Stage 3's summary, or `{}` when nothing promoted.

    Written by the orchestrator rather than by Stage 3, because Stage 3 stages
    into the incubator and never commits - the commit is Stage 5's, and the
    hash only exists once it has run. The card reads it and changes its
    HEADING; it never infers a promotion from a seal, because a sealed
    configuration is one Stage 3 staged and nobody committed.
    """
    blk = blob.get("auto_promotion")
    return blk if isinstance(blk, dict) else {}


def promotion_command(strat: str, row: dict[str, Any],
                      source: str | None) -> str:
    """
    The exact Stage 5 command for ONE configuration, on one line.

    This is the RECOVERY path and not the card's normal instruction: it is
    printed only for a configuration whose automatic promotion failed, where
    an operator needs the one pair rather than the whole certified set. It is
    spelled out in full - strategy, version, source module and that pair's own
    `gate_audit_<SYMBOL>_<TF>.json` - because the alternative is reconstructing
    it against a directory holding one audit per timeframe, and picking the
    wrong file promotes a verdict about different bars. `--source` comes from
    the handoff; when Stage 3 did not record one the placeholder is left
    visible rather than the flag dropped, so the command fails loudly instead
    of promoting the module's defaults.
    """
    version = str(row.get("version") or "A")
    audit = str(row.get("audit_file") or "NOT RECORDED")
    src = str(source or "<path to the strategy module>")
    return (f"python3 backtest/promote.py --strat {strat} "
            f"--version {version} --source {src} --audit-file {audit}")


def unified_promote_command(strat: str) -> list[str]:
    """
    The one command that promotes everything Stage 3 certified, as the two
    lines it is printed on.

    `--promote-only`, never `--auto-promote`: the second re-runs Stages 1-4
    first and OVERWRITES the handoff this card was built from, so the winners
    it then promotes are a fresh sweep's and not the ones the reader is
    looking at. One flag is the difference between promoting what was
    certified and silently re-certifying from scratch.

    Split on a backslash continuation rather than run out to 90 characters:
    a code block that overruns the viewport wraps, and a wrapped command is
    one a reader copies half of.
    """
    return ["python3 backtest/run_pipeline.py \\",
            f"    --strat {strat} --promote-only"]


def format_stage3_promotions(strat: str, blob: dict[str, Any],
                             max_rows: int = STAGE3_PROMO_MAX_ROWS
                             ) -> tuple[str, list[str], list[str], int]:
    """
    The promotion block: `(title, lines, pairs, hidden)`.

    `pairs` is the one-line answer the description carries (`` `CL 15m` ``);
    `lines` are MARKDOWN, not a code fence - a bullet per certified
    configuration with its quadrant, the factor Gate R scored, the winning
    parameter plateau, the blended pair behind it and its seals, and ONE
    command underneath that promotes all of them.

    **One command, not one per pair.** Three certified configurations used to
    print three three-line bash blocks carrying four absolute filesystem
    paths each, which is 12 lines of wrapped path on a phone and the reason
    this section was unreadable. `run_pipeline.py --promote-only` iterates
    every certified row against that row's OWN
    `gate_audit_<SYMBOL>_<TF>.json`, which is exactly what the per-pair
    commands spelled out by hand. The per-pair command survives for the one
    case that needs it: a promotion that FAILED, where an operator is
    finishing a single pair rather than running the set.

    Only rows Stage 3 recorded as `certified` appear. Never a row whose
    numbers look like a pass: this module transcribes, and a section headed
    READY FOR PROMOTION is exactly where a re-derived verdict would do the
    most damage.

    The heading is the auto-promotion record's, not a guess. A configuration
    Stage 3 SEALED has been staged into the incubator and not committed, and
    presenting that as promoted is the failure this whole card is written to
    avoid.
    """
    rows = [r for r in stage3_rows(blob) if r.get("certified")]
    rows.sort(key=_stage3_sort_key)
    shown = rows[: max(0, int(max_rows))]

    auto = stage3_auto_promotion(blob)
    ran = bool(auto.get("ran"))
    commit = str(auto.get("commit") or "").strip()
    title = (f"{PROMO_DONE_TITLE} (Commit {commit})" if ran and commit
             else PROMO_DONE_TITLE if ran else PROMO_READY_TITLE)
    # Keyed by the same (symbol, timeframe, version) the orchestrator promoted
    # under, so a partially-failed auto-promotion labels each row by what
    # actually happened to it rather than by what happened to the run.
    done = {(str(d.get("symbol")), str(d.get("timeframe")),
             str(d.get("version") or "A")): d
            for d in (auto.get("promotions") or [])
            if isinstance(d, dict)}
    source = blob.get("strategy_source")

    pairs: list[str] = []
    lines: list[str] = []
    if any(not (done.get((str(r.get("symbol")), str(r.get("timeframe")),
                          str(r.get("version") or "A"))) or {}).get("promoted")
           for r in shown):
        # Said ONCE, above the rows, rather than on every line: staged and
        # promoted are different states and the difference is the whole point
        # of this section - a configuration Stage 3 sealed is in the incubator
        # directory and in nobody's git history.
        lines.append("_Sealed and STAGED by Stage 3 — nothing is committed "
                     "until the command below runs._")
        lines.append("")
    for row in shown:
        sym = str(row.get("symbol") or "?")
        tf = str(row.get("timeframe") or "?")
        ver = str(row.get("version") or "A")
        quad = str(row.get("quadrant") or row.get("target_regime")
                   or "quadrant not recorded")
        pairs.append(f"`{sym} {tf}`")
        record = done.get((sym, tf, ver))
        if record and record.get("promoted"):
            sha = str(record.get("commit") or commit or "").strip()
            state = f"promoted `{sha}`" if sha else "promoted"
        elif record:
            state = "**NOT PROMOTED**"
        else:
            state = "staged" if row.get("incubator_dir") else "not staged"
        lines.append(f"• **{sym} {tf}** V{ver} `{quad}` · PF "
                     f"**{_regime_pf_cell(row)}** (n="
                     f"{_fmt_count(row.get('oos_trade_count'))}) · {state}")
        lines.append(f"  params `{compact_params(stage3_param_text(row.get('params')))}`")
        # The blended pair, for the rows somebody is about to act on. A
        # holdout factor read alone lets an edge that collapsed from 2.40 to
        # 1.10 look like a healthy 1.10, and this is the section where that
        # reading costs something.
        lines.append(f"  blended IS {_fmt_metric(row.get('is_profit_factor'))}"
                     f" → OOS {_fmt_metric(row.get('holdout_profit_factor'))}")
        code, params = _seal_prefix(row), _seal_prefix(row,
                                                       "winning_parameters")
        if code != "--" or params != "--":
            lines.append(f"  seals code `{code}` · params `{params}`")
        if record and not record.get("promoted"):
            lines.append(f"  ⚠ {record.get('error') or 'promote.py failed'}")
            lines.append(f"  `{promotion_command(strat, row, source)}`")
        lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    return title, lines, pairs, len(rows) - len(shown)


def outstanding_promotions(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """
    The certified configurations that have NOT been committed yet.

    A configuration Stage 3 sealed is STAGED and not promoted, and one whose
    automatic promotion failed is neither - both still need the command. One
    the orchestrator committed does not, and telling a reader to run a command
    that has already run is how the same strategy gets promoted twice under
    two commits.
    """
    done = {(str(d.get("symbol")), str(d.get("timeframe")),
             str(d.get("version") or "A"))
            for d in (stage3_auto_promotion(blob).get("promotions") or [])
            if isinstance(d, dict) and d.get("promoted")}
    return [r for r in stage3_rows(blob) if r.get("certified")
            and (str(r.get("symbol")), str(r.get("timeframe")),
                 str(r.get("version") or "A")) not in done]


def promotion_footer(strat: str, blob: dict[str, Any]) -> str:
    """
    The single promotion command, as one field value, or `""` when there is
    nothing left to run.

    Its own field rather than the last lines of the bullet block, because the
    field chunker splits a long block across fields on a blank line and half a
    command is a command that runs and does something else. A field cannot be
    split.
    """
    pending = outstanding_promotions(blob)
    if not pending:
        return ("_Every certified configuration is promoted and committed; "
                "nothing is left to run._" if stage3_rows(blob)
                and any(r.get("certified") for r in stage3_rows(blob)) else "")
    return (f"Promotes the {len(pending)} configuration(s) above and nothing "
            f"else — it reads this handoff and promotes only what Gate R "
            f"certified.\n" + FENCE_OPEN
            + "\n".join(unified_promote_command(strat)) + FENCE_CLOSE)


def build_stage3_embed(strat: str, blob: dict[str, Any],
                       source: str | Path | None = None,
                       max_rows: int = STAGE3_MAX_ROWS) -> dict[str, Any]:
    """
    Stage 3's card. Pure - sends nothing, computes nothing, and every value on
    it is transcribed from the summary Stage 3 wrote.

    Both windows are on the card because a certification is a claim about two
    date ranges and is meaningless with either one missing: the in-sample
    window says what the parameters were fitted to, and the holdout says what
    they were then measured on. `Verdict` states in words that Gates 1-3 are
    advisory, because a reader who has seen this pipeline before the charter
    would otherwise read a `CERTIFIED` beside a failing Gate 1 as a bug.
    """
    rows = stage3_rows(blob)
    certified = [r for r in rows if r.get("certified")]
    audited = [r for r in rows
               if str(r.get("status") or "").upper() != NOT_AUDITED]
    is_window = blob.get("in_sample") or {}
    ho_window = blob.get("holdout") or {}
    rule = blob.get("certification_rule") or {}
    table, hidden, legend = format_stage3_table(rows, max_rows, rule)
    coverage = blob.get("coverage") or {}
    # Every timeframe the summary indexes, not only the one the last Stage 3
    # invocation certified. A card headed `15m` above a table carrying 5m rows
    # describes neither, and that is exactly what a multi-timeframe campaign
    # produced before the summary merged.
    tfs = stage3_timeframes(blob)
    tf = " · ".join(f"`{t}`" for t in tfs) if tfs else None

    description = [
        f"**In-sample** `{is_window.get('start') or 'not recorded'} → "
        f"{is_window.get('end') or 'not recorded'}` "
        f"· fitted at Stage 2, evidence only",
        f"**Out-of-sample holdout** `{ho_window.get('start') or 'not recorded'}"
        f" → {ho_window.get('end') or 'present'}` · the verdict",
        f"**Verdict** Gate R — profit factor `>= "
        f"{_fmt_metric(rule.get('min_profit_factor'))}` over "
        f"`{_fmt_count(rule.get('min_trades'))}`+ trades INSIDE the target "
        f"quadrant. Gates 1–3 are reported as evidence and cannot fail a "
        f"certification.",
        "```text",
        table if table.strip() else STAGE3_NO_ROWS_NOTE,
        "```",
        # What the two number columns ARE. They are Gate R's own quadrant
        # numbers and not the blended sample, and an unlabelled profit factor
        # under a regime-gated verdict is the one value on this card a reader
        # must not have to guess at.
        "`PF` `N` — Gate R's factor and trades INSIDE the target quadrant, "
        "on the holdout. Blended IS/OOS: on the certified rows below.",
        # The table is filtered and says so. The counts in the fields below
        # are NOT: they describe every configuration the run covered, so a
        # short table can never read as a short certification run.
        "_The table lists CERTIFIED configurations only — the counts below "
        "cover every configuration audited, and each one's verdict is in its "
        "own per-pair gate audit file._",
    ]
    promo_title, promo_lines, promo_pairs, promo_hidden = \
        format_stage3_promotions(strat, blob)
    if promo_pairs:
        # The passing pairs in one line, above the fold. The block below
        # carries the parameters and the command; this is the answer to "did
        # anything certify", which is the question the card is opened with.
        description.append(f"**{promo_title}** " + " · ".join(promo_pairs)
                           + (f" _(+{promo_hidden} more)_" if promo_hidden
                              else ""))
    if legend:
        description.append("**Target regimes** " + " · ".join(
            f"`{q}` {legend[q]}" for q in sorted(legend)))
    if hidden:
        description.append(
            f"_{hidden} further CERTIFIED configuration(s) are not shown — "
            f"the full summary is in the handoff._")

    text = "\n".join(description)
    if len(text) > MAX_EMBED_DESCRIPTION:
        # Trim the TABLE and never the header lines: without the two windows
        # and the verdict rule the numbers underneath are unlabelled.
        keep = MAX_EMBED_DESCRIPTION - 64
        text = text[:keep] + "\n```\n_truncated — see the handoff._"

    fields = [
        {"name": "Strategy", "value": f"`{strat}`", "inline": True},
        # Named in the singular because a single-timeframe campaign is still
        # the common case and the value reads as one; it carries every
        # timeframe the summary indexes, separated.
        {"name": "Timeframe", "value": tf or "not recorded",
         "inline": True},
        {"name": "Configurations", "value": str(len(rows)), "inline": True},
        # "Certified" is Gate R and nothing else. Counted from the handoff's
        # own `certified` flag rather than re-derived from the numbers on the
        # card, so this can never announce a pass Stage 3 did not record.
        {"name": "Certified → Incubator", "value": str(len(certified)),
         "inline": True},
        {"name": "Audited", "value": f"{len(audited)}/{len(rows)}",
         "inline": True},
        {"name": "Pruning",
         "value": ("none on an aggregate metric; no prop-firm rule applied"
                   if coverage.get("complete") is not False
                   else f"none by design; "
                        f"{coverage.get('errors', 0)} error(s), "
                        f"{coverage.get('skipped', 0)} skipped"),
         "inline": False},
    ]

    embed = {
        "title": f"\U0001F512 Stage 3 · Gate Audit & Certification: {strat}",
        "description": text,
        # Teal when something was certified, amber when nothing was. Amber
        # rather than red for the same reason as the other cards: a holdout
        # that certified nothing is a result to read, not a crash.
        "color": TEAL if certified else AMBER,
        "fields": fields,
        "footer": {"text": "backtest/discord_reporter.py · Stage 3 gate audit "
                           "· values as recorded by audit_gates.py, not "
                           "recomputed"},
    }

    # The promotion block goes in LAST, on whatever the rest of the card left
    # of Discord's 6000 characters - sized against the FINISHED embed rather
    # than a constant, exactly as the Stage 2 parameter block is, because the
    # description holding the table is most of the budget.
    handoff = {"name": "Handoff", "value": _fmt_report(str(source or "")),
               "inline": False}
    footer = promotion_footer(strat, blob)
    budget = (MAX_EMBED_TOTAL - _embed_size(embed)
              - len(handoff["name"]) - len(handoff["value"])
              - (len(PROMO_COMMAND_FIELD) + len(footer) if footer else 0))
    # It is the one part of this card somebody ACTS on, which is why it is
    # budgeted ahead of everything optional and why the command that promotes
    # the set is reserved out of the budget before the bullets are chunked: a
    # card that listed three certified configurations and dropped the command
    # sends a reader into the handoff to reconstruct one against a directory
    # holding an audit per timeframe, and picking the wrong file promotes a
    # verdict about different bars.
    promo_fields, promo_dropped = _fenced_fields(
        promo_lines, budget, promo_title, f"{promo_title} (cont.)",
        STAGE3_PROMO_MAX_FIELDS, fence=False)
    if promo_fields and promo_dropped:
        promo_fields[-1]["value"] += (
            f"\n… {promo_dropped} further line(s) — see the handoff.")
    if footer:
        promo_fields.append({"name": PROMO_COMMAND_FIELD, "value": footer,
                             "inline": False})
    embed["fields"] = fields + promo_fields + [handoff]
    return embed


def _fenced_fields(lines: list[str], budget: int, name: str, cont: str,
                   max_fields: int, fence: bool = True
                   ) -> tuple[list[dict], int]:
    """
    A block of lines as Discord fields, and how many did not fit.

    One field per chunk under `MAX_FIELD_VALUE`, at most `max_fields` of them,
    and never past `budget`. A line that does not fit is COUNTED rather than
    truncated - half a SHA-256 is not a shorter checksum but a different string
    that looks like one, and half a promote command is a command that runs and
    does something else.

    `fence=False` renders the lines as MARKDOWN instead of wrapping them in a
    code fence. Stage 3's promotion block needs it: bold, backticks and
    bullets render in a field value and do not inside a fence, and the block is
    a list of configurations rather than a fixed-width table. The chunking is
    otherwise identical, including the preference for cutting on a blank line.
    """
    wrap = (len(FENCE_OPEN) + len(FENCE_CLOSE)) if fence else 0
    if not lines or budget <= wrap + NOTE_RESERVE:
        return [], len(lines)

    fields: list[dict] = []
    spent = 0
    i = 0
    while i < len(lines) and len(fields) < max_fields:
        chunk: list[str] = []
        size = wrap
        while i < len(lines):
            need = len(lines[i]) + (1 if chunk else 0)
            if size + need > MAX_FIELD_VALUE:
                break
            label = name if not fields else cont
            if spent + size + need + len(label) + NOTE_RESERVE > budget:
                break
            chunk.append(lines[i])
            size += need
            i += 1
        if not chunk:
            break
        # Prefer to end a field on a BLANK line when there is more to come.
        # The promotion block is a stack of records separated by blank lines,
        # and a field boundary through the middle of one splits a promote
        # command across two Discord fields - where the half a reader copies
        # is a command that runs and does something else. A block with no
        # blank line in it (a fixed-width table) is chunked straight through.
        if i < len(lines) and "" in chunk[:-1]:
            cut = len(chunk) - 1 - chunk[::-1].index("")
            if cut > 0:
                i -= len(chunk) - cut
                chunk = chunk[:cut]
        label = name if not fields else cont
        # The blank line a chunk was cut on belongs to the break, not to the
        # next field: rendered, it is an empty first row above the record.
        while chunk and not chunk[0]:
            chunk.pop(0)
        if not chunk:
            continue
        body = "\n".join(chunk)
        fields.append({"name": label,
                       "value": (FENCE_OPEN + body + FENCE_CLOSE) if fence
                       else body,
                       "inline": False})
        spent += size + len(label)
    return fields, len(lines) - i


# --------------------------------------------------------------------------
# Stage 4 · full-lifecycle verification
# --------------------------------------------------------------------------

def default_verify_dir(strat: str, out_dir: str | None = None) -> Path:
    """
    The NEWEST `verify_<stamp>/` directory under the strategy's pipeline dir.

    Stage 4 writes into a directory stamped with the run's UTC time precisely
    so a re-run never overwrites the evidence an earlier decision was made on,
    which means there is no stable path to default to. Sorted by NAME rather
    than by mtime: the name carries the stamp Stage 4 wrote it under, and an
    mtime moves when a directory is copied off the NFS mount or a report is
    regenerated inside it.

    Whichever directory this resolves to is PRINTED on the card and in the
    success line, so the choice is visible rather than silent - that is the
    whole reason it is allowed to be a default at all.
    """
    base = pipeline_dir(strat, out_dir)
    runs = sorted((p for p in base.glob("verify_*") if p.is_dir()),
                  key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError(
            f"no verify_<stamp>/ directory in {base}. Run stage 4 first "
            f"(python3 backtest/verify_full.py --strat {strat} ...), or name "
            f"the directory with --artifacts.")
    return runs[-1]


def _finite(value: Any) -> float | None:
    """
    A float, or None for anything that is not a measurement.

    NaN is the shape a missing metric arrives in: the engine writes NaN for a
    ratio it could not compute and `json.dump` round-trips it as a float, so an
    unguarded format call renders `nan%` where the truthful cell is `--`.
    """
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _fmt_money(value: Any) -> str:
    """
    One money cell, compact, so the table holds `STAGE4_TABLE_WIDTH`.

    `123.5k` rather than `123,456.78`: net P&L over sixteen years and an alpha
    score (net P&L x profit factor) are both wide enough on their own to wrap
    this table on a phone, and a wrapped fixed-width table is worse than a
    coarse one. The exact figures are in `dual_metrics_<SYMBOL>.json` and on
    the tear sheet, and the card names the directory holding both.
    """
    number = _finite(value)
    if number is None:
        return "--"
    size = abs(number)
    if size >= 1_000_000:
        return f"{number / 1_000_000:,.2f}M"
    if size >= 10_000:
        return f"{number / 1_000:,.1f}k"
    return f"{number:,.0f}"


def _fmt_pct(value: Any, decimals: int = 1) -> str:
    """One percentage cell. `--` for a metric nobody measured, never `0.0%`."""
    number = _finite(value)
    return "--" if number is None else f"{number:.{decimals}f}%"


def friction_share(metrics: dict[str, Any]) -> float | None:
    """
    Costs as a percentage of GROSS profit, or None when that is undefined.

    The one derived number on this card, and it is a division of two figures
    the run already recorded (`total_costs`, `gross_pnl`) rather than a metric
    re-scored from bars. It is here because it is the number that decides
    whether an edge is real - an edge handing 85% of its gross to the broker
    dies on one extra tick of slippage while every ratio above it still reads
    fine - and `dual_metrics.json` carries the two totals but not their ratio.

    **The undefined rule is `backtest/verify_full.cost_drag`'s, exactly**: a
    non-positive gross P&L has no profit for costs to be a share of, so the
    answer is None rather than 0.0. Printing 0% for a strategy that lost money
    gross reads as a run that cost nothing, which is the opposite of what
    happened.
    """
    gross = _finite(metrics.get("gross_pnl"))
    costs = _finite(metrics.get("total_costs"))
    if gross is None or costs is None or gross <= 0:
        return None
    return 100.0 * costs / gross


def top_regime_alpha(artifacts: Path, symbol: str, tf: str) -> dict[str, Any] | None:
    """
    The designated quadrant and its ALPHA SCORE, read from Stage 4's own
    regime profile - or None when this run wrote none.

    The score is `net P&L x profit factor` inside one quadrant, which is what
    `backtest/profiler.designate` ranks on; it is transcribed, never
    recomputed. Nothing else on this card knows how a home regime is chosen and
    nothing here re-derives one, because a quadrant named by a notifier is a
    live-trading instruction nobody certified.

    Only the UNSUFFIXED `regime_profile_<SYMBOL>_<TF>.json` is read. The
    suffixed `_version_a` / `_version_b` files in the same directory are Stage
    1's, profiled over the charter window alone - printing one of those under a
    lifecycle heading would attach an in-sample score to a whole-lifecycle row
    with every column still lining up.

    Looked up in the artifacts directory first and then in its PARENT, because
    `verify_full.py` passes the profiler `art_dir.parent` - the stage's own
    pipeline directory - so the live supervisor can find the latest profile
    without being told a timestamp.
    """
    name = REGIME_PROFILE_FILE.format(symbol=symbol, tf=tf)
    for candidate in (artifacts / name, artifacts.parent / name):
        try:
            blob = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        regime = str(blob.get("optimal_regime") or NO_HOME_REGIME)
        designated = regime != NO_HOME_REGIME
        return {
            "regime": regime if designated else None,
            "quadrant": blob.get("optimal_quadrant") if designated else None,
            "score": blob.get("optimal_score") if designated else None,
            "profit_factor": (blob.get("optimal_profit_factor")
                              if designated else None),
            "trade_count": (blob.get("optimal_trade_count")
                            if designated else None),
            "designated": designated,
            "source": str(candidate),
        }
    return None


def _symbol_from_name(path: Path) -> str:
    """`dual_metrics_NQ.json` -> `NQ`. The fallback when a file has no meta."""
    stem = path.stem
    return stem[len(DUAL_METRICS_PREFIX):] if stem.startswith(
        DUAL_METRICS_PREFIX) else stem


def _check_strategy(blob: dict[str, Any], path: Path,
                    strat: str | None) -> None:
    """
    Refuse another strategy's snapshot, the way `pipeline.read_stage` refuses
    another strategy's handoff.

    A `dual_metrics_<SYMBOL>.json` is not written through `write_stage` and
    carries no stage number, so that check cannot be delegated - but the
    failure it prevents is the same one and is worse here: a card posts one
    strategy's lifecycle numbers under another's name, and a Discord card is
    exactly the artifact nobody cross-checks.

    `strat` is the name the OPERATOR typed and `meta.strategy` is the MODULE's,
    and for a promoted strategy those differ by construction:
    `approved_incubator/<strat>/strat.py` is module `strat` under directory
    name `<strat>`. That one spelling is accepted; anything else is refused.
    """
    recorded = str((blob.get("meta") or {}).get("strategy") or "").strip()
    wanted = str(strat or "").strip()
    if not wanted or not recorded:
        return
    if _owns(recorded, wanted):
        return
    raise ValueError(
        f"{path.name} was written for strategy {recorded!r}, not {wanted!r}. "
        f"Posting it under --strat {wanted} would announce one strategy's "
        f"lifecycle under another's name.")


def stage4_rows(artifacts: str | Path,
                strat: str | None = None) -> list[dict[str, Any]]:
    """
    One row per `dual_metrics_<SYMBOL>.json` in the artifacts directory.

    Every file in the directory becomes a row, INCLUDING one that could not be
    read: a card shorter than the run it announces reads as a shorter run, and
    "the snapshot is corrupt" and "this contract was never verified" are fixed
    by completely different work. An unreadable row carries `error` and renders
    every metric as `--`; it never renders as a contract that measured zero.

    Ordered by (symbol, timeframe) and NOT ranked. Stage 4 selects nothing -
    it is one full-lifecycle run per contract, no gate and no verdict - and
    sorting by CAGR or net P&L would put a leaderboard's shape on a stage that
    produced no leaderboard.
    """
    directory = Path(artifacts)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory. Name the "
                                f"stage 4 run's artifacts directory with "
                                f"--artifacts.")
    files = sorted(directory.glob(DUAL_METRICS_GLOB))
    if not files:
        # An empty screen is a result; an artifacts directory with no metrics
        # snapshot in it is the wrong directory. Refused rather than posted as
        # an empty card, because the numbers a card would need are not missing
        # - they were never looked for here.
        raise FileNotFoundError(
            f"no {DUAL_METRICS_GLOB} in {directory}. Stage 4 writes them into "
            f"its verify_<stamp>/ directory; name that one with --artifacts.")

    rows: list[dict[str, Any]] = []
    for path in files:
        symbol = _symbol_from_name(path)
        try:
            blob = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            rows.append({"symbol": symbol, "tf": None, "source": str(path),
                         "error": f"{type(exc).__name__}: {exc}"})
            continue

        _check_strategy(blob, path, strat)
        meta = blob.get("meta") or {}
        version_a = blob.get("version_a") or {}
        metrics = version_a.get("metrics") or {}
        symbol = str(meta.get("symbol") or symbol)
        tf = str(meta.get("timeframe") or "") or None
        regime = top_regime_alpha(directory, symbol, tf) if tf else None

        rows.append({
            "symbol": symbol,
            "tf": tf,
            # CAGR, net P&L and the trade count as the run recorded them. The
            # engine computed `annualized_return_pct` off its DAILY equity
            # curve, which is what makes a 15m run and a 1d run comparable;
            # recomputing it from anything on this card would not be.
            "cagr_pct": metrics.get("annualized_return_pct"),
            "net_pnl": metrics.get("total_pnl"),
            "trades": metrics.get("trade_count"),
            "gross_pnl": metrics.get("gross_pnl"),
            "total_costs": metrics.get("total_costs"),
            "friction_pct": friction_share(metrics),
            "regime": regime,
            "window": {"start": meta.get("start"), "end": meta.get("end")},
            # Whether the snapshot holds a Version B at all. None is "not run"
            # and must not collapse into a B that ran and scored nothing.
            "version_b": blob.get("version_b") is not None,
            "source": str(path),
            "error": None,
        })
    return rows


def format_stage4_table(rows: list[dict[str, Any]],
                        max_rows: int = STAGE4_MAX_ROWS
                        ) -> tuple[str, int, dict[str, str]]:
    """
    The lifecycle table as one fixed-width block, plus the quadrant legend.

    Returns `(text, hidden, legend)`. `hidden` is how many contracts did not
    fit and is printed on the card by the caller.

    The QD column carries the `Q1`..`Q4` id the profiler recorded and the
    legend maps only the ids that appear, built FROM the rows - no short
    spelling of a regime name lives in this module, for the reason the Stage 1
    table gives: a second one would be free to disagree with `mdlib.regimes`,
    and a card naming the wrong environment is caught only in live trading.
    """
    header = ["SYM", "TF", "CAGR", "NET P&L", "TRD", "FRIC", "QD", "ALPHA"]
    body: list[list[str]] = []
    legend: dict[str, str] = {}

    ordered = sorted(rows, key=lambda r: (str(r.get("symbol") or ""),
                                          str(r.get("tf") or "")))
    shown = ordered[: max(0, int(max_rows))]
    for row in shown:
        regime = row.get("regime") or {}
        quad, name = regime.get("quadrant"), regime.get("regime")
        if quad and name:
            legend[str(quad)] = str(name)
        body.append([
            str(row.get("symbol") or "?"),
            str(row.get("tf") or "--"),
            _fmt_pct(row.get("cagr_pct"), decimals=2),
            _fmt_money(row.get("net_pnl")),
            _fmt_count(row.get("trades")),
            _fmt_pct(row.get("friction_pct")),
            str(quad) if quad else "--",
            _fmt_money(regime.get("score")),
        ])

    widths = [max(len(header[i]), *(len(r[i]) for r in body)) if body
              else len(header[i]) for i in range(len(header))]
    align = ["<", "<", ">", ">", ">", ">", "<", ">"]

    def line(cells: list[str]) -> str:
        return "  ".join(format(c, f"{align[i]}{widths[i]}")
                         for i, c in enumerate(cells)).rstrip()

    out = [line(header), line(["-" * w for w in widths])]
    out.extend(line(r) for r in body)
    return "\n".join(out), len(ordered) - len(shown), legend


def stage4_regime_note(rows: list[dict[str, Any]]) -> str:
    """
    What the ALPHA column is, and why a `--` in it is one of two things.

    A contract with NO profile and a contract whose profile designated no home
    regime both print `--`, and they are fixed by different work: the first is
    a profiler that did not run (or wrote elsewhere), the second is a strategy
    with no environment on these bars. Counted separately rather than left as
    one dash.
    """
    missing = sum(1 for r in rows if not r.get("error") and not r.get("regime"))
    undesignated = sum(1 for r in rows
                       if (r.get("regime") or {}).get("designated") is False)
    note = ("_ALPHA is the designated quadrant's alpha score (net P&L × PF) "
            "from `regime_profile_<SYM>_<TF>.json`, as the profiler ranked "
            "it._")
    if missing:
        note += f" _{missing} contract(s) have no profile._"
    if undesignated:
        note += (f" _{undesignated} designated no home quadrant — no quadrant "
                 f"cleared the designation bars._")
    return note


def stage4_window(rows: list[dict[str, Any]]) -> str:
    """
    The lifecycle window, or a statement that it varies.

    Every contract in one Stage 4 run is asked for the same `--start`/`--end`,
    but the bars each one HAS are its own - the window recorded here is the
    first and last bar the run actually saw. Where they differ the card says
    so rather than printing one contract's span above another's numbers.
    """
    spans = {(r["window"].get("start"), r["window"].get("end"))
             for r in rows if not r.get("error") and r.get("window")}
    spans = {s for s in spans if s[0] and s[1]}
    if not spans:
        return "not recorded"
    if len(spans) > 1:
        return "varies by contract — see the tear sheets"
    start, end = spans.pop()
    return f"{start} → {end}"


def build_stage4_embed(strat: str, rows: list[dict[str, Any]],
                       source: str | Path | None = None,
                       max_rows: int = STAGE4_MAX_ROWS) -> dict[str, Any]:
    """
    Stage 4's card. Pure - sends nothing, reads nothing, and every value on it
    except the friction share is transcribed from the metrics snapshot Stage 4
    wrote (see `friction_share` for the one division and the rule it inherits).

    It carries no gate table and no verdict, because Stage 4 produces neither:
    the window contains the holdout Stage 3 spent, so a badge here would be a
    certification of contaminated bars wearing the same shape as a real one.
    What is on it is the lifecycle question set - what it compounded at, what
    it made, how often it traded, what the broker took, and which environment
    the alpha actually came from.
    """
    table, hidden, legend = format_stage4_table(rows, max_rows)
    readable = [r for r in rows if not r.get("error")]
    broken = len(rows) - len(readable)
    with_b = sum(1 for r in readable if r.get("version_b"))

    description = [
        f"**Lifecycle window** `{stage4_window(readable)}`",
        STAGE4_NOT_CERTIFICATION,
        f"**Metrics** {STAGE4_VERSION_NOTE} — the version Stage 4 profiles, "
        f"drags the costs of, and writes the trade log for",
        "```text",
        table if table.strip() else "no contract was verified",
        "```",
        # FRIC is the number a reader acts on and it is the one that is
        # meaningless unlabelled: costs as a share of GROSS profit, undefined
        # (`--`) where there was no gross profit for them to be a share of.
        "_FRIC is total costs as a share of GROSS profit — `--` where gross "
        "P&L was not positive, which is not the same as costing nothing._",
        stage4_regime_note(readable),
    ]
    if legend:
        description.append("**Regimes** " + " · ".join(
            f"`{q}` {legend[q]}" for q in sorted(legend)))
    if hidden:
        description.append(
            f"_{hidden} further contract(s) are not shown — every snapshot is "
            f"in the artifacts directory below._")
    if broken:
        description.append(
            f"_{broken} snapshot(s) could not be read and carry no metrics — "
            f"that is a corrupt file, not a contract that traded nothing._")

    text = "\n".join(description)
    if len(text) > MAX_EMBED_DESCRIPTION:
        # Trim the TABLE and never the header lines: without the window and
        # the not-a-certification statement the numbers underneath are
        # unlabelled, which is the one way this card can mislead.
        keep = MAX_EMBED_DESCRIPTION - 64
        text = text[:keep] + "\n```\n_truncated — see the artifacts._"

    fields = [
        {"name": "Contracts", "value": str(len(rows)), "inline": True},
        {"name": "Verified", "value": str(len(readable)), "inline": True},
        {"name": "Unreadable", "value": str(broken), "inline": True},
        # NOT RUN rather than 0: a Version B that was never asked for and one
        # that ran and filtered nothing are different runs, and `--ml` is
        # opt-in on this stage.
        {"name": "Version B",
         "value": (f"{with_b}/{len(readable)} snapshot(s)" if with_b
                   else "NOT RUN"),
         "inline": True},
        {"name": "Certifies", "value": "nothing — Stage 3 holds the verdict",
         "inline": True},
        {"name": "Artifacts", "value": _fmt_report(str(source or "")),
         "inline": False},
    ]

    return {
        "title": f"\U0001F4C8 Stage 4 · Lifecycle Verification & Tear Sheet: {strat}",
        "description": text,
        # Graphite when something was verified, amber when nothing could be
        # read - amber rather than red for the reason Stage 1 gives: a stage
        # that produced no rows is a result to look at, not a crash.
        "color": GRAPHITE if readable else AMBER,
        "fields": fields,
        "footer": {"text": "backtest/discord_reporter.py · Stage 4 full "
                           "lifecycle · values as recorded by verify_full.py, "
                           "not recomputed · NOT a certification"},
    }


def build_payload(embed: dict[str, Any]) -> dict[str, Any]:
    return {"embeds": [embed]}


def _embed_size(embed: dict[str, Any]) -> int:
    """
    Total characters Discord counts against the 6000 embed limit.

    The DESCRIPTION counts too, and on the Stage 1 card it is most of the
    embed. Omitting it here would let a 40-row leaderboard sail past a check
    that reported 300 characters and be rejected with a 400 nobody reads.
    """
    total = (len(embed.get("title", ""))
             + len(embed.get("description", ""))
             + len(embed.get("footer", {}).get("text", "")))
    for field in embed.get("fields", []):
        total += len(field["name"]) + len(field["value"])
    return total


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

def post_embed(webhook: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    POST the payload. Never raises: the outcome is RETURNED, because a notifier
    that throws loses the record of what it attempted.

    Returns {ok, http_status, error}. `http_status` is None when the request
    never reached a server.
    """
    try:
        response = requests.post(
            webhook,
            json=payload,
            timeout=POST_TIMEOUT_SECONDS,
            headers={"Content-Type": "application/json"},
        )
    except requests.exceptions.RequestException as exc:
        # Never echo the URL: it is a credential, and requests' own exception
        # text embeds the FULL url including the token - so the exception
        # message is deliberately not forwarded. The type and the host are
        # enough to tell a typo'd domain from a dead network from a timeout.
        host = urlparse(webhook).netloc or "<unparseable url>"
        return {
            "ok": False,
            "http_status": None,
            "error": f"{type(exc).__name__} contacting {host}",
        }

    if response.status_code in SUCCESS_STATUS:
        return {"ok": True, "http_status": response.status_code, "error": None}

    body = (response.text or "").strip()
    if len(body) > 500:
        body = body[:500] + "..."
    return {
        "ok": False,
        "http_status": response.status_code,
        "error": body or response.reason or "no response body",
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discord_reporter.py",
        description="Post a pipeline card to a Discord webhook: a promotion "
                    "scorecard (--mode promotion), Stage 1's regime-firewall "
                    "leaderboard (--stage 1), Stage 2's parameter "
                    "optimization summary (--stage 2), Stage 3's gate "
                    "audit and certification (--stage 3), or Stage 4's "
                    "full-lifecycle metrics (--stage 4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Values are printed as supplied - nothing here recomputes a metric.\n"
            "--webhook may be omitted when any of "
            + ", ".join("$" + name for name in DISCORD_WEBHOOK_VARS)
            + " is set, in the environment or in .env.\n"
            "\n"
            "  --stage 5          --strat X   (everything else auto-resolved\n"
            "                     from approved_incubator/X/, overridable with\n"
            "                     --symbol --tf --pf --win --dd --regime\n"
            "                     --report, --audit-file and --metrics)\n"
            "  --stage 1          --strat X [--survivors <surviving_assets.json>]\n"
            "  --stage 2          --strat X [--summary <stage2_summary.json>]\n"
            "  --stage 3          --strat X [--audit <stage3_audit_summary.json\n"
            "                     | gate_audit_<SYMBOL>_<TF>.json>]\n"
            "  --stage 4          --strat X [--artifacts <verify_<stamp>/>]"
        ),
    )
    # Resolved in main() rather than defaulted here, so the NAME that supplied
    # it can be reported. A default computed at parse time cannot say whether
    # the URL came from the flag, from the environment or from .env, and
    # "which variable is this posting with" is the question an operator with
    # two channels configured actually has.
    parser.add_argument("--webhook", default=None,
                        help="Discord webhook URL (default: the first of "
                             + ", ".join("$" + n for n in DISCORD_WEBHOOK_VARS)
                             + " that is set)")
    # `--mode` and `--stage` are two spellings of one choice, and they share a
    # dest so they cannot disagree. A card labelled Stage 1 that was built by
    # the promotion path would announce a screen as a promotion.
    parser.add_argument("--mode", dest="mode", default=None,
                        choices=["promotion", "baseline", "scan", "audit",
                                 "verify"],
                        help="promotion (default): the Stage 5 scorecard. "
                             "baseline: Stage 1's regime-firewall leaderboard. "
                             "scan: Stage 2's parameter optimization summary. "
                             "audit: Stage 3's gate audit and certification. "
                             "verify: Stage 4's full-lifecycle metrics.")
    parser.add_argument("--stage", dest="stage", default=None,
                        choices=["1", "2", "3", "4", "5"],
                        help="1 == --mode baseline, 2 == --mode scan, "
                             "3 == --mode audit, 4 == --mode verify, "
                             "5 == --mode promotion")
    parser.add_argument("--strat", required=True, help="strategy name, e.g. sma_momentum_crossover")
    parser.add_argument("--symbol", default="", help="promotion mode: the contract the decision rests on, e.g. NQ")
    parser.add_argument("--tf", default="", help="promotion mode: timeframe, e.g. 15m")
    parser.add_argument("--survivors", default=None,
                        help="baseline mode: path to surviving_assets.json "
                             "(default: <BT_ARTIFACTS>/pipeline/<strat>/"
                             f"{SURVIVORS_FILE})")
    parser.add_argument("--summary", default=None,
                        help=f"scan mode: path to {STAGE2_SUMMARY_FILE}. "
                             f"audit mode: path to {STAGE3_SUMMARY_FILE} - "
                             f"the campaign index ONLY; a single per-pair "
                             f"gate audit is passed with --audit (default: "
                             f"<BT_ARTIFACTS>/pipeline/<strat>/<that file>)")
    parser.add_argument("--audit", default=None,
                        help=f"audit mode: {STAGE3_SUMMARY_FILE}, or ONE "
                             f"{GATE_AUDIT_FILE.format(symbol='<SYMBOL>_<TF>')}"
                             f" - whichever it is told, read as what it is. "
                             f"Default: {STAGE3_SUMMARY_FILE} under "
                             f"<BT_ARTIFACTS>/pipeline/<strat>/, else every "
                             f"per-pair gate audit in that directory")
    parser.add_argument("--artifacts", default=None,
                        help="verify mode: the Stage 4 run's artifacts "
                             "directory, holding its "
                             f"{DUAL_METRICS_GLOB} snapshots (default: the "
                             "NEWEST verify_<stamp>/ under "
                             "<BT_ARTIFACTS>/pipeline/<strat>/, named on the "
                             "card either way)")
    parser.add_argument("--out-dir", default=None,
                        help="baseline, scan, audit and verify modes: "
                             "override the pipeline directory the handoff - "
                             "or, in verify mode, the verify_<stamp>/ run - "
                             "is looked up in")
    parser.add_argument("--max-rows", type=int, default=None,
                        help=f"leaderboard rows on the card. Each mode keeps "
                             f"its OWN default, because the rows are different "
                             f"widths: baseline {STAGE1_MAX_ROWS}, scan "
                             f"{STAGE2_MAX_ROWS} (each row carries a parameter "
                             f"set), audit {STAGE3_MAX_ROWS}, verify "
                             f"{STAGE4_MAX_ROWS}. Whatever does "
                             f"not fit is COUNTED on the card, never dropped "
                             f"in silence.")
    parser.add_argument("--pf", default="", help="out-of-sample profit factor, or a token like 'NOT EVALUATED'")
    parser.add_argument("--win", default="",
                        help="win rate in PERCENT (52.0), or a token like "
                             "'NOT EVALUATED'. Default: Gate R's own quadrant "
                             "on the holdout, then the blended holdout, then "
                             "the metrics snapshot - the card names which")
    parser.add_argument("--dd", default="", help="max drawdown in percent, or a token like 'NOT EVALUATED'")
    parser.add_argument("--regime", default="", help="certified regime, e.g. 'High-Vol/Trending'")
    parser.add_argument("--report", default="", help="artifact URL or path to the tear sheet")
    # The two files a promotion is made from, spelled the way `promote.py`
    # spells them. They are aliases in the sense that matters: the command an
    # operator already has in their shell history from Stage 5 now runs here
    # unchanged instead of dying on an unrecognised argument, which is the
    # failure that sends somebody to retype four numbers by hand.
    parser.add_argument("--audit-file", dest="audit_file", default=None,
                        help="promotion mode: the Stage 3 certification "
                             "(gate_audit_<SYMBOL>_<TF>.json) to read the "
                             "contract, Gate R's out-of-sample profit factor, "
                             "the certified quadrant and the holdout drawdown "
                             "from (default: the one meta.json cites)")
    parser.add_argument("--metrics", dest="metrics", default=None,
                        help="promotion mode: the locked "
                             f"{PROMOTED_METRICS_FILE} to read the contract, "
                             "the drawdown and the tear sheet path from "
                             "(default: the promotion's own copy). It supplies "
                             "no profit factor: its window contains the "
                             "holdout, and the card's is Gate R's")
    parser.add_argument("--incubator", default=None,
                        help="promotion mode: override the directory "
                             f"{PROMOTED_META_FILE} and "
                             f"{PROMOTED_METRICS_FILE} are looked up under "
                             "(default: strategies/approved_incubator/<strat>/)")
    parser.add_argument("--portfolios", default=None,
                        help="promotion mode: the routing table the portfolio "
                             "membership is read from (default: "
                             f"{PORTFOLIO_CONFIG_FILE}). Read only - this "
                             "never allocates anything")
    parser.add_argument("--force", action="store_true",
                        help=("post even if an identical payload went out in "
                              f"the last {int(DUPLICATE_WINDOW_SECONDS)}s"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the payload and send nothing")
    return parser


def resolve_mode(mode: str | None, stage: str | None) -> str:
    """
    One mode from the two flags, or a refusal.

    `--mode baseline --stage 5` is not a typo worth guessing at: one of the two
    is what the operator meant and picking either silently posts the wrong
    card. Neither flag given is `promotion`, which is what this script was
    before the Stage 1 mode existed.
    """
    from_stage = {"1": "baseline", "2": "scan", "3": "audit",
                  "4": "verify", "5": "promotion"}.get(stage or "")
    if mode and from_stage and mode != from_stage:
        raise ValueError(f"--mode {mode} and --stage {stage} disagree "
                         f"(--stage {stage} means --mode {from_stage}).")
    return mode or from_stage or "promotion"


# The flags that only mean something on the promotion card. Refused elsewhere
# rather than ignored: `--stage 3 --metrics <file>` parses cleanly, changes
# nothing, and posts a card built from an entirely different file - which is
# the shape of every silently-inert flag this repository has had to fix.
PROMOTION_ONLY_FLAGS = (("--audit-file", "audit_file"),
                        ("--metrics", "metrics"),
                        ("--incubator", "incubator"),
                        ("--portfolios", "portfolios"))


def _build_card(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """The embed and the one-line summary its success message prints."""
    mode = resolve_mode(args.mode, args.stage)

    if mode != "promotion":
        stray = [flag for flag, dest in PROMOTION_ONLY_FLAGS
                 if getattr(args, dest, None)]
        if stray:
            raise ValueError(
                f"{', '.join(stray)}: promotion-mode flag(s) that would change "
                f"nothing on the {mode} card. Stage 3's summary is named with "
                f"--audit, Stage 4's run directory with --artifacts.")

    if mode == "baseline":
        path = Path(args.survivors) if args.survivors else \
            default_survivors_path(args.strat, args.out_dir)
        blob = load_stage1(path, args.strat)
        # `--max-rows` defaults to None so each card keeps its OWN row cap: a
        # Stage 2 row carries a parameter set and is roughly twice as wide, and
        # one shared default would either waste the Stage 1 card's description
        # or overflow the Stage 2 one.
        embed = build_stage1_embed(args.strat, blob, source=path,
                                   max_rows=(args.max_rows
                                             if args.max_rows is not None
                                             else STAGE1_MAX_ROWS))
        rows = stage1_rows(blob)
        kept = sum(1 for r in rows
                   if str(r.get("status") or "").upper() == PROMOTED)
        return embed, (f"Stage 1 screen '{args.strat}' "
                       f"({kept}/{len(rows)} promoted)")

    if mode == "scan":
        path = Path(args.summary) if args.summary else \
            default_scan_summary_path(args.strat, args.out_dir)
        blob = load_stage2(path, args.strat)
        embed = build_stage2_embed(args.strat, blob, source=path,
                                   max_rows=(args.max_rows
                                             if args.max_rows is not None
                                             else STAGE2_MAX_ROWS))
        rows = stage2_rows(blob)
        done = sum(1 for r in rows
                   if str(r.get("status") or "").upper() == OPTIMIZED)
        return embed, (f"Stage 2 optimization '{args.strat}' "
                       f"({done}/{len(rows)} optimised)")

    if mode == "audit":
        # EITHER shape, and the file named by hand always wins - see
        # `resolve_stage3_input`. `--audit` takes the campaign summary or a
        # single `gate_audit_<SYMBOL>_<TF>.json`; `--summary` takes the
        # summary alone; with neither, the index is preferred and the per-pair
        # audits are the fallback. Whatever it resolved to is named on the
        # card and in the success line, so a defaulted choice is never silent.
        blob, path, what = resolve_stage3_input(
            args.strat, args.audit, args.summary, args.out_dir)
        embed = build_stage3_embed(args.strat, blob, source=path,
                                   max_rows=(args.max_rows
                                             if args.max_rows is not None
                                             else STAGE3_MAX_ROWS))
        rows = stage3_rows(blob)
        # `certified` is Stage 3's own flag, not a count re-derived from the
        # numbers on the card. A reporter that recomputed a verdict would be
        # free to announce a pass the audit did not record.
        passed = sum(1 for r in rows if r.get("certified"))
        return embed, (f"Stage 3 certification '{args.strat}' "
                       f"({passed}/{len(rows)} certified, from {what})")

    if mode == "verify":
        # The DIRECTORY is the input here, not a handoff file: Stage 4 writes
        # one `dual_metrics_<SYMBOL>.json` per contract into a run-stamped
        # directory, and the card describes that run. Whichever directory this
        # resolves to is named on the card and in the success line, so a
        # defaulted choice is never a silent one.
        path = (Path(args.artifacts) if args.artifacts
                else default_verify_dir(args.strat, args.out_dir))
        rows = stage4_rows(path, args.strat)
        embed = build_stage4_embed(args.strat, rows, source=path,
                                   max_rows=(args.max_rows
                                             if args.max_rows is not None
                                             else STAGE4_MAX_ROWS))
        read = sum(1 for r in rows if not r.get("error"))
        return embed, (f"Stage 4 lifecycle '{args.strat}' "
                       f"({read}/{len(rows)} contract(s) from {path.name})")

    # Everything the promotion card needs, from the command line first and
    # from what Stages 3 and 5 already wrote for anything left over. A file
    # named explicitly and missing raises here rather than being defaulted
    # around: an operator who typed a path meant that path.
    res = resolve_promotion_fields(
        args.strat,
        symbol=args.symbol, tf=args.tf, pf=args.pf, win=args.win, dd=args.dd,
        regime=args.regime, report=args.report,
        audit_file=args.audit_file, metrics_file=args.metrics,
        incubator=args.incubator, portfolio_config=args.portfolios)

    # The card names ONE contract, so these two are still required - but only
    # after the resolution has had its turn. Checked rather than defaulted: a
    # card headed `?` · `?` is a promotion announcement for a strategy on no
    # instrument. The refusal names where this looked, because "pass --symbol"
    # and "your meta.json cites an audit that is not on disk" send an operator
    # to two completely different places.
    missing = [f for f, v in (("--symbol", res["symbol"]), ("--tf", res["tf"]))
               if not v]
    if missing:
        # An ambiguous MODULE name is a different failure from a missing
        # measurement, and it has a different fix. Stage 5 writes one package
        # per certified pair, so a module with several of them cannot be
        # reduced to the one contract this card names - and telling the
        # operator to pass --symbol would have them describe a package by
        # hand when the id they want is already on disk. Name them instead.
        packages = res.get("packages") or []
        if len(packages) > 1:
            listed = "\n  ".join(packages)
            raise ValueError(
                f"--mode promotion needs {' and '.join(missing)}: "
                f"'{args.strat}' is a MODULE name and Stage 5 promoted "
                f"{len(packages)} packages from it. A promotion card names "
                f"one contract, so name one package:\n  {listed}")
        looked = ", ".join(res["inspected"]) or f"nothing under {res['home']}"
        raise ValueError(
            f"--mode promotion needs {' and '.join(missing)}: the contract "
            f"could not be resolved from {looked}.")

    embed = build_embed(
        # The PACKAGE's id, not what was typed: where a module name resolved
        # to a single promotion, the card describes that package and must be
        # headed by its name, or the announcement reads as though the module
        # itself were promoted at one contract.
        strat=res["strat_id"],
        symbol=res["symbol"],
        tf=res["tf"],
        pf=res["pf"],
        win=res["win"],
        dd=res["dd"],
        regime=res["regime"],
        report=res["report"],
        membership=res["membership"],
        resolution=res,
    )
    auto = len(res["resolved"])
    return embed, (f"'{args.strat}' ({res['symbol']} {res['tf']})"
                   + (f", {auto} value(s) auto-resolved" if auto else ""))


#: How long an identical payload is treated as a repeat rather than as a new
#: card. Ten seconds is long enough to catch a double-invocation - a re-run
#: after a scrollback, an orchestrator and a hand-typed command racing - and
#: far too short to suppress a genuine re-post minutes later.
DUPLICATE_WINDOW_SECONDS = 10.0


def _guard_path() -> Path:
    """
    Where the last-post record lives: LOCAL disk, never the NFS mount.

    The lake is mounted hard and the artifact tree is shared; a guard file
    there would be one more thing taking a lock over NLM for a convenience
    that only ever concerns one box. It is also not an artifact - nothing
    downstream reads it - so it does not belong beside the handoffs.
    """
    return Path(tempfile.gettempdir()) / f"discord_reporter_guard_{os.getuid()}.json"


def payload_fingerprint(payload: dict[str, Any]) -> str:
    """
    A stable digest of exactly what would be sent.

    Keyed on the PAYLOAD rather than on (stage, strategy), because those two
    are equal for a re-post that legitimately carries new numbers - a screen
    re-run after a fix is the same stage and the same strategy and is not a
    duplicate. Two posts collide here only when every character Discord would
    receive is identical.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def seconds_since_identical_post(fingerprint: str, *, now: float | None = None,
                                 path: Path | None = None) -> float | None:
    """
    Age of an identical post, or None if there is not one inside the window.

    UNREADABLE STATE IS NOT A DUPLICATE. A corrupt, truncated or absent guard
    file returns None and the card goes out. The guard is a convenience against
    double-invocation; failing the other way would let a broken temp file
    silence a stage card, and a notification that is missing is a far worse
    failure than one that arrives twice.
    """
    now = time.time() if now is None else now
    try:
        record = json.loads((path or _guard_path()).read_text())
        if record.get("fingerprint") != fingerprint:
            return None
        age = now - float(record["posted_at"])
    except (OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError):
        return None
    if 0 <= age < DUPLICATE_WINDOW_SECONDS:
        return age
    return None


def record_post(fingerprint: str, *, now: float | None = None,
                path: Path | None = None) -> None:
    """Remember what was just sent. A write that fails is not a failed post."""
    now = time.time() if now is None else now
    try:
        (path or _guard_path()).write_text(json.dumps(
            {"fingerprint": fingerprint, "posted_at": now}))
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        embed, summary = _build_card(args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        # Nothing is posted. A card built from a handoff that could not be read
        # would have to invent the numbers on it, which is the one thing this
        # module refuses to do.
        print(f"FAILED  {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    size = _embed_size(embed)
    if size > MAX_EMBED_TOTAL:
        print(f"FAILED  embed is {size} characters, over Discord's {MAX_EMBED_TOTAL} limit; nothing sent.",
              file=sys.stderr)
        return 1

    payload = build_payload(embed)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        print("DRY RUN  nothing was sent.")
        return 0

    # Checked AFTER --dry-run, so a dry run neither consults nor arms the
    # guard, and BEFORE the webhook is resolved, so a suppressed duplicate
    # never touches the credential.
    fingerprint = payload_fingerprint(payload)
    age = None if args.force else seconds_since_identical_post(fingerprint)
    if age is not None:
        # Exit 0, not 1. Nothing failed - the card the operator wanted is
        # already in the channel - and run_pipeline posts these with
        # check=False beside stages that must not be aborted by a notifier.
        print(f"SKIPPED  an identical payload was posted "
              f"{age:.1f}s ago; not sending it again. "
              f"Use --force to override.")
        return 0

    webhook, webhook_source = describe_webhook(args.webhook)
    if not webhook:
        print(f"FAILED  no webhook: {WEBHOOK_HINT}.", file=sys.stderr)
        return 1

    result = post_embed(webhook, payload)

    if result["ok"]:
        # Armed only on a post Discord ACCEPTED. Recording a rejected send
        # would make the retry look like a duplicate and swallow the card.
        record_post(fingerprint)
        # The SOURCE is a variable name and is safe to print; the URL is a
        # credential and is not. Naming it is what tells an operator with a
        # test channel and a live one which of the two just received the card.
        print(f"SUCCESS  posted {summary} to Discord "
              f"[HTTP {result['http_status']}] via {webhook_source}")
        return 0

    status = result["http_status"]
    where = f"HTTP {status}" if status is not None else "no response"
    print(f"FAILED  Discord rejected the post [{where}]: {result['error']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
