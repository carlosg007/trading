#!/usr/bin/env bash
# The shebang is for TOOLING, not for execution. This file is SOURCED
# from ~/.bashrc and is never run as a program, so the kernel never
# reads this line. Editors, linters and `shellcheck` do: without it they
# default to POSIX sh, where a hyphen in a function name is illegal, and
# report `bt-check()` at line 66 as "Bad function name" - a real error
# message about a file that is not broken. `bash -n` has always passed.
# Interactive shell helpers for the research pipeline.
#
# Sourced from ~/.bashrc. It lives in the REPO rather than inline in .bashrc so
# it is reviewable, revertible and versioned like the systemd units — and so a
# fix here reaches the box through git rather than through somebody editing a
# dotfile from memory.
#
# BASH ONLY, AND `sh -n` ON THIS FILE IS EXPECTED TO FAIL. Thirteen helpers are
# named with a hyphen — precompute-all, pipeline-all, bt-check, bt-inventory —
# because that is the vocabulary CLAUDE.md documents and the one people type.
# Bash permits a hyphen in a function name; POSIX sh does not, so `dash -n`
# stops at the first of them with "Bad function name" and a reader can mistake
# a deliberate bash file for a broken one. `bash -n` is the check that applies
# here, and it passes. Renaming them to satisfy a shell that never sources this
# file would break every command in the runbook.
#
#   precompute_all_regimes   (precompute-all)  build the ATR/ADX regime cache
#   run_pipeline_all         (pipeline-all)    run the multi-asset pipeline
#
# BOTH ARE LONG JOBS. They run at nice 19 / ionice idle so they yield to the
# live stack — the NT8 listener, the regime daemon and the execution loop share
# this box, and a backtest that wins a CPU fight with the loop is a backtest
# that cost money.

_TRADING_REPO="/home/cgrullon/src/trading"
_TRADING_PY="${_TRADING_REPO}/.venv/bin/python3"

# The 23 contracts that have BOTH 1m lake data and a ContractSpec AND a live
# NT8 feed. HG, ZM and ZL are absent because neither is in the lake and neither
# has a spec, so including them buys a failed run rather than more coverage.
#
# CL was removed on 2026-08-29 for a DIFFERENT reason, and the difference
# decides whether putting it back is right. CL has 16 years of 1m data, a
# verified ContractSpec and all ten regime caches, and it cleared Stage 1 at
# 15m, 30m and 1h on t3_braid_scalp_20260823. What it does not have is a live
# NT8 series, so nothing screened on it can be traded from this box. Restore it
# here the day the feed carries it, and check the spool rather than this
# comment:
#
#     ls /mnt/backtest/artifacts/nt8_bars/ | grep '^CL' 
_TRADING_UNIVERSE="ES,NQ,RTY,YM,NG,RB,HO,GC,SI,PL,ZB,ZN,ZF,ZT,6E,6J,6B,6A,6C,6S,ZC,ZS,ZW"

# Fail loudly here rather than three screens into a run.
_trading_preflight() {
    [ -d "$_TRADING_REPO" ] || { echo "no repo at $_TRADING_REPO" >&2; return 1; }
    [ -x "$_TRADING_PY" ]   || { echo "no venv interpreter at $_TRADING_PY" >&2; return 1; }
}

# 1. Regime cache for every contract, every timeframe.
#
# --is-start/--is-end are NOT passed. The script already defaults to the pinned
# 2013-01-01..2022-12-31 anchor, and theta_vol is LOADED from that anchor
# rather than recomputed per request — an override here would silently rewrite
# the boundary every certified strategy was drawn against.
precompute_all_regimes() {
    _trading_preflight || return 1
    echo "==> regime precompute: 24 contracts x 10 timeframes, nice 19 / ionice idle"
    echo "    add --force to rebuild caches that already exist"
    # A SUBSHELL: the cd is scoped to the job. Without it the helper leaves your
    # interactive shell in the repo, which is not what "works from any
    # directory" means.
    (
        cd "$_TRADING_REPO" || exit 1
        nice -n 19 ionice -c 3 "$_TRADING_PY" scripts/precompute_regimes.py \
            --symbols "$_TRADING_UNIVERSE" \
            --tf 1m,2m,3m,5m,15m,30m,1h,2h,4h,1d "$@"
    )
}
# A FUNCTION, not an alias: bash does not expand aliases in a
# non-interactive shell, so `precompute-all` in a script was "command not
# found" while the function it points at worked. The two job launchers
# and the bt-tf family are functions for this reason; the read-only
# status cards below are left as aliases.
precompute-all() { precompute_all_regimes "$@"; }

# 2. The pipeline, prompting for the strategy when none is given.
#
# --tf names the GROUP, not the four timeframes. `CORE_DAY_TRADING` is
# `backtest.run.TF_GROUPS`'s own tuple — 5m, 15m, 30m, 1h — and
# `run_pipeline.parse_timeframes` expands it to exactly what every other stage
# gets from that same constant. Spelling the list out here worked and was one
# edit away from not: two spellings of "the day-trading ladder" drifting apart
# would put Stage 1 and Stage 3 on different timeframe sets with every log
# line reading correctly. Changing the ladder is a change to ONE tuple.
#
# It was ALL_DAY_TRADING until 2026-08-27, which added 1m/2m/3m. Those are
# excluded by default on FRICTION — a sub-5m bar spends much of its own range
# on a tick of slippage each way plus commission, so the screen largely
# measures the cost model — and because they tripled the search for the
# resolutions least likely to survive it. `pipeline-all ... --tf
# ALL_DAY_TRADING` overrides this in one word; the later --tf wins.
run_pipeline_all() {
    _trading_preflight || return 1

    local strat_name="$1"
    # The prompt runs OUTSIDE the subshell below, so it reads from the terminal.
    if [ -z "$strat_name" ]; then
        read -rp "Strategy to run (e.g. double_rsi_macd_scalp_20260823): " strat_name
    fi
    [ -n "$strat_name" ] || { echo "Error: strategy name cannot be empty." >&2; return 1; }

    echo "==> pipeline: ${strat_name}  |  24 contracts x 4 timeframes [5m,15m,30m,1h]  |  2013-01-01..2022-12-31"
    # Said out loud every time, because it is the one flag here that CHANGES
    # STATE somebody has to live with: Stage 5 runs for every configuration
    # Stage 3 certified, up to 168 of them (24 contracts x 7 timeframes),
    # without a human seeing a card
    # first. It never overrides a gate. To look before registering, pass
    # --promote-only later, or --dry-run now to print the plan and run nothing.
    echo "    --auto-promote IS ON: every certified configuration registers unattended."
    (
        cd "$_TRADING_REPO" || exit 1
        nice -n 19 ionice -c 3 "$_TRADING_PY" backtest/run_pipeline.py \
            --strat "$strat_name" \
            --symbols "$_TRADING_UNIVERSE" \
            --tf CORE_DAY_TRADING \
            --start 2013-01-01 \
            --end 2022-12-31 \
            --report-discord \
            --auto-promote "${@:2}"
    )
}
# A FUNCTION, not an alias: bash does not expand aliases in a
# non-interactive shell, so `pipeline-all` in a script was "command not
# found" while the function it points at worked. The two job launchers
# and the bt-tf family are functions for this reason; the read-only
# status cards below are left as aliases.
pipeline-all() { run_pipeline_all "$@"; }

# 2b. The same pipeline on a CHOSEN SUBSET of timeframes.
#
# `run_pipeline_all` runs the whole day-trading ladder and takes the strategy
# as its first argument. This one inverts that: the TIMEFRAMES come first,
# because they are what varies when you are iterating on one strategy. It is a
# SEPARATE function rather than an extra flag on `run_pipeline_all` — that
# helper's signature is `run_pipeline_all <strategy>` and is in muscle memory
# and in the runbook; giving it a new first argument would silently reinterpret
# every invocation anyone already types.
#
# The name is NOT `bt-run`. That is already the alias for `backtest/run.py` —
# the single-symbol runner — in ~/.bashrc and in CLAUDE.md's helper table.
# Rebinding it here would repoint a documented command from one entry point to
# a different one, and the failure would look like a backtest behaving oddly
# rather than like the wrong program running.
#
# --auto-promote IS DELIBERATELY OFF HERE, unlike `run_pipeline_all`. This is
# the wrapper for a narrow, exploratory sweep — "what does this look like at
# 1h" — and that is exactly the run whose output you have not read yet.
# Auto-promotion registers certified configurations and COMMITS THEM TO GIT
# without a human seeing a card. Pass --auto-promote explicitly when you mean
# it; it is forwarded like any other flag.
_TRADING_TF_KNOWN="1m 2m 3m 5m 15m 30m 1h 2h 4h 1d 1w"

# A MIRROR of `mdlib.lake.NATIVE_TFS | DERIVED`, which is the authority. It is
# restated here because the check has to happen before a Python process
# starts — the point is to fail on a typo in the shell, in a millisecond,
# rather than after the interpreter, the imports and the first lake read.
#
# It earns its keep: `run_pipeline.parse_timeframes` does NOT validate. It
# splits on commas and returns whatever it was handed, so `--tf 1hr` reaches
# Stage 1 as a timeframe nobody serves. Add a timeframe to lake.py's DERIVED
# and add it here too.
_trading_check_tf() {
    local raw="$1" tf out="" seen
    [ -n "$raw" ] || { echo "Error: no timeframe given." >&2; return 1; }
    local IFS=,
    for tf in $raw; do
        # Strip surrounding whitespace so `5m, 15m` is not a typo.
        tf="${tf#"${tf%%[![:space:]]*}"}"
        tf="${tf%"${tf##*[![:space:]]}"}"
        [ -n "$tf" ] || continue
        case " $_TRADING_TF_KNOWN " in
            *" $tf "*) ;;
            *) echo "Error: unknown timeframe '$tf'." >&2
               echo "       The lake serves: $_TRADING_TF_KNOWN" >&2
               return 1 ;;
        esac
        # De-duplicate, preserving the order given.
        case " $out " in *" $tf "*) continue ;; esac
        out="${out:+$out,}$tf"
    done
    [ -n "$out" ] || { echo "Error: '$raw' names no timeframe." >&2; return 1; }
    printf '%s' "$out"
}

run_pipeline_tf() {
    _trading_preflight || return 1

    local a
    for a in "$@"; do
        case "$a" in
            -h|--help)
                cat <<'USAGE'
bt-tf <timeframes> [strategy] [extra flags...]

  <timeframes>  one or a comma-separated list: 1m 2m 3m 5m 15m 30m 1h 2h 4h 1d 1w
  [strategy]    module name; prompted for when omitted
  extra flags   forwarded verbatim to backtest/run_pipeline.py

  Runs 24 contracts over 2013-01-01..2022-12-31 at nice 19 / ionice idle.
  --auto-promote is OFF unless you pass it. --dry-run prints the plan only.

  bt-tf 1h double_rsi_macd_scalp_20260823
  bt-tf 5m,15m,30m my_strat --dry-run
  bt-1h my_strat     bt-30m     bt-15m     bt-5m     bt-swing
USAGE
                return 0 ;;
        esac
    done

    local tf_list
    tf_list="$(_trading_check_tf "$1")" || return 1
    shift

    # A leading flag means the strategy was omitted, not that it is named
    # "--dry-run". Without this, `bt-tf 1h --dry-run` would run a strategy
    # module by that name and fail on the import instead of prompting.
    local strat_name=""
    case "$1" in
        -*|"") ;;
        *) strat_name="$1"; shift ;;
    esac
    # The prompt runs OUTSIDE the subshell below, so it reads from the terminal.
    if [ -z "$strat_name" ]; then
        read -rp "Strategy to run (e.g. double_rsi_macd_scalp_20260823): " strat_name
    fi
    [ -n "$strat_name" ] || { echo "Error: strategy name cannot be empty." >&2; return 1; }

    # awk counts FIELDS. `printf | tr ',' '\n' | wc -l` counted NEWLINES, which
    # is one fewer than the number of timeframes — a single-tf run announced
    # itself as "0 timeframe(s)".
    local n_tf; n_tf="$(printf '%s' "$tf_list" | awk -F, '{print NF}')"
    echo "==> pipeline: ${strat_name}  |  24 contracts x ${n_tf} timeframe(s) [${tf_list}]  |  2013-01-01..2022-12-31"
    case " $* " in
        *" --auto-promote "*)
            echo "    --auto-promote IS ON: every certified configuration registers unattended." ;;
        *)  echo "    --auto-promote is off: nothing is promoted or committed." ;;
    esac
    (
        cd "$_TRADING_REPO" || exit 1
        nice -n 19 ionice -c 3 "$_TRADING_PY" backtest/run_pipeline.py \
            --strat "$strat_name" \
            --symbols "$_TRADING_UNIVERSE" \
            --tf "$tf_list" \
            --start 2013-01-01 \
            --end 2022-12-31 \
            --report-discord "$@"
    )
}
# FUNCTIONS, not aliases. Bash does not expand aliases in a non-interactive
# shell, so an aliased `bt-tf` works when typed and is "command not found" the
# moment anyone puts it in a script — while the bt-1h family beside it keeps
# working. Two spellings of the same helper should not differ in where they
# are available.
bt-tf()  { run_pipeline_tf "$@"; }
# The spelling to reach for when the sentence in your head is "run the
# backtest" rather than "pick the timeframes". One implementation, two names —
# and NOT `bt-run`, which is the single-symbol runner.
run-bt() { run_pipeline_tf "$@"; }

# The single-timeframe shortcuts. Functions, not aliases, because an alias
# cannot put its arguments BEFORE the fixed timeframe — `alias bt-1h='bt-tf 1h'`
# would build `bt-tf 1h` and then append, which happens to work, but breaks the
# moment a shortcut needs anything after the strategy.
bt-1h()    { run_pipeline_tf 1h  "$@"; }
bt-30m()   { run_pipeline_tf 30m "$@"; }
bt-15m()   { run_pipeline_tf 15m "$@"; }
bt-5m()    { run_pipeline_tf 5m  "$@"; }
# The swing ladder: the four timeframes a multi-hour hold is actually screened
# on. Deliberately NOT ALL_DAY_TRADING — 1m/2m/3m are scalping resolutions and
# carry the most friction per unit of edge, so including them in a "swing" run
# would spend hours on configurations the name says you are not looking for.
bt-swing() { run_pipeline_tf 5m,15m,30m,1h "$@"; }

# 3. The status card: where a run has got to, in English.
#
# ALIASES, NOT FUNCTIONS, and not run under nice/ionice like the two jobs
# above. This one reads `ps`, `stat` and a handful of small JSON handoffs — it
# opens no bar, imports no strategy and finishes in well under a second, so
# there is nothing here for the live stack to lose a CPU fight with. Yielding
# it to idle priority would only make the answer arrive late.
#
# It is also the ONLY part of the pipeline that is safe to run from inside a
# Claude Code session: it starts no backtest and takes nobody out of the loop.
#
# Aliases pass their arguments through, so `bt-check --watch 10` and
# `bt-check --strategy <name>` work without a wrapper. The interpreter is
# spelled absolutely for the same reason `_TRADING_PY` exists: the venv must
# not depend on which one happens to be active in the calling shell.
# FUNCTIONS rather than aliases, unlike the cards below. Two reasons specific
# to this one: `complete -F` attaches to a function reliably, and an alias is
# not expanded in a non-interactive shell, so `bt-check` in a script was
# "command not found". "$@" is forwarded, so `bt-check <strategy>`,
# `bt-check --all`, `bt-check -l` and `bt-check --watch 10` all reach argparse.
bt-check()    { "$_TRADING_PY" "${_TRADING_REPO}/backtest/check_progress.py" "$@"; }
# The same tool under the name people reach for when the question is "how far
# along is it" rather than "is it alive". One implementation, two spellings —
# a second script would be one more thing to keep in step with the stages.
bt-progress() { bt-check "$@"; }

# Completion: the campaign directories, which are the only valid values for
# the positional argument. $BT_ARTIFACTS is honoured because `pipeline.
# artifacts_root()` honours it - completion offering names from a tree the
# tool will not read is worse than no completion.
_bt_check_complete() {
    local cur root
    cur="${COMP_WORDS[COMP_CWORD]}"
    if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "-h --help -a --all -l --list --strategy \
                                 --strat --out-dir --watch" -- "$cur"))
        return
    fi
    root="${BT_ARTIFACTS:-/mnt/backtest/artifacts}/pipeline"
    # -maxdepth/-mindepth 1 so this lists campaigns, not the tree under them,
    # and 2>/dev/null so an unmounted NFS completes to nothing instead of
    # printing a find error over the prompt.
    COMPREPLY=($(compgen -W "$(find "$root" -mindepth 1 -maxdepth 1 -type d \
                              -printf '%f\n' 2>/dev/null)" -- "$cur"))
}
complete -F _bt_check_complete bt-check bt-progress

# 4. The live feed: is NinjaTrader still sending bars?
#
# Same shape as bt-check above and safe for the same reasons — it reads
# /health, the spool, the routing table and the two files downstream of the
# feed. It opens no bar, imports no engine, writes nothing, and finishes in
# under a tenth of a second, so it is not run under nice/ionice either.
#
# Deliberately NOT `curl localhost:8000/health | jq`, which the runbook's
# pre-flight already gives you. That is the RECEIVER's view and it cannot see
# the three things you actually need: whether the process is there at all (a
# refused connection and a STARVED listener look identical through curl),
# whether the streams arriving are the ones config/portfolios.json needs, and
# whether the regime file and watchdog downstream of the feed have moved.
#
# Exits non-zero when the listener cannot be reached, so it chains:
#     nt8-check && systemctl restart trading-master-live
alias nt8-check="${_TRADING_PY} ${_TRADING_REPO}/realtime/check_nt8_feed.py"
# The name people reach for when the question is "is the feed up" rather than
# "what is the listener doing". One implementation, two spellings.
alias feed-status='nt8-check'

# 5. The other half of the live question: is anything DECIDING?
#
# `nt8-check` says bars are arriving. This says what is being done with them —
# which strategy is allocated, what quadrant the gate thinks it is in, whether
# entries are permitted, and what the firewall has counted today.
#
# It reads state other processes WROTE and recomputes nothing. That is the
# whole design: the engine evaluates on the bars IT loaded, with its own
# warm-up and its own last-closed-bar rule, so a status tool that recomputed
# an indicator would differ at exactly the boundaries that matter and would
# carry a status tool's authority while doing it.
#
# Exit code answers "is it evaluating", NOT "is something wrong": 1 when the
# loop is not running, which on this box is the normal resting state — the
# shipped unit is --dry-run and is not armed.
alias signal-check="${_TRADING_PY} ${_TRADING_REPO}/realtime/check_live_signals.py"
# The name for "show me what the strategies are doing" rather than "is the
# engine up". One implementation, two spellings.
alias live-signals='signal-check'

# 6. Why is nothing trading?
#
# The third of the three cards, and the one to reach for when the other two
# look fine and no order has gone out. `nt8-check` says bars arrive,
# `signal-check` says what the loop decided, this says which layer is stopping
# an entry — interlock, kill switch, session caps, execution bridge, regime
# gate, feed — and ends with the blockers listed in the order the stack
# applies them.
#
# IT NEVER PROBES THE ORDER ENDPOINT. "Reachable" for CrossTrade means POSTing
# to the thing that places orders on a funded account; configuration is
# checked and the first real request is left to the loop, under the interlock
# and the kill switch where it belongs.
#
# Exit code answers "would an entry go through": 1 while anything blocks. On
# this box, in dry run, that is deliberately non-zero.
alias firewall-check="${_TRADING_PY} ${_TRADING_REPO}/realtime/check_trade_firewall.py"
# The name for "why did my trade not fire" rather than "is the firewall up".
alias trade-gate='firewall-check'

# 7. Is the order bridge actually reachable from this box?
#
# `firewall-check` reports the bridge as configured or not. This one connects:
# DNS, TCP, and a real TLS handshake with certificate verification — and sends
# ZERO bytes of HTTP. A completed handshake proves the name resolves, the route
# works, something is listening and it presents a certificate this box trusts.
# No path is requested, so no endpoint can act on it.
#
# The webhook URL's path IS the credential and the key IS the account, so
# neither is ever printed: the origin, a path SEGMENT COUNT, and the key's
# length plus last four characters. That is enough to tell two keys apart and
# useless to anyone reading over a shoulder.
#
# `--http-probe` adds a GET, and only ever to the bare origin — it refuses a
# URL carrying a path, query or fragment.
#
# Exits 0 when configured AND reachable, 1 otherwise, so it chains.
alias crosstrade-check="${_TRADING_PY} ${_TRADING_REPO}/realtime/check_crosstrade_connection.py"
alias ct-check='crosstrade-check'

# 10. Is this box in a fit state to be trusted?
#
# The post-power-on check, run before anything is armed. It walks the mount,
# the interpreter and its packages, the NT8 spool, the CrossTrade bridge, the
# routing table and outbound DNS, and prints a badge per subsystem.
#
# IT CHANGES NOTHING and CONTACTS NO ORDER ENDPOINT. Every probe is a read, a
# DNS lookup or a TLS handshake; the one write is a probe file on the mount,
# removed immediately, because that is the only honest way to answer "is it
# writable". `check_crosstrade_connection`'s rule holds - the webhook's PATH is
# the credential, so a completed handshake is the whole proof a preflight needs.
#
# It will NOT print "ready for live trading". It can see machinery; the account
# balance lives in CrossTrade, the evidence behind each strategy lives in its
# gate audit, and the decision is a human's with the runbook open.
#
# Exits non-zero on any subsystem FAULT, so it chains:
#     check-system && systemctl restart trading-master-live
check-system()    { "$_TRADING_PY" "${_TRADING_REPO}/tools/system_preflight_check.py" "$@"; }
# The spelling for "run the preflight" rather than "check the system". One
# implementation, two names.
preflight-check() { check-system "$@"; }

_check_system_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    COMPREPLY=($(compgen -W "--skip-network --json --no-color -h --help" \
                         -- "$cur"))
}
complete -F _check_system_complete check-system preflight-check

# 9. What is allocated, on what evidence.
#
# The other cards ask about MOTION and CONFIGURATION. This one joins the two
# to the EVIDENCE: it walks config/portfolios.json, the promoted package under
# strategies/approved_incubator/, and the gate audit each package cites, and
# puts them on one row.
#
# It exists because nothing joined those three. On 2026-08-29 the routing
# table carried 18 allocations naming packages that had been deleted, and the
# live dispatcher imports a strategy by that path - so every one was an entry
# that could never load. Finding it took a hand-written diff.
#
# A FUNCTION, so `complete -F` attaches and it works in a script. Exits
# non-zero when any allocation names a package that is not on disk, so it
# chains:  bt-inventory && systemctl restart trading-master-live
bt-inventory() { "$_TRADING_PY" "${_TRADING_REPO}/tools/portfolio_inventory.py" "$@"; }
# The spelling for "what is in the portfolios" rather than "is the config
# readable". One implementation, two names.
inv-portfolios() { bt-inventory "$@"; }

_bt_inventory_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    COMPREPLY=($(compgen -W "--out --portfolio --symbol --version --no-csv \
                             --config -h --help" -- "$cur"))
}
complete -F _bt_inventory_complete bt-inventory inv-portfolios

# 8. Who is allowed to trade what, and where does the order go?
#
# The other cards ask about MOTION — are bars arriving, what did the loop
# decide, why is nothing firing. This one asks about CONFIGURATION: which
# portfolios exist, which account each addresses, what each may trade, and
# which strategy is allocated to which contract.
#
# Four files meet in it and none is authoritative alone — the routing table,
# the promoted meta.json, the switchboard, and the spool. An id present in one
# and absent from another is a real and quiet fault, so each is reported as
# found or missing rather than merged into a row that hides which is empty.
#
# Empty portfolios are SHOWN. Three of the four here are empty by design, and
# filtering them out would make "nothing is allocated" indistinguishable from
# "this portfolio does not exist".
#
# Exits non-zero only when the configuration cannot be READ. An empty
# portfolio is a legitimate state, not a failure.
alias portfolio-check="${_TRADING_PY} ${_TRADING_REPO}/realtime/check_portfolio_assets.py"
alias assets-check='portfolio-check'
