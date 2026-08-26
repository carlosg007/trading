# Interactive shell helpers for the research pipeline.
#
# Sourced from ~/.bashrc. It lives in the REPO rather than inline in .bashrc so
# it is reviewable, revertible and versioned like the systemd units — and so a
# fix here reaches the box through git rather than through somebody editing a
# dotfile from memory.
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

# The 24 contracts that have BOTH 1m lake data and a ContractSpec. HG, ZM and
# ZL are deliberately absent: neither is in the lake and neither has a spec, so
# including them buys a failed run, not more coverage.
_TRADING_UNIVERSE="ES,NQ,RTY,YM,CL,NG,RB,HO,GC,SI,PL,ZB,ZN,ZF,ZT,6E,6J,6B,6A,6C,6S,ZC,ZS,ZW"

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
alias precompute-all='precompute_all_regimes'

# 2. The pipeline, prompting for the strategy when none is given.
run_pipeline_all() {
    _trading_preflight || return 1

    local strat_name="$1"
    # The prompt runs OUTSIDE the subshell below, so it reads from the terminal.
    if [ -z "$strat_name" ]; then
        read -rp "Strategy to run (e.g. double_rsi_macd_scalp_20260823): " strat_name
    fi
    [ -n "$strat_name" ] || { echo "Error: strategy name cannot be empty." >&2; return 1; }

    echo "==> pipeline: ${strat_name}  |  24 contracts x 6 timeframes  |  2013-01-01..2022-12-31"
    # Said out loud every time, because it is the one flag here that CHANGES
    # STATE somebody has to live with: Stage 5 runs for every configuration
    # Stage 3 certified, up to 144 of them, without a human seeing a card
    # first. It never overrides a gate. To look before registering, pass
    # --promote-only later, or --dry-run now to print the plan and run nothing.
    echo "    --auto-promote IS ON: every certified configuration registers unattended."
    (
        cd "$_TRADING_REPO" || exit 1
        nice -n 19 ionice -c 3 "$_TRADING_PY" backtest/run_pipeline.py \
            --strat "$strat_name" \
            --symbols "$_TRADING_UNIVERSE" \
            --tf 1m,2m,3m,5m,15m,30m \
            --start 2013-01-01 \
            --end 2022-12-31 \
            --report-discord \
            --auto-promote "${@:2}"
    )
}
alias pipeline-all='run_pipeline_all'

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
alias bt-check="${_TRADING_PY} ${_TRADING_REPO}/backtest/check_progress.py"
# The same tool under the name people reach for when the question is "how far
# along is it" rather than "is it alive". One implementation, two spellings —
# a second script would be one more thing to keep in step with the stages.
alias bt-progress='bt-check'

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
