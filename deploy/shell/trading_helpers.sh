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
