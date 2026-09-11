#!/usr/bin/env bash

if [ -z "$1" ]; then
  echo "Usage: $0 <campaign_name> [symbols] [timeframes]"
  exit 1
fi

STRAT_NAME=$1
# Default to the full known symbol universe if not explicitly overridden
SYMBOLS=${2:-"6A,6B,6C,6E,6J,6S,BTC,CL,ES,ETH,GC,HO,LE,NG,NQ,PL,RB,RTY,SI,YM,ZB,ZC,ZF,ZN,ZS,ZT,ZW"}
# Default to all active intraday timeframes
TIMEFRAMES=${3:-"5m,15m,30m,1h"}
END_DATE=2022-12-31  # in-sample charter end; the holdout begins 2023-01-01 and is handled by Stage 3

echo "=== Executing Stages 1-4.5 (Generation, ML, DOW Gate) ==="
OMP_NUM_THREADS=4 .venv/bin/python3 -u backtest/run_pipeline.py \
  --strat "$STRAT_NAME" \
  --symbols "$SYMBOLS" \
  --tf "$TIMEFRAMES" \
  --start 2013-01-01 \
  --end "$END_DATE" \
  --ml-threshold 0.48 \
  --report-discord \
  --auto-promote

echo "=== Executing Stage 5 (Promotion & Sealing) ==="
# Note: For brand new campaigns, a blanket promote is safe. 
# If running on an existing campaign with live packages, the --only flag must be appended.
.venv/bin/python3 -u backtest/run_pipeline.py \
  --strat "$STRAT_NAME" \
  --promote-only

echo "=== Executing Stage 6 (Dry-Run Audit) ==="
.venv/bin/python3 scripts/register_incubator_batch.py 2>/dev/null || \
.venv/bin/python3 backtest/register_incubator_batch.py 2>/dev/null || \
echo "WARNING: register_incubator_batch.py not found in scripts/ or backtest/"

echo "=== Phase A Complete. Awaiting human review before --write ==="
