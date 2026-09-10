#!/usr/bin/env bash
# ~/src/trading/scripts/cron_git_push.sh
# Daily automated push of the trading repo to GitHub (vectorbt-engine branch).
# Schedule: midnight US/Eastern via user crontab (CRON_TZ=America/New_York).
set -euo pipefail

REPO_DIR="$HOME/src/trading"
LOG_FILE="$REPO_DIR/logs/cron_git_push.log"
BRANCH="vectorbt-engine"
MAX_LOG_BYTES=$((10 * 1024 * 1024))  # 10 MiB — truncate if exceeded

# --- SSH key for non-interactive cron execution ---
# Cron runs without a login shell or desktop keyring, so SSH_AUTH_SOCK is
# almost always unset at midnight. Pin the key explicitly so git push never
# hangs on a missing agent or an unlocked passphrase prompt.
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

mkdir -p "$REPO_DIR/logs"

# --- Log rotation ---
if [[ -f "$LOG_FILE" ]]; then
  size=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
  if (( size > MAX_LOG_BYTES )); then
    mv "$LOG_FILE" "${LOG_FILE}.old"
  fi
fi

# --- Timestamp helper ---
ts() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

log() {
  printf "[%s] %s\n" "$(ts)" "$*" >> "$LOG_FILE"
}

log "=== cron_git_push START ==="

# --- Working dir ---
cd "$REPO_DIR"

# --- SSH environment check ---
if [[ -n "${SSH_AUTH_SOCK:-}" ]]; then
  log "SSH_AUTH_SOCK=$SSH_AUTH_SOCK (agent available)"
else
  log "WARNING: SSH_AUTH_SOCK not set — SSH key must be unlocked or agent must be running"
fi

# --- Branch check ---
current_branch=$(git branch --show-current 2>/dev/null || true)
log "Current branch: ${current_branch:-<detached>}"

# --- Pre-flight: local status ---
if git status --porcelain | grep -q .; then
  log "WARNING: working tree is dirty — uncommitted changes present"
else
  log "Working tree is clean"
fi

# --- Behind remote? ---
remote_head=$(git rev-parse --abbrev-ref @{u} 2>/dev/null || true)
if [[ -n "$remote_head" ]]; then
  # ahead  = commits HERE that the remote does not have -> @{u}..HEAD
  # behind = commits on the REMOTE that we do not have     -> HEAD..@{u}
  # These were the wrong way round, so every log line read backwards.
  ahead=$(git rev-list --count "@{u}..HEAD" 2>/dev/null || echo 0)
  behind=$(git rev-list --count "HEAD..@{u}" 2>/dev/null || echo 0)
  log "Tracking ${remote_head}: ahead=${ahead} behind=${behind}"
  if (( behind > 0 )); then
    # Not fatal here - the push below will be refused as a non-fast-forward
    # and reported as one. Logged first so the reason is in the file before
    # the refusal is, rather than being inferred from git's wording.
    log "WARNING: behind the remote by ${behind} commit(s); a push will be refused as non-fast-forward"
  fi
else
  log "No upstream tracking branch configured"
fi

# --- Push ---
#
# THE STATUS HAS TO COME FROM `git push` ITSELF. The original piped git into
# `tee` and read the pipeline's status: under `pipefail` that is the rightmost
# non-zero one, so a `tee` that failed on a full disk or an unwritable log
# reported a FAILED push on a push that had worked. Command substitution
# removes the pipeline entirely - `$?` on the next line is git's own code and
# there is nothing else in the chain to borrow a status from.
#
# The output is captured rather than streamed, so it reaches both the log and
# stdout AFTER the status is known. A midnight cron has no terminal reading
# stdout anyway; the log is the record.
#
# `set +e` around it because `errexit` would abort the script on a failed push
# BEFORE the status could be captured and logged, which is the one outcome
# this script exists to record.
log "Pushing $BRANCH to origin ..."
set +e
push_out=$(git push origin "$BRANCH" 2>&1)
push_status=$?
set -e
printf "%s\n" "$push_out" >> "$LOG_FILE"
printf "%s\n" "$push_out"

if (( push_status == 0 )); then
  log "PUSH OK"
else
  log "PUSH FAILED (git exit ${push_status})"
  log "=== cron_git_push END (failed) ==="
  exit "${push_status}"
fi

log "=== cron_git_push END ==="
