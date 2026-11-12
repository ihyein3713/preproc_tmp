#!/usr/bin/env bash
# Re-run custom/run_preprocess_adni.sh every INTERVAL_MIN minutes while raw series are still being added
# to ADNI_t1w_raw. Each pass is resumable: it only processes series that are new (or changed) and have
# settled for 60 min, so a download in progress is picked up by a later pass. Run from AD_data_processing/:
#   tmux new-session -d -s adni_preproc "bash custom/watch_adni.sh"
# Stop after the current pass: touch $PROCESSED_DIR/logs/stop_watch   (or kill the tmux session)
# Extra arguments and environment variables (GPUS, SPLIT_DIR, ...) are passed through to every pass.
set -uo pipefail
cd "$(dirname "$0")/.."

LOGS="${PROCESSED_DIR:-/mnt/aix22308/team_data/ADNI_processed}/logs"
INTERVAL_MIN="${INTERVAL_MIN:-60}"
mkdir -p "$LOGS"
rm -f "$LOGS/stop_watch"

while [ ! -e "$LOGS/stop_watch" ]; do
    log="$LOGS/run_$(date +%Y%m%d_%H%M).log"
    echo "$(date '+%F %T') pass -> $log"
    bash custom/run_preprocess_adni.sh "$@" > "$log" 2>&1
    echo "$(date '+%F %T') pass exited $?: $(grep -E 'already complete|done:' "$log" | tr '\n' ' ')"
    for _ in $(seq "$INTERVAL_MIN"); do
        [ -e "$LOGS/stop_watch" ] && break
        sleep 60
    done
done
echo "$(date '+%F %T') stop_watch found, exiting"
