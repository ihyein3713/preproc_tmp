#!/usr/bin/env bash
# Continue an OASIS-3 / AIBL run on a local machine after the server run stopped (STOP_NEW_AT).
# The server tree /mnt/aix22308/team_data is mounted read-only at $MOUNT (sshfs); raws, the remaining-scan
# manifest and the MNI template are read through it, outputs go to a local disk ($LOCAL_OUT).
#   sshfs -o ro,reconnect <user>@<server>:/mnt/aix22308/team_data ~/longi_v2/server_data
#   COHORT=aibl   bash custom/run_local.sh --list-only
#   COHORT=aibl   bash custom/run_local.sh            # then COHORT=oasis3
# Same pipeline and settings as the server run (run_preprocess_cohort.sh); tools must already be on PATH
# (FreeSurfer 8.2 mri_synthstrip/mri_synthseg, ANTs, intensity-normalize) with FS_LICENSE set, or give
# ENV_SCRIPT=<setup script>. Resumable like the server run: re-running skips finished scans.
set -euo pipefail
cd "$(dirname "$0")/.."

COHORT="${COHORT:?set COHORT=aibl or COHORT=oasis3}"
MOUNT="${MOUNT:-$HOME/longi_v2/server_data}"
LOCAL_OUT="${LOCAL_OUT:-$HOME/longi_v2/local_processed}"
SERVER_DIR="$MOUNT/${COHORT^^}_processed"

# newest remaining_<cohort>_*.csv written by the stopped server run
REMAINING="${REMAINING:-$(ls -1t "$SERVER_DIR"/logs/remaining_"${COHORT}"_*.csv 2>/dev/null | head -1 || true)}"
if [ -z "$REMAINING" ] || [ ! -f "$REMAINING" ]; then
    echo "no $SERVER_DIR/logs/remaining_${COHORT}_*.csv yet: the server run has not finished stopping" >&2
    exit 1
fi
echo "manifest: $REMAINING ($(($(wc -l < "$REMAINING") - 1)) scans)"

export REFERENCE_BRAIN="${REFERENCE_BRAIN:-$MOUNT/OASIS_AIBL_rerun/ref/MNI152_T1_1mm_brain.nii.gz}"
# '' = FreeSurfer's TensorFlow uses the local CUDA install as is (the server needs a pip CUDA env)
export TFCUDA_SITE_PACKAGES="${TFCUDA_SITE_PACKAGES-}"

COHORT="$COHORT" \
AFFINE=register \
MANIFEST="$REMAINING" \
PATH_PREFIX="" \
RAW_MAP="/data1/miela/workspace/longi=$MOUNT" \
PROCESSED_DIR="$LOCAL_OUT/${COHORT^^}_processed" \
ENV_SCRIPT="${ENV_SCRIPT-}" \
GPUS="${GPUS-0}" \
CPU_WORKERS="${CPU_WORKERS:-8}" \
STRIP_WORKERS="${STRIP_WORKERS:-3}" \
STRIP_THREADS="${STRIP_THREADS:-4}" \
STOP_NEW_AT="" \
    bash custom/run_preprocess_cohort.sh --min-age-min 0 "$@"
