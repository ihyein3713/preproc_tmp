#!/usr/bin/env bash
# OASIS-3 / AIBL T1 -> $PROCESSED_DIR/<subject>/<image_uid>/ with the same pipeline, settings and outputs
# as run_preprocess_adni.sh, driven by rerun/manifest.csv copied from the other server. Run from AD_data_processing/.
# Resumable: finished scans are skipped, so re-running after a crash or a GPU change is safe.
#   COHORT=aibl AFFINE=manifest GPUS= bash custom/run_preprocess_cohort.sh --list-only
#   COHORT=aibl AFFINE=manifest GPUS=<id> bash custom/run_preprocess_cohort.sh --limit 2      # sanity check
#   nohup env COHORT=aibl AFFINE=manifest GPUS=<id> bash custom/run_preprocess_cohort.sh \
#       > /mnt/aix22308/team_data/AIBL_processed/logs/run_$(date +%Y%m%d_%H%M).log 2>&1 &
# Extra arguments are passed through to preprocess_cohort.py.
set -euo pipefail
cd "$(dirname "$0")/.."

# tools (FreeSurfer, ANTs, intensity-normalize): the aix server setup by default; on another machine set
# ENV_SCRIPT=<its setup script>, or ENV_SCRIPT= when the tools are already on PATH
ENV_SCRIPT="${ENV_SCRIPT-custom/env_server.sh}"
if [ -n "$ENV_SCRIPT" ]; then source "$ENV_SCRIPT"; fi

# cohort value as written in the manifest's `cohort` column: aibl | oasis3
COHORT="${COHORT:?set COHORT=aibl or COHORT=oasis3}"
# rsync destination on this server: <XFER_DIR>/files/<source absolute path>, <XFER_DIR>/rerun/
XFER_DIR="${XFER_DIR:-/mnt/aix22308/team_data/OASIS_AIBL_rerun}"
MANIFEST="${MANIFEST:-$XFER_DIR/rerun/manifest.csv}"
# saved affines stay under <XFER_DIR>/files/data2/...; the raw folders were moved from
# <XFER_DIR>/files/data1/miela/workspace/longi/{aibl,oasis3}_t1w to /mnt/aix22308/team_data/{aibl,oasis3}_t1w
PATH_PREFIX="${PATH_PREFIX-$XFER_DIR/files}"
RAW_MAP="${RAW_MAP-/data1/miela/workspace/longi=/mnt/aix22308/team_data}"
# rerun/exclude_visits.csv is NOT passed by default: its rows that are in the manifest are the scans this
# rerun fixes (reregister / reregister_new_raw) plus one REVIEW case; QC exclusion belongs to the CSV step
EXCLUDE="${EXCLUDE:-}"
# manifest columns: cohort,subject_id,image_uid,action,note,raw_path,mat_path,old_final_path
COL_SUBJECT="${COL_SUBJECT:-subject_id}"
COL_IMAGE="${COL_IMAGE:-image_uid}"        # days since the subject's first scan (OASIS-3: dNNNN)
COL_PATH="${COL_PATH:-raw_path}"
COL_DATE="${COL_DATE:-}"
FILTER="${FILTER-cohort=$COHORT}"
# registration: register (fresh antsRegistration for every scan, as ADNI) | manifest (saved affine where
# action=apply_saved_affine, fresh registration for reregister*). No default on purpose.
AFFINE="${AFFINE:?set AFFINE=register or AFFINE=manifest}"
# outputs: $PROCESSED_DIR/<subject>/<image_uid>/<image_uid>_{final,synthseg}_{1mm,1.5mm}.nii.gz
PROCESSED_DIR="${PROCESSED_DIR:-/mnt/aix22308/team_data/${COHORT^^}_processed}"
# SynthSeg GPU(s): no default on purpose -- only a GPU assigned for this job. GPUS= (empty) = CPU.
GPUS="${GPUS?set GPUS to the assigned GPU id(s), or GPUS= for SynthSeg on the CPU}"
# same CPU settings as run_preprocess_adni.sh
CPU_WORKERS="${CPU_WORKERS:-16}"
CPU_MIN="${CPU_MIN:-2}"
CPU_START="${CPU_START:-4}"
KEEP="${KEEP:-stripped,registered,transforms}"
ITK_THREADS="${ITK_THREADS:-4}"
STRIP_WORKERS="${STRIP_WORKERS:-4}"
STRIP_THREADS="${STRIP_THREADS:-8}"
SEG_BATCH="${SEG_BATCH:-48}"
# start no new scan after this local time (YYYY-MM-DDTHH:MM; touching $PROCESSED_DIR/logs/STOP does the same);
# the unfinished manifest rows are then written to $PROCESSED_DIR/logs/remaining_<cohort>_<time>.csv
STOP_NEW_AT="${STOP_NEW_AT:-}"

mkdir -p "$PROCESSED_DIR/logs"

FILTER_ARGS=()
[ -n "$FILTER" ] && FILTER_ARGS+=(--filter "$FILTER")
[ -n "$RAW_MAP" ] && FILTER_ARGS+=(--path-map "$RAW_MAP")

python custom/preprocess_cohort.py \
    --cohort        "$COHORT" \
    --manifest      "$MANIFEST" \
    --col-subject   "$COL_SUBJECT" \
    --col-image     "$COL_IMAGE" \
    --col-path      "$COL_PATH" \
    --col-date      "$COL_DATE" \
    --affine        "$AFFINE" \
    --path-prefix   "$PATH_PREFIX" \
    --exclude       "$EXCLUDE" \
    --out-dir       "$PROCESSED_DIR" \
    --resolutions   1.0,1.5 \
    --gpus          "$GPUS" \
    --cpu-workers   "$CPU_WORKERS" \
    --cpu-min       "$CPU_MIN" \
    --cpu-start     "$CPU_START" \
    --itk-threads   "$ITK_THREADS" \
    --strip-workers "$STRIP_WORKERS" \
    --strip-threads "$STRIP_THREADS" \
    --seg-batch     "$SEG_BATCH" \
    --keep          "$KEEP" \
    --stop-new-at   "$STOP_NEW_AT" \
    ${FILTER_ARGS[@]+"${FILTER_ARGS[@]}"} \
    "$@"
