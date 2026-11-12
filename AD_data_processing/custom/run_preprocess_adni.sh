#!/usr/bin/env bash
# ADNI DICOM -> $PROCESSED_DIR/<subject>/<image_id>/ on the aix server. Run from AD_data_processing/.
# Resumable: finished series are skipped, so re-running after a crash or a GPU change is safe.
#   bash custom/run_preprocess_adni.sh --list-only      # check what gets picked up
#   bash custom/run_preprocess_adni.sh --limit 5        # sanity check
#   nohup bash custom/run_preprocess_adni.sh > /mnt/aix22308/team_data/ADNI_processed/logs/run_$(date +%Y%m%d_%H%M).log 2>&1 &
# Extra arguments are passed through to preprocess_adni.py.
set -euo pipefail
cd "$(dirname "$0")/.."

source custom/env_server.sh

RAW_DIR="${RAW_DIR:-/mnt/aix22308/team_data/ADNI_t1w_raw}"
# outputs: $PROCESSED_DIR/<subject>/<image_id>/<image_id>_{final,synthseg}_{1mm,1.5mm}.nii.gz
PROCESSED_DIR="${PROCESSED_DIR:-/mnt/aix22308/team_data/ADNI_processed}"
# SynthSeg on GPU 4 only (GPU 5 was handed back on 2026-10-28). Time windows are still possible,
# e.g. GPUS=5@..2026-10-27T22:30,4 (5 until then, 4 always)
GPUS="${GPUS:-4}"
# only the series acquired on a visit listed in the split files (first series of each visit first);
# SPLIT_DIR= (empty) processes every series
SPLIT_DIR="${SPLIT_DIR-/mnt/aix22307/workspace_prognosis_prediction/longiv2/data/split/ADNI_split}"
# CPU-stage concurrency follows the load average: starts at CPU_START, -1 above 0.9 x cores, +1 below
# 0.75 x cores, within [CPU_MIN, CPU_WORKERS]. To fix it by hand while running:
#   echo 6 > $PROCESSED_DIR/logs/cpu_workers     (rm the file to go back to automatic)
CPU_WORKERS="${CPU_WORKERS:-16}"
CPU_MIN="${CPU_MIN:-2}"
CPU_START="${CPU_START:-4}"
# step3_registered (N4 + SynthStrip + MNI affine, before WhiteStripe) lets intensity normalization be
# redone without the GPU; stripped (native space) lets registration be redone; the affine .mat is a few KB
KEEP="${KEEP:-stripped,registered,transforms}"
ITK_THREADS="${ITK_THREADS:-4}"
# SynthStrip runs on the CPU (FreeSurfer 8.2 ships CPU-only torch); ~8-10 GB RAM per process
STRIP_WORKERS="${STRIP_WORKERS:-4}"
STRIP_THREADS="${STRIP_THREADS:-8}"
# volumes per mri_synthseg call: its start-up (TensorFlow import from NFS) takes 6-10 min, ~20 s per volume
SEG_BATCH="${SEG_BATCH:-48}"

mkdir -p "$PROCESSED_DIR/logs"

python custom/preprocess_adni.py \
    --raw-dir      "$RAW_DIR" \
    --out-dir      "$PROCESSED_DIR" \
    --sequences    all \
    --resolutions  1.0,1.5 \
    --gpus         "$GPUS" \
    --cpu-workers  "$CPU_WORKERS" \
    --cpu-min      "$CPU_MIN" \
    --cpu-start    "$CPU_START" \
    --split-dir    "$SPLIT_DIR" \
    --itk-threads  "$ITK_THREADS" \
    --strip-workers "$STRIP_WORKERS" \
    --strip-threads "$STRIP_THREADS" \
    --seg-batch    "$SEG_BATCH" \
    --keep         "$KEEP" \
    "$@"
