#!/usr/bin/env bash
# OASIS / AIBL -> Delta-LFM CSVs (the run_adni.sh counterpart). Run from AD_data_processing/.
# Needs the `longi` env (nibabel, pandas>=2). Images are not modified; Image/ is a symlink tree.
set -euo pipefail

COHORT="${COHORT:?set COHORT, e.g. OASIS or AIBL}"
PROCESSED_DIR="${PROCESSED_DIR:-/mnt/aix22308/team_data/${COHORT}_processed}"   # written by run_preprocess_cohort.sh
# Image/ (symlinks) and derived/ (CSVs); keep it outside PROCESSED_DIR, whose subfolders are read as subjects
DATA_DIR="${DATA_DIR:?set DATA_DIR, e.g. /mnt/aix22308/data/${COHORT}_d_lfm}"
# per-scan clinical table, columns in make_cohort_csv.py: sub_id, image_id, acq_date|days, sex, diagnosis,
# dob|age_at_day0|age [, split]   (dob / age_at_day0 give ADNI-like ages that move with the scan dates)
VISITS_CSV="${VISITS_CSV:?set VISITS_CSV}"
QC_FAILED="${QC_FAILED:-}"                  # subject_id,image_id to drop; empty = no exclusion
PYTHON="${PYTHON:-$HOME/miniconda3/envs/longi/bin/python}"

# ---- decisions (EDIT): same switches as run_adni.sh
EXTRA_ARGS="${EXTRA_ARGS:-}"
DT_ARGS="${DT_ARGS:-}"                      # e.g. "--max_dt_months 180" ; empty = no filter
PAIR_DX_MISSING="${PAIR_DX_MISSING:-keep}"  # drop | keep
VOL_NORM_REF="${VOL_NORM_REF:-train}"       # train | all
VISIT_DIR="${VISIT_DIR:-date}"              # date | image_id

cd "$(dirname "$0")/.."
"$PYTHON" custom/make_cohort_csv.py \
    --cohort          "$COHORT" \
    --processed_dir   "$PROCESSED_DIR" \
    --data_dir        "$DATA_DIR" \
    --visits_csv      "$VISITS_CSV" \
    --exclude_csv     "$QC_FAILED" \
    --visit_dir       "$VISIT_DIR" \
    --pair_dx_missing "$PAIR_DX_MISSING" \
    --vol_norm_ref    "$VOL_NORM_REF" \
    $EXTRA_ARGS \
    $DT_ARGS

DERIVED_DIR="$DATA_DIR/derived" "$PYTHON" build_multiprior.py
