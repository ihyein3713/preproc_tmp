#!/usr/bin/env bash
# ADNI -> Delta-LFM CSVs (replaces s1/s2/s3 for ADNI). Run from AD_data_processing/.
# Needs the `longi` env (nibabel, pandas>=2). Images are not modified; Image/ is a symlink tree.
set -euo pipefail

PROCESSED_DIR="${PROCESSED_DIR:-/mnt/aix22308/team_data/ADNI_processed}"   # written by run_preprocess_adni.sh
# Image/ (symlinks) and derived/ (CSVs); keep it outside PROCESSED_DIR, whose subfolders are read as subjects
DATA_DIR="${DATA_DIR:?set DATA_DIR, e.g. /mnt/aix22308/team_data/ADNI_dlfm}"
SPLIT_DIR="${SPLIT_DIR:-/mnt/aix22307/workspace_prognosis_prediction/longiv2/data/split/ADNI_split}"
META_DIR="${META_DIR:-/mnt/aix22307/workspace_prognosis_prediction/longiv2/data/ADNI_metadata}"
DXSUM="${DXSUM:-$META_DIR/DXSUM_19Jul2026.csv}"
PTDEMOG="${PTDEMOG:-$META_DIR/PTDEMOG_19Jul2026.csv}"
# QC list for this run (the old one, adni_qc_failed_old.csv, is for the previous local run); empty = no exclusion
QC_FAILED="${QC_FAILED:-}"
# nibabel/pandas live in the `longi` env; override with PYTHON=... if it sits elsewhere
PYTHON="${PYTHON:-$HOME/miniconda3/envs/longi/bin/python}"

# ---- decisions (EDIT): dt15 filter, missing-DX pairs, volume-normalization reference
EXTRA_ARGS="${EXTRA_ARGS:-}"
DT_ARGS="${DT_ARGS:-}"                      # e.g. "--max_dt_months 60" ; empty = no filter
PAIR_DX_MISSING="${PAIR_DX_MISSING:-keep}"  # drop | keep
VOL_NORM_REF="${VOL_NORM_REF:-train}"       # train split only (avoids the leakage of the released make_csv_A) | all
DX_WINDOW_DAYS="${DX_WINDOW_DAYS:-0}"       # 0 = no limit; nearest DXSUM exam always wins
VISIT_DIR="${VISIT_DIR:-date}"              # Image/<subject>/<visit>/ : date | image_id

"$PYTHON" custom/make_adni_csv.py \
    --processed_dir   "$PROCESSED_DIR" \
    --data_dir        "$DATA_DIR" \
    --split_dir       "$SPLIT_DIR" \
    --dxsum           "$DXSUM" \
    --ptdemog         "$PTDEMOG" \
    --exclude_csv     "$QC_FAILED" \
    --dx_window_days  "$DX_WINDOW_DAYS" \
    --visit_dir       "$VISIT_DIR" \
    --pair_dx_missing "$PAIR_DX_MISSING" \
    --vol_norm_ref    "$VOL_NORM_REF" \
    $EXTRA_ARGS \
    $DT_ARGS

DERIVED_DIR="$DATA_DIR/derived" "$PYTHON" build_multiprior.py
