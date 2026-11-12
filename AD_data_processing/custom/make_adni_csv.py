#!/usr/bin/env python
"""Build the Delta-LFM CSVs for ADNI from the preprocessed tree (replaces s1/s2/s3,
which are OASIS-specific).

Input  : <processed_dir>/<sub_id>/<img_id>/<img_id>_{final,synthseg}_1.5mm.nii.gz
         <processed_dir>/preprocess_log.jsonl   acquisition date/time + sequence name, parsed
                                                from the raw DICOM folder path
         <split_dir>/{train,val,test}.txt       one scan per line: <sub_id>_MR_<YYYY-MM-DD>
         DXSUM csv                              diagnosis (PTID, EXAMDATE, DIAGNOSIS 1/2/3)
         PTDEMOG csv                            sex and date of birth (PTGENDER, PTDOB)
Output : <data_dir>/Image/ADNI_<sub_id>/<visit>/{t1,seg}.nii.gz    symlinks, images untouched
                                                                   visit = scan date (default) or image id
         <data_dir>/derived/AD-Progression-All.csv          one row per visit, image_path relative to data_dir
         <data_dir>/derived/AD-Progression-All-abs.csv      same, image_path absolute
         <data_dir>/derived/AD-Progression-pairs-dt15.csv   all (i<j) pairs per subject
         <data_dir>/derived/adni_volumes_raw.csv            voxel-count cache
         <data_dir>/derived/adni_build_report.txt           counts and validation checks
Then run build_multiprior.py with DERIVED_DIR=<data_dir>/derived.

Encoding: age in years (All) and years/100 (pairs); follow_up in months since the subject's
first kept visit (days/30.44); sex F=0, M=1; diagnosis CN=0, MCI=0.5, AD=1, missing=-1.
Every table carries a `split` column (train/val/test) taken from the split files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import const  # noqa: E402

DAYS_PER_MONTH = 30.44
SPLITS = ("train", "val", "test")
PAIR_CONTEXT_COLS = ["starting_age", "sex", "starting_diagnosis"] + [f"starting_{r}" for r in const.CONDITIONING_REGIONS]
TARGET_SHAPE = (128, 144, 128)   # ResizeWithPadOrCrop target in the Delta-LFM loaders
NUM_CLASS_EMBEDS = 15            # init_large_latent_diffusion(num_class_embeds=15): the flow model
                                 # embeds the interval in whole years, so age_gap must stay <= 14
DXSUM_CODE = {1: 0.0, 2: 0.5, 3: 1.0}             # CN, MCI, Dementia
PTGENDER_CODE = {1: 1.0, 2: 0.0}                   # ADNI: 1=Male, 2=Female -> M=1, F=0
DX_NAME = {0.0: "CN", 0.5: "MCI", 1.0: "AD"}

DEFAULT_SPLIT_DIR = "/mnt/aix22307/workspace_prognosis_prediction/longiv2/data/split/ADNI_split"
DEFAULT_META_DIR = "/mnt/aix22307/workspace_prognosis_prediction/longiv2/data/ADNI_metadata"


# --------------------------------------------------------------------------- helpers

def split_v3_is_train(subject_str: str) -> bool:
    """Same hash as oasis_dataset_3D_pair_latent.get_dataframe(split_v3=1)."""
    return int(hashlib.md5(("v3|" + str(subject_str)).encode()).hexdigest()[:8], 16) % 1000 < 800


def count_labels(seg_path: str) -> dict:
    import nibabel as nib
    seg = np.rint(np.asanyarray(nib.load(seg_path).dataobj)).astype(np.int64).ravel()
    counts = np.bincount(seg[seg >= 0], minlength=256)
    rec = {"head_size": int((seg > 0).sum())}
    for region in const.COARSE_REGIONS:
        rec[region] = 0
    for code, region in const.SYNTHSEG_CODEMAP.items():
        if region == "background":
            continue
        rec[region.replace("left_", "").replace("right_", "")] += int(counts[code])
    return rec


def pad_crop_bounds(size: int, target: int) -> tuple[int, int, int]:
    """(pad_before, crop_start, crop_end) matching MONAI ResizeWithPadOrCrop (checked on monai 1.6)."""
    if size < target:
        return (target - size) // 2, 0, size
    start = size // 2 - target // 2
    return 0, start, start + target


# --------------------------------------------------------------------------- loading

def discover(processed_dir: Path, tag: str) -> tuple[pd.DataFrame, list]:
    rows, incomplete = [], []
    for sub_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir() and p.name != "logs"):
        for img_dir in sorted(p for p in sub_dir.iterdir() if p.is_dir()):
            img = img_dir.name
            t1 = img_dir / f"{img}_final_{tag}.nii.gz"
            seg = img_dir / f"{img}_synthseg_{tag}.nii.gz"
            if t1.exists() and seg.exists():
                rows.append({"sub_id": sub_dir.name, "image_id": img,
                             "src_t1": str(t1.resolve()), "src_seg": str(seg.resolve())})
            else:
                incomplete.append(f"{sub_dir.name}/{img}")
    return pd.DataFrame(rows), incomplete


def load_log(log_path: Path) -> pd.DataFrame:
    """Last entry per image -> acquisition datetime and sequence folder name.

    Failed entries are kept: the date comes from the raw DICOM path, not from the outcome, and a
    scan can have outputs on disk while its successful re-run was never appended to the log.
    Raw layout: <raw>/<subject>/<sequence>/<YYYY-MM-DD_HH_MM_SS.f>/<image_id>."""
    recs = {}
    with open(log_path) as f:
        for line in f:
            r = json.loads(line)
            parts = r["input"].rstrip("/").split("/")
            m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})", parts[-2])
            recs[parts[-1]] = {
                "image_id": parts[-1],
                "sequence": parts[-3],
                "acq_datetime": pd.Timestamp(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}") if m else pd.NaT,
                "log_status": r.get("status", ""),
            }
    return pd.DataFrame(list(recs.values()))


def load_split(split_dir: Path) -> pd.DataFrame:
    rows = []
    for sp in SPLITS:
        for line in (split_dir / f"{sp}.txt").read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.fullmatch(r"(\d{3}_S_\d+)_MR_(\d{4}-\d{2}-\d{2})", line)
            if m is None:
                raise ValueError(f"unexpected line in {sp}.txt: {line!r}")
            rows.append({"sub_id": m.group(1), "acq_date": pd.Timestamp(m.group(2)), "split": sp, "scan_key": line})
    df = pd.DataFrame(rows)
    multi = df.groupby("sub_id")["split"].nunique()
    if (multi > 1).any():
        raise ValueError(f"subjects in more than one split: {multi[multi > 1].index.tolist()[:10]}")
    return df.drop_duplicates(["sub_id", "acq_date"])


def load_dxsum(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    date = pd.to_datetime(df["EXAMDATE"], errors="coerce").fillna(pd.to_datetime(df["USERDATE"], errors="coerce"))
    out = pd.DataFrame({"sub_id": df["PTID"].astype(str).str.strip(), "examdate": date,
                        "dx": pd.to_numeric(df["DIAGNOSIS"], errors="coerce").map(DXSUM_CODE)})
    return out.dropna(subset=["examdate", "dx"])


def load_ptdemog(path: str, R: list) -> pd.DataFrame:
    """One row per subject: sex (mode of the valid codes) and date of birth (earliest record).
    PTDOB is MM/YYYY; the 15th of the month is used, so the age error is under 0.05 y."""
    df = pd.read_csv(path, low_memory=False)
    df["sub_id"] = df["PTID"].astype(str).str.strip()
    df["_date"] = pd.to_datetime(df["VISDATE"], errors="coerce").fillna(pd.to_datetime(df["USERDATE"], errors="coerce"))
    df = df.sort_values(["sub_id", "_date"])
    df["sex"] = pd.to_numeric(df["PTGENDER"], errors="coerce").map(PTGENDER_CODE)
    df["dob"] = pd.to_datetime(df["PTDOB"], format="%m/%Y", errors="coerce") + pd.Timedelta(days=14)
    valid_sex = df.dropna(subset=["sex"]).groupby("sub_id")["sex"]
    sex = valid_sex.agg(lambda s: s.mode().iloc[0] if s.mode().size == 1 else s.iloc[0])
    dob = df.dropna(subset=["dob"]).groupby("sub_id")["dob"].first()
    n_sex = int((valid_sex.nunique() > 1).sum())
    n_dob = int((df.dropna(subset=["dob"]).groupby("sub_id")["dob"].nunique() > 1).sum())
    R.append(f"PTDEMOG: {len(sex)} subjects with sex, {len(dob)} with DOB; "
             f"conflicting records across visits: sex {n_sex}, DOB {n_dob} (mode / earliest record used)")
    return pd.concat([sex, dob], axis=1).reset_index()


# --------------------------------------------------------------------------- per-visit table

def build_visits(scans: pd.DataFrame, split: pd.DataFrame | None, dxsum: pd.DataFrame, demog: pd.DataFrame,
                 args, R: list) -> pd.DataFrame:
    df = scans.copy()
    no_date = df["acq_datetime"].isna()
    if no_date.any():
        R.append(f"DROP {int(no_date.sum())} scans without an acquisition date in the log: "
                 f"{df.loc[no_date, 'image_id'].tolist()[:10]}")
        df = df[~no_date].copy()
    df["acq_date"] = df["acq_datetime"].dt.normalize()

    # ---- one scan per (subject, day): repeats / several T1 series of the same session.
    # Same rule as the Delta-LFM triplet loader, which sorts by ["follow_up", "capture_time"] and
    # keeps the first row of each follow_up: the earliest acquisition time of that day wins.
    df["is_repeat"] = df["sequence"].fillna("").str.contains("repeat", case=False)
    df["_img_num"] = df["image_id"].str[1:].astype(np.int64)
    df = df.sort_values(["sub_id", "acq_date", "acq_datetime", "_img_num"], kind="mergesort")
    n_before = len(df)
    if not args.keep_same_day:
        df = df.drop_duplicates(["sub_id", "acq_date"], keep="first")
    R.append(f"same-day dedup: {n_before} scans -> {len(df)} visits "
             f"({'disabled' if args.keep_same_day else 'earliest acquisition time, as in the Delta-LFM loader'})")

    # ---- split: keep only the visits listed in the split files
    if split is not None:
        ours_sub = set(df["sub_id"])
        df = df.merge(split[["sub_id", "acq_date", "split", "scan_key"]], on=["sub_id", "acq_date"], how="left")
        out_sub = sorted(ours_sub - set(split["sub_id"]))
        in_sub = df["sub_id"].isin(set(split["sub_id"]))
        R.append(f"split: {len(ours_sub) - len(out_sub)}/{len(ours_sub)} of our subjects are in the split; "
                 f"{len(out_sub)} subjects not in any split file dropped, e.g. {out_sub[:5]}")
        R.append(f"  visits of split subjects not listed in the split (e.g. <30 day gap): "
                 f"{int((in_sub & df['split'].isna()).sum())} dropped")
        have = set(zip(df["sub_id"], df["acq_date"]))
        listed = split[split["sub_id"].isin(ours_sub)]
        miss = listed[[k not in have for k in zip(listed["sub_id"], listed["acq_date"])]]
        R.append(f"  listed visits of our subjects with no processed scan: {len(miss)} {miss['scan_key'].tolist()[:6]}")
        df = df[df["split"].notna()].copy()
    else:
        df["split"] = np.where(df["sub_id"].map(lambda s: split_v3_is_train("ADNI_" + s)), "train", "test")
        df["scan_key"] = df["sub_id"] + "_MR_" + df["acq_date"].dt.strftime("%Y-%m-%d")
        R.append("split: no --split_dir, split column derived from the split_v3 hash (train/test)")

    # ---- sex and age (age from date of birth, so age deltas match the scan dates)
    df = df.merge(demog, on="sub_id", how="left")
    df["age"] = (df["acq_date"] - df["dob"]).dt.days / 365.25
    bad = df["age"].isna() | df["sex"].isna()
    if bad.any():
        R.append(f"DROP {int(bad.sum())} visits / {df.loc[bad, 'sub_id'].nunique()} subjects with no sex or DOB "
                 f"in PTDEMOG: {sorted(df.loc[bad, 'sub_id'].unique())[:10]}")
        df = df[~bad].copy()

    # ---- follow_up / capture_time (after all drops, so each subject starts at 0)
    first = df.groupby("sub_id")["acq_date"].transform("min")
    df["follow_up"] = (df["acq_date"] - first).dt.days / DAYS_PER_MONTH
    df["capture_time"] = df["acq_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # ---- diagnosis: DXSUM row of the same subject with the nearest EXAMDATE, within the window
    df["diagnosis"] = -1.0
    df["dx_gap_days"] = np.nan
    by = {k: g for k, g in dxsum.groupby("sub_id")}
    for i, r in df.iterrows():
        g = by.get(r.sub_id)
        if g is None:
            continue
        gap = (g["examdate"] - r.acq_date).dt.days.abs()
        j = gap.idxmin()
        df.at[i, "dx_gap_days"] = gap[j]
        if args.dx_window_days <= 0 or gap[j] <= args.dx_window_days:
            df.at[i, "diagnosis"] = g.at[j, "dx"]
    return df


def add_volumes(df: pd.DataFrame, cache_path: Path, workers: int, recompute: bool, R: list) -> pd.DataFrame:
    cache = pd.read_csv(cache_path) if cache_path.exists() and not recompute else pd.DataFrame({"src_seg": []})
    todo = sorted(set(df["src_seg"]) - set(cache["src_seg"]))
    if todo:
        print(f"[volumes] counting labels in {len(todo)} segmentations", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(count_labels, todo, chunksize=8))
        cache = pd.concat([cache, pd.DataFrame(recs).assign(src_seg=todo)], ignore_index=True)
        cache = cache.drop_duplicates("src_seg", keep="last")
        cache.to_csv(cache_path, index=False)
    R.append(f"volumes: {len(todo)} computed, {len(df) - len(todo)} from cache ({cache_path.name})")
    return df.merge(cache, on="src_seg", how="left")


def normalize_volumes(df: pd.DataFrame, ref: str, R: list) -> pd.DataFrame:
    """Min-max per coarse region (as make_csv_A); reference = train-split visits or all visits."""
    mask = df["split"].eq("train") if ref == "train" else pd.Series(True, index=df.index)
    for region in const.COARSE_REGIONS + ["head_size"]:
        v = df.loc[mask, region].astype(float)
        lo, hi = v.min(), v.max()
        df[region] = (df[region].astype(float) - lo) / (hi - lo if hi > lo else 1.0)
    msg = f"volume min-max reference: {ref} ({int(mask.sum())} visits)"
    if ref == "train":
        out = df.loc[~mask, const.CONDITIONING_REGIONS]
        msg += f"; val/test visits outside [0,1] in a conditioning region: {int(((out < 0) | (out > 1)).any(axis=1).sum())}"
    R.append(msg)
    return df


# --------------------------------------------------------------------------- pairs

def make_pairs(allv: pd.DataFrame, args, R: list) -> pd.DataFrame:
    per_visit = ["image_path", "visit", "image_id", "scan_key", "age", "follow_up", "capture_time", "diagnosis", "source",
                 "head_size"] + const.COARSE_REGIONS
    recs = []
    for sid, g in allv.groupby("subject_id", sort=True):
        g = g.sort_values(["follow_up", "capture_time"], kind="mergesort").reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                s, e = g.iloc[i], g.iloc[j]
                rec = {"subject_id": sid, "split": s["split"], "sex": s["sex"]}
                for c in per_visit:
                    rec[f"starting_{c}"] = s[c]
                    rec[f"followup_{c}"] = e[c]
                recs.append(rec)
    pairs = pd.DataFrame(recs)
    pairs["starting_age"] = pairs["starting_age"] / 100.0
    pairs["followup_age"] = pairs["followup_age"] / 100.0
    dt = pairs["followup_follow_up"] - pairs["starting_follow_up"]
    R.append(f"pairs: {len(pairs)} (all i<j)")

    keep = dt > 0
    R.append(f"  drop dt==0: {int((~keep).sum())}")
    n0 = int(keep.sum())
    if args.min_dt_months is not None:
        keep &= dt >= args.min_dt_months
    if args.max_dt_months is not None:
        keep &= dt <= args.max_dt_months
    R.append(f"  dt filter [{args.min_dt_months}, {args.max_dt_months}] months: {n0} -> {int(keep.sum())}")
    if args.pair_dx_missing == "drop":
        n1 = int(keep.sum())
        keep &= (pairs["starting_diagnosis"] >= 0) & (pairs["followup_diagnosis"] >= 0)
        R.append(f"  drop pairs with a missing diagnosis: {n1} -> {int(keep.sum())}")
    pairs = pairs[keep].copy()

    # train, val, test blocks in that order: _cond_stats takes its statistics from the first 80% of rows
    rank = pairs["split"].map({s: i for i, s in enumerate(SPLITS)})
    if args.pair_order == "subject":
        rank = 0
    pairs = pairs.assign(_r=rank).sort_values(["_r", "subject_id", "starting_follow_up", "followup_follow_up"],
                                              kind="mergesort")
    return pairs.drop(columns="_r").reset_index(drop=True)


# --------------------------------------------------------------------------- validation

def validate(allv: pd.DataFrame, allv_abs: pd.DataFrame, pairs: pd.DataFrame, audit: pd.DataFrame | None,
             n_check: int, R: list):
    R.append("\n=== validation ===")
    # 1. files
    t1 = allv_abs["image_path"].map(os.path.exists)
    seg = allv_abs["seg_path"].map(os.path.exists)
    R.append(f"[1] t1 exists {int(t1.sum())}/{len(t1)}, seg exists {int(seg.sum())}/{len(seg)}")

    # 2. counts
    nv = allv.groupby("subject_id").size()
    R.append(f"[2] subjects {allv.subject_id.nunique()}, visits {len(allv)}, pairs {len(pairs)}; subjects with "
             f">=3 visits {int((nv >= 3).sum())} (step1 min_timepoints=3), 2 visits {int((nv == 2).sum())}, "
             f"1 visit {int((nv == 1).sum())}")
    for sp in SPLITS:
        a, p = allv[allv.split == sp], pairs[pairs.split == sp]
        R.append(f"    {sp:5s}: {a.subject_id.nunique()} subjects, {len(a)} visits, {len(p)} pairs")

    # 3. ranges / monotonicity / duplicates
    mono = allv.sort_values(["subject_id", "capture_time"]).groupby("subject_id")["follow_up"] \
        .apply(lambda s: s.is_monotonic_increasing).all()
    dup = int(allv.duplicated(["subject_id", "follow_up"]).sum())
    R.append(f"[3] age {allv.age.min():.1f}-{allv.age.max():.1f} y; follow_up 0-{allv.follow_up.max():.1f} months; "
             f"follow_up monotone in capture_time: {mono}; duplicate (subject, follow_up): {dup}")
    if len(pairs):
        dtm = pairs["followup_follow_up"] - pairs["starting_follow_up"]
        R.append(f"    pair dt months: min {dtm.min():.1f}, median {dtm.median():.1f}, max {dtm.max():.1f}; "
                 f"pair age range {pairs.starting_age.min():.3f}-{pairs.followup_age.max():.3f} (/100)")

    # 4. diagnosis / sex
    miss = int((allv.diagnosis < 0).sum())
    R.append(f"[4] diagnosis unmatched {miss}/{len(allv)} ({100 * miss / max(len(allv), 1):.1f}%); "
             f"distribution {allv.diagnosis.map(DX_NAME).fillna('missing').value_counts().to_dict()}; "
             f"sex (F=0, M=1) {allv.sex.value_counts().sort_index().to_dict()}")
    gap = allv["dx_gap_days"].dropna()
    if len(gap):
        R.append(f"    nearest EXAMDATE gap days: median {gap.median():.0f}, p90 {gap.quantile(.9):.0f}, max {gap.max():.0f}")
    if audit is not None:
        g = allv.sort_values("follow_up").groupby("sub_id")
        ours = pd.DataFrame({"first": g["diagnosis"].first().map(DX_NAME), "last": g["diagnosis"].last().map(DX_NAME),
                             "split": g["split"].first()})
        a = audit.drop_duplicates("subject_id").set_index("subject_id").reindex(ours.index)
        ok = a["dx_first"].notna()
        if ok.any():
            R.append(f"    vs split audit (subject_audit.csv, {int(ok.sum())} subjects): "
                     f"dx_first agree {100 * (ours.loc[ok, 'first'] == a.loc[ok, 'dx_first']).mean():.1f}%, "
                     f"dx_last agree {100 * (ours.loc[ok, 'last'] == a.loc[ok, 'dx_last']).mean():.1f}%, "
                     f"split agree {100 * (ours.loc[ok, 'split'] == a.loc[ok, 'split']).mean():.1f}%")
    nan_ctx = pairs[PAIR_CONTEXT_COLS + ["followup_diagnosis"]].isna().sum()
    R.append(f"    NaN in pair context columns: {int(nan_ctx.sum())}"
             + (f" -> {nan_ctx[nan_ctx > 0].to_dict()}" if nan_ctx.sum() else ""))
    if (pairs[["starting_diagnosis", "followup_diagnosis"]] < 0).any(axis=None):
        R.append("    WARNING pairs contain diagnosis=-1; step3 feeds followup_dia-starting_dia into the context")

    # 4b. the model's class embedding indexes the interval in whole years
    gap = ((pairs["followup_age"] - pairs["starting_age"]) * 100).astype(int)
    over = int((gap >= NUM_CLASS_EMBEDS).sum())
    R.append(f"[4b] age_gap = int((followup_age - starting_age) * 100), the class index of the flow model: "
             f"max {int(gap.max()) if len(gap) else 0}, pairs >= num_class_embeds ({NUM_CLASS_EMBEDS}): {over}")
    if over:
        R.append(f"    ERROR these pairs index past the class embedding and abort training with a CUDA "
                 f"device-side assert; filter them with --max_dt_months (180 keeps every gap <= 14)")

    # 5. split as the Delta-LFM loaders would see it
    sid3 = pairs["starting_image_path"].map(lambda x: str(x).split("/")[-3])
    h_tr = sid3.map(split_v3_is_train)
    R.append(f"[5] current loader --split_v3 1 (md5 hash): train {sid3[h_tr].nunique()} subj / {int(h_tr.sum())} rows, "
             f"test {sid3[~h_tr].nunique()} subj / {int((~h_tr).sum())} rows")
    cross = pd.crosstab(pairs["split"], h_tr.map({True: "hash_train", False: "hash_test"}))
    R.append("    split files vs split_v3 hash (pair rows):\n" + "\n".join("      " + l for l in cross.to_string().splitlines()))
    n80 = int(0.8 * len(pairs))
    R.append(f"    rows 0..80% (row-wise split and _cond_stats): {pairs.iloc[:n80].split.value_counts().to_dict()}; "
             f"last 20%: {pairs.iloc[n80:].split.value_counts().to_dict()}")

    # 6. geometry sample
    import nibabel as nib
    rng = np.random.default_rng(0)
    idx = rng.choice(len(allv_abs), size=min(n_check, len(allv_abs)), replace=False)
    R.append(f"[6] geometry check on {len(idx)} random visits (pad/crop to {TARGET_SHAPE})")
    for k in idx:
        r = allv_abs.iloc[k]
        im = nib.load(r.image_path)
        sg = np.asanyarray(nib.load(r.seg_path).dataobj)
        zooms = tuple(float(z) for z in im.header.get_zooms()[:3])
        fg = np.argwhere(sg > 0)
        kept, margins = True, []
        for ax in range(3):
            pb, cs, ce = pad_crop_bounds(im.shape[ax], TARGET_SHAPE[ax])
            lo, hi = fg[:, ax].min(), fg[:, ax].max()
            kept &= bool(lo >= cs and hi < ce)
            margins.append(int(min(lo - cs, ce - 1 - hi) + pb))
        d = np.asanyarray(im.dataobj).astype(np.float32)
        bg01 = (float(np.median(d[sg == 0])) - d.min()) / (d.max() - d.min() + 1e-8)
        R.append(f"    {r.subject_id}/{r.visit}: shape {im.shape} spacing {zooms} "
                 f"{'OK' if np.allclose(zooms, 1.5) else 'SPACING!=1.5'}; brain kept {kept}, "
                 f"min margin to border (vox) {margins}; background after ScaleIntensity {bg01:.3f} (0 expected)")

    # 7. prior matching
    abs_set = set(allv_abs["image_path"])
    ms = pairs["starting_image_path"].isin(abs_set).mean() if len(pairs) else 1.0
    mf = pairs["followup_image_path"].isin(abs_set).mean() if len(pairs) else 1.0
    R.append(f"[7] pair paths found in All-abs: starting {100 * ms:.1f}%, followup {100 * mf:.1f}%")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True, help="<sub_id>/<img_id>/ tree written by preprocess_adni.py")
    ap.add_argument("--data_dir", required=True, help="Delta-LFM DATA_DIR; Image/ and derived/ are created here")
    ap.add_argument("--preprocess_log", default=None, help="default: <processed_dir>/preprocess_log.jsonl")
    ap.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR,
                    help="dir with train/val/test.txt (<sub_id>_MR_<date>) and audit/; '' = split_v3 hash instead")
    ap.add_argument("--dxsum", default=f"{DEFAULT_META_DIR}/DXSUM_19Jul2026.csv")
    ap.add_argument("--ptdemog", default=f"{DEFAULT_META_DIR}/PTDEMOG_19Jul2026.csv")
    ap.add_argument("--exclude_csv", default="",
                    help="CSV with subject_id,image_id to drop (visual QC); '' (default) = no exclusion")
    ap.add_argument("--resolution_tag", default="1.5mm")
    ap.add_argument("--visit_dir", choices=["date", "image_id"], default="date",
                    help="name of the visit folder under Image/<subject>/. date (default) keeps the eval pair "
                         "keys (subject__startdate__followdate) readable and comparable with other methods; "
                         "image_id mirrors the processed tree. Must be unique within a subject")
    ap.add_argument("--dx_window_days", type=int, default=0,
                    help="max |EXAMDATE - scan date| for a DX match; 0 = no limit (default). The nearest DXSUM "
                         "exam of the same subject reproduces the team's per-scan labels in 303/308 cases, and "
                         "ADNI diagnoses are carried between annual exams, so a hard cutoff only creates gaps")
    ap.add_argument("--keep_same_day", action="store_true",
                    help="keep every T1 of a session (default: one per day, the earliest acquisition time)")
    ap.add_argument("--vol_norm_ref", choices=["train", "all"], default="train",
                    help="min-max reference: train-split visits only (default; the released make_csv_A uses every "
                         "visit, which leaks val/test statistics into the conditioning variables) or all visits")
    ap.add_argument("--min_dt_months", type=float, default=None, help="pair filter: keep dt >= this")
    ap.add_argument("--max_dt_months", type=float, default=None, help="pair filter: keep dt <= this")
    ap.add_argument("--pair_dx_missing", choices=["drop", "keep"], default="keep",
                    help="pairs with diagnosis=-1 at either end: drop, or keep as -1 (default)")
    ap.add_argument("--pair_order", choices=["split", "subject"], default="split",
                    help="row order of the pair CSV: train/val/test blocks (default) or subject only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--recompute_volumes", action="store_true")
    ap.add_argument("--n_check", type=int, default=6, help="volumes to load for the geometry check")
    ap.add_argument("--force_links", action="store_true", help="replace existing links that point elsewhere")
    args = ap.parse_args()

    processed = Path(args.processed_dir).resolve()
    data_dir = Path(os.path.abspath(args.data_dir))       # abspath, not resolve: keep the link-tree path
    derived = data_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    R: list[str] = [
        "NOTE Delta-LFM splits by md5 hash (step1: 'ap|', step3/eval: split_v3 'v3|') and does not read the",
        "     `split` column; the loaders need a patch to follow the split files.",
        "",
    ]

    # ---- scans
    scans, incomplete = discover(processed, args.resolution_tag)
    R.append(f"discovered {len(scans)} complete scans / {scans.sub_id.nunique()} subjects; "
             f"incomplete dirs skipped: {len(incomplete)} {incomplete[:5]}")
    if args.exclude_csv:
        ex = pd.read_csv(args.exclude_csv)
        bad = set(zip(ex["subject_id"].astype(str), ex["image_id"].astype(str)))
        m = np.array([(s, i) in bad for s, i in zip(scans.sub_id, scans.image_id)], dtype=bool)
        scans = scans[~m]
        R.append(f"QC exclusion ({args.exclude_csv}): dropped {int(m.sum())}/{len(bad)} listed scans")
    log = load_log(Path(args.preprocess_log or processed / "preprocess_log.jsonl"))
    scans = scans.merge(log, on="image_id", how="left")
    R.append(f"preprocess log: acquisition date for {int(scans.acq_datetime.notna().sum())}/{len(scans)} scans")
    _lf = scans["log_status"].fillna("") .ne("ok")
    if _lf.any():
        # outputs are on disk although the last log entry is a failure (re-run not appended to the log)
        R.append(f"  {int(_lf.sum())} scans have complete outputs but no 'ok' log entry, kept: "
                 f"{scans.loc[_lf, 'image_id'].tolist()[:10]}")

    split = load_split(Path(args.split_dir)) if args.split_dir else None
    audit_path = Path(args.split_dir) / "audit" / "subject_audit.csv" if args.split_dir else None
    audit = pd.read_csv(audit_path) if audit_path is not None and audit_path.exists() else None
    dxsum = load_dxsum(args.dxsum)
    demog = load_ptdemog(args.ptdemog, R)

    df = build_visits(scans, split, dxsum, demog, args, R)
    df["subject_id"] = "ADNI_" + df["sub_id"]
    df["source"] = "ADNI"

    # ---- visit folder name: one per subject and visit, used as Image/<subject>/<visit>/ and in the
    # latent tree, the derived artifacts and the eval pair keys
    df["visit"] = df["acq_date"].dt.strftime("%Y-%m-%d") if args.visit_dir == "date" else df["image_id"]
    dup = df.duplicated(["subject_id", "visit"])
    if dup.any():
        raise SystemExit(f"--visit_dir {args.visit_dir}: {int(dup.sum())} duplicate (subject, visit) names, e.g. "
                         f"{df.loc[dup, ['subject_id', 'visit']].head(5).values.tolist()}; "
                         f"use --visit_dir image_id or drop --keep_same_day")

    # ---- link tree
    n_new = 0
    for r in df.itertuples():
        d = data_dir / "Image" / r.subject_id / r.visit
        d.mkdir(parents=True, exist_ok=True)
        for name, src in (("t1.nii.gz", r.src_t1), ("seg.nii.gz", r.src_seg)):
            link = d / name
            if link.is_symlink() or link.exists():
                if os.path.realpath(link) == src:
                    continue
                if not args.force_links:
                    raise SystemExit(f"{link} exists and points to {os.path.realpath(link)}, not {src} (use --force_links)")
                link.unlink()
            link.symlink_to(src)
            n_new += 1
    R.append(f"links: {n_new} created under {data_dir / 'Image'} (visit folder = {args.visit_dir})")
    df["image_path"] = "Image/" + df["subject_id"] + "/" + df["visit"] + "/t1.nii.gz"
    df["seg_path"] = "Image/" + df["subject_id"] + "/" + df["visit"] + "/seg.nii.gz"

    # ---- volumes
    df = add_volumes(df, derived / "adni_volumes_raw.csv", args.workers, args.recompute_volumes, R)
    df = normalize_volumes(df, args.vol_norm_ref, R)

    # ---- per-visit tables
    cols = ["subject_id", "sub_id", "split", "scan_key", "visit", "image_id", "image_path", "seg_path", "follow_up",
            "capture_time", "age", "sex", "diagnosis", "dx_gap_days", "source", "sequence", "head_size"] \
        + const.COARSE_REGIONS
    allv = df.sort_values(["subject_id", "follow_up", "capture_time"])[cols].reset_index(drop=True)
    allv_abs = allv.copy()
    for c in ("image_path", "seg_path"):
        allv_abs[c] = str(data_dir) + "/" + allv_abs[c]
    allv.to_csv(derived / "AD-Progression-All.csv", index=False)
    allv_abs.to_csv(derived / "AD-Progression-All-abs.csv", index=False)

    # ---- pairs (absolute paths, same strings as All-abs)
    pairs = make_pairs(allv_abs, args, R)
    pairs.to_csv(derived / "AD-Progression-pairs-dt15.csv", index=False)

    validate(allv, allv_abs, pairs, audit, args.n_check, R)
    R.append(f"\nwritten to {derived}: AD-Progression-All.csv, AD-Progression-All-abs.csv, AD-Progression-pairs-dt15.csv")
    R.append(f"next: DERIVED_DIR={derived} python build_multiprior.py")
    text = "\n".join(R)
    print(text)
    (derived / "adni_build_report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
