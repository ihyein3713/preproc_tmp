#!/usr/bin/env python
"""Build the Delta-LFM CSVs for a non-ADNI cohort (OASIS, AIBL) from the tree written by
preprocess_cohort.py and a per-scan clinical table. Outputs, encodings, pair rules and validation
are those of make_adni_csv.py, whose functions are reused; only the inputs differ.

Input  : <processed_dir>/<sub_id>/<image_id>/<image_id>_{final,synthseg}_1.5mm.nii.gz
         --visits_csv   one row per scan, cohort-specific codes already mapped (see below)
Output : <data_dir>/Image/<COHORT>_<sub_id>/<visit>/{t1,seg}.nii.gz   symlinks, images untouched
         <data_dir>/derived/AD-Progression-All.csv, AD-Progression-All-abs.csv,
         AD-Progression-pairs-dt15.csv, <cohort>_volumes_raw.csv, <cohort>_build_report.txt
Then run build_multiprior.py with DERIVED_DIR=<data_dir>/derived.

--visits_csv columns (strict: unknown codes stop the build instead of being guessed):
  sub_id      subject folder name under processed_dir
  image_id    image folder name under processed_dir/<sub_id>/
  acq_date    scan date YYYY-MM-DD                                   } exactly one
  days        integer days from a per-subject reference (OASIS-3 dNNNN)  } of the two
  acq_time    optional HH:MM:SS, orders same-day scans
  sex         F or M (per subject: the mode of its rows, as make_adni_csv.py does with PTDEMOG)
  diagnosis   CN, MCI, AD, or empty (-> -1, missing)
  split       optional train/val/test; absent -> split_v3 hash (train/test only), as make_adni_csv.py
  age source, first one present is used (ADNI-like = age moves exactly with the scan dates):
  dob           date of birth, YYYY-MM-DD, YYYY-MM (-> the 15th, as ADNI's PTDOB MM/YYYY) or YYYY (-> July 2,
                the middle of the year; AIBL PTDOB is year only); acq_date axis only;
                age = (acq_date - dob) / 365.25 days, exactly make_adni_csv.py. Per subject: earliest row's value
  age_at_day0   age in years at days = 0 (OASIS-3 AgeatEntry); days axis only; age = age_at_day0 + days/365.25
  age           age in years at this scan, used as given (not ADNI-like when it is rounded)

Encoding (as make_adni_csv.py): age in years (All) and years/100 (pairs); follow_up in months since
the subject's first kept visit (days/30.44); sex F=0, M=1; diagnosis CN=0, MCI=0.5, AD=1, missing=-1.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_adni_csv as M  # noqa: E402  (also puts AD_data_processing/ on sys.path for const)
import const  # noqa: E402

SEX_CODE = {"F": 0.0, "M": 1.0}
DX_CODE = {"CN": 0.0, "MCI": 0.5, "AD": 1.0}


def load_visits(path: str, R: list) -> pd.DataFrame:
    v = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"sub_id", "image_id", "sex", "diagnosis"}
    if need - set(v.columns):
        raise SystemExit(f"--visits_csv: missing columns {sorted(need - set(v.columns))} (has {list(v.columns)})")
    has_date, has_days = "acq_date" in v.columns, "days" in v.columns
    if has_date == has_days:
        raise SystemExit("--visits_csv: give exactly one of acq_date / days")
    age_src = next((c for c in ("dob", "age_at_day0", "age") if c in v.columns), None)
    if age_src is None:
        raise SystemExit("--visits_csv: no age source column (dob, age_at_day0 or age)")
    if age_src == "dob" and not has_date:
        raise SystemExit("--visits_csv: dob needs the acq_date axis")
    if age_src == "age_at_day0" and not has_days:
        raise SystemExit("--visits_csv: age_at_day0 needs the days axis")
    v = v.apply(lambda s: s.str.strip())
    out = pd.DataFrame({"sub_id": v["sub_id"], "image_id": v["image_id"]})

    bad = ~v["sex"].isin(SEX_CODE)
    if bad.any():
        raise SystemExit(f"--visits_csv: sex must be F or M, got {sorted(set(v.loc[bad, 'sex']))[:10]}")
    out["sex"] = v["sex"].map(SEX_CODE)
    bad = ~(v["diagnosis"].isin(DX_CODE) | v["diagnosis"].eq(""))
    if bad.any():
        raise SystemExit(f"--visits_csv: diagnosis must be CN/MCI/AD or empty, got {sorted(set(v.loc[bad, 'diagnosis']))[:10]}")
    out["diagnosis"] = v["diagnosis"].map(DX_CODE).fillna(-1.0)
    out["age_src"] = age_src
    if age_src == "dob":
        full = pd.to_datetime(v["dob"], format="%Y-%m-%d", errors="coerce")
        month = pd.to_datetime(v["dob"], format="%Y-%m", errors="coerce") + pd.Timedelta(days=14)
        # year only (AIBL PTDOB '/YYYY'): the middle of the year, July 2 -- same rule as the 15th for MM/YYYY
        year = pd.to_datetime(v["dob"].where(v["dob"].str.fullmatch(r"\d{4}")) + "-07-02", format="%Y-%m-%d", errors="coerce")
        out["dob"] = full.fillna(month).fillna(year)
        bad = out["dob"].isna() & v["dob"].ne("")
        if bad.any():
            raise SystemExit(f"--visits_csv: dob must be YYYY-MM-DD, YYYY-MM or YYYY, got {sorted(set(v.loc[bad, 'dob']))[:10]}")
        n_year = int(year.notna().sum())
        if n_year:
            R.append(f"dob: {n_year} rows give the year only -> July 2 of that year (middle of the year)")
    elif age_src == "age_at_day0":
        out["age_at_day0"] = pd.to_numeric(v["age_at_day0"], errors="coerce")
    if "age" in v.columns:
        out["age_given"] = pd.to_numeric(v["age"], errors="coerce")

    t = v["acq_time"] if "acq_time" in v.columns else pd.Series("", index=v.index)
    if has_date:
        d = pd.to_datetime(v["acq_date"], format="%Y-%m-%d", errors="coerce")
        out["day"] = (d - pd.Timestamp("1970-01-01")).dt.days.astype("Int64")
        out["visit_label"] = v["acq_date"]
        out["date_str"] = v["acq_date"]
    else:
        out["day"] = pd.to_numeric(v["days"], errors="coerce").astype("Int64")
        out["visit_label"] = "d" + out["day"].astype(str)
        out["date_str"] = ""
    out["acq_time"] = t.where(t.ne(""), "00:00:00")
    if "split" in v.columns:
        bad = ~v["split"].isin(M.SPLITS)
        if bad.any():
            raise SystemExit(f"--visits_csv: split must be train/val/test, got {sorted(set(v.loc[bad, 'split']))[:10]}")
        out["split"] = v["split"]
    dup = out.duplicated(["sub_id", "image_id"])
    if dup.any():
        raise SystemExit(f"--visits_csv: duplicate (sub_id, image_id): {out.loc[dup, ['sub_id', 'image_id']].head().values.tolist()}")
    R.append(f"visits_csv {path}: {len(out)} rows / {out.sub_id.nunique()} subjects; "
             f"time axis = {'acq_date' if has_date else 'days'}; split column: {'yes' if 'split' in out else 'no'}")
    return out


def build_visits(scans: pd.DataFrame, visits: pd.DataFrame, args, R: list) -> pd.DataFrame:
    df = scans.merge(visits, on=["sub_id", "image_id"], how="left", indicator=True)
    no_row = df["_merge"].eq("left_only")
    if no_row.any():
        R.append(f"DROP {int(no_row.sum())} processed scans with no visits_csv row: "
                 f"{(df.loc[no_row, 'sub_id'] + '/' + df.loc[no_row, 'image_id']).tolist()[:6]}")
    df = df[~no_row].drop(columns="_merge")
    unused = len(visits) - len(df)
    R.append(f"visits_csv rows with no processed scan: {unused}")

    bad = df["day"].isna()
    if bad.any():
        R.append(f"DROP {int(bad.sum())} scans without a valid date/days: "
                 f"{(df.loc[bad, 'sub_id'] + '/' + df.loc[bad, 'image_id']).tolist()[:6]}")
        df = df[~bad].copy()
    df["day"] = df["day"].astype(np.int64)

    # sex and date of birth per subject, as load_ptdemog in make_adni_csv.py: sex = mode of the subject's
    # rows (first row by date on a tie), dob = earliest row's value; conflicts are reported
    df = df.sort_values(["sub_id", "day"], kind="mergesort")
    sx = df.groupby("sub_id")["sex"]
    df["sex"] = df["sub_id"].map(sx.agg(lambda s: s.mode().iloc[0] if s.mode().size == 1 else s.iloc[0]))
    msg = f"sex per subject: conflicting rows in {int((sx.nunique() > 1).sum())} subjects (mode used)"
    if "dob" in df.columns:
        db = df.dropna(subset=["dob"]).groupby("sub_id")["dob"]
        df["dob"] = df["sub_id"].map(db.first())
        msg += f"; dob: conflicting rows in {int((db.nunique() > 1).sum())} subjects (earliest row used)"
    if "age_at_day0" in df.columns:
        a0 = df.dropna(subset=["age_at_day0"]).groupby("sub_id")["age_at_day0"]
        df["age_at_day0"] = df["sub_id"].map(a0.first())
        msg += f"; age_at_day0: conflicting rows in {int((a0.nunique() > 1).sum())} subjects (earliest row used)"
    R.append(msg)

    # one scan per (subject, day), same rule as make_adni_csv.py: earliest acquisition time first,
    # then image id (ADNI: image number) -- the Delta-LFM loader keeps the first row of each follow_up
    df = df.sort_values(["sub_id", "day", "acq_time", "image_id"], kind="mergesort")
    n_before = len(df)
    if not args.keep_same_day:
        df = df.drop_duplicates(["sub_id", "day"], keep="first")
    R.append(f"same-day dedup: {n_before} scans -> {len(df)} visits "
             f"({'disabled' if args.keep_same_day else 'earliest acq_time, then image_id'})")

    if "split" in df.columns:
        multi = df.groupby("sub_id")["split"].nunique()
        if (multi > 1).any():
            raise SystemExit(f"subjects in more than one split: {multi[multi > 1].index.tolist()[:10]}")
    else:
        df["split"] = np.where(df["sub_id"].map(lambda s: M.split_v3_is_train(f"{args.cohort}_{s}")), "train", "test")
        R.append("split: no split column in visits_csv, derived from the split_v3 hash (train/test)")

    # ---- age. ADNI-like: (scan date - dob), or age at day 0 + days, so age deltas equal the scan
    # intervals; the flow model's class index is int(100 * (followup_age - starting_age))
    src = df["age_src"].iloc[0] if len(df) else "age"
    if src == "dob":
        df["age"] = (pd.Timestamp("1970-01-01") + pd.to_timedelta(df["day"], unit="D") - df["dob"]).dt.days / 365.25
        R.append("age: (acq_date - dob) / 365.25, as make_adni_csv.py")
    elif src == "age_at_day0":
        df["age"] = df["age_at_day0"] + df["day"] / 365.25
        R.append("age: age_at_day0 + days / 365.25")
    else:
        df["age"] = df["age_given"]
        R.append("age: the visits_csv age column as given (no dob / age_at_day0)")
    if src != "age" and "age_given" in df.columns:
        e = (df["age"] - df["age_given"]).abs().dropna()
        if len(e):
            R.append(f"  vs the visits_csv age column: |diff| median {e.median():.3f} y, max {e.max():.3f} y")
    g = df.groupby("sub_id")
    err = (g["age"].diff() - g["day"].diff() / 365.25).abs().dropna()
    if len(err):
        R.append(f"  age step vs scan interval (consecutive visits): |d_age - d_days/365.25| median {err.median():.3f} y, "
                 f"max {err.max():.3f} y")
    bad = df["age"].isna() | df["sex"].isna()
    if bad.any():
        R.append(f"DROP {int(bad.sum())} visits / {df.loc[bad, 'sub_id'].nunique()} subjects with no sex or age source: "
                 f"{sorted(df.loc[bad, 'sub_id'].unique())[:10]}")
        df = df[~bad].copy()

    # follow_up / capture_time after all drops, so each subject starts at 0
    first = df.groupby("sub_id")["day"].transform("min")
    df["follow_up"] = (df["day"] - first) / M.DAYS_PER_MONTH
    if df["date_str"].ne("").all():
        df["capture_time"] = df["date_str"] + "T" + df["acq_time"]
    else:
        df["capture_time"] = "d" + (df["day"] - first).map(lambda x: f"{x:05d}") + "T" + df["acq_time"]
    df["dx_gap_days"] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True, help="subject_id prefix and `source` value, e.g. OASIS or AIBL")
    ap.add_argument("--processed_dir", required=True, help="<sub_id>/<image_id>/ tree written by preprocess_cohort.py")
    ap.add_argument("--data_dir", required=True, help="Delta-LFM DATA_DIR; Image/ and derived/ are created here")
    ap.add_argument("--visits_csv", required=True, help="per-scan clinical table, columns in the module docstring")
    ap.add_argument("--exclude_csv", default="", help="CSV with subject_id,image_id to drop (visual QC); '' = none")
    ap.add_argument("--resolution_tag", default="1.5mm")
    ap.add_argument("--visit_dir", choices=["date", "image_id"], default="date",
                    help="Image/<subject>/<visit>/ name: date (YYYY-MM-DD, or dNNNN on a days axis) or image_id")
    ap.add_argument("--keep_same_day", action="store_true")
    ap.add_argument("--vol_norm_ref", choices=["train", "all"], default="train")
    ap.add_argument("--min_dt_months", type=float, default=None)
    ap.add_argument("--max_dt_months", type=float, default=None)
    ap.add_argument("--pair_dx_missing", choices=["drop", "keep"], default="keep")
    ap.add_argument("--pair_order", choices=["split", "subject"], default="split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--recompute_volumes", action="store_true")
    ap.add_argument("--n_check", type=int, default=6)
    ap.add_argument("--force_links", action="store_true")
    args = ap.parse_args()

    tag = args.cohort.lower()
    processed = Path(args.processed_dir).resolve()
    data_dir = Path(os.path.abspath(args.data_dir))
    derived = data_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    R: list[str] = [
        "NOTE Delta-LFM splits by md5 hash (step1: 'ap|', step3/eval: split_v3 'v3|') unless run with",
        "     --split_col split, which follows the `split` column written here.",
        "",
    ]

    scans, incomplete = M.discover(processed, args.resolution_tag)
    if scans.empty:
        raise SystemExit(f"no complete scans under {processed}")
    R.append(f"discovered {len(scans)} complete scans / {scans.sub_id.nunique()} subjects; "
             f"incomplete dirs skipped: {len(incomplete)} {incomplete[:5]}")
    if args.exclude_csv:
        ex = pd.read_csv(args.exclude_csv, dtype=str)
        bad = set(zip(ex["subject_id"].str.strip(), ex["image_id"].str.strip()))
        m = np.array([(s, i) in bad for s, i in zip(scans.sub_id, scans.image_id)], dtype=bool)
        scans = scans[~m]
        R.append(f"QC exclusion ({args.exclude_csv}): dropped {int(m.sum())}/{len(bad)} listed scans")

    visits = load_visits(args.visits_csv, R)
    df = build_visits(scans, visits, args, R)
    df["subject_id"] = f"{args.cohort}_" + df["sub_id"]
    df["source"] = args.cohort
    df["scan_key"] = df["sub_id"] + "_MR_" + df["visit_label"]
    df["sequence"] = ""
    df["visit"] = df["visit_label"] if args.visit_dir == "date" else df["image_id"]
    dup = df.duplicated(["subject_id", "visit"])
    if dup.any():
        raise SystemExit(f"--visit_dir {args.visit_dir}: {int(dup.sum())} duplicate (subject, visit) names, e.g. "
                         f"{df.loc[dup, ['subject_id', 'visit']].head(5).values.tolist()}")

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

    df = M.add_volumes(df, derived / f"{tag}_volumes_raw.csv", args.workers, args.recompute_volumes, R)
    df = M.normalize_volumes(df, args.vol_norm_ref, R)

    cols = ["subject_id", "sub_id", "split", "scan_key", "visit", "image_id", "image_path", "seg_path", "follow_up",
            "capture_time", "age", "sex", "diagnosis", "dx_gap_days", "source", "sequence", "head_size"] \
        + const.COARSE_REGIONS
    allv = df.sort_values(["subject_id", "follow_up", "capture_time"])[cols].reset_index(drop=True)
    allv_abs = allv.copy()
    for c in ("image_path", "seg_path"):
        allv_abs[c] = str(data_dir) + "/" + allv_abs[c]
    allv.to_csv(derived / "AD-Progression-All.csv", index=False)
    allv_abs.to_csv(derived / "AD-Progression-All-abs.csv", index=False)

    pairs = M.make_pairs(allv_abs, args, R)
    pairs.to_csv(derived / "AD-Progression-pairs-dt15.csv", index=False)

    M.validate(allv, allv_abs, pairs, None, args.n_check, R)
    R.append(f"\nwritten to {derived}: AD-Progression-All.csv, AD-Progression-All-abs.csv, AD-Progression-pairs-dt15.csv")
    R.append(f"next: DERIVED_DIR={derived} python build_multiprior.py")
    text = "\n".join(R)
    print(text)
    (derived / f"{tag}_build_report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
