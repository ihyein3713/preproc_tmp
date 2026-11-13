#!/usr/bin/env python
"""Manifest-driven T1 preprocessing for cohorts other than ADNI (OASIS, AIBL).

Same pipeline, settings and output layout as preprocess_adni.py, whose machinery is reused
(CPU stage, SynthSeg batching, load-adaptive concurrency, resume and stale detection):
    N4 -> SynthStrip -> antsRegistration (affine to MNI152 1 mm) -> WhiteStripe -> resample,
    SynthSeg on the registered volume, label map resampled.
Outputs: <out_dir>/<subject>/<image_id>/<image_id>_{final,synthseg}_{1mm,1.5mm}.nii.gz

What differs from preprocess_adni.py is only how the scans are found. preprocess_adni.py walks an
ADNI DICOM tree (<subject>/<sequence>/<date_time>/<image_id>/*.dcm) and takes subject, image id
and date from the folder names. Here every scan is one row of a manifest CSV:
    --col-subject  subject id          -> <out_dir>/<subject>/
    --col-image    image id            -> <out_dir>/<subject>/<image_id>/ (unique within a subject)
    --col-path     input: a .nii/.nii.gz file, or a folder holding the .dcm files of one series
    --col-date     optional; copied into the result log (acq_date) for make_cohort_csv.py
--path-prefix is put in front of every path: the rsync --files-from copy keeps the source absolute
path under <dest>/files/, so the manifest paths stay valid with --path-prefix <dest>/files.

NIfTI inputs skip dcm2niix: a .nii.gz is linked as <image_id>_step0_raw.nii.gz (the link is removed
afterwards, the raw file is never modified), a .nii is gzip-copied there.

Registration (--affine):
    register  every scan gets a fresh antsRegistration (seed --reg-seed), as preprocess_adni.py.
    manifest  follow the manifest's --col-action: 'apply_saved_affine' applies the saved ANTs affine
              in --col-mat (path under --path-prefix like the raw) with antsApplyTransforms, same
              reference grid and BSpline interpolation as the registration -- on an ADNI scan this
              reproduces the antsRegistration output bit for bit; 'reregister' / 'reregister_new_raw'
              register afresh. Any other action value stops the run. The applied .mat is copied next
              to the output under the name antsRegistration would have written.

Stopping early (handing the rest to another machine): --stop-new-at YYYY-MM-DDTHH:MM, or creating the
file <out_dir>/logs/STOP, makes the run start no new scan; scans already in the CPU stage finish, the
SynthSeg queue is flushed, then the run exits. At the end, the manifest rows of every scan that is not
complete (deferred, failed, still being copied) are written to <out_dir>/logs/remaining_<cohort>_<time>.csv,
a manifest the other machine can run as is.

Two machines on one manifest (--claim-dir, --claim-as, --reverse): each machine writes to its own --out-dir
and, right before starting a scan, creates <claim-dir>/<subject>__<image> holding its --claim-as label
(exclusive create). A scan claimed by the other machine is skipped, so one machine can work from the front
and the other from the back (--reverse) until they meet. Scans already complete in --out-dir are claimed at
start-up. A claim with the own label is honoured on a restart (resume).

Run --list-only first. Without the --col-* options it prints the manifest columns.

Usage (see run_preprocess_cohort.sh, which sets the paths and the CPU settings):
    source custom/env_server.sh
    python custom/preprocess_cohort.py --cohort OASIS --manifest <manifest.csv> \
        --col-subject <col> --col-image <col> --col-path <col> \
        --path-prefix /mnt/aix22308/team_data/OASIS_AIBL_rerun/files \
        --out-dir /mnt/aix22308/team_data/OASIS_processed --gpus "" --list-only
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import shutil
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline  # noqa: E402
import preprocess_adni as P  # noqa: E402

logger = logging.getLogger("preprocess_cohort")


# --------------------------------------------------------------------------- manifest

def read_table(path: Path, sep: str) -> tuple[list, list]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def read_exclusions(path: Path, sep: str, col_subject: str, col_image: str) -> tuple[set, set]:
    """-> (set of (subject, image), set of image). A .csv/.tsv with both the subject and the image
    column excludes by pair; with the image column only, by image id; any other file is read as one
    image id per line."""
    if path.suffix.lower() in (".csv", ".tsv"):
        cols, rows = read_table(path, "\t" if path.suffix.lower() == ".tsv" else sep)
        if col_image not in cols:
            raise SystemExit(f"--exclude {path}: no column {col_image!r} (columns: {cols})")
        if col_subject in cols:
            return {(r[col_subject].strip(), r[col_image].strip()) for r in rows}, set()
        return set(), {r[col_image].strip() for r in rows}
    return set(), {line.strip() for line in path.read_text().splitlines() if line.strip()}


def input_kind(src: Path) -> tuple[str, list]:
    """('nifti', []) | ('dicom', dcm file names) | ('missing', [])"""
    if src.is_file() and src.name.endswith((".nii.gz", ".nii")):
        return "nifti", []
    if src.is_dir():
        files = [f for f in os.listdir(src) if f.lower().endswith(".dcm")]
        if files:
            return "dicom", files
    return "missing", []


APPLY_ACTIONS = {"apply_saved_affine"}
REGISTER_ACTIONS = {"reregister", "reregister_new_raw"}


def fingerprint(kind: str, src: Path, files: list, mat: Path | None = None) -> str:
    """Input identity for stale detection; a saved affine is part of it, so switching between
    registering and applying a saved affine (or a changed .mat) redoes the scan."""
    if kind == "dicom":
        fp = P.source_fingerprint(files)
    else:
        st = src.stat()
        fp = f"nii:{st.st_size}:{int(st.st_mtime)}"
    if mat is not None:
        st = mat.stat()
        fp += f"|affine:{mat.name}:{st.st_size}:{int(st.st_mtime)}"
    return fp


def rsync_temp_present(folder: Path, name: str | None) -> bool:
    """rsync writes into a hidden '.<name>.XXXXXX' file next to the target and renames it at the end."""
    try:
        entries = os.listdir(folder)
    except OSError:
        return False
    if name is None:
        return any(e.startswith(".") and e.count(".") >= 2 for e in entries)
    return any(e.startswith(f".{name}.") for e in entries)


def is_settled(kind: str, src: Path, files: list, min_age_s: float) -> bool:
    if kind == "dicom":
        return P.is_settled(src, files, min_age_s) and not rsync_temp_present(src, None)
    if rsync_temp_present(src.parent, src.name):
        return False
    return time.time() - src.stat().st_ctime >= min_age_s


def output_paths(out_dir: Path, subject: str, image_id: str, resolutions: list) -> dict:
    # preprocess_adni.output_paths takes the subject from the first folder below raw_dir and the image
    # id from the last one; a placeholder <root>/<subject>/<image_id> gives the same layout.
    root = Path("/_manifest_")
    return P.output_paths(root / subject / image_id, root, out_dir, resolutions)


def stage_nifti(src: Path, paths: dict) -> None:
    """Put the raw NIfTI where pipeline.dicom_to_nifti looks for an existing conversion."""
    d = paths["raw_nifti_dir"]
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{paths['raw_nifti_stem']}.nii.gz"
    if dst.is_symlink() and not dst.exists():
        dst.unlink()                                     # dangling link from an earlier run
    if dst.exists():
        return
    if src.name.endswith(".nii.gz"):
        dst.symlink_to(src.resolve())
        return
    tmp = d / f".{dst.name}.partial"
    with open(src, "rb") as fi, gzip.open(tmp, "wb", compresslevel=1) as fo:
        shutil.copyfileobj(fi, fo)
    tmp.rename(dst)


def apply_saved_affine(src: Path, dst: Path, reference: Path, mat: Path) -> None:
    """antsRegistration's warped output, from a saved affine: same reference grid, same BSpline."""
    if dst.exists():
        return
    tmp = pipeline._prepare(dst)
    pipeline._run("apply_affine", [
        "antsApplyTransforms", "-d", "3", "-i", str(src), "-r", str(reference),
        "-o", str(tmp), "-n", "BSpline", "-t", str(mat),
    ])
    shutil.copyfile(mat, f"{dst}_transform_0GenericAffine.mat")
    tmp.rename(dst)


def cpu_stage(run: P.Run, job: dict) -> None:
    """preprocess_adni.cpu_stage, with the registration step replaced by the saved affine when the
    job carries one."""
    paths, args = job["paths"], run.args
    job["raw_nifti"] = pipeline.dicom_to_nifti(job["dicom_dir"], paths["raw_nifti_dir"], paths["raw_nifti_stem"])
    pipeline.n4_bias_correction(job["raw_nifti"], paths["biasfield"])
    if not paths["stripped"].exists():
        with P.strip_cpu_slots:
            pipeline.skull_strip(paths["biasfield"], paths["stripped"], gpu=False, threads=args.strip_threads)
    if job["mat"] is not None:
        apply_saved_affine(paths["stripped"], paths["registered"], run.reference_brain, job["mat"])
    else:
        pipeline.register_to_reference(paths["stripped"], paths["registered"], run.reference_brain, seed=args.reg_seed)
    pipeline.normalize_intensity(paths["registered"], paths["final"])
    for res in run.resolutions:
        pipeline.resample(paths["final"], paths["final_resampled"][res], res, is_segmentation=False)


# --------------------------------------------------------------------------- run

class CohortRun(P.Run):
    """P.Run with the cohort, subject, image id and date in every log entry: preprocess_adni.py takes
    them from the DICOM path later, which a manifest-driven input does not encode."""

    def record(self, job: dict, error: Exception | None) -> None:
        result = {"cohort": self.args.cohort, "subject_id": job["subject_id"], "image_id": job["image_id"],
                  "acq_date": job["acq_date"], "input": str(job["dicom_dir"]), "input_kind": job["kind"],
                  "registration": "saved_affine" if job["mat"] is not None else "antsRegistration",
                  "affine": str(job["mat"]) if job["mat"] is not None else ""}
        if error is None:
            result.update(status="ok", seconds=round(time.time() - job["start"], 1), source=job["fingerprint"])
        else:
            step = error.step if isinstance(error, pipeline.StepError) else type(error).__name__
            result.update(status="failed", step=step, error=str(error), source=job["fingerprint"])
        with self.lock:
            self.log_f.write(json.dumps(result) + "\n")
            self.log_f.flush()
            self.n_done += 1
            self.n_ok += error is None
            if self.n_done % 20 == 0 or self.n_done == self.total:
                logger.info("progress %d/%d (%d failed, %.0fs elapsed)", self.n_done, self.total,
                            self.n_done - self.n_ok, time.time() - self.start)


class StopRequest:
    """True once --stop-new-at has passed or <out_dir>/logs/STOP exists; logged once."""

    def __init__(self, at: datetime | None, stop_file: Path):
        self.at, self.stop_file = at, stop_file
        self.lock = threading.Lock()
        self.n_deferred = 0
        self.seen = False

    def __call__(self) -> bool:
        hit = (self.at is not None and datetime.now() >= self.at) or self.stop_file.exists()
        if hit:
            with self.lock:
                if not self.seen:
                    self.seen = True
                    logger.info("stop requested (%s): no new scans; finishing the ones in progress",
                                f"--stop-new-at {self.at:%Y-%m-%d %H:%M}" if self.at is not None and datetime.now() >= self.at
                                else f"{self.stop_file} exists")
        return hit

    def defer(self) -> None:
        with self.lock:
            self.n_deferred += 1


class Claims:
    """Per-scan claim files shared by two machines (see the module doc). Atomicity only avoids duplicate work:
    the machines write to different --out-dir, so a double claim never corrupts an output."""

    def __init__(self, claim_dir: Path, me: str):
        self.dir, self.me = claim_dir, me
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.n_other = 0

    def path(self, job: dict) -> Path:
        return self.dir / f"{job['subject_id']}__{job['image_id']}"

    def owner(self, job: dict) -> str | None:
        try:
            text = self.path(job).read_text().split()
        except FileNotFoundError:
            return None
        except OSError:
            return ""
        return text[0] if text else ""          # "" = being written by its creator right now

    def take(self, job: dict) -> bool:
        p = self.path(job)
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        except FileExistsError:
            if self.owner(job) == self.me:
                return True
            with self.lock:
                self.n_other += 1
            return False
        with os.fdopen(fd, "w") as f:
            f.write(f"{self.me} {socket.gethostname()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        return True


def process(run: CohortRun, batcher: P.SegBatcher, gate: P.CpuGate, job: dict, stop: StopRequest,
            claims: Claims | None = None) -> None:
    if stop():
        stop.defer()
        return
    job["start"] = time.time()
    try:
        with gate:
            if stop():                       # waited at the gate past the stop time: nothing started yet
                stop.defer()
                return
            if claims is not None and not claims.take(job):
                return                       # the other machine has it
            if job["kind"] == "nifti":
                stage_nifti(job["dicom_dir"], job["paths"])
            cpu_stage(run, job)
    except Exception as e:
        if isinstance(e, pipeline.StepError):
            logger.error("failed %s/%s: %s", job["subject_id"], job["image_id"], e)
        else:
            logger.exception("unexpected failure on %s/%s", job["subject_id"], job["image_id"])
        run.record(job, e)
        return
    if job["paths"]["synthseg"].exists():
        P.finish(run, job)
        run.record(job, None)
    else:
        batcher.submit(job)


def main():
    ap = argparse.ArgumentParser(description="Preprocess OASIS/AIBL T1 scans listed in a manifest (no CSV output).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--cohort", required=True, help="Cohort tag written into the log, e.g. OASIS or AIBL.")
    ap.add_argument("--manifest", required=True, help="CSV with one row per T1 scan.")
    ap.add_argument("--sep", default=",", help="Manifest delimiter (default ',').")
    ap.add_argument("--col-subject", default="", help="Manifest column with the subject id.")
    ap.add_argument("--col-image", default="", help="Manifest column with the image id (unique within a subject).")
    ap.add_argument("--col-path", default="", help="Manifest column with the input path (NIfTI file or DICOM folder).")
    ap.add_argument("--col-date", default="", help="Optional manifest column with the scan date (copied into the log).")
    ap.add_argument("--affine", choices=["register", "manifest"], required=True,
                    help="register: fresh antsRegistration for every scan (as ADNI); manifest: follow --col-action "
                         "(apply the saved affine in --col-mat, or register afresh). Required on purpose.")
    ap.add_argument("--col-action", default="action", help="Manifest column with the action (--affine manifest).")
    ap.add_argument("--col-mat", default="mat_path", help="Manifest column with the saved .mat (--affine manifest).")
    ap.add_argument("--filter", action="append", default=[], metavar="COL=VALUE",
                    help="Keep only rows with this value (repeatable, all must match), e.g. for a manifest "
                         "that lists both cohorts.")
    ap.add_argument("--path-prefix", default="", help="Prepended to every manifest path not matched by --path-map.")
    ap.add_argument("--path-map", action="append", default=[], metavar="/OLD=/NEW",
                    help="Replace a leading path prefix (repeatable), e.g. for raw folders moved after the copy: "
                         "/data1/miela/workspace/longi=/mnt/aix22308/team_data")
    ap.add_argument("--exclude", default="",
                    help="Scans to skip: CSV/TSV with the image column (and optionally the subject column) "
                         "named as in the manifest, or a text file with one image id per line.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reference-brain",
                    default=os.environ.get("REFERENCE_BRAIN", str(P.TOOLS_DIR / "MNI152_T1_1mm_brain.nii.gz")))
    ap.add_argument("--fs-license", default=os.environ.get("FS_LICENSE", str(P.TOOLS_DIR / "license.txt")))
    ap.add_argument("--resolutions", default="1.0,1.5")
    ap.add_argument("--gpus", required=True,
                    help="Physical GPU ids for SynthSeg as in preprocess_adni.py ('id', 'id@FROM..UNTIL'); "
                         "required on purpose, so a run never takes a GPU that was not assigned. '' = CPU.")
    ap.add_argument("--slots-per-gpu", type=int, default=1)
    ap.add_argument("--seg-batch", type=int, default=48)
    ap.add_argument("--seg-wait", type=float, default=900)
    ap.add_argument("--seg-threads", type=int, default=8)
    ap.add_argument("--cpu-workers", type=int, default=4)
    ap.add_argument("--cpu-min", type=int, default=2)
    ap.add_argument("--cpu-start", type=int, default=4)
    ap.add_argument("--load-high", type=float, default=0.9)
    ap.add_argument("--load-low", type=float, default=0.75)
    ap.add_argument("--cpu-adjust-s", type=float, default=120)
    ap.add_argument("--itk-threads", type=int, default=4)
    ap.add_argument("--strip-threads", type=int, default=None)
    ap.add_argument("--strip-workers", type=int, default=1)
    ap.add_argument("--reg-seed", type=int, default=1)
    ap.add_argument("--tfcuda-site-packages",
                    default=os.environ.get("TFCUDA_SITE_PACKAGES", "/mnt/aix22308/envs/tfcuda/lib/python3.8/site-packages"))
    ap.add_argument("--keep", default="", help=f"Intermediates to keep, out of {','.join(P.INTERMEDIATES)}.")
    ap.add_argument("--keep-native-res", action="store_true")
    ap.add_argument("--min-age-min", type=float, default=60,
                    help="Skip inputs changed less than this many minutes ago or with an rsync temp file next to "
                         "them (copy still running); a later run picks them up. 0 = no wait.")
    ap.add_argument("--list-only", action="store_true", help="Check the manifest and the inputs, no processing.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-file", default=None, help="default: <out_dir>/preprocess_log.jsonl")
    ap.add_argument("--reverse", action="store_true", help="Process the manifest from the last row backwards.")
    ap.add_argument("--claim-dir", default="", help="Shared folder of per-scan claims for two machines on one manifest.")
    ap.add_argument("--claim-as", default="", help="This machine's label in --claim-dir, e.g. server or local.")
    ap.add_argument("--stop-new-at", default="", metavar="YYYY-MM-DDTHH:MM",
                    help="Start no new scan after this local time (also: create <out_dir>/logs/STOP); scans in "
                         "progress finish, then the unfinished manifest rows go to <out_dir>/logs/remaining_*.csv.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    keep = {k.strip() for k in args.keep.split(",") if k.strip()}
    if keep - set(P.INTERMEDIATES):
        ap.error(f"--keep: unknown {sorted(keep - set(P.INTERMEDIATES))}, choose from {P.INTERMEDIATES}")
    if args.reg_seed == 0:
        ap.error("--reg-seed must be nonzero (antsRegistration treats 0 as unseeded)")
    if bool(args.claim_dir) != bool(args.claim_as) or (args.claim_as and not args.claim_as.isidentifier()):
        ap.error("--claim-dir and --claim-as go together; --claim-as is one word, e.g. server or local")
    try:
        stop_at = datetime.fromisoformat(args.stop_new_at) if args.stop_new_at else None
    except ValueError:
        ap.error(f"--stop-new-at {args.stop_new_at!r}: expected YYYY-MM-DDTHH:MM")

    # ---- manifest
    manifest = Path(args.manifest)
    cols, rows = read_table(manifest, args.sep)
    need = {"--col-subject": args.col_subject, "--col-image": args.col_image, "--col-path": args.col_path}
    if args.col_date:
        need["--col-date"] = args.col_date
    if args.affine == "manifest":
        need["--col-action"], need["--col-mat"] = args.col_action, args.col_mat
    unset = [k for k, v in need.items() if not v]
    unknown = [f"{k} {v!r}" for k, v in need.items() if v and v not in cols]
    if unset or unknown:
        print(f"manifest {manifest}: {len(rows)} rows, columns:\n  " + "\n  ".join(cols))
        for r in rows[:3]:
            print("  e.g.", {c: r[c] for c in cols})
        ap.error("set " + ", ".join(unset + [f"(not a column) {u}" for u in unknown]))
    for f in args.filter:
        col, sep_, val = f.partition("=")
        if not sep_ or col not in cols:
            ap.error(f"--filter {f!r}: expected COL=VALUE with COL in {cols}")
        rows = [r for r in rows if r[col].strip() == val]
    logger.info("manifest %s: %d rows after filters %s", manifest, len(rows), args.filter)

    if args.exclude:
        ex_pairs, ex_images = read_exclusions(Path(args.exclude), args.sep, args.col_subject, args.col_image)
        n0 = len(rows)
        rows = [r for r in rows if (r[args.col_subject].strip(), r[args.col_image].strip()) not in ex_pairs
                and r[args.col_image].strip() not in ex_images]
        logger.info("exclusions %s: %d listed, %d manifest rows dropped", args.exclude,
                    len(ex_pairs) + len(ex_images), n0 - len(rows))

    path_maps = []
    for pm in args.path_map:
        old, sep_, new = pm.partition("=")
        if not sep_ or not old.startswith("/") or not new.startswith("/"):
            ap.error(f"--path-map {pm!r}: expected /OLD/PREFIX=/NEW/PREFIX")
        path_maps.append((old.rstrip("/") + "/", new.rstrip("/") + "/"))

    def under_prefix(p: str) -> Path:
        for old, new in path_maps:                     # first matching --path-map wins over --path-prefix
            if p.startswith(old):
                return Path(new + p[len(old):])
        return Path(args.path_prefix.rstrip("/") + "/" + p.lstrip("/")) if args.path_prefix else Path(p)

    out_dir = Path(args.out_dir)
    resolutions = [float(r.strip()) for r in args.resolutions.split(",") if r.strip()]
    jobs, seen, missing, bad_ids, bad_actions = [], {}, [], [], set()
    for r in rows:
        sub, img, raw = r[args.col_subject].strip(), r[args.col_image].strip(), r[args.col_path].strip()
        if not sub or not img or "/" in sub or "/" in img or sub.startswith(".") or img.startswith("."):
            bad_ids.append((sub, img))
            continue
        mat = None
        if args.affine == "manifest":
            action = r[args.col_action].strip()
            if action in APPLY_ACTIONS:
                mat = under_prefix(r[args.col_mat].strip()) if r[args.col_mat].strip() else None
                if mat is None or not mat.is_file():
                    missing.append(f"{mat} (saved affine of {sub}/{img})")
                    continue
            elif action not in REGISTER_ACTIONS:
                bad_actions.add(action)
                continue
        src = under_prefix(raw)
        kind, files = input_kind(src)
        if kind == "missing":
            missing.append(str(src))
            continue
        if (sub, img) in seen:
            raise SystemExit(f"duplicate (subject, image) {sub}/{img}: {seen[(sub, img)]} and {src}")
        seen[(sub, img)] = src
        jobs.append({"row": r, "subject_id": sub, "image_id": img, "dicom_dir": src, "kind": kind, "files": files, "mat": mat,
                     "acq_date": r[args.col_date].strip() if args.col_date else "",
                     "paths": output_paths(out_dir, sub, img, resolutions)})
    if bad_ids:
        raise SystemExit(f"{len(bad_ids)} rows with an empty or unusable subject/image id (no '/', no leading '.'): "
                         f"{bad_ids[:5]}")
    if bad_actions:
        raise SystemExit(f"unknown {args.col_action} values {sorted(bad_actions)}; known: "
                         f"{sorted(APPLY_ACTIONS)} (saved affine), {sorted(REGISTER_ACTIONS)} (register)")
    kinds = {k: sum(j["kind"] == k for j in jobs) for k in ("nifti", "dicom")}
    n_apply = sum(j["mat"] is not None for j in jobs)
    logger.info("%d scans / %d subjects with an input on disk (%s); %d inputs not found",
                len(jobs), len({j['subject_id'] for j in jobs}), kinds, len(missing))
    logger.info("registration: %d saved affine, %d antsRegistration (--affine %s)",
                n_apply, len(jobs) - n_apply, args.affine)
    if missing:
        logger.warning("  inputs not found, e.g. %s", missing[:5])

    if args.list_only:
        for j in jobs[:20]:
            print(j["subject_id"], j["image_id"], j["acq_date"], j["kind"],
                  "saved_affine" if j["mat"] is not None else "register", j["dicom_dir"])
        if len(jobs) > 20:
            print(f"... and {len(jobs) - 20} more")
        if missing:
            print(f"{len(missing)} inputs not found:")
            print("\n".join(missing[:50]))
        return

    # ---- preflight (as preprocess_adni.py)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    if missing:
        miss_path = out_dir / "logs" / f"missing_inputs_{time.strftime('%Y%m%d_%H%M')}.txt"
        miss_path.write_text("\n".join(missing) + "\n")
        logger.warning("  full list: %s", miss_path)
    reference_brain = Path(args.reference_brain)
    if not reference_brain.exists():
        ap.error(f"reference brain not found: {reference_brain}")
    if args.fs_license:
        if not Path(args.fs_license).exists():
            ap.error(f"FreeSurfer license not found: {args.fs_license}")
        os.environ["FS_LICENSE"] = args.fs_license
    pipeline.require_tools([t for t in pipeline.REQUIRED_TOOLS if t != "dcm2niix" or kinds["dicom"]]
                           + (["antsApplyTransforms"] if n_apply else []))
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.itk_threads)
    P.strip_cpu_slots = threading.Semaphore(args.strip_workers)

    seg_env = None
    gpu_specs = P.parse_gpus(args.gpus)
    if gpu_specs:
        # '' = the CUDA libraries FreeSurfer's TensorFlow finds by itself (e.g. a local install);
        # the preflight below still fails if TensorFlow sees no GPU
        seg_env = (pipeline.synthseg_gpu_env(Path(args.tfcuda_site_packages)) if args.tfcuda_site_packages
                   else dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="1"))
        free_mb = pipeline.gpu_free_mb()
        for gid, start, dl in gpu_specs:
            name = pipeline.check_synthseg_gpu(pipeline.on_gpu(seg_env, gid))
            logger.info("SynthSeg GPU preflight OK: GPU %d (%s, %s MiB free)%s%s", gid, name, free_mb.get(gid, "?"),
                        f", from {start}" if start else "", f", until {dl}" if dl else "")
            if free_mb.get(gid, 1 << 30) < 12000 * args.slots_per_gpu:
                logger.warning("GPU %d has only %d MiB free; each SynthSeg slot needs ~11 GB", gid, free_mb[gid])
    else:
        logger.info("--gpus '': SynthSeg runs on the CPU")

    # ---- resume: complete and unchanged -> skip; complete but input changed -> redo; still copying -> later
    todo, n_done, pending, stale = [], 0, [], []
    for j in jobs:
        j["fingerprint"] = fingerprint(j["kind"], j["dicom_dir"], j["files"], j["mat"])
        if P.is_complete(j["paths"]):
            old = P.recorded_fingerprint(j["paths"])
            if old is None or old == j["fingerprint"]:
                n_done += 1
                continue
            stale.append((j, old))
        if not is_settled(j["kind"], j["dicom_dir"], j["files"], args.min_age_min * 60):
            pending.append(j)
            continue
        todo.append(j)
    pend = {id(j) for j in pending}
    for j, old in stale:
        if id(j) in pend:
            continue
        logger.warning("input changed since it was processed (%s -> %s), redoing: %s", old, j["fingerprint"], j["dicom_dir"])
        P.discard_outputs(j["paths"])
    logger.info("%d already complete, %d to process, %d skipped as still being copied (changed < %g min ago or "
                "rsync temp file; re-run later), %d redone because their input changed",
                n_done, len(todo), len(pending), args.min_age_min, sum(id(j) not in pend for j, _ in stale))
    if pending:
        logger.info("  e.g. %s", [str(j["dicom_dir"]) for j in pending[:3]])
    if args.reverse:
        todo.reverse()
    claims = None
    if args.claim_dir:
        claims = Claims(Path(args.claim_dir), args.claim_as)
        mine = [j for j in jobs if P.is_complete(j["paths"])]
        for j in mine:
            claims.take(j)
        n_other0 = claims.n_other
        todo = [j for j in todo if claims.owner(j) in (None, args.claim_as)]
        logger.info("claims %s as %r: %d complete here claimed (%d of them also claimed by the other machine), "
                    "%d to process not claimed by the other machine%s", claims.dir, args.claim_as, len(mine),
                    n_other0, len(todo), ", in reverse order" if args.reverse else "")
        claims.n_other = 0
    elif args.reverse:
        logger.info("processing in reverse manifest order")
    if args.limit:
        todo = todo[: args.limit]

    stop = StopRequest(stop_at, out_dir / "logs" / "STOP")
    if stop_at is not None:
        logger.info("no new scans after %s (or once %s exists)", stop_at, stop.stop_file)
    log_path = Path(args.log_file) if args.log_file else out_dir / "preprocess_log.jsonl"
    with log_path.open("a") as log_f:
        run = CohortRun(args, None, out_dir, reference_brain, resolutions, keep, len(todo), log_f)
        batcher = P.SegBatcher(run, gpu_specs, args.slots_per_gpu, seg_env)
        gate = P.CpuGate(args, out_dir / "logs" / "cpu_workers")
        with ThreadPoolExecutor(max_workers=args.cpu_workers) as executor:
            for future in [executor.submit(process, run, batcher, gate, j, stop, claims) for j in todo]:
                future.result()
        logger.info("CPU stage finished, waiting for the remaining SynthSeg batches")
        batcher.close_and_join()
    logger.info("done: %d ok, %d failed, %d not started (stop requested)%s. log: %s",
                run.n_ok, run.n_done - run.n_ok, stop.n_deferred,
                f", {claims.n_other} skipped (claimed by the other machine)" if claims else "", log_path)

    # manifest rows of everything not complete now: input for a run elsewhere (same columns as --manifest)
    remaining = [j for j in jobs if not P.is_complete(j["paths"])
                 and (claims is None or claims.owner(j) in (None, args.claim_as))]
    if remaining:
        rem_path = out_dir / "logs" / f"remaining_{args.cohort}_{time.strftime('%Y%m%d_%H%M')}.csv"
        with rem_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter=args.sep)
            w.writeheader()
            w.writerows(j["row"] for j in remaining)
        logger.info("%d of %d scans not complete; their manifest rows: %s", len(remaining), len(jobs), rem_path)
    else:
        logger.info("all %d scans complete", len(jobs))


if __name__ == "__main__":
    main()
