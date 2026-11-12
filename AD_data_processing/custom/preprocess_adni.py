#!/usr/bin/env python
"""ADNI T1 series preprocessing (no cohort/pair CSVs -- that's s1/s2/s3).

Raw tree is DICOM: <raw_dir>/<subject_id>/<sequence_folder>/<date_time>/<image_id>/*.dcm
Run with --list-only first to check which series get picked up.

Usage (on the server, see run_preprocess_adni.sh, which sets the paths below):
    source custom/env_server.sh
    python custom/preprocess_adni.py \
        --raw-dir /mnt/aix22308/team_data/ADNI_t1w_raw \
        --out-dir /mnt/aix22308/team_data/ADNI_processed \
        --sequences all --list-only

    python custom/preprocess_adni.py \
        --raw-dir /mnt/aix22308/team_data/ADNI_t1w_raw \
        --out-dir /mnt/aix22308/team_data/ADNI_processed \
        --sequences all --gpus 4,5@2026-10-27T22:30 \
        --limit 5   # sanity-check a handful before the full batch

Two stages. The CPU workers run dcm2niix -> N4 -> SynthStrip -> registration -> WhiteStripe ->
resampling of the image. Registered volumes then queue for SynthSeg, which runs in batches: one
mri_synthseg process per GPU slot segments up to --seg-batch volumes, because its start-up
(TensorFlow import and model load from the NFS FreeSurfer install) takes minutes while each
prediction takes ~20 s. The label map is then resampled and the intermediates are removed.

--gpus takes physical GPU ids (nvidia-smi index); `id@YYYY-MM-DDTHH:MM` stops starting new
SynthSeg batches on that GPU when they would not finish before the given local time, so a GPU
that has to be handed back is freed without stopping the run.

Env var fallbacks: RAW_DATA_DIR, OUT_DIR, REFERENCE_BRAIN, FS_LICENSE, TFCUDA_SITE_PACKAGES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pipeline

logger = logging.getLogger("preprocess_adni")

# Substring match, not an exact-name allowlist: Siemens/Philips use MPRAGE/
# MP-RAGE, GE uses (IR-)SPGR/(IR-)FSPGR. Excludes FLAIR/DTI/rsfMRI/ASL/
# field-map/localizer/HighResHippocampus, which this T1-tuned pipeline isn't for.
DEFAULT_T1_PATTERNS = ["MPRAGE", "MP-RAGE", "SPGR"]

INTERMEDIATES = ["raw", "biasfield", "stripped", "registered", "transforms"]

TOOLS_DIR = Path(os.environ.get("TOOLS_DIR", "/mnt/aix22308/tools"))

# Conservative SynthSeg batch duration estimate, used only to decide whether a batch still fits
# before a GPU's deadline (measured on this server: ~6-10 min start-up, ~20 s per volume).
SEG_STARTUP_EST = timedelta(minutes=15)
SEG_PER_VOLUME_EST = timedelta(seconds=30)

# CPU SynthStrip peaks at ~8-10 GB RSS per process on 16-25 MB inputs; --strip-workers
# bounds how many run at once (main() resizes this).
strip_cpu_slots = threading.Semaphore(1)


def parse_gpus(text: str) -> list:
    """'5@..2026-10-27T22:30,4@2026-10-27T22:00..' -> [(5, None, until), (4, from, None)].
    'id@T' alone means until T; 'id' alone means the whole run."""
    def when(t):
        return datetime.fromisoformat(t) if t else None
    specs = []
    for item in (t.strip() for t in text.split(",") if t.strip()):
        gid, _, window = item.partition("@")
        start, sep, until = window.partition("..")
        specs.append((int(gid), when(start), when(until)) if sep else (int(gid), None, when(start)))
    return specs


class CpuGate:
    """Bounds how many series are in the CPU stage at once. The bound follows the server's 1-min load
    average (one step every --cpu-adjust-s: down above --load-high x cores, up below --load-low x cores,
    within [--cpu-min, --cpu-workers]), unless <out>/logs/cpu_workers holds a number, which then fixes
    it; the file is re-read at every step, so it can be edited while the run goes on."""

    def __init__(self, args, control_file: Path):
        self.args, self.control_file = args, control_file
        self.cap = max(args.cpu_min, min(args.cpu_start, args.cpu_workers))
        self.active = 0
        self.cond = threading.Condition()
        threading.Thread(target=self._adjust, daemon=True).start()

    def __enter__(self):
        with self.cond:
            while self.active >= self.cap:
                self.cond.wait(timeout=30)
            self.active += 1

    def __exit__(self, *exc):
        with self.cond:
            self.active -= 1
            self.cond.notify_all()

    def _adjust(self) -> None:
        a, ncpu = self.args, os.cpu_count() or 1
        while True:
            time.sleep(a.cpu_adjust_s)
            load = os.getloadavg()[0]
            try:
                fixed = int(self.control_file.read_text().split()[0])
            except (OSError, ValueError, IndexError):
                fixed = None
            if fixed is not None:
                new, why = max(1, fixed), f"fixed by {self.control_file}"
            elif load > a.load_high * ncpu:
                new, why = max(a.cpu_min, self.cap - 1), f"load {load:.0f} > {a.load_high * ncpu:.0f}"
            elif load < a.load_low * ncpu:
                new, why = min(a.cpu_workers, self.cap + 1), f"load {load:.0f} < {a.load_low * ncpu:.0f}"
            else:
                continue
            if new != self.cap:
                logger.info("CPU stage concurrency %d -> %d (%s)", self.cap, new, why)
                with self.cond:
                    self.cap = new
                    self.cond.notify_all()


def discover_series(raw_dir: Path, patterns: list) -> list:
    """Return (subject_id, dicom_series_dir, dicom_file_names) triples: one per leaf directory
    containing .dcm files, under any <subject>/<sequence>/ folder whose name
    contains one of `patterns` (case-sensitive substring match; empty = every
    folder, for a raw tree that is already T1-only). Visit date
    and image ID depth aren't assumed fixed -- we just walk from the
    sequence folder down to wherever the .dcm files are.
    """
    series = []
    for subject_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        subject_id = subject_dir.name
        for seq_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            if patterns and not any(pattern in seq_dir.name for pattern in patterns):
                continue
            for dirpath, _dirnames, filenames in os.walk(seq_dir):
                if any(f.lower().endswith(".dcm") for f in filenames):
                    series.append((subject_id, Path(dirpath), filenames))
    return series


def select_split_days(series: list, raw_dir: Path, split_dir: Path) -> list:
    """Keep the series acquired on a (subject, date) listed in the split files. Order: the first series
    of every listed day (earliest acquisition time, then image number -- the visit make_adni_csv.py
    keeps), then the second ones, and so on."""
    listed = set()
    for sp in ("train", "val", "test"):
        for line in (split_dir / f"{sp}.txt").read_text().split():
            sub, _, day = line.partition("_MR_")
            listed.add((sub, day))
    by_day = {}
    for s, d, files in series:
        acq = d.relative_to(raw_dir).parts[2]          # <subject>/<sequence>/<YYYY-MM-DD_HH_MM_SS.f>/<image_id>
        if (s, acq[:10]) in listed:
            by_day.setdefault((s, acq[:10]), []).append((acq, int(d.name[1:]) if d.name[1:].isdigit() else 0, s, d, files))
    ranked = []
    for key, items in by_day.items():
        for rank, (_, _, s, d, files) in enumerate(sorted(items)):
            ranked.append((rank, key, (s, d, files)))
    ranked.sort(key=lambda r: (r[0], r[1]))
    logger.info("--split-dir: %d series on %d of the %d listed (subject, date) visits; %d series outside them skipped",
                len(ranked), len(by_day), len(listed), len(series) - len(ranked))
    return [r[2] for r in ranked]


def source_fingerprint(filenames: list) -> str:
    """Number and names of the .dcm files. The raw tree is filled by extract_adni_t1w.py, which writes
    every file as .part and renames it when complete, so a slice appears under its final name only
    once it is fully written; an added or removed slice changes this."""
    dcm = sorted(f for f in filenames if f.lower().endswith(".dcm"))
    return f"{len(dcm)}:{hashlib.sha1(chr(10).join(dcm).encode()).hexdigest()[:16]}"


def is_settled(dicom_dir: Path, filenames: list, min_age_s: float) -> bool:
    """Not being written: no .part file and nothing added/renamed in the folder for `min_age_s`
    (the folder mtime changes with every file the extractor renames into it)."""
    if any(f.endswith(".part") for f in filenames):
        return False
    return time.time() - dicom_dir.stat().st_mtime >= min_age_s


def output_paths(dicom_dir: Path, raw_dir: Path, out_dir: Path, resolutions: list) -> dict:
    # Flattened to <subject>/<image_id>/: the sequence and visit-date levels are
    # dropped since image_id is already unique within a subject.
    subject_id = dicom_dir.relative_to(raw_dir).parts[0]
    image_id = dicom_dir.name
    base_dir = out_dir / subject_id / image_id
    paths = {
        "raw_nifti_dir": base_dir,
        "raw_nifti_stem": f"{image_id}_step0_raw",
        "biasfield": base_dir / f"{image_id}_step1_biasfield.nii.gz",
        "stripped": base_dir / f"{image_id}_step2_stripped.nii.gz",
        "registered": base_dir / f"{image_id}_step3_registered.nii.gz",
        "synthseg": base_dir / f"{image_id}_synthseg_native.nii.gz",
        "final": base_dir / f"{image_id}_final_native.nii.gz",
        "source": base_dir / f"{image_id}_source.json",
        "synthseg_resampled": {},
        "final_resampled": {},
    }
    for res in resolutions:
        tag = f"{res:g}mm"
        paths["synthseg_resampled"][res] = base_dir / f"{image_id}_synthseg_{tag}.nii.gz"
        paths["final_resampled"][res] = base_dir / f"{image_id}_final_{tag}.nii.gz"
    return paths


def is_complete(paths: dict) -> bool:
    # Intermediates are deleted once a series finishes, so the per-step existence
    # checks alone would redo the whole chain; the resampled outputs are the marker.
    finals = list(paths["synthseg_resampled"].values()) + list(paths["final_resampled"].values())
    return all(p.exists() for p in finals)


def recorded_fingerprint(paths: dict) -> str | None:
    try:
        return json.loads(paths["source"].read_text())["fingerprint"]
    except (OSError, ValueError, KeyError):
        return None


def discard_outputs(paths: dict) -> None:
    """Delete every output of one image (its raw series changed after it was processed)."""
    image_id = paths["source"].name[: -len("_source.json")]
    pipeline.cleanup(sorted(paths["raw_nifti_dir"].glob(f"{image_id}_*")))


class Run:
    """State shared by the CPU workers and the SynthSeg batch workers: settings, the result log
    and the completion count."""

    def __init__(self, args, raw_dir, out_dir, reference_brain, resolutions, keep, total, log_f):
        self.args = args
        self.raw_dir, self.out_dir, self.reference_brain = raw_dir, out_dir, reference_brain
        self.resolutions, self.keep, self.total = resolutions, keep, total
        self.log_f = log_f
        self.lock = threading.Lock()
        self.n_done = self.n_ok = 0
        self.start = time.time()

    def record(self, job: dict, error: Exception | None) -> None:
        if error is None:
            result = {"subject_id": job["subject_id"], "input": str(job["dicom_dir"]), "status": "ok",
                      "seconds": round(time.time() - job["start"], 1), "source": job["fingerprint"]}
        else:
            step = error.step if isinstance(error, pipeline.StepError) else type(error).__name__
            result = {"subject_id": job["subject_id"], "input": str(job["dicom_dir"]), "status": "failed",
                      "step": step, "error": str(error), "source": job["fingerprint"]}
        with self.lock:
            self.log_f.write(json.dumps(result) + "\n")
            self.log_f.flush()
            self.n_done += 1
            self.n_ok += error is None
            if self.n_done % 20 == 0 or self.n_done == self.total:
                logger.info("progress %d/%d (%d failed, %.0fs elapsed)", self.n_done, self.total,
                            self.n_done - self.n_ok, time.time() - self.start)


def cpu_stage(run: Run, job: dict) -> None:
    """Everything up to the SynthSeg input; also WhiteStripe and the image resampling, which
    don't need the label map."""
    paths, args = job["paths"], run.args
    job["raw_nifti"] = pipeline.dicom_to_nifti(job["dicom_dir"], paths["raw_nifti_dir"], paths["raw_nifti_stem"])
    pipeline.n4_bias_correction(job["raw_nifti"], paths["biasfield"])
    if not paths["stripped"].exists():
        with strip_cpu_slots:
            pipeline.skull_strip(paths["biasfield"], paths["stripped"], gpu=False, threads=args.strip_threads)
    pipeline.register_to_reference(paths["stripped"], paths["registered"], run.reference_brain, seed=args.reg_seed)
    pipeline.normalize_intensity(paths["registered"], paths["final"])
    for res in run.resolutions:
        pipeline.resample(paths["final"], paths["final_resampled"][res], res, is_segmentation=False)


def finish(run: Run, job: dict) -> None:
    """After SynthSeg: resample the label map, then delete what is not kept."""
    paths = job["paths"]
    for res in run.resolutions:
        pipeline.resample(paths["synthseg"], paths["synthseg_resampled"][res], res, is_segmentation=True)
    intermediates = {
        "raw": [job["raw_nifti"]],
        "biasfield": [paths["biasfield"]],
        "stripped": [paths["stripped"]],
        "registered": [paths["registered"]],
        "transforms": list(paths["raw_nifti_dir"].glob(f"{paths['registered'].name}_transform_*")),
    }
    pipeline.cleanup([p for step, ps in intermediates.items() if step not in run.keep for p in ps])
    if not run.args.keep_native_res:
        pipeline.cleanup([paths["synthseg"], paths["final"]])
    paths["source"].write_text(json.dumps({"fingerprint": job["fingerprint"], "dicom_dir": str(job["dicom_dir"])}) + "\n")


class SegBatcher:
    """Queue of registered volumes waiting for SynthSeg, drained by one worker thread per GPU
    slot (or a single CPU worker when no GPU is given). A worker takes a batch once --seg-batch
    volumes are waiting, the oldest has waited --seg-wait seconds, or the CPU stage is over."""

    def __init__(self, run: Run, specs: list, slots_per_gpu: int, seg_env: dict | None):
        self.run, self.seg_env = run, seg_env
        self.queue = []                      # (enqueue time, job)
        self.cond = threading.Condition()
        self.closed = False
        slots = [spec for spec in specs for _ in range(slots_per_gpu)] or [(None, None, None)]
        self.workers = [threading.Thread(target=self._work, args=slot, daemon=True) for slot in slots]
        for t in self.workers:
            t.start()

    def submit(self, job: dict) -> None:
        with self.cond:
            self.queue.append((time.time(), job))
            self.cond.notify_all()

    def close_and_join(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        for t in self.workers:
            t.join()
        for _, job in self.queue:            # every worker retired before the queue emptied
            self.run.record(job, RuntimeError("SynthSeg not run: every GPU passed its deadline (re-run to finish)"))

    def _take(self, deadline: datetime | None) -> list | None:
        size, wait = self.run.args.seg_batch, self.run.args.seg_wait
        with self.cond:
            while True:
                retry = bool(self.queue) and self.queue[0][1].get("retry", False)
                if self.queue and (retry or len(self.queue) >= size or self.closed
                                   or time.time() - self.queue[0][0] >= wait):
                    n = 1 if retry else min(size, len(self.queue))
                    if deadline is not None:
                        fit = (deadline - datetime.now() - SEG_STARTUP_EST) // SEG_PER_VOLUME_EST
                        n = min(n, int(fit))
                        if n < 1:
                            return None
                    batch = [job for _, job in self.queue[:n]]
                    del self.queue[:n]
                    return batch
                if self.closed:
                    return None
                self.cond.wait(timeout=10)

    def _work(self, gid: int | None, start: datetime | None, deadline: datetime | None) -> None:
        env = pipeline.on_gpu(self.seg_env, gid) if gid is not None else None
        args = self.run.args
        if start is not None and datetime.now() < start:
            logger.info("GPU %s: SynthSeg starts at %s", gid, start)
            with self.cond:
                while datetime.now() < start and not (self.closed and not self.queue):
                    self.cond.wait(timeout=60)
            if datetime.now() < start:
                return
            logger.info("GPU %s: start time reached, taking SynthSeg batches", gid)
        while True:
            batch = self._take(deadline)
            if batch is None:
                if deadline is not None and self.queue:
                    logger.warning("GPU %s: no SynthSeg batch fits before %s any more, retiring it", gid, deadline)
                return
            t0 = time.time()
            items = [(job["paths"]["registered"], job["paths"]["synthseg"]) for job in batch]
            try:
                done, err = pipeline.segment_batch(items, gpu=gid is not None, list_dir=self.run.out_dir / "logs" / "seg_batches",
                                                   env=env, threads=args.seg_threads)
            except Exception as e:           # e.g. mri_synthseg not startable: fail the batch, keep going
                done, err = [], f"{type(e).__name__}: {e}"
            logger.info("GPU %s: SynthSeg batch of %d in %.0fs, %d written%s", gid, len(batch), time.time() - t0,
                        len(done), f" ({err.splitlines()[0] if err else ''})" if err else "")
            done = set(done)
            for job in batch:
                if job["paths"]["synthseg"] in done:
                    try:
                        finish(self.run, job)
                        self.run.record(job, None)
                    except Exception as e:
                        logger.exception("finishing %s failed", job["dicom_dir"])
                        self.run.record(job, e)
                elif len(batch) > 1:
                    job["retry"] = True          # alone, so one bad volume can't sink a whole batch again
                    with self.cond:
                        self.queue.insert(0, (time.time(), job))
                        self.cond.notify_all()
                else:
                    self.run.record(job, pipeline.StepError("segment", ["mri_synthseg"], -1, err or "no output"))


def process_series(run: Run, batcher: SegBatcher, gate: CpuGate, subject_id: str, dicom_dir: Path,
                   fingerprint: str) -> None:
    job = {"subject_id": subject_id, "dicom_dir": dicom_dir, "fingerprint": fingerprint, "start": time.time(),
           "paths": output_paths(dicom_dir, run.raw_dir, run.out_dir, run.resolutions)}
    try:
        with gate:
            cpu_stage(run, job)
    except Exception as e:
        if isinstance(e, pipeline.StepError):
            logger.error("failed %s: %s", dicom_dir, e)
        else:
            logger.exception("unexpected failure on %s", dicom_dir)
        run.record(job, e)
        return
    if job["paths"]["synthseg"].exists():
        finish(run, job)
        run.record(job, None)
    else:
        batcher.submit(job)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ADNI T1 series (no CSV output).")
    parser.add_argument("--raw-dir", default=os.environ.get("RAW_DATA_DIR"))
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    parser.add_argument("--reference-brain",
                         default=os.environ.get("REFERENCE_BRAIN", str(TOOLS_DIR / "MNI152_T1_1mm_brain.nii.gz")))
    parser.add_argument("--fs-license", default=os.environ.get("FS_LICENSE", str(TOOLS_DIR / "license.txt")))
    parser.add_argument("--sequences", default=",".join(DEFAULT_T1_PATTERNS),
                         help="Comma-separated substrings (case-sensitive); a sequence folder matches if its name contains any of them. "
                              "'all' takes every sequence folder (the team_data raw tree is already T1-only).")
    parser.add_argument("--resolutions", default="1.0,1.5",
                         help="Comma-separated isotropic resolutions (mm) to resample synthseg/final to. All are kept -- disk is not the constraint here.")
    parser.add_argument("--cpu-workers", type=int, default=4,
                         help="Max concurrency of the CPU stage (dcm2niix, N4, SynthStrip, registration, WhiteStripe).")
    parser.add_argument("--cpu-min", type=int, default=2, help="Lower bound of the load-adaptive CPU concurrency.")
    parser.add_argument("--cpu-start", type=int, default=4, help="CPU concurrency at start (then load-adaptive).")
    parser.add_argument("--load-high", type=float, default=0.9,
                         help="Lower the CPU concurrency while the 1-min load average exceeds this x cores.")
    parser.add_argument("--load-low", type=float, default=0.75,
                         help="Raise the CPU concurrency while the 1-min load average is below this x cores.")
    parser.add_argument("--cpu-adjust-s", type=float, default=120, help="Seconds between concurrency adjustments.")
    parser.add_argument("--split-dir", default=None,
                         help="Only the series acquired on a date listed in <split-dir>/{train,val,test}.txt "
                              "(<sub_id>_MR_<YYYY-MM-DD> per line). The first series of each such day goes first, "
                              "the same-day repeats after them.")
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "4"),
                         help="Physical GPU ids (nvidia-smi index), comma-separated, each with an optional time window: "
                              "'id@FROM..UNTIL' (either end may be empty; 'id@UNTIL' = until). A GPU takes SynthSeg "
                              "batches only inside its window. Empty = SynthSeg on the CPU.")
    parser.add_argument("--slots-per-gpu", type=int, default=1,
                         help="Concurrent SynthSeg processes per GPU (each peaks at ~10.7 GB).")
    parser.add_argument("--seg-batch", type=int, default=48,
                         help="Max volumes per mri_synthseg call (start-up is paid once per call).")
    parser.add_argument("--seg-wait", type=float, default=900,
                         help="Seconds the oldest queued volume waits for a full batch before a partial one starts.")
    parser.add_argument("--seg-threads", type=int, default=8,
                         help="mri_synthseg --threads (CPU-side pre/post-processing).")
    parser.add_argument("--itk-threads", type=int, default=4,
                         help="Threads per N4/antsRegistration call (ITK otherwise takes every core of the shared server).")
    parser.add_argument("--strip-threads", type=int, default=None,
                         help="PyTorch CPU threads per SynthStrip process. SynthStrip always runs on the CPU: "
                              "FreeSurfer 8.2's bundled torch is CPU-only, so mri_synthstrip -g aborts.")
    parser.add_argument("--strip-workers", type=int, default=1,
                         help="Concurrent SynthStrip processes (~8-10 GB RAM each).")
    parser.add_argument("--reg-seed", type=int, default=1,
                         help="antsRegistration --random-seed (nonzero), for a reproducible affine.")
    parser.add_argument("--tfcuda-site-packages",
                         default=os.environ.get("TFCUDA_SITE_PACKAGES",
                                                "/mnt/aix22308/envs/tfcuda/lib/python3.8/site-packages"),
                         help="site-packages holding the pip nvidia-* CUDA 11.8/cuDNN 8.6 wheels that FreeSurfer's TensorFlow needs for GPU SynthSeg.")
    parser.add_argument("--keep", default="",
                         help=f"Comma-separated intermediates to keep once a series finishes, out of {','.join(INTERMEDIATES)} "
                              "(transforms = the ANTs affine .mat). The rest is deleted.")
    parser.add_argument("--keep-intermediates", action="store_true", help="Same as --keep with every intermediate.")
    parser.add_argument("--keep-native-res", action="store_true",
                         help="Keep native-resolution synthseg/final instead of deleting them once resampling finishes.")
    parser.add_argument("--list-only", action="store_true", help="Discover series and print them -- no processing.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N discovered series (debugging).")
    parser.add_argument("--min-age-min", type=float, default=60,
                         help="Skip series whose folder changed less than this many minutes ago or still holds .part "
                              "files (raw tree still being filled); a later run picks them up.")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()
    keep = set(INTERMEDIATES) if args.keep_intermediates else {k.strip() for k in args.keep.split(",") if k.strip()}
    if keep - set(INTERMEDIATES):
        parser.error(f"--keep: unknown {sorted(keep - set(INTERMEDIATES))}, choose from {INTERMEDIATES}")
    if args.reg_seed == 0:
        parser.error("--reg-seed must be nonzero (antsRegistration treats 0 as unseeded)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.raw_dir or not args.out_dir:
        parser.error("--raw-dir/--out-dir (or RAW_DATA_DIR/OUT_DIR) are required")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.exists():
        parser.error(f"raw dir not found: {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = [] if args.sequences == "all" else [s.strip() for s in args.sequences.split(",") if s.strip()]
    series = discover_series(raw_dir, patterns)
    if args.split_dir:
        series = select_split_days(series, raw_dir, Path(args.split_dir))
    subjects = sorted({s for s, _, _ in series})
    logger.info("discovered %d T1 series across %d subjects under %s (patterns=%s)",
                len(series), len(subjects), raw_dir, patterns)

    if args.list_only:
        for subject_id, dicom_dir, _ in series[:50]:
            n_dcm = sum(1 for _ in dicom_dir.glob("*.dcm"))
            print(subject_id, dicom_dir, f"({n_dcm} dcm files)")
        if len(series) > 50:
            print(f"... and {len(series) - 50} more")
        return

    reference_brain = Path(args.reference_brain)
    if not reference_brain.exists():
        parser.error(f"reference brain not found: {reference_brain} (put MNI152_T1_1mm_brain.nii.gz there, or pass --reference-brain)")

    if args.fs_license:
        if not Path(args.fs_license).exists():
            parser.error(f"FreeSurfer license not found: {args.fs_license}")
        os.environ["FS_LICENSE"] = args.fs_license

    pipeline.require_tools()
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.itk_threads)
    global strip_cpu_slots
    strip_cpu_slots = threading.Semaphore(args.strip_workers)

    seg_env = None
    gpu_specs = parse_gpus(args.gpus)
    if gpu_specs:
        seg_env = pipeline.synthseg_gpu_env(Path(args.tfcuda_site_packages))
        free_mb = pipeline.gpu_free_mb()
        for gid, start, dl in gpu_specs:
            name = pipeline.check_synthseg_gpu(pipeline.on_gpu(seg_env, gid))
            logger.info("SynthSeg GPU preflight OK: GPU %d (%s, %s MiB free)%s%s", gid, name, free_mb.get(gid, "?"),
                        f", from {start}" if start else "", f", until {dl}" if dl else "")
            if free_mb.get(gid, 1 << 30) < 12000 * args.slots_per_gpu:
                logger.warning("GPU %d has only %d MiB free; each SynthSeg slot needs ~11 GB, batches on it may "
                               "fail with CUDA OOM (their volumes are then retried one by one)", gid, free_mb[gid])

    resolutions = [float(r.strip()) for r in args.resolutions.split(",") if r.strip()]

    todo, n_done, pending, stale = [], 0, [], []
    for s, d, files in series:
        paths, fp = output_paths(d, raw_dir, out_dir, resolutions), source_fingerprint(files)
        if is_complete(paths):
            old = recorded_fingerprint(paths)
            if old is None or old == fp:
                n_done += 1
                continue
            stale.append((paths, d, old, fp))
        if not is_settled(d, files, args.min_age_min * 60):
            pending.append(d)
            continue
        todo.append((s, d, fp))
    for paths, d, old, fp in stale:
        if d in pending:
            continue
        logger.warning("raw series changed since it was processed (%s -> %s), redoing: %s", old, fp, d)
        discard_outputs(paths)
    logger.info("%d already complete, %d to process, %d skipped as still being written (changed < %g min ago "
                "or .part files; re-run later), %d redone because their raw series changed",
                n_done, len(todo), len(pending), args.min_age_min, sum(d not in pending for _, d, _, _ in stale))
    if pending:
        logger.info("  e.g. %s", [str(d) for d in pending[:3]])
    series = todo

    if args.limit:
        series = series[: args.limit]

    log_path = Path(args.log_file) if args.log_file else out_dir / "preprocess_log.jsonl"
    with log_path.open("a") as log_f:
        run = Run(args, raw_dir, out_dir, reference_brain, resolutions, keep, len(series), log_f)
        batcher = SegBatcher(run, gpu_specs, args.slots_per_gpu, seg_env)
        gate = CpuGate(args, out_dir / "logs" / "cpu_workers")
        with ThreadPoolExecutor(max_workers=args.cpu_workers) as executor:
            for future in [executor.submit(process_series, run, batcher, gate, s, d, fp) for s, d, fp in series]:
                future.result()
        logger.info("CPU stage finished, waiting for the remaining SynthSeg batches")
        batcher.close_and_join()

    logger.info("done: %d ok, %d failed. log: %s", run.n_ok, run.n_done - run.n_ok, log_path)


if __name__ == "__main__":
    main()
