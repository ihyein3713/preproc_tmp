"""Re-register the series that manual QC marked fail, without touching ADNI_processed.

The main run calls antsRegistration with no initial transform, so the moving image starts at its
scanner position. When the header origin puts the brain far from MNI's (e.g. a centre of mass at
z = -160 mm) the two barely overlap, MI has nothing to work with, and the affine collapses (scales
~0.3, brain pushed out of the field of view). Here each series is registered from its kept
step2_stripped image with a few alternatives that start from an aligned centre and add a rigid
stage first; the candidate whose brain mask overlaps the MNI mask best is kept, and only that one
is carried through WhiteStripe, resampling and SynthSeg (GPU, on the registered image as in the
main run).

    source custom/env_server.sh
    python custom/retry_qcfailed.py [--workers 6] [--gpu 4]

Outputs (same file names as ADNI_processed) go to <out>/<subject>/<image_id>/, plus
<out>/summary.csv (original vs every candidate) and <out>/compare/<image_id>.png
(top: original final 1.5 mm, bottom: retry). Re-runs skip finished series, so newly saved fails
can be picked up by running it again.
"""
from __future__ import annotations

import argparse, csv, os, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline  # noqa: E402

sys.path.insert(0, "/mnt/aix22308/team_data/manual_qc_adni")
import qc_render  # noqa: E402

PROCESSED = Path("/mnt/aix22308/team_data/ADNI_processed")
QC_FAILED = PROCESSED / "qc_failed.csv"
OUT = Path("/mnt/aix22308/team_data/retry_adni_qcfailed")
TEMPLATE = Path("/mnt/aix22308/tools/MNI152_T1_1mm_brain.nii.gz")
RESOLUTIONS = (1.0, 1.5)

# name -> (initial-moving-transform mode, sampling, seed); mode 1 = centre of mass, 0 = geometric centre
CANDIDATES = {
    "com_s1":  (1, "Regular", 1),
    "com_s7":  (1, "Random", 7),
    "geom_s1": (0, "Regular", 1),
}


def tag(res):
    return f"{res:g}mm"


def register(src: Path, dst: Path, init: int, sampling: str, seed: int) -> None:
    if dst.exists():
        return
    tmp = pipeline._prepare(dst)
    prefix = str(dst) + "_transform_"
    pipeline._run("register", [
        "antsRegistration", "-d", "3", "--random-seed", str(seed),
        "--initial-moving-transform", f"[{TEMPLATE},{src},{init}]",
        "-n", "BSpline", "-o", f"[{prefix},{tmp}]",
        "--transform", "Rigid[0.1]",
        "-m", f"MI[{TEMPLATE},{src},1,32,{sampling},0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]", "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--transform", "Affine[0.1]",
        "-m", f"MI[{TEMPLATE},{src},1,32,{sampling},0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]", "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ])
    tmp.rename(dst)


_T = {}


def overlap(path: Path) -> dict:
    """Dice of the brain mask with the MNI brain mask and correlation with the template over their
    union, on the 1 mm registration output (same grid as the template). The mask uses the same
    threshold as pipeline.normalize_intensity."""
    if "v" not in _T:
        _T["v"] = np.asanyarray(nib.load(str(TEMPLATE)).dataobj).astype(np.float32)
    tv = _T["v"]; tm = tv > 0
    v = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)
    pos = v[v > 0]
    m = v > 1e-3 * np.percentile(pos, 99) if pos.size else np.zeros(v.shape, bool)
    dice = 2 * (m & tm).sum() / max(1, m.sum() + tm.sum())
    u = m | tm
    a, b = v[u], tv[u]
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    return dict(dice=round(float(dice), 4), corr=round(corr, 4), brain_ml=round(m.sum() / 1000, 1))


def paths(base: Path, iid: str) -> dict:
    return dict(
        registered=base / f"{iid}_step3_registered.nii.gz",
        final=base / f"{iid}_final_native.nii.gz",
        synthseg=base / f"{iid}_synthseg_native.nii.gz",
        final_res={r: base / f"{iid}_final_{tag(r)}.nii.gz" for r in RESOLUTIONS},
        seg_res={r: base / f"{iid}_synthseg_{tag(r)}.nii.gz" for r in RESOLUTIONS},
    )


def cpu_job(sub: str, iid: str) -> dict:
    """Candidates -> pick best -> WhiteStripe -> resample image. Returns a summary row."""
    src_dir = PROCESSED / sub / iid
    base = OUT / sub / iid
    cand_dir = base / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    row = dict(subject_id=sub, image_id=iid)
    for k, v in overlap(src_dir / f"{iid}_step3_registered.nii.gz").items():
        row[f"orig_{k}"] = v
    scores = {}
    for name, (init, sampling, seed) in CANDIDATES.items():
        dst = cand_dir / f"{iid}_{name}.nii.gz"
        try:
            register(src_dir / f"{iid}_step2_stripped.nii.gz", dst, init, sampling, seed)
            scores[name] = overlap(dst)
        except Exception as e:
            scores[name] = dict(dice=-1.0, corr=-1.0, brain_ml=0.0)
            row[f"{name}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        for k, v in scores[name].items():
            row[f"{name}_{k}"] = v
    best = max(scores, key=lambda n: (scores[n]["dice"], scores[n]["corr"]))
    row["best"] = best
    p = paths(base, iid)
    if not p["registered"].exists():
        chosen = cand_dir / f"{iid}_{best}.nii.gz"
        for t in cand_dir.glob(f"{chosen.name}_transform_*"):
            shutil.copy2(t, base / t.name.replace(f"{iid}_{best}.nii.gz", p["registered"].name))
        os.replace(chosen, p["registered"])
    shutil.rmtree(cand_dir, ignore_errors=True)     # 39 MB per candidate; scores are in summary.csv
    pipeline.normalize_intensity(p["registered"], p["final"])
    for r in RESOLUTIONS:
        pipeline.resample(p["final"], p["final_res"][r], r, is_segmentation=False)
    return row


def compare_png(sub: str, iid: str, dst: Path) -> None:
    """Sagittal | coronal | axial centre slices, original (top) over retry (bottom)."""
    rows = []
    for vol_path in (PROCESSED / sub / iid / f"{iid}_final_1.5mm.nii.gz",
                     OUT / sub / iid / f"{iid}_final_1.5mm.nii.gz"):
        vol = np.asanyarray(nib.load(str(vol_path)).dataobj).astype(np.float32)
        lo, hi, _ = qc_render.window(vol)
        panels = [np.repeat(np.repeat(qc_render.to_u8(f(vol), lo, hi), 2, 0), 2, 1)
                  for f in (qc_render.sagittal, qc_render.coronal, qc_render.axial)]
        h = max(x.shape[0] for x in panels)
        padded = []
        for n, x in enumerate(panels):
            if n:
                padded.append(np.zeros((h, 6), np.uint8))
            padded.append(np.concatenate([x, np.zeros((h - x.shape[0], x.shape[1]), np.uint8)], 0))
        rows.append(np.concatenate(padded, 1))
    img = np.concatenate([rows[0], np.full((6, rows[0].shape[1]), 128, np.uint8), rows[1]], 0)
    dst.write_bytes(qc_render.png(img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--itk-threads", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--tfcuda-site-packages",
                    default=os.environ.get("TFCUDA_SITE_PACKAGES",
                                           "/mnt/aix22308/envs/tfcuda/lib/python3.8/site-packages"))
    args = ap.parse_args()
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.itk_threads)
    pipeline.require_tools(["antsRegistration", "mri_synthseg", "intensity-normalize"])
    (OUT / "compare").mkdir(parents=True, exist_ok=True)

    with open(QC_FAILED) as f:
        todo = [(r["subject_id"], r["image_id"]) for r in csv.DictReader(f)]
    summary_path = OUT / "summary.csv"
    old = {}
    if summary_path.exists():
        with open(summary_path) as f:
            old = {r["image_id"]: r for r in csv.DictReader(f)}
    done = {iid for sub, iid in todo
            if iid in old and paths(OUT / sub / iid, iid)["seg_res"][1.5].exists()}
    jobs = [(s, i) for s, i in todo if i not in done]
    print(f"{len(todo)} in qc_failed.csv, {len(done)} already retried, {len(jobs)} to do", flush=True)

    rows = dict(old)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(cpu_job, s, i): (s, i) for s, i in jobs}
        for n, fu in enumerate(as_completed(futs), 1):
            s, i = futs[fu]
            try:
                r = fu.result()
                rows[i] = r
                print(f"[{n}/{len(jobs)}] {i}: orig dice {r['orig_dice']} -> {r['best']} "
                      f"dice {r[r['best'] + '_dice']}", flush=True)
            except Exception as e:
                rows[i] = dict(subject_id=s, image_id=i, error=f"{type(e).__name__}: {str(e)[:300]}")
                print(f"[{n}/{len(jobs)}] {i}: FAILED {e}", flush=True)

    seg_env = pipeline.on_gpu(pipeline.synthseg_gpu_env(Path(args.tfcuda_site_packages)), args.gpu)
    items = []
    for s, i in jobs:
        p = paths(OUT / s / i, i)
        if p["registered"].exists() and not p["seg_res"][1.5].exists():
            items.append((p["registered"], p["synthseg"]))     # same input as the main run (before WhiteStripe)
    if items:
        print(f"SynthSeg on GPU {args.gpu}: {len(items)} volumes", flush=True)
        _, err = pipeline.segment_batch(items, gpu=True, list_dir=OUT / "seg_batches", env=seg_env, threads=8)
        if err:
            print("SynthSeg error:", err, flush=True)
    for s, i in jobs:
        p = paths(OUT / s / i, i)
        if p["synthseg"].exists():
            for r in RESOLUTIONS:
                pipeline.resample(p["synthseg"], p["seg_res"][r], r, is_segmentation=True)
            p["synthseg"].unlink()
            p["final"].unlink(missing_ok=True)      # native = 1 mm grid; the main run doesn't keep it either
        if p["final_res"][1.5].exists():
            compare_png(s, i, OUT / "compare" / f"{i}.png")

    cols = ["subject_id", "image_id", "best", "orig_dice", "orig_corr", "orig_brain_ml"]
    for n in CANDIDATES:
        cols += [f"{n}_dice", f"{n}_corr", f"{n}_brain_ml"]
    extra = sorted({k for r in rows.values() for k in r} - set(cols))
    order = [i for _, i in todo] + [i for i in rows if i not in {i for _, i in todo}]
    tmp = summary_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, cols + extra)
        w.writeheader()
        for i in order:
            if i in rows:
                w.writerow(rows[i])
    os.replace(tmp, summary_path)
    print(f"summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
