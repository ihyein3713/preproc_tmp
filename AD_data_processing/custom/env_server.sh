# Source me (bash): tools for custom/preprocess_adni.py on the aix server.
#   FreeSurfer 8.2 (mri_synthstrip, mri_synthseg) from the extracted rpm under /mnt/aix22308/tools,
#   dcm2niix / ANTs / intensity-normalize / nibabel from the adni_preproc env,
#   CUDA 11.8 + cuDNN 8.6 pip wheels (tfcuda) for FreeSurfer's TensorFlow 2.13 (GPU SynthSeg).
# FreeSurfer's bundled torch is CPU-only, so SynthStrip runs on the CPU (--strip-device cpu).

# A RAM-disk copy (bin, python, SynthSeg/SynthStrip models only; see custom/fs_to_shm.sh) is used when
# present: loading FreeSurfer's Python/TensorFlow from the shared NFS stalls every SynthStrip/SynthSeg
# call while /mnt is congested. /dev/shm is cleared on reboot, then the NFS install is used again.
FS_NFS=/mnt/aix22308/tools/fs_extract/usr/local/freesurfer/8.2.0-1
FS_SHM=/dev/shm/aix22308_freesurfer/8.2.0-1
if [ -z "${FREESURFER_HOME:-}" ]; then
    if [ -e "$FS_SHM/.copy_complete" ]; then FREESURFER_HOME="$FS_SHM"; else FREESURFER_HOME="$FS_NFS"; fi
fi
export FREESURFER_HOME
export FS_LICENSE="${FS_LICENSE:-/mnt/aix22308/tools/license.txt}"
export TFCUDA_SITE_PACKAGES="${TFCUDA_SITE_PACKAGES:-/mnt/aix22308/envs/tfcuda/lib/python3.8/site-packages}"
ADNI_PREPROC_ENV="${ADNI_PREPROC_ENV:-/mnt/aix22308/envs/adni_preproc}"

# SetUpFreeSurfer.sh reads unset variables and has failing greps, which `set -euo pipefail` in the
# calling script would reject: relax those options while sourcing it, then restore them
# ($(set +o) alone would not do: command substitution drops errexit)
_saved_opts="$(set +o)"; case $- in *e*) _saved_opts="$_saved_opts; set -e" ;; esac
set +euo pipefail
source "$FREESURFER_HOME/SetUpFreeSurfer.sh" > /dev/null
eval "$_saved_opts"
unset _saved_opts

# after FreeSurfer, so the env's python / ANTs come first on PATH
export PATH="$ADNI_PREPROC_ENV/bin:$PATH"
