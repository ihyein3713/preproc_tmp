#!/usr/bin/env bash
# Copy the parts of FreeSurfer 8.2 that mri_synthstrip / mri_synthseg need to /dev/shm (RAM disk, ~4 GB),
# so env_server.sh can use it instead of the NFS install. Re-run after a reboot (/dev/shm is cleared).
set -euo pipefail
SRC=/mnt/aix22308/tools/fs_extract/usr/local/freesurfer/8.2.0-1
DST=/dev/shm/aix22308_freesurfer/8.2.0-1

rm -f "$DST/.copy_complete"
mkdir -p "$DST/models"
cd "$SRC"
rsync -a --exclude='*/' ./ "$DST/"                    # top-level setup scripts and LUTs
rsync -a bin etc luts python sessions "$DST/"
rsync -a models/synthseg* models/synthstrip* "$DST/models/"
touch "$DST/.copy_complete"
du -sh "$DST"
