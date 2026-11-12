# 0817 ADNI 전처리 → 0917 Δ-LFM 학습 입력 호환성 + 실행 준비 (2026-10-23)

0917 학습 코드: `d_lfm_preproc0917/Disease-Progression-World-Model/ICLR 2026 Delta-LFM/` (sparse checkout에서 빠져 있어 `git show HEAD:...`로 읽음).
검증: `002_S_0295`의 3개 series를 실제로 전처리 → `run_adni.sh` → 0917의 `build_triplets` / `oasis_dataset_3D_pair_latent.get_dataframe` / MONAI transform에 그대로 넣어 확인.

## 1. 볼륨 (전처리 산출물 → 로더)

| 항목 | 0917 로더가 기대하는 것 | 산출물 | 상태 |
| --- | --- | --- | --- |
| 해상도/방향 | 1.5 mm, 재정렬 없음 (Step1/2는 Spacing 없음, Step3/eval은 `Spacing(1.5)`) | 1.5 mm, LAS (MNI), Spacing은 no-op → 두 경로 텐서 차이 0 | ✅ |
| 크기 | `ResizeWithPadOrCrop(128,144,128)` → `res_scale 0.82` → 112³ | (121,145,121): x/z pad, y 1 slice crop, 뇌 bbox 안 잘림 | ✅ |
| 배경 | `ScaleIntensity(0,1)` 후 배경 0 (CropForeground `>0.02`, eval Ω `>0.05` 전제) | **수정 전 0.08~0.10** → 수정 후 0.000 | ✅ (수정) |
| seg | SynthSeg 라벨 | NN 리샘플, codemap 밖 라벨 없음, t1과 affine 동일 | ✅ |
| 파일 트리 | `$DATA_DIR/Image/<subject>/<date>/t1.nii.gz`, latent = `<latent>/<path[-3]>/<path[-2]>/t1.npz` | `make_adni_csv.py`가 symlink tree 생성, 경로 규칙 일치 | ✅ |

배경 문제 원인: antsRegistration `-n BSpline` 출력이 뇌 경계에서 음수로 ringing(최저 약 -1300)하고 배경 전체에 ~1e-13 잡음이 있음. 마스크 없는 WhiteStripe가 잡음을 foreground로 세고, ringing 값이 배경보다 낮아서 min-max 후 배경이 0이 안 됨. → `pipeline.normalize_intensity`에서 `99퍼센타일 × 1e-3` 미만을 0으로 두고 그 마스크로 WhiteStripe (`--mask`). 등록 결과(`step3_registered`)는 그대로.

## 2. CSV (`make_adni_csv.py` + `build_multiprior.py` → 0917)

| 항목 | 상태 / 할 일 |
| --- | --- |
| Step3 `dataset_csv: ${DATA_DIR}/derived/AD-Progression-multiprior-dt15.csv` | ✅ 이름·위치·컬럼 일치 (prior/prior2 포함) |
| Step1/2 `dataset_csv: Final/AD-Progression-All.csv` | ⚠️ 우리 파일은 `derived/AD-Progression-All.csv`. `--dataset_csv derived/AD-Progression-All.csv`를 넘기거나 `Final/` 링크 필요 (`scripts/01,02`는 안 넘김) |
| Step2 latent 저장 위치 | ⚠️ `step2`는 `--output_dir`에 저장. RECIPE 4절 명령은 `--output_dir`가 빠져 있음 → `scripts/02_extract_latents.sh` 사용 |
| `sex` | ✅ F=0/M=1 (subject가 1명이면 0.5 fallback — 실제 데이터는 해당 없음) |
| diagnosis | 0917 Step3/eval도 여전히 `followup_dia - starting_dia`를 context에 넣음 (`step3_train_flowmatching.py:546,981`, `eval_fm.py:1017`). CN=0/MCI=0.5/AD=1이라 의미는 있음. 결측 -1이 pair에 남으면 망가짐 → 리포트 `[4]` 확인 후 필요하면 `PAIR_DX_MISSING=drop` |
| Δt class index | `int((followup_age-starting_age)*100)`가 `num_class_embeds=15` 미만이어야 함. ADNI는 15년 넘는 pair가 있을 수 있음 → 리포트 `[4b]`가 ERROR면 `DT_ARGS="--max_dt_months 179"` |
| split | Step1은 `ap|` md5, Step3/eval은 `split_v3`(`v3|` md5). `split` 컬럼은 안 읽음 → 팀 split 파일과 다름. `vol_norm_ref=train`의 train도 hash train과 다름. (CSV 작업은 보류하기로 함) |

## 3. 실행 환경 (서버, 이번에 준비)

- `custom/env_server.sh` 신규: FreeSurfer 8.2 (`/mnt/aix22308/tools/fs_extract/...`), `adni_preproc` env, `FS_LICENSE`, `TFCUDA_SITE_PACKAGES`. `set -euo pipefail` 스크립트에서 source 가능.
- `adni_preproc` env에 `intensity-normalization 3.0.1` 설치 (없었음).
- `/mnt/aix22308/envs/tfcuda/lib/python3.8/site-packages`에 CUDA 11.8 / cuDNN 8.6 pip wheel 설치 (FreeSurfer TF 2.13용). GPU 4 preflight 통과.
- FreeSurfer 8.2 내장 torch가 CPU 전용 → `mri_synthstrip -g`는 즉시 실패. SynthStrip은 CPU로 고정 (`--strip-workers`로 동시 실행 수).
- `antsRegistration --random-seed 1` (old.md #7).

## 4. 속도 (측정)

| 단계 | 시간 (1 series) |
| --- | --- |
| N4 | ~50 s |
| SynthStrip (CPU, 8 threads) | ~85 s |
| 등록 | ~25 s |
| WhiteStripe | ~30 s |
| SynthSeg | **호출당 시작 6~10분** (NFS에서 TF import) + scan당 ~20 s |

SynthSeg를 series마다 호출하면 5903 series × ~6분 / GPU 2장 ≈ 300시간 → **batch로 변경**: CPU 단계가 끝난 볼륨을 큐에 모으고 GPU slot마다 `mri_synthseg --i list.txt`를 최대 48개씩 실행. 한 batch에서 실패한 볼륨은 하나씩 재시도.
예상: CPU 단계 ~120 series/h (CPU_WORKERS=8, STRIP_WORKERS=4) → 5903 series ≈ 2일. GPU 1장이 ~120/h를 처리하므로 GPU5가 빠져도 병목은 CPU.

## 5. 토요일 시작 절차

```bash
cd /mnt/aix22308/longi_v2/d_lfm_preproc0817/AD_data_processing
nvidia-smi --query-gpu=index,memory.free --format=csv     # GPU 5는 현재 다른 사용자(~40 GB 사용 중), 12 GB 미만이면 GPUS=4
bash custom/run_preprocess_adni.sh --list-only             # raw 정리 끝난 뒤, series 수 확인 (~5900)
bash custom/run_preprocess_adni.sh --limit 5               # 실제 출력 경로에 5개 (≈15분)
tmux new -s adni   # 또는 nohup
bash custom/run_preprocess_adni.sh 2>&1 | tee /mnt/aix22308/team_data/ADNI_processed/logs/run_$(date +%Y%m%d_%H%M).log
```

- GPU 5: `GPUS=4,5@2026-10-27T22:30` 기본값. batch 예상 종료(시작 15분 + 30 s/볼륨)가 22:30을 넘으면 GPU 5에서 새 batch를 시작하지 않음 → 10/28 0시 전에 비워짐. 진행 중인 batch는 끝까지 감.
- 중단/재시작 안전: 끝난 series는 건너뛰고, CPU 단계까지 끝난 series는 SynthSeg부터 이어서 함. 모든 GPU가 은퇴해 남은 볼륨은 log에 failed로 남고, 다시 실행하면 처리됨.
- raw 정리(`team_data/code/extract_adni_t1w.py`)가 끝나기 전에 시작해도 됨 (6절).
- 디스크: series당 ~75 MB (`KEEP=stripped,registered,transforms` 포함, registered가 53 MB) → 5903개 ≈ 440 GB. `/mnt` 여유 44 TB.

## 6. raw 정리 중에 시작해도 되나 (2026-10-25 검토)

**된다.** 볼륨 전처리는 series 하나만 보고 끝난다 (N4, SynthStrip, 고정 MNI 템플릿 등록, 볼륨별 WhiteStripe, SynthSeg). SynthSeg batch도 볼륨을 서로 섞지 않는다. 여러 영상이 얽히는 일(같은 날 중복 제거, 부피 정규화, pair, split)은 전부 나중의 CSV 단계에서 출력 트리를 보고 한다. 그래서 처리 순서·시점은 결과에 영향이 없다.

위험은 하나: **추출이 덜 된 series를 읽는 것.** `extract_adni_t1w.py`는 파일을 `.part`로 쓰고 rename하지만 series 폴더는 파일 단위로 채워진다. 중간에 읽으면 dcm2niix가 일부 slice만으로 볼륨을 만들고, 그 결과가 "완료"로 남는다. DICOM 헤더로는 못 잡는다 (표본 40명 확인: `ImagesInAcquisition` 없음, Philips는 multi-frame 1파일도 있음).

그래서 `preprocess_adni.py`에 추가:
- `--min-age-min 60` (기본): `.part`가 있거나 폴더가 60분 안에 바뀐 series는 이번 실행에서 건너뜀 (로그에 개수). 다음 실행 때 처리.
- 완료한 영상마다 `<image_id>_source.json`에 `.dcm` 개수+이름 해시를 기록. 다시 실행했을 때 raw series가 달라져 있으면 (slice 추가/삭제) 그 영상의 출력을 지우고 다시 처리.
- `preprocess_log.jsonl`에도 `source` 기록.

운영: 추출이 진행 중이어도 지금 실행 → 추출이 끝나고 60분 뒤 같은 명령 한 번 더 (완료된 것은 건너뜀, 새로 생긴/바뀐 series만 처리). 시작할 때 목록을 한 번 만들기 때문에 실행 중에 추가된 series는 다음 실행에서 잡힌다.

## 7. 실제 실행 (2026-10-26 00:53 시작)

- 범위: split 파일에 있는 (subject, 날짜) 방문의 series만 (`--split-dir`) → 14,649 / 22,035. 방문마다 첫 series(가장 이른 촬영)를 먼저, 같은 날 repeat는 그 뒤.
- GPU: `GPUS=5@..2026-10-27T22:30,4@2026-10-27T21:30..` — 화요일 21:30까지 GPU 5만, 21:30부터 GPU 4 합류, GPU 5는 22:30 전에 끝날 batch만 시작. 수동 전환 불필요.
- CPU: load 적응형. 시작 4, 1분 load > 0.9×112 이면 -1, < 0.75×112 이면 +1 (2분마다, 2~12). 고정하려면 `echo 6 > /mnt/aix22308/team_data/ADNI_processed/logs/cpu_workers`, 지우면 자동으로 복귀.
- tmux 세션 `adni_preproc`, 로그 `ADNI_processed/logs/run_20261026_0053.log`.
- 10/26 09:33 재시작: CPU(~290/h)가 GPU 1장(~222/h)보다 빨라 SynthSeg 대기열이 쌓여서 GPU 4·5 동시 사용으로 변경 (`GPUS=5@..2026-10-27T22:30,4`). 로그 `ADNI_processed/logs/run_20261026_0933.log`. 재시작 전 1,632개 완료.
- 10/26 20:42 재시작: 16시부터 /mnt NFS 혼잡으로 처리량이 시간당 60~100개로 떨어져, FreeSurfer 필요 부분(5.8 GB)을 `/dev/shm/aix22308_freesurfer`로 복사 (`custom/fs_to_shm.sh`, 재부팅 후 다시 실행). `env_server.sh`는 `.copy_complete`가 있으면 복사본 사용. 측정: TF import 72~137 s → 3 s, SynthStrip 85 s → 19 s (출력 동일), GPU preflight ~60 s → 3 s. 로그 `run_20261026_2042.log`. 재시작 시점 4,204개 완료.
