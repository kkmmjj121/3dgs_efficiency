# 3DGS Efficiency

3D Gaussian Splatting에서 고정 해상도 학습과 점진적 해상도 학습을 비교하는 실험 스크립트입니다. Blender Synthetic의 8개 장면을 대상으로 학습 시간, GPU 메모리 사용량, 가우시안 개수와 렌더링 품질을 기록합니다.

## 비교 설정

| 방식 | 입력 해상도 | 반복 횟수 |
| --- | --- | --- |
| Baseline | 원본 해상도 (`-r 1`) | 30,000 |
| Progressive 1단계 | 가로·세로 1/4 (`-r 4`) | 0 → 10,000 |
| Progressive 2단계 | 가로·세로 1/2 (`-r 2`) | 10,000 → 20,000 |
| Progressive 3단계 | 원본 해상도 (`-r 1`) | 20,000 → 30,000 |

Progressive는 이전 단계의 체크포인트를 이어서 학습합니다. 최종 모델은 두 방식 모두 원본 해상도의 test split에서 평가합니다. 장면은 `chair`, `drums`, `ficus`, `hotdog`, `lego`, `materials`, `mic`, `ship`입니다. 0.5K/1K/2K 표기는 원본 이미지가 약 2K일 때의 구분입니다.

## 준비

[Gaussian Splatting 공식 구현](https://github.com/graphdeco-inria/gaussian-splatting)의 학습·렌더링·평가 환경과 Blender Synthetic 데이터를 별도로 준비해야 합니다. 이 저장소에는 해당 구현, 데이터, 학습 가중치를 포함하지 않습니다.

기본 경로는 실행 위치와 관계없이 이 저장소를 기준으로 합니다.

```text
3dgs_efficiency/
├── data/<scene>/             # Blender Synthetic 데이터
├── gaussian-splatting/       # train.py, render.py, metrics.py
└── output/                  # 실험 결과
```

다른 위치를 사용하려면 환경변수를 지정합니다.

```bash
export GS_ROOT=/path/to/gaussian-splatting
export DATA_ROOT=/path/to/nerf_synthetic
export OUT_ROOT=/path/to/experiment_output
export CONDA_ENV=gs
```

`GS_TRAIN`, `GS_RENDER`, `GS_METRICS`로 개별 실행 파일도 지정할 수 있습니다. 경로에 공백이 있는 경우 따옴표로 감싸세요. Conda 환경에는 3DGS와 지표 계산 의존성, `plyfile`이 설치되어 있어야 하며, GPU 메모리 기록에는 `nvidia-smi`를 사용합니다.

## 실행

저장소 폴더에서 실행합니다. 마지막 숫자는 사용할 GPU 번호입니다.

```bash
bash run_baseline.sh 0
bash run_progressive.sh 0
bash run_eval.sh 0
python summarize.py
```

```bash
# 다른 출력 폴더의 결과를 정리할 때
python summarize.py --output-dir /path/to/experiment_output
```

학습 스크립트가 사용하는 `--start_checkpoint`, `--checkpoint_iterations`, `--disable_viewer` 등의 옵션을 지원하는 3DGS 구현이 필요합니다. 이 저장소의 경로 수정 검증에서는 학습을 다시 수행하지 않았습니다.

## 결과와 파일

- `run_baseline.sh`: 고정 해상도 학습 후 렌더링·평가.
- `run_progressive.sh`: 3단계 해상도 학습 후 렌더링·평가.
- `run_eval.sh`: 저장된 모델의 test split 재평가.
- `summarize.py`: `train_stats.json`과 `results.json`을 읽어 장면별·전체 평균 비교표 출력.
- `paths.sh`: 공통 경로 및 Conda 환경 설정.

결과는 `output/<scene>_baseline_2k_30k/`와 `output/<scene>_progressive_3stage/`에 저장합니다. GPU 메모리는 5초 간격으로 측정한 장치 전체 사용량이므로, 같은 GPU의 다른 프로세스 사용량과 샘플 사이의 피크가 측정에 영향을 줄 수 있습니다.

이전 실험 기록에서는 평균 메모리 약 17.5%, 학습 시간 약 17.0%, 가우시안 개수 약 58.7% 감소와 PSNR 약 0.49 dB 하락을 보고했습니다. 원본 결과 파일은 이 저장소에 포함되어 있지 않으며, 해당 수치는 이전 실행 기록이지 모든 환경에서 보장하는 성능은 아닙니다. 당시 기록 환경은 Ubuntu 20.04, Xeon Silver 4214R, RTX A6000 48GB였습니다.
