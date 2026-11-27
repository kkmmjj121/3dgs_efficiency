#!/bin/bash
# ============================================================
# Baseline: 2K (full res, -r 1) / 30,000 iter
# 측정: 학습 시간, 피크 VRAM, PSNR/SSIM/LPIPS (test set)
# 사용법: bash run_baseline.sh [GPU번호]  (기본: GPU 2)
# ============================================================

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
GPU=${1:-2}
ITERS=30000

SCENES=(chair drums ficus hotdog lego materials mic ship)

# ------ VRAM 모니터링 함수 ------
start_vram_monitor() {
    local LOG_FILE=$1
    rm -f "${LOG_FILE}"
    # 해당 GPU의 used memory를 5초마다 기록 (MiB 단위)
    nvidia-smi -i ${GPU} --query-gpu=memory.used --format=csv,noheader,nounits -l 5 >> "${LOG_FILE}" &
    echo $!        # PID 반환
}

stop_vram_monitor() {
    local PID=$1
    kill "${PID}" 2>/dev/null
    wait "${PID}" 2>/dev/null
}

get_peak_vram() {
    local LOG_FILE=$1
    sort -n "${LOG_FILE}" 2>/dev/null | tail -1 || echo "0"
}

# ------ 메인 루프 ------
echo "[Baseline] GPU=${GPU}, Iter=${ITERS}, Resolution=2K(-r 1)"
echo "Scenes: ${SCENES[*]}"
echo "============================================"

for SCENE in "${SCENES[@]}"; do
    OUT_DIR="${OUT_ROOT}/${SCENE}_baseline_2k_30k"
    VRAM_LOG="${OUT_DIR}/vram_monitor.txt"
    STATS_FILE="${OUT_DIR}/train_stats.json"
    mkdir -p "${OUT_DIR}"

    echo ""
    echo ">>> [${SCENE}] Training Start  $(date '+%Y-%m-%d %H:%M:%S')"

    # 시간 측정 시작
    T_START=$(date +%s)

    # VRAM 모니터링 시작
    VRAM_PID=$(start_vram_monitor "${VRAM_LOG}")

    CUDA_VISIBLE_DEVICES=${GPU} conda run -n "${CONDA_ENV}" python "${GS_TRAIN}" \
        -s "${DATA_ROOT}/${SCENE}" \
        -m "${OUT_DIR}" \
        -r 1 \
        --iterations ${ITERS} \
        --test_iterations ${ITERS} \
        --save_iterations ${ITERS} \
        --checkpoint_iterations ${ITERS} \
        --white_background \
        --eval \
        --disable_viewer

    TRAIN_EXIT=$?

    # VRAM 모니터링 종료
    stop_vram_monitor "${VRAM_PID}"
    T_END=$(date +%s)

    ELAPSED=$((T_END - T_START))
    PEAK_VRAM=$(get_peak_vram "${VRAM_LOG}")

    echo "    Train time : ${ELAPSED}s"
    echo "    Peak VRAM  : ${PEAK_VRAM} MiB"

    # 학습 실패 시 건너뜀
    if [ ${TRAIN_EXIT} -ne 0 ]; then
        echo "[ERROR] Training failed for ${SCENE} (exit code ${TRAIN_EXIT}), skipping render & metrics."
        continue
    fi

    # 가우시안 개수 & PLY 파일 크기 측정
    PLY_PATH="${OUT_DIR}/point_cloud/iteration_${ITERS}/point_cloud.ply"
    if [ -f "${PLY_PATH}" ]; then
        NUM_GAUSSIANS=$(conda run -n "${CONDA_ENV}" python -c 'import sys; from plyfile import PlyData; print(len(PlyData.read(sys.argv[1])["vertex"]))' "${PLY_PATH}" 2>/dev/null || echo "0")
        PLY_SIZE_MB=$(python3 -c 'import os, sys; print(round(os.path.getsize(sys.argv[1])/1024/1024, 2))' "${PLY_PATH}" 2>/dev/null || echo "0")
    else
        NUM_GAUSSIANS=0
        PLY_SIZE_MB=0
    fi
    echo "    Gaussians  : ${NUM_GAUSSIANS}"
    echo "    PLY size   : ${PLY_SIZE_MB} MB"

    # train_stats.json 저장
    cat > "${STATS_FILE}" <<EOF
{
  "scene": "${SCENE}",
  "experiment": "baseline_2k_30k",
  "gpu": ${GPU},
  "iterations": ${ITERS},
  "resolution": "2K",
  "train_seconds": ${ELAPSED},
  "peak_vram_mib": ${PEAK_VRAM},
  "num_gaussians": ${NUM_GAUSSIANS},
  "ply_size_mb": ${PLY_SIZE_MB}
}
EOF

    # 렌더링 (test split)
    echo "    Rendering test split..."
    CUDA_VISIBLE_DEVICES=${GPU} conda run -n "${CONDA_ENV}" python "${GS_RENDER}" \
        -s "${DATA_ROOT}/${SCENE}" \
        -m "${OUT_DIR}" \
        -r 1 \
        --white_background \
        --eval \
        --skip_train

    # 지표 측정 (PSNR / SSIM / LPIPS)
    echo "    Computing metrics..."
    CUDA_VISIBLE_DEVICES=${GPU} conda run -n "${CONDA_ENV}" python "${GS_METRICS}" \
        -m "${OUT_DIR}"

    echo "<<< [${SCENE}] Done   $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "============================================"
echo "All baseline training done!"
echo "Summary: python \"${SCRIPT_DIR}/summarize.py\" --output-dir \"${OUT_ROOT}\""
