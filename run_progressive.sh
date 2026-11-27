#!/bin/bash
# ============================================================
# Progressive Resolution Training
# Stage1: 0.5K (-r 4)  iter    0 ~ 10,000
# Stage2: 1K   (-r 2)  iter 10,001 ~ 20,000  (resume stage1)
# Stage3: 2K   (-r 1)  iter 20,001 ~ 30,000  (resume stage2)
# 측정: 단계별/전체 학습 시간, 단계별/피크 VRAM, PSNR/SSIM/LPIPS
# 사용법: bash run_progressive.sh [GPU번호]  (기본: GPU 3)
# ============================================================

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
GPU=${1:-3}

STAGE1_ITER=10000
STAGE2_ITER=20000
STAGE3_ITER=30000

SCENES=(chair drums ficus hotdog lego materials mic ship)

# ------ VRAM 모니터링 함수 ------
start_vram_monitor() {
    local LOG_FILE=$1
    rm -f "${LOG_FILE}"
    nvidia-smi -i ${GPU} --query-gpu=memory.used --format=csv,noheader,nounits -l 5 >> "${LOG_FILE}" &
    echo $!
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
echo "[Progressive] GPU=${GPU}, 3-Stage (0.5K->1K->2K, each 10k iter)"
echo "Scenes: ${SCENES[*]}"
echo "============================================"

for SCENE in "${SCENES[@]}"; do
    OUT_DIR="${OUT_ROOT}/${SCENE}_progressive_3stage"
    mkdir -p "${OUT_DIR}"
    STATS_FILE="${OUT_DIR}/train_stats.json"

    echo ""
    echo ">>> [${SCENE}] Start  $(date '+%Y-%m-%d %H:%M:%S')"

    T_TOTAL_START=$(date +%s)

    # ========== Stage 1: 0.5K ==========
    echo "  -- Stage1: 0.5K (-r 4), iter 0~${STAGE1_ITER}"
    VRAM_LOG_S1="${OUT_DIR}/vram_s1.txt"
    T_S1_START=$(date +%s)
    VRAM_PID=$(start_vram_monitor "${VRAM_LOG_S1}")

    CUDA_VISIBLE_DEVICES=${GPU} conda run -n "${CONDA_ENV}" python "${GS_TRAIN}" \
        -s "${DATA_ROOT}/${SCENE}" \
        -m "${OUT_DIR}" \
        -r 4 \
        --iterations ${STAGE1_ITER} \
        --test_iterations ${STAGE1_ITER} \
        --save_iterations ${STAGE1_ITER} \
        --checkpoint_iterations ${STAGE1_ITER} \
        --white_background \
        --eval \
        --disable_viewer

    stop_vram_monitor "${VRAM_PID}"
    T_S1_END=$(date +%s)
    S1_ELAPSED=$((T_S1_END - T_S1_START))
    S1_PEAK_VRAM=$(get_peak_vram "${VRAM_LOG_S1}")
    echo "     Stage1 time: ${S1_ELAPSED}s | Peak VRAM: ${S1_PEAK_VRAM} MiB"

    CKPT_S1="${OUT_DIR}/chkpnt${STAGE1_ITER}.pth"
    if [ ! -f "${CKPT_S1}" ]; then
        echo "[ERROR] Stage1 checkpoint not found: ${CKPT_S1}"
        exit 1
    fi

    # ========== Stage 2: 1K ==========
    echo "  -- Stage2: 1K (-r 2), iter ${STAGE1_ITER}~${STAGE2_ITER}"
    VRAM_LOG_S2="${OUT_DIR}/vram_s2.txt"
    T_S2_START=$(date +%s)
    VRAM_PID=$(start_vram_monitor "${VRAM_LOG_S2}")

    CUDA_VISIBLE_DEVICES=${GPU} conda run -n "${CONDA_ENV}" python "${GS_TRAIN}" \
        -s "${DATA_ROOT}/${SCENE}" \
        -m "${OUT_DIR}" \
        -r 2 \
        --iterations ${STAGE2_ITER} \
        --test_iterations ${STAGE2_ITER} \
        --save_iterations ${STAGE2_ITER} \
        --checkpoint_iterations ${STAGE2_ITER} \
        --start_checkpoint "${CKPT_S1}" \
        --white_background \
        --eval \
        --disable_viewer

    stop_vram_monitor "${VRAM_PID}"
    T_S2_END=$(date +%s)
    S2_ELAPSED=$((T_S2_END - T_S2_START))
    S2_PEAK_VRAM=$(get_peak_vram "${VRAM_LOG_S2}")
    echo "     Stage2 time: ${S2_ELAPSED}s | Peak VRAM: ${S2_PEAK_VRAM} MiB"

    CKPT_S2="${OUT_DIR}/chkpnt${STAGE2_ITER}.pth"
    if [ ! -f "${CKPT_S2}" ]; then
        echo "[ERROR] Stage2 checkpoint not found: ${CKPT_S2}"
        exit 1
    fi

    # ========== Stage 3: 2K ==========
    echo "  -- Stage3: 2K (-r 1), iter ${STAGE2_ITER}~${STAGE3_ITER}"
    VRAM_LOG_S3="${OUT_DIR}/vram_s3.txt"
    T_S3_START=$(date +%s)
    VRAM_PID=$(start_vram_monitor "${VRAM_LOG_S3}")

    CUDA_VISIBLE_DEVICES=${GPU} conda run -n "${CONDA_ENV}" python "${GS_TRAIN}" \
        -s "${DATA_ROOT}/${SCENE}" \
        -m "${OUT_DIR}" \
        -r 1 \
        --iterations ${STAGE3_ITER} \
        --test_iterations ${STAGE3_ITER} \
        --save_iterations ${STAGE3_ITER} \
        --checkpoint_iterations ${STAGE3_ITER} \
        --start_checkpoint "${CKPT_S2}" \
        --white_background \
        --eval \
        --disable_viewer

    stop_vram_monitor "${VRAM_PID}"
    T_S3_END=$(date +%s)
    S3_ELAPSED=$((T_S3_END - T_S3_START))
    S3_PEAK_VRAM=$(get_peak_vram "${VRAM_LOG_S3}")
    echo "     Stage3 time: ${S3_ELAPSED}s | Peak VRAM: ${S3_PEAK_VRAM} MiB"

    T_TOTAL_END=$(date +%s)
    TOTAL_ELAPSED=$((T_TOTAL_END - T_TOTAL_START))

    # 피크 VRAM = 세 단계 중 최대
    PEAK_VRAM=$(echo -e "${S1_PEAK_VRAM}\n${S2_PEAK_VRAM}\n${S3_PEAK_VRAM}" | sort -n | tail -1)

    echo "  Total time : ${TOTAL_ELAPSED}s | Overall Peak VRAM: ${PEAK_VRAM} MiB"

    # 가우시안 개수 & PLY 파일 크기 측정 (최종 stage3 모델 기준)
    PLY_PATH="${OUT_DIR}/point_cloud/iteration_${STAGE3_ITER}/point_cloud.ply"
    if [ -f "${PLY_PATH}" ]; then
        NUM_GAUSSIANS=$(conda run -n "${CONDA_ENV}" python -c 'import sys; from plyfile import PlyData; print(len(PlyData.read(sys.argv[1])["vertex"]))' "${PLY_PATH}" 2>/dev/null || echo "0")
        PLY_SIZE_MB=$(python3 -c 'import os, sys; print(round(os.path.getsize(sys.argv[1])/1024/1024, 2))' "${PLY_PATH}" 2>/dev/null || echo "0")
    else
        NUM_GAUSSIANS=0
        PLY_SIZE_MB=0
    fi
    echo "  Gaussians  : ${NUM_GAUSSIANS}"
    echo "  PLY size   : ${PLY_SIZE_MB} MB"

    # train_stats.json 저장
    cat > "${STATS_FILE}" <<EOF
{
  "scene": "${SCENE}",
  "experiment": "progressive_3stage",
  "gpu": ${GPU},
  "total_iterations": ${STAGE3_ITER},
  "peak_vram_mib": ${PEAK_VRAM},
  "total_train_seconds": ${TOTAL_ELAPSED},
  "num_gaussians": ${NUM_GAUSSIANS},
  "ply_size_mb": ${PLY_SIZE_MB},
  "stages": [
    {"name": "stage1_0.5K", "resolution": "0.5K", "r_flag": 4,
     "iter_start": 0, "iter_end": ${STAGE1_ITER},
     "train_seconds": ${S1_ELAPSED}, "peak_vram_mib": ${S1_PEAK_VRAM}},
    {"name": "stage2_1K", "resolution": "1K", "r_flag": 2,
     "iter_start": ${STAGE1_ITER}, "iter_end": ${STAGE2_ITER},
     "train_seconds": ${S2_ELAPSED}, "peak_vram_mib": ${S2_PEAK_VRAM}},
    {"name": "stage3_2K", "resolution": "2K", "r_flag": 1,
     "iter_start": ${STAGE2_ITER}, "iter_end": ${STAGE3_ITER},
     "train_seconds": ${S3_ELAPSED}, "peak_vram_mib": ${S3_PEAK_VRAM}}
  ]
}
EOF


    # 렌더링 (test split, 최종 모델은 2K(-r 1)로 평가)
    echo "  Rendering test split..."
    CUDA_VISIBLE_DEVICES=${GPU} conda run -n "${CONDA_ENV}" python "${GS_RENDER}" \
        -s "${DATA_ROOT}/${SCENE}" \
        -m "${OUT_DIR}" \
        -r 1 \
        --white_background \
        --eval \
        --skip_train

    echo "  Computing metrics..."
    CUDA_VISIBLE_DEVICES=${GPU} conda run -n "${CONDA_ENV}" python "${GS_METRICS}" \
        -m "${OUT_DIR}"

    echo "<<< [${SCENE}] Done   $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "============================================"
echo "All progressive training done!"
