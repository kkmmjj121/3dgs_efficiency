#!/bin/bash
# ============================================================
# Evaluation: render test split & print PSNR/SSIM/LPIPS
# ============================================================

# ---- 설정 ----
GPU=${1:-2}
OUT_ROOT=/NasData/home/kmj/blender_exp/output
DATA_ROOT=/NasData/home/kmj/data
GS_RENDER=/NasData/home/kmj/soccernet/sn-nvs/submodules/gaussian-splatting/render.py
GS_METRICS=/NasData/home/kmj/soccernet/sn-nvs/submodules/gaussian-splatting/metrics.py
CONDA_ENV=gs

SCENES=(chair drums ficus hotdog lego materials mic ship)

echo "============================================"
echo "[Eval] GPU=${GPU}"
echo "============================================"

for SCENE in "${SCENES[@]}"; do
    for EXP in "baseline_2k_30k" "progressive_3stage"; do
        MODEL_DIR="${OUT_ROOT}/${SCENE}_${EXP}"
        if [ ! -d "${MODEL_DIR}" ]; then
            echo "[SKIP] Not found: ${MODEL_DIR}"
            continue
        fi
        echo ""
        echo ">>> Rendering [${SCENE}] [${EXP}]"

        conda run -n ${CONDA_ENV} python ${GS_RENDER} \
            -s ${DATA_ROOT}/${SCENE} \
            -m ${MODEL_DIR} \
            -r 1 \
            --white_background \
            --skip_train \
            --gpu ${GPU}

        echo ">>> Metrics [${SCENE}] [${EXP}]"
        conda run -n ${CONDA_ENV} python ${GS_METRICS} \
            -m ${MODEL_DIR}
    done
done

echo ""
echo "============================================"
echo "Eval done! Results saved in each model_dir/results.json"
