#!/usr/bin/env bash
# Train the WT segmentation model inside the pathology-pipeline Docker image.
#
# Required env vars:
#   WT_DATA_ROOT    — host path to Samba mount containing images/ and masks/
#   WT_OUTPUT_ROOT  — host path for checkpoints and WandB logs
#
# Optional:
#   WT_RUN_NAME     — override the auto-generated run name
#   WT_IMAGE        — Docker image name (default: pathology-pipeline)
#
# Usage:
#   export WT_DATA_ROOT=/mnt/smb/wt_data
#   export WT_OUTPUT_ROOT=~/wt_runs
#   bash scripts/train.sh [run_name]
#
# The script needs -it because pytorch_exp_run.py prompts on stdin for
# architecture selection at startup.

set -euo pipefail

: "${WT_DATA_ROOT:?Set WT_DATA_ROOT to the Samba mount containing images/ and masks/}"
: "${WT_OUTPUT_ROOT:?Set WT_OUTPUT_ROOT to the host path for checkpoints/logs}"

WT_IMAGE="${WT_IMAGE:-pathology-pipeline}"
RUN_NAME="${WT_RUN_NAME:-${1:-wt_configB_$(date +%Y%m%d_%H%M)}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Run name:    ${RUN_NAME}"
echo "==> Data root:   ${WT_DATA_ROOT}"
echo "==> Output root: ${WT_OUTPUT_ROOT}"
echo "==> Docker image: ${WT_IMAGE}"

mkdir -p "${WT_OUTPUT_ROOT}/${RUN_NAME}"

# Pass WANDB_API_KEY if present in .env or environment
ENV_FLAGS=()
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    ENV_FLAGS+=(--env-file "${PROJECT_ROOT}/.env")
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    ENV_FLAGS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")
fi

docker run --gpus all -it --rm \
    "${ENV_FLAGS[@]}" \
    -v "${WT_DATA_ROOT}:/home/user/data:ro" \
    -v "${WT_OUTPUT_ROOT}:/home/user/process" \
    -v "${PROJECT_ROOT}/configs:/home/user/project/configs:ro" \
    -v "${PROJECT_ROOT}/data:/home/user/project/data:ro" \
    -v "${PROJECT_ROOT}/src:/home/user/project/src:ro" \
    "${WT_IMAGE}" \
    python3 /home/user/source/code/pytorch_exp_run.py \
        --project_name  wt-segmentation \
        --data_path     /home/user/project/data/wt_train.yaml \
        --config_path   /home/user/project/configs/wt_network_configuration.yaml \
        --alb_config_path /home/user/project/configs/wt_albumentations.yaml \
        --output_path   "/home/user/process/${RUN_NAME}"
