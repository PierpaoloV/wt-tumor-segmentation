#!/usr/bin/env bash
# Run held-out evaluation with awesomedice.py on a completed training run.
#
# Reads class names and mappings from src/wt_segmentation/labels.py via
# python3 -m wt_segmentation.labels --dump=... so the dictionaries never
# drift from the canonical source of truth.
#
# Required env vars:
#   WT_DATA_ROOT    — host path to Samba mount (same as train.sh)
#   WT_OUTPUT_ROOT  — host path for run outputs (same as train.sh)
#
# Usage:
#   export WT_DATA_ROOT=/mnt/smb/wt_data
#   export WT_OUTPUT_ROOT=~/wt_runs
#   bash scripts/evaluate.sh <run_name>
#
# The run directory must already contain a predictions/ subfolder produced
# by the pipeline's async tile inference.

set -euo pipefail

: "${WT_DATA_ROOT:?Set WT_DATA_ROOT}"
: "${WT_OUTPUT_ROOT:?Set WT_OUTPUT_ROOT}"
: "${1:?Usage: $0 <run_name>}"

WT_IMAGE="${WT_IMAGE:-pathology-pipeline}"
RUN_NAME="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Evaluating run: ${RUN_NAME}"

# Dump class dicts from labels.py (runs on the host, no Docker needed for this step)
CLASSES_JSON=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}/src')
from wt_segmentation.labels import INT_TO_ENGLISH
import json
print(json.dumps({v: k for k, v in INT_TO_ENGLISH.items()}))
")

MAPPING_JSON=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}/src')
from wt_segmentation.labels import LABEL_MAP
import json
print(json.dumps({str(k): v for k, v in LABEL_MAP.items()}))
")

echo "==> Classes: ${CLASSES_JSON}"
echo "==> Mapping: ${MAPPING_JSON}"

docker run --gpus all --rm \
    -v "${WT_DATA_ROOT}:/home/user/data:ro" \
    -v "${WT_OUTPUT_ROOT}:/home/user/process" \
    -v "${PROJECT_ROOT}/src:/home/user/project/src:ro" \
    "${WT_IMAGE}" \
    python3 /home/user/source/code/awesomedice.py \
        --input_mask_path  "/home/user/process/${RUN_NAME}/predictions/*.tif" \
        --ground_truth_path "/home/user/data/masks/{image}.tif" \
        --classes          "${CLASSES_JSON}" \
        --mapping          "${MAPPING_JSON}" \
        --spacing          2.0 \
        --all_cm \
        --output_path      "/home/user/process/${RUN_NAME}/scores.yaml"

echo "==> Scores written to ${WT_OUTPUT_ROOT}/${RUN_NAME}/scores.yaml"
