#!/bin/bash
# Run all round* configs for selected methods on the MATH dataset using llama3.1-8b.

set -euo pipefail

MODEL_NAME="llama3.1-8b-local"
DATASET_NAME="MATH"
METHODS=("div_mad" "llm_debate" "dmad")

for method in "${METHODS[@]}"; do
    config_dir="methods/${method}/configs"
    if [[ ! -d "${config_dir}" ]]; then
        echo "[WARN] Config directory not found for ${method}, skipping."
        continue
    fi

    shopt -s nullglob
    configs=("${config_dir}"/round*.yaml)
    shopt -u nullglob

    if [[ ${#configs[@]} -eq 0 ]]; then
        echo "[INFO] No round* configs for ${method}, skipping."
        continue
    fi

    for config_path in "${configs[@]}"; do
        config_name=$(basename "${config_path}" .yaml)
        output_path="results/${DATASET_NAME}/${MODEL_NAME}/${method}_${config_name}.jsonl"

        echo ">>> Running ${method} with config ${config_name}"
        python inference.py \
            --method_name "${method}" \
            --method_config_name "${config_name}" \
            --model_name "${MODEL_NAME}" \
            --test_dataset_name "${DATASET_NAME}" \
            --output_path "${output_path}"
    done
done
