#!/bin/bash
#PBS -q prepost
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe

set -euo pipefail

module purge
module load singularity squashfuse

cd "${HOME}/CodeScope"
source env_c/bin/activate
cd code_translation

export MULTIPLE_E_EVALUATION_DIR="${HOME}/MultiPL-E/evaluation"
image="${HOME}/MultiPL-E/multipl-e-eval_sandbox"

runs=(
  "qwen3.5_4b_grpo_0702_reward35b_nothink_step100_instruct_max8192_defaultpp_seed42"
  "qwen3.5_4b_grpo_0702_reward35b_nothink_step100_instruct_max8192_pp0_seed42"
  "qwen3.5_9b_grpo_0702_reward35b_nothink_step100_instruct_max8192_defaultpp_seed42"
  "qwen3.5_9b_grpo_0702_reward35b_nothink_step100_instruct_max8192_pp0_seed42"
)

for run_name in "${runs[@]}"; do
  input_path="result/code_translation_eval_${run_name}.jsonl"
  output_path="result/eval_${run_name}_singularity_fixed.json"
  if [[ -s "$output_path" ]]; then
    echo "Skipping completed evaluation: ${output_path}"
    continue
  fi
  echo "Evaluating ${input_path}"
  python evaluator/run_multiple_singularity.py \
    --jsonl_path "$input_path" \
    --output_path "$output_path" \
    --singularity_image "$image" \
    --singularity_runtime singularity \
    --singularity_pwd /code \
    --request_timeout 900 \
    --compile_timeout 120 \
    --max_workers 112 \
    --max_inflight_total 112 \
    --max_inflight_per_lang 16
done
