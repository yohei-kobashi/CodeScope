#!/bin/bash
#PBS -q short-c
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe

set -euo pipefail

module purge
module load singularity squashfuse

cd "${HOME}/CodeScope"
source env_c/bin/activate
cd code_translation

SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${HOME}/multipl-e-eval_sandbox}"
MULTIPLE_E_EVALUATION_DIR="${MULTIPLE_E_EVALUATION_DIR:-${HOME}/MultiPL-E/evaluation}"
SINGULARITY_RUNTIME="${SINGULARITY_RUNTIME:-singularity}"
SINGULARITY_PWD="${SINGULARITY_PWD:-/code}"
export MULTIPLE_E_EVALUATION_DIR

MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_INFLIGHT_TOTAL="${MAX_INFLIGHT_TOTAL:-4}"
MAX_INFLIGHT_PER_LANG="${MAX_INFLIGHT_PER_LANG:-2}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-120}"
BATCH_TIMEOUT="${BATCH_TIMEOUT:-}"

if [[ ! -e "${SINGULARITY_IMAGE}" ]]; then
  echo "Missing Singularity image/sandbox: ${SINGULARITY_IMAGE}" >&2
  echo "Set SINGULARITY_IMAGE=/path/to/multipl-e-eval_sandbox or .sif" >&2
  exit 1
fi

if [[ ! -d "${MULTIPLE_E_EVALUATION_DIR}" ]]; then
  echo "Missing MultiPL-E evaluation directory: ${MULTIPLE_E_EVALUATION_DIR}" >&2
  echo "Set MULTIPLE_E_EVALUATION_DIR=/path/to/MultiPL-E/evaluation" >&2
  exit 1
fi

if ! python -c "import func_timeout" >/dev/null 2>&1; then
  echo "Missing Python package: func_timeout" >&2
  echo "Install it in the active environment with:" >&2
  echo "  python -m pip install func-timeout" >&2
  echo "or reinstall code_translation dependencies with:" >&2
  echo "  python -m pip install -r requirement.txt" >&2
  exit 1
fi

COMMON_ARGS=(
  --singularity_image "${SINGULARITY_IMAGE}"
  --singularity_runtime "${SINGULARITY_RUNTIME}"
  --singularity_pwd "${SINGULARITY_PWD}"
  --request_timeout "${REQUEST_TIMEOUT}"
  --compile_timeout "${COMPILE_TIMEOUT}"
  --max_workers "${MAX_WORKERS}"
  --max_inflight_total "${MAX_INFLIGHT_TOTAL}"
  --max_inflight_per_lang "${MAX_INFLIGHT_PER_LANG}"
)

if [[ -n "${BATCH_TIMEOUT}" ]]; then
  COMMON_ARGS+=(--batch_timeout "${BATCH_TIMEOUT}")
fi

EVALUATION_RUNS=(
  "qwen3.5_4b_instruct_max8192_defaultpp_seed42"
  "qwen3.5_4b_instruct_max8192_pp0_seed42"
  "qwen3.5_9b_instruct_max8192_defaultpp_seed42"
  "qwen3.5_9b_instruct_max8192_pp0_seed42"
  "qwen3.5_4b_grpo_0702_reward35b_nothink_step100_instruct_max8192_defaultpp_seed42"
  "qwen3.5_4b_grpo_0702_reward35b_nothink_step100_instruct_max8192_pp0_seed42"
  "qwen3.5_9b_grpo_0702_reward35b_nothink_step100_instruct_max8192_defaultpp_seed42"
  "qwen3.5_9b_grpo_0702_reward35b_nothink_step100_instruct_max8192_pp0_seed42"
)

for run_name in "${EVALUATION_RUNS[@]}"; do
  input_path="result/code_translation_eval_${run_name}.jsonl"
  output_path="result/eval_${run_name}_singularity.json"

  if [[ ! -s "$input_path" ]]; then
    echo "Missing or empty inference result: ${input_path}" >&2
    exit 1
  fi

  echo "Evaluating ${input_path}"
  python evaluator/run_multiple_singularity.py \
    --jsonl_path "$input_path" \
    --output_path "$output_path" \
    "${COMMON_ARGS[@]}"
done
