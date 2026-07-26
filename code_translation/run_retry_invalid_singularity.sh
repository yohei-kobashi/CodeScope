#!/bin/bash

set -euo pipefail

module purge
module load singularity squashfuse

cd "${HOME}/CodeScope"
source env_c/bin/activate
cd code_translation

export MULTIPLE_E_EVALUATION_DIR="${HOME}/MultiPL-E/evaluation"
image="${HOME}/MultiPL-E/multipl-e-eval_sandbox"
retry_dir="result/invalid_retries"

python evaluator/prepare_invalid_retries.py \
  --result-dir result \
  --output-dir "$retry_dir"

python -c '
import json
for item in json.load(open("result/invalid_retries/manifest.json")):
    print(item["input_path"] + "\t" + item["output_path"])
' | while IFS=$'\t' read -r input_path output_path; do
  echo "Retrying ${input_path}"
  python evaluator/run_multiple_singularity.py \
    --jsonl_path "$input_path" \
    --output_path "$output_path" \
    --no-resume \
    --singularity_image "$image" \
    --singularity_runtime singularity \
    --singularity_pwd /code \
    --request_timeout 900 \
    --compile_timeout 120 \
    --max_workers 8 \
    --max_inflight_total 8 \
    --max_inflight_per_lang 2
done
