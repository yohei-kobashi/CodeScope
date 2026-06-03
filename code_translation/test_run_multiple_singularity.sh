#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if command -v module >/dev/null 2>&1; then
  module load singularity squashfuse 2>/dev/null || true
fi

TEST_JSONL="${TEST_JSONL:-test.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-result/test_run_multiple_singularity.json}"
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-${HOME}/multipl-e-eval_sandbox}"
MULTIPLE_E_EVALUATION_DIR="${MULTIPLE_E_EVALUATION_DIR:-${HOME}/MultiPL-E/evaluation}"
SINGULARITY_RUNTIME="${SINGULARITY_RUNTIME:-singularity}"
SINGULARITY_PWD="${SINGULARITY_PWD:-/code}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_INFLIGHT_TOTAL="${MAX_INFLIGHT_TOTAL:-4}"
MAX_INFLIGHT_PER_LANG="${MAX_INFLIGHT_PER_LANG:-2}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
BATCH_TIMEOUT="${BATCH_TIMEOUT:-}"
RANDOM_SAMPLE_SIZE="${RANDOM_SAMPLE_SIZE:-0}"
RANDOM_SAMPLE_SEED="${RANDOM_SAMPLE_SEED:-0}"

export MULTIPLE_E_EVALUATION_DIR

if [[ ! -f "${TEST_JSONL}" ]]; then
  echo "Missing test JSONL: ${TEST_JSONL}" >&2
  exit 1
fi

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

mkdir -p "$(dirname "${OUTPUT_PATH}")"

COMMON_ARGS=(
  --jsonl_path "${TEST_JSONL}"
  --output_path "${OUTPUT_PATH}"
  --singularity_image "${SINGULARITY_IMAGE}"
  --singularity_runtime "${SINGULARITY_RUNTIME}"
  --singularity_pwd "${SINGULARITY_PWD}"
  --request_timeout "${REQUEST_TIMEOUT}"
  --max_workers "${MAX_WORKERS}"
  --max_inflight_total "${MAX_INFLIGHT_TOTAL}"
  --max_inflight_per_lang "${MAX_INFLIGHT_PER_LANG}"
  --random_sample_size "${RANDOM_SAMPLE_SIZE}"
  --random_sample_seed "${RANDOM_SAMPLE_SEED}"
)

if [[ -n "${BATCH_TIMEOUT}" ]]; then
  COMMON_ARGS+=(--batch_timeout "${BATCH_TIMEOUT}")
fi

echo "[Test] Evaluating known-correct data: ${TEST_JSONL}"
echo "[Test] Output: ${OUTPUT_PATH}"

python evaluator/run_multiple_singularity.py "${COMMON_ARGS[@]}"

python - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
with output_path.open(encoding="utf-8") as f:
    result = json.load(f)

info = result.get("info", {})
code_sum = int(info.get("code_sum", 0))
correct_sum = int(info.get("correct_sum", 0))
wrong_num = int(info.get("wrong_num", 0))
error_num = int(info.get("error_num", 0))
invalid_num = int(info.get("invalid_num", 0))
aborted = bool(info.get("aborted", False))
accuracy = float(info.get("accuracy", 0.0))

print(
    "[Test] Summary: "
    f"code_sum={code_sum} correct_sum={correct_sum} "
    f"wrong_num={wrong_num} error_num={error_num} "
    f"invalid_num={invalid_num} accuracy={accuracy:.6f}"
)

failures = []
if aborted:
    failures.append("evaluation aborted")
if code_sum <= 0:
    failures.append("no code was evaluated")
if correct_sum != code_sum:
    failures.append(f"expected all evaluated programs to pass, got {correct_sum}/{code_sum}")
if wrong_num != 0:
    failures.append(f"wrong_num is {wrong_num}")
if error_num != 0:
    failures.append(f"error_num is {error_num}")
if invalid_num != 0:
    failures.append(f"invalid_num is {invalid_num}")

if failures:
    print("[Test] FAILED")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("[Test] PASSED")
PY
