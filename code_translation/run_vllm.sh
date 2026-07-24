#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -W group_list=gj26
#PBS -j oe

set -euo pipefail

module purge
module load nvidia/25.9
module load singularity/4.2.1

export CC=gcc
export CXX=g++
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0"

cd "${PBS_O_WORKDIR}"

singularity exec --nv \
  --bind /work/go25:/work/go25 \
  /work/gj26/share/sif/vllm_v0.21.0.sif \
  bash <<'EOF'

set -euo pipefail

export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Miyabi uses a 96 GB GH200. These defaults leave headroom for CUDA graphs.
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS_4B="${MAX_NUM_SEQS_4B:-512}"
MAX_NUM_SEQS_9B="${MAX_NUM_SEQS_9B:-384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
CANDIDATE_NUM="${CANDIDATE_NUM:-1}"
USE_SFT_PROMPT_TEMPLATE=true

MODEL_NAMES=(
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-9B"
)

RUN_NAMES=(
  "evaluation_qwen3.5_4b"
  "evaluation_qwen3.5_9b"
)

common_args=(
  --batch_size "$REQUEST_BATCH_SIZE"
  --candidate_num "$CANDIDATE_NUM"
  --max_model_len "$MAX_MODEL_LEN"
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
  --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS"
  --tensor_parallel_size 1
  --kv_cache_dtype fp8
  --enable_chunked_prefill
  --enable_prefix_caching
  --language_model_only
  --mtp_tokens 0
  --seed 42
)

if [[ "$USE_SFT_PROMPT_TEMPLATE" == "true" ]]; then
  common_args+=(--use_sft_prompt_template)
fi

for model_index in "${!MODEL_NAMES[@]}"; do
  model_name="${MODEL_NAMES[$model_index]}"
  run_name="${RUN_NAMES[$model_index]}"
  max_new_tokens="$MAX_NEW_TOKENS"
  if [[ "$model_name" == *9B* ]]; then
    max_num_seqs="$MAX_NUM_SEQS_9B"
  else
    max_num_seqs="$MAX_NUM_SEQS_4B"
  fi

  for thinking in 0 1; do
    if [[ "$thinking" == "1" ]]; then
      mode_name="thinking"
      sampling_args=(
        --do_sample true
        --temperature 0.6
        --top_p 0.95
        --top_k 20
        --min_p 0.0
        --presence_penalty 0.0
        --repetition_penalty 1.0
        --enable_thinking
      )
    else
      mode_name="instruct"
      sampling_args=(
        --do_sample true
        --temperature 0.7
        --top_p 0.8
        --top_k 20
        --min_p 0.0
        --presence_penalty 1.5
        --repetition_penalty 1.0
      )
    fi

    result_save_name="code_translation_eval_${run_name}_${mode_name}_seed42.jsonl"
    log_file_name="code_translation_eval_${run_name}_${mode_name}_seed42.log"

    echo "Running ${model_name}; mode: ${mode_name}; result: ${result_save_name}; max_new_tokens: ${max_new_tokens}"

    python3 inference/run_vllm.py \
      --model "$model_name" \
      --max_new_tokens "$max_new_tokens" \
      --result_save_name "$result_save_name" \
      --log_file_name "$log_file_name" \
      --random_sample_ratio 1.0 \
      --max_num_seqs "$max_num_seqs" \
      "${common_args[@]}" \
      "${sampling_args[@]}"
  done
done
EOF
