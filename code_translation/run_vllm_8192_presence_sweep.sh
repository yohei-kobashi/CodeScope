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

request_batch_size="${REQUEST_BATCH_SIZE:-512}"
max_new_tokens=8192
max_model_len=16384
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-65536}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
seed="${SEED:-42}"

model_paths=(
  "Qwen/Qwen3.5-4B"
  "Qwen/Qwen3.5-9B"
  "/work/go25/share/model/code_trans_grpo_model_0702/Qwen3.5_4B_grpo_0702_reward35b_nothink/global_step_100"
  "/work/go25/share/model/code_trans_grpo_model_0702/Qwen3.5_9B_grpo_0702_reward35b_nothink/global_step_100"
)
model_tags=(
  "qwen3.5_4b"
  "qwen3.5_9b"
  "qwen3.5_4b_grpo_0702_reward35b_nothink_step100"
  "qwen3.5_9b_grpo_0702_reward35b_nothink_step100"
)
model_variants=("4B" "9B" "4B" "9B")

common_args=(
  --batch_size "$request_batch_size"
  --candidate_num 1
  --max_model_len "$max_model_len"
  --gpu_memory_utilization "$gpu_memory_utilization"
  --max_num_batched_tokens "$max_num_batched_tokens"
  --tensor_parallel_size 1
  --kv_cache_dtype fp8
  --enable_chunked_prefill
  --enable_prefix_caching
  --language_model_only
  --mtp_tokens 0
  --seed "$seed"
  --use_sft_prompt_template
)

run_one() {
  local model_path="$1"
  local model_tag="$2"
  local model_variant="$3"
  local mode_name="$4"
  local penalty_name="$5"
  local presence_penalty="$6"
  local max_num_seqs
  local -a sampling_args

  if [[ "$model_variant" == "9B" ]]; then
    max_num_seqs="${MAX_NUM_SEQS_9B:-384}"
  else
    max_num_seqs="${MAX_NUM_SEQS_4B:-512}"
  fi

  if [[ "$mode_name" == "thinking" ]]; then
    sampling_args=(
      --do_sample true
      --temperature 0.6
      --top_p 0.95
      --top_k 20
      --min_p 0.0
      --presence_penalty "$presence_penalty"
      --repetition_penalty 1.0
      --enable_thinking
    )
  else
    sampling_args=(
      --do_sample true
      --temperature 0.7
      --top_p 0.8
      --top_k 20
      --min_p 0.0
      --presence_penalty "$presence_penalty"
      --repetition_penalty 1.0
    )
  fi

  local run_tag="${model_tag}_${mode_name}_max8192_${penalty_name}_seed${seed}"
  echo "Starting: model=${model_path} mode=${mode_name} presence_penalty=${presence_penalty}"

  python3 inference/run_vllm.py \
    --model "$model_path" \
    --max_new_tokens "$max_new_tokens" \
    --result_save_name "code_translation_eval_${run_tag}.jsonl" \
    --log_file_name "code_translation_eval_${run_tag}.log" \
    --random_sample_ratio 1.0 \
    --max_num_seqs "$max_num_seqs" \
    "${common_args[@]}" \
    "${sampling_args[@]}"
}

for i in "${!model_paths[@]}"; do
  for mode_name in instruct thinking; do
    if [[ "$mode_name" == "thinking" ]]; then
      default_penalty=0.0
    else
      default_penalty=1.5
    fi

    run_one \
      "${model_paths[$i]}" "${model_tags[$i]}" "${model_variants[$i]}" \
      "$mode_name" "defaultpp" "$default_penalty"
    run_one \
      "${model_paths[$i]}" "${model_tags[$i]}" "${model_variants[$i]}" \
      "$mode_name" "pp0" "0.0"
  done
done
EOF
