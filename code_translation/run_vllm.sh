#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe
# module load cuda/12.8
# module load cudnn/9.10.1.4
# module load nvidia/25.3
# module load nv-hpcx/25.3
# source /work/gj26/b20048/miniconda3/etc/profile.d/conda.sh
# conda activate inference_env
# export CUDA_VISIBLE_DEVICES=0
# export PATH="$CONDA_PREFIX/bin:/opt/rh/gcc-toolset-14/root/usr/bin:$PATH"

# export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
# export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
# export TRITON_CC="$CC"
# export TRITON_CXX="$CXX"
# export CUDAHOSTCXX="$CXX"

# export PYTHONNOUSERSITE=1
module purge
export CUDA_VISIBLE_DEVICES=0
module load nvidia/25.9
module load singularity/4.2.1

singularity exec --nv --bind /work/go25:/work/go25 /work/gj26/share/sif/vllm_v0.21.0.sif bash <<'EOF'
cd CodeScope/code_translation

BATCH_SIZE=128
MAX_NEW_TOKENS=2048
CANDIDATE_NUM=1
USE_SFT_PROMPT_TEMPLATE=true
ENFORCE_EAGER=True

MODEL_NAMES=(
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen2.5_Coder_7B_grpo_reward30b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward7b/global_step_194"
  "/work/go25/share/model/code_trans_grpo_model_0409/Qwen3.5_4B_grpo_reward80b/global_step_194"
)

RUN_NAMES=(
  "Qwen2.5_7B_grpo_reward30b_2"
  "Qwen3.5_4B_grpo_reward7b"
  "Qwen3.5_4B_grpo_reward80b_2"
)

common_args=(
  --batch_size "$BATCH_SIZE"
  --candidate_num "$CANDIDATE_NUM"
  --enforce_eager "$ENFORCE_EAGER"
)

if [[ "$USE_SFT_PROMPT_TEMPLATE" == "true" ]]; then
  common_args+=(--use_sft_prompt_template)
fi

max_tokens_for_model() {
  local model_name="$1"

  case "$model_name" in
    *Qwen3.5_4B_grpo_reward80b*|*Qwen3.5-4B*reward80b*)
      echo 4096
      ;;
    *)
      echo "$MAX_NEW_TOKENS"
      ;;
  esac
}

sampling_args_for_model() {
  local model_name="$1"

  case "$model_name" in
    *Qwen3.5*|*qwen3.5*)
      echo "--temperature 0.6 --top_p 0.95 --top_k 20"
      ;;
    *Qwen2.5-Coder-7B-Instruct*|*Qwen2.5_Coder_7B*|*qwen2.5_coder_7b*)
      echo "--temperature 0.7 --top_p 0.8 --top_k 20"
      ;;
    *)
      echo "--temperature 0.5 --top_p 0.95 --top_k 50"
      ;;
  esac
}

for model_index in "${!MODEL_NAMES[@]}"; do
  model_name="${MODEL_NAMES[$model_index]}"
  run_name="${RUN_NAMES[$model_index]}"
  max_new_tokens="$(max_tokens_for_model "$model_name")"
  read -r -a sampling_args <<< "$(sampling_args_for_model "$model_name")"

  result_save_name="code_translation_eval_${run_name}.jsonl"
  log_file_name="code_translation_eval_${run_name}.log"

  echo "Running ${model_name}; result: ${result_save_name}; max_new_tokens: ${max_new_tokens}; sampling args: ${sampling_args[*]}"

  python3 inference/run_vllm.py \
    --model "$model_name" \
    --max_new_tokens "$max_new_tokens" \
    --result_save_name "$result_save_name" \
    --log_file_name "$log_file_name" \
    "${common_args[@]}" \
    "${sampling_args[@]}"
done
EOF
