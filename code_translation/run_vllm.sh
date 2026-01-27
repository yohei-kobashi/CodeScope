#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe
module purge
module load cuda/12.8
module load cudnn/9.10.1.4
module load nvidia/25.3
module load nv-hpcx/25.3
source /work/gj26/b20048/miniconda3/etc/profile.d/conda.sh
conda activate inference_env
export CUDA_VISIBLE_DEVICES=0
export PATH="$CONDA_PREFIX/bin:/opt/rh/gcc-toolset-14/root/usr/bin:$PATH"

export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
export TRITON_CC="$CC"
export TRITON_CXX="$CXX"
export CUDAHOSTCXX="$CXX"

export PYTHONNOUSERSITE=1
cd CodeScope/code_translation
# Qwen3-Coder 30B base model
python inference/run_vllm.py --model /work/go25/share/model/Qwen2.5_Coder_7B_grpo_0123_code/checkpoint-68 --result_save_name code_translation_eval_Qwen2.5_Coder_7B_grpo_0123_code.jsonl --log_file_name code_translation_eval_Qwen2.5_Coder_7B_grpo_0123_code.log --use_sft_prompt_template --enforce_eager True

# # Seed-Coder-8B base model
python inference/run_vllm.py --model /work/go25/share/model/Qwen2.5_Coder_7B_grpo_0123_md/checkpoint-68 --result_save_name code_translation_eval_Qwen2.5_Coder_7B_grpo_0123_md.jsonl --log_file_name code_translation_eval_Qwen2.5_Coder_7B_grpo_0123_md.log --use_sft_prompt_template --enforce_eager True