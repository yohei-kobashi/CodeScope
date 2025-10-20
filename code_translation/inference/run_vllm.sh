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
cd 
# Qwen3-Coder 30B base model
python run_vllm.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct --result_save_name code_translation_eval_vllm.jsonl --log_file_name code_translation_eval_vllm.log --use_sft_prompt_template

# Seed-Coder-8B base model
python run_vllm.py --model ByteDance-Seed/Seed-Coder-8B-Instruct --result_save_name code_translation_eval_vllm.jsonl --log_file_name code_translation_eval_vllm.log --use_sft_prompt_template

# Qwen2.5-Coder-7B-Instruct base model
python run_vllm.py --model Qwen/Qwen2.5-Coder-7B-Instruct --result_save_name code_translation_eval_vllm.jsonl --log_file_name code_translation_eval_vllm.log --use_sft_prompt_template