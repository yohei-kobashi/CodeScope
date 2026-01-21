#!/bin/bash
#PBS -q short-c
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe

cd CodeScope
source env_c/bin/activate
cd code_translation
# Qwen3-Coder 30B base model
python evaluator/run_multiple.py --jsonl_path result/code_translation_eval_Qwen2.5_Coder_7B_dapo_0116.jsonl --output_path result/eval_Qwen2.5_Coder_7B_dapo_0116.jsonl --request_retries 5 --max_workers 128

# # Seed-Coder-8B base model
python evaluator/run_multiple.py --jsonl_path result/code_translation_eval_Qwen2.5_Coder_7B_drgrpo_0116.jsonl --output_path result/eval_Qwen2.5_Coder_7B_drgrpo_0116.jsonl --request_retries 5 --max_workers 128

# Qwen2.5-Coder-7B-Instruct base model
python evaluator/run_multiple.py --jsonl_path result/code_translation_eval_Qwen2.5_Coder_7B_grpo_0116.jsonl --output_path result/eval_Qwen2.5_Coder_7B_grpo_0116.jsonl --request_retries 5 --max_workers 128
