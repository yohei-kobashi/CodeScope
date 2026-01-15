#!/bin/bash
#PBS -q short-c
#PBS -l select=1
#PBS -W group_list=go25
#PBS -j oe

cd CodeScope
source env_c/bin/activate
cd code_translation
# Qwen3-Coder 30B base model
python evaluator/run_multiple.py --jsonl_path result/code_translation_eval_Qwen3.jsonl --output_path result/eval_Qwen3.jsonl --request_retries 5 --max_workers 128

# # Seed-Coder-8B base model
python evaluator/run_multiple.py --jsonl_path result/code_translation_eval_Seed-Coder.jsonl --output_path result/eval_Seed-Coder.jsonl --request_retries 5 --max_workers 128

# Qwen2.5-Coder-7B-Instruct base model
python evaluator/run_multiple.py --jsonl_path result/code_translation_eval_Qwen2.5.jsonl --output_path result/eval_Qwen2.5.jsonl --request_retries 5 --max_workers 128
