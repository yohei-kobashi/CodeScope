import argparse
import logging
import math
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams


def _str_to_bool(value: Optional[str]) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    lowered = value.strip().lower()
    if lowered in {'true', 't', 'yes', 'y', '1'}:
        return True
    if lowered in {'false', 'f', 'no', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='codellama/CodeLlama-7b-Instruct-hf', type=str,
                        help='Model identifier or local path for vLLM.')
    parser.add_argument('--tokenizer', default=None, type=str,
                        help='Optional tokenizer identifier. Defaults to the model id.')
    parser.add_argument('--tensor_parallel_size', default=1, type=int,
                        help='Tensor parallelism for vLLM.')
    parser.add_argument('--max_model_len', default=8192, type=int,
                        help='Override maximum sequence length handled by vLLM.')
    parser.add_argument('--dtype', default=None, type=str,
                        help='Model weights dtype for vLLM (e.g., float16, bfloat16).')
    parser.add_argument('--download_dir', default=None, type=str,
                        help='Cache directory for model weights.')
    parser.add_argument('--trust_remote_code', action='store_true',
                        help='Allow execution of remote code during model loading.')
    parser.add_argument('--data_load_name', default='code_translation_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='code_translation_eval_vllm.jsonl', type=str)
    parser.add_argument('--log_file_name', default='code_translation_eval_vllm.log', type=str)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--top_k', default=50, type=int)
    parser.add_argument('--candidate_num', default=1, type=int)
    parser.add_argument('--max_new_tokens', default=2048, type=int)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Number of prompts to send per vLLM generate call.')
    parser.add_argument('--random_sample_ratio', default=1.0, type=float,
                        help='Randomly sample this ratio of tasks before inference. '
                             'Use 1.0 to evaluate all tasks.')
    parser.add_argument('--random_sample_seed', default=None, type=int,
                        help='Seed for --random_sample_ratio.')
    parser.add_argument('--use_sft_prompt_template', action='store_true',
                        help='Use chat-style SFT prompt template requiring tokenizer.apply_chat_template.')
    parser.add_argument('--enforce_eager', nargs='?', default=None, type=_str_to_bool,
                        const=True,
                        help='Optionally force vLLM to execute in eager mode. '
                             'Provide an explicit boolean value to override.')
    return parser.parse_args()


LANGUAGE_NAME_MAP: Dict[str, str] = {
    'c++': 'C++',
    'c#': 'C#',
    'java': 'Java',
    'javascript': 'JavaScript',
    'c': 'C',
    'python': 'Python',
    'php': 'PHP',
    'ruby': 'Ruby',
    'kotlin': 'Kotlin',
    'rust': 'Rust',
    'go': 'Go',
    'd': 'DLang',
    'delphi': 'Delphi',
    'perl': 'Perl'
}

shortcode_to_markdowncode = {
    'c++': 'cpp',
    'c#': 'csharp',
    'java': 'java',
    'javascript': 'javascript',
    'c': 'c',
    'python': 'py',
    'php': 'php',
    'ruby': 'ruby',
    'kotlin': 'kotlin',
    'rust': 'rust',
    'go': 'go',
    'd': 'd',
    'delphi': 'dpr',
    'perl': 'perl',
}

def canonical_language_name(name: str) -> str:
    key = (name or '').strip().lower()
    if not key:
        return ''
    # return LANGUAGE_NAME_MAP.get(key, key.capitalize())
    return shortcode_to_markdowncode.get(key, key.capitalize())


def build_prompt(example: Dict[str, Any], tokenizer, use_sft_prompt_template: bool) -> str:
    source_lang_key = (example.get('source_lang_cluster') or '').strip().lower()
    target_lang_key = (example.get('target_lang_cluster') or '').strip().lower()
    source_lang = canonical_language_name(source_lang_key)
    target_lang = canonical_language_name(target_lang_key)
    source_code = (example.get('source_code') or '').rstrip()
    target_declaration = (example.get('target_declaration') or '')

    if use_sft_prompt_template:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when use_sft_prompt_template=True")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert software engineer proficient in a wide range of programming languages."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Your task is to carefully translate the following {source_lang_key} code into {target_lang_key}.\n"
                    "The translated code MUST preserve exactly the same functionality as the original.\n\n"
                    "```{source_lang}\n{source_code}```"
                ).format(source_lang_key=source_lang_key, target_lang_key=target_lang_key, source_lang=source_lang, source_code=source_code),
            },
        ]
        chat_prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        suffix = "```{target_lang}\n".format(target_lang=target_lang)
        if suffix and not suffix.endswith("\n"):
            suffix += "\n"
        prompt = chat_prefix + suffix
    else:
        prompt = "code translation\n"
        prompt += f"{source_lang}:\n"
        prompt += source_code + "\n"
        prompt += f"{target_lang}:\n"
        prompt += target_declaration
    return prompt


def count_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        return 0
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def _normalize_dtype_name(dtype_value: Any) -> Optional[str]:
    if dtype_value is None:
        return None
    if isinstance(dtype_value, str):
        dtype_str = dtype_value
    else:
        dtype_str = str(dtype_value)
    dtype_str = dtype_str.replace('torch.', '').replace('torch', '').strip().lower()
    if dtype_str == 'float':
        dtype_str = 'float32'
    return dtype_str or None


def infer_model_dtype(model_id: str, dtype_arg: Optional[str], trust_remote_code: bool,
                      cache_dir: Optional[str]) -> Optional[str]:
    if dtype_arg and dtype_arg.lower() != 'auto':
        return dtype_arg

    try:
        from transformers import AutoConfig  # type: ignore
    except Exception:
        logging.warning('transformers not available; defaulting dtype to float16.')
        return 'float16'

    try:
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        logging.warning('Failed to load model config for dtype detection (%s); defaulting to float16.', exc)
        return 'float16'

    config_dtype = getattr(config, 'torch_dtype', None) or getattr(config, 'dtype', None)
    normalized = _normalize_dtype_name(config_dtype)
    if not normalized:
        logging.warning('Model config does not specify torch_dtype; defaulting to float16.')
        return 'float16'

    logging.info('Auto-detected dtype=%s from model config.', normalized)
    return normalized


def batched(iterable: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start_idx in range(0, len(iterable), batch_size):
        yield iterable[start_idx:start_idx + batch_size]


def sample_records(records: List[Dict[str, Any]], sample_ratio: float,
                   seed: Optional[int]) -> List[Dict[str, Any]]:
    if sample_ratio <= 0.0 or sample_ratio > 1.0:
        raise ValueError('--random_sample_ratio must be in the range (0.0, 1.0].')
    if sample_ratio >= 1.0 or not records:
        logging.info('Using all %d records.', len(records))
        return records

    sample_size = max(1, math.ceil(len(records) * sample_ratio))
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(records)), sample_size))
    logging.info('Randomly sampled %d/%d records with ratio %.6f and seed %s.',
                 sample_size, len(records), sample_ratio, seed)
    return [records[idx] for idx in sampled_indices]


def output_token_count(tokenizer, sequence: Any, text: str) -> int:
    token_ids = getattr(sequence, 'token_ids', None)
    if token_ids is not None:
        return len(token_ids)
    return count_tokens(tokenizer, text)


def reached_max_tokens(sequence: Any, token_count: int, max_tokens: int) -> bool:
    finish_reason = (getattr(sequence, 'finish_reason', None) or '').lower()
    if finish_reason == 'length':
        return True
    return token_count >= max_tokens


def main() -> None:
    load_path = Path(__file__).parent.parent.parent / Path('data') / Path(args.data_load_name)
    save_path = Path(__file__).parent.parent / Path('result') / Path(args.result_save_name)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files()
    records: List[Dict[str, Any]] = dataset.to_list()
    records = sample_records(records, args.random_sample_ratio, args.random_sample_seed)

    resolved_dtype = infer_model_dtype(
        args.model,
        args.dtype,
        args.trust_remote_code,
        args.download_dir,
    )
    args.dtype = resolved_dtype

    llm_kwargs: Dict[str, Any] = {
        'model': args.model,
        'tokenizer': args.tokenizer or args.model,
        'tensor_parallel_size': args.tensor_parallel_size,
        'trust_remote_code': args.trust_remote_code,
        'dtype': args.dtype,
        'max_model_len': args.max_model_len,
        'download_dir': args.download_dir,
    }
    if args.enforce_eager is not None:
        llm_kwargs['enforce_eager'] = args.enforce_eager

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    if tokenizer and tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts: List[str] = []
    for example in records:
        prompt = build_prompt(example, tokenizer, args.use_sft_prompt_template)
        prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k >= 0 else -1,
        max_tokens=args.max_new_tokens,
        n=args.candidate_num,
    )

    all_generations: List[List[str]] = [[] for _ in records]
    all_generation_infos: List[List[Dict[str, Any]]] = [[] for _ in records]
    total_generations = 0
    max_token_generations = 0
    request_indices = list(range(len(prompts)))
    for batch_indices in batched(request_indices, args.batch_size):
        batch_prompts = [prompts[i] for i in batch_indices]
        batch_records = [records[i] for i in batch_indices]

        for record, prompt in zip(batch_records, batch_prompts):
            tokens = count_tokens(tokenizer, prompt)
            logging.info('problem src_id: %s', record.get('src_uid'))
            logging.info('input tokens: %d', tokens)

        batch_outputs = llm.generate(batch_prompts, sampling_params)

        for req_idx, request_output in zip(batch_indices, batch_outputs):
            sequences = sorted(request_output.outputs, key=lambda item: item.index)
            generation_texts = [seq.text.strip() for seq in sequences]
            generation_infos = []
            for seq, generated in zip(sequences, generation_texts):
                output_tokens = output_token_count(tokenizer, seq, generated)
                finish_reason = getattr(seq, 'finish_reason', None)
                hit_max_tokens = reached_max_tokens(seq, output_tokens, args.max_new_tokens)
                total_generations += 1
                if hit_max_tokens:
                    max_token_generations += 1
                generation_infos.append({
                    'finish_reason': finish_reason,
                    'output_tokens': output_tokens,
                    'reached_max_tokens': hit_max_tokens,
                })
            all_generations[req_idx] = generation_texts
            all_generation_infos[req_idx] = generation_infos
            logging.info('response: %s', generation_texts)

    for example, generations, generation_infos in zip(records, all_generations, all_generation_infos):
        for idx, generated in enumerate(generations):
            info = generation_infos[idx] if idx < len(generation_infos) else {}
            output_tokens = info.get('output_tokens')
            if output_tokens is None:
                output_tokens = count_tokens(tokenizer, generated)
            logging.info('output tokens: %d', output_tokens)
            if info.get('reached_max_tokens'):
                logging.warning('Reached max_new_tokens %s src_uid: %s lang: %s candidate: %d finish_reason: %s',
                                args.max_new_tokens, example.get('src_uid'),
                                example.get('target_lang_cluster'), idx, info.get('finish_reason'))
            if output_tokens > args.max_new_tokens:
                logging.warning('Over total tokens limit %s lang: %s', example.get('src_uid'),
                                example.get('target_lang_cluster'))
                generated = ''
            example[f'code_translation_{idx}'] = generated
            example[f'code_translation_{idx}_finish_reason'] = info.get('finish_reason')
            example[f'code_translation_{idx}_output_tokens'] = output_tokens
            example[f'code_translation_{idx}_reached_max_tokens'] = bool(info.get('reached_max_tokens'))
        for pad_idx in range(len(generations), args.candidate_num):
            example[f'code_translation_{pad_idx}'] = ''
            example[f'code_translation_{pad_idx}_finish_reason'] = None
            example[f'code_translation_{pad_idx}_output_tokens'] = 0
            example[f'code_translation_{pad_idx}_reached_max_tokens'] = False

    max_token_rate = max_token_generations / total_generations if total_generations else 0.0
    logging.info('max_new_tokens reached: %d/%d (%.4f)',
                 max_token_generations, total_generations, max_token_rate)
    print('max_new_tokens reached:', max_token_generations, '/', total_generations,
          f'({max_token_rate:.4f})')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_with_outputs = Dataset.from_list(records)
    dataset_with_outputs.to_json(save_path, lines=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_arguments()

    log_file_path = Path(__file__).parent.parent / Path('log') / Path(args.log_file_name)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    main()
