"""
Usage:
1. Prepare a Lambda endpoint mapping such as `lang2url.json` and pass it via `--lang_url_config`.
2. Provide the JSONL file with the code you want to evaluate through `--jsonl_path`, and choose where to write the results with `--output_path`.
3. Provide generated programs in each JSON record as `code_translation_<index>` (multiple variants per record are supported and are evaluated independently) and set the desired runtime via `target_lang_cluster`.
4. If `target_lang_cluster` is missing or you need to override for the entire file, use `--language`.
5. Cold starts can make the first request fail, so this script pings each endpoint via `/healthz` before running the evaluations. Tune the check with `--healthcheck_*` flags if needed.
6. Example:
   `python CodeScope/code_translation/evaluator/run_multiple.py --jsonl_path data/sample.jsonl --output_path results/executed_result.json --lang_url_config config/lang2url.json`
"""

import argparse
import json
import ast
import time
import math
import re
import random
import threading
import concurrent.futures
from contextlib import ExitStack, contextmanager
from collections import defaultdict
from typing import MutableSet, Sequence, Tuple, Dict, List
from pathlib import Path
from urllib.parse import urljoin

import func_timeout
import requests
# from func_timeout import func_set_timeout

LANGUAGE_KEY_MAP = {
    "d": "d",
    "dlang": "d",
    "delphi": "delphi",
    "c": "c",
    "go": "go_test.go",
    "go_test.go": "go_test.go",
    "kotlin": "kotlin",
    "kt": "kotlin",
    "javascript": "javascript",
    "js": "javascript",
    "ruby": "ruby",
    "rb": "ruby",
    "c#": "cs",
    "csharp": "cs",
    "cs": "cs",
    "python": "python",
    "py": "python",
    "php": "php",
    "java": "java",
    "rust": "rust",
    "rs": "rust",
    "c++": "cpp",
    "cpp": "cpp",
    "perl": "pl",
    "pl": "pl",
}

LANGUAGE_URL_FALLBACKS = {}

LANG_URLS = {}
TOTAL_INFLIGHT_LIMITER: threading.BoundedSemaphore | None = None
LANG_INFLIGHT_LIMITERS: dict[str, threading.BoundedSemaphore] = {}
LANG_LIMITER_LOCK = threading.Lock()


def load_lang_urls(path: str):
    global LANG_URLS
    config_path = Path(path)
    if not config_path.is_file():
        config_path = Path(__file__).resolve().parent / path
    with config_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    urls = data.get("urls") if isinstance(data, dict) else None
    if not isinstance(urls, dict):
        raise ValueError(f"Invalid lang2url config format in {config_path}")
    LANG_URLS = {str(k): str(v) for k, v in urls.items()}


def normalize_language_key(lang: str) -> str:
    if not lang:
        return ""
    normalized = lang.lower()
    return LANGUAGE_KEY_MAP.get(normalized, normalized)


def resolve_language_url(lang_key: str) -> str:
    if lang_key in LANG_URLS:
        return LANG_URLS[lang_key]

    for candidate in LANGUAGE_URL_FALLBACKS.get(lang_key, []):
        if candidate in LANG_URLS:
            return LANG_URLS[candidate]

    raise KeyError(f"No lambda URL configured for language '{lang_key}'")


@contextmanager
def _semaphore_guard(semaphore: threading.BoundedSemaphore | None):
    if semaphore is None:
        yield
        return
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def get_lang_inflight_limiter(lang_key: str) -> threading.BoundedSemaphore | None:
    max_per_lang = getattr(args, "max_inflight_per_lang", 0)
    if not max_per_lang or max_per_lang <= 0:
        return None
    with LANG_LIMITER_LOCK:
        limiter = LANG_INFLIGHT_LIMITERS.get(lang_key)
        if limiter is None:
            limiter = threading.BoundedSemaphore(max_per_lang)
            LANG_INFLIGHT_LIMITERS[lang_key] = limiter
        return limiter


def check_language_server_health(lang_key: str, attempts: int = 5, interval: float = 10.0,
                                 timeout: float = 5.0) -> bool:
    base_url = resolve_language_url(lang_key)
    health_url = urljoin(base_url if base_url.endswith("/") else base_url + "/", "healthz")
    for idx in range(attempts):
        print(f"[HealthCheck] {lang_key}: attempt {idx + 1}/{attempts} -> {health_url}")
        try:
            resp = requests.get(health_url, timeout=timeout)
            if resp.status_code == requests.codes.ok:
                print(f"[HealthCheck] {lang_key}: server is ready")
                return True
            print(f"[HealthCheck] {lang_key}: unexpected status {resp.status_code}")
        except requests.RequestException as exc:
            print(f"[HealthCheck] {lang_key}: request failed ({exc})")
        if idx < attempts - 1:
            time.sleep(interval)
    print(f"[HealthCheck] {lang_key}: failed after {attempts} attempts, continuing anyway")
    return False


def ensure_language_health_checked(lang: str, prepared_languages: MutableSet[str],
                                   attempts: int, interval: float, timeout: float):
    normalized = normalize_language_key(lang)
    if not normalized or normalized in prepared_languages:
        return
    try:
        check_language_server_health(normalized, attempts, interval, timeout)
    except KeyError as exc:
        print(f"[HealthCheck] Skipping for '{lang}': {exc}")
    prepared_languages.add(normalized)


def pass_at_k(total: int, correct: int, k: int) -> float:
    if total < k or correct == 0:
        return 0.0
    if total - correct < k:
        return 1.0
    return 1.0 - math.comb(total - correct, k) / math.comb(total, k)


def summarize_pass_metrics(group_results: Dict[Tuple[str, str], List[bool]]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for (src_uid, lang), outcomes in group_results.items():
        total = len(outcomes)
        correct = sum(1 for result in outcomes if result)
        metrics = {}
        if total >= 1:
            metrics["pass@1"] = pass_at_k(total, correct, 1)
        if total >= 5:
            metrics["pass@5"] = pass_at_k(total, correct, 5)
        if total >= 10:
            metrics["pass@10"] = pass_at_k(total, correct, 10)
        if not metrics:
            continue
        key = f"{src_uid}_{lang}"
        summary[key] = {
            "src_uid": src_uid,
            "language": lang,
            "total": total,
            "correct": correct,
            **metrics,
        }
    return summary


def extract_translations(content: dict) -> Sequence[Tuple[str, str]]:
    translations: List[Tuple[str, str]] = []
    prefix = "code_translation_"
    for key, value in content.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if not suffix.isdigit():
            continue
        if isinstance(value, str):
            translations.append((key, value))
    translations.sort(key=lambda item: int(item[0].split("_")[-1]))
    return translations


def normalize_for_compare(text: str | None) -> str:
    if text is None:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace(" ", "").lower().strip()
    return normalized


def strip_code_block_wrappers(source_code: str) -> str:
    """Remove markdown-style code fences and surrounding text."""
    if not source_code:
        return ""
    stripped = re.sub(r"^.*?```[^\n]*\n", "", source_code, flags=re.DOTALL)
    stripped = re.sub(r"```\n.*$", "", stripped, flags=re.DOTALL)
    stripped = stripped.replace("```", "")
    return stripped.strip()


def preprocess_source_code(raw_code: str | None) -> str:
    """Normalize source text without destroying literal escape sequences."""
    if not raw_code:
        return ""

    code = raw_code

    # Drop any BOM that sneaks into the start of the snippet (common with C#/Java files).
    if code.startswith("\ufeff"):
        code = code.lstrip("\ufeff")

    # Some datasets keep newline characters double-escaped (\\n) for the entire blob.
    # Only decode those placeholders when the text does not already contain real line feeds,
    # so legitimate escape sequences such as '\n' or '\\n' survive untouched.
    if "\n" not in code and "\\n" in code:
        code = code.replace("\\r\\n", "\n")
        code = code.replace("\\n", "\n")

    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = strip_code_block_wrappers(code)
    return code


def iter_jsonl_entries(jsonl_path: str, sample_size: int = 0, seed: int | None = None):
    """Yield (line_idx, line_text) pairs, optionally limited to a random subset."""
    if sample_size and sample_size > 0:
        rng = random.Random(seed)
        reservoir = []
        total_lines = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                total_lines = line_idx + 1
                if len(reservoir) < sample_size:
                    reservoir.append((line_idx, line))
                    continue
                swap_idx = rng.randint(0, line_idx)
                if swap_idx < sample_size:
                    reservoir[swap_idx] = (line_idx, line)
        if not reservoir:
            return
        reservoir.sort(key=lambda item: item[0])
        print(f"[RandomSample] Processing {len(reservoir)} randomly selected entries "
              f"(requested {sample_size}) out of {total_lines} total lines")
        for item in reservoir:
            yield item
        return

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            yield line_idx, line


def invoke_lambda_executor(lang_key: str, source_code: str, input_data: str, expected_output: list,
                           submission_id: str, verbose: bool = False, debug_info: dict | None = None):
    if debug_info is None:
        debug_info = {}
    base_url = resolve_language_url(lang_key)
    eval_url = urljoin(base_url if base_url.endswith("/") else base_url + "/", "evaluate")
    max_attempts = max(1, getattr(args, "request_retries", 0) + 1)
    base_retry_delay = max(0.0, getattr(args, "request_retry_delay", 1.0))
    max_retry_delay = max(base_retry_delay, getattr(args, "request_retry_max_delay", base_retry_delay))
    retry_statuses = {429, 502, 503, 504}
    request_timeout = float(getattr(args, "request_timeout", 60))
    connect_timeout = getattr(args, "request_connect_timeout", None)
    read_timeout = getattr(args, "request_read_timeout", None)
    if connect_timeout is None:
        connect_timeout = min(10.0, request_timeout)
    if read_timeout is None:
        read_timeout = request_timeout
    payload = {
        "language": lang_key,
        "source_code": source_code,
        "input": input_data,
        "output": expected_output,
        "name": f"{submission_id}_{lang_key}",
        "eval_timeout": 800,
    }

    debug_info["eval_url"] = eval_url
    if verbose:
        debug_info["payload"] = payload

    resp = None
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with ExitStack() as stack:
                stack.enter_context(_semaphore_guard(TOTAL_INFLIGHT_LIMITER))
                stack.enter_context(_semaphore_guard(get_lang_inflight_limiter(lang_key)))
                resp = requests.post(eval_url, json=payload, timeout=(connect_timeout, read_timeout))
            debug_info.setdefault("attempts", []).append(
                {"attempt": attempt, "response_status": resp.status_code}
            )
            should_retry = resp.status_code >= 500 or resp.status_code in retry_statuses
            if should_retry and attempt < max_attempts:
                retry_after = 0.0
                if resp.status_code == 429:
                    retry_after_header = resp.headers.get("Retry-After")
                    if retry_after_header:
                        try:
                            retry_after = float(retry_after_header)
                        except ValueError:
                            retry_after = 0.0
                backoff = min(max_retry_delay, base_retry_delay * (2 ** (attempt - 1)))
                jitter = random.uniform(0.0, min(1.0, backoff))
                delay = max(backoff + jitter, retry_after)
                print(f"[Retry] HTTP {resp.status_code} from {eval_url} (attempt {attempt}/{max_attempts}); "
                      f"retrying in {delay:.2f}s")
                time.sleep(delay)
                continue
            break
        except requests.RequestException as exc:
            last_exc = exc
            debug_info.setdefault("attempts", []).append({"attempt": attempt, "error": str(exc)})
            if attempt >= max_attempts:
                raise
            backoff = min(max_retry_delay, base_retry_delay * (2 ** (attempt - 1)))
            jitter = random.uniform(0.0, min(1.0, backoff))
            delay = backoff + jitter
            print(f"[Retry] Request failed ({exc}) (attempt {attempt}/{max_attempts}); "
                  f"retrying in {delay:.2f}s")
            time.sleep(delay)

    if resp is None:
        # Should only happen if the retry loop never ran; keep a clear error for safety.
        raise RuntimeError(f"Failed to invoke evaluator at {eval_url}: {last_exc or 'Unknown error'}")

    debug_info["response_status"] = resp.status_code
    data = None
    parse_error = None
    try:
        data = resp.json()
    except ValueError:
        parse_error = resp.text
    if verbose:
        if data is not None:
            debug_info["response_body"] = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            debug_info["response_body"] = parse_error or "<empty response>"

    result_entry = (data.get("results") or [{}])[0] if isinstance(data, dict) else {}
    expected_from_result = result_entry.get("expected_output")
    if not expected_from_result:
        expected_from_result = expected_output

    return {
        "http_status": resp.status_code,
        "status": result_entry.get("status"),
        "stdout": result_entry.get("stdout"),
        "stderr": result_entry.get("stderr") if isinstance(result_entry, dict) else parse_error,
        "matched": result_entry.get("matched"),
        "program": result_entry.get("program"),
        "expected_output": expected_from_result,
        "raw_error": parse_error if parse_error and not result_entry else None,
    }, debug_info


def status_to_errtype(status: str | None) -> str:
    if not status:
        return "RUNTIME_ERROR"
    upper_status = status.upper()
    if "SYNTAX" in upper_status or "COMPILE" in upper_status:
        return "COMPILATION_ERROR"
    if "TIME" in upper_status:
        return "TIMEOUT"
    if "RUNTIME" in upper_status:
        return "RUNTIME_ERROR"
    return upper_status

def record_result(output_dict, src_uid, submission_id, difficulty, id, answer, output, outerr, errtype=None):
    output_dict[submission_id] = {}
    output_dict[submission_id]["src_uid"] = src_uid
    output_dict[submission_id]["submission_id"] = submission_id
    if difficulty:
        output_dict[submission_id]["difficulty"] = difficulty
    if id:
        output_dict[submission_id]["id"] = id
    if answer:
        output_dict[submission_id]["answer"] = answer
    if output:
        output_dict[submission_id]["output"] = output
    if outerr:
        output_dict[submission_id]["error"] = outerr
    if errtype:
        output_dict[submission_id]["errtype"] = errtype
    return output_dict


def exe_testcase(source_code, answer, input_data, lang, output_dict, wrong_case, src_uid,
                 submission_id, difficulty, id, ):
    err = 0
    errtype = None
    outerr = None
    output_value = None
    invalid_case = 0
    debug_info: dict = {}
    verbose_enabled = getattr(args, "verbose", False)

    normalized_lang = normalize_language_key(lang)
    if not normalized_lang:
        err = 1
        errtype = "UNSUPPORTED_LANGUAGE"
        outerr = f"Unable to normalize language key from '{lang}'"
        output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                             None, outerr, errtype)
        wrong_case += 1
        return output_dict, wrong_case, err

    def evaluate_with_lang(lang_key: str):
        local_debug: dict = {}
        local_err = 0
        local_errtype = None
        local_outerr = None
        local_output_value = None
        lambda_error = False

        try:
            response, local_debug = invoke_lambda_executor(
                lang_key,
                source_code,
                input_data,
                answer,
                submission_id,
                verbose=verbose_enabled,
                debug_info=local_debug,
            )
        except KeyError as exc:
            local_err = 1
            local_errtype = "UNSUPPORTED_LANGUAGE"
            local_outerr = str(exc)
        except requests.RequestException as exc:
            local_err = 1
            local_errtype = "LAMBDA_ERROR"
            local_outerr = str(exc)
            lambda_error = True
        except Exception as exc:
            local_err = 1
            local_errtype = "RUNTIME_ERROR"
            local_outerr = str(exc)
        else:
            http_status = response.get("http_status")
            status = response.get("status")
            local_output_value = response.get("stdout") or ""
            stderr_value = response.get("stderr") or response.get("raw_error")
            expected_outputs = response.get("expected_output") or ([answer] if answer else [])
            raw_error = response.get("raw_error")

            normalized_stdout = normalize_for_compare(local_output_value)
            normalized_expecteds = [normalize_for_compare(item) for item in expected_outputs if item is not None]
            matched = response.get("matched")
            normalized_match = bool(normalized_expecteds) and any(
                normalized_stdout == candidate for candidate in normalized_expecteds
            )
            if matched is None:
                matched = normalized_match
            elif matched is False and normalized_match:
                matched = True

            if raw_error:
                local_err = 1
                local_errtype = "LAMBDA_ERROR"
                local_outerr = f"Malformed response: {raw_error}"
                lambda_error = True
            elif http_status != 200:
                local_err = 1
                if http_status >= 500 or http_status in {429, 502, 503, 504}:
                    local_errtype = "LAMBDA_ERROR"
                    lambda_error = True
                else:
                    local_errtype = status_to_errtype(status or "HTTP_ERROR")
                local_outerr = f"HTTP {http_status}: {stderr_value or 'No response body'}"
            elif status is not None and status.upper() != "OK":
                local_err = 1
                local_errtype = status_to_errtype(status)
                local_outerr = stderr_value
            elif matched is False:
                local_err = 1
                local_errtype = "WRONG_ANSWER"
                local_outerr = stderr_value

        if local_err != 0 and local_outerr is None:
            local_outerr = "Unknown error"

        return {
            "err": local_err,
            "errtype": local_errtype,
            "outerr": local_outerr,
            "output_value": local_output_value,
            "debug_info": local_debug,
            "lambda_error": lambda_error,
        }

    result = None

    if normalized_lang == "python":
        # Prefer python2 executor, fall back to python3 if it fails or mismatches.
        python_attempts: list[dict] = []
        py2_result = evaluate_with_lang("python2")
        python_attempts.append({"lang": "python2", "debug": py2_result.get("debug_info")})

        if py2_result["err"] == 0:
            result = py2_result
        else:
            py3_result = evaluate_with_lang("python3")
            python_attempts.append({"lang": "python3", "debug": py3_result.get("debug_info")})
            if py3_result["err"] == 0:
                result = py3_result
            else:
                combined_error_parts = []
                if py2_result.get("outerr"):
                    combined_error_parts.append(f"python2: {py2_result['outerr']}")
                if py3_result.get("outerr"):
                    combined_error_parts.append(f"python3: {py3_result['outerr']}")
                combined_outerr = " | ".join(combined_error_parts) or "python2/python3 evaluation failed"
                result = {
                    "err": 1,
                    "errtype": py3_result.get("errtype") or py2_result.get("errtype"),
                    "outerr": combined_outerr,
                    "output_value": py3_result.get("output_value") or py2_result.get("output_value"),
                    "debug_info": py3_result.get("debug_info"),
                }

        if result is None:
            result = py2_result

        debug_info = result.get("debug_info") if isinstance(result.get("debug_info"), dict) else {}
        debug_info.setdefault("python_attempts", python_attempts)
        result["debug_info"] = debug_info
    else:
        result = evaluate_with_lang(normalized_lang)

    err = result.get("err", 1)
    errtype = result.get("errtype")
    outerr = result.get("outerr")
    output_value = result.get("output_value")
    debug_info = result.get("debug_info") if isinstance(result.get("debug_info"), dict) else result.get("debug_info")

    if err == 0:
        return output_dict, wrong_case, err, invalid_case

    if verbose_enabled:
        eval_url = debug_info.get("eval_url")
        if eval_url:
            print(f"[Verbose] POST {eval_url}")
        payload = debug_info.get("payload")
        if payload is not None:
            print("[Verbose] Payload:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        response_status = debug_info.get("response_status")
        if response_status is not None:
            print(f"[Verbose] Response status: {response_status}")
        response_body = debug_info.get("response_body")
        if response_body is not None:
            print("[Verbose] Response body:")
            print(response_body)

    if errtype == "LAMBDA_ERROR":
        invalid_case = 1
        output_dict["invalid"] = record_result(
            output_dict["invalid"],
            src_uid,
            submission_id,
            difficulty,
            id,
            None,
            None,
            outerr,
            errtype,
        )
    elif errtype == "WRONG_ANSWER":
        print("-----------------answer: ", answer, "-------------------")
        print("-----------------output: ", output_value, "-------------------")
        print("WRONG_ANSWER in src_uid: ", src_uid)
        try:
            output_dict["wrong"] = record_result(
                output_dict["wrong"],
                src_uid,
                submission_id,
                difficulty,
                id,
                answer,
                output_value,
                outerr,
                errtype,
            )
        except func_timeout.exceptions.FunctionTimedOut:
            print("Time Limit Exceeded while recording wrong answer")
            output_dict["error"] = record_result(
                output_dict["error"],
                src_uid,
                submission_id,
                difficulty,
                id,
                None,
                None,
                outerr,
                "TIMEOUT",
            )
    else:
        if outerr is None:
            outerr = "Unknown error"
        output_dict["error"] = record_result(
            output_dict["error"],
            src_uid,
            submission_id,
            difficulty,
            id,
            None,
            None,
            outerr,
            errtype or "RUNTIME_ERROR",
        )

    wrong_case += 1
    return output_dict, wrong_case, err, invalid_case


def exe_question(content, lang, output_dict, source_code: str, translation_label: str):
    source_code = preprocess_source_code(source_code)

    id = content.get("id")
    src_uid = str(content["src_uid"])
    difficulty = str(content["difficulty"])
    testcases = content["testcases"]
    if isinstance(content['testcases'], str):
        testcases = ast.literal_eval(testcases)
    if "code_uid" in content:
        submission_id = str(content["code_uid"])
    elif "submission_id" in content:
        submission_id = str(content["submission_id"])
    else:
        submission_id = src_uid
    if translation_label:
        submission_id = f"{submission_id}_{translation_label}"

    if source_code == "":
        print(f"No source code detected for {translation_label or 'entry'}")
        output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None, None,
                                             "No Source Code", "No_Source_Code")
        return output_dict, 1, 0

    wrong_case = 0
    err = 0
    for testcase in testcases:
        raw_input = testcase.get("input", "")
        raw_output = testcase.get("output", "")

        if isinstance(raw_input, list):
            input_data = "\n".join(str(item) for item in raw_input)
        else:
            input_data = str(raw_input)

        if isinstance(raw_output, list):
            answer = raw_output
        else:
            answer = [str(raw_output)]

        input_data = input_data.replace("\r\n", "\n").replace("\r", "")

        invalid_case = 0
        try:
            output_dict, wrong_case, err, invalid_case = exe_testcase(
                source_code,
                answer,
                input_data,
                lang,
                output_dict,
                wrong_case,
                src_uid,
                submission_id,
                difficulty,
                id,
            )
        except func_timeout.exceptions.FunctionTimedOut:
            err, wrong_case = 1, 1
            print("Time Limit Exceeded")
            output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                                 None, "Time Limit Exceeded", "RUNTIME_ERROR")

        if invalid_case:
            return output_dict, 1, 1

        if err == 1:
            wrong_case = 1
            break
    if err == 0:
        output_dict["accepted"] = record_result(output_dict["accepted"], src_uid, submission_id, difficulty, id, None,
                                                None, None, None)

    return output_dict, wrong_case, 0


def exe_main():
    global TOTAL_INFLIGHT_LIMITER

    try:
        load_lang_urls(args.lang_url_config)
    except Exception as exc:
        raise RuntimeError(f"Failed to load language URL config: {exc}") from exc

    jsonl_path = args.jsonl_path
    if getattr(args, "max_inflight_total", 0) and args.max_inflight_total > 0:
        TOTAL_INFLIGHT_LIMITER = threading.BoundedSemaphore(args.max_inflight_total)
    if args.language:
        lang_hint = args.language
    else:
        lang_hint = jsonl_path.split(".")[0].split("_")[-1]

    code_sum, correct_sum = 0, 0
    output_dict = {"accepted": {}, "wrong": {}, "error": {}, "invalid": {}}
    prepared_languages: MutableSet[str] = set()
    group_results: Dict[Tuple[str, str], List[bool]] = defaultdict(list)
    per_language_totals: Dict[str, dict] = defaultdict(lambda: {
        "code_sum": 0,
        "correct_sum": 0,
        "wrong_num": 0,
        "error_num": 0,
        "invalid_num": 0,
    })
    per_lang_pair_totals: Dict[Tuple[str, str], dict] = defaultdict(lambda: {
        "code_sum": 0,
        "correct_sum": 0,
        "wrong_num": 0,
        "error_num": 0,
        "invalid_num": 0,
    })
    prepared_lock = threading.Lock()

    def ensure_language_prepared(lang: str):
        normalized = normalize_language_key(lang)
        if not normalized:
            return
        with prepared_lock:
            if normalized in prepared_languages:
                return
            prepared_languages.add(normalized)
        try:
            check_language_server_health(
                normalized,
                args.healthcheck_attempts,
                args.healthcheck_interval,
                args.healthcheck_timeout,
            )
        except KeyError as exc:
            print(f"[HealthCheck] Skipping for '{lang}': {exc}")

    def process_line(line_idx: int, line: str):
        local_output = {"accepted": {}, "wrong": {}, "error": {}, "invalid": {}}
        local_group_results: Dict[Tuple[str, str], List[bool]] = defaultdict(list)
        local_per_language_totals: Dict[str, dict] = defaultdict(lambda: {
            "code_sum": 0,
            "correct_sum": 0,
            "wrong_num": 0,
            "error_num": 0,
            "invalid_num": 0,
        })
        local_per_lang_pair_totals: Dict[Tuple[str, str], dict] = defaultdict(lambda: {
            "code_sum": 0,
            "correct_sum": 0,
            "wrong_num": 0,
            "error_num": 0,
            "invalid_num": 0,
        })
        local_code_sum = 0
        local_correct_sum = 0
        local_invalid_sum = 0

        content = json.loads(line)
        translations = list(extract_translations(content))
        if not translations and "source_code" in content:
            translations = [("source_code", content["source_code"])]
        if not translations:
            print(f"No code translations found in line {line_idx + 1}, skipping")
            return {
                "code_sum": 0,
                "correct_sum": 0,
                "output": local_output,
                "group_results": local_group_results,
                "per_language_totals": local_per_language_totals,
            }

        entry_lang = content.get("target_lang_cluster") or content.get("language") or lang_hint
        source_lang = content.get("source_lang_cluster") or "unknown"
        ensure_language_prepared(entry_lang)
        src_uid = str(content["src_uid"])

        for translation_label, source_code in translations:
            prev_wrong = len(local_output["wrong"])
            prev_error = len(local_output["error"])
            try:
                local_output, wrong_case, invalid_case = exe_question(
                    content,
                    entry_lang,
                    local_output,
                    source_code,
                    translation_label,
                )
            except func_timeout.exceptions.FunctionTimedOut:
                print("Time Limit Exceeded")
                wrong_case = 1
                invalid_case = 0

            lang_totals = local_per_language_totals[entry_lang]
            pair_key = (source_lang, entry_lang)
            pair_totals = local_per_lang_pair_totals[pair_key]
            if invalid_case:
                local_invalid_sum += 1
                lang_totals["invalid_num"] += 1
                pair_totals["invalid_num"] += 1
            else:
                local_code_sum += 1
                lang_totals["code_sum"] += 1
                pair_totals["code_sum"] += 1
                success = wrong_case == 0
                if success:
                    local_correct_sum += 1
                    lang_totals["correct_sum"] += 1
                    pair_totals["correct_sum"] += 1
                local_group_results[(src_uid, entry_lang)].append(success)
                if not success:
                    new_wrong = len(local_output["wrong"]) - prev_wrong
                    new_error = len(local_output["error"]) - prev_error
                    if new_wrong > 0:
                        lang_totals["wrong_num"] += 1
                        pair_totals["wrong_num"] += 1
                    elif new_error > 0:
                        lang_totals["error_num"] += 1
                        pair_totals["error_num"] += 1
                    else:
                        lang_totals["error_num"] += 1
                        pair_totals["error_num"] += 1

        return {
            "code_sum": local_code_sum,
            "correct_sum": local_correct_sum,
            "invalid_sum": local_invalid_sum,
            "output": local_output,
            "group_results": local_group_results,
            "per_language_totals": local_per_language_totals,
            "per_lang_pair_totals": local_per_lang_pair_totals,
        }

    processed_any = False
    futures = []
    total_entries = 0
    processed_entries = 0
    invalid_sum = 0
    aborted = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for line_idx, line in iter_jsonl_entries(jsonl_path, args.random_sample_size, args.random_sample_seed):
            processed_any = True
            futures.append(executor.submit(process_line, line_idx, line))

        total_entries = len(futures)
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            code_sum += result["code_sum"]
            correct_sum += result["correct_sum"]
            invalid_sum += result["invalid_sum"]
            processed_entries += 1

            for key in ("accepted", "wrong", "error", "invalid"):
                output_dict[key].update(result["output"][key])

            for (src_uid, lang), outcomes in result["group_results"].items():
                group_results[(src_uid, lang)].extend(outcomes)

            for lang, stats in result["per_language_totals"].items():
                lang_totals = per_language_totals[lang]
                lang_totals["code_sum"] += stats["code_sum"]
                lang_totals["correct_sum"] += stats["correct_sum"]
                lang_totals["wrong_num"] += stats["wrong_num"]
                lang_totals["error_num"] += stats["error_num"]
                lang_totals["invalid_num"] += stats["invalid_num"]
            for pair_key, stats in result["per_lang_pair_totals"].items():
                pair_totals = per_lang_pair_totals[pair_key]
                pair_totals["code_sum"] += stats["code_sum"]
                pair_totals["correct_sum"] += stats["correct_sum"]
                pair_totals["wrong_num"] += stats["wrong_num"]
                pair_totals["error_num"] += stats["error_num"]
                pair_totals["invalid_num"] += stats["invalid_num"]

            print(f"[Progress] entries {processed_entries}/{total_entries} "
                  f"| done: {code_sum} not accepted: {code_sum - correct_sum}")

            total_attempts = code_sum + invalid_sum
            max_invalid_rate = getattr(args, "max_lambda_error_rate", None)
            min_samples = getattr(args, "lambda_error_min_samples", 0)
            if (max_invalid_rate is not None and max_invalid_rate >= 0
                    and total_attempts >= min_samples and total_attempts > 0):
                invalid_rate = invalid_sum / total_attempts
                if invalid_rate > max_invalid_rate:
                    print(f"[Abort] Lambda error rate {invalid_rate:.3f} exceeded "
                          f"threshold {max_invalid_rate:.3f} after {total_attempts} attempts")
                    aborted = True
                    executor.shutdown(cancel_futures=True)
                    break

    if not processed_any:
        print(f"No entries found in {jsonl_path}")
        return

    wrong_num = len(output_dict["wrong"].keys())
    error_num = len(output_dict["error"].keys())
    invalid_num = len(output_dict["invalid"].keys())
    overall_accuracy = correct_sum / code_sum if code_sum else 0
    print("code_sum:", code_sum, " correct_sum: ", correct_sum, " wrong_num: ", wrong_num, " error_num: ", error_num,
          " accurancy: ", overall_accuracy)
    for lang, stats in per_language_totals.items():
        lang_code_sum = stats["code_sum"]
        lang_accuracy = stats["correct_sum"] / lang_code_sum if lang_code_sum else 0
        print(f"[ByLanguage] {lang} -> code_sum: {lang_code_sum} correct_sum: {stats['correct_sum']} "
              f"wrong_num: {stats['wrong_num']} error_num: {stats['error_num']} "
              f"invalid_num: {stats['invalid_num']} accuracy: {lang_accuracy}")
    for (src_lang, tgt_lang), stats in per_lang_pair_totals.items():
        pair_code_sum = stats["code_sum"]
        pair_accuracy = stats["correct_sum"] / pair_code_sum if pair_code_sum else 0
        print(f"[ByLangPair] {src_lang}->{tgt_lang} -> code_sum: {pair_code_sum} "
              f"correct_sum: {stats['correct_sum']} wrong_num: {stats['wrong_num']} "
              f"error_num: {stats['error_num']} invalid_num: {stats['invalid_num']} "
              f"accuracy: {pair_accuracy}")
    pass_summary = summarize_pass_metrics(group_results)
    total_attempts = code_sum + invalid_sum
    lambda_error_rate = invalid_sum / total_attempts if total_attempts else 0
    output_dict["info"] = {"code_sum": code_sum, "correct_sum": correct_sum, "wrong_num": wrong_num, "error_num":
        error_num, "invalid_num": invalid_num, "lambda_error_rate": lambda_error_rate, "aborted": aborted,
        "accurancy": overall_accuracy}
    per_language_summary = {}
    for lang, stats in per_language_totals.items():
        lang_code_sum = stats["code_sum"]
        lang_accuracy = stats["correct_sum"] / lang_code_sum if lang_code_sum else 0
        per_language_summary[lang] = {**stats, "accuracy": lang_accuracy}
    output_dict["info_by_language"] = per_language_summary
    per_lang_pair_summary = {}
    for (src_lang, tgt_lang), stats in per_lang_pair_totals.items():
        pair_code_sum = stats["code_sum"]
        pair_accuracy = stats["correct_sum"] / pair_code_sum if pair_code_sum else 0
        key = f"{src_lang}__{tgt_lang}"
        per_lang_pair_summary[key] = {
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            **stats,
            "accuracy": pair_accuracy,
        }
    output_dict["info_by_lang_pair"] = per_lang_pair_summary
    output_dict["pass_metrics"] = pass_summary

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f)

    if aborted:
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_path', type=str, default="program_synthesis_eval_palm_d.jsonl")
    parser.add_argument('--output_path', type=str, default="./results/executed_result.json")
    parser.add_argument('--lang_url_config', type=str, default="evaluator/lang2url.json")
    parser.add_argument('--random_sample_size', type=int, default=0,
                        help="If >0, randomly select this many entries from --jsonl_path (e.g., 10) for a quick test run")
    parser.add_argument('--random_sample_seed', type=int, default=None,
                        help="Optional seed to make the random subset deterministic")
    parser.add_argument('--language', type=str, default=None,
                        help="Override language hint instead of inferring from file name")
    parser.add_argument('--request_timeout', type=float, default=900.0)
    parser.add_argument('--request_retries', type=int, default=5,
                        help="How many times to retry evaluator requests after failures/timeouts")
    parser.add_argument('--request_retry_delay', type=float, default=30.0,
                        help="Seconds to wait before retrying a failed evaluator request")
    parser.add_argument('--request_retry_max_delay', type=float, default=60.0,
                        help="Upper bound for exponential backoff delays (seconds)")
    parser.add_argument('--request_connect_timeout', type=float, default=5.0,
                        help="Connection timeout (seconds) for evaluator requests")
    parser.add_argument('--request_read_timeout', type=float, default=None,
                        help="Read timeout (seconds) for evaluator requests; defaults to --request_timeout")
    parser.add_argument('--healthcheck_attempts', type=int, default=5,
                        help="Number of attempts when probing Lambda /healthz endpoints")
    parser.add_argument('--healthcheck_interval', type=float, default=10.0,
                        help="Seconds to wait between health check attempts")
    parser.add_argument('--healthcheck_timeout', type=float, default=30.0,
                        help="Per-request timeout (seconds) for health checks")
    parser.add_argument('--max_lambda_error_rate', type=float, default=-1.0,
                        help="Abort if invalid (Lambda-side) error rate exceeds this value; set <0 to disable")
    parser.add_argument('--lambda_error_min_samples', type=int, default=20,
                        help="Minimum attempts before enforcing max_lambda_error_rate")
    parser.add_argument('--verbose', action='store_true',
                        help="Print request payloads and API responses for Lambda invocations")
    parser.add_argument('--max_workers', type=int, default=32,
                        help="Maximum number of entries to process in parallel")
    parser.add_argument('--max_inflight_total', type=int, default=64,
                        help="Limit total in-flight Lambda requests; set <=0 to disable. "
                             "Defaults to min(--max_workers, 64).")
    parser.add_argument('--max_inflight_per_lang', type=int, default=8,
                        help="Limit in-flight Lambda requests per language; set <=0 to disable. "
                             "Defaults to min(8, --max_inflight_total).")

    args = parser.parse_args()
    if args.max_inflight_total is None:
        args.max_inflight_total = min(args.max_workers, 64)
    if args.max_inflight_per_lang is None:
        if args.max_inflight_total and args.max_inflight_total > 0:
            args.max_inflight_per_lang = min(8, args.max_inflight_total)
        else:
            args.max_inflight_per_lang = 0

    exe_main()
