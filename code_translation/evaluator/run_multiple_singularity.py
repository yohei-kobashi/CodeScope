"""
Usage:
1. Build or prepare a Singularity sandbox/SIF for the MultiPL-E evaluation image.
2. Provide the JSONL file with code through `--jsonl_path`, and choose where to write results with `--output_path`.
3. Provide generated programs in each JSON record as `code_translation_<index>`; multiple variants per record are evaluated independently.
4. Set the desired runtime via `target_lang_cluster`, or override for the entire file with `--language`.
5. Example:
   `python CodeScope/code_translation/evaluator/run_multiple_singularity.py --jsonl_path data/sample.jsonl --output_path results/executed_result.json --singularity_image /path/to/multipl-e-eval_sandbox`
"""

from __future__ import annotations

import argparse
import json
import ast
import os
import sys
import math
import re
import random
import subprocess
import threading
import concurrent.futures
from contextlib import contextmanager
from collections import defaultdict
from typing import MutableSet, Sequence, Tuple, Dict, List
from pathlib import Path

import func_timeout
# from func_timeout import func_set_timeout

WRAPPER_DIR = Path(os.environ.get("MULTIPLE_E_EVALUATION_DIR", "/home/kobashi/MultiPL-E/evaluation"))
if str(WRAPPER_DIR) not in sys.path:
    sys.path.insert(0, str(WRAPPER_DIR))

from singularity_eval_wrapper import SingularityEvaluationError, SingularityEvaluator

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

TOTAL_INFLIGHT_LIMITER: threading.BoundedSemaphore | None = None
LANG_INFLIGHT_LIMITERS: dict[str, threading.BoundedSemaphore] = {}
LANG_LIMITER_LOCK = threading.Lock()
SINGULARITY_EVALUATOR: SingularityEvaluator | None = None


CPP_COMPILE_ONCE_RUNNER = r"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import eval_cpp
from safe_subprocess import run


def _response(result):
    return {"ok": True, "result": result}


def _compile_error_response(build_result, testcase):
    return _response({
        "status": "SyntaxError",
        "exit_code": build_result.exit_code,
        "stdout": build_result.stdout,
        "stderr": build_result.stderr,
        "input": testcase.get("input", ""),
        "expected_output": testcase.get("expected_output", []),
        "matched": False,
    })


try:
    payload = json.load(sys.stdin)
    source_code = payload["source_code"]
    testcases = payload.get("testcases", [])
    compile_timeout = int(payload.get("compile_timeout", 120))

    source_path = None
    binary_path = None
    responses = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as source_file:
            source_file.write(source_code.encode("utf-8"))
            source_file.flush()
            source_path = Path(source_file.name)

        binary_path = source_path.with_suffix("")
        build_result = run(
            ["g++", str(source_path), "-o", str(binary_path), "-std=c++17"],
            timeout_seconds=compile_timeout,
            max_output_size=8192,
        )
        if build_result.exit_code != 0:
            responses = [_compile_error_response(build_result, testcase) for testcase in testcases]
        else:
            for testcase in testcases:
                input_data = testcase.get("input", "")
                expected_outputs = testcase.get("expected_output", [])
                run_result = eval_cpp._run_cpp(binary_path, input_data=input_data)
                if run_result.timeout:
                    status = "Timeout"
                elif run_result.exit_code != 0:
                    status = "Exception"
                else:
                    status = "OK"

                converted_expected = [eval_cpp._convert_for_compare(item) for item in expected_outputs]
                converted_stdout = eval_cpp._convert_for_compare(run_result.stdout)
                matched = bool(converted_expected) and len(converted_stdout) == len(converted_expected[0]) and any(
                    all(
                        eval_cpp._compare_values(output, expected)
                        for output, expected in zip(converted_stdout, candidate)
                    )
                    for candidate in converted_expected
                )

                responses.append(_response({
                    "status": status,
                    "exit_code": run_result.exit_code,
                    "stdout": run_result.stdout,
                    "stderr": run_result.stderr,
                    "input": input_data,
                    "expected_output": expected_outputs,
                    "matched": matched,
                }))
    finally:
        for path in (binary_path, source_path):
            try:
                if path is not None and Path(path).exists():
                    os.remove(path)
            except Exception:
                pass

    print(json.dumps({"responses": responses}, ensure_ascii=True))
except Exception:
    print(json.dumps({
        "responses": [{
            "ok": False,
            "error": traceback.format_exc(),
            "result": {
                "status": "Exception",
                "exit_code": -1,
                "stdout": "",
                "stderr": traceback.format_exc(),
                "matched": False,
            },
        }]
    }, ensure_ascii=True))
"""


def normalize_language_key(lang: str) -> str:
    if not lang:
        return ""
    normalized = lang.lower()
    return LANGUAGE_KEY_MAP.get(normalized, normalized)


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


def summarize_pass_metric_averages(pass_summary: Dict[str, dict]) -> Dict[str, float]:
    aggregate_metrics: Dict[str, float] = {}
    for metric_name in ("pass@1", "pass@5", "pass@10"):
        values = [entry[metric_name] for entry in pass_summary.values() if metric_name in entry]
        if not values:
            continue
        mean_value = sum(values) / len(values)
        aggregate_metrics[f"{metric_name}_mean"] = mean_value
        if metric_name == "pass@1":
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            aggregate_metrics["pass@1_std"] = math.sqrt(variance)
    return aggregate_metrics


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
    text = remove_runtime_noise(text)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace(" ", "").lower().strip()
    return normalized


def remove_runtime_noise(text: str) -> str:
    noise_patterns = (
        r"^\[[0-9.]+s\]\[warning\]\[perf,memops\] Cannot use file /tmp/hsperfdata_.*$",
    )
    lines = []
    for line in text.splitlines():
        if any(re.match(pattern, line) for pattern in noise_patterns):
            continue
        lines.append(line)
    return "\n".join(lines)


def strip_code_block_wrappers(source_code: str) -> str:
    """Remove markdown-style code fences and surrounding text."""
    if not source_code:
        return ""
    source_code = re.sub(r"```\s*[#\*]* *(?:Explanation|Note).*", "", source_code, flags=re.DOTALL)
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


def invoke_singularity_executor(lang_key: str, source_code: str, input_data: str, expected_output: list,
                                submission_id: str, verbose: bool = False, debug_info: dict | None = None):
    if debug_info is None:
        debug_info = {}
    if SINGULARITY_EVALUATOR is None:
        raise RuntimeError("Singularity evaluator is not initialized")

    request_timeout = float(getattr(args, "request_timeout", 60))
    payload = {
        "language": lang_key,
        "source_code": source_code,
        "input": input_data,
        "output": expected_output,
        "name": f"{submission_id}_{lang_key}",
        "eval_timeout": 800,
    }

    debug_info["singularity_image"] = SINGULARITY_EVALUATOR.image
    if verbose:
        debug_info["payload"] = payload

    with _semaphore_guard(TOTAL_INFLIGHT_LIMITER), _semaphore_guard(get_lang_inflight_limiter(lang_key)):
        result_entry = SINGULARITY_EVALUATOR.eval_source_with_io(
            lang_key,
            source_code,
            input_data,
            expected_output,
            timeout=request_timeout,
        )

    if verbose:
        debug_info["response_body"] = json.dumps(result_entry, indent=2, ensure_ascii=False)

    expected_from_result = result_entry.get("expected_output")
    if not expected_from_result:
        expected_from_result = expected_output

    return {
        "http_status": 200,
        "status": result_entry.get("status"),
        "stdout": result_entry.get("stdout"),
        "stderr": result_entry.get("stderr"),
        "matched": result_entry.get("matched"),
        "program": result_entry.get("program"),
        "expected_output": expected_from_result,
        "raw_error": None,
    }, debug_info


def run_singularity_python_runner(runner_source: str, payload: dict, timeout: float) -> list[dict]:
    if SINGULARITY_EVALUATOR is None:
        raise RuntimeError("Singularity evaluator is not initialized")

    cmd = [
        SINGULARITY_EVALUATOR.runtime,
        "exec",
    ]
    if SINGULARITY_EVALUATOR.cleanenv:
        cmd.append("--cleanenv")
    for bind in SINGULARITY_EVALUATOR.binds:
        cmd.extend(["--bind", bind])
    if SINGULARITY_EVALUATOR.pwd:
        cmd.extend(["--pwd", SINGULARITY_EVALUATOR.pwd])
    cmd.extend([SINGULARITY_EVALUATOR.image, "python3", "-c", runner_source])

    proc = subprocess.run(
        cmd,
        input=json.dumps(payload),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise SingularityEvaluationError(
            f"{SINGULARITY_EVALUATOR.runtime} exited with status {proc.returncode}",
            returncode=proc.returncode,
            stderr=proc.stderr,
        )

    try:
        output_payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SingularityEvaluationError(
            "Evaluator returned non-JSON output",
            returncode=proc.returncode,
            stderr=(proc.stderr + "\nstdout:\n" + proc.stdout),
        ) from exc

    responses = output_payload.get("responses")
    if not isinstance(responses, list):
        raise SingularityEvaluationError(
            "Evaluator JSON did not contain a responses list",
            returncode=proc.returncode,
            stderr=(proc.stderr + "\nstdout:\n" + proc.stdout),
        )
    return responses


def invoke_singularity_batch(lang_key: str, source_code: str, testcases: list[dict],
                             submission_id: str, verbose: bool = False, debug_info: dict | None = None):
    if debug_info is None:
        debug_info = {}
    if SINGULARITY_EVALUATOR is None:
        raise RuntimeError("Singularity evaluator is not initialized")

    requests = [
        {
            "mode": "io",
            "language": lang_key,
            "source_code": source_code,
            "input": testcase["input"],
            "expected_output": testcase["answer"],
        }
        for testcase in testcases
    ]
    request_timeout = float(getattr(args, "request_timeout", 60))
    batch_timeout = getattr(args, "batch_timeout", None)
    if batch_timeout is None:
        batch_timeout = request_timeout * max(1, len(requests))

    debug_info["singularity_image"] = SINGULARITY_EVALUATOR.image
    debug_info["batch_size"] = len(requests)
    if verbose:
        debug_info["payload"] = {
            "language": lang_key,
            "name": f"{submission_id}_{lang_key}",
            "batch_size": len(requests),
            "requests": requests,
        }

    with _semaphore_guard(TOTAL_INFLIGHT_LIMITER), _semaphore_guard(get_lang_inflight_limiter(lang_key)):
        if lang_key == "cpp":
            responses = run_singularity_python_runner(
                CPP_COMPILE_ONCE_RUNNER,
                {
                    "source_code": source_code,
                    "testcases": requests,
                    "compile_timeout": int(getattr(args, "compile_timeout", 120)),
                },
                timeout=batch_timeout,
            )
        else:
            responses = SINGULARITY_EVALUATOR.eval_batch(requests, timeout=batch_timeout)

    if len(responses) != len(testcases):
        raise SingularityEvaluationError(
            f"Evaluator returned {len(responses)} responses for {len(testcases)} requests"
        )

    if verbose:
        debug_info["response_body"] = json.dumps(responses, indent=2, ensure_ascii=False)

    normalized = []
    for response, testcase in zip(responses, testcases):
        result_entry = response.get("result", {}) if isinstance(response, dict) else {}
        expected_from_result = result_entry.get("expected_output") or testcase["answer"]
        normalized.append({
            "http_status": 200,
            "status": result_entry.get("status"),
            "stdout": result_entry.get("stdout"),
            "stderr": result_entry.get("stderr") or (response.get("error") if isinstance(response, dict) else None),
            "matched": result_entry.get("matched"),
            "program": result_entry.get("program"),
            "expected_output": expected_from_result,
            "raw_error": None,
        })
    return normalized, debug_info


def prepare_testcases(raw_testcases) -> list[dict]:
    prepared = []
    for testcase in raw_testcases:
        raw_input = testcase.get("input", "")
        raw_output = testcase.get("output", "")

        if isinstance(raw_input, list):
            input_data = "\n".join(str(item) for item in raw_input)
        else:
            input_data = str(raw_input)

        if isinstance(raw_output, list):
            answer = [str(item) for item in raw_output]
        else:
            answer = [str(raw_output)]

        input_data = input_data.replace("\r\n", "\n").replace("\r", "")
        prepared.append({"input": input_data, "answer": answer})
    return prepared


def classify_response(response: dict) -> dict:
    local_err = 0
    local_errtype = None
    local_outerr = None
    local_output_value = response.get("stdout") or ""
    container_error = False

    http_status = response.get("http_status")
    status = response.get("status")
    stderr_value = response.get("stderr") or response.get("raw_error")
    expected_outputs = response.get("expected_output") or []
    raw_error = response.get("raw_error")

    normalized_stdout = normalize_for_compare(local_output_value)
    normalized_expecteds = [normalize_for_compare(str(item)) for item in expected_outputs if item is not None]
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
        local_errtype = "SINGULARITY_ERROR"
        local_outerr = f"Malformed response: {raw_error}"
        container_error = True
    elif http_status != 200:
        local_err = 1
        if http_status >= 500 or http_status in {429, 502, 503, 504}:
            local_errtype = "SINGULARITY_ERROR"
            container_error = True
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
        "container_error": container_error,
    }


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


def record_failed_case(output_dict, src_uid, submission_id, difficulty, id, answer, output_value, outerr, errtype):
    invalid_case = 0
    if errtype == "SINGULARITY_ERROR":
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
    else:
        output_dict["error"] = record_result(
            output_dict["error"],
            src_uid,
            submission_id,
            difficulty,
            id,
            None,
            None,
            outerr or "Unknown error",
            errtype or "RUNTIME_ERROR",
        )
    return output_dict, invalid_case


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
        container_error = False

        try:
            response, local_debug = invoke_singularity_executor(
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
        except SingularityEvaluationError as exc:
            local_err = 1
            local_errtype = "SINGULARITY_ERROR"
            local_outerr = str(exc)
            container_error = True
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
            normalized_expecteds = [normalize_for_compare(str(item)) for item in expected_outputs if item is not None]
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
                local_errtype = "SINGULARITY_ERROR"
                local_outerr = f"Malformed response: {raw_error}"
                container_error = True
            elif http_status != 200:
                local_err = 1
                if http_status >= 500 or http_status in {429, 502, 503, 504}:
                    local_errtype = "SINGULARITY_ERROR"
                    container_error = True
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
            "container_error": container_error,
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
        singularity_image = debug_info.get("singularity_image")
        if singularity_image:
            print(f"[Verbose] Singularity image: {singularity_image}")
        payload = debug_info.get("payload")
        if payload is not None:
            print("[Verbose] Payload:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        response_body = debug_info.get("response_body")
        if response_body is not None:
            print("[Verbose] Response body:")
            print(response_body)

    if errtype == "SINGULARITY_ERROR":
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

    prepared_testcases = prepare_testcases(testcases)
    normalized_lang = normalize_language_key(lang)
    if not normalized_lang:
        output_dict["error"] = record_result(
            output_dict["error"],
            src_uid,
            submission_id,
            difficulty,
            id,
            None,
            None,
            f"Unable to normalize language key from '{lang}'",
            "UNSUPPORTED_LANGUAGE",
        )
        return output_dict, 1, 0

    lang_candidates = ["python2", "python3"] if normalized_lang == "python" else [normalized_lang]
    verbose_enabled = getattr(args, "verbose", False)
    candidate_failures = []

    for lang_key in lang_candidates:
        debug_info: dict = {}
        try:
            responses, debug_info = invoke_singularity_batch(
                lang_key,
                source_code,
                prepared_testcases,
                submission_id,
                verbose=verbose_enabled,
                debug_info=debug_info,
            )
        except SingularityEvaluationError as exc:
            output_dict, invalid_case = record_failed_case(
                output_dict,
                src_uid,
                submission_id,
                difficulty,
                id,
                None,
                None,
                str(exc),
                "SINGULARITY_ERROR",
            )
            return output_dict, 1, invalid_case
        except func_timeout.exceptions.FunctionTimedOut:
            print("Time Limit Exceeded")
            output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                                 None, "Time Limit Exceeded", "RUNTIME_ERROR")
            return output_dict, 1, 0
        except Exception as exc:
            candidate_failures.append({
                "lang": lang_key,
                "case_index": 0,
                "answer": None,
                "output_value": None,
                "outerr": str(exc),
                "errtype": "RUNTIME_ERROR",
            })
            continue

        if verbose_enabled:
            singularity_image = debug_info.get("singularity_image")
            if singularity_image:
                print(f"[Verbose] Singularity image: {singularity_image}")
            print(f"[Verbose] Batch size: {debug_info.get('batch_size')}")
            payload = debug_info.get("payload")
            if payload is not None:
                print("[Verbose] Payload:")
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            response_body = debug_info.get("response_body")
            if response_body is not None:
                print("[Verbose] Response body:")
                print(response_body)

        first_failure = None
        for case_index, (response, testcase) in enumerate(zip(responses, prepared_testcases)):
            result = classify_response(response)
            if result["err"] == 0:
                continue
            first_failure = {
                "lang": lang_key,
                "case_index": case_index,
                "answer": testcase["answer"],
                "output_value": result.get("output_value"),
                "outerr": result.get("outerr"),
                "errtype": result.get("errtype"),
            }
            break

        if first_failure is None:
            output_dict["accepted"] = record_result(output_dict["accepted"], src_uid, submission_id, difficulty, id, None,
                                                    None, None, None)
            return output_dict, 0, 0

        candidate_failures.append(first_failure)

    failure = candidate_failures[-1] if candidate_failures else {
        "answer": None,
        "output_value": None,
        "outerr": "Unknown error",
        "errtype": "RUNTIME_ERROR",
    }
    if normalized_lang == "python" and len(candidate_failures) > 1:
        parts = []
        for item in candidate_failures:
            parts.append(f"{item['lang']}: {item.get('outerr') or item.get('errtype') or 'failed'}")
        failure["outerr"] = " | ".join(parts)

    output_dict, invalid_case = record_failed_case(
        output_dict,
        src_uid,
        submission_id,
        difficulty,
        id,
        failure.get("answer"),
        failure.get("output_value"),
        failure.get("outerr"),
        failure.get("errtype"),
    )
    if invalid_case:
        return output_dict, 1, 1

    return output_dict, 1, 0


def exe_main():
    global SINGULARITY_EVALUATOR, TOTAL_INFLIGHT_LIMITER

    SINGULARITY_EVALUATOR = SingularityEvaluator(
        image=args.singularity_image,
        runtime=args.singularity_runtime,
        pwd=args.singularity_pwd,
        cleanenv=not args.no_cleanenv,
        binds=args.singularity_bind or [],
        timeout=args.request_timeout,
    )

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
        print(f"[Runtime] {normalized}: using Singularity image {args.singularity_image}")

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
            max_invalid_rate = getattr(args, "max_container_error_rate", None)
            min_samples = getattr(args, "container_error_min_samples", 0)
            if (max_invalid_rate is not None and max_invalid_rate >= 0
                    and total_attempts >= min_samples and total_attempts > 0):
                invalid_rate = invalid_sum / total_attempts
                if invalid_rate > max_invalid_rate:
                    print(f"[Abort] Container error rate {invalid_rate:.3f} exceeded "
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
          " accuracy: ", overall_accuracy)
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
    pass_metric_averages = summarize_pass_metric_averages(pass_summary)
    if pass_metric_averages:
        pass_metric_text = " ".join(
            f"{metric_name}: {metric_value}"
            for metric_name, metric_value in sorted(pass_metric_averages.items())
        )
        print(f"[PassMetrics] {pass_metric_text}")
    total_attempts = code_sum + invalid_sum
    container_error_rate = invalid_sum / total_attempts if total_attempts else 0
    output_dict["info"] = {"code_sum": code_sum, "correct_sum": correct_sum, "wrong_num": wrong_num, "error_num":
        error_num, "invalid_num": invalid_num, "container_error_rate": container_error_rate, "aborted": aborted,
        "accuracy": overall_accuracy, **pass_metric_averages}
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
    parser.add_argument('--singularity_image', type=str, required=True,
                        help="Singularity sandbox/SIF path, e.g. multipl-e-eval_sandbox")
    parser.add_argument('--singularity_runtime', type=str, default="singularity",
                        help="Singularity executable name or path")
    parser.add_argument('--singularity_pwd', type=str, default="/code",
                        help="Working directory inside the Singularity container")
    parser.add_argument('--singularity_bind', action='append', default=[],
                        help="Bind mount passed to singularity exec. May be specified multiple times")
    parser.add_argument('--no_cleanenv', action='store_true',
                        help="Do not pass --cleanenv to singularity exec")
    parser.add_argument('--random_sample_size', type=int, default=0,
                        help="If >0, randomly select this many entries from --jsonl_path (e.g., 10) for a quick test run")
    parser.add_argument('--random_sample_seed', type=int, default=None,
                        help="Optional seed to make the random subset deterministic")
    parser.add_argument('--language', type=str, default=None,
                        help="Override language hint instead of inferring from file name")
    parser.add_argument('--request_timeout', type=float, default=900.0)
    parser.add_argument('--compile_timeout', type=int, default=120,
                        help="Seconds to allow for one compile step in compile-once evaluators such as C++")
    parser.add_argument('--batch_timeout', type=float, default=None,
                        help="Total timeout for one eval_batch call. Defaults to request_timeout * batch_size")
    parser.add_argument('--max_container_error_rate', type=float, default=-1.0,
                        help="Abort if invalid Singularity/container error rate exceeds this value; set <0 to disable")
    parser.add_argument('--container_error_min_samples', type=int, default=20,
                        help="Minimum attempts before enforcing max_container_error_rate")
    parser.add_argument('--verbose', action='store_true',
                        help="Print evaluator payloads and responses")
    parser.add_argument('--max_workers', type=int, default=4,
                        help="Maximum number of entries to process in parallel")
    parser.add_argument('--max_inflight_total', type=int, default=4,
                        help="Limit total in-flight Singularity evaluations; set <=0 to disable")
    parser.add_argument('--max_inflight_per_lang', type=int, default=2,
                        help="Limit in-flight Singularity evaluations per language; set <=0 to disable")

    args = parser.parse_args()
    if args.max_inflight_total is None:
        args.max_inflight_total = args.max_workers
    if args.max_inflight_per_lang is None:
        if args.max_inflight_total and args.max_inflight_total > 0:
            args.max_inflight_per_lang = min(2, args.max_inflight_total)
        else:
            args.max_inflight_per_lang = 0

    exe_main()
