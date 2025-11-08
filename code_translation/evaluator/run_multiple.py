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
from collections import defaultdict
from typing import MutableSet, Sequence, Tuple, Dict, List
from pathlib import Path
from urllib.parse import urljoin

import func_timeout
import requests
from func_timeout import func_set_timeout

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


def invoke_lambda_executor(lang_key: str, source_code: str, input_data: str, expected_output: str,
                           submission_id: str):
    base_url = resolve_language_url(lang_key)
    eval_url = urljoin(base_url if base_url.endswith("/") else base_url + "/", "evaluate")
    expected_list = [expected_output] if expected_output is not None else []
    payload = {
        "language": lang_key,
        "source_code": source_code,
        "input": input_data,
        "output": expected_list,
        "name": f"{submission_id}_{lang_key}",
    }

    resp = requests.post(eval_url, json=payload, timeout=getattr(args, "request_timeout", 60))
    data = None
    parse_error = None
    try:
        data = resp.json()
    except ValueError:
        parse_error = resp.text

    result_entry = (data.get("results") or [{}])[0] if isinstance(data, dict) else {}
    expected_from_result = result_entry.get("expected_output")
    if not expected_from_result:
        expected_from_result = expected_list

    return {
        "http_status": resp.status_code,
        "status": result_entry.get("status"),
        "stdout": result_entry.get("stdout"),
        "stderr": result_entry.get("stderr") if isinstance(result_entry, dict) else parse_error,
        "matched": result_entry.get("matched"),
        "program": result_entry.get("program"),
        "expected_output": expected_from_result,
        "raw_error": parse_error if parse_error and not result_entry else None,
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

@func_set_timeout(5)
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


@func_set_timeout(60)
def exe_testcase(source_code, answer, input_data, lang, output_dict, wrong_case, src_uid,
                 submission_id, difficulty, id, ):
    err = 0
    errtype = None
    outerr = None
    output_value = None

    normalized_lang = normalize_language_key(lang)
    if not normalized_lang:
        err = 1
        errtype = "UNSUPPORTED_LANGUAGE"
        outerr = f"Unable to normalize language key from '{lang}'"
        output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                             None, outerr, errtype)
        wrong_case += 1
        return output_dict, wrong_case, err

    try:
        response = invoke_lambda_executor(normalized_lang, source_code, input_data, answer, submission_id)
    except KeyError as exc:
        err = 1
        errtype = "UNSUPPORTED_LANGUAGE"
        outerr = str(exc)
    except requests.RequestException as exc:
        err = 1
        errtype = "NETWORK_ERROR"
        outerr = str(exc)
    except Exception as exc:
        err = 1
        errtype = "RUNTIME_ERROR"
        outerr = str(exc)
    else:
        http_status = response.get("http_status")
        status = response.get("status")
        output_value = response.get("stdout") or ""
        stderr_value = response.get("stderr") or response.get("raw_error")
        expected_outputs = response.get("expected_output") or ([answer] if answer else [])

        normalized_stdout = normalize_for_compare(output_value)
        normalized_expecteds = [normalize_for_compare(item) for item in expected_outputs if item is not None]
        matched = response.get("matched")
        if matched is None and normalized_expecteds:
            matched = any(normalized_stdout == candidate for candidate in normalized_expecteds)

        if http_status != 200:
            err = 1
            errtype = status_to_errtype(status or "HTTP_ERROR")
            outerr = f"HTTP {http_status}: {stderr_value or 'No response body'}"
        elif status is not None and status.upper() != "OK":
            err = 1
            errtype = status_to_errtype(status)
            outerr = stderr_value
        elif matched is False:
            err = 1
            errtype = "WRONG_ANSWER"
            outerr = stderr_value
        else:
            err = 0

    if err == 0:
        return output_dict, wrong_case, err

    if errtype == "WRONG_ANSWER":
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
    return output_dict, wrong_case, err


@func_set_timeout(300)
def exe_question(content, lang, output_dict, source_code: str, translation_label: str):
    source_code = source_code or ""

    id = content.get("id")
    src_uid = str(content["src_uid"])
    difficulty = str(content["difficulty"])
    testcases = ast.literal_eval(content['testcases'])
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
        return output_dict, 1

    source_code = source_code.replace("\\\\", "\\")
    source_code = source_code.replace("\\r", "\r")
    source_code = source_code.replace("\\n", "\n")
    source_code = source_code.replace("\\\"", "\"")
    source_code = source_code.replace("\r", "")
    source_code = source_code.replace("\r\n", "\n")
    source_code = strip_code_block_wrappers(source_code)

    wrong_case = 0
    err = 0
    for testcase in testcases:
        input = testcase["input"][0]
        answer = testcase["output"][0]

        input = input.replace("\r", "")
        input = input.replace("\r\n", "\n")

        try:
            output_dict, wrong_case, err = exe_testcase(source_code, answer, input, lang,
                                                        output_dict, wrong_case,
                                                        src_uid, submission_id, difficulty, id)
        except func_timeout.exceptions.FunctionTimedOut:
            err, wrong_case = 1, 1
            print("Time Limit Exceeded")
            output_dict["error"] = record_result(output_dict["error"], src_uid, submission_id, difficulty, id, None,
                                                 None, "Time Limit Exceeded", "RUNTIME_ERROR")

        if err == 1:
            wrong_case = 1
            break
    if err == 0:
        output_dict["accepted"] = record_result(output_dict["accepted"], src_uid, submission_id, difficulty, id, None,
                                                None, None, None)

    return output_dict, wrong_case


def exe_main():

    try:
        load_lang_urls(args.lang_url_config)
    except Exception as exc:
        raise RuntimeError(f"Failed to load language URL config: {exc}") from exc

    jsonl_path = args.jsonl_path
    if args.language:
        lang_hint = args.language
    else:
        lang_hint = jsonl_path.split(".")[0].split("_")[-1]

    code_sum, correct_sum = 0, 0
    output_dict = {"accepted": {}, "wrong": {}, "error": {}}
    prepared_languages: MutableSet[str] = set()
    group_results: Dict[Tuple[str, str], List[bool]] = defaultdict(list)
    per_language_totals: Dict[str, dict] = defaultdict(lambda: {
        "code_sum": 0,
        "correct_sum": 0,
        "wrong_num": 0,
        "error_num": 0,
    })
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            content = json.loads(line)
            translations = list(extract_translations(content))
            if not translations and "source_code" in content:
                translations = [("source_code", content["source_code"])]
            if not translations:
                print(f"No code translations found in line {line_idx + 1}, skipping")
                continue

            entry_lang = content.get("target_lang_cluster") or content.get("language") or lang_hint
            ensure_language_health_checked(
                entry_lang,
                prepared_languages,
                args.healthcheck_attempts,
                args.healthcheck_interval,
                args.healthcheck_timeout,
            )
            for translation_label, source_code in translations:
                prev_wrong = len(output_dict["wrong"])
                prev_error = len(output_dict["error"])
                try:
                    output_dict, wrong_case = exe_question(
                        content,
                        entry_lang,
                        output_dict,
                        source_code,
                        translation_label,
                    )
                except func_timeout.exceptions.FunctionTimedOut:
                    print("Time Limit Exceeded")
                    wrong_case = 1

                code_sum += 1
                lang_totals = per_language_totals[entry_lang]
                lang_totals["code_sum"] += 1
                success = wrong_case == 0
                if success:
                    correct_sum += 1
                    lang_totals["correct_sum"] += 1
                src_uid = str(content["src_uid"])
                group_results[(src_uid, entry_lang)].append(success)
                if not success:
                    new_wrong = len(output_dict["wrong"]) - prev_wrong
                    new_error = len(output_dict["error"]) - prev_error
                    if new_wrong > 0:
                        lang_totals["wrong_num"] += 1
                    elif new_error > 0:
                        lang_totals["error_num"] += 1
                    else:
                        lang_totals["error_num"] += 1

                print("done: ", code_sum, " not accepted: ", code_sum - correct_sum)

    wrong_num = len(output_dict["wrong"].keys())
    error_num = len(output_dict["error"].keys())
    print("code_sum:", code_sum, " correct_sum: ", correct_sum, " wrong_num: ", wrong_num, " error_num: ", error_num,
          " accurancy: ", correct_sum / code_sum)
    pass_summary = summarize_pass_metrics(group_results)
    overall_accuracy = correct_sum / code_sum if code_sum else 0
    output_dict["info"] = {"code_sum": code_sum, "correct_sum": correct_sum, "wrong_num": wrong_num, "error_num":
        error_num, "accurancy": overall_accuracy}
    per_language_summary = {}
    for lang, stats in per_language_totals.items():
        lang_code_sum = stats["code_sum"]
        lang_accuracy = stats["correct_sum"] / lang_code_sum if lang_code_sum else 0
        per_language_summary[lang] = {**stats, "accuracy": lang_accuracy}
    output_dict["info_by_language"] = per_language_summary
    output_dict["pass_metrics"] = pass_summary

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_path', type=str, default="program_synthesis_eval_palm_d.jsonl")
    parser.add_argument('--output_path', type=str, default="./results/executed_result.json")
    parser.add_argument('--lang_url_config', type=str, default="lang2url.json")
    parser.add_argument('--language', type=str, default=None,
                        help="Override language hint instead of inferring from file name")
    parser.add_argument('--request_timeout', type=float, default=60.0)
    parser.add_argument('--healthcheck_attempts', type=int, default=5,
                        help="Number of attempts when probing Lambda /healthz endpoints")
    parser.add_argument('--healthcheck_interval', type=float, default=10.0,
                        help="Seconds to wait between health check attempts")
    parser.add_argument('--healthcheck_timeout', type=float, default=5.0,
                        help="Per-request timeout (seconds) for health checks")

    args = parser.parse_args()

    exe_main()
