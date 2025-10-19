#!/usr/bin/env python3
"""
Run all sample test codes against the ExecEval HTTP API from a Singularity container.

How to use (from host):
1) Build the image (once):
   singularity build ExecEval.sif Singularity.def

2) Start the service in one terminal (runs Gunicorn on port 5000):
   singularity run ExecEval.sif

3) In another terminal, run this script to evaluate all samples:
   singularity exec ExecEval.sif python3 eval_scripts/run_all_tests.py --endpoint http://127.0.0.1:5000

Notes:
- The service must be running before executing this script.
- Languages must match the keys defined in execution_engine/config.yaml.
- By default, network is blocked for executed code (block_network=true).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "execution_engine" / "test_codes"


LANG_TO_FILE = {
    # Core set already in the repo
    "GNU C": "test.c",
    "GNU C++17": "test.cpp",
    "Go": "test.go",
    "Java 17": "test.java",
    "Node js": "test.js",
    "Kotlin 1.7": "test.kt",
    "PHP": "test.php",
    "Python 3": "test.py",
    "Ruby": "test.rb",
    "Rust 2021": "test.rs",
    # Newly added languages
    "D": "test.d",
    "Perl": "test.pl",
    "Delphi": "test.pas",
    # C# via Mono (pick one existing key from config.yaml)
    "MS C#": "test.cs",
}


DEFAULT_TESTS = [
    {"input": "1 2", "output": ["3"]},
    {"input": "10 20", "output": ["30"]},
]


def post_json(url: str, payload: dict) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.getcode()
            body = resp.read()
            try:
                return status, json.loads(body.decode("utf-8"))
            except Exception:
                return status, body.decode("utf-8", errors="ignore")
    except Exception as e:
        return 0, f"HTTP error: {e}"


def run_sample(endpoint: str, language: str, code_path: pathlib.Path) -> tuple[bool, str]:
    try:
        source = code_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Failed to read {code_path.name}: {e}"

    payload = {
        "language": language,
        "source_code": source,
        "unittests": DEFAULT_TESTS,
        "block_network": True,
        "stop_on_first_fail": True,
    }
    status, body = post_json(f"{endpoint}/api/execute_code", payload)
    if status != 200:
        return False, f"{language}: HTTP {status} -> {body}"

    try:
        results = body.get("data", [])
        # Passed if all testcases report ExecOutcome.PASSED
        passed = all((tc.get("exec_outcome") == "PASSED") for tc in results)
        if passed:
            return True, f"{language}: PASS ({len(results)} tests)"
        else:
            return False, f"{language}: FAIL -> {results}"
    except Exception as e:
        return False, f"{language}: Response parse error: {e} | body={body}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:5000", help="ExecEval HTTP endpoint")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of language keys to run")
    args = ap.parse_args(argv)

    langs = LANG_TO_FILE
    if args.only:
        langs = {k: v for k, v in LANG_TO_FILE.items() if k in set(args.only)}
        missing = set(args.only) - set(langs.keys())
        if missing:
            print(f"[warn] Unknown languages ignored: {sorted(missing)}", file=sys.stderr)

    print(f"Running {len(langs)} languages against {args.endpoint}\n")
    any_fail = False
    for lang, fname in langs.items():
        path = TEST_DIR / fname
        ok, msg = run_sample(args.endpoint, lang, path)
        print(msg)
        any_fail |= (not ok)

    print("\nDONE.")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
