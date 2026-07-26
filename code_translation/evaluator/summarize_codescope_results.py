#!/usr/bin/env python3
"""Summarize CodeScope Singularity evaluations with Invalid retries applied."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RESULT_SUFFIX = "_singularity_fixed.json"
RESULT_PREFIX = "eval_"


def count_jsonl(path: Path) -> int:
    with path.open("rb") as input_file:
        return sum(1 for line in input_file if line.strip())


def parse_run_name(run_name: str) -> dict[str, object]:
    size_match = re.search(r"qwen3\.5_(4b|9b)", run_name, flags=re.IGNORECASE)
    if not size_match:
        raise ValueError(f"Cannot determine model size from run name: {run_name}")

    if "_pp0_" in run_name:
        penalty_label = "pp0"
        presence_penalty = 0.0
    elif "_defaultpp_" in run_name:
        penalty_label = "defaultpp"
        presence_penalty = 1.5
    else:
        raise ValueError(f"Cannot determine presence penalty from run name: {run_name}")

    return {
        "model": f"Qwen3.5-{size_match.group(1).upper()}",
        "training": "GRPO" if "_grpo_" in run_name else "Original",
        "mode": "thinking" if "_thinking_" in run_name else "instruct",
        "max_tokens": 8192 if "_max8192_" in run_name else None,
        "penalty_label": penalty_label,
        "presence_penalty": presence_penalty,
        "seed": 42 if run_name.endswith("_seed42") else None,
    }


def retry_counts(retry: dict) -> dict[str, int]:
    return {
        "accepted": len(retry.get("accepted", {})),
        "wrong_answer": len(retry.get("wrong", {})),
        "error": len(retry.get("error", {})),
        "invalid": len(retry.get("invalid", {})),
    }


def summarize(result_dir: Path, retry_dir: Path) -> list[dict]:
    rows = []
    for result_path in sorted(result_dir.glob(f"{RESULT_PREFIX}*{RESULT_SUFFIX}")):
        run_name = result_path.name.removeprefix(RESULT_PREFIX).removesuffix(RESULT_SUFFIX)
        input_path = result_dir / f"code_translation_eval_{run_name}.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing inference JSONL: {input_path}")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        info = result.get("info", {})
        original_invalid = len(result.get("invalid", {}))
        reported_invalid = int(info.get("invalid_num", 0))
        if original_invalid != reported_invalid:
            raise RuntimeError(
                f"{run_name}: invalid bucket has {original_invalid} entries, "
                f"but info.invalid_num is {reported_invalid}"
            )

        retry_path = retry_dir / f"{run_name}_invalid_retry.json"
        if original_invalid:
            if not retry_path.is_file():
                raise FileNotFoundError(
                    f"{run_name}: {original_invalid} Invalid entries but retry result is missing: "
                    f"{retry_path}"
                )
            retry = json.loads(retry_path.read_text(encoding="utf-8"))
            retry_summary = retry_counts(retry)
            if sum(retry_summary.values()) != original_invalid:
                raise RuntimeError(
                    f"{run_name}: retry classified {sum(retry_summary.values())} of "
                    f"{original_invalid} Invalid entries"
                )
        else:
            retry_summary = {
                "accepted": 0,
                "wrong_answer": 0,
                "error": 0,
                "invalid": 0,
            }

        total = count_jsonl(input_path)
        original_correct = int(info["correct_sum"])
        original_code_sum = int(info["code_sum"])
        corrected_correct = original_correct + retry_summary["accepted"]
        accounting_gap = total - original_code_sum - original_invalid
        if accounting_gap < 0:
            raise RuntimeError(f"{run_name}: aggregate counts exceed input count")

        row = {
            "run_name": run_name,
            **parse_run_name(run_name),
            "total": total,
            "original_correct": original_correct,
            "original_accuracy_all": original_correct / total if total else 0.0,
            "original_invalid": original_invalid,
            "retry_accepted": retry_summary["accepted"],
            "retry_wrong_answer": retry_summary["wrong_answer"],
            "retry_error": retry_summary["error"],
            "retry_invalid": retry_summary["invalid"],
            "corrected_correct": corrected_correct,
            "corrected_failed": total - corrected_correct,
            "corrected_accuracy": corrected_correct / total if total else 0.0,
            "accounting_gap": accounting_gap,
            "result_path": str(result_path),
            "retry_path": str(retry_path) if original_invalid else None,
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            int(str(row["model"]).split("-")[-1].removesuffix("B")),
            0 if row["training"] == "Original" else 1,
            -float(row["presence_penalty"]),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# CodeScope evaluation summary",
        "",
        "| Model | Training | Presence penalty | Correct / Total | Accuracy | "
        "Original Invalid | Retry result (A/W/E/I) | Accounting gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        retry_text = (
            f"{row['retry_accepted']}/{row['retry_wrong_answer']}/"
            f"{row['retry_error']}/{row['retry_invalid']}"
        )
        lines.append(
            f"| {row['model']} | {row['training']} | {row['presence_penalty']:.1f} "
            f"| {row['corrected_correct']} / {row['total']} "
            f"| {100 * row['corrected_accuracy']:.2f}% "
            f"| {row['original_invalid']} | {retry_text} | {row['accounting_gap']} |"
        )
    lines.extend([
        "",
        "Accuracy uses the inference JSONL line count as a fixed denominator. "
        "A/W/E/I means Accepted, Wrong Answer, Error, and Invalid in the retry.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument(
        "--retry-dir",
        type=Path,
        default=Path("result/invalid_retries"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("result/codescope_evaluation_summary"),
    )
    args = parser.parse_args()

    rows = summarize(args.result_dir, args.retry_dir)
    if not rows:
        raise RuntimeError(f"No completed evaluation results found in {args.result_dir}")

    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    markdown_path = args.output_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows)

    print(markdown_path.read_text(encoding="utf-8"), end="")
    print(f"Wrote {json_path}, {csv_path}, and {markdown_path}")


if __name__ == "__main__":
    main()
