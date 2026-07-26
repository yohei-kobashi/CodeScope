#!/usr/bin/env python3
"""Extract only records reported as Invalid by completed evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def submission_ids(record: dict) -> set[str]:
    base_id = str(
        record.get("code_uid")
        or record.get("submission_id")
        or record["src_uid"]
    )
    ids = set()
    for key, value in record.items():
        if not key.startswith("code_translation_") or not isinstance(value, str):
            continue
        suffix = key.removeprefix("code_translation_")
        if suffix.isdigit():
            ids.add(f"{base_id}_{key}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--output-dir", type=Path, default=Path("result/invalid_retries"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    pattern = "eval_*_singularity_fixed.json"
    for result_path in sorted(args.result_dir.glob(pattern)):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        invalid_ids = set(result.get("invalid", {}))
        if not invalid_ids:
            continue

        run_name = result_path.name.removeprefix("eval_").removesuffix(
            "_singularity_fixed.json"
        )
        input_path = args.result_dir / f"code_translation_eval_{run_name}.jsonl"
        subset_path = args.output_dir / f"{run_name}_invalid.jsonl"
        retry_path = args.output_dir / f"{run_name}_invalid_retry.json"

        selected = []
        with input_path.open(encoding="utf-8") as input_file:
            for line in input_file:
                record = json.loads(line)
                if submission_ids(record) & invalid_ids:
                    selected.append(line)

        if len(selected) != len(invalid_ids):
            raise RuntimeError(
                f"{run_name}: found {len(selected)} records for "
                f"{len(invalid_ids)} Invalid submission IDs"
            )
        subset_path.write_text("".join(selected), encoding="utf-8")
        manifest.append({
            "run_name": run_name,
            "count": len(selected),
            "input_path": str(subset_path),
            "output_path": str(retry_path),
        })

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} retry sets to {manifest_path}")
    for item in manifest:
        print(f"{item['run_name']}: {item['count']}")


if __name__ == "__main__":
    main()
