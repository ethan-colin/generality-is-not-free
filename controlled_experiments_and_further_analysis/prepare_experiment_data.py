"""Create controlled training-data variants from unified_train.jsonl."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from qa_utils import read_jsonl, save_json, write_jsonl

MODES = (
    "full",
    "drop_only",
    "quoref_only",
    "balanced",
    "multispan_x2",
    "multispan_x4",
    "single_only",
    "multi_only",
)


def add_duplicate(record: Dict[str, Any], copy_number: int) -> Dict[str, Any]:
    duplicate = deepcopy(record)
    duplicate["id"] = f"{record['id']}::oversample-{copy_number}"
    duplicate["augmentation"] = "multi_span_oversample"
    duplicate["source_id"] = record["id"]
    return duplicate


def create_variant(
    records: List[Dict[str, Any]],
    mode: str,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    if mode == "full":
        output = list(records)

    elif mode == "drop_only":
        output = [record for record in records if record.get("dataset") == "drop"]

    elif mode == "quoref_only":
        output = [record for record in records if record.get("dataset") == "quoref"]

    elif mode == "single_only":
        output = [
            record
            for record in records
            if record.get("answer_type") == "single_span"
        ]

    elif mode == "multi_only":
        output = [
            record
            for record in records
            if record.get("answer_type") == "multi_span"
        ]

    elif mode == "balanced":
        # Balance single- and multi-span examples separately inside each dataset.
        # We keep every multi-span example and randomly downsample single-span
        # examples to the same count. This directly tests whether the large
        # single-span majority is suppressing multi-span learning.
        output = []
        by_dataset = defaultdict(lambda: defaultdict(list))
        for record in records:
            by_dataset[str(record.get("dataset"))][
                str(record.get("answer_type"))
            ].append(record)

        for dataset_name in sorted(by_dataset):
            singles = list(by_dataset[dataset_name]["single_span"])
            multis = list(by_dataset[dataset_name]["multi_span"])
            rng.shuffle(singles)
            rng.shuffle(multis)

            keep_count = min(len(singles), len(multis))
            output.extend(singles[:keep_count])
            output.extend(multis[:keep_count])

    elif mode in {"multispan_x2", "multispan_x4"}:
        factor = 2 if mode == "multispan_x2" else 4
        output = list(records)

        for record in records:
            if record.get("answer_type") != "multi_span":
                continue
            for copy_number in range(1, factor):
                output.append(add_duplicate(record, copy_number))

    else:
        raise ValueError(f"Unknown mode: {mode}")

    rng.shuffle(output)
    return output


def count_groups(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for record in records:
        dataset = str(record.get("dataset", "unknown"))
        answer_type = str(record.get("answer_type", "unknown"))
        counts["total"] += 1
        counts[f"dataset__{dataset}"] += 1
        counts[f"answer_type__{answer_type}"] += 1
        counts[f"group__{dataset}__{answer_type}"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create controlled training-data variants."
    )
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest_file", type=Path, default=None)
    args = parser.parse_args()

    records = read_jsonl(args.input_file)
    output = create_variant(records, args.mode, args.seed)

    if not output:
        raise ValueError(f"Mode {args.mode!r} produced an empty training file.")

    written = write_jsonl(output, args.output_file)
    manifest = {
        "mode": args.mode,
        "seed": args.seed,
        "source_file": str(args.input_file),
        "output_file": str(args.output_file),
        "source_counts": count_groups(records),
        "output_counts": count_groups(output),
        "written": written,
    }

    manifest_path = args.manifest_file or args.output_file.with_suffix(
        args.output_file.suffix + ".manifest.json"
    )
    save_json(manifest, manifest_path)

    print(f"Created experiment mode: {args.mode}")
    print(f"Wrote {written} records to: {args.output_file}")
    print("Output counts:")
    for name, value in manifest["output_counts"].items():
        print(f"  {name}: {value}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
