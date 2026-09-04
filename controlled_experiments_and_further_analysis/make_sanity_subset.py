"""Create a small stratified subset for the required overfitting sanity check."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from qa_utils import read_jsonl, save_json, write_jsonl


def group_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(record.get("dataset", "unknown")),
        str(record.get("answer_type", "unknown")),
    )


def stratified_sample(
    records: List[Dict[str, Any]],
    size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if size <= 0:
        raise ValueError("--size must be positive.")
    if size > len(records):
        raise ValueError(
            f"Requested {size} examples, but the input contains only {len(records)}."
        )

    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)

    for values in groups.values():
        rng.shuffle(values)

    group_names = sorted(groups)
    selected: List[Dict[str, Any]] = []

    # Round-robin selection gives each available dataset/answer-type group
    # approximately equal representation, including the rarer multi-span groups.
    index = 0
    while len(selected) < size:
        made_progress = False
        for name in group_names:
            values = groups[name]
            if index < len(values) and len(selected) < size:
                selected.append(values[index])
                made_progress = True
        if not made_progress:
            break
        index += 1

    if len(selected) < size:
        chosen_ids = {id(record) for record in selected}
        leftovers = [record for record in records if id(record) not in chosen_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: size - len(selected)])

    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small stratified train-equals-evaluation sanity subset."
    )
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest_file",
        type=Path,
        default=None,
        help="Defaults to <output_file>.manifest.json",
    )
    args = parser.parse_args()

    records = read_jsonl(args.input_file)
    selected = stratified_sample(records, args.size, args.seed)
    count = write_jsonl(selected, args.output_file)

    group_counts = Counter(
        f"{record.get('dataset')}__{record.get('answer_type')}"
        for record in selected
    )
    manifest = {
        "source_file": str(args.input_file),
        "output_file": str(args.output_file),
        "seed": args.seed,
        "requested_size": args.size,
        "written_size": count,
        "group_counts": dict(sorted(group_counts.items())),
        "ids": [record.get("id") for record in selected],
    }

    manifest_path = args.manifest_file or args.output_file.with_suffix(
        args.output_file.suffix + ".manifest.json"
    )
    save_json(manifest, manifest_path)

    print(f"Wrote {count} sanity examples to: {args.output_file}")
    for group, group_count in sorted(group_counts.items()):
        print(f"  {group}: {group_count}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
