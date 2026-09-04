"""Context-only post-processing experiments for generated answer lists."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from qa_utils import (
    ANSWER_SEPARATOR,
    answer_is_grounded,
    deduplicate_answers,
    get_predicted_answers,
    read_jsonl,
    write_jsonl,
)

MODES = (
    "none",
    "deduplicate",
    "drop_ungrounded",
    "deduplicate_and_drop_ungrounded",
)


def apply_mode(answers: List[str], context: str, mode: str) -> List[str]:
    output = list(answers)

    if mode in {"deduplicate", "deduplicate_and_drop_ungrounded"}:
        output = deduplicate_answers(output)

    if mode in {"drop_ungrounded", "deduplicate_and_drop_ungrounded"}:
        output = [answer for answer in output if answer_is_grounded(answer, context)]

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply label-free, context-only post-processing to generated answers."
        )
    )
    parser.add_argument("--gold_file", type=Path, required=True)
    parser.add_argument("--predictions_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()

    gold_records = read_jsonl(args.gold_file)
    prediction_records = read_jsonl(args.predictions_file)

    gold_by_id: Dict[str, Dict[str, Any]] = {}
    for record in gold_records:
        example_id = str(record.get("id", ""))
        if not example_id:
            raise ValueError("A gold record has an empty ID.")
        if example_id in gold_by_id:
            raise ValueError(f"Duplicate gold ID: {example_id}")
        gold_by_id[example_id] = record

    output_records = []
    changed = 0
    removed_spans = 0

    for prediction in prediction_records:
        example_id = str(prediction.get("id", ""))
        if example_id not in gold_by_id:
            raise KeyError(f"Prediction ID is absent from gold file: {example_id}")

        original_answers = get_predicted_answers(prediction)
        context = str(gold_by_id[example_id].get("context", ""))
        processed_answers = apply_mode(original_answers, context, args.mode)

        if processed_answers != original_answers:
            changed += 1
            removed_spans += max(0, len(original_answers) - len(processed_answers))

        output = dict(prediction)
        output["prediction_text"] = ANSWER_SEPARATOR.join(processed_answers)
        output["predicted_answers"] = processed_answers
        output["postprocessing"] = {
            "mode": args.mode,
            "original_answers": original_answers,
            "removed_count": len(original_answers) - len(processed_answers),
        }
        output_records.append(output)

    write_jsonl(output_records, args.output_file)

    print(f"Post-processing mode: {args.mode}")
    print(f"Prediction rows: {len(output_records)}")
    print(f"Changed rows: {changed}")
    print(f"Removed answer spans: {removed_spans}")
    print(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
