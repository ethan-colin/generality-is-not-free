import argparse
import json
import re
from pathlib import Path


SEPARATOR = " ||| "
UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {path}"
                ) from error


def load_gold_ids(path: Path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        str(qa["query_id"])
        for passage in data.values()
        for qa in passage.get("qa_pairs", [])
    }


def get_answers(record):
    answers = record.get("predicted_answers")

    if isinstance(answers, list):
        return [
            str(answer).strip()
            for answer in answers
            if str(answer).strip()
        ]

    text = str(record.get("prediction_text", ""))

    return [
        answer.strip()
        for answer in text.split(SEPARATOR)
        if answer.strip()
    ]


def resolve_id(full_id: str, gold_ids: set[str]):
    """
    Find the original gold query ID inside a unified/composite ID.

    Examples:
        drop:abc                       -> abc
        nfl_1184:f37e...faa9:0         -> f37e...faa9
        quoref:some-id                 -> some-id
    """
    full_id = str(full_id).strip()

    # Exact match.
    if full_id in gold_ids:
        return full_id

    # Try each colon-separated component.
    for part in full_id.split(":"):
        if part in gold_ids:
            return part

    # DROP IDs are normally UUIDs. Extract a UUID from a composite ID.
    for candidate in UUID_PATTERN.findall(full_id):
        if candidate in gold_ids:
            return candidate

    return None


def convert_dataset(
    records,
    dataset_name: str,
    gold_ids: set[str],
):
    predictions = {}
    unmatched = []
    duplicates = []

    prefix = dataset_name + ":"

    for record in records:
        full_id = str(record["id"])

        # Select only this dataset.
        if not full_id.startswith(prefix):
            continue

        without_dataset_prefix = full_id[len(prefix):]
        gold_id = resolve_id(without_dataset_prefix, gold_ids)

        if gold_id is None:
            unmatched.append(full_id)
            continue

        if gold_id in predictions:
            duplicates.append((full_id, gold_id))
            # Keep the first prediction deterministically.
            continue

        predictions[gold_id] = get_answers(record)

    return predictions, unmatched, duplicates


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def print_report(
    name,
    predictions,
    gold_ids,
    unmatched,
    duplicates,
):
    matched = len(set(predictions) & gold_ids)
    missing = sorted(gold_ids - set(predictions))

    print(f"{name} predictions written: {len(predictions)}")
    print(f"{name} matched gold IDs: {matched}/{len(gold_ids)}")
    print(f"{name} unmatched input rows: {len(unmatched)}")
    print(f"{name} duplicate mappings skipped: {len(duplicates)}")
    print(f"{name} missing gold IDs: {len(missing)}")

    if unmatched:
        print(f"Example unmatched {name} ID:")
        print(unmatched[0])

    if duplicates:
        print(f"Example duplicate {name} mapping:")
        print(f"{duplicates[0][0]} -> {duplicates[0][1]}")

    if missing:
        print(f"Example missing {name} gold ID:")
        print(missing[0])

    if missing:
        raise ValueError(
            f"{name} still has {len(missing)} gold IDs "
            "without predictions."
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--gold_dir",
        type=Path,
        default=Path("official_gold"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("official_predictions"),
    )

    args = parser.parse_args()

    records = list(read_jsonl(args.input_file))

    drop_gold_ids = load_gold_ids(
        args.gold_dir / "drop_all_spans.json"
    )
    quoref_gold_ids = load_gold_ids(
        args.gold_dir / "quoref_all_spans.json"
    )

    drop_predictions, drop_unmatched, drop_duplicates = (
        convert_dataset(
            records,
            "drop",
            drop_gold_ids,
        )
    )

    quoref_predictions, quoref_unmatched, quoref_duplicates = (
        convert_dataset(
            records,
            "quoref",
            quoref_gold_ids,
        )
    )

    print_report(
        "DROP",
        drop_predictions,
        drop_gold_ids,
        drop_unmatched,
        drop_duplicates,
    )

    print()

    print_report(
        "QUOREF",
        quoref_predictions,
        quoref_gold_ids,
        quoref_unmatched,
        quoref_duplicates,
    )

    drop_output = (
        args.output_dir / "drop_predictions.json"
    )
    quoref_output = (
        args.output_dir / "quoref_predictions.json"
    )

    save_json(drop_predictions, drop_output)
    save_json(quoref_predictions, quoref_output)

    print()
    print(f"Saved: {drop_output}")
    print(f"Saved: {quoref_output}")


if __name__ == "__main__":
    main()