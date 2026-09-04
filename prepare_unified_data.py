import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ANSWER_SEPARATOR = " ||| "


def valid_answers(answers: Iterable[Any]) -> List[str]:
    """
    Keep only non-empty textual answers.

    We preserve the answer order because that order will also be used
    in target_text.
    """
    result = []

    for answer in answers:
        if isinstance(answer, str) and answer.strip():
            result.append(answer.strip())

    return result


def create_record(
    dataset: str,
    split: str,
    question_id: str,
    question: str,
    context: str,
    answers: List[str],
    passage_id: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create one dataset-independent training record.

    Every record has exactly the same fields.
    Dataset-specific fields use an empty string when unavailable.
    """
    return {
        "id": f"{dataset}:{question_id}",
        "dataset": dataset,
        "split": split,
        "question": question.strip(),
        "context": context,
        "input_text": (
            f"question: {question.strip()} "
            f"context: {context}"
        ),
        "answers": answers,
        "target_text": ANSWER_SEPARATOR.join(answers),
        "answer_type": (
            "multi_span" if len(answers) > 1 else "single_span"
        ),
        "num_answers": len(answers),

        # Always present in every record:
        "passage_id": passage_id or "",
        "title": title or "",
    }


def read_drop(
    input_path: Path,
    split: str,
) -> Iterable[Dict[str, Any]]:
    """
    Read the original DROP JSON structure.

    DROP structure:

    {
        "passage_id": {
            "passage": "...",
            "qa_pairs": [
                {
                    "query_id": "...",
                    "question": "...",
                    "answer": {
                        "spans": [...],
                        "number": "...",
                        "date": {...}
                    }
                }
            ]
        }
    }

    Only examples with non-empty answer["spans"] are retained.
    Number-only and date-only answers are skipped.
    """
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    for passage_id, passage_data in data.items():
        context = passage_data.get("passage", "")

        for qa_index, qa in enumerate(
            passage_data.get("qa_pairs", [])
        ):
            answer_object = qa.get("answer", {})
            answers = valid_answers(
                answer_object.get("spans", [])
            )

            # Skip DROP number/date answers.
            if not answers:
                continue

            question = qa.get("question", "")
            original_question_id = qa.get(
                "query_id",
                f"question-{qa_index}",
            )

            question_id = (
                f"{passage_id}:"
                f"{original_question_id}:"
                f"{qa_index}"
            )

            yield create_record(
                dataset="drop",
                split=split,
                question_id=str(question_id),
                question=question,
                context=context,
                answers=answers,
                passage_id=str(passage_id),
            )


def read_quoref(
    input_path: Path,
    split: str,
) -> Iterable[Dict[str, Any]]:
    """
    Read the original SQuAD-style QUOREF JSON structure.

    QUOREF structure:

    {
        "data": [
            {
                "title": "...",
                "paragraphs": [
                    {
                        "context": "...",
                        "qas": [
                            {
                                "id": "...",
                                "question": "...",
                                "answers": [
                                    {
                                        "text": "...",
                                        "answer_start": 123
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    """
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    for article_index, article in enumerate(data.get("data", [])):
        title = article.get("title", "")

        for paragraph_index, paragraph in enumerate(
            article.get("paragraphs", [])
        ):
            context = paragraph.get("context", "")

            for qa_index, qa in enumerate(paragraph.get("qas", [])):
                answers = valid_answers(
                    answer.get("text", "")
                    for answer in qa.get("answers", [])
                )

                if not answers:
                    continue

                question = qa.get("question", "")
                fallback_id = (
                    f"{article_index}-"
                    f"{paragraph_index}-"
                    f"{qa_index}"
                )
                question_id = qa.get("id", fallback_id)

                yield create_record(
                    dataset="quoref",
                    split=split,
                    question_id=str(question_id),
                    question=question,
                    context=context,
                    answers=answers,
                    title=title,
                )


def normalize_question_for_comparison(question: str) -> str:
    """
    Normalize minor trailing punctuation differences when checking duplicates.
    """
    return question.strip().rstrip("/").strip()


def write_jsonl(
    records: Iterable[Dict[str, Any]],
    output_path: Path,
) -> int:
    """
    Write one JSON object per line.

    Equivalent duplicate examples are skipped.
    Truly conflicting duplicate IDs cause an error.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_by_id = {}
    count = 0

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            example_id = record["id"]

            if example_id in seen_by_id:
                previous = seen_by_id[example_id]

                same_example = (
                    previous["dataset"] == record["dataset"]
                    and previous["passage_id"] == record["passage_id"]
                    and previous["context"] == record["context"]
                    and previous["answers"] == record["answers"]
                    and normalize_question_for_comparison(
                        previous["question"]
                    )
                    == normalize_question_for_comparison(
                        record["question"]
                    )
                )

                if same_example:
                    print(
                        f"Warning: skipping equivalent duplicate: "
                        f"{example_id}"
                    )
                    continue

                raise ValueError(
                    f"Conflicting records have the same ID: "
                    f"{example_id}"
                )

            seen_by_id[example_id] = record

            json.dump(record, file, ensure_ascii=False)
            file.write("\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DROP and QUOREF into one unified JSONL file."
        )
    )

    parser.add_argument(
        "--drop",
        type=Path,
        help="Path to a DROP train or development JSON file.",
    )

    parser.add_argument(
        "--quoref",
        type=Path,
        help="Path to a QUOREF train or development JSON file.",
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "validation", "test"],
        help="Name of the split being converted.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path of the output JSONL file.",
    )

    args = parser.parse_args()

    if args.drop is None and args.quoref is None:
        parser.error(
            "At least one of --drop or --quoref must be supplied."
        )

    all_records = []

    if args.drop is not None:
        if not args.drop.exists():
            raise FileNotFoundError(
                f"DROP file does not exist: {args.drop}"
            )

        drop_records = list(
            read_drop(args.drop, args.split)
        )
        all_records.extend(drop_records)

        print(
            f"Loaded {len(drop_records)} span-answer "
            f"examples from DROP."
        )

    if args.quoref is not None:
        if not args.quoref.exists():
            raise FileNotFoundError(
                f"QUOREF file does not exist: {args.quoref}"
            )

        quoref_records = list(
            read_quoref(args.quoref, args.split)
        )
        all_records.extend(quoref_records)

        print(
            f"Loaded {len(quoref_records)} examples "
            f"from QUOREF."
        )

    total = write_jsonl(all_records, args.output)

    print(f"Wrote {total} examples to {args.output}")


if __name__ == "__main__":
    main()