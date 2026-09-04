import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def add_passage(
    output: Dict[str, Any],
    passage_id: str,
    passage_info: Dict[str, Any],
    selected_qa_pairs: List[Dict[str, Any]],
) -> None:
    """
    Add the passage only when it contains at least one selected question.
    """
    if not selected_qa_pairs:
        return

    copied_passage = dict(passage_info)
    copied_passage["qa_pairs"] = selected_qa_pairs
    output[passage_id] = copied_passage


def split_drop_dataset(
    drop_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Split a DROP-format dataset into:
      - all span-answer questions
      - single-span questions
      - multi-span questions

    Number and date answers are excluded.
    """
    all_spans: Dict[str, Any] = {}
    single_span: Dict[str, Any] = {}
    multi_span: Dict[str, Any] = {}

    for passage_id, passage_info in drop_data.items():
        all_questions = []
        single_questions = []
        multi_questions = []

        for qa_pair in passage_info.get("qa_pairs", []):
            answer = qa_pair.get("answer", {})
            spans = answer.get("spans", [])

            if not spans:
                # Ignore DROP number/date questions.
                continue

            all_questions.append(qa_pair)

            if len(spans) == 1:
                single_questions.append(qa_pair)
            else:
                multi_questions.append(qa_pair)

        add_passage(
            all_spans,
            passage_id,
            passage_info,
            all_questions,
        )
        add_passage(
            single_span,
            passage_id,
            passage_info,
            single_questions,
        )
        add_passage(
            multi_span,
            passage_id,
            passage_info,
            multi_questions,
        )

    return {
        "all_spans": all_spans,
        "single_span": single_span,
        "multi_span": multi_span,
    }


def convert_quoref_to_drop_format(
    quoref_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert original SQuAD-style QUOREF into DROP format.

    This follows the same basic conversion used by the paper's
    tools/dropify_quoref.py script.
    """
    converted: Dict[str, Any] = {}
    paragraph_counter = 0

    for article in quoref_data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            passage_id = str(
                paragraph.get("context_id", paragraph_counter)
            )

            passage_info: Dict[str, Any] = {
                "passage": paragraph.get("context", ""),
                "qa_pairs": [],
            }

            if "title" in article:
                passage_info["title"] = article["title"]

            if "url" in article:
                passage_info["wiki_url"] = article["url"]

            for qa in paragraph.get("qas", []):
                answer_objects = qa.get("answers", [])
                answer_texts = [
                    str(answer.get("text", "")).strip()
                    for answer in answer_objects
                    if str(answer.get("text", "")).strip()
                ]

                # A development question should have at least one answer.
                if not answer_texts:
                    continue

                qa_pair = {
                    "question": qa.get("question", ""),
                    "query_id": str(qa.get("id", "")),
                    "answer": {
                        "number": "",
                        "date": {
                            "day": "",
                            "month": "",
                            "year": "",
                        },
                        "spans": answer_texts,
                    },
                    "original_answer": answer_objects,
                }

                passage_info["qa_pairs"].append(qa_pair)

            if passage_info["qa_pairs"]:
                converted[passage_id] = passage_info

            paragraph_counter += 1

    return converted


def count_questions(dataset: Dict[str, Any]) -> int:
    return sum(
        len(passage.get("qa_pairs", []))
        for passage in dataset.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create official DROP-format all-span, single-span, "
            "and multi-span validation files for DROP and QUOREF."
        )
    )

    parser.add_argument(
        "--drop_dev",
        required=True,
        type=Path,
        help="Original DROP development JSON file.",
    )

    parser.add_argument(
        "--quoref_dev",
        required=True,
        type=Path,
        help=(
            "Original SQuAD-style QUOREF development JSON file, "
            "or an already dropified QUOREF file."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("official_gold"),
    )

    args = parser.parse_args()

    if not args.drop_dev.exists():
        raise FileNotFoundError(
            f"DROP file does not exist: {args.drop_dev}"
        )

    if not args.quoref_dev.exists():
        raise FileNotFoundError(
            f"QUOREF file does not exist: {args.quoref_dev}"
        )

    drop_data = load_json(args.drop_dev)
    quoref_input = load_json(args.quoref_dev)

    # Original QUOREF has a top-level "data" field.
    # The paper's converted version is already in DROP format.
    if "data" in quoref_input:
        quoref_drop = convert_quoref_to_drop_format(quoref_input)
    else:
        quoref_drop = quoref_input

    drop_subsets = split_drop_dataset(drop_data)
    quoref_subsets = split_drop_dataset(quoref_drop)

    files = {
        "drop_all_spans.json": drop_subsets["all_spans"],
        "drop_single_span.json": drop_subsets["single_span"],
        "drop_multi_span.json": drop_subsets["multi_span"],
        "quoref_all_spans.json": quoref_subsets["all_spans"],
        "quoref_single_span.json": quoref_subsets["single_span"],
        "quoref_multi_span.json": quoref_subsets["multi_span"],
    }

    for file_name, dataset in files.items():
        output_path = args.output_dir / file_name
        save_json(dataset, output_path)

        print(
            f"{file_name}: "
            f"{count_questions(dataset)} questions"
        )

    print(f"\nSaved files in: {args.output_dir}")


if __name__ == "__main__":
    main()
