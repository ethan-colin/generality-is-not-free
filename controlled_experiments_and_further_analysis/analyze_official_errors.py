"""
Per-example error analysis using the official DROP metric implementation.
Works for DROP and for the dropified QUOREF gold files.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from qa_utils import (
    answer_is_grounded,
    deduplicate_answers,
    get_predicted_answers,
    load_json,
    load_python_module,
    normalize_answer,
    save_json,
)

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def explicit_count_cue(question: str) -> int | None:
    normalized = normalize_answer(question)
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", normalized):
            return value
    digit_match = re.search(r"\b([1-9]|10)\b", normalized)
    if digit_match:
        return int(digit_match.group(1))
    return None


def best_official_scores(
    evaluator,
    qa_pair: Dict[str, Any],
    predicted_answers: Sequence[str],
) -> Tuple[float, float, List[str], str]:
    candidates = [qa_pair["answer"]]
    candidates.extend(qa_pair.get("validated_answers", []) or [])

    max_em = 0.0
    max_f1 = 0.0
    best_gold: List[str] = []
    best_type = "unknown"
    best_f1_for_selection = -1.0
    best_em_for_selection = -1.0

    for candidate in candidates:
        gold_tuple, gold_type = evaluator.answer_json_to_strings(candidate)
        gold_answers = list(gold_tuple)
        em, f1 = evaluator.get_metrics(list(predicted_answers), gold_tuple)
        max_em = max(max_em, float(em))
        max_f1 = max(max_f1, float(f1))

        if (float(f1), float(em)) > (
            best_f1_for_selection,
            best_em_for_selection,
        ):
            best_f1_for_selection = float(f1)
            best_em_for_selection = float(em)
            best_gold = gold_answers
            best_type = gold_type

    return max_em, max_f1, best_gold, best_type


def looks_like_split(gold: Sequence[str], predicted: Sequence[str]) -> bool:
    if len(predicted) <= len(gold) or len(predicted) < 2:
        return False

    normalized_predictions = [normalize_answer(x) for x in predicted]
    joined_predictions = normalize_answer(" ".join(predicted))

    for gold_answer in gold:
        normalized_gold = normalize_answer(gold_answer)
        if joined_predictions == normalized_gold:
            return True

        contained = [
            part
            for part in normalized_predictions
            if part and part in normalized_gold
        ]
        if len(contained) >= 2:
            joined_contained = " ".join(contained)
            if joined_contained == normalized_gold:
                return True

    return False


def looks_like_merge(gold: Sequence[str], predicted: Sequence[str]) -> bool:
    if len(predicted) >= len(gold) or len(gold) < 2:
        return False

    normalized_gold = [normalize_answer(x) for x in gold]
    for predicted_answer in predicted:
        normalized_prediction = normalize_answer(predicted_answer)
        contained_count = sum(
            1
            for gold_answer in normalized_gold
            if gold_answer and gold_answer in normalized_prediction
        )
        if contained_count >= 2:
            return True

    return False


def classify_error(
    em: float,
    f1: float,
    gold: Sequence[str],
    predicted: Sequence[str],
    context: str,
) -> str:
    if em == 1.0:
        return "correct"
    if not predicted:
        return "empty_prediction"

    all_grounded = all(answer_is_grounded(answer, context) for answer in predicted)
    if not all_grounded:
        return "ungrounded_prediction"

    normalized = [normalize_answer(answer) for answer in predicted]
    if len(normalized) != len(set(normalized)):
        return "duplicate_prediction"

    if looks_like_split(gold, predicted):
        return "split_gold_span"
    if looks_like_merge(gold, predicted):
        return "merged_gold_spans"

    if len(predicted) < len(gold):
        return "too_few_spans"
    if len(predicted) > len(gold):
        return "too_many_spans"
    if f1 > 0:
        return "boundary_or_partial_match"
    return "wrong_answer"


def context_excerpt(context: str, gold: Sequence[str], predicted: Sequence[str], width: int = 500) -> str:
    normalized_context = context.lower()
    search_terms = [x for x in list(gold) + list(predicted) if str(x).strip()]
    location = -1
    for term in search_terms:
        location = normalized_context.find(str(term).lower())
        if location >= 0:
            break

    if location < 0:
        return context[:width].replace("\n", " ")

    start = max(0, location - width // 2)
    end = min(len(context), start + width)
    excerpt = context[start:end].replace("\n", " ")
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(context):
        excerpt += "..."
    return excerpt


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-example official-metric error analysis."
    )
    parser.add_argument("--gold_file", type=Path, required=True)
    parser.add_argument("--predictions_file", type=Path, required=True)
    parser.add_argument("--drop_eval", type=Path, required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manual_examples_per_category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for path in (args.gold_file, args.predictions_file, args.drop_eval):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = load_python_module(args.drop_eval, "official_drop_eval_for_analysis")
    gold_data = load_json(args.gold_file)
    predictions = load_json(args.predictions_file)

    details: List[Dict[str, Any]] = []
    category_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing_predictions = 0

    for passage_id, passage in gold_data.items():
        context = str(passage.get("passage", ""))

        for qa_pair in passage.get("qa_pairs", []):
            query_id = str(qa_pair.get("query_id", ""))
            question = str(qa_pair.get("question", ""))
            predicted_answers = get_predicted_answers(predictions.get(query_id, []))
            if query_id not in predictions:
                missing_predictions += 1

            em, f1, gold_answers, gold_type = best_official_scores(
                evaluator,
                qa_pair,
                predicted_answers,
            )
            all_grounded = bool(predicted_answers) and all(
                answer_is_grounded(answer, context)
                for answer in predicted_answers
            )
            grounded_count = sum(
                answer_is_grounded(answer, context)
                for answer in predicted_answers
            )
            deduplicated_count = len(deduplicate_answers(predicted_answers))
            cue = explicit_count_cue(question)
            category = classify_error(
                em,
                f1,
                gold_answers,
                predicted_answers,
                context,
            )

            row = {
                "dataset": args.dataset_name,
                "passage_id": passage_id,
                "query_id": query_id,
                "question": question,
                "gold_answers": " ||| ".join(gold_answers),
                "predicted_answers": " ||| ".join(predicted_answers),
                "gold_type": gold_type,
                "gold_span_count": len(gold_answers),
                "predicted_span_count": len(predicted_answers),
                "span_count_difference": len(predicted_answers) - len(gold_answers),
                "em": em,
                "f1": f1,
                "all_predictions_grounded": int(all_grounded),
                "grounded_prediction_count": grounded_count,
                "duplicate_prediction_count": len(predicted_answers) - deduplicated_count,
                "explicit_count_cue": cue if cue is not None else "",
                "count_cue_followed": (
                    int(cue == len(predicted_answers)) if cue is not None else ""
                ),
                "error_category": category,
                "context_excerpt": context_excerpt(
                    context,
                    gold_answers,
                    predicted_answers,
                ),
            }
            details.append(row)
            category_rows[category].append(row)

    write_csv(details, args.output_dir / "error_details.csv")

    with (args.output_dir / "error_details.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        import json

        for row in details:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")

    category_summary = []
    for category, rows in sorted(category_rows.items()):
        category_summary.append(
            {
                "category": category,
                "count": len(rows),
                "percent": round(100 * len(rows) / len(details), 2) if details else 0,
                "mean_em": round(100 * safe_mean(float(row["em"]) for row in rows), 2),
                "mean_f1": round(100 * safe_mean(float(row["f1"]) for row in rows), 2),
                "grounded_percent": round(
                    100
                    * safe_mean(float(row["all_predictions_grounded"]) for row in rows),
                    2,
                ),
            }
        )
    write_csv(category_summary, args.output_dir / "error_category_summary.csv")

    by_gold_count: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in details:
        count = int(row["gold_span_count"])
        bucket = str(count) if count <= 4 else "5+"
        by_gold_count[bucket].append(row)

    span_count_summary = []
    bucket_order = ["1", "2", "3", "4", "5+"]
    for bucket in bucket_order:
        rows = by_gold_count.get(bucket, [])
        if not rows:
            continue
        span_count_summary.append(
            {
                "gold_span_count": bucket,
                "count": len(rows),
                "em": round(100 * safe_mean(float(row["em"]) for row in rows), 2),
                "f1": round(100 * safe_mean(float(row["f1"]) for row in rows), 2),
                "average_predicted_span_count": round(
                    safe_mean(float(row["predicted_span_count"]) for row in rows),
                    3,
                ),
                "all_grounded_percent": round(
                    100
                    * safe_mean(float(row["all_predictions_grounded"]) for row in rows),
                    2,
                ),
            }
        )
    write_csv(span_count_summary, args.output_dir / "performance_by_gold_spans.csv")

    confusion = Counter(
        (int(row["gold_span_count"]), int(row["predicted_span_count"]))
        for row in details
    )
    confusion_rows = [
        {
            "gold_span_count": gold_count,
            "predicted_span_count": pred_count,
            "count": count,
        }
        for (gold_count, pred_count), count in sorted(confusion.items())
    ]
    write_csv(confusion_rows, args.output_dir / "span_count_confusion.csv")

    cue_rows = [row for row in details if row["explicit_count_cue"] != ""]
    cue_summary = {
        "examples_with_explicit_count_cue": len(cue_rows),
        "count_cue_followed": sum(int(row["count_cue_followed"]) for row in cue_rows),
        "count_cue_followed_percent": round(
            100
            * safe_mean(float(row["count_cue_followed"]) for row in cue_rows),
            2,
        )
        if cue_rows
        else 0.0,
    }

    rng = random.Random(args.seed)
    manual_rows: List[Dict[str, Any]] = []
    for category, rows in sorted(category_rows.items()):
        if category == "correct":
            continue
        sampled = list(rows)
        rng.shuffle(sampled)
        manual_rows.extend(sampled[: args.manual_examples_per_category])
    write_csv(manual_rows, args.output_dir / "manual_review_sample.csv")

    summary = {
        "dataset": args.dataset_name,
        "examples": len(details),
        "missing_predictions": missing_predictions,
        "official_style_em": round(
            100 * safe_mean(float(row["em"]) for row in details), 2
        ),
        "official_style_f1": round(
            100 * safe_mean(float(row["f1"]) for row in details), 2
        ),
        "all_predictions_grounded_percent": round(
            100
            * safe_mean(float(row["all_predictions_grounded"]) for row in details),
            2,
        ),
        "error_categories": {
            row["category"]: row["count"] for row in category_summary
        },
        "explicit_count_cue_analysis": cue_summary,
    }
    save_json(summary, args.output_dir / "error_analysis_summary.json")

    print(f"Dataset: {args.dataset_name}")
    print(f"Examples: {len(details)}")
    print(f"EM: {summary['official_style_em']:.2f}")
    print(f"F1: {summary['official_style_f1']:.2f}")
    print(f"Missing predictions: {missing_predictions}")
    print("Error categories:")
    for row in category_summary:
        print(f"  {row['category']}: {row['count']} ({row['percent']:.2f}%)")
    print(f"Saved analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
