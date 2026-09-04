"""Run the small-sample overfitting sanity check using only official DROP evaluation.

Pipeline:
1. sample a small stratified subset from the unified training JSONL;
2. train T5-small on that subset and validate on the same subset;
3. greedily predict on the same subset;
4. convert the unified gold examples and T5 predictions to JSON structures accepted
   by the official DROP evaluator;
5. call drop_eval.py directly for EM/F1.

No metrics from evaluate_predictions.py are used.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ANSWER_SEPARATOR = " ||| "
GROUP_ORDER = (
    ("drop", "single_span"),
    ("drop", "multi_span"),
    ("quoref", "single_span"),
    ("quoref", "multi_span"),
)
GROUP_LABELS = {
    ("drop", "single_span"): ("DROP", "Single-span"),
    ("drop", "multi_span"): ("DROP", "Multi-span"),
    ("quoref", "single_span"): ("QUOREF", "Single-span"),
    ("quoref", "multi_span"): ("QUOREF", "Multi-span"),
}
AGGREGATE_GROUPS = (
    ("Overall", GROUP_ORDER),
    (
        "Single-span",
        (("drop", "single_span"), ("quoref", "single_span")),
    ),
    (
        "Multi-span",
        (("drop", "multi_span"), ("quoref", "multi_span")),
    ),
    ("DROP", (("drop", "single_span"), ("drop", "multi_span"))),
    (
        "QUOREF",
        (("quoref", "single_span"), ("quoref", "multi_span")),
    ),
)


def run(command: List[str]) -> None:
    print("\n> " + " ".join(command))
    subprocess.run(command, check=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}."
                )
            records.append(record)
    return records


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def percent_rounded(value: float) -> float:
    """Convert a metric fraction to a two-decimal percentage using half-up.

    The evaluator averages binary floating-point scores, so a mathematical
    value such as 0.99575 may arrive as 0.9957499999999999. Normalizing to 12
    decimal places before reporting prevents that representation artifact from
    turning 99.575 into 99.57 instead of the conventional 99.58.
    """
    normalized = Decimal(f"{value:.12f}") * Decimal("100")
    return float(normalized.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def save_breakdown_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Save the four paper-ready rows, with EM/F1 expressed as percentages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "answer_type", "num_examples", "em", "f1"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "answer_type": row["answer_type"],
                    "num_examples": row["num_examples"],
                    "em": f"{row['exact_match_percent']:.2f}",
                    "f1": f"{row['f1_percent']:.2f}",
                }
            )


def save_overall_table_csv(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    """Save the requested Overall/answer-type/dataset aggregate table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group", "examples", "em", "f1"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "group": row["group"],
                    "examples": row["num_examples"],
                    "em": f"{row['exact_match_percent']:.2f}",
                    "f1": f"{row['f1_percent']:.2f}",
                }
            )


def print_breakdown_table(rows: Sequence[Mapping[str, Any]]) -> None:
    """Print a compact four-row table without requiring a table package."""
    headers = ("Dataset", "Answer type", "N", "EM", "F1")
    body = [
        (
            str(row["dataset"]),
            str(row["answer_type"]),
            str(row["num_examples"]),
            f"{row['exact_match_percent']:.2f}",
            f"{row['f1_percent']:.2f}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in body))
        for index in range(len(headers))
    ]

    def format_row(values: Sequence[str]) -> str:
        return " | ".join(
            value.ljust(widths[index])
            if index < 2
            else value.rjust(widths[index])
            for index, value in enumerate(values)
        )

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in body:
        print(format_row(row))


def print_overall_table(rows: Sequence[Mapping[str, Any]]) -> None:
    """Print the requested five-row aggregate table."""
    headers = ("Group", "Examples", "EM", "F1")
    body = [
        (
            str(row["group"]),
            str(row["num_examples"]),
            f"{row['exact_match_percent']:.2f}",
            f"{row['f1_percent']:.2f}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in body))
        for index in range(len(headers))
    ]

    def format_row(values: Sequence[str]) -> str:
        return " | ".join(
            value.ljust(widths[index])
            if index == 0
            else value.rjust(widths[index])
            for index, value in enumerate(values)
        )

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in body:
        print(format_row(row))


def clean_answer_list(values: Iterable[Any]) -> List[str]:
    return [
        str(value).strip()
        for value in values
        if str(value).strip()
    ]


def prediction_answers(record: Mapping[str, Any]) -> List[str]:
    """Extract the answer list produced by predict_t5_experiments.py."""
    answers = record.get("predicted_answers")
    if isinstance(answers, list):
        return clean_answer_list(answers)

    prediction_text = str(record.get("prediction_text", ""))
    return [
        part.strip()
        for part in prediction_text.split(ANSWER_SEPARATOR)
        if part.strip()
    ]


def make_drop_answer(spans: Sequence[str]) -> Dict[str, Any]:
    """Create the answer object used by the released DROP JSON format."""
    return {
        "number": "",
        "date": {
            "day": "",
            "month": "",
            "year": "",
        },
        "spans": list(spans),
    }


def build_official_sanity_files(
    subset_file: Path,
    raw_predictions_file: Path,
    gold_output: Path,
    predictions_output: Path,
) -> Tuple[
    Dict[str, Any],
    Dict[str, List[str]],
    Dict[Tuple[str, str], List[str]],
]:
    """Convert the unified sanity subset and model outputs for drop_eval.py.

    The official evaluator matches by query_id; it does not require the IDs to be
    original DROP UUIDs. We therefore retain each unified ID (for example,
    ``drop:<id>`` or ``quoref:<id>``) as the query_id in both files. This also
    prevents cross-dataset ID collisions inside the mixed sanity subset.

    Each unified example is serialized as its own DROP-style passage entry. The
    prediction file is the standard evaluator dictionary:

        {"query_id": ["predicted span 1", "predicted span 2"]}
    """
    subset = read_jsonl(subset_file)
    prediction_rows = read_jsonl(raw_predictions_file)

    if not subset:
        raise ValueError(f"Sanity subset is empty: {subset_file}")

    gold: Dict[str, Any] = {}
    gold_ids: set[str] = set()
    group_ids: Dict[Tuple[str, str], List[str]] = {
        group: [] for group in GROUP_ORDER
    }

    for index, record in enumerate(subset):
        query_id = str(record.get("id", "")).strip()
        if not query_id:
            raise ValueError(f"Subset row {index + 1} has no non-empty 'id'.")
        if query_id in gold_ids:
            raise ValueError(f"Duplicate sanity gold ID: {query_id}")

        answers = record.get("answers")
        if not isinstance(answers, list):
            raise ValueError(
                f"Subset row {index + 1} ({query_id}) has no answer list."
            )
        spans = clean_answer_list(answers)
        if not spans:
            raise ValueError(
                f"Subset row {index + 1} ({query_id}) has no non-empty spans."
            )

        group = (
            str(record.get("dataset", "")).strip().lower(),
            str(record.get("answer_type", "")).strip().lower(),
        )
        if group not in group_ids:
            expected = ", ".join(
                f"{dataset}/{answer_type}"
                for dataset, answer_type in GROUP_ORDER
            )
            raise ValueError(
                f"Subset row {index + 1} ({query_id}) has unsupported "
                f"dataset/answer_type values: {group!r}. Expected one of: "
                + expected
            )

        # A unique passage key is sufficient because drop_eval.py matches the
        # prediction to qa_pair['query_id']; it does not score passage IDs.
        passage_key = f"sanity_{index:04d}"
        gold[passage_key] = {
            "passage": str(record.get("context", "")),
            "qa_pairs": [
                {
                    "question": str(record.get("question", "")),
                    "query_id": query_id,
                    "answer": make_drop_answer(spans),
                }
            ],
        }
        gold_ids.add(query_id)
        group_ids[group].append(query_id)

    predictions: Dict[str, List[str]] = {}
    duplicate_prediction_ids: List[str] = []

    for index, record in enumerate(prediction_rows):
        query_id = str(record.get("id", "")).strip()
        if not query_id:
            raise ValueError(
                f"Prediction row {index + 1} has no non-empty 'id'."
            )
        if query_id in predictions:
            duplicate_prediction_ids.append(query_id)
            continue
        predictions[query_id] = prediction_answers(record)

    if duplicate_prediction_ids:
        examples = ", ".join(duplicate_prediction_ids[:3])
        raise ValueError(
            "Duplicate prediction IDs were produced. "
            f"Examples: {examples}"
        )

    prediction_ids = set(predictions)
    missing = sorted(gold_ids - prediction_ids)
    extra = sorted(prediction_ids - gold_ids)

    if missing:
        raise ValueError(
            f"Missing predictions for {len(missing)} sanity examples. "
            f"Example: {missing[0]}"
        )
    if extra:
        raise ValueError(
            f"Found {len(extra)} prediction IDs not present in the sanity subset. "
            f"Example: {extra[0]}"
        )

    save_json(gold, gold_output)
    save_json(predictions, predictions_output)

    return gold, predictions, group_ids


def evaluate_group_breakdown(
    evaluator: Any,
    gold: Mapping[str, Any],
    predictions: Mapping[str, List[str]],
    group_ids: Mapping[Tuple[str, str], Sequence[str]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Run the official evaluator independently on each of the four groups."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for group in GROUP_ORDER:
        ids = list(group_ids.get(group, []))
        if not ids:
            dataset, answer_type = group
            raise ValueError(
                "Cannot create the requested four-row table because the sanity "
                f"subset contains no examples for {dataset}/{answer_type}."
            )

        selected_ids = set(ids)
        group_gold: Dict[str, Any] = {}
        for passage_key, passage_record in gold.items():
            qa_pairs = passage_record.get("qa_pairs", [])
            if len(qa_pairs) != 1:
                raise ValueError(
                    f"Expected one sanity QA pair in passage {passage_key}."
                )
            query_id = str(qa_pairs[0].get("query_id", ""))
            if query_id in selected_ids:
                group_gold[passage_key] = passage_record

        group_predictions = {
            query_id: predictions[query_id] for query_id in ids
        }
        if len(group_gold) != len(ids) or len(group_predictions) != len(ids):
            raise ValueError(
                f"Could not construct a complete official-evaluation group for {group}."
            )

        dataset, answer_type = group
        stem = f"{dataset}_{answer_type}"
        gold_file = output_dir / f"{stem}_gold.json"
        predictions_file = output_dir / f"{stem}_predictions.json"
        metrics_file = output_dir / f"{stem}_metrics.json"
        save_json(group_gold, gold_file)
        save_json(group_predictions, predictions_file)

        em, f1 = evaluator.evaluate_prediction_file(
            str(predictions_file),
            str(gold_file),
            str(metrics_file),
        )
        dataset_label, answer_type_label = GROUP_LABELS[group]
        rows.append(
            {
                "dataset": dataset_label,
                "answer_type": answer_type_label,
                "num_examples": len(ids),
                "exact_match": em,
                "f1": f1,
                "exact_match_percent": percent_rounded(em),
                "f1_percent": percent_rounded(f1),
                "official_gold_file": str(gold_file),
                "official_predictions_file": str(predictions_file),
                "metrics_file": str(metrics_file),
            }
        )

    return rows


def evaluate_aggregate_table(
    evaluator: Any,
    gold: Mapping[str, Any],
    predictions: Mapping[str, List[str]],
    group_ids: Mapping[Tuple[str, str], Sequence[str]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Evaluate Overall, answer-type, and dataset aggregates officially."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for label, included_groups in AGGREGATE_GROUPS:
        ids = [
            query_id
            for group in included_groups
            for query_id in group_ids.get(group, [])
        ]
        if not ids:
            raise ValueError(
                f"Cannot evaluate aggregate group {label!r}: it has no examples."
            )

        selected_ids = set(ids)
        aggregate_gold: Dict[str, Any] = {}
        for passage_key, passage_record in gold.items():
            qa_pairs = passage_record.get("qa_pairs", [])
            if len(qa_pairs) != 1:
                raise ValueError(
                    f"Expected one sanity QA pair in passage {passage_key}."
                )
            query_id = str(qa_pairs[0].get("query_id", ""))
            if query_id in selected_ids:
                aggregate_gold[passage_key] = passage_record

        aggregate_predictions = {
            query_id: predictions[query_id] for query_id in ids
        }
        if (
            len(aggregate_gold) != len(ids)
            or len(aggregate_predictions) != len(ids)
        ):
            raise ValueError(
                f"Could not construct a complete official aggregate for {label}."
            )

        stem = label.lower().replace("-", "_")
        gold_file = output_dir / f"{stem}_gold.json"
        predictions_file = output_dir / f"{stem}_predictions.json"
        metrics_file = output_dir / f"{stem}_metrics.json"
        save_json(aggregate_gold, gold_file)
        save_json(aggregate_predictions, predictions_file)

        em, f1 = evaluator.evaluate_prediction_file(
            str(predictions_file),
            str(gold_file),
            str(metrics_file),
        )
        rows.append(
            {
                "group": label,
                "num_examples": len(ids),
                "exact_match": em,
                "f1": f1,
                "exact_match_percent": percent_rounded(em),
                "f1_percent": percent_rounded(f1),
                "official_gold_file": str(gold_file),
                "official_predictions_file": str(predictions_file),
                "metrics_file": str(metrics_file),
            }
        )

    return rows


def load_drop_evaluator(path: Path):
    """Dynamically load the supplied official drop_eval.py."""
    spec = importlib.util.spec_from_file_location("official_drop_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "evaluate_prediction_file"):
        raise AttributeError(
            f"{path} does not contain evaluate_prediction_file()."
        )
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample 50-100 training examples, overfit T5-small on them, predict "
            "on the same examples, convert gold/predictions to DROP-compatible "
            "JSON, and score only with the official drop_eval.py evaluator."
        )
    )
    parser.add_argument(
        "--base_train_file",
        type=Path,
        default=None,
        help="Required unless --evaluation_only is used.",
    )
    parser.add_argument(
        "--drop_eval",
        type=Path,
        default=Path("drop_eval.py"),
        help="Path to the official DROP drop_eval.py file.",
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=Path("experiments/sanity_overfit"),
    )
    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--epochs", type=float, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--prediction_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--evaluation_only",
        action="store_true",
        help=(
            "Skip sampling, training, and prediction; rebuild all official "
            "metrics and the four-row table from sanity_subset.jsonl and "
            "sanity_predictions.jsonl already in --work_dir."
        ),
    )
    args = parser.parse_args()

    if args.evaluation_only and args.overwrite:
        parser.error("--evaluation_only and --overwrite cannot be used together.")
    if not args.evaluation_only and args.base_train_file is None:
        parser.error("--base_train_file is required unless --evaluation_only is used.")
    if args.base_train_file is not None and not args.base_train_file.exists():
        raise FileNotFoundError(args.base_train_file)
    if not args.drop_eval.exists():
        raise FileNotFoundError(args.drop_eval)

    if args.size <= 0:
        raise ValueError("--size must be positive.")

    if (
        not args.evaluation_only
        and args.work_dir.exists()
        and any(args.work_dir.iterdir())
    ):
        if not args.overwrite:
            raise FileExistsError(
                f"Work directory is not empty: {args.work_dir}. Use --overwrite."
            )
        shutil.rmtree(args.work_dir)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent

    subset_file = args.work_dir / "sanity_subset.jsonl"
    model_dir = args.work_dir / "model"
    raw_predictions_file = args.work_dir / "sanity_predictions.jsonl"
    official_gold_file = args.work_dir / "sanity_gold_drop_format.json"
    official_predictions_file = (
        args.work_dir / "sanity_predictions_drop_format.json"
    )
    metrics_file = args.work_dir / "sanity_official_metrics.json"
    summary_file = args.work_dir / "sanity_official_summary.json"
    group_evaluation_dir = args.work_dir / "official_group_evaluation"
    breakdown_json_file = args.work_dir / "sanity_official_breakdown.json"
    breakdown_csv_file = args.work_dir / "sanity_official_breakdown.csv"
    aggregate_evaluation_dir = args.work_dir / "official_aggregate_evaluation"
    overall_table_json_file = args.work_dir / "sanity_official_table.json"
    overall_table_csv_file = args.work_dir / "sanity_official_table.csv"

    if args.evaluation_only:
        for path in (subset_file, raw_predictions_file):
            if not path.exists():
                raise FileNotFoundError(
                    f"--evaluation_only requires an existing file: {path}"
                )
        print("\nEvaluation-only mode: reusing the existing subset and predictions.")
    else:
        # These are the same project components used by the previous runner.
        make_subset_script = script_dir / "make_sanity_subset.py"
        train_script = script_dir / "train_t5_experiments.py"
        predict_script = script_dir / "predict_t5_experiments.py"
        for path in (make_subset_script, train_script, predict_script):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required companion script does not exist: {path}"
                )

        run(
            [
                sys.executable,
                str(make_subset_script),
                "--input_file",
                str(args.base_train_file),
                "--output_file",
                str(subset_file),
                "--size",
                str(args.size),
                "--seed",
                str(args.seed),
            ]
        )

        run(
            [
                sys.executable,
                str(train_script),
                "--train_file",
                str(subset_file),
                "--validation_file",
                str(subset_file),
                "--output_dir",
                str(model_dir),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--eval_batch_size",
                str(args.batch_size),
                "--gradient_accumulation_steps",
                "1",
                "--learning_rate",
                str(args.learning_rate),
                "--logging_steps",
                "5",
                "--save_total_limit",
                "1",
                "--precision",
                args.precision,
                "--seed",
                str(args.seed),
                "--overwrite_output_dir",
            ]
        )

        run(
            [
                sys.executable,
                str(predict_script),
                "--model_dir",
                str(model_dir),
                "--input_file",
                str(subset_file),
                "--output_file",
                str(raw_predictions_file),
                "--batch_size",
                str(args.prediction_batch_size),
                "--num_beams",
                "1",
                "--seed",
                str(args.seed),
            ]
        )

    print("\nConverting sanity gold and predictions to DROP-compatible JSON.")
    gold, predictions, group_ids = build_official_sanity_files(
        subset_file=subset_file,
        raw_predictions_file=raw_predictions_file,
        gold_output=official_gold_file,
        predictions_output=official_predictions_file,
    )
    gold_count = len(gold)
    prediction_count = len(predictions)
    print(f"Official-format gold questions: {gold_count}")
    print(f"Official-format predictions:    {prediction_count}")
    print(f"Gold file:        {official_gold_file}")
    print(f"Prediction file:  {official_predictions_file}")

    print("\nEvaluating with official DROP evaluator only.")
    evaluator = load_drop_evaluator(args.drop_eval)
    em, f1 = evaluator.evaluate_prediction_file(
        str(official_predictions_file),
        str(official_gold_file),
        str(metrics_file),
    )

    print("\nEvaluating the four dataset/answer-type groups separately.")
    breakdown_rows = evaluate_group_breakdown(
        evaluator=evaluator,
        gold=gold,
        predictions=predictions,
        group_ids=group_ids,
        output_dir=group_evaluation_dir,
    )
    save_json({"rows": breakdown_rows}, breakdown_json_file)
    save_breakdown_csv(breakdown_rows, breakdown_csv_file)

    print("\nEvaluating the five requested aggregate rows.")
    overall_table_rows = evaluate_aggregate_table(
        evaluator=evaluator,
        gold=gold,
        predictions=predictions,
        group_ids=group_ids,
        output_dir=aggregate_evaluation_dir,
    )
    save_json({"rows": overall_table_rows}, overall_table_json_file)
    save_overall_table_csv(overall_table_rows, overall_table_csv_file)

    summary = {
        "evaluator": str(args.drop_eval),
        "num_examples": gold_count,
        "exact_match": em,
        "f1": f1,
        "exact_match_percent": percent_rounded(em),
        "f1_percent": percent_rounded(f1),
        "subset_file": str(subset_file),
        "raw_predictions_file": str(raw_predictions_file),
        "official_gold_file": str(official_gold_file),
        "official_predictions_file": str(official_predictions_file),
        "metrics_file": str(metrics_file),
        "breakdown": breakdown_rows,
        "breakdown_json_file": str(breakdown_json_file),
        "breakdown_csv_file": str(breakdown_csv_file),
        "overall_table": overall_table_rows,
        "overall_table_json_file": str(overall_table_json_file),
        "overall_table_csv_file": str(overall_table_csv_file),
    }
    save_json(summary, summary_file)

    print("\nSanity experiment finished.")
    print("Evaluation path: T5 output -> DROP-compatible JSON -> drop_eval.py")
    print(f"Official EM: {em * 100:.2f}%")
    print(f"Official F1: {f1 * 100:.2f}%")
    print("\nOverall sanity table (EM/F1 are percentages):")
    print_overall_table(overall_table_rows)
    print(
        "Because training and evaluation use the same tiny subset, EM/F1 "
        "should be very high."
    )
    print(
        "A low score indicates a problem in preprocessing, target formatting, "
        "training, generation, conversion, or official evaluation."
    )
    print(f"Official metrics: {metrics_file}")
    print(f"Summary:          {summary_file}")
    print(f"4-row JSON:       {breakdown_json_file}")
    print(f"4-row CSV:        {breakdown_csv_file}")
    print(f"Overall JSON:     {overall_table_json_file}")
    print(f"Overall CSV:      {overall_table_csv_file}")
    trainer_state_file = model_dir / "trainer_state.json"
    if trainer_state_file.exists():
        print(
            "Trainer state for the learning-curve plot: "
            f"{trainer_state_file}"
        )


if __name__ == "__main__":
    main()
