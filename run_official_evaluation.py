import argparse
import csv
import importlib.util
import json
from pathlib import Path


EVALUATIONS = [
    (
        "DROP",
        "All Spans",
        "drop_all_spans.json",
        "drop_predictions.json",
        "drop_all_spans_metrics.json",
    ),
    (
        "DROP",
        "Single Span",
        "drop_single_span.json",
        "drop_predictions.json",
        "drop_single_span_metrics.json",
    ),
    (
        "DROP",
        "Multi Span",
        "drop_multi_span.json",
        "drop_predictions.json",
        "drop_multi_span_metrics.json",
    ),
    (
        "QUOREF",
        "All Spans",
        "quoref_all_spans.json",
        "quoref_predictions.json",
        "quoref_all_spans_metrics.json",
    ),
    (
        "QUOREF",
        "Single Span",
        "quoref_single_span.json",
        "quoref_predictions.json",
        "quoref_single_span_metrics.json",
    ),
    (
        "QUOREF",
        "Multi Span",
        "quoref_multi_span.json",
        "quoref_predictions.json",
        "quoref_multi_span_metrics.json",
    ),
]


def load_drop_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location(
        "official_drop_eval",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load evaluator from: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "evaluate_prediction_file"):
        raise AttributeError(
            "drop_eval.py does not contain "
            "evaluate_prediction_file()."
        )

    return module


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the official DROP evaluator on all six "
            "DROP/QUOREF validation subsets."
        )
    )

    parser.add_argument(
        "--drop_eval",
        type=Path,
        default=Path("drop_eval.py"),
        help="Path to the official DROP drop_eval.py file.",
    )

    parser.add_argument(
        "--gold_dir",
        type=Path,
        default=Path("official_gold"),
    )

    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=Path("official_predictions"),
    )

    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("official_results"),
    )

    args = parser.parse_args()

    require_file(args.drop_eval)

    evaluator = load_drop_evaluator(args.drop_eval)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for (
        dataset,
        subset,
        gold_name,
        prediction_name,
        metrics_name,
    ) in EVALUATIONS:

        gold_path = args.gold_dir / gold_name
        prediction_path = (
            args.predictions_dir / prediction_name
        )
        metrics_path = args.results_dir / metrics_name

        require_file(gold_path)
        require_file(prediction_path)

        print()
        print(f"Evaluating {dataset} - {subset}")

        em, f1 = evaluator.evaluate_prediction_file(
            str(prediction_path),
            str(gold_path),
            str(metrics_path),
        )

        rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                "em": round(em * 100, 2),
                "f1": round(f1 * 100, 2),
            }
        )

    summary_json = args.results_dir / "summary.json"
    summary_csv = args.results_dir / "summary.csv"

    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    with summary_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["dataset", "subset", "em", "f1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"{'Dataset':<10}"
        f"{'Subset':<15}"
        f"{'EM':>10}"
        f"{'F1':>10}"
    )
    print("-" * 45)

    for row in rows:
        print(
            f"{row['dataset']:<10}"
            f"{row['subset']:<15}"
            f"{row['em']:>10.2f}"
            f"{row['f1']:>10.2f}"
        )

    print()
    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
