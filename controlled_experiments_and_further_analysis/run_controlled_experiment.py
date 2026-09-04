"""Run one reproducible controlled experiment from prediction through official scores."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from prepare_experiment_data import MODES
from postprocess_predictions import MODES as POSTPROCESS_MODES


def run(command: List[str]) -> None:
    print("\n> " + " ".join(command))
    subprocess.run(command, check=True)


def append_results(summary_path: Path, aggregate_path: Path, metadata: dict) -> None:
    with summary_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    output_rows = [{**metadata, **row} for row in rows]

    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "train_mode",
        "model_dir",
        "num_beams",
        "postprocess_mode",
        "seed",
        "dataset",
        "subset",
        "em",
        "f1",
    ]

    existing_rows = []
    if aggregate_path.exists():
        with aggregate_path.open("r", encoding="utf-8", newline="") as file:
            existing_rows = [
                row
                for row in csv.DictReader(file)
                if row.get("experiment") != metadata["experiment"]
            ]

    with aggregate_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a training variant, train or reuse a model, generate predictions, "
            "optionally post-process them, and run the official evaluation."
        )
    )
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--validation_file", type=Path, required=True)
    parser.add_argument("--gold_dir", type=Path, required=True)
    parser.add_argument("--drop_eval", type=Path, required=True)
    parser.add_argument("--make_official_script", type=Path, required=True)
    parser.add_argument("--official_eval_script", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, default=Path("experiments"))
    parser.add_argument("--aggregate_csv", type=Path, default=None)

    parser.add_argument("--base_train_file", type=Path, default=None)
    parser.add_argument("--train_mode", choices=MODES, default="full")
    parser.add_argument(
        "--existing_model_dir",
        type=Path,
        default=None,
        help="Reuse an already-trained model and skip training.",
    )

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prediction_batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--postprocess_mode", choices=POSTPROCESS_MODES, default="none")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required_paths = [
        args.validation_file,
        args.gold_dir,
        args.drop_eval,
        args.make_official_script,
        args.official_eval_script,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    script_dir = Path(__file__).resolve().parent
    experiment_dir = args.work_dir / args.experiment_name

    if experiment_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Experiment directory already exists: {experiment_dir}. "
                "Use --overwrite or a new experiment name."
            )
        shutil.rmtree(experiment_dir)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    data_dir = experiment_dir / "data"
    model_dir = experiment_dir / "model"
    prediction_dir = experiment_dir / "predictions"
    official_predictions_dir = experiment_dir / "official_predictions"
    official_results_dir = experiment_dir / "official_results"
    data_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    if args.existing_model_dir is not None:
        if not args.existing_model_dir.exists():
            raise FileNotFoundError(args.existing_model_dir)
        selected_model_dir = args.existing_model_dir
    else:
        if args.base_train_file is None:
            parser.error("--base_train_file is required unless --existing_model_dir is used.")
        if not args.base_train_file.exists():
            raise FileNotFoundError(args.base_train_file)

        experiment_train = data_dir / "train.jsonl"
        run(
            [
                sys.executable,
                str(script_dir / "prepare_experiment_data.py"),
                "--input_file",
                str(args.base_train_file),
                "--output_file",
                str(experiment_train),
                "--mode",
                args.train_mode,
                "--seed",
                str(args.seed),
            ]
        )

        train_command = [
            sys.executable,
            str(script_dir / "train_t5_experiments.py"),
            "--train_file",
            str(experiment_train),
            "--validation_file",
            str(args.validation_file),
            "--output_dir",
            str(model_dir),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--learning_rate",
            str(args.learning_rate),
            "--early_stopping_patience",
            str(args.early_stopping_patience),
            "--precision",
            args.precision,
            "--seed",
            str(args.seed),
            "--overwrite_output_dir",
        ]
        if args.eval_batch_size is not None:
            train_command.extend(["--eval_batch_size", str(args.eval_batch_size)])
        run(train_command)
        selected_model_dir = model_dir

    raw_predictions = prediction_dir / "raw_predictions.jsonl"
    run(
        [
            sys.executable,
            str(script_dir / "predict_t5_experiments.py"),
            "--model_dir",
            str(selected_model_dir),
            "--input_file",
            str(args.validation_file),
            "--output_file",
            str(raw_predictions),
            "--batch_size",
            str(args.prediction_batch_size),
            "--num_beams",
            str(args.num_beams),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--length_penalty",
            str(args.length_penalty),
            "--seed",
            str(args.seed),
        ]
    )

    predictions_for_official = raw_predictions
    if args.postprocess_mode != "none":
        processed_predictions = prediction_dir / "processed_predictions.jsonl"
        run(
            [
                sys.executable,
                str(script_dir / "postprocess_predictions.py"),
                "--gold_file",
                str(args.validation_file),
                "--predictions_file",
                str(raw_predictions),
                "--output_file",
                str(processed_predictions),
                "--mode",
                args.postprocess_mode,
            ]
        )
        predictions_for_official = processed_predictions

    run(
        [
            sys.executable,
            str(args.make_official_script),
            "--input_file",
            str(predictions_for_official),
            "--gold_dir",
            str(args.gold_dir),
            "--output_dir",
            str(official_predictions_dir),
        ]
    )

    run(
        [
            sys.executable,
            str(args.official_eval_script),
            "--drop_eval",
            str(args.drop_eval),
            "--gold_dir",
            str(args.gold_dir),
            "--predictions_dir",
            str(official_predictions_dir),
            "--results_dir",
            str(official_results_dir),
        ]
    )

    aggregate_path = args.aggregate_csv or (args.work_dir / "controlled_results.csv")
    append_results(
        official_results_dir / "summary.json",
        aggregate_path,
        {
            "experiment": args.experiment_name,
            "train_mode": args.train_mode if args.existing_model_dir is None else "reused_model",
            "model_dir": str(selected_model_dir),
            "num_beams": args.num_beams,
            "postprocess_mode": args.postprocess_mode,
            "seed": args.seed,
        },
    )

    run_metadata = {
        "experiment_name": args.experiment_name,
        "experiment_dir": str(experiment_dir),
        "model_dir": str(selected_model_dir),
        "train_mode": args.train_mode,
        "existing_model_reused": args.existing_model_dir is not None,
        "num_beams": args.num_beams,
        "postprocess_mode": args.postprocess_mode,
        "seed": args.seed,
        "aggregate_csv": str(aggregate_path),
    }
    with (experiment_dir / "experiment_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(run_metadata, file, indent=2)

    print("\nExperiment completed.")
    print(f"Results: {official_results_dir / 'summary.csv'}")
    print(f"Aggregate results: {aggregate_path}")


if __name__ == "__main__":
    main()
