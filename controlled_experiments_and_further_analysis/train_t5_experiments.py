"""Train T5-small with reproducible settings for baseline and controlled runs."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


def build_training_arguments(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    """Build arguments while tolerating Transformers version differences."""
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    supported = set(signature.parameters)

    kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size or args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": "none",
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "group_by_length": True,
        "predict_with_generate": False,
        "save_safetensors": True,
    }

    if "overwrite_output_dir" in supported:
        kwargs["overwrite_output_dir"] = args.overwrite_output_dir

    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        raise TypeError(
            "This Transformers version exposes neither eval_strategy nor "
            "evaluation_strategy in Seq2SeqTrainingArguments."
        )

    precision = args.precision
    if precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            kwargs["bf16"] = True
        elif torch.cuda.is_available():
            kwargs["fp16"] = True
    elif precision == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested, but this GPU does not support it.")
        kwargs["bf16"] = True
    elif precision == "fp16":
        if not torch.cuda.is_available():
            raise RuntimeError("fp16 was requested, but CUDA is unavailable.")
        kwargs["fp16"] = True
    elif precision != "fp32":
        raise ValueError(f"Unknown precision: {precision}")

    # Transformers occasionally removes or renames optional arguments.
    # Keep only parameters accepted by the installed version rather than
    # failing on a harmless compatibility difference.
    unsupported = sorted(key for key in kwargs if key not in supported)
    if unsupported:
        print(
            "Compatibility note: ignoring unsupported "
            "Seq2SeqTrainingArguments fields: "
            + ", ".join(unsupported)
        )
        kwargs = {key: value for key, value in kwargs.items() if key in supported}

    return Seq2SeqTrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a reproducible T5 model on unified multi-span QA JSONL."
    )
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--validation_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="google-t5/t5-small")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    args = parser.parse_args()

    for path in (args.train_file, args.validation_file):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print("Loading JSONL files...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_file),
            "validation": str(args.validation_file),
        },
    )

    required_columns = {"input_text", "target_text"}
    for split_name in ("train", "validation"):
        missing = required_columns - set(dataset[split_name].column_names)
        if missing:
            raise ValueError(
                f"{split_name} is missing required columns: {sorted(missing)}"
            )

    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    print(f"Loading tokenizer and model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA was not detected; training on CPU will be slow.")

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing data...")
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )
    training_arguments = build_training_arguments(args)

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_arguments,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": data_collator,
        "callbacks": callbacks,
    }

    # `processing_class` is the newer name; older releases use `tokenizer`.
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    trainer_supported = set(trainer_signature.parameters)
    if "processing_class" in trainer_supported:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_supported:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    run_config = {
        "model_name": args.model_name,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size or args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_train_batch_size_per_device": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "seed": args.seed,
        "precision": args.precision,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    with (args.output_dir / "run_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(run_config, file, indent=2)

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving final/best model...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.save_state()

    final_metrics = trainer.evaluate()
    with (args.output_dir / "final_eval_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in final_metrics.items()
            },
            file,
            indent=2,
        )

    print(f"Model saved to: {args.output_dir}")
    print(f"Final validation loss: {final_metrics.get('eval_loss')}")


if __name__ == "__main__":
    main()
