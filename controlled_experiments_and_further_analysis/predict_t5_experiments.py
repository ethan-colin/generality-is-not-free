"""Generate deterministic T5 predictions with configurable decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from qa_utils import ANSWER_SEPARATOR, iter_jsonl


def split_prediction(prediction_text: str) -> List[str]:
    return [
        answer.strip()
        for answer in prediction_text.split(ANSWER_SEPARATOR)
        if answer.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate predictions with a trained T5 model."
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)
    if not args.input_file.exists():
        raise FileNotFoundError(args.input_file)
    if args.num_beams < 1:
        raise ValueError("--num_beams must be at least 1.")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    examples = list(iter_jsonl(args.input_file))
    print(f"Loaded {len(examples)} examples.")
    print(
        "Decoding: "
        f"beams={args.num_beams}, "
        f"length_penalty={args.length_penalty}, "
        f"repetition_penalty={args.repetition_penalty}, "
        f"no_repeat_ngram_size={args.no_repeat_ngram_size}"
    )

    seen_ids = set()
    with args.output_file.open("w", encoding="utf-8") as output_file:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            input_texts = [str(example["input_text"]) for example in batch]

            encoded_inputs = tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
                return_tensors="pt",
            )
            encoded_inputs = {
                name: tensor.to(device)
                for name, tensor in encoded_inputs.items()
            }

            generation_kwargs: Dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "num_beams": args.num_beams,
                "length_penalty": args.length_penalty,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
            }
            if args.num_beams > 1:
                generation_kwargs["early_stopping"] = True

            with torch.inference_mode():
                generated_ids = model.generate(
                    **encoded_inputs,
                    **generation_kwargs,
                )

            prediction_texts = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            for example, prediction_text in zip(batch, prediction_texts):
                example_id = str(example.get("id", ""))
                if not example_id:
                    raise ValueError("An input example has an empty ID.")
                if example_id in seen_ids:
                    raise ValueError(f"Duplicate input ID: {example_id}")
                seen_ids.add(example_id)

                prediction_text = prediction_text.strip()
                record = {
                    "id": example_id,
                    "dataset": example.get("dataset", ""),
                    "prediction_text": prediction_text,
                    "predicted_answers": split_prediction(prediction_text),
                    "generation": {
                        "num_beams": args.num_beams,
                        "max_new_tokens": args.max_new_tokens,
                        "length_penalty": args.length_penalty,
                        "repetition_penalty": args.repetition_penalty,
                        "no_repeat_ngram_size": args.no_repeat_ngram_size,
                        "seed": args.seed,
                    },
                }
                json.dump(record, output_file, ensure_ascii=False)
                output_file.write("\n")

            completed = min(start + args.batch_size, len(examples))
            print(f"\rGenerated {completed}/{len(examples)}", end="")

    print()
    print(f"Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
