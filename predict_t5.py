import argparse
import json
from pathlib import Path

import torch
from transformers import ( AutoModelForSeq2SeqLM, AutoTokenizer )

ANSWER_SEPARATOR = " ||| "

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError( f"Invalid JSON on line {line_number} of {path}" ) from error


def split_prediction(prediction_text: str):
    """
    Convert:

        Barack Obama ||| George W. Bush

    into:

        ["Barack Obama", "George W. Bush"]
    """

    return [ answer.strip() for answer in prediction_text.split(ANSWER_SEPARATOR) if answer.strip() ]


def main() -> None:
    parser = argparse.ArgumentParser( description="Generate predictions with the trained T5 model." )

    parser.add_argument( "--model_dir", type=Path, required=True, help="Directory containing the trained model" )
    parser.add_argument( "--input_file", type=Path, required=True, help="Unified validation or test JSONL file" )
    parser.add_argument( "--output_file", type=Path, required=True, help="Output predictions JSONL file" )
    parser.add_argument( "--batch_size", type=int, default=4 )
    parser.add_argument( "--max_input_length", type=int, default=512 )
    parser.add_argument( "--max_new_tokens", type=int, default=64 )

    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError( f"Model directory does not exist: {args.model_dir}")

    if not args.input_file.exists():
        raise FileNotFoundError( f"Input file does not exist: {args.input_file}" )

    args.output_file.parent.mkdir( parents=True, exist_ok=True )

    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

    print(f"Using device: {device}")
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained( args.model_dir )

    model = AutoModelForSeq2SeqLM.from_pretrained( args.model_dir )

    model.to(device)
    model.eval()

    examples = list(read_jsonl(args.input_file))

    print(f"Loaded {len(examples)} examples.")

    with args.output_file.open( "w", encoding="utf-8" ) as output_file:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]

            input_texts = [ example["input_text"] for example in batch ]

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

            with torch.no_grad():
                generated_ids = model.generate(
                    **encoded_inputs,
                    max_new_tokens=args.max_new_tokens,

                    # Deterministic generation.
                    do_sample=False,

                    # Simple greedy decoding.
                    num_beams=1,
                )

            prediction_texts = tokenizer.batch_decode( generated_ids, skip_special_tokens=True )

            for example, prediction_text in zip( batch, prediction_texts ):
                prediction_text = prediction_text.strip()

                prediction_record = {
                    "id": example["id"],
                    "dataset": example["dataset"],
                    "prediction_text": prediction_text,
                    "predicted_answers": split_prediction(
                        prediction_text
                    ),
                }

                json.dump( prediction_record, output_file, ensure_ascii=False )
                output_file.write("\n")

            completed = min( start + args.batch_size, len(examples) )

            print( f"\rGenerated {completed}/{len(examples)}", end="" )

    print()
    print(f"Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()