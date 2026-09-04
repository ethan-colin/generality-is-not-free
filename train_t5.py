import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM,AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,Seq2SeqTrainingArguments)


MODEL_NAME = "google-t5/t5-small"

def main() -> None:
    parser = argparse.ArgumentParser( description="Train T5-small on the unified DROP and QUOREF data.")

    parser.add_argument("--train_file", type=str, required=True, help="Path to unified_train.jsonl")
    parser.add_argument( "--validation_file", type=str, required=True, help="Path to unified_validation.jsonl")
    parser.add_argument( "--output_dir", type=str, default="models/t5_multispan", help="Directory in which the trained model will be saved")
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument( "--batch_size", type=int, default=2 )
    parser.add_argument( "--max_input_length", type=int, default=512 )
    parser.add_argument( "--max_target_length", type=int, default=64 )
    parser.add_argument( "--gradient_accumulation_steps", type=int, default=4)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading JSONL files...")

    dataset = load_dataset( "json", data_files={ "train": args.train_file, "validation": args.validation_file } )

    print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained( MODEL_NAME )

    # Display the device that will be used for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "WARNING: CUDA was not detected. "
            "Training will run on the CPU and may be very slow."
        )


    def tokenize_batch(batch):
        """
        Convert input_text and target_text into token IDs.
        """
        model_inputs = tokenizer(batch["input_text"], max_length=args.max_input_length, truncation=True )
        labels = tokenizer( text_target=batch["target_text"], max_length=args.max_target_length, truncation=True, )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing data...")

    tokenized_dataset = dataset.map( tokenize_batch, batched=True, remove_columns=dataset["train"].column_names)

    data_collator = DataCollatorForSeq2Seq( tokenizer=tokenizer, model=model )

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),

        # Training configuration
        num_train_epochs=args.epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        # Four small batches behave like one larger batch.
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        train_sampling_strategy="group_by_length",

        # Evaluate and save after every epoch.
        eval_strategy="epoch",
        save_strategy="epoch",

        # Keep only the two most recent checkpoints.
        save_total_limit=2,

        # At the end, restore the checkpoint with lowest validation loss.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=100,
        seed=42,

        # Use half precision when training on an NVIDIA GPU.
        fp16=torch.cuda.is_available(),

        # Prevent integrations such as Weights & Biases from starting.
        report_to="none",

        # Safer default on Windows.
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")

    trainer.train()

    print("Saving final model...")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()


