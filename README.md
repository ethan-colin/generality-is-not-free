# Generality Is Not Free: Task-Specific Constraints versus Text Generation for Multi-Span Question Answering

Note: we utilized an LLM to create a README based on our own custom initial draft our README and report, we found many of its suggestions very helpful and assisting to "steer in the fog" so we verified them and kept them. We re-utilized the LLM for further suggestions and corrections in this README after we finished writing the paper but we removed or heavily modified what was too "grandiose" or blatantly innacurate and kept the validated and constructive propositions.
The scripts were vibe-coded and have gone through several iterations after correctness and validation checks to get to the correct behavior.

_______________________________________


The project compares a small general-purpose encoder-decoder model, `google-t5/t5-small`, with the more task-specific extractive formulation studied by Segal et al. (2020), *A Simple and Effective Model for Answering Multi-span Questions*.

The experiments use:

- **DROP**, restricted to examples whose answers contain textual spans. Number-only and date-only answers are excluded.
- **QUOREF**, using its extractive answer annotations.
- A unified text-to-text format:
  - input: `question: <question> context: <passage>`
  - target: answer strings separated by ` ||| `

The main reported T5 configuration uses deterministic greedy decoding unless otherwise stated.

The project pipeline is:

1. Load span-answer examples from DROP and extractive examples from QUOREF.
2. Convert both datasets to a unified JSONL representation.
3. Serialize each target as `a1 ||| a2 ||| ...`.
4. Fine-tune T5-small jointly on the unified training records.
5. Generate predictions using greedy decoding by default and split the generated string on `|||`.
6. Convert predictions to the evaluation representation and compute EM/F1.

The commands below reproduce this pipeline using the directory layout shown in the next section. Additional environment and reproducibility details are provided at the end of the README.



---

## 1. Repository structure

The important files are expected to be arranged roughly as follows:

```text
.
├── prepare_unified_data.py
├── train_t5.py
├── predict_t5.py
├── prepare_official_gold.py
├── make_official_predictions.py
├── run_official_evaluation.py
├── drop_eval.py
├── audit_all_datasets.py
├── requirements.txt
│
├── controlled_experiments_and_further_analysis/
│   ├── make_sanity_subset.py
│   ├── run_sanity_overfit_official.py
│   ├── train_t5_experiments.py
│   ├── predict_t5_experiments.py
│   ├── prepare_experiment_data.py
│   ├── postprocess_predictions.py
│   ├── run_controlled_experiment.py
│   ├── analyze_official_errors.py
│   └── qa_utils.py
│
├── data/
│   ├── drop_dataset_train.json
│   ├── drop_dataset_dev.json
│   ├── quoref-train-v0.1.json
│   └── quoref-dev-v0.1.json
```

Some output directories are created automatically by the scripts.

All commands below are run from the project root.

---

## 2. Environment setup

The experiments were run with Python 3.12. A GPU is strongly recommended for full training re-creation.

Commands below use `python`. If your environment exposes Python as `python3`, substitute to `python3`.

Install the packages used by the project:

```bash
pip install -r requirements.txt
```

---

## 3. Obtain the datasets

Note: QUOREF input format. The preprocessing script expects the original SQuAD-style QUOREF format, in which examples are stored under data → paragraphs → qas → answers. Therefore, the experiments use quoref-train-v0.1.json (and the corresponding development file). Another available QUOREF representation, such as quoref_dataset_train.json, contains the same underlying examples but uses a DROP-like structure with top-level passage IDs, qa_pairs, and answer["spans"], and is therefore not directly compatible with prepare_unified_data.py.

We used the DROP and QUOREF files distributed through the repository associated with Segal et al. (2020). The commands below assume the following paths:

```text
data/drop_dataset_train.json
data/drop_dataset_dev.json
data/quoref-train-v0.1.json
data/quoref-dev-v0.1.json
```

That paper's source repository is:

```text
https://github.com/eladsegal/tag-based-multi-span-extraction/tree/master
```

If your filenames differ, change the paths in the commands below.

The expected formats are:

- **DROP**: the original DROP JSON structure with passages and `qa_pairs`.
- **QUOREF**: the original SQuAD-style JSON structure with `data -> paragraphs -> qas`.

---

## 4. Preprocessing: prepare the unified training and validation data

`prepare_unified_data.py` converts both datasets into the common JSONL format used by the T5 model.

### Training split

```bash
python prepare_unified_data.py --drop data/drop_dataset_train.json --quoref data/quoref-train-v0.1.json --split train --output processed/unified_train.jsonl
```

### Validation split

```bash
python prepare_unified_data.py --drop data/drop_dataset_dev.json --quoref data/quoref-dev-v0.1.json --split validation --output processed/unified_validation.jsonl
```

The generated files are written to `processed/`.

Each JSONL record contains fields such as:

```json
{
  "id": "drop:<question-id>",
  "dataset": "drop",
  "split": "train",
  "question": "...",
  "context": "...",
  "input_text": "question: ... context: ...",
  "answers": ["answer 1", "answer 2"],
  "target_text": "answer 1 ||| answer 2",
  "answer_type": "multi_span",
  "num_answers": 2
}
```

For DROP, only examples with non-empty textual span answers are retained. Number-only and date-only examples are skipped.

The final project data contained:

```text
Training records:   48,594
Validation records:  5,947
```

---

## 5. Optional but recommended: sanity check before full training

Before training the full model, run the small overfitting experiment below. This checks that preprocessing, target formatting, training, generation, conversion, and evaluation are connected correctly.

```bash
python controlled_experiments_and_further_analysis/run_sanity_overfit_official.py \
  --base_train_file processed/unified_train.jsonl \
  --drop_eval drop_eval.py \
  --work_dir experiments/sanity_overfit \
  --size 80 \
  --epochs 30 \
  --batch_size 8 \
  --learning_rate 0.0003 \
  --overwrite
```

To evaluate an already trained sanity model without retraining:

```bash
python controlled_experiments_and_further_analysis/run_sanity_overfit_official.py \
  --evaluation_only \
  --drop_eval drop_eval.py \
  --work_dir experiments/sanity_overfit
```

The results are written to `experiments/sanity_overfit/`.

For the sanity experiment, the script builds an 80-example balanced subset covering both datasets and both answer types. The model is trained directly on this subset and then scored on it again, so the experiment serves as a basic overfitting check for the end-to-end pipeline.

Because the model is trained and evaluated on the same tiny subset, the obtained scores are expected to be very high. A low score would've likely indicated a pipeline problem rather than a generalization problem.

---

## 6. Train the final T5-small model

Train jointly on the unified DROP + QUOREF training data:

```bash
python train_t5.py --train_file processed/unified_train.jsonl --validation_file processed/unified_validation.jsonl --output_dir models/t5_multispan --epochs 3 --batch_size 2 --gradient_accumulation_steps 4
```

Important settings in `train_t5.py` include:

```text
Base model:                   google-t5/t5-small
Maximum input length:         512
Maximum target length:         64
Learning rate:               5e-5
Epochs:                         3
Batch size per device:          2
Gradient accumulation:          4
Effective batch size:           8
Seed:                          42
```

The script automatically uses CUDA when available and otherwise falls back to CPU.

After training, the model and tokenizer are saved under:

```text
models/t5_multispan/
```

---

## 7. Generate predictions with the final greedy model

The simple prediction script uses deterministic greedy decoding (`num_beams=1`, no sampling):

```bash
python predict_t5.py --model_dir models/t5_multispan --input_file processed/unified_validation.jsonl --output_file predictions/validation_predictions.jsonl --batch_size 4
```

Each output row contains the original example ID, generated text, and the parsed answer list.

Example:

```json
{
  "id": "quoref:<question-id>",
  "dataset": "quoref",
  "prediction_text": "Barack Obama ||| George W. Bush",
  "predicted_answers": ["Barack Obama", "George W. Bush"]
}
```

---

## 8. Prepare evaluation gold files

The project evaluates both datasets with the official DROP-style evaluator . In order for this to be done ,we create the all-span, single-span, and multi-span gold subsets from the development/validation datasets, and also parse the QUOREF data into DROP format to be used by the official evaluator for later.

```bash
python prepare_official_gold.py --drop_dev data/drop_dataset_dev.json --quoref_dev data/quoref-dev-v0.1.json --output_dir official_gold
```

This creates:

```text
official_gold/
├── drop_all_spans.json
├── drop_single_span.json
├── drop_multi_span.json
├── quoref_all_spans.json
├── quoref_single_span.json
└── quoref_multi_span.json
```

Note: As we said, these are used by the project to later evaluate both datasets with the official DROP evaluation implementation which is 'drop_eval.py' ( also available from https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py ). DROP is evaluated in its native representation, while QUOREF is first converted to the corresponding DROP-style representation. This is the evaluation procedure used for all T5 inference conditions in this work.

---

## 9. Convert generated predictions to the evaluation format

Convert the model's JSONL predictions into the per-question JSON dictionaries as DROP format, which is expected by the official evaluator:

```bash
python make_official_predictions.py --input_file predictions/validation_predictions.jsonl --gold_dir official_gold --output_dir official_predictions
```

This creates:

```text
official_predictions/drop_predictions.json
official_predictions/quoref_predictions.json
```

Note: section 8 processes the ground-truth/reference answers, while section 9 processes the model's generated answers. These are to be used by the official DROP evaluator.

---

## 10. Run evaluation

At this point, the pipeline has two corresponding components: the reference answers prepared in Section 8 and the model predictions prepared in Section 9. run_official_evaluation.py compares them to compute EM and F1.

Run the evaluator on all six subsets:

```bash
python run_official_evaluation.py --drop_eval drop_eval.py --gold_dir official_gold --predictions_dir official_predictions --results_dir official_results
```

The script evaluates:

```text
DROP   - All Spans
DROP   - Single Span
DROP   - Multi Span
QUOREF - All Spans
QUOREF - Single Span
QUOREF - Multi Span
```

It writes per-subset metric files as well as:

```text
official_results/summary.json
official_results/summary.csv
```

 Results may differ because if the underlying checkpoint, processed validation file, evaluation files, model initialization, training hardware, running on CPU or GPU, or package versions differ or scripts may not be identical even when the relative path names are the same.

---

## 11. Reproduce the three controlled experiments

The controlled experiments all reuse the same trained model in `models/t5_multispan`. They change decoding or post-processing only.

Before running them, make sure the following files and directories exist:

```text
models/t5_multispan/
official_gold/
drop_eval.py
make_official_predictions.py
run_official_evaluation.py
processed/unified_validation.jsonl
```

### 11.1 Greedy baseline

```bash
python controlled_experiments_and_further_analysis/run_controlled_experiment.py --experiment_name baseline_greedy --existing_model_dir models/t5_multispan --validation_file processed/unified_validation.jsonl --gold_dir official_gold --drop_eval drop_eval.py --make_official_script make_official_predictions.py --official_eval_script run_official_evaluation.py --num_beams 1 --work_dir experiments --overwrite
```

Note: results for the greedy baseline will be the same as in section 10, but we wrote this here anyway for the sake of completeness.


### 11.2 Beam-search experiment

```bash
python controlled_experiments_and_further_analysis/run_controlled_experiment.py --experiment_name beam4 --existing_model_dir models/t5_multispan --validation_file processed/unified_validation.jsonl --gold_dir official_gold --drop_eval drop_eval.py --make_official_script make_official_predictions.py --official_eval_script run_official_evaluation.py --num_beams 4 --work_dir experiments --overwrite
```



### 11.3 Generate-then-ground experiment

This condition uses greedy decoding followed by deduplication and removal of answer items that do not match the passage.

```bash
python controlled_experiments_and_further_analysis/run_controlled_experiment.py --experiment_name grounded_filter --existing_model_dir models/t5_multispan --validation_file processed/unified_validation.jsonl --gold_dir official_gold --drop_eval drop_eval.py --make_official_script make_official_predictions.py --official_eval_script run_official_evaluation.py --num_beams 1 --postprocess_mode deduplicate_and_drop_ungrounded --work_dir experiments --overwrite
```


Each controlled experiment creates its own directory, for example:

```text
experiments/baseline_greedy/
experiments/beam4/
experiments/grounded_filter/
```

The runner also maintains:

```text
experiments/controlled_results.csv
```

---

## 12. Official error analysis

After producing the evaluation predictions, analyze DROP errors:

```bash
python controlled_experiments_and_further_analysis/analyze_official_errors.py --gold_file official_gold/drop_all_spans.json --predictions_file official_predictions/drop_predictions.json --drop_eval drop_eval.py --dataset_name DROP --output_dir analysis/errors/drop
```

Analyze QUOREF errors:

```bash
python controlled_experiments_and_further_analysis/analyze_official_errors.py --gold_file official_gold/quoref_all_spans.json --predictions_file official_predictions/quoref_predictions.json --drop_eval drop_eval.py --dataset_name QUOREF --output_dir analysis/errors/quoref
```

The analysis groups examples into categories including:

```text
correct
wrong_answer
boundary_or_partial_match
too_few_spans
too_many_spans
duplicate_prediction
merged_gold_spans
ungrounded_prediction
```

---

## 13. Reproducibility notes

### Random seed

The main training code uses seed `42`, and the experimental scripts use deterministic decoding unless explicitly changed. The reported full-model results are from a single training seed, so retraining from initialization may produce somewhat different scores if trained on a GPU.

Repeated greedy decoding of the same saved checkpoint with the same tokenizer, inputs, generation code, and evaluation artifacts should reproduce the same predictions.

### CPU vs GPU

Training on CPU is possible but very slow. A CUDA-capable GPU is strongly recommended for the full model. Prediction and evaluation can also run on CPU.

### Model download

`google-t5/t5-small` is loaded through Hugging Face Transformers. Here's the current link ( as of now ): https://huggingface.co/google-t5/t5-small


### Dataset paths

The dataset filenames in this README are examples. The scripts do not require those exact names, they only require valid paths to the original QUOREF and DROP JSON files.

### Package versions

The experiments were run with Python 3.12. For the most reproducible setup, use the package versions recorded in `requirements.txt`.

### Evaluation protocol

To re-emphasize: Segal et al. (2020) report using the dataset-specific official evaluation scripts for DROP and QUOREF. In this repository, both T5 datasets are scored through the official DROP evaluation implementation, with QUOREF converted to a DROP-style representation first. This protocol difference should be kept in mind when comparing QUOREF scores directly with the published Segal et al. baselines.
We re-provide the link of the official evaluator again ( even though we did so above ), for the sake of completeness: https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py
