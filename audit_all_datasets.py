#!/usr/bin/env python3
"""
Unified audit for the processed DROP and QUOREF data used by the project.

The script audits BOTH datasets with the same checks and the same summary
wording as much as possible.

Common checks for DROP and QUOREF:
    - record counts
    - single-span / multi-span counts
    - required fields and field types
    - dataset and split values
    - input_text format
    - target_text format
    - num_answers / answer_type consistency
    - duplicate IDs within a split
    - train/validation ID leakage
    - train/validation question+context leakage
    - gold-answer grounding in the full context
    - source-token truncation
    - whether at least one gold answer is lost after source truncation
    - target-token truncation

Optional source-consistency checks:
    - DROP: compare the processed records to the original DROP JSON files
    - QUOREF: compare the processed records to the original SQuAD-style
      QUOREF JSON files

Recommended command:

python audit_all_datasets.py `
  --train_file processed/unified_train.jsonl `
  --validation_file processed/unified_validation.jsonl `
  --drop_train_json data/drop_dataset_train.json `
  --drop_validation_json data/drop_dataset_dev.json `
  --quoref_train_json <PATH_TO_QUOREF_TRAIN_JSON> `
  --quoref_validation_json <PATH_TO_QUOREF_VALIDATION_JSON> `
  --report_dir reports/data_audit `
  --tokenizer_name google-t5/t5-small `
  --max_source_length 512 `
  --max_target_length 64

If you do not want to compare to the original source files, omit the
corresponding --drop_*_json / --quoref_*_json arguments.

Outputs:
    <report_dir>/data_audit_summary.json
    <report_dir>/data_audit_issues.csv
    <report_dir>/data_audit_examples.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


TARGET_SEPARATOR = " ||| "
DATASETS = ("drop", "quoref")

REQUIRED_FIELDS = {
    "dataset",
    "split",
    "id",
    "question",
    "context",
    "answers",
    "input_text",
    "target_text",
}


@dataclass
class Issue:
    severity: str
    dataset: str
    split: str
    line_number: int
    example_id: str
    issue: str
    details: str


def normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_answer(text: Any) -> str:
    """
    Project-style normalized grounding check:
      - lowercase
      - remove ASCII punctuation
      - remove English articles
      - collapse whitespace

    This is intentionally a grounding diagnostic, not the official
    DROP/QUOREF evaluation metric.
    """
    value = str(text).lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_is_grounded(answer: Any, context: Any) -> bool:
    normalized_answer = normalize_answer(answer)
    normalized_context = normalize_answer(context)

    if not normalized_answer:
        return False

    return normalized_answer in normalized_context


def count_normalized_occurrences(answer: Any, context: Any) -> int:
    normalized_answer = normalize_answer(answer)
    normalized_context = normalize_answer(context)

    if not normalized_answer:
        return 0

    return normalized_context.count(normalized_answer)


def clean_answer(answer: Any) -> str:
    return str(answer).strip()


def canonical_answer(answer: Any) -> str:
    return normalize_whitespace(answer).casefold()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path, expected_split: str) -> tuple[list[dict[str, Any]], list[Issue]]:
    records: list[dict[str, Any]] = []
    issues: list[Issue] = []

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                issues.append(
                    Issue(
                        severity="WARNING",
                        dataset="unknown",
                        split=expected_split,
                        line_number=line_number,
                        example_id="",
                        issue="blank_line",
                        details=f"Blank line in {path}",
                    )
                )
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(
                    Issue(
                        severity="ERROR",
                        dataset="unknown",
                        split=expected_split,
                        line_number=line_number,
                        example_id="",
                        issue="invalid_json",
                        details=str(exc),
                    )
                )
                continue

            if not isinstance(obj, dict):
                issues.append(
                    Issue(
                        severity="ERROR",
                        dataset="unknown",
                        split=expected_split,
                        line_number=line_number,
                        example_id="",
                        issue="record_is_not_object",
                        details=f"Found {type(obj).__name__}",
                    )
                )
                continue

            obj["_audit_line_number"] = line_number
            records.append(obj)

    return records, issues


def add_issue(
    issues: list[Issue],
    severity: str,
    dataset: str,
    split: str,
    record: dict[str, Any] | None,
    issue: str,
    details: str,
) -> None:
    issues.append(
        Issue(
            severity=severity,
            dataset=dataset,
            split=str(record.get("split", split)) if record else split,
            line_number=int(record.get("_audit_line_number", -1)) if record else -1,
            example_id=str(record.get("id", "")) if record else "",
            issue=issue,
            details=details,
        )
    )


def get_dataset_records(
    records: Iterable[dict[str, Any]],
    dataset: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if str(record.get("dataset", "")).strip().casefold() == dataset
    ]


def valid_answers(values: Iterable[Any]) -> list[str]:
    answers: list[str] = []

    for value in values:
        if isinstance(value, str) and value.strip():
            answers.append(value.strip())

    return answers


def expected_record(
    *,
    dataset: str,
    split: str,
    question_id: str,
    question: str,
    context: str,
    answers: list[str],
    passage_id: str = "",
    title: str = "",
) -> dict[str, Any]:
    question = str(question).strip()
    context = str(context)

    return {
        "id": f"{dataset}:{question_id}",
        "dataset": dataset,
        "split": split,
        "question": question,
        "context": context,
        "answers": answers,
        "input_text": f"question: {question} context: {context}",
        "target_text": TARGET_SEPARATOR.join(answers),
        "answer_type": "multi_span" if len(answers) > 1 else "single_span",
        "num_answers": len(answers),
        "passage_id": passage_id,
        "title": title,
    }


def normalize_question_for_duplicate_check(question: Any) -> str:
    """
    Mirrors the project's preprocessing duplicate comparison:
    ignore surrounding whitespace and a trailing slash.
    """
    return str(question).strip().rstrip("/").strip()


def expected_records_equivalent(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    return (
        first["dataset"] == second["dataset"]
        and first.get("passage_id", "") == second.get("passage_id", "")
        and first["context"] == second["context"]
        and first["answers"] == second["answers"]
        and normalize_question_for_duplicate_check(first["question"])
        == normalize_question_for_duplicate_check(second["question"])
    )


def insert_expected_record(
    expected: dict[str, dict[str, Any]],
    record: dict[str, Any],
    source_counts: Counter,
) -> None:
    example_id = record["id"]

    if example_id not in expected:
        expected[example_id] = record
        return

    if expected_records_equivalent(expected[example_id], record):
        source_counts["equivalent_duplicate_source_records"] += 1
        return

    raise ValueError(
        f"Conflicting source records have the same ID: {example_id}"
    )


def build_expected_drop(
    path: Path,
    split: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """
    Reconstruct the processed DROP span-only records using the same ID scheme
    as the latest project preprocessing:
        drop:<passage_id>:<original_query_id>:<qa_index>
    """
    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError(f"DROP source must be a JSON object: {path}")

    expected: dict[str, dict[str, Any]] = {}
    counts = Counter()

    for passage_id, passage_data in data.items():
        if not isinstance(passage_data, dict):
            continue

        context = str(passage_data.get("passage", ""))

        for qa_index, qa in enumerate(passage_data.get("qa_pairs", [])):
            if not isinstance(qa, dict):
                continue

            counts["source_questions"] += 1

            answer_object = qa.get("answer", {}) or {}
            spans = valid_answers(answer_object.get("spans", []))

            if not spans:
                continue

            counts["retained_questions"] += 1
            counts["multi_span" if len(spans) > 1 else "single_span"] += 1

            original_query_id = qa.get("query_id", f"question-{qa_index}")
            question_id = f"{passage_id}:{original_query_id}:{qa_index}"

            record = expected_record(
                dataset="drop",
                split=split,
                question_id=str(question_id),
                question=str(qa.get("question", "")),
                context=context,
                answers=spans,
                passage_id=str(passage_id),
                title="",
            )
            insert_expected_record(expected, record, counts)

    counts["expected_records_after_deduplication"] = len(expected)
    return expected, dict(counts)


def build_expected_quoref(
    path: Path,
    split: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """
    Reconstruct processed SQuAD-style QUOREF records using the same ID scheme
    as the project preprocessing:
        quoref:<question_id>
    """
    data = load_json(path)

    if not isinstance(data, dict) or "data" not in data:
        raise ValueError(
            "QUOREF source must be the original SQuAD-style JSON "
            f"with a top-level 'data' field: {path}"
        )

    expected: dict[str, dict[str, Any]] = {}
    counts = Counter()

    for article_index, article in enumerate(data.get("data", [])):
        if not isinstance(article, dict):
            continue

        title = str(article.get("title", ""))

        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            if not isinstance(paragraph, dict):
                continue

            context = str(paragraph.get("context", ""))

            for qa_index, qa in enumerate(paragraph.get("qas", [])):
                if not isinstance(qa, dict):
                    continue

                counts["source_questions"] += 1

                answer_objects = qa.get("answers", [])
                answers = valid_answers(
                    answer.get("text", "")
                    for answer in answer_objects
                    if isinstance(answer, dict)
                )

                if not answers:
                    continue

                counts["retained_questions"] += 1
                counts["multi_span" if len(answers) > 1 else "single_span"] += 1

                fallback_id = f"{article_index}-{paragraph_index}-{qa_index}"
                question_id = qa.get("id", fallback_id)

                record = expected_record(
                    dataset="quoref",
                    split=split,
                    question_id=str(question_id),
                    question=str(qa.get("question", "")),
                    context=context,
                    answers=answers,
                    passage_id="",
                    title=title,
                )
                insert_expected_record(expected, record, counts)

    counts["expected_records_after_deduplication"] = len(expected)
    return expected, dict(counts)


def audit_record(
    record: dict[str, Any],
    dataset: str,
    expected_split: str,
    issues: list[Issue],
) -> dict[str, Any]:
    missing_fields = sorted(REQUIRED_FIELDS - set(record))
    if missing_fields:
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "missing_required_fields",
            ", ".join(missing_fields),
        )

    actual_dataset = str(record.get("dataset", "")).strip().casefold()
    actual_split = str(record.get("split", "")).strip().casefold()
    example_id = record.get("id")
    question = record.get("question")
    context = record.get("context")
    answers = record.get("answers")
    input_text = record.get("input_text")
    target_text = record.get("target_text")

    if actual_dataset != dataset:
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "wrong_dataset_value",
            f"Expected {dataset!r}, found {record.get('dataset')!r}",
        )

    if actual_split != expected_split.casefold():
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "wrong_split_value",
            f"Expected {expected_split!r}, found {record.get('split')!r}",
        )

    if not isinstance(example_id, str) or not example_id.strip():
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "invalid_id",
            "id must be a non-empty string",
        )

    if not isinstance(question, str) or not question.strip():
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "invalid_question",
            "question must be a non-empty string",
        )

    if not isinstance(context, str) or not context.strip():
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "invalid_context",
            "context must be a non-empty string",
        )

    cleaned_answers: list[str] = []

    if not isinstance(answers, list):
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "answers_not_list",
            f"Found {type(answers).__name__}",
        )
    else:
        cleaned_answers = [clean_answer(answer) for answer in answers]

        if not cleaned_answers:
            add_issue(
                issues,
                "ERROR",
                dataset,
                expected_split,
                record,
                "empty_answers",
                "answers list is empty",
            )

        for answer_index, answer in enumerate(cleaned_answers):
            if not answer:
                add_issue(
                    issues,
                    "ERROR",
                    dataset,
                    expected_split,
                    record,
                    "empty_answer_string",
                    f"answers[{answer_index}] is empty",
                )

        canonical_answers = [canonical_answer(answer) for answer in cleaned_answers]
        if canonical_answers and len(set(canonical_answers)) < len(canonical_answers):
            add_issue(
                issues,
                "WARNING",
                dataset,
                expected_split,
                record,
                "duplicate_answer_text",
                f"answers={cleaned_answers!r}",
            )

    # Match the project's preprocessing format:
    # input_text = "question: <question.strip()> context: <context>"
    if isinstance(question, str) and isinstance(context, str):
        expected_input = f"question: {question.strip()} context: {context}"

        if not isinstance(input_text, str):
            add_issue(
                issues,
                "ERROR",
                dataset,
                expected_split,
                record,
                "invalid_input_text",
                "input_text must be a string",
            )
        elif input_text != expected_input:
            if normalize_whitespace(input_text) != normalize_whitespace(expected_input):
                add_issue(
                    issues,
                    "ERROR",
                    dataset,
                    expected_split,
                    record,
                    "input_text_content_mismatch",
                    "input_text differs from 'question: <question> context: <context>'",
                )

    expected_target = TARGET_SEPARATOR.join(cleaned_answers)

    if not isinstance(target_text, str):
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "invalid_target_text",
            "target_text must be a string",
        )
    elif target_text != expected_target:
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "target_text_mismatch",
            f"Expected {expected_target!r}, found {target_text!r}",
        )

    expected_multispan = len(cleaned_answers) > 1
    expected_answer_type = "multi_span" if expected_multispan else "single_span"

    if "num_answers" in record and record.get("num_answers") != len(cleaned_answers):
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "wrong_num_answers",
            f"Expected {len(cleaned_answers)}, found {record.get('num_answers')!r}",
        )

    if "answer_type" in record and record.get("answer_type") != expected_answer_type:
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "wrong_answer_type",
            f"Expected {expected_answer_type!r}, found {record.get('answer_type')!r}",
        )

    if "is_multispan" in record and record.get("is_multispan") != expected_multispan:
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "wrong_is_multispan",
            f"Expected {expected_multispan}, found {record.get('is_multispan')!r}",
        )

    grounded_flags = (
        [answer_is_grounded(answer, context) for answer in cleaned_answers]
        if isinstance(context, str)
        else []
    )
    all_grounded = bool(grounded_flags) and all(grounded_flags)

    occurrence_counts = (
        [count_normalized_occurrences(answer, context) for answer in cleaned_answers]
        if isinstance(context, str)
        else []
    )
    any_repeated = any(count > 1 for count in occurrence_counts)

    if cleaned_answers and isinstance(context, str) and not all_grounded:
        missing_answers = [
            answer
            for answer, grounded in zip(cleaned_answers, grounded_flags)
            if not grounded
        ]
        add_issue(
            issues,
            "WARNING",
            dataset,
            expected_split,
            record,
            "gold_answer_not_found_in_full_context",
            f"Ungrounded answer(s): {missing_answers!r}",
        )

    if "all_answers_groundable" in record and isinstance(context, str):
        if record.get("all_answers_groundable") != all_grounded:
            add_issue(
                issues,
                "ERROR",
                dataset,
                expected_split,
                record,
                "wrong_all_answers_groundable",
                f"Expected {all_grounded}, found {record.get('all_answers_groundable')!r}",
            )

    # Validate preserved offsets for either dataset, if the processed schema has them.
    answer_starts = record.get("answer_starts")
    has_offsets = isinstance(answer_starts, list)

    if answer_starts is not None and not isinstance(answer_starts, list):
        add_issue(
            issues,
            "ERROR",
            dataset,
            expected_split,
            record,
            "answer_starts_not_list",
            f"Found {type(answer_starts).__name__}",
        )
    elif isinstance(answer_starts, list):
        if len(answer_starts) != len(cleaned_answers):
            add_issue(
                issues,
                "ERROR",
                dataset,
                expected_split,
                record,
                "answer_start_count_mismatch",
                f"{len(answer_starts)} starts for {len(cleaned_answers)} answers",
            )
        elif isinstance(context, str):
            for answer_index, (answer, start) in enumerate(
                zip(cleaned_answers, answer_starts)
            ):
                if not isinstance(start, int) or isinstance(start, bool):
                    add_issue(
                        issues,
                        "ERROR",
                        dataset,
                        expected_split,
                        record,
                        "invalid_answer_start",
                        f"answer_starts[{answer_index}]={start!r}",
                    )
                    continue

                if start < 0 or start + len(answer) > len(context):
                    add_issue(
                        issues,
                        "ERROR",
                        dataset,
                        expected_split,
                        record,
                        "answer_start_out_of_range",
                        (
                            f"answer={answer!r}, start={start}, "
                            f"context_length={len(context)}"
                        ),
                    )
                    continue

                extracted = context[start:start + len(answer)]
                if extracted != answer:
                    add_issue(
                        issues,
                        "ERROR",
                        dataset,
                        expected_split,
                        record,
                        "answer_offset_mismatch",
                        (
                            f"answer={answer!r}, start={start}, "
                            f"context_slice={extracted!r}"
                        ),
                    )

    return {
        "num_answers": len(cleaned_answers),
        "is_multispan": expected_multispan,
        "all_gold_answers_grounded": all_grounded,
        "any_gold_answer_repeated": any_repeated,
        "max_normalized_occurrences": max(occurrence_counts, default=0),
        "has_offsets": has_offsets,
    }


def audit_duplicate_ids(
    records: list[dict[str, Any]],
    dataset: str,
    split: str,
    issues: list[Issue],
) -> int:
    id_to_records: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        example_id = str(record.get("id", "")).strip()
        if example_id:
            id_to_records.setdefault(example_id, []).append(record)

    duplicate_id_count = 0

    for example_id, duplicate_records in id_to_records.items():
        if len(duplicate_records) <= 1:
            continue

        duplicate_id_count += 1

        for record in duplicate_records:
            add_issue(
                issues,
                "ERROR",
                dataset,
                split,
                record,
                "duplicate_id_within_split",
                f"id={example_id!r}, count={len(duplicate_records)}",
            )

    return duplicate_id_count


def audit_cross_split_leakage(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    dataset: str,
    issues: list[Issue],
) -> dict[str, int]:
    train_ids = {
        str(record.get("id", "")).strip()
        for record in train_records
        if str(record.get("id", "")).strip()
    }
    validation_ids = {
        str(record.get("id", "")).strip()
        for record in validation_records
        if str(record.get("id", "")).strip()
    }

    overlapping_ids = train_ids & validation_ids

    for record in validation_records:
        if str(record.get("id", "")).strip() in overlapping_ids:
            add_issue(
                issues,
                "ERROR",
                dataset,
                "validation",
                record,
                "id_overlap_between_splits",
                f"id={record.get('id')!r}",
            )

    def content_key(record: dict[str, Any]) -> tuple[str, str]:
        return (
            normalize_whitespace(record.get("question", "")).casefold(),
            normalize_whitespace(record.get("context", "")).casefold(),
        )

    train_content = {
        content_key(record)
        for record in train_records
        if record.get("question") and record.get("context")
    }

    overlapping_content = 0

    for record in validation_records:
        if not record.get("question") or not record.get("context"):
            continue

        if content_key(record) in train_content:
            overlapping_content += 1
            add_issue(
                issues,
                "ERROR",
                dataset,
                "validation",
                record,
                "question_context_overlap_between_splits",
                "The same normalized question and context appear in training",
            )

    return {
        "overlapping_ids": len(overlapping_ids),
        "overlapping_question_context_pairs": overlapping_content,
    }


def audit_tokenization(
    record: dict[str, Any],
    tokenizer,
    max_source_length: int,
    max_target_length: int,
) -> dict[str, Any]:
    question = str(record.get("question", ""))
    context = str(record.get("context", ""))
    answers = (
        [clean_answer(answer) for answer in record.get("answers", [])]
        if isinstance(record.get("answers"), list)
        else []
    )

    input_text = record.get("input_text")
    if not isinstance(input_text, str):
        input_text = f"question: {question.strip()} context: {context}"

    target_text = record.get("target_text")
    if not isinstance(target_text, str):
        target_text = TARGET_SEPARATOR.join(answers)

    full_input_ids = tokenizer(
        input_text,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )["input_ids"]

    target_ids = tokenizer(
        text_target=target_text,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )["input_ids"]

    # Reproduce the project's "keep the beginning of the passage" truncation.
    prefix = f"question: {question.strip()} context: "
    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )["input_ids"]
    context_ids = tokenizer(
        context,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )["input_ids"]

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    available_context_tokens = max(
        0,
        max_source_length - len(prefix_ids) - special_tokens,
    )
    retained_context_ids = context_ids[:available_context_tokens]

    retained_context = tokenizer.decode(
        retained_context_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    visible_flags = [
        answer_is_grounded(answer, retained_context)
        for answer in answers
    ]

    return {
        "input_token_count": len(full_input_ids),
        "target_token_count": len(target_ids),
        "input_truncated": len(full_input_ids) > max_source_length,
        "target_truncated": len(target_ids) > max_target_length,
        "available_context_tokens": available_context_tokens,
        "all_answers_visible_after_truncation": (
            bool(visible_flags) and all(visible_flags)
        ),
        "visible_answer_count_after_truncation": sum(visible_flags),
    }


def compare_to_source(
    records: list[dict[str, Any]],
    expected_by_id: dict[str, dict[str, Any]],
    source_counts: dict[str, int],
    dataset: str,
    split: str,
    issues: list[Issue],
) -> dict[str, Any]:
    unified_by_id: dict[str, dict[str, Any]] = {}

    for record in records:
        example_id = str(record.get("id", "")).strip()
        if example_id and example_id not in unified_by_id:
            unified_by_id[example_id] = record

    expected_ids = set(expected_by_id)
    unified_ids = set(unified_by_id)

    missing_ids = sorted(expected_ids - unified_ids)
    extra_ids = sorted(unified_ids - expected_ids)

    for example_id in missing_ids:
        expected = expected_by_id[example_id]
        add_issue(
            issues,
            "ERROR",
            dataset,
            split,
            None,
            "missing_from_processed",
            (
                f"id={example_id!r}; "
                f"question={expected.get('question', '')!r}"
            ),
        )

    for example_id in extra_ids:
        record = unified_by_id[example_id]
        add_issue(
            issues,
            "ERROR",
            dataset,
            split,
            record,
            "unexpected_processed_record",
            f"id={example_id!r}",
        )

    mismatch_count = 0

    fields_to_compare = (
        "question",
        "context",
        "answers",
        "answer_type",
        "num_answers",
        "passage_id",
        "title",
    )

    for example_id in sorted(expected_ids & unified_ids):
        expected = expected_by_id[example_id]
        record = unified_by_id[example_id]

        for field in fields_to_compare:
            if field not in record:
                continue

            actual_value = record.get(field)
            expected_value = expected.get(field)

            # Question is stripped during preprocessing.
            if field == "question":
                actual_value = str(actual_value).strip()
                expected_value = str(expected_value).strip()

            if actual_value != expected_value:
                mismatch_count += 1
                add_issue(
                    issues,
                    "ERROR",
                    dataset,
                    split,
                    record,
                    f"source_{field}_mismatch",
                    f"Expected {expected_value!r}, found {actual_value!r}",
                )

    return {
        "enabled": True,
        "source_questions": int(source_counts.get("source_questions", 0)),
        "source_retained_questions": int(source_counts.get("retained_questions", 0)),
        "expected_records": len(expected_by_id),
        "equivalent_duplicate_source_records": int(
            source_counts.get("equivalent_duplicate_source_records", 0)
        ),
        "missing_expected_records": len(missing_ids),
        "unexpected_records": len(extra_ids),
        "field_mismatches": mismatch_count,
        "missing_ids_preview": missing_ids[:100],
        "unexpected_ids_preview": extra_ids[:100],
    }


def summarize_split(
    records: list[dict[str, Any]],
    record_stats: list[dict[str, Any]],
    duplicate_id_count: int,
    token_stats: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    records_with_offsets = sum(item["has_offsets"] for item in record_stats)

    summary = {
        "records": len(records),
        "unique_ids": len(
            {
                str(record.get("id", "")).strip()
                for record in records
                if str(record.get("id", "")).strip()
            }
        ),
        "duplicate_ids": duplicate_id_count,
        "single_span_records": sum(not item["is_multispan"] for item in record_stats),
        "multi_span_records": sum(item["is_multispan"] for item in record_stats),
        "all_gold_answers_grounded": sum(
            item["all_gold_answers_grounded"] for item in record_stats
        ),
        "records_with_repeated_gold_answer": sum(
            item["any_gold_answer_repeated"] for item in record_stats
        ),
        "records_with_answer_starts": records_with_offsets,
        "records_without_answer_starts": len(records) - records_with_offsets,
    }

    if token_stats is None:
        summary["tokenization"] = {"enabled": False}
    else:
        summary["tokenization"] = {
            "enabled": True,
            "input_truncated": sum(item["input_truncated"] for item in token_stats),
            "examples_losing_answers": sum(
                item["input_truncated"]
                and not item["all_answers_visible_after_truncation"]
                for item in token_stats
            ),
            "target_truncated": sum(item["target_truncated"] for item in token_stats),
            "answers_lost_total": sum(
                max(
                    0,
                    int(record_stat["num_answers"])
                    - int(token_stat["visible_answer_count_after_truncation"]),
                )
                if token_stat["input_truncated"]
                else 0
                for record_stat, token_stat in zip(record_stats, token_stats)
            ),
        }

    return summary


def write_issues_csv(path: Path, issues: list[Issue]) -> None:
    fieldnames = [
        "severity",
        "dataset",
        "split",
        "line_number",
        "example_id",
        "issue",
        "details",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            writer.writerow(asdict(issue))


def write_examples_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        fieldnames = [
            "dataset",
            "split",
            "line_number",
            "id",
            "answer_type",
            "num_answers",
            "all_gold_answers_grounded",
            "input_truncated",
            "target_truncated",
            "question",
        ]
    else:
        fieldnames = sorted({key for row in rows for key in row})

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_example_row(
    record: dict[str, Any],
    dataset: str,
    split: str,
    record_stat: dict[str, Any],
    token_stat: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "dataset": dataset,
        "split": split,
        "line_number": int(record.get("_audit_line_number", -1)),
        "id": str(record.get("id", "")),
        "answer_type": (
            "multi_span" if record_stat["is_multispan"] else "single_span"
        ),
        "num_answers": record_stat["num_answers"],
        "all_gold_answers_grounded": int(
            record_stat["all_gold_answers_grounded"]
        ),
        "any_gold_answer_repeated": int(
            record_stat["any_gold_answer_repeated"]
        ),
        "max_normalized_occurrences": record_stat[
            "max_normalized_occurrences"
        ],
        "has_answer_starts": int(record_stat["has_offsets"]),
        "question": str(record.get("question", "")),
    }

    if token_stat is not None:
        row.update(
            {
                "input_token_count": token_stat["input_token_count"],
                "target_token_count": token_stat["target_token_count"],
                "input_truncated": int(token_stat["input_truncated"]),
                "target_truncated": int(token_stat["target_truncated"]),
                "all_answers_visible_after_truncation": int(
                    token_stat["all_answers_visible_after_truncation"]
                ),
                "visible_answer_count_after_truncation": token_stat[
                    "visible_answer_count_after_truncation"
                ],
            }
        )

    return row


def add_tokenization_issues(
    records: list[dict[str, Any]],
    token_stats: list[dict[str, Any]],
    dataset: str,
    split: str,
    max_source_length: int,
    max_target_length: int,
    issues: list[Issue],
) -> None:
    for record, token_stat in zip(records, token_stats):
        if token_stat["target_truncated"]:
            add_issue(
                issues,
                "WARNING",
                dataset,
                split,
                record,
                "target_longer_than_max_target_length",
                (
                    f"target_tokens={token_stat['target_token_count']}, "
                    f"max_target_length={max_target_length}"
                ),
            )

        if (
            token_stat["input_truncated"]
            and not token_stat["all_answers_visible_after_truncation"]
        ):
            add_issue(
                issues,
                "WARNING",
                dataset,
                split,
                record,
                "gold_answer_removed_by_input_truncation",
                (
                    f"input_tokens={token_stat['input_token_count']}, "
                    f"max_source_length={max_source_length}, "
                    f"visible_answers="
                    f"{token_stat['visible_answer_count_after_truncation']}"
                ),
            )


def issue_counts_for_dataset(
    issues: list[Issue],
    dataset: str,
) -> dict[str, Any]:
    selected = [issue for issue in issues if issue.dataset == dataset]
    severity_counts = Counter(issue.severity for issue in selected)
    type_counts = Counter(issue.issue for issue in selected)

    return {
        "errors": severity_counts.get("ERROR", 0),
        "warnings": severity_counts.get("WARNING", 0),
        "by_type": dict(sorted(type_counts.items())),
    }


def print_source_line(
    prefix: str,
    source_summary: dict[str, Any],
    key: str,
) -> None:
    if source_summary.get("enabled"):
        print(f"{prefix:<48}{source_summary.get(key, 0)}")
    else:
        print(f"{prefix:<48}not checked")


def print_dataset_summary(
    dataset: str,
    summary: dict[str, Any],
    max_source_length: int,
    max_target_length: int,
) -> None:
    name = dataset.upper()
    train = summary["train"]
    validation = summary["validation"]
    train_source = summary["source_consistency"]["train"]
    validation_source = summary["source_consistency"]["validation"]
    leakage = summary["cross_split_leakage"]
    issues = summary["issues"]

    print()
    print("=" * 78)
    print(f"{name} AUDIT SUMMARY")
    print("=" * 78)

    # Same wording/order for both datasets.
    print(f"Training {name} records:".ljust(48) + str(train["records"]))
    print(f"Validation {name} records:".ljust(48) + str(validation["records"]))

    print(
        f"Training {name} single/multi:".ljust(48)
        + f"{train['single_span_records']} / {train['multi_span_records']}"
    )
    print(
        f"Validation {name} single/multi:".ljust(48)
        + f"{validation['single_span_records']} / "
          f"{validation['multi_span_records']}"
    )

    print(
        f"Training {name} gold answers grounded:".ljust(48)
        + f"{train['all_gold_answers_grounded']}/{train['records']}"
    )
    print(
        f"Validation {name} gold answers grounded:".ljust(48)
        + f"{validation['all_gold_answers_grounded']}/"
          f"{validation['records']}"
    )

    if train["tokenization"].get("enabled"):
        print(
            f"Training {name} inputs longer than {max_source_length}:".ljust(48)
            + str(train["tokenization"]["input_truncated"])
        )
        print(
            f"Validation {name} inputs longer than {max_source_length}:".ljust(48)
            + str(validation["tokenization"]["input_truncated"])
        )
        print(
            f"Training {name} examples losing answers:".ljust(48)
            + str(train["tokenization"]["examples_losing_answers"])
        )
        print(
            f"Validation {name} examples losing answers:".ljust(48)
            + str(validation["tokenization"]["examples_losing_answers"])
        )
        print(
            f"Training {name} targets longer than {max_target_length}:".ljust(48)
            + str(train["tokenization"]["target_truncated"])
        )
        print(
            f"Validation {name} targets longer than {max_target_length}:".ljust(48)
            + str(validation["tokenization"]["target_truncated"])
        )
    else:
        print(f"Training {name} tokenizer audit:".ljust(48) + "not run")
        print(f"Validation {name} tokenizer audit:".ljust(48) + "not run")

    print(
        f"Training {name} duplicate IDs:".ljust(48)
        + str(train["duplicate_ids"])
    )
    print(
        f"Validation {name} duplicate IDs:".ljust(48)
        + str(validation["duplicate_ids"])
    )
    print(
        f"{name} cross-split ID overlap:".ljust(48)
        + str(leakage["overlapping_ids"])
    )
    print(
        f"{name} question/context overlap:".ljust(48)
        + str(leakage["overlapping_question_context_pairs"])
    )

    print_source_line(
        f"Training {name} source questions:",
        train_source,
        "source_questions",
    )
    print_source_line(
        f"Validation {name} source questions:",
        validation_source,
        "source_questions",
    )
    print_source_line(
        f"Training {name} expected processed records:",
        train_source,
        "expected_records",
    )
    print_source_line(
        f"Validation {name} expected processed records:",
        validation_source,
        "expected_records",
    )
    print_source_line(
        f"Training {name} missing expected records:",
        train_source,
        "missing_expected_records",
    )
    print_source_line(
        f"Validation {name} missing expected records:",
        validation_source,
        "missing_expected_records",
    )
    print_source_line(
        f"Training {name} unexpected records:",
        train_source,
        "unexpected_records",
    )
    print_source_line(
        f"Validation {name} unexpected records:",
        validation_source,
        "unexpected_records",
    )

    print(f"{name} errors:".ljust(48) + str(issues["errors"]))
    print(f"{name} warnings:".ljust(48) + str(issues["warnings"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit processed DROP and QUOREF data with one unified script."
    )

    parser.add_argument(
        "--train_file",
        required=True,
        type=Path,
        help="Combined processed training JSONL.",
    )
    parser.add_argument(
        "--validation_file",
        required=True,
        type=Path,
        help="Combined processed validation JSONL.",
    )

    parser.add_argument(
        "--drop_train_json",
        type=Path,
        help="Optional original DROP training JSON for source-consistency checks.",
    )
    parser.add_argument(
        "--drop_validation_json",
        "--drop_dev_json",
        dest="drop_validation_json",
        type=Path,
        help="Optional original DROP development/validation JSON.",
    )
    parser.add_argument(
        "--quoref_train_json",
        type=Path,
        help="Optional original SQuAD-style QUOREF training JSON.",
    )
    parser.add_argument(
        "--quoref_validation_json",
        "--quoref_dev_json",
        dest="quoref_validation_json",
        type=Path,
        help="Optional original SQuAD-style QUOREF development/validation JSON.",
    )

    parser.add_argument(
        "--report_dir",
        "--output_dir",
        dest="report_dir",
        type=Path,
        default=Path("reports/data_audit"),
        help="Directory for summary, issue, and per-example reports.",
    )
    parser.add_argument(
        "--tokenizer_name",
        "--model_name",
        dest="tokenizer_name",
        default="google-t5/t5-small",
        help="Tokenizer used for length/truncation checks.",
    )
    parser.add_argument(
        "--max_source_length",
        "--max_input_length",
        dest="max_source_length",
        type=int,
        default=512,
        help="Maximum source length used during training.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=64,
        help="Maximum target length used during training.",
    )
    parser.add_argument(
        "--skip_tokenizer_audit",
        action="store_true",
        help="Skip tokenizer-dependent source/target length checks.",
    )
    parser.add_argument(
        "--fail_on_errors",
        action="store_true",
        help="Return exit code 1 when audit errors are found.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    all_issues: list[Issue] = []
    example_rows: list[dict[str, Any]] = []

    train_all, train_loading_issues = load_jsonl(args.train_file, "train")
    validation_all, validation_loading_issues = load_jsonl(
        args.validation_file,
        "validation",
    )
    all_issues.extend(train_loading_issues)
    all_issues.extend(validation_loading_issues)

    tokenizer = None
    tokenizer_error = None

    if not args.skip_tokenizer_audit:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name,
                use_fast=True,
            )
        except Exception as exc:
            tokenizer_error = f"{type(exc).__name__}: {exc}"
            print("Tokenizer audit could not run:")
            print(tokenizer_error)

    source_paths = {
        "drop": {
            "train": args.drop_train_json,
            "validation": args.drop_validation_json,
        },
        "quoref": {
            "train": args.quoref_train_json,
            "validation": args.quoref_validation_json,
        },
    }

    dataset_summaries: dict[str, Any] = {}

    for dataset in DATASETS:
        train_records = get_dataset_records(train_all, dataset)
        validation_records = get_dataset_records(validation_all, dataset)

        if not train_records:
            print(f"WARNING: No {dataset.upper()} records found in {args.train_file}")
        if not validation_records:
            print(
                f"WARNING: No {dataset.upper()} records found in "
                f"{args.validation_file}"
            )

        train_record_stats = [
            audit_record(record, dataset, "train", all_issues)
            for record in train_records
        ]
        validation_record_stats = [
            audit_record(record, dataset, "validation", all_issues)
            for record in validation_records
        ]

        train_duplicate_ids = audit_duplicate_ids(
            train_records,
            dataset,
            "train",
            all_issues,
        )
        validation_duplicate_ids = audit_duplicate_ids(
            validation_records,
            dataset,
            "validation",
            all_issues,
        )

        leakage_summary = audit_cross_split_leakage(
            train_records,
            validation_records,
            dataset,
            all_issues,
        )

        train_token_stats: list[dict[str, Any]] | None = None
        validation_token_stats: list[dict[str, Any]] | None = None

        if tokenizer is not None:
            train_token_stats = [
                audit_tokenization(
                    record,
                    tokenizer,
                    args.max_source_length,
                    args.max_target_length,
                )
                for record in train_records
            ]
            validation_token_stats = [
                audit_tokenization(
                    record,
                    tokenizer,
                    args.max_source_length,
                    args.max_target_length,
                )
                for record in validation_records
            ]

            add_tokenization_issues(
                train_records,
                train_token_stats,
                dataset,
                "train",
                args.max_source_length,
                args.max_target_length,
                all_issues,
            )
            add_tokenization_issues(
                validation_records,
                validation_token_stats,
                dataset,
                "validation",
                args.max_source_length,
                args.max_target_length,
                all_issues,
            )

        for index, (record, record_stat) in enumerate(
            zip(train_records, train_record_stats)
        ):
            token_stat = (
                train_token_stats[index]
                if train_token_stats is not None
                else None
            )
            example_rows.append(
                make_example_row(
                    record,
                    dataset,
                    "train",
                    record_stat,
                    token_stat,
                )
            )

        for index, (record, record_stat) in enumerate(
            zip(validation_records, validation_record_stats)
        ):
            token_stat = (
                validation_token_stats[index]
                if validation_token_stats is not None
                else None
            )
            example_rows.append(
                make_example_row(
                    record,
                    dataset,
                    "validation",
                    record_stat,
                    token_stat,
                )
            )

        source_consistency: dict[str, dict[str, Any]] = {}

        for split, records in (
            ("train", train_records),
            ("validation", validation_records),
        ):
            source_path = source_paths[dataset][split]

            if source_path is None:
                source_consistency[split] = {"enabled": False}
                continue

            if dataset == "drop":
                expected_by_id, source_counts = build_expected_drop(
                    source_path,
                    split,
                )
            else:
                expected_by_id, source_counts = build_expected_quoref(
                    source_path,
                    split,
                )

            source_consistency[split] = compare_to_source(
                records,
                expected_by_id,
                source_counts,
                dataset,
                split,
                all_issues,
            )

        dataset_summaries[dataset] = {
            "train": summarize_split(
                train_records,
                train_record_stats,
                train_duplicate_ids,
                train_token_stats,
            ),
            "validation": summarize_split(
                validation_records,
                validation_record_stats,
                validation_duplicate_ids,
                validation_token_stats,
            ),
            "cross_split_leakage": leakage_summary,
            "source_consistency": source_consistency,
        }

    # Add issue summaries only after every audit has finished.
    for dataset in DATASETS:
        dataset_summaries[dataset]["issues"] = issue_counts_for_dataset(
            all_issues,
            dataset,
        )

    global_severity_counts = Counter(issue.severity for issue in all_issues)
    global_issue_type_counts = Counter(issue.issue for issue in all_issues)

    summary = {
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "target_separator": TARGET_SEPARATOR,
        "tokenizer": {
            "enabled": tokenizer is not None,
            "tokenizer_name": args.tokenizer_name,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
            "error": tokenizer_error,
        },
        "datasets": dataset_summaries,
        "combined": {
            "training_records": sum(
                dataset_summaries[dataset]["train"]["records"]
                for dataset in DATASETS
            ),
            "validation_records": sum(
                dataset_summaries[dataset]["validation"]["records"]
                for dataset in DATASETS
            ),
            "errors": global_severity_counts.get("ERROR", 0),
            "warnings": global_severity_counts.get("WARNING", 0),
            "issues_by_type": dict(sorted(global_issue_type_counts.items())),
        },
    }

    summary_path = args.report_dir / "data_audit_summary.json"
    issues_path = args.report_dir / "data_audit_issues.csv"
    examples_path = args.report_dir / "data_audit_examples.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    write_issues_csv(issues_path, all_issues)
    write_examples_csv(examples_path, example_rows)

    for dataset in DATASETS:
        print_dataset_summary(
            dataset,
            dataset_summaries[dataset],
            args.max_source_length,
            args.max_target_length,
        )

    print()
    print("=" * 78)
    print("COMBINED AUDIT SUMMARY")
    print("=" * 78)
    print(
        "Training combined records:".ljust(48)
        + str(summary["combined"]["training_records"])
    )
    print(
        "Validation combined records:".ljust(48)
        + str(summary["combined"]["validation_records"])
    )
    print("Total errors:".ljust(48) + str(summary["combined"]["errors"]))
    print("Total warnings:".ljust(48) + str(summary["combined"]["warnings"]))
    print(f"Summary report: {summary_path}")
    print(f"Issues report:  {issues_path}")
    print(f"Examples report:{examples_path}")

    if summary["combined"]["errors"] == 0:
        print("\nRESULT: PASS — no audit errors were found.")
    else:
        print("\nRESULT: REVIEW REQUIRED — inspect data_audit_issues.csv.")

    if args.fail_on_errors and summary["combined"]["errors"] > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
