"""Shared helpers for the NLP multi-span QA experiments."""

from __future__ import annotations

import importlib.util
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

ANSWER_SEPARATOR = " ||| "


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return its non-empty records."""
    path = Path(path)
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


def iter_jsonl(path: Path | str) -> Iterator[Dict[str, Any]]:
    """Yield JSONL records one at a time."""
    path = Path(path)

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

            yield record


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path | str) -> int:
    """Write JSON objects as JSONL and return the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(dict(record), file, ensure_ascii=False)
            file.write("\n")
            count += 1

    return count


def load_json(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


def normalize_answer(text: Any) -> str:
    """A simple DROP/SQuAD-style normalization for diagnostics."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def get_predicted_answers(record_or_value: Any) -> List[str]:
    """Read a prediction list from JSONL records or official JSON values."""
    if isinstance(record_or_value, dict):
        answers = record_or_value.get("predicted_answers")
        if isinstance(answers, list):
            return [
                str(answer).strip()
                for answer in answers
                if str(answer).strip()
            ]

        prediction_text = str(record_or_value.get("prediction_text", ""))
        return [
            answer.strip()
            for answer in prediction_text.split(ANSWER_SEPARATOR)
            if answer.strip()
        ]

    if isinstance(record_or_value, (list, tuple)):
        return [
            str(answer).strip()
            for answer in record_or_value
            if str(answer).strip()
        ]

    value = str(record_or_value or "").strip()
    return [value] if value else []


def answer_is_grounded(answer: str, context: str) -> bool:
    normalized_answer = normalize_answer(answer)
    normalized_context = normalize_answer(context)
    return bool(normalized_answer) and normalized_answer in normalized_context


def count_normalized_occurrences(answer: str, context: str) -> int:
    """Count non-overlapping normalized substring occurrences."""
    normalized_answer = normalize_answer(answer)
    normalized_context = normalize_answer(context)
    if not normalized_answer:
        return 0
    return normalized_context.count(normalized_answer)


def deduplicate_answers(answers: Sequence[str]) -> List[str]:
    """Remove normalized duplicates while preserving first occurrence order."""
    seen = set()
    output: List[str] = []

    for answer in answers:
        key = normalize_answer(answer)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(answer.strip())

    return output


def exact_match_lists(gold_answers: Sequence[str], predicted_answers: Sequence[str]) -> int:
    gold = Counter(normalize_answer(answer) for answer in gold_answers)
    predicted = Counter(normalize_answer(answer) for answer in predicted_answers)
    return int(gold == predicted)


def load_python_module(path: Path | str, module_name: str):
    """Load a Python file as a module."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
