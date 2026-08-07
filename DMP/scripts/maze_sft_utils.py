#!/usr/bin/env python3
"""Shared helpers for DMP maze SFT scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DIFFICULTY_LEVELS = ("Easy", "Medium", "Hard")


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level list in {path}, got {type(data).__name__}.")

    return [row for row in data if isinstance(row, dict)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_id, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_id}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    return load_json_list(path)


def write_json_or_jsonl(rows: list[dict[str, Any]], output_path: Path, indent: int = 2) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def build_dataset_info_snippet(dataset_name: str, output_path: Path) -> dict[str, Any]:
    return {
        dataset_name: {
            "file_name": output_path.name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }


def normalize_difficulty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "easy":
        return "Easy"
    if lowered == "medium":
        return "Medium"
    if lowered == "hard":
        return "Hard"
    return text


def longest_prefix_ratio(pred_codes: list[Any], target_codes: list[Any]) -> float:
    if not target_codes:
        return 0.0

    prefix = 0
    for index, target in enumerate(target_codes):
        if index >= len(pred_codes):
            break
        if pred_codes[index] != target:
            break
        prefix += 1
    return prefix / len(target_codes)
