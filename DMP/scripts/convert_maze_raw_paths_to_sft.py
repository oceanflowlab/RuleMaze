#!/usr/bin/env python3
"""Convert raw 3x3 maze JSON data into LLaMA-Factory SFT datasets.

Expected source format (top-level list):
{
  "_meta_difficulty": "Easy",
  "rule_id": "...",
  "data": {
    "maze_index": 310084,
    "matched_actions": [["down", "right", "down"]],
    "start": [1, 2],
    "end": [3, 3],
    "image_path": "/abs/path/maze_310084.png",
    "wrong_actions": [["right", "right", "up", "left"],
    ...
  }
}

The script writes a LLaMA-Factory compatible multimodal ShareGPT dataset with
`messages` and `images` columns.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from maze_sft_utils import build_dataset_info_snippet, load_json_list, write_json_or_jsonl

ALLOWED_DIRECTIONS = {"up", "down", "left", "right"}
SPECIAL_STATES = {"blue", "yellow", "pink", "purple", "orange", "blue_green"}
GOAL_INSTRUCTION = "Your task is to find a path from the starting position to the goal position while following the given rule. The starting position is marked in green, and the goal position is marked in red."
# GOAL_INSTRUCTION = "Your task is to find a path from the starting position to the goal position while following the given rule. The starting position is marked as prince, and the goal position is marked as princess or treasure."

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw maze JSON files into a LLaMA-Factory SFT dataset."
    )
    parser.add_argument("input_path", type=Path, help="Path to the raw maze JSON file.")
    parser.add_argument("output_path", type=Path, help="Path to the output JSON or JSONL file.")
    parser.add_argument(
        "--response-style",
        choices=(
            "actions_only",
            "coordinates_then_actions",
            "ascii_then_actions",
            "all_then_actions",
        ),
        default="actions_only",
        help=(
            "`actions_only` asks the model to output the gt action sequence only. "
            "`coordinates_then_actions` asks for a coordinate-based reasoning trace first, then the action sequence. "
            "`ascii_then_actions` asks for an ASCII-based reasoning trace first, then the action sequence. "
            "`all_then_actions` asks for all actions from start to end first, then the action sequence."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "auto", "none"),
        default="auto",
        help="Optional split tag stored in the output metadata.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset name used to print a ready-to-copy dataset_info snippet.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation for JSON output. Ignored for JSONL.",
    )
    parser.add_argument(
        "--retain-percent",
        type=float,
        default=100.0,
        help="Keep only this percentage of loaded records (0-100).",
    )
    return parser.parse_args()


def load_records(input_path: Path) -> list[dict[str, Any]]:
    return load_json_list(input_path)


def retain_records(records: list[dict[str, Any]], retain_percent: float) -> list[dict[str, Any]]:
    if not 0.0 <= retain_percent <= 100.0:
        raise ValueError(f"`retain_percent` must be between 0 and 100, got {retain_percent}.")

    if retain_percent >= 100.0 or not records:
        return records

    keep_count = math.ceil(len(records) * retain_percent / 100.0)
    if retain_percent > 0.0:
        keep_count = max(1, keep_count)

    return records[:keep_count]


def infer_split(input_path: Path, split: str) -> str | None:
    if split == "none":
        return None
    if split != "auto":
        return split

    stem = input_path.stem.lower()
    if "train" in stem:
        return "train"
    if "test" in stem:
        return "test"
    return None


def format_position(position: Any) -> str:
    if isinstance(position, (list, tuple)) and len(position) == 2:
        return f"({position[0]}, {position[1]})"
    return str(position)


def serialize_maze_structure(maze_structure: Any) -> str:
    if not isinstance(maze_structure, dict) or not maze_structure:
        raise ValueError("Missing or empty `data.maze_structure`.")
    return json.dumps(maze_structure, ensure_ascii=False, sort_keys=True, indent=2)


def get_cell_state(cell_value: Any) -> str:
    if not isinstance(cell_value, dict):
        return "unknown"
    state = cell_value.get("state")
    return state if isinstance(state, str) and state else "unknown"


def get_grid_coordinates(maze_structure: dict[str, Any]) -> list[tuple[int, int]]:
    coordinates: list[tuple[int, int]] = []
    for key in maze_structure:
        if not isinstance(key, str):
            continue
        text = key.strip().lstrip("(").rstrip(")")
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2:
            continue
        try:
            coordinates.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return coordinates


def build_coordinate_layout_prompt(maze_structure: dict[str, Any], start: Any, end: Any) -> str:
    starting_state = "green"
    goal_state = "red"
    normal_cells: list[str] = []
    special_cells: list[str] = []

    for coordinate in get_grid_coordinates(maze_structure):
        coordinate_key = f"({coordinate[0]}, {coordinate[1]})"
        cell_state = get_cell_state(maze_structure.get(coordinate_key))
        if cell_state == starting_state:
            continue
        if cell_state == goal_state:
            continue
        if cell_state == "normal":
            normal_cells.append(coordinate_key)
        else:
            special_cells.append(f"{coordinate_key} ({cell_state})")

    normal_text = ", ".join(normal_cells) if normal_cells else "none"
    special_text = ", ".join(special_cells) if special_cells else "none"

    return (
        "The layout of the given image is as follows:\n"
        f"- Starting position: {format_position(start)}\n"
        f"- Goal: {format_position(end)}\n"
        f"- Normal Cells: {normal_text}\n"
        f"- Special Tokens: {special_text}\n"
    )


def build_ascii_layout_prompt(maze_structure: dict[str, Any]) -> str:
    rows = [1, 2, 3]
    cols = [1, 2, 3]
    symbol_map = {
        "green": "S",
        "red": "G",
        "normal": "N",
    }

    lines: list[str] = [
        "w/ ASCII：",
        "The symbols used to represent the grid are:",
        "- S denotes the starting position, G denotes the goal, N denotes the normal cells, X denotes the special cells.",
    ]

    for row in rows:
        row_symbols: list[str] = []
        for col in cols:
            cell_key = f"({row}, {col})"
            cell_state = get_cell_state(maze_structure.get(cell_key))
            row_symbols.append(symbol_map.get(cell_state, "X"))
        lines.append("".join(row_symbols))

    return "\n".join(lines) + "\n"


def build_wrong_actions_prompt(wrong_actions: Any) -> str:
    sequences = normalize_action_sequences(wrong_actions)
    flattened_actions = [action for sequence in sequences for action in sequence]
    return ' '.join(flattened_actions)


def build_matched_actions_prompt(matched_actions: Any) -> str:
    sequences = normalize_action_sequences(matched_actions)
    flattened_actions = [action for sequence in sequences for action in sequence]
    return ' '.join(flattened_actions)

def extract_data(record: dict[str, Any]) -> dict[str, Any]:
    nested = record.get("data")
    if isinstance(nested, dict):
        return nested
    return record


def normalize_action_sequences(raw_actions: Any) -> list[list[str]]:
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("Missing or empty `matched_actions`.")

    if all(isinstance(item, str) for item in raw_actions):
        return [list(raw_actions)]

    sequences: list[list[str]] = []
    for candidate in raw_actions:
        if not isinstance(candidate, list) or not all(isinstance(item, str) for item in candidate):
            raise ValueError("`matched_actions` must contain lists of action strings.")
        sequences.append(list(candidate))

    if not sequences:
        raise ValueError("`matched_actions` does not contain any valid action sequence.")

    return sequences


def actions_to_answer(actions: list[str]) -> str:
    validated_actions: list[str] = []
    for action in actions:
        if action not in ALLOWED_DIRECTIONS:
            raise ValueError(f"Unsupported action direction: {action!r}")
        validated_actions.append(action)

    return "\n".join(validated_actions)


def build_user_prompt(record: dict[str, Any], include_rule: bool, response_style: str) -> str:
    data = extract_data(record)
    rule_id = record.get("rule_id", "unknown")
    start = format_position(data.get("start"))
    end = format_position(data.get("end"))

    rule_line = f"Rule: {rule_id}\n" if include_rule else ""
    if response_style == "actions_only":
        instruction = "Return only the final action sequence wrapped in <ANSWER>...</ANSWER>.\n"
    else:
        instruction = "Provide your text-based planning first, then the final answer wrapped in <ANSWER>...</ANSWER>.\n"

    return (
        "<image>"
        "You are solving a 3x3 maze.\n"
        f"{GOAL_INSTRUCTION}\n"
        f"{rule_line}"
        # f"Start: {start}\n"
        # f"Goal: {end}\n"
        f"{instruction}"
    )


def build_reasoning_text(record: dict[str, Any], response_style: str) -> str:
    data = extract_data(record)
    maze_structure = data.get("maze_structure")

    if response_style == "coordinates_then_actions":
        if not isinstance(maze_structure, dict):
            raise ValueError("Missing `data.maze_structure` for coordinate response style.")
        return build_coordinate_layout_prompt(maze_structure, data.get("start"), data.get("end"))

    if response_style == "ascii_then_actions":
        if not isinstance(maze_structure, dict):
            raise ValueError("Missing `data.maze_structure` for ASCII response style.")
        return build_ascii_layout_prompt(maze_structure)

    if response_style == "all_then_actions":
        wrong_path_text = build_wrong_actions_prompt(data.get("wrong_actions"))
        matched_path_text = build_matched_actions_prompt(data.get("matched_actions"))
        return f"If not considering the rule, the paths from the start to the goal include: {wrong_path_text}, and {matched_path_text}.\n"

    return ""


def build_assistant_content(record: dict[str, Any], response_style: str) -> str:
    reasoning_text = build_reasoning_text(record, response_style)
    data = extract_data(record)

    action_sequences = normalize_action_sequences(data.get("matched_actions"))
    selected_actions: list[str] = []
    for action_sequence in action_sequences:
        selected_actions.extend(action_sequence)
    if not selected_actions:
        raise ValueError("`matched_actions` does not contain any action.")

    answer_text = actions_to_answer(selected_actions)
    answer_text = f"<ANSWER>{answer_text.replace(chr(10), ' ')}</ANSWER>"

    if reasoning_text:
        if response_style == "all_then_actions":
            return (
                f"{reasoning_text}"
                f"Given the rule, the final answer is: {answer_text}"
            )
        return f"{reasoning_text}Based on the above layout, the final answer is: {answer_text}"

    return answer_text


def build_sample(
    record: dict[str, Any],
    response_style: str,
    split: str | None,
) -> dict[str, Any]:
    data = extract_data(record)
    image_path = data.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("Missing `data.image_path`.")

    sample: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": build_user_prompt(record, include_rule=True, response_style=response_style),
            },
            {
                "role": "assistant",
                "content": build_assistant_content(record, response_style=response_style),
            },
        ],
        "images": [image_path],
        "maze_index": data.get("maze_index"),
        "difficulty": record.get("_meta_difficulty") or record.get("difficulty"),
        "rule_id": record.get("rule_id"),
        "start_pos": data.get("start"),
        "end_pos": data.get("end"),
    }

    if split is not None:
        sample["source_split"] = split

    return sample


def convert_records(
    records: list[dict[str, Any]],
    response_style: str,
    split: str | None,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            converted.append(build_sample(record=record, response_style=response_style, split=split))
        except Exception as exc:
            raise ValueError(f"Failed to convert record #{index}: {exc}") from exc

    return converted


def main() -> None:
    args = parse_args()
    records = load_records(args.input_path)
    records = retain_records(records, args.retain_percent)
    split = infer_split(args.input_path, args.split)
    samples = convert_records(
        records=records,
        response_style=args.response_style,
        split=split,
    )
    write_json_or_jsonl(samples, args.output_path, args.indent)

    print(f"Converted {len(samples)} samples -> {args.output_path}")
    if args.dataset_name:
        print("\nSuggested dataset_info.json entry:")
        print(json.dumps(build_dataset_info_snippet(args.dataset_name, args.output_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
