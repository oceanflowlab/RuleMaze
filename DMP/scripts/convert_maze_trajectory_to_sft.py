#!/usr/bin/env python3
"""Convert maze trajectory data into LLaMA-Factory SFT datasets.

Supported source format example:
`../DATA/Training_Data/train_sample_trajectory_heuristic_with_step_images.json`

This script produces a LLaMA-Factory-compatible multimodal SFT dataset in
sharegpt format with `messages` and `images` fields.
"""

from __future__ import annotations

import argparse
import json
import re
import random
from pathlib import Path
from typing import Any

from maze_sft_utils import build_dataset_info_snippet, load_json_list, write_json_or_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert maze trajectory data into a LLaMA-Factory SFT dataset."
    )
    parser.add_argument("input_path", type=Path, help="Path to the source maze JSON file.")
    parser.add_argument("output_path", type=Path, help="Path to the converted output JSON/JSONL file.")
    parser.add_argument(
        "--trajectory-source",
        choices=("trajectory", "wrong_trajectory", "both"),
        default="trajectory",
        help=(
            "Choose which trajectory field to convert. `trajectory` uses ground-truth path, "
            "`wrong_trajectory` uses heuristic wrong path, and `both` outputs both variants."
        ),
    )
    parser.add_argument(
        "--add-wrong-trajectory-hint",
        action="store_true",
        help=(
            "Add an extra step-1 training sample for the correct trajectory with a "
            "wrong-trajectory description appended to the user prompt."
        ),
    )
    parser.add_argument(
        "--use-step0-image-path",
        action="store_true",
        help=(
            "When set, every step in a trajectory uses step 0's input_image_path instead of "
            "its own current-step image."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset name used to print a ready-to-copy dataset_info snippet.",
    )
    parser.add_argument(
        "--retain-percent",
        type=float,
        default=100.0,
        help="Keep only this percentage of loaded records (0-100).",
    )
    parser.add_argument(
        "--retain-difficulties",
        nargs="+",
        choices=("Easy", "Medium", "Hard"),
        default=None,
        help="Keep only records whose difficulty is in this set. Defaults to all difficulties.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation for JSON output. Ignored for JSONL.",
    )
    return parser.parse_args()


def load_records(
    input_path: Path,
    retain_percent: float = 100.0,
    retain_difficulties: set[str] | None = None,
) -> list[dict[str, Any]]:
    data = load_json_list(input_path)

    if retain_difficulties is not None:
        data = [record for record in data if record.get("difficulty") in retain_difficulties]
    
    if retain_percent < 100:
        target_count = max(1, int(len(data) * retain_percent / 100))
        rng = random.Random(42)
        data = rng.sample(data, target_count) if target_count < len(data) else data

    return data


def build_stepwise_user_prompt(
    record: dict[str, Any],
    completed_codes: list[str],
    include_rule: bool,
    wrong_trajectory_note: list[str] | None = None,
) -> str:
    completed_text = "\n".join(f"{index + 1}. {code}" for index, code in enumerate(completed_codes))
    if not completed_text:
        completed_text = "None"

    rule_line = f"Rule: {record.get('rule_id', 'unknown')}\n" if include_rule else ""
    wrong_line = (
        f"Wrong trajectory (do NOT follow):\n{', '.join(wrong_trajectory_note)}\n"
        if wrong_trajectory_note
        else ""
    )

    return (
        "<image>"
        "You are solving a 3x3 maze step by step.\n"
        f"{rule_line}"
        f"{wrong_line}"
        f"Completed actions so far:\n{completed_text}\n"
        "Look at the image and output the exact code for this step.\n"
        "Allowed actions are LocateStart(), ExecuteMove('up'|'down'|'left'|'right'), VerifyRule(), and VerifyEnd().\n"
        "Output only the code.\n"
    )


def build_wrong_trajectory_note(record: dict[str, Any]) -> list[str]:
    wrong_steps = record.get("wrong_trajectory", {}).get("steps", [])
    wrong_moves: list[str] = []
    pattern = re.compile(r"ExecuteMove\(\s*['\"]([^'\"]+)['\"]\s*\)")

    for step in wrong_steps:
        code = step.get("code")
        if not code:
            continue
        wrong_moves.extend(pattern.findall(code))
    return wrong_moves


def is_training_split(input_path: Path) -> bool:
    path_text = str(input_path).lower()
    return "train" in path_text and "test" not in path_text


def build_stepwise_samples(
    record: dict[str, Any],
    trajectory_key: str,
    include_rule: bool,
    use_step0_image_path: bool = False,
    only_last: bool = False,
) -> list[dict[str, Any]]:
    all_steps = record[trajectory_key]["steps"]
    step0_image_path = all_steps[0].get("input_image_path") if all_steps else None
    steps = all_steps
    if only_last:
        steps = steps[-1:]
    completed_codes: list[str] = []
    samples: list[dict[str, Any]] = []

    for step in steps:
        code = step.get("code")
        step_id = step.get("step_id", len(completed_codes))
        image_path = step0_image_path if use_step0_image_path else step.get("input_image_path")
        if not code or not image_path:
            raise ValueError(f"Step {step_id} is missing `code` or `input_image_path`.")

        sample: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": build_stepwise_user_prompt(record, completed_codes, include_rule),
                },
                {
                    "role": "assistant",
                    "content": code,
                },
            ],
            "images": [image_path],
            "prompt_variant": "standard",
            "maze_index": record.get("maze_index"),
            "difficulty": record.get("difficulty"),
            "rule_id": record.get("rule_id"),
            "step_id": step_id,
            "trajectory_source": trajectory_key,
        }
        samples.append(sample)
        completed_codes.append(code)

    return samples


def convert_records(
    records: list[dict[str, Any]],
    trajectory_source: str,
    add_wrong_trajectory_hint: bool,
    use_step0_image_path: bool,
    only_last_wrong: bool,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if trajectory_source == "both":
        trajectory_keys = ["trajectory", "wrong_trajectory"]
    else:
        trajectory_keys = [trajectory_source]

    for index, record in enumerate(records):
        for trajectory_key in trajectory_keys:
            trajectory = record.get(trajectory_key, {})
            steps = trajectory.get("steps", [])
            if not steps:
                raise ValueError(f"Missing `{trajectory_key}.steps`.")
            step0_image_path = steps[0].get("input_image_path")

            # include_rule = trajectory_key == "trajectory"
            include_rule = True  # Always include the rule in the prompt

            use_only_last_wrong = only_last_wrong and trajectory_key == "wrong_trajectory"
            step_samples = build_stepwise_samples(
                record=record,
                trajectory_key=trajectory_key,
                include_rule=include_rule,
                use_step0_image_path=use_step0_image_path,
                only_last=use_only_last_wrong,
            )
            converted.extend(step_samples)
            
            if add_wrong_trajectory_hint and trajectory_key == "trajectory":
                wrong_note = build_wrong_trajectory_note(record)
                if wrong_note:
                    first_step = steps[1]
                    code = first_step.get("code")
                    image_path = first_step.get("input_image_path")
                    step_id = first_step.get("step_id", 1)
                    if not code or not image_path:
                        raise ValueError("First step is missing `code` or `input_image_path`.")
                    hint_sample: dict[str, Any] = {
                        "messages": [
                            {
                                "role": "user",
                                "content": build_stepwise_user_prompt(
                                    record=record,
                                    completed_codes=[steps[0].get("code", "")],  # step0 should be LocateStart()
                                    include_rule=include_rule,
                                    wrong_trajectory_note=wrong_note,
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": code,
                            },
                        ],
                        "images": [step0_image_path if use_step0_image_path else image_path],
                        "prompt_variant": "wrong_hint",
                        "maze_index": record.get("maze_index"),
                        "difficulty": record.get("difficulty"),
                        "rule_id": record.get("rule_id"),
                        "step_id": step_id,
                        "trajectory_source": trajectory_key,
                    }
                    converted.append(hint_sample)

    return converted


def main() -> None:
    args = parse_args()
    records = load_records(
        args.input_path,
        retain_percent=args.retain_percent,
        retain_difficulties=set(args.retain_difficulties) if args.retain_difficulties else None,
    )
    samples = convert_records(
        records=records,
        trajectory_source=args.trajectory_source,
        add_wrong_trajectory_hint=args.add_wrong_trajectory_hint,
        use_step0_image_path=args.use_step0_image_path,
        only_last_wrong=is_training_split(args.input_path),
    )
    write_json_or_jsonl(samples, args.output_path, args.indent)

    print(f"Converted {len(samples)} samples -> {args.output_path}")
    if args.dataset_name:
        print("\nSuggested dataset_info.json entry:")
        print(json.dumps(build_dataset_info_snippet(args.dataset_name, args.output_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
