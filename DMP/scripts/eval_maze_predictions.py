#!/usr/bin/env python3
"""Evaluate maze action-code predictions from LLaMA-Factory outputs.

Expected input is `generated_predictions.jsonl` created by SFT predict mode.
Each line should contain at least: {"predict": ..., "label": ...}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"ExecuteMove\('\s*([^'\n]+?)\s*'\)", re.IGNORECASE)
COMMAND_RE = re.compile(
    r"LocateStart\(\)|VerifyRule\(\)|VerifyEnd\(\)|ExecuteMove\('\s*([^'\n]+?)\s*'\)",
    re.IGNORECASE,
)
DIFFICULTY_LEVELS = ("Easy", "Medium", "Hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate maze predictions with exact-match metrics.")
    parser.add_argument(
        "prediction_file",
        type=Path,
        help="Path to generated_predictions.jsonl produced by LLaMA-Factory.",
    )
    parser.add_argument(
        "--save-errors",
        type=Path,
        default=None,
        help="Optional path to save mismatch details in JSONL.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help=(
            "Optional converted SFT dataset JSON/JSONL used in prediction. "
            "When provided, computes maze-level metrics by aligning rows by index."
        ),
    )
    parser.add_argument(
        "--save-metrics-json",
        type=Path,
        default=None,
        help="Optional path to save aggregate metrics as JSON.",
    )
    parser.add_argument(
        "--save-maze-details",
        type=Path,
        default=None,
        help="Optional path to save per-maze EM/PR details in JSONL.",
    )
    return parser.parse_args()


def extract_action_code(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    m = ACTION_RE.search(text)
    if not m:
        return None
    direction = m.group(1).strip()
    return f"ExecuteMove('{direction}')"


def extract_command_sequence(text: str) -> list[str]:
    if not isinstance(text, str):
        return []

    commands: list[str] = []
    for match in COMMAND_RE.finditer(text):
        move_direction = match.group(1)
        matched_text = match.group(0).strip().lower()
        if move_direction is not None:
            commands.append(f"ExecuteMove('{move_direction.strip()}')")
        elif matched_text.startswith("locatestart"):
            commands.append("LocateStart()")
        elif matched_text.startswith("verifyrule"):
            commands.append("VerifyRule()")
        elif matched_text.startswith("verifyend"):
            commands.append("VerifyEnd()")
    return commands


def flatten_step_commands(entries: list[dict[str, Any]], key: str) -> list[str]:
    codes: list[str] = []
    for entry in entries:
        codes.extend(extract_command_sequence(str(entry.get(key, "")).strip()))
    return codes


def first_execute_move(commands: list[str]) -> str | None:
    for cmd in commands:
        if cmd.startswith("ExecuteMove("):
            return cmd
    return None


def full_sequence_em(pred_codes: list[str], target_codes: list[str]) -> bool:
    if len(pred_codes) != len(target_codes):
        return False
    return all(pred_codes[i] == target_codes[i] for i in range(len(target_codes)))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_id}: {exc}") from exc
    return rows


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    raise ValueError(f"Expected list in {path}, got {type(data).__name__}.")


def normalize_rule_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def longest_prefix_ratio(pred_codes: list[str | None], target_codes: list[str]) -> float:
    if not target_codes:
        return 0.0

    prefix = 0
    for idx, target in enumerate(target_codes):
        if idx >= len(pred_codes):
            break
        pred = pred_codes[idx]
        if pred is None or pred != target:
            break
        prefix += 1
    return prefix / len(target_codes)


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


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.prediction_file)
    if not rows:
        raise ValueError(f"No records found in {args.prediction_file}")

    total = 0
    exact_text_match = 0
    exact_action_match = 0
    invalid_pred = 0
    errors: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        pred = str(row.get("predict", "")).strip()
        label = str(row.get("label", "")).strip()
        total += 1

        if pred == label:
            exact_text_match += 1

        pred_code = extract_action_code(pred)
        label_code = extract_action_code(label)

        if pred_code is None:
            invalid_pred += 1

        if pred_code is not None and label_code is not None and pred_code == label_code:
            exact_action_match += 1
        elif label_code is not None:
            errors.append(
                {
                    "index": idx,
                    "predict": pred,
                    "label": label,
                    "predict_action": pred_code,
                    "label_action": label_code,
                }
            )

    print(f"Total: {total}")
    print(f"Exact text match: {exact_text_match}/{total} = {exact_text_match / total:.4f}")
    print(f"Exact action match: {exact_action_match}/{total} = {exact_action_match / total:.4f}")
    print(f"Invalid action format rate: {invalid_pred}/{total} = {invalid_pred / total:.4f}")

    metrics: dict[str, Any] = {
        "total": total,
        "exact_text_match": exact_text_match / total,
        "exact_action_match": exact_action_match / total,
        "invalid_action_format_rate": invalid_pred / total,
    }

    if args.dataset_file is not None:
        dataset_rows = load_json_or_jsonl(args.dataset_file)
        if len(dataset_rows) != len(rows):
            raise ValueError(
                "Length mismatch: prediction rows = "
                f"{len(rows)} but dataset rows = {len(dataset_rows)}. "
                "Please ensure `--dataset-file` is exactly the eval dataset used for prediction."
            )

        grouped: dict[tuple[int, str], dict[str, Any]] = defaultdict(
            lambda: {
                "entries": [],
                "by_source": defaultdict(list),
                "difficulty": None,
                "maze_index": None,
                "rule_id": "",
            }
        )
        for idx, (pred_row, sample_row) in enumerate(zip(rows, dataset_rows), start=1):
            maze_index = sample_row.get("maze_index")
            step_id = sample_row.get("step_id")
            if maze_index is None or step_id is None:
                raise ValueError(
                    "`--dataset-file` rows must include both `maze_index` and `step_id` for maze-level metrics. "
                    f"Missing at row index {idx - 1}."
                )

            try:
                maze_index_key = int(maze_index)
                step = int(step_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid `maze_index` or `step_id` at row index {idx - 1}: "
                    f"maze_index={maze_index}, step_id={step_id}"
                ) from exc
            rule_id_key = normalize_rule_id(sample_row.get("rule_id"))
            maze_key = (maze_index_key, rule_id_key)
            trajectory_source = str(sample_row.get("trajectory_source", "trajectory") or "trajectory").strip()

            pred_code = extract_action_code(str(pred_row.get("predict", "")).strip())
            label_code = extract_action_code(str(pred_row.get("label", "")).strip())
            difficulty = normalize_difficulty(sample_row.get("difficulty"))
            grouped_item = grouped[maze_key]
            grouped_item["maze_index"] = maze_index_key
            grouped_item["rule_id"] = rule_id_key

            if difficulty is not None:
                current_diff = grouped_item["difficulty"]
                if current_diff is None:
                    grouped_item["difficulty"] = difficulty
                elif current_diff != difficulty:
                    print(
                        "[WARN] Inconsistent difficulty in maze "
                        f"(maze_index={maze_index_key}, rule_id={rule_id_key!r}): "
                        f"{current_diff} vs {difficulty}. Keep first value."
                    )

            item = {
                "step_id": step,
                "predict": str(pred_row.get("predict", "")).strip(),
                "label": str(pred_row.get("label", "")).strip(),
                "predict_action": pred_code,
                "label_action": label_code,
                "trajectory_source": trajectory_source,
            }
            grouped_item["entries"].append(item)
            grouped_item["by_source"][trajectory_source].append(item)

        maze_details: list[dict[str, Any]] = []
        em_count = 0
        pr_sum = 0.0
        branching_maze_total = 0

        difficulty_stats: dict[str, dict[str, float | int]] = {
            level: {"maze_total": 0, "maze_em_sum": 0, "maze_pr_sum": 0.0}
            for level in DIFFICULTY_LEVELS
        }

        for maze_key in sorted(grouped.keys()):
            maze_group = grouped[maze_key]
            entries = sorted(maze_group["entries"], key=lambda x: int(x["step_id"]))
            difficulty = maze_group["difficulty"]
            maze_index_key = int(maze_group["maze_index"])
            rule_id_key = str(maze_group["rule_id"])
            by_source: dict[str, list[dict[str, Any]]] = {
                source: sorted(source_entries, key=lambda x: int(x["step_id"]))
                for source, source_entries in dict(maze_group["by_source"]).items()
            }

            branch_mode = "trajectory" in by_source and "wrong_trajectory" in by_source

            if branch_mode:
                branching_maze_total += 1
                trajectory_entries = by_source["trajectory"]
                wrong_entries = by_source["wrong_trajectory"]

                trajectory_pred_commands = flatten_step_commands(trajectory_entries, "predict")
                trajectory_label_commands = flatten_step_commands(trajectory_entries, "label")
                wrong_pred_commands = flatten_step_commands(wrong_entries, "predict")
                wrong_label_commands = flatten_step_commands(wrong_entries, "label")

                decision_pred_move = first_execute_move(trajectory_pred_commands)
                trajectory_first_move = first_execute_move(trajectory_label_commands)
                wrong_first_move = first_execute_move(wrong_label_commands)

                selected_source = "unknown"
                if decision_pred_move is not None and decision_pred_move == trajectory_first_move:
                    selected_source = "trajectory"
                elif decision_pred_move is not None and decision_pred_move == wrong_first_move:
                    selected_source = "wrong_trajectory"

                if selected_source == "trajectory":
                    selected_pred_commands = trajectory_pred_commands
                    selected_label_commands = trajectory_label_commands
                elif selected_source == "wrong_trajectory":
                    selected_pred_commands = wrong_pred_commands
                    selected_label_commands = wrong_label_commands
                else:
                    selected_pred_commands = []
                    selected_label_commands = []

                trajectory_em = full_sequence_em(trajectory_pred_commands, trajectory_label_commands)
                trajectory_pr = (
                    longest_prefix_ratio(trajectory_pred_commands, trajectory_label_commands)
                    if trajectory_label_commands
                    else 0.0
                )
                wrong_em = full_sequence_em(wrong_pred_commands, wrong_label_commands)
                wrong_pr = (
                    longest_prefix_ratio(wrong_pred_commands, wrong_label_commands)
                    if wrong_label_commands
                    else 0.0
                )
                selected_em = full_sequence_em(selected_pred_commands, selected_label_commands)
                selected_pr = (
                    longest_prefix_ratio(selected_pred_commands, selected_label_commands)
                    if selected_label_commands
                    else 0.0
                )
                rounds = [
                    {
                        "round_id": 1,
                        "selected_source": selected_source,
                        "round_em": selected_em,
                        "round_pr": selected_pr,
                        "trajectory_em": trajectory_em,
                        "trajectory_pr": trajectory_pr,
                        "wrong_trajectory_em": wrong_em,
                        "wrong_trajectory_pr": wrong_pr,
                        "num_commands": len(selected_label_commands),
                    }
                ]

                # Branch-mode maze metrics are defined on the correct trajectory.
                # This keeps single-round and multi-round checkpoint metrics consistent.
                maze_em = bool(trajectory_em)
                maze_pr = float(trajectory_pr)
                candidate_trajectories = []
            else:
                pred_codes = [item["predict_action"] for item in entries if item["label_action"] is not None]
                label_codes = [item["label_action"] for item in entries if item["label_action"] is not None]

                if not label_codes:
                    candidate_trajectories: list[list[str]] = []
                else:
                    candidate_trajectories = [label_codes]  # type: ignore[list-item]

                maze_em = False
                maze_pr = 0.0

                for traj in candidate_trajectories:
                    if len(pred_codes) == len(traj) and all(pred_codes[i] == traj[i] for i in range(len(traj))):
                        maze_em = True
                    maze_pr = max(maze_pr, longest_prefix_ratio(pred_codes, traj))
                rounds = []
                selected_source = next(iter(by_source.keys()), "trajectory")

            if maze_em:
                em_count += 1
            pr_sum += maze_pr

            if difficulty in DIFFICULTY_LEVELS:
                stat = difficulty_stats[difficulty]
                stat["maze_total"] = int(stat["maze_total"]) + 1
                stat["maze_em_sum"] = int(stat["maze_em_sum"]) + int(maze_em)
                stat["maze_pr_sum"] = float(stat["maze_pr_sum"]) + maze_pr

            step_outputs = []
            for item in entries:
                step_action_match = item["predict_action"] == item["label_action"]
                step_outputs.append(
                    {
                        "step_id": item["step_id"],
                        "predict": item["predict"],
                        "label": item["label"],
                        "predict_action": item["predict_action"],
                        "label_action": item["label_action"],
                        "trajectory_source": item.get("trajectory_source"),
                        "step_action_match": step_action_match,
                    }
                )

            maze_details.append(
                {
                    "maze_index": maze_index_key,
                    "rule_id": rule_id_key,
                    "maze_key": f"{maze_index_key}::{rule_id_key}",
                    "difficulty": difficulty,
                    "num_steps": len(entries),
                    "branch_mode": branch_mode,
                    "selected_source": selected_source,
                    "maze_em": maze_em,
                    "maze_pr": maze_pr,
                    "rounds": rounds,
                    "num_optimal_trajectories": len(candidate_trajectories),
                    "step_outputs": step_outputs,
                }
            )

        maze_total = len(maze_details)
        maze_em = (em_count / maze_total) if maze_total > 0 else 0.0
        maze_pr = (pr_sum / maze_total) if maze_total > 0 else 0.0

        print(f"Maze total: {maze_total}")
        print(f"Maze Exact Match (EM): {em_count}/{maze_total} = {maze_em:.4f}")
        print(f"Maze Progress Rate (PR): {maze_pr:.4f}")

        by_difficulty_metrics: dict[str, dict[str, float | int]] = {}
        em_macro_sum = 0.0
        pr_macro_sum = 0.0
        macro_count = 0
        for level in DIFFICULTY_LEVELS:
            stat = difficulty_stats[level]
            level_total = int(stat["maze_total"])
            level_em = (int(stat["maze_em_sum"]) / level_total) if level_total > 0 else 0.0
            level_pr = (float(stat["maze_pr_sum"]) / level_total) if level_total > 0 else 0.0
            by_difficulty_metrics[level] = {
                "maze_total": level_total,
                "maze_em": level_em,
                "maze_pr": level_pr,
            }
            print(f"Maze {level}: total={level_total}, EM={level_em:.4f}, PR={level_pr:.4f}")

            if level_total > 0:
                em_macro_sum += level_em
                pr_macro_sum += level_pr
                macro_count += 1

        maze_em_macro = (em_macro_sum / macro_count) if macro_count > 0 else 0.0
        maze_pr_macro = (pr_macro_sum / macro_count) if macro_count > 0 else 0.0
        print(
            "Maze macro average across difficulties (Easy/Medium/Hard, non-empty): "
            f"EM={maze_em_macro:.4f}, PR={maze_pr_macro:.4f}"
        )

        metrics.update(
            {
                "maze_total": maze_total,
                "maze_em": maze_em,
                "maze_pr": maze_pr,
                "branching_maze_total": branching_maze_total,
                "maze_by_difficulty": by_difficulty_metrics,
                "maze_em_macro_avg": maze_em_macro,
                "maze_pr_macro_avg": maze_pr_macro,
            }
        )

        if args.save_maze_details is not None:
            args.save_maze_details.parent.mkdir(parents=True, exist_ok=True)
            with args.save_maze_details.open("w", encoding="utf-8") as f:
                for item in maze_details:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved maze-level details to: {args.save_maze_details}")

    if args.save_errors is not None:
        args.save_errors.parent.mkdir(parents=True, exist_ok=True)
        with args.save_errors.open("w", encoding="utf-8") as f:
            for item in errors:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(errors)} mismatches to: {args.save_errors}")

    if args.save_metrics_json is not None:
        args.save_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_metrics_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Saved aggregate metrics to: {args.save_metrics_json}")


if __name__ == "__main__":
    main()
