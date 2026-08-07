#!/usr/bin/env python3
"""Run prediction and metric evaluation for one maze checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from maze_sft_utils import (
    DIFFICULTY_LEVELS,
    build_dataset_info_snippet,
    load_json_or_jsonl,
    load_jsonl,
    longest_prefix_ratio,
    normalize_difficulty,
    write_json_or_jsonl,
)

COMMAND_RE = re.compile(
    r"LocateStart\(\)|VerifyRule\(\)|VerifyEnd\(\)|ExecuteMove\(['\"]?\s*([^'\"\n]+?)\s*['\"]?\)",
    re.IGNORECASE,
)
ANSWER_RE = re.compile(r"<ANSWER>\s*(.*?)\s*</ANSWER>", re.IGNORECASE | re.DOTALL)
PATH_ACTION_RE = re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one LLaMA-Factory maze checkpoint.")
    parser.add_argument(
        "--predict-yaml",
        type=Path,
        required=True,
        help="Prediction yaml passed to `llamafactory-cli train`.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory, for example `.../checkpoint-1600`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated predictions and metric files.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Converted SFT dataset used for prediction. If omitted, resolve it from the yaml dataset_info.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip LLaMA-Factory prediction and evaluate an existing generated_predictions.jsonl.",
    )
    parser.add_argument(
        "--max-reset-rounds",
        type=int,
        default=3,
        help="Maximum per-sample reset/retry rounds. Failed maze samples are rerun in later rounds.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def load_predict_yaml_kv(yaml_file: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with yaml_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("###") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if value.startswith(("\"", "'")) and value.endswith(("\"", "'")) and len(value) >= 2:
                value = value[1:-1]
            if key:
                data[key] = value
    return data


def resolve_eval_dataset_from_predict_yaml(predict_yaml: Path) -> tuple[str, Path]:
    if not predict_yaml.exists():
        raise ValueError(f"Prediction yaml not found: {predict_yaml}")

    yaml_kv = load_predict_yaml_kv(predict_yaml)
    eval_dataset = str(yaml_kv.get("eval_dataset", "")).strip()
    dataset_dir_text = str(yaml_kv.get("dataset_dir", "")).strip()
    if not eval_dataset:
        raise ValueError(f"Missing `eval_dataset` in {predict_yaml}")
    if not dataset_dir_text:
        raise ValueError(f"Missing `dataset_dir` in {predict_yaml}")
    return eval_dataset, Path(dataset_dir_text)


def resolve_dataset_file(predict_yaml: Path) -> Path | None:
    eval_dataset, dataset_dir = resolve_eval_dataset_from_predict_yaml(predict_yaml)
    dataset_info_file = dataset_dir / "dataset_info.json"
    if not dataset_info_file.exists():
        return None

    with dataset_info_file.open("r", encoding="utf-8") as handle:
        dataset_info = json.load(handle)
    dataset_meta = dataset_info.get(eval_dataset)
    if not isinstance(dataset_meta, dict):
        return None

    file_name = dataset_meta.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        return None

    dataset_file = dataset_dir / file_name
    return dataset_file if dataset_file.exists() else None


def extract_command_sequence(text: Any) -> list[str]:
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


def extract_first_action_code(text: Any) -> str | None:
    commands = extract_command_sequence(text)
    for command in commands:
        if command.startswith("ExecuteMove("):
            return command
    return commands[0] if commands else None


def extract_path_actions(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []

    answer_match = ANSWER_RE.search(text)
    content = answer_match.group(1) if answer_match else text
    return [token.lower() for token in PATH_ACTION_RE.findall(content)]


def exact_match(pred_items: list[Any], label_items: list[Any]) -> bool:
    return pred_items == label_items


def update_difficulty_stats(
    stats: dict[str, dict[str, float | int]],
    difficulty: str | None,
    em: bool,
    pr: float,
) -> None:
    if difficulty not in DIFFICULTY_LEVELS:
        return
    bucket = stats[difficulty]
    bucket["total"] = int(bucket["total"]) + 1
    bucket["em_sum"] = int(bucket["em_sum"]) + int(em)
    bucket["pr_sum"] = float(bucket["pr_sum"]) + pr


def summarize_difficulty_stats(stats: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for difficulty in DIFFICULTY_LEVELS:
        bucket = stats[difficulty]
        total = int(bucket["total"])
        summary[difficulty] = {
            "total": total,
            "em": (int(bucket["em_sum"]) / total) if total else 0.0,
            "pr": (float(bucket["pr_sum"]) / total) if total else 0.0,
        }
    return summary


def evaluate_raw_path_predictions(
    prediction_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stats: dict[str, dict[str, float | int]] = {
        level: {"total": 0, "em_sum": 0, "pr_sum": 0.0} for level in DIFFICULTY_LEVELS
    }
    details: list[dict[str, Any]] = []
    em_sum = 0
    pr_sum = 0.0

    for index, (pred_row, dataset_row) in enumerate(zip(prediction_rows, dataset_rows)):
        pred_actions = extract_path_actions(pred_row.get("predict", ""))
        label_actions = extract_path_actions(pred_row.get("label", ""))
        difficulty = normalize_difficulty(dataset_row.get("difficulty") or dataset_row.get("_meta_difficulty"))
        em = exact_match(pred_actions, label_actions)
        pr = longest_prefix_ratio(pred_actions, label_actions)

        em_sum += int(em)
        pr_sum += pr
        update_difficulty_stats(stats, difficulty, em, pr)
        details.append(
            {
                "index": index,
                "maze_index": dataset_row.get("maze_index"),
                "rule_id": dataset_row.get("rule_id"),
                "difficulty": difficulty,
                "em": em,
                "pr": pr,
            }
        )

    total = len(prediction_rows)
    metrics = {
        "mode": "raw_path",
        "total": total,
        "em": (em_sum / total) if total else 0.0,
        "pr": (pr_sum / total) if total else 0.0,
        "by_difficulty": summarize_difficulty_stats(stats),
    }
    return metrics, details


def evaluate_stepwise_predictions(
    prediction_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_total = len(prediction_rows)
    exact_text_match = 0
    exact_step_match = 0
    invalid_prediction = 0

    grouped: dict[tuple[int, str], dict[str, Any]] = defaultdict(
        lambda: {"entries": [], "difficulty": None, "maze_index": None, "rule_id": ""}
    )

    for index, (pred_row, sample_row) in enumerate(zip(prediction_rows, dataset_rows)):
        predict_text = str(pred_row.get("predict", "")).strip()
        label_text = str(pred_row.get("label", "")).strip()
        if predict_text == label_text:
            exact_text_match += 1

        pred_code = extract_first_action_code(predict_text)
        label_code = extract_first_action_code(label_text)
        if pred_code is None:
            invalid_prediction += 1
        if pred_code is not None and label_code is not None and pred_code == label_code:
            exact_step_match += 1

        try:
            maze_index = int(sample_row.get("maze_index"))
            step_id = int(sample_row.get("step_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "`dataset_file` rows must include integer-compatible `maze_index` and `step_id` "
                f"for stepwise evaluation. Failed at row {index}."
            ) from exc

        rule_id = str(sample_row.get("rule_id", "") or "").strip()
        trajectory_source = str(sample_row.get("trajectory_source", "trajectory") or "trajectory").strip()
        difficulty = normalize_difficulty(sample_row.get("difficulty"))
        group = grouped[(maze_index, rule_id)]
        group["maze_index"] = maze_index
        group["rule_id"] = rule_id
        if group["difficulty"] is None:
            group["difficulty"] = difficulty
        group["entries"].append(
            {
                "step_id": step_id,
                "trajectory_source": trajectory_source,
                "predict_action": pred_code,
                "label_action": label_code,
                "step_match": pred_code is not None and pred_code == label_code,
            }
        )

    difficulty_stats: dict[str, dict[str, float | int]] = {
        level: {"total": 0, "em_sum": 0, "pr_sum": 0.0} for level in DIFFICULTY_LEVELS
    }
    details: list[dict[str, Any]] = []
    maze_em_sum = 0
    maze_pr_sum = 0.0

    for maze_key in sorted(grouped.keys()):
        group = grouped[maze_key]
        entries = group["entries"]
        sources = {str(item["trajectory_source"]) for item in entries}
        metric_source = "trajectory" if "trajectory" in sources else sorted(sources)[0]
        metric_entries = sorted(
            [item for item in entries if item["trajectory_source"] == metric_source],
            key=lambda item: int(item["step_id"]),
        )
        pred_codes = [item["predict_action"] for item in metric_entries if item["label_action"] is not None]
        label_codes = [item["label_action"] for item in metric_entries if item["label_action"] is not None]
        maze_em = exact_match(pred_codes, label_codes)
        maze_pr = longest_prefix_ratio(pred_codes, label_codes)

        maze_em_sum += int(maze_em)
        maze_pr_sum += maze_pr
        update_difficulty_stats(difficulty_stats, group["difficulty"], maze_em, maze_pr)
        details.append(
            {
                "maze_index": group["maze_index"],
                "rule_id": group["rule_id"],
                "difficulty": group["difficulty"],
                "metric_source": metric_source,
                "num_steps": len(metric_entries),
                "maze_em": maze_em,
                "maze_pr": maze_pr,
            }
        )

    maze_total = len(details)
    metrics = {
        "mode": "stepwise",
        "total": sample_total,
        "exact_text_match": (exact_text_match / sample_total) if sample_total else 0.0,
        "exact_step_match": (exact_step_match / sample_total) if sample_total else 0.0,
        "invalid_prediction_rate": (invalid_prediction / sample_total) if sample_total else 0.0,
        "maze_total": maze_total,
        "maze_em": (maze_em_sum / maze_total) if maze_total else 0.0,
        "maze_pr": (maze_pr_sum / maze_total) if maze_total else 0.0,
        "maze_by_difficulty": summarize_difficulty_stats(difficulty_stats),
    }
    return metrics, details


def evaluate_predictions(
    prediction_file: Path,
    dataset_file: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_rows = load_jsonl(prediction_file)
    dataset_rows = load_json_or_jsonl(dataset_file)
    if len(prediction_rows) != len(dataset_rows):
        raise ValueError(
            "Length mismatch: prediction rows = "
            f"{len(prediction_rows)} but dataset rows = {len(dataset_rows)}."
        )

    if dataset_rows and "step_id" in dataset_rows[0]:
        return evaluate_stepwise_predictions(prediction_rows, dataset_rows)
    return evaluate_raw_path_predictions(prediction_rows, dataset_rows)


def write_outputs(output_dir: Path, metrics: dict[str, Any], details: list[dict[str, Any]]) -> None:
    metrics_file = output_dir / "maze_metrics.json"
    details_file = output_dir / "maze_details.jsonl"

    with metrics_file.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with details_file.open("w", encoding="utf-8") as handle:
        for item in details:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved metrics to: {metrics_file}")
    print(f"Saved details to: {details_file}")


def detail_key(detail: dict[str, Any]) -> tuple[Any, ...]:
    if "maze_index" in detail:
        return (int(detail.get("maze_index")), str(detail.get("rule_id", "") or "").strip())
    return (detail.get("index"),)


def detail_passed(detail: dict[str, Any]) -> bool:
    if "maze_em" in detail:
        return bool(detail.get("maze_em"))
    return bool(detail.get("em"))


def dataset_row_key(row: dict[str, Any], index: int) -> tuple[Any, ...]:
    if "maze_index" in row:
        return (int(row.get("maze_index")), str(row.get("rule_id", "") or "").strip())
    return (index,)


def write_subset_dataset(rows: list[dict[str, Any]], output_dir: Path, source_file: Path, round_id: int) -> tuple[str, Path]:
    dataset_name = f"{source_file.stem}__round_{round_id}"
    subset_dir = output_dir / f"round_{round_id}" / "subset_dataset"
    subset_file = subset_dir / f"{dataset_name}{source_file.suffix or '.json'}"
    write_json_or_jsonl(rows, subset_file)

    dataset_info_file = subset_dir / "dataset_info.json"
    with dataset_info_file.open("w", encoding="utf-8") as handle:
        json.dump(build_dataset_info_snippet(dataset_name, subset_file), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return dataset_name, subset_file


def run_single_round(
    args: argparse.Namespace,
    eval_dataset: str,
    dataset_dir: Path,
    dataset_file: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not args.skip_predict:
        command = [
            "llamafactory-cli",
            "train",
            str(args.predict_yaml),
            f"adapter_name_or_path={args.checkpoint}",
            f"eval_dataset={eval_dataset}",
            f"dataset_dir={dataset_dir}",
            f"output_dir={output_dir}",
        ]
        print("[RUN]", " ".join(command))
        run_command(command)

    prediction_file = output_dir / "generated_predictions.jsonl"
    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

    metrics, details = evaluate_predictions(prediction_file, dataset_file)
    write_outputs(output_dir, metrics, details)
    return metrics, details


def summarize_retry_details(
    details: list[dict[str, Any]],
    round_metrics: list[dict[str, Any]],
    max_reset_rounds: int,
) -> dict[str, Any]:
    if not details:
        return {
            "mode": "retry",
            "total": 0,
            "em": 0.0,
            "pr": 0.0,
            "round_metrics": round_metrics,
            "max_reset_rounds": max_reset_rounds,
            "round_budget": max_reset_rounds + 1,
            "executed_rounds": [item["round_id"] for item in round_metrics],
        }

    if "maze_em" in details[0]:
        difficulty_stats: dict[str, dict[str, float | int]] = {
            level: {"total": 0, "em_sum": 0, "pr_sum": 0.0} for level in DIFFICULTY_LEVELS
        }
        em_sum = 0
        pr_sum = 0.0
        for detail in details:
            em = bool(detail.get("maze_em"))
            pr = float(detail.get("maze_pr", 0.0))
            em_sum += int(em)
            pr_sum += pr
            update_difficulty_stats(difficulty_stats, detail.get("difficulty"), em, pr)

        maze_total = len(details)
        return {
            "mode": "stepwise_retry",
            "maze_total": maze_total,
            "maze_em": em_sum / maze_total,
            "maze_pr": pr_sum / maze_total,
            "maze_by_difficulty": summarize_difficulty_stats(difficulty_stats),
            "round_metrics": round_metrics,
            "max_reset_rounds": max_reset_rounds,
            "round_budget": max_reset_rounds + 1,
            "executed_rounds": [item["round_id"] for item in round_metrics],
        }

    total = len(details)
    em_sum = sum(int(bool(detail.get("em"))) for detail in details)
    pr_sum = sum(float(detail.get("pr", 0.0)) for detail in details)
    return {
        "mode": "raw_path_retry",
        "total": total,
        "em": em_sum / total,
        "pr": pr_sum / total,
        "round_metrics": round_metrics,
        "max_reset_rounds": max_reset_rounds,
        "round_budget": max_reset_rounds + 1,
        "executed_rounds": [item["round_id"] for item in round_metrics],
    }


def run_retry_rounds(
    args: argparse.Namespace,
    initial_dataset_file: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original_rows = load_json_or_jsonl(initial_dataset_file)
    pending_keys = {dataset_row_key(row, index) for index, row in enumerate(original_rows)}
    final_details: dict[tuple[Any, ...], dict[str, Any]] = {}
    round_metrics: list[dict[str, Any]] = []

    for round_id in range(1, args.max_reset_rounds + 2):
        subset_rows = [
            row for index, row in enumerate(original_rows) if dataset_row_key(row, index) in pending_keys
        ]
        if not subset_rows:
            break

        dataset_name, subset_file = write_subset_dataset(
            rows=subset_rows,
            output_dir=args.output_dir,
            source_file=initial_dataset_file,
            round_id=round_id,
        )
        round_output_dir = args.output_dir / f"round_{round_id}"
        metrics, details = run_single_round(
            args=args,
            eval_dataset=dataset_name,
            dataset_dir=subset_file.parent,
            dataset_file=subset_file,
            output_dir=round_output_dir,
        )
        round_metrics.append({"round_id": round_id, **metrics})

        next_pending_keys: set[tuple[Any, ...]] = set()
        for detail in details:
            key = detail_key(detail)
            selected_detail = {**detail, "selected_round": round_id}
            if key not in final_details or not detail_passed(final_details[key]):
                final_details[key] = selected_detail
            if not detail_passed(detail):
                next_pending_keys.add(key)

        pending_keys = next_pending_keys

    details = [final_details[key] for key in sorted(final_details.keys())]
    metrics = summarize_retry_details(details, round_metrics, args.max_reset_rounds)
    return metrics, details


def main() -> None:
    args = parse_args()
    if args.max_reset_rounds < 0:
        raise ValueError("`--max-reset-rounds` must be >= 0.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_dataset, dataset_dir = resolve_eval_dataset_from_predict_yaml(args.predict_yaml)
    dataset_file = args.dataset_file or resolve_dataset_file(args.predict_yaml)
    if dataset_file is None:
        raise ValueError("Could not resolve dataset file. Pass `--dataset-file` explicitly.")

    if args.max_reset_rounds > 0:
        metrics, details = run_retry_rounds(args, dataset_file)
    else:
        metrics, details = run_single_round(
            args=args,
            eval_dataset=eval_dataset,
            dataset_dir=dataset_dir,
            dataset_file=dataset_file,
            output_dir=args.output_dir,
        )

    metrics["checkpoint"] = args.checkpoint.name
    metrics["checkpoint_path"] = str(args.checkpoint)
    metrics["dataset_file"] = str(dataset_file)
    write_outputs(args.output_dir, metrics, details)


if __name__ == "__main__":
    main()
