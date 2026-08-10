#!/usr/bin/env python3
"""Run prediction and evaluation for every saved LoRA checkpoint.

This script helps preserve test results at each saved training step.
It invokes LLaMA-Factory predict mode on every `checkpoint-*` directory and
optionally runs `scripts/eval_maze_predictions.py` to compute exact-match stats.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all saved maze checkpoints.")
    parser.add_argument(
        "--predict-yaml",
        type=Path,
        default=Path("examples/inference/qwen25vl_3b_maze_lora_predict.yaml"),
        help="Base prediction yaml used by llamafactory-cli.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Training output directory containing checkpoint-* folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("saves/qwen2.5-vl-3b/lora/maze_checkpoint_eval"),
        help="Directory to store per-checkpoint prediction outputs.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip prediction and only evaluate existing generated_predictions.jsonl files.",
    )
    parser.add_argument(
        "--max-reset-rounds",
        type=int,
        default=1,
        help=(
            "Maximum additional re-inference rounds for pending mazes. "
            "Total round budget is 1 + max_reset_rounds."
        ),
    )
    return parser.parse_args()


def list_checkpoints(checkpoint_root: Path) -> list[Path]:
    checkpoints = [path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir()]
    return sorted(checkpoints, key=lambda path: int(path.name.split("-")[-1]))


def run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}.")
    return [row for row in data if isinstance(row, dict)]


def normalize_rule_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def maze_key_from_values(maze_index: Any, rule_id: Any) -> str | None:
    try:
        maze_index_key = int(maze_index)
    except (TypeError, ValueError):
        return None
    return f"{maze_index_key}::{normalize_rule_id(rule_id)}"


def maze_key_from_dataset_row(row: dict[str, Any]) -> str | None:
    return maze_key_from_values(row.get("maze_index"), row.get("rule_id"))


def normalize_prompt_variant(value: Any) -> str:
    if value is None:
        return "standard"
    text = str(value).strip()
    return text or "standard"


def step_id_from_row(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("step_id"))
    except (TypeError, ValueError):
        return None


def trajectory_source_from_row(row: dict[str, Any]) -> str:
    return str(row.get("trajectory_source", "trajectory") or "trajectory").strip() or "trajectory"


def build_round_subset_rows(
    full_dataset_rows: list[dict[str, Any]],
    pending_maze_keys: set[str] | None,
    round_id: int,
) -> list[dict[str, Any]]:
    combo_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    combo_order: list[tuple[str, str, int]] = []

    for row in full_dataset_rows:
        maze_key = maze_key_from_dataset_row(row)
        if maze_key is None:
            continue
        if pending_maze_keys is not None and maze_key not in pending_maze_keys:
            continue

        step_id = step_id_from_row(row)
        if step_id is None:
            continue

        combo = (maze_key, trajectory_source_from_row(row), step_id)
        if combo not in combo_rows:
            combo_order.append(combo)
        combo_rows[combo].append(row)

    selected_rows: list[dict[str, Any]] = []
    for maze_key, source, step_id in combo_order:
        candidates = combo_rows[(maze_key, source, step_id)]
        chosen_row: dict[str, Any] | None = None

        if round_id == 1:
            for row in candidates:
                if normalize_prompt_variant(row.get("prompt_variant")) == "standard":
                    chosen_row = row
                    break
        else:
            if source == "trajectory" and step_id == 1:
                for row in candidates:
                    if normalize_prompt_variant(row.get("prompt_variant")) == "wrong_hint":
                        chosen_row = row
                        break
            if chosen_row is None:
                for row in candidates:
                    if normalize_prompt_variant(row.get("prompt_variant")) == "standard":
                        chosen_row = row
                        break

        if chosen_row is None:
            chosen_row = candidates[0]

        selected_rows.append(chosen_row)

    return selected_rows


def load_predict_yaml_kv(yaml_file: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with yaml_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("###"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if value.startswith(("\"", "'")) and value.endswith(("\"", "'")) and len(value) >= 2:
                value = value[1:-1]
            data[key] = value
    return data


def resolve_eval_dataset_from_predict_yaml(predict_yaml: Path) -> tuple[str, Path]:
    if not predict_yaml.exists():
        raise ValueError(f"predict yaml not found: {predict_yaml}")
    yaml_kv = load_predict_yaml_kv(predict_yaml)
    eval_dataset = str(yaml_kv.get("eval_dataset", "")).strip()
    dataset_dir_text = str(yaml_kv.get("dataset_dir", "")).strip()
    if not eval_dataset:
        raise ValueError(f"Missing `eval_dataset` in {predict_yaml}")
    if not dataset_dir_text:
        raise ValueError(f"Missing `dataset_dir` in {predict_yaml}")
    return eval_dataset, Path(dataset_dir_text)


def resolve_eval_dataset_meta(dataset_dir: Path, eval_dataset: str) -> dict[str, Any] | None:
    dataset_info_file = dataset_dir / "dataset_info.json"
    if not dataset_info_file.exists():
        return None
    try:
        with dataset_info_file.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    except Exception:
        return None
    dataset_meta = dataset_info.get(eval_dataset)
    if isinstance(dataset_meta, dict):
        return dataset_meta
    return None


def build_round_subset_dataset(
    round_output_dir: Path,
    round_id: int,
    base_dataset_name: str,
    base_dataset_meta: dict[str, Any],
    full_dataset_rows: list[dict[str, Any]],
    pending_maze_keys: set[str] | None,
) -> tuple[str, Path, Path, int]:
    subset_rows = build_round_subset_rows(
        full_dataset_rows=full_dataset_rows,
        pending_maze_keys=pending_maze_keys,
        round_id=round_id,
    )
    if not subset_rows:
        raise ValueError("No pending rows found when building subset dataset.")

    subset_dataset_name = f"{base_dataset_name}__round_{round_id}"
    subset_dir = round_output_dir / "subset_dataset"
    subset_dir.mkdir(parents=True, exist_ok=True)

    subset_file_name = f"{subset_dataset_name}.json"
    subset_file = subset_dir / subset_file_name
    with subset_file.open("w", encoding="utf-8") as f:
        json.dump(subset_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    subset_meta = dict(base_dataset_meta)
    subset_meta["file_name"] = subset_file_name
    subset_info = {subset_dataset_name: subset_meta}
    subset_info_file = subset_dir / "dataset_info.json"
    with subset_info_file.open("w", encoding="utf-8") as f:
        json.dump(subset_info, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return subset_dataset_name, subset_dir, subset_file, len(subset_rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
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


def _sorted_source_steps(detail: dict[str, Any], source: str) -> list[dict[str, Any]]:
    steps = detail.get("step_outputs", [])
    filtered = [item for item in steps if str(item.get("trajectory_source", "")).strip() == source]
    return sorted(filtered, key=lambda x: int(x.get("step_id", 0)))


def _source_em(detail: dict[str, Any], source: str) -> bool | None:
    source_steps = _sorted_source_steps(detail, source)
    if not source_steps:
        return None
    return all(bool(item.get("step_action_match", False)) for item in source_steps)


def _source_prefix_ratio(detail: dict[str, Any], source: str) -> float:
    source_steps = _sorted_source_steps(detail, source)
    if not source_steps:
        return 0.0
    prefix = 0
    for item in source_steps:
        if bool(item.get("step_action_match", False)):
            prefix += 1
        else:
            break
    return prefix / len(source_steps)


def _source_last_predict_contains(detail: dict[str, Any], source: str, needle: str) -> bool:
    source_steps = _sorted_source_steps(detail, source)
    if not source_steps:
        return False
    last_predict = str(source_steps[-1].get("predict", ""))
    return needle in last_predict


def needs_rerun(detail: dict[str, Any]) -> bool:
    if not bool(detail.get("branch_mode", False)):
        return False
    trajectory_em = _source_em(detail, "trajectory")
    wrong_em = _source_em(detail, "wrong_trajectory")
    wrong_last_has_verify_rule = _source_last_predict_contains(detail, "wrong_trajectory", "VerifyRule()")
    return trajectory_em is False and wrong_em is True and wrong_last_has_verify_rule


def detail_effective_scores(detail: dict[str, Any]) -> tuple[bool, float]:
    if bool(detail.get("branch_mode", False)):
        trajectory_em = _source_em(detail, "trajectory")
        return bool(trajectory_em is True), _source_prefix_ratio(detail, "trajectory")
    return bool(detail.get("maze_em", False)), float(detail.get("maze_pr", 0.0))


def choose_final_detail(
    maze_key: str,
    details_by_round: dict[int, dict[str, dict[str, Any]]],
    executed_rounds: list[int],
    total_round_budget: int,
) -> tuple[dict[str, Any] | None, int | None]:
    last_seen: tuple[dict[str, Any], int] | None = None
    for round_id in executed_rounds:
        round_details = details_by_round.get(round_id, {})
        detail = round_details.get(maze_key)
        if detail is None:
            continue
        last_seen = (detail, round_id)

        if not bool(detail.get("branch_mode", False)):
            return detail, round_id

        trajectory_em = _source_em(detail, "trajectory")
        wrong_em = _source_em(detail, "wrong_trajectory")
        if trajectory_em is True:
            return detail, round_id

        wrong_last_has_verify_rule = _source_last_predict_contains(detail, "wrong_trajectory", "VerifyRule()")
        if (
            trajectory_em is False
            and wrong_em is True
            and wrong_last_has_verify_rule
            and round_id < total_round_budget
        ):
            continue

        return detail, round_id

    if last_seen is None:
        return None, None
    return last_seen


def aggregate_multi_round_metrics(
    details_by_round: dict[int, dict[str, dict[str, Any]]],
    executed_rounds: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    maze_keys: set[str] = set()
    for round_id in executed_rounds:
        maze_keys.update(details_by_round.get(round_id, {}).keys())

    selected_details: list[dict[str, Any]] = []
    em_sum = 0
    pr_sum = 0.0
    branching_maze_total = 0

    difficulty_levels = ("Easy", "Medium", "Hard")
    difficulty_stats: dict[str, dict[str, float | int]] = {
        level: {"maze_total": 0, "maze_em_sum": 0, "maze_pr_sum": 0.0}
        for level in difficulty_levels
    }

    for maze_key in sorted(maze_keys):
        detail, selected_round = choose_final_detail(
            maze_key=maze_key,
            details_by_round=details_by_round,
            executed_rounds=executed_rounds,
            total_round_budget=max(executed_rounds) if executed_rounds else 1,
        )
        if detail is None:
            continue

        maze_em, maze_pr = detail_effective_scores(detail)
        if maze_em:
            em_sum += 1
        pr_sum += maze_pr

        if bool(detail.get("branch_mode", False)):
            branching_maze_total += 1

        difficulty = detail.get("difficulty")
        if isinstance(difficulty, str) and difficulty in difficulty_levels:
            stat = difficulty_stats[difficulty]
            stat["maze_total"] = int(stat["maze_total"]) + 1
            stat["maze_em_sum"] = int(stat["maze_em_sum"]) + int(maze_em)
            stat["maze_pr_sum"] = float(stat["maze_pr_sum"]) + float(maze_pr)

        merged = dict(detail)
        merged["selected_round"] = selected_round
        merged["maze_em"] = maze_em
        merged["maze_pr"] = maze_pr
        selected_details.append(merged)

    maze_total = len(selected_details)
    maze_em = (em_sum / maze_total) if maze_total > 0 else 0.0
    maze_pr = (pr_sum / maze_total) if maze_total > 0 else 0.0

    by_difficulty: dict[str, dict[str, float | int]] = {}
    em_macro_sum = 0.0
    pr_macro_sum = 0.0
    macro_count = 0
    for level in difficulty_levels:
        stat = difficulty_stats[level]
        level_total = int(stat["maze_total"])
        level_em = (int(stat["maze_em_sum"]) / level_total) if level_total > 0 else 0.0
        level_pr = (float(stat["maze_pr_sum"]) / level_total) if level_total > 0 else 0.0
        by_difficulty[level] = {
            "maze_total": level_total,
            "maze_em": level_em,
            "maze_pr": level_pr,
        }
        if level_total > 0:
            em_macro_sum += level_em
            pr_macro_sum += level_pr
            macro_count += 1

    metrics = {
        "maze_total": maze_total,
        "maze_em": maze_em,
        "maze_pr": maze_pr,
        "branching_maze_total": branching_maze_total,
        "maze_by_difficulty": by_difficulty,
        "maze_em_macro_avg": (em_macro_sum / macro_count) if macro_count > 0 else 0.0,
        "maze_pr_macro_avg": (pr_macro_sum / macro_count) if macro_count > 0 else 0.0,
        "executed_rounds": executed_rounds,
    }
    return metrics, selected_details


def resolve_dataset_file(args: argparse.Namespace) -> Path | None:
    eval_dataset, dataset_dir = resolve_eval_dataset_from_predict_yaml(args.predict_yaml)
    dataset_info_file = dataset_dir / "dataset_info.json"
    if not dataset_info_file.exists():
        print(f"[WARN] dataset_info.json not found: {dataset_info_file}")
        return None

    try:
        with dataset_info_file.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Failed to load dataset info from {dataset_info_file}: {exc}")
        return None

    dataset_meta = dataset_info.get(eval_dataset)
    if not isinstance(dataset_meta, dict):
        print(f"[WARN] Dataset `{eval_dataset}` not found in {dataset_info_file}")
        return None

    file_name = dataset_meta.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        print(f"[WARN] Missing `file_name` for dataset `{eval_dataset}` in {dataset_info_file}")
        return None

    resolved = dataset_dir / file_name
    if not resolved.exists():
        print(f"[WARN] Resolved dataset file does not exist: {resolved}")
        return None

    return resolved


def main() -> None:
    args = parse_args()
    eval_dataset_name, dataset_dir = resolve_eval_dataset_from_predict_yaml(args.predict_yaml)
    checkpoints = list_checkpoints(args.checkpoint_root)
    if not checkpoints:
        raise ValueError(f"No checkpoint-* directories found in {args.checkpoint_root}")

    resolved_dataset_file = resolve_dataset_file(args)
    eval_dataset_meta = resolve_eval_dataset_meta(dataset_dir, eval_dataset_name)
    full_dataset_rows: list[dict[str, Any]] = []
    if resolved_dataset_file is not None:
        full_dataset_rows = load_json_or_jsonl(resolved_dataset_file)

    if resolved_dataset_file is not None:
        print(f"[INFO] Using dataset file for maze-level metrics: {resolved_dataset_file}")
    else:
        print("[INFO] No dataset file resolved, maze-level metrics will be skipped.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, str | int | float | None]] = []

    for checkpoint in checkpoints:
        checkpoint_output_dir = args.output_root / checkpoint.name
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

        total_round_budget = 1 + max(args.max_reset_rounds, 0)
        if resolved_dataset_file is None and total_round_budget > 1:
            print(
                "[WARN] Multi-round re-inference requires maze-level details from --dataset-file. "
                "Falling back to single-round evaluation."
            )
            total_round_budget = 1
        if eval_dataset_meta is None and total_round_budget > 1:
            print(
                "[WARN] Multi-round subset re-inference requires eval_dataset metadata in dataset_info.json. "
                "Later rounds will fall back to full-dataset inference."
            )

        details_by_round: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        executed_rounds: list[int] = []
        last_round_metrics: dict[str, Any] = {}
        pending_maze_keys: set[str] | None = None

        for round_id in range(1, total_round_budget + 1):
            if round_id > 1 and (pending_maze_keys is None or not pending_maze_keys):
                break

            round_output_dir = checkpoint_output_dir / f"round_{round_id}"
            prediction_file = round_output_dir / "generated_predictions.jsonl"
            errors_file = round_output_dir / "errors.jsonl"
            maze_details_file = round_output_dir / "maze_details.jsonl"
            round_metrics_file = round_output_dir / "maze_metrics.json"

            current_eval_dataset_name = eval_dataset_name
            current_dataset_dir = dataset_dir
            current_eval_dataset_file = resolved_dataset_file

            if resolved_dataset_file is not None and eval_dataset_meta is not None:
                (
                    current_eval_dataset_name,
                    current_dataset_dir,
                    current_eval_dataset_file,
                    subset_size,
                ) = build_round_subset_dataset(
                    round_output_dir=round_output_dir,
                    round_id=round_id,
                    base_dataset_name=eval_dataset_name,
                    base_dataset_meta=eval_dataset_meta,
                    full_dataset_rows=full_dataset_rows,
                    pending_maze_keys=None if round_id == 1 else pending_maze_keys,
                )
                print(
                    f"[INFO] Round {round_id}: subset rows={subset_size}, "
                    f"pending mazes={len(pending_maze_keys) if pending_maze_keys else 0}"
                )
            elif round_id > 1 and pending_maze_keys:
                print(
                    f"[WARN] Round {round_id}: cannot build subset dataset, "
                    "fallback to full eval dataset inference."
                )

            if not args.skip_predict:
                round_output_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    "llamafactory-cli",
                    "train",
                    str(args.predict_yaml),
                    f"adapter_name_or_path={checkpoint}",
                    f"eval_dataset={current_eval_dataset_name}",
                    f"dataset_dir={current_dataset_dir}",
                    f"output_dir={round_output_dir}",
                ]
                print("[RUN]", " ".join(command))
                run_command(command)

            if not prediction_file.exists():
                print(f"[WARN] Missing prediction file: {prediction_file}")
                break

            eval_command = [
                "python",
                "scripts/eval_maze_predictions.py",
                str(prediction_file),
                "--save-errors",
                str(errors_file),
                "--save-metrics-json",
                str(round_metrics_file),
            ]
            if current_eval_dataset_file is not None:
                eval_command.extend(["--dataset-file", str(current_eval_dataset_file)])
                eval_command.extend(["--save-maze-details", str(maze_details_file)])

            print("[RUN]", " ".join(eval_command))
            result = subprocess.run(eval_command, check=True, capture_output=True, text=True)
            print(result.stdout)

            executed_rounds.append(round_id)

            if round_metrics_file.exists():
                with round_metrics_file.open("r", encoding="utf-8") as f:
                    loaded_metrics = json.load(f)
                    if isinstance(loaded_metrics, dict):
                        last_round_metrics = loaded_metrics

            if resolved_dataset_file is None or not maze_details_file.exists():
                break

            round_details_rows = load_jsonl(maze_details_file)
            for detail in round_details_rows:
                maze_key = str(detail.get("maze_key", "")).strip()
                if not maze_key:
                    maze_index = detail.get("maze_index")
                    rule_id = str(detail.get("rule_id", "")).strip()
                    maze_key = f"{maze_index}::{rule_id}"
                    detail["maze_key"] = maze_key
                details_by_round[round_id][maze_key] = detail

            pending = {
                maze_key
                for maze_key, detail in details_by_round[round_id].items()
                if needs_rerun(detail)
            }
            print(
                f"[INFO] Round {round_id}: pending rerun mazes = {len(pending)} / "
                f"{len(details_by_round[round_id])}"
            )

            pending_maze_keys = pending
            if not pending:
                break

        metrics_file = checkpoint_output_dir / "maze_metrics.json"
        final_maze_details_file = checkpoint_output_dir / "maze_details.jsonl"

        if details_by_round and executed_rounds:
            merged_metrics, selected_details = aggregate_multi_round_metrics(details_by_round, executed_rounds)
            final_metrics: dict[str, Any] = dict(last_round_metrics)
            final_metrics.update(merged_metrics)
            final_metrics["checkpoint"] = checkpoint.name
            final_metrics["max_reset_rounds"] = args.max_reset_rounds
            final_metrics["round_budget"] = total_round_budget

            with final_maze_details_file.open("w", encoding="utf-8") as f:
                for item in selected_details:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved merged maze details to {final_maze_details_file}")
        else:
            final_metrics = dict(last_round_metrics)
            final_metrics["checkpoint"] = checkpoint.name
            final_metrics["max_reset_rounds"] = args.max_reset_rounds
            final_metrics["round_budget"] = total_round_budget
            final_metrics["executed_rounds"] = executed_rounds

        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)
            f.write("\n")
        summary.append(final_metrics)

    summary_file = args.output_root / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
