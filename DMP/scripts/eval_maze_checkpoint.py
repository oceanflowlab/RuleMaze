#!/usr/bin/env python3
"""Run prediction and evaluation for one saved LoRA checkpoint.

This is the single-checkpoint companion to `eval_maze_checkpoints.py`.
It uses the same dataset subset, branch rerun, and metric merge logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_maze_checkpoints import (
    aggregate_multi_round_metrics,
    build_round_subset_dataset,
    load_json_or_jsonl,
    load_jsonl,
    needs_rerun,
    resolve_dataset_file,
    resolve_eval_dataset_from_predict_yaml,
    resolve_eval_dataset_meta,
    run_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one saved maze checkpoint.")
    parser.add_argument(
        "--predict-yaml",
        type=Path,
        default=Path("examples/inference/qwen25vl_3b_maze_lora_predict.yaml"),
        help="Base prediction yaml used by llamafactory-cli.",
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
        help="Directory to store prediction outputs and metric files.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Converted SFT dataset used for prediction. Defaults to dataset_info resolution in the yaml.",
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


def main() -> None:
    args = parse_args()
    if args.max_reset_rounds < 0:
        raise ValueError("`--max-reset-rounds` must be >= 0.")

    eval_dataset_name, dataset_dir = resolve_eval_dataset_from_predict_yaml(args.predict_yaml)
    resolved_dataset_file = args.dataset_file or resolve_dataset_file(args)
    eval_dataset_meta = resolve_eval_dataset_meta(dataset_dir, eval_dataset_name)
    eval_script = Path(__file__).resolve().with_name("eval_maze_predictions.py")

    full_dataset_rows: list[dict[str, Any]] = []
    if resolved_dataset_file is not None:
        full_dataset_rows = load_json_or_jsonl(resolved_dataset_file)

    if resolved_dataset_file is not None:
        print(f"[INFO] Using dataset file for maze-level metrics: {resolved_dataset_file}")
    else:
        print("[INFO] No dataset file resolved, maze-level metrics will be skipped.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

        round_output_dir = args.output_dir / f"round_{round_id}"
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
                f"adapter_name_or_path={args.checkpoint}",
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
            sys.executable,
            str(eval_script),
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

    metrics_file = args.output_dir / "maze_metrics.json"
    final_maze_details_file = args.output_dir / "maze_details.jsonl"

    if details_by_round and executed_rounds:
        merged_metrics, selected_details = aggregate_multi_round_metrics(details_by_round, executed_rounds)
        final_metrics: dict[str, Any] = dict(last_round_metrics)
        final_metrics.update(merged_metrics)
        final_metrics["checkpoint"] = args.checkpoint.name
        final_metrics["checkpoint_path"] = str(args.checkpoint)
        final_metrics["max_reset_rounds"] = args.max_reset_rounds
        final_metrics["round_budget"] = total_round_budget

        with final_maze_details_file.open("w", encoding="utf-8") as f:
            for item in selected_details:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved merged maze details to {final_maze_details_file}")
    else:
        final_metrics = dict(last_round_metrics)
        final_metrics["checkpoint"] = args.checkpoint.name
        final_metrics["checkpoint_path"] = str(args.checkpoint)
        final_metrics["max_reset_rounds"] = args.max_reset_rounds
        final_metrics["round_budget"] = total_round_budget
        final_metrics["executed_rounds"] = executed_rounds

    if resolved_dataset_file is not None:
        final_metrics["dataset_file"] = str(resolved_dataset_file)

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved metrics to {metrics_file}")


if __name__ == "__main__":
    main()
