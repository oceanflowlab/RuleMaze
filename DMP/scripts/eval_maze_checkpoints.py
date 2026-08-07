#!/usr/bin/env python3
"""Evaluate every checkpoint-* directory with eval_maze_checkpoint.py."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all saved maze checkpoints.")
    parser.add_argument(
        "--predict-yaml",
        type=Path,
        required=True,
        help="Prediction yaml passed to `llamafactory-cli train`.",
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
        required=True,
        help="Directory to store per-checkpoint prediction outputs.",
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
        help="Skip LLaMA-Factory prediction and evaluate existing generated_predictions.jsonl files.",
    )
    parser.add_argument(
        "--max-reset-rounds",
        type=int,
        default=0,
        help="Maximum per-sample reset/retry rounds inside each checkpoint evaluation.",
    )
    return parser.parse_args()


def list_checkpoints(checkpoint_root: Path) -> list[Path]:
    checkpoints = [path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir()]
    return sorted(checkpoints, key=lambda path: int(path.name.split("-")[-1]))


def run_checkpoint_eval(
    script_path: Path,
    predict_yaml: Path,
    checkpoint: Path,
    output_dir: Path,
    dataset_file: Path | None,
    skip_predict: bool,
    max_reset_rounds: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script_path),
        "--predict-yaml",
        str(predict_yaml),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
    ]
    if dataset_file is not None:
        command.extend(["--dataset-file", str(dataset_file)])
    if skip_predict:
        command.append("--skip-predict")

    command.extend(["--max-reset-rounds", str(max_reset_rounds)])

    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)

    metrics_file = output_dir / "maze_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metric file not found after evaluation: {metrics_file}")
    with metrics_file.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected metric object in {metrics_file}")
    return metrics


def main() -> None:
    args = parse_args()
    if args.max_reset_rounds < 0:
        raise ValueError("`--max-reset-rounds` must be >= 0.")

    checkpoints = list_checkpoints(args.checkpoint_root)
    if not checkpoints:
        raise ValueError(f"No checkpoint-* directories found in {args.checkpoint_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().with_name("eval_maze_checkpoint.py")
    summary: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        output_dir = args.output_root / checkpoint.name
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.append(
            run_checkpoint_eval(
                script_path=script_path,
                predict_yaml=args.predict_yaml,
                checkpoint=checkpoint,
                output_dir=output_dir,
                dataset_file=args.dataset_file,
                skip_predict=args.skip_predict,
                max_reset_rounds=args.max_reset_rounds,
            )
        )

    summary_file = args.output_root / "summary.json"
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Saved summary to: {summary_file}")


if __name__ == "__main__":
    main()
