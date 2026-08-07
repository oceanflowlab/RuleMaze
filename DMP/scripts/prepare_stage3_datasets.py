#!/usr/bin/env python3
"""Prepare DMP SFT datasets from RuleMaze stage-2 trajectory outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_DMP_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _DMP_ROOT.parent
_DATAGEN_ROOT = _REPO_ROOT / "DataGeneration"

sys.path.append(str(_DATAGEN_ROOT))
sys.path.append(str(_SCRIPT_DIR))

from Utils.utils import get_config  # noqa: E402
from convert_maze_trajectory_to_sft import convert_records, is_training_split, load_records  # noqa: E402
from maze_sft_utils import build_dataset_info_snippet, write_json_or_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert stage-2 RuleMaze trajectory files into DMP SFT datasets."
    )
    parser.add_argument("--setting", type=str, default="local",
                        help="Config setting name from DataGeneration/path_setting")
    parser.add_argument("--scene", type=str, default="regular", choices=("regular", "quest"),
                        help="Scene type to convert")
    parser.add_argument("--split", choices=("all", "train", "test_unseen", "test_seen"),
                        default="all", help="Which split to convert")
    parser.add_argument("--trajectory-source", choices=("trajectory", "wrong_trajectory", "both"),
                        default="both", help="Which trajectory fields to convert")
    parser.add_argument("--no-wrong-trajectory-hint", action="store_true",
                        help="Do not add wrong-trajectory hints to correct-trajectory training samples")
    parser.add_argument("--use-step0-image-path", action="store_true",
                        help="Use step-0 image for every converted step")
    parser.add_argument("--retain-percent", type=float, default=100.0,
                        help="Keep only this percentage of records")
    parser.add_argument("--retain-difficulties", nargs="+",
                        choices=("Easy", "Medium", "Hard"), default=None,
                        help="Keep only selected difficulties")
    parser.add_argument("--indent", type=int, default=2,
                        help="JSON indent for converted DMP datasets")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite converted output files if they already exist")
    parser.add_argument("--no-update-dataset-info", action="store_true",
                        help="Do not update the training dataset_info.json")
    return parser.parse_args()


def _resolve_from_base(base_dir: Path, path_value: str | os.PathLike[str]) -> Path:
    path = Path(os.path.expanduser(str(path_value)))
    if path.is_absolute():
        return path
    return base_dir / path


def _data_root(config: dict[str, Any]) -> Path:
    base_dir = _resolve_from_base(_REPO_ROOT, config.get("BASED_DIR", _REPO_ROOT))
    data_root_dir = config.get("DATA_ROOT_DIR", "DATA")
    return _resolve_from_base(base_dir, data_root_dir)


def _trajectory_file_name(config: dict[str, Any], combined_file_name: str) -> str:
    trajectory_file_suffix = config.get(
        "TRAJECTORY_FILE_SUFFIX",
        "_trajectories_without_code_thought.jsonl",
    )
    trajectory_with_images_suffix = config.get(
        "TRAJECTORY_WITH_STEP_IMAGES_SUFFIX",
        "_traj_with_step_images.jsonl",
    )
    intermediate_name = combined_file_name.replace(".json", trajectory_file_suffix)
    return intermediate_name.replace(".jsonl", trajectory_with_images_suffix)


def _split_specs(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        "train": {
            "combined_file": config.get(
                "COMBINED_TRAIN_DATASET_FILE_NAME",
                "combined_train_all_difficulties.json",
            ),
            "output_file": config.get(
                "DMP_TRAIN_SFT_FILE_NAME",
                "maze_train_stepwise_sft.json",
            ),
            "dataset_name": config.get(
                "DMP_TRAIN_DATASET_NAME",
                "maze_train_stepwise_sft",
            ),
        },
        "test_unseen": {
            "combined_file": config.get(
                "COMBINED_TEST_UNSEEN_DATASET_FILE_NAME",
                "combined_test_unseen_all_difficulties.json",
            ),
            "output_file": config.get(
                "DMP_TEST_UNSEEN_SFT_FILE_NAME",
                "maze_test_unseen_stepwise_sft.json",
            ),
            "dataset_name": config.get(
                "DMP_TEST_UNSEEN_DATASET_NAME",
                "maze_test_unseen_stepwise_sft",
            ),
        },
        "test_seen": {
            "combined_file": config.get(
                "COMBINED_TEST_SEEN_DATASET_FILE_NAME",
                "combined_test_seen_all_difficulties.json",
            ),
            "output_file": config.get(
                "DMP_TEST_SEEN_SFT_FILE_NAME",
                "maze_test_seen_stepwise_sft.json",
            ),
            "dataset_name": config.get(
                "DMP_TEST_SEEN_DATASET_NAME",
                "maze_test_seen_stepwise_sft",
            ),
        },
    }


def _selected_splits(split: str) -> list[str]:
    if split == "all":
        return ["train", "test_unseen", "test_seen"]
    return [split]


def _format_retain_percent(value: float) -> str:
    text = f"{value:g}".replace(".", "_")
    return text.replace("-", "minus_")


def _content_variant_suffixes(args: argparse.Namespace) -> list[str]:
    suffixes = []
    if args.no_wrong_trajectory_hint and args.trajectory_source in {"trajectory", "both"}:
        suffixes.append("no_wrong_hint")
    if args.use_step0_image_path:
        suffixes.append("step0_image")
    if args.retain_percent != 100.0:
        suffixes.append(f"retain_{_format_retain_percent(args.retain_percent)}")
    if args.retain_difficulties:
        difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2}
        ordered_difficulties = sorted(args.retain_difficulties, key=difficulty_order.get)
        difficulties = "_".join(difficulty.lower() for difficulty in ordered_difficulties)
        suffixes.append(f"diff_{difficulties}")
    return suffixes


def _name_for_conversion_args(name: str, args: argparse.Namespace) -> str:
    path = Path(name)
    stem = path.stem
    stem = stem + f"_{args.trajectory_source}"

    suffixes = _content_variant_suffixes(args)
    if suffixes:
        stem = stem + "_" + "_".join(suffixes)
    return str(path.with_name(stem + path.suffix))


def _stage2_trajectory_dir(config: dict[str, Any], scene: str) -> Path:
    scene_dir = config.get("SCENE_DIR", {}).get(scene, scene)
    return (
        _data_root(config)
        / config.get("RULEMAZE_DATASET_DIR", "RuleMaze_Dataset")
        / scene_dir
        / config.get("COMBINE_DATASETS_DIR_NAME", "process_datasets")
        / config.get("TRAJECTORIES_DIR_NAME", "trajectories")
    )


def _dmp_data_dir(config: dict[str, Any]) -> Path:
    dmp_root = _resolve_from_base(_REPO_ROOT, config.get("DMP_DIR", "DMP"))
    return _resolve_from_base(dmp_root, config.get("DMP_DATA_DIR", "../DATA/Training_Data"))


def _dataset_info_path(config: dict[str, Any], scene_dmp_data_dir: Path) -> Path:
    return scene_dmp_data_dir / config.get("DMP_DATASET_INFO_FILE_NAME", "dataset_info.json")


def _dataset_file_name(output_path: Path, dataset_dir: Path) -> str:
    try:
        return output_path.relative_to(dataset_dir).as_posix()
    except ValueError:
        return output_path.name


def _update_dataset_info(dataset_info_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as handle:
            dataset_info = json.load(handle)
    else:
        dataset_info = {}
    dataset_info.update(entries)
    with dataset_info_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Updated dataset info: {dataset_info_path}")


def _convert_one(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Trajectory input file not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        print(f"Converted dataset already exists: {output_path}. Skipping conversion.")
        snippet = build_dataset_info_snippet(dataset_name, output_path)
        snippet[dataset_name]["file_name"] = _dataset_file_name(output_path, dataset_dir)
        return snippet

    records = load_records(
        input_path,
        retain_percent=args.retain_percent,
        retain_difficulties=set(args.retain_difficulties) if args.retain_difficulties else None,
    )
    samples = convert_records(
        records=records,
        trajectory_source=args.trajectory_source,
        add_wrong_trajectory_hint=not args.no_wrong_trajectory_hint,
        use_step0_image_path=args.use_step0_image_path,
        only_last_wrong=is_training_split(input_path),
    )
    write_json_or_jsonl(samples, output_path, args.indent)
    print(f"Converted {len(samples)} samples -> {output_path}")
    snippet = build_dataset_info_snippet(dataset_name, output_path)
    snippet[dataset_name]["file_name"] = _dataset_file_name(output_path, dataset_dir)
    return snippet


def main() -> None:
    args = parse_args()
    config = get_config(args.setting)
    trajectory_dir = _stage2_trajectory_dir(config, args.scene)
    dmp_data_dir = _dmp_data_dir(config)
    scene_dmp_data_dir = dmp_data_dir / args.scene

    print(f"Stage-2 trajectory dir: {trajectory_dir}")
    print(f"Training data dir: {dmp_data_dir}")
    print(f"Scene training data dir: {scene_dmp_data_dir}")

    dataset_info_entries: dict[str, dict[str, Any]] = {}
    split_specs = _split_specs(config)
    for split in _selected_splits(args.split):
        spec = split_specs[split]
        input_path = trajectory_dir / _trajectory_file_name(config, spec["combined_file"])
        output_file = _name_for_conversion_args(spec["output_file"], args)
        dataset_name = _name_for_conversion_args(spec["dataset_name"], args)
        output_path = scene_dmp_data_dir / output_file
        dataset_info_entries.update(
            _convert_one(input_path, output_path, dataset_name, scene_dmp_data_dir, args)
        )

    if not args.no_update_dataset_info:
        _update_dataset_info(_dataset_info_path(config, scene_dmp_data_dir), dataset_info_entries)
    else:
        print("Suggested dataset_info.json entries:")
        print(json.dumps(dataset_info_entries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
