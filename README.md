# Rule-Compliant Visual Spatial Planning for Multimodal Large Language Models

[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-blue)](https://huggingface.co/datasets/Fish-03/RuleMaze)
[![Paper](https://img.shields.io/badge/Paper-Project%20Page-green)](https://fish-03.github.io/RULEMAZE/paper)
[![Demo Page](https://img.shields.io/badge/Demo%20Page-Online-purple)](https://fish-03.github.io/RULEMAZE/)

Official implementation for **RuleMaze**, a controllable benchmark and training pipeline for studying rule-compliant visual spatial planning in multimodal large language models (MLLMs).

## Introduction

RuleMaze evaluates whether MLLMs can navigate visual maze environments while obeying explicit natural-language rules. The task requires a model to jointly perceive the spatial layout, interpret rule constraints, plan multi-step actions, and verify that the generated trajectory remains rule-compliant.

The project provides:

- A scalable **Language-Logic-Function Hybridization** pipeline that generates natural-language rules, formalizes them into logical constraints, and synthesizes executable Python validators.
- Procedurally generated visual maze environments with rule-matched positive and negative trajectories.
- **Disentangled Multimodal Planning (DMP)** data and scripts that separate perception, action execution, and rule verification into interpretable reasoning primitives.
- Conversion, training, and evaluation utilities for LLaMA-Factory.

RuleMaze currently supports two scene types:

- `regular`: grid mazes with symbolic visual legends.
- `quest`: adventure-style mazes with richer object and tool semantics.

## Highlights

- Rule-conditioned maze generation with staged rule creation, validator synthesis, maze-pool generation, and rule-maze matching.
- Multimodal trajectory data with step-wise maze images and action traces.
- Positive and negative reasoning paths for training with correct trajectories, wrong trajectories, or both.
- LLaMA-Factory integration for ShareGPT-style SFT conversion, LoRA training, prediction, and checkpoint evaluation.
- Seen-rule and unseen-rule evaluation splits across difficulty levels.

## Table of Contents

- [News and TODOs](#news-and-todos)
- [Data Download](#data-download)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
  - [Stage 1: Rule-Maze Generation](#stage-1-rule-maze-generation)
  - [Stage 2: Dataset Construction](#stage-2-dataset-construction)
  - [Stage 3: DMP Conversion, Training, and Evaluation](#stage-3-dmp-conversion-training-and-evaluation)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## News and TODOs

- [ ] Release public dataset download links.
- [ ] Release pretrained RuleMaze model checkpoints.
- [ ] Release paper and project-page links.


## Data Download

Coming soon.

## Installation

### Requirements

- Python 3.10+
- Conda
- LLaMA-Factory for stage-3 SFT training and prediction

### Environment Setup

```bash
git clone <repo_url>
cd RULEMAZE

conda create -n rulemaze python=3.10 -y
conda activate rulemaze
```

Install the Python dependencies required by your environment. If your release package includes a `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

Stage-3 training and prediction use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Please install LLaMA-Factory before running SFT or checkpoint evaluation.

## Configuration

Copy the example config and update the absolute paths:

```bash
cp DataGeneration/path_setting/local_setting_example.yml \
   DataGeneration/path_setting/local_setting.yml
```

Configure paths in:

```text
DataGeneration/path_setting/local_setting.yml
```

Important fields include:

| Key | Description |
| --- | --- |
| `BASED_DIR` | Absolute path to the RuleMaze repository. |
| `MODEL_DIR` | Directory for local model files. |
| `DATA_ROOT_DIR` | Root directory for generated artifacts, usually `DATA`. |
| `MAZE_SIZE` | Maze grid size used by generation and matching. |
| `NUM_MAZES` | Number of mazes to generate in the maze pool. |
| `NUM_PROCESSES` | Number of parallel workers for maze-pool generation. |
| `SCENE_DIR` | Scene mapping for `regular` and `quest`. |
| `RULEMAZE_DATASET_DIR` | Stage-2 dataset output directory. |
| `DMP_DATA_DIR` | Converted SFT data directory. |
| `DMP_TRAINING_RESULT_DIR` | Training checkpoints and evaluation outputs. |

All generated artifacts are stored under:

```text
${BASED_DIR}/${DATA_ROOT_DIR}/
```

If you use LLM-based rule or validator generation, add your API key to:

```text
DataGeneration/Generate_rule_maze/apikey.yaml
```

Example:

```yaml
API_KEY: your_api_key_here
API_BASE: https://your-api-base/v1
```

## Quick Start

The following example runs the complete `regular` pipeline from data generation to checkpoint evaluation.

### 1. Generate Rule-Maze Data

```bash
cd DataGeneration

python Generate_rule_maze/generate_data_pipeline.py \
  --setting local \
  --state 1 \
  --mode regular \
  --num_iterations 5

python Generate_rule_maze/generate_data_pipeline.py \
  --setting local \
  --state 2 \
  --mode regular \
  --num_rules 20

python Generate_rule_maze/generate_data_pipeline.py \
  --setting local \
  --state 3 \
  --mode regular

python Generate_rule_maze/generate_data_pipeline.py \
  --setting local \
  --state 4 \
  --mode regular

python Generate_rule_maze/generate_data_pipeline.py \
  --setting local \
  --state 5 \
  --mode regular

python Generate_rule_maze/generate_data_pipeline.py \
  --setting local \
  --state 6 \
  --mode regular
```

### 2. Build Datasets and Trajectories

```bash
python Training_Dataset_Preparation/Build_Training_Dataset/build_training_dataset.py \
  --setting local \
  --state 1 \
  --mode regular

python Training_Dataset_Preparation/Build_Training_Dataset/build_training_dataset.py \
  --setting local \
  --state 2 \
  --mode regular

python Training_Dataset_Preparation/Build_Training_Dataset/build_training_dataset.py \
  --setting local \
  --state 3 \
  --mode regular

python Training_Dataset_Preparation/Generate_Training_Trajectories/generate_training_trajectories.py \
  --setting local \
  --scene regular
```

### 3. Convert, Train, and Evaluate

```bash
cd ../DMP

python scripts/prepare_stage3_datasets.py \
  --setting local \
  --scene regular \
  --split all

CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
  examples/train_lora/qwen25vl_3b_maze_lora_sft_both.yaml

CUDA_VISIBLE_DEVICES=0 python scripts/eval_maze_checkpoints.py \
  --predict-yaml examples/inference/qwen25vl_3b_maze_lora_predict_both.yaml \
  --checkpoint-root ../DATA/Training_Result/qwen2.5-vl-3b/lora/maze_sft_both_50 \
  --output-root ../DATA/Training_Result/qwen2.5-vl-3b/lora/maze_checkpoint_eval_both
```

Training takes approximately 12 hours on 4 RTX 4090 GPUs.

## Pipeline

RuleMaze contains three main stages.

```text
Stage 1: Rule-Maze Generation
  Generate rules -> select rule sets -> generate validators
  -> extract validator code -> generate maze pool -> match rules and mazes

Stage 2: Dataset Construction
  Load matched examples -> split train/test data
  -> combine difficulties -> generate step-wise trajectories

Stage 3: DMP
  Convert to SFT format -> train with LLaMA-Factory
  -> evaluate every checkpoint
```

### Stage 1: Rule-Maze Generation

Working directory:

```bash
cd DataGeneration
```

Main entry:

```bash
python Generate_rule_maze/generate_data_pipeline.py \
  --setting local \
  --state <1-6> \
  --mode regular
```

`--mode` can be `regular` or `quest`. `--setting local` loads `DataGeneration/path_setting/local_setting.yml`.

| State | Function | Main Output |
| --- | --- | --- |
| 1 | Generate natural-language rules with an LLM. | `maze_navigation_rules.json` |
| 2 | Select rule sets by difficulty. | `rule_sets/Easy.json`, `Medium.json`, `Hard.json` |
| 3 | Generate validator-code responses. | `<Difficulty>_with_code.json` |
| 4 | Extract validator code to Python files. | `rules_checking_code_new.py` |
| 5 | Generate maze pools and rendered images. | `Mazes_Pool/<scene>/maze_size_<N>/` |
| 6 | Match rule sets with satisfying and violating mazes. | `matched_mazes_<N>/.../matched_mazes.json` |

Stage 6 is interactive: run it for each target difficulty file when prompted.

### Stage 2: Dataset Construction

Working directory:

```bash
cd DataGeneration
```

Build train/test datasets:

```bash
python Training_Dataset_Preparation/Build_Training_Dataset/build_training_dataset.py \
  --setting local \
  --state <1-3> \
  --mode regular
```

| State | Function | Main Output |
| --- | --- | --- |
| 1 | Load matched maze data and attach split labels. | `saved_raw_train_data.json`, `saved_raw_test_data.json` |
| 2 | Split data by difficulty and evaluation setting. | `train_*`, `test_seen_*`, `test_unseen_*` |
| 3 | Merge all difficulties. | `combined_train_all_difficulties.json`, `combined_test_seen_all_difficulties.json`, `combined_test_unseen_all_difficulties.json` |

Generate step-wise trajectories:

```bash
python Training_Dataset_Preparation/Generate_Training_Trajectories/generate_training_trajectories.py \
  --setting local \
  --scene regular
```

The trajectory script writes `*_traj_with_step_images.jsonl` files and corresponding step images under the configured `trajectories/` directory.

### Stage 3: DMP Conversion, Training, and Evaluation

Working directory:

```bash
cd DMP
```

Convert stage-2 trajectories into LLaMA-Factory SFT data:

```bash
python scripts/prepare_stage3_datasets.py \
  --setting local \
  --scene regular \
  --split all
```

Useful conversion options:

| Option | Description |
| --- | --- |
| `--trajectory-source trajectory` | Use only correct trajectories. |
| `--trajectory-source wrong_trajectory` | Use only wrong trajectories. |
| `--trajectory-source both` | Use both correct and wrong trajectories. |
| `--no-wrong-trajectory-hint` | Disable wrong-trajectory hint samples. |
| `--retain-percent 50.0` | Keep a percentage of records. |
| `--retain-difficulties Easy Medium` | Keep selected difficulties only. |
| `--overwrite` | Rebuild existing converted files. |

Train with LLaMA-Factory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
  examples/train_lora/qwen25vl_3b_maze_lora_sft_both.yaml
```

Training takes approximately 12 hours on 4 RTX 4090 GPUs.

Evaluate all checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_maze_checkpoints.py \
  --predict-yaml examples/inference/qwen25vl_3b_maze_lora_predict_both.yaml \
  --checkpoint-root ../DATA/Training_Result/qwen2.5-vl-3b/lora/maze_sft_both_50 \
  --output-root ../DATA/Training_Result/qwen2.5-vl-3b/lora/maze_checkpoint_eval_both
```

Each checkpoint evaluation directory contains:

```text
checkpoint-XXXX/
|-- generated_predictions.jsonl
|-- maze_metrics.json
`-- maze_details.jsonl
```

Evaluation metrics include exact step match, maze-level exact match, prefix progress rate, and difficulty-wise summaries.

## Project Structure

```text
.
|-- DataGeneration/
|   |-- path_setting/
|   |   `-- local_setting_example.yml
|   |-- Generate_rule_maze/
|   |   |-- generate_data_pipeline.py
|   |   |-- common.py
|   |   |-- generate_maze.py
|   |   |-- LLM_Agent.py
|   |   `-- states/
|   |-- Training_Dataset_Preparation/
|   |   |-- Build_Training_Dataset/
|   |   `-- Generate_Training_Trajectories/
|   |-- Function/
|   |-- Utils/
|   `-- legend_images/
|-- DMP/
|   |-- scripts/
|   `-- examples/
|       |-- train_lora/
|       `-- inference/
|-- DATA/
|   |-- Generate_rule_maze/
|   |-- RuleMaze_Dataset/
|   |-- Training_Data/
|   `-- Training_Result/
|-- readme/
`-- README.md
```

## Acknowledgements



## Citation

If you find RuleMaze useful for your research, please cite our work:

```bibtex
@misc{rulemaze,
  title  = {Rule-Compliant Visual Spatial Planning for Multimodal Large Language Models},
  author = {Chen, Yu and Lei, Ting and Li, Yaoyi and Cai, Jia and Wu, Zhecen and Liu, Yang},
  year   = {2026},
  note   = {Code and dataset release}
}
```
