# Probing the Trajectories of Reasoning Traces in Large Language Models

This repository contains the code for reproducing the experiments in the paper *"Probing the Trajectories of Reasoning Traces in Large Language Models"* (ICML 2026 submission).

## Abstract

Large language models (LLMs) increasingly solve difficult problems by producing "reasoning traces" before emitting a final response. However, it remains unclear how accuracy and decision commitment evolve along a reasoning trajectory, and whether intermediate trace segments provide answer-relevant information beyond generic length or stylistic effects. We propose a protocol to systematically probe the trajectories of reasoning traces in LLMs by (1) generating a model's reasoning trace, (2) truncating it at fixed token-percentiles, and (3) injecting each partial trace back into the model (or a different model) to measure the induced distribution over answer choices via next-token probabilities.

We apply this protocol to the open-source Qwen3-4B/-8B/-14B and gpt-oss-20b/-120b models across the GPQA Diamond and MMLU-Pro benchmarks. We find that accuracy and decision commitment consistently increase as the percentage of provided reasoning tokens grows. These gains are primarily driven by relevant content in the model generation rather than context length or generic "reasoning style" effects. Stronger models often backtrack successfully from incorrect partial traces, but immediate answers often remain anchored in the weaker model's incorrect response.

## Repository Structure

```
.
├── scripts/                    # All experiment and analysis scripts
│   ├── 01_generate_reasoning_traces.py
│   ├── 02_create_reasoning_deciles.py
│   ├── 03_score_reasoning_deciles.py
│   ├── 04_control_decile_probes.py
│   ├── 05_transfer_rescue.py
│   ├── 06_process_outputs.py
│   ├── 07_process_controls.py
│   ├── 08_process_rescue.py
│   ├── 09_process_ablation_outputs.py
│   ├── prompts.py
│   ├── utils.py
│   ├── figures/                # Figure generation scripts
│   └── tables/                 # Table generation scripts
├── data/                       # Datasets and processed deciles (via anonymous link)
│   ├── datasets/               # GPQA Diamond and MMLU-Pro parquet files
│   └── deciles/                # Pre-computed reasoning deciles per model/run
├── outputs/                    # Raw experiment outputs (via anonymous link)
│   ├── reasoning_traces/       # Full reasoning traces
│   ├── logprobs/               # Decile probing log-probabilities
│   ├── logprobs_control/       # Control condition log-probabilities
│   └── rescue/                 # Cross-model transfer outputs
├── results/                    # Processed results, figures, and tables
│   ├── figures/                # Generated figures (PNG)
│   └── tables/                 # Generated tables (LaTeX/CSV)
└── requirements.txt
```

## Data and Outputs

Due to the size of the results, data, and outputs (~10GB), they are provided via an anonymous link:

**[Anonymous Link to Data and Outputs]**

After downloading, place the `results/`, `data/`, and `outputs/` folders in the repository root.

### Datasets

- **GPQA Diamond** (n=198): Graduate-level multiple-choice questions in biology, chemistry, and physics. Source: [fingertap/GPQA-Diamond](https://huggingface.co/datasets/fingertap/GPQA-Diamond)
- **MMLU-Pro** (n=12,032): Enhanced MMLU benchmark with 10 answer choices across 14 subject categories. Source: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)

## Requirements

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

**For experiments (requires GPU):**
- vLLM (tested with v0.12.0)
- PyTorch with CUDA support

**For figures and tables:**
- pandas
- matplotlib
- seaborn
- transformers
- statsmodels

### Models

The experiments use the following models:
- **Qwen3 family**: Qwen3-4B, Qwen3-8B, Qwen3-14B (context: 32,768 tokens)
- **gpt-oss family**: gpt-oss-20b, gpt-oss-120b (context: 131,072 tokens)

## Reproducing the Experiments

### Pipeline Overview

The experimental pipeline consists of 9 scripts executed sequentially:

#### Step 1: Generate Reasoning Traces
```bash
python scripts/01_generate_reasoning_traces.py \
    --model-name qwen3_4b \
    --dataset gpqa \
    --run-id 1
```
Generates full reasoning traces for each model/dataset/run combination.

#### Step 2: Create Reasoning Deciles
```bash
python scripts/02_create_reasoning_deciles.py \
    --dataset gpqa \
    --model-family qwen3 \
    --model-size 4b \
    --run 1
```
Slices reasoning traces into token-based deciles (10%, 20%, ..., 100%).

#### Step 3: Score Reasoning Deciles
```bash
python scripts/03_score_reasoning_deciles.py \
    --dataset gpqa \
    --model-key qwen3_4b \
    --run 1
```
Probes each decile to extract next-token log-probabilities over answer choices.

#### Step 4: Control Experiments
```bash
python scripts/04_control_decile_probes.py \
    --dataset gpqa \
    --model-key qwen3_4b \
    --run 1 \
    --control-type random  # or: swap, shuffle
```
Runs control conditions (random tokens, trace-swap, token-shuffle) to isolate semantic content effects.

#### Step 5: Cross-Model Transfer (Rescue Experiments)
```bash
python scripts/05_transfer_rescue.py \
    --dataset gpqa \
    --base-model qwen3_4b \
    --target-model qwen3_14b \
    --run 1
```
Injects partial traces from weaker models into stronger models to measure rescue and anchoring rates.

#### Steps 6-9: Process Results
```bash
python scripts/06_process_outputs.py      # Main decile results
python scripts/07_process_controls.py     # Control experiment results
python scripts/08_process_rescue.py       # Rescue experiment results
python scripts/09_process_ablation_outputs.py  # Ablation results
```
Aggregates raw outputs into processed parquet files for analysis.

### Generating Figures and Tables

After processing, generate figures and tables:

```bash
# Main figures
python scripts/figures/main_figure_2.py
python scripts/figures/main_figure_3.py
python scripts/figures/main_figure_4.py

# Appendix figures
python scripts/figures/appendix_accuracy_length.py
python scripts/figures/appendix_confidence.py
python scripts/figures/appendix_mmlu_categories.py
python scripts/figures/appendix_non_choice.py
python scripts/figures/appendix_qwen_ablation.py
python scripts/figures/appendix_trajectories.py

# Tables
python scripts/tables/summary_statistics.py
python scripts/tables/rescue_summary.py
python scripts/tables/appendix_variance.py
python scripts/tables/qwen8B_issue.py
```

Generated outputs are saved to `results/figures/` and `results/tables/`.
