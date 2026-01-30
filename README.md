# Probing the Trajectories of Reasoning Traces in Large Language Models

This repository contains the code for reproducing the experiments in the paper *"Probing the Trajectories of Reasoning Traces in Large Language Models"*.

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
├── results/                    # Processed results, figures, and tables (via anonymous link)
│   ├── figures/                # Generated figures (PNG)
│   └── tables/                 # Generated tables (LaTeX/CSV)
└── requirements.txt
```

## Data and Outputs

Due to the size of the results, data, and outputs (~12GB), they are provided via an anonymous link:

**[Data and Outputs](https://figshare.com/s/68347c2a5cc02cdf9d08)**

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
- pandas (pyarrow)
- matplotlib
- seaborn
- transformers
- statsmodels

### Models

The experiments use the following models:
- **Qwen3 family**: Qwen3-4B, Qwen3-8B, Qwen3-14B (context: 32,768 tokens)
- **gpt-oss family**: gpt-oss-20b, gpt-oss-120b (context: 131,072 tokens)
