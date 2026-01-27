from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from transformers import AutoTokenizer

from utils import (
    build_deciles,
    build_deciles_with_special_tokens,
    clean_and_split_gpt_oss_solution,
    clean_and_split_solution,
)


DATASETS: dict[str, dict[str, str]] = {
    "gpqa": {"prompt_column": "question"},
    "mmlu": {"prompt_column": "prompt"},
}


@dataclass(frozen=True)
class ModelFamily:
    """Bundle model-specific processing details."""

    file_prefix: str
    tokenizer_name: Callable[[str], str]
    clean_fn: Callable[[pd.DataFrame], pd.DataFrame]
    decile_fn: Callable[[str, AutoTokenizer], dict]


MODEL_FAMILIES: dict[str, ModelFamily] = {
    "qwen3": ModelFamily(
        file_prefix="qwen3",
        tokenizer_name=lambda size: f"Qwen/Qwen3-{size.upper()}",
        clean_fn=clean_and_split_solution,
        decile_fn=build_deciles,
    ),
    "gpt_oss": ModelFamily(
        file_prefix="gpt_oss",
        tokenizer_name=lambda size: f"openai/gpt-oss-{size}",
        clean_fn=clean_and_split_gpt_oss_solution,
        decile_fn=build_deciles_with_special_tokens,
    ),
}


@dataclass(frozen=True)
class RunFile:
    """Metadata extracted from a reasoning trace JSONL filename."""

    dataset: str
    model_family: str
    model_size: str
    run: int
    path: Path

    @property
    def model_slug(self) -> str:
        """Underscore-separated model string used in output file names."""
        family_prefix = MODEL_FAMILIES[self.model_family].file_prefix
        return f"{family_prefix}_{self.model_size}"


def compile_pattern(dataset: str, family: ModelFamily) -> re.Pattern[str]:
    """
    Build a regex that extracts model size and optional run id from file names.

    Example match: gpqa_qwen3_14b_run2_reasoning_chains.jsonl
                   mmlu_gpt_oss_20b_reasoning_chains.jsonl
    """
    return re.compile(
        rf"^{dataset}_{family.file_prefix}_(?P<size>[^_]+)"
        r"(?:_run(?P<run>\d+))?_reasoning_chains\.jsonl$"
    )


def discover_runs(
    dataset: str,
    families: set[str],
    input_root: Path,
) -> list[RunFile]:
    """Locate all matching reasoning trace files for the dataset/families."""
    dataset_dir = input_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {dataset_dir}")

    discovered: list[RunFile] = []
    for family_name, family in MODEL_FAMILIES.items():
        if family_name not in families:
            continue

        pattern = compile_pattern(dataset, family)
        for path in dataset_dir.glob(f"{dataset}_{family.file_prefix}_*_reasoning_chains.jsonl"):
            match = pattern.match(path.name)
            if not match:
                continue

            size = match.group("size")
            run_str = match.group("run")
            run = int(run_str) if run_str else 1
            discovered.append(
                RunFile(
                    dataset=dataset,
                    model_family=family_name,
                    model_size=size,
                    run=run,
                    path=path,
                )
            )

    return sorted(
        discovered,
        key=lambda r: (r.dataset, r.model_family, r.model_size, r.run, r.path)
    )


def load_dataset(dataset: str, data_root: Path) -> pd.DataFrame:
    """Load only id + prompt columns for the chosen dataset."""
    cfg = DATASETS[dataset]
    dataset_path = data_root / f"{dataset}.parquet"
    df = pd.read_parquet(dataset_path, columns=["id", cfg["prompt_column"]])
    df["id"] = df["id"].astype(str)
    return df


def process_run(
    run_file: RunFile,
    dataset_rows: pd.DataFrame,
    tokenizer_cache: dict[str, AutoTokenizer],
    output_root: Path,
) -> Path:
    """Build decile slices for one run and write parquet to disk."""
    family = MODEL_FAMILIES[run_file.model_family]

    tokenizer_name = family.tokenizer_name(run_file.model_size)
    if tokenizer_name not in tokenizer_cache:
        tokenizer_cache[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = tokenizer_cache[tokenizer_name]

    try:
        responses = pd.read_json(run_file.path, lines=True)
    except ValueError as exc:
        raise ValueError(f"Failed to read JSONL {run_file.path}: {exc}") from exc

    if responses.empty:
        raise ValueError(f"No responses found in {run_file.path}")
    if "id" not in responses.columns:
        raise KeyError(f"Expected an 'id' column in {run_file.path}")

    responses["id"] = responses["id"].astype(str)

    if "run" in responses.columns:
        responses["run"] = pd.to_numeric(
            responses["run"], errors="coerce"
        ).fillna(run_file.run).astype(int)
        runs = responses["run"].unique()
        if len(runs) != 1:
            raise ValueError(f"{run_file.path}: multiple run ids found: {runs[:10]}")
        run_value = int(runs[0])
    else:
        run_value = run_file.run
        responses["run"] = run_value
    if "solution" not in responses and "response" in responses:
        responses = responses.rename(columns={"response": "solution"})

    dup_count = responses["id"].duplicated().sum()
    if dup_count:
        print(f"WARNING {run_file.path.name}: {dup_count} duplicate ids found; keeping last occurrence.")
        responses = responses.drop_duplicates(subset=["id"], keep="last")

    # Normalize raw generations into separate reasoning and final columns.
    responses = family.clean_fn(responses)

    # Join the source prompt for context (outer join keeps any stray ids).
    dataset_rows = dataset_rows.copy()
    dataset_rows["id"] = dataset_rows["id"].astype(str)
    prompt_col = DATASETS[run_file.dataset]["prompt_column"]
    merged = responses.merge(dataset_rows, on="id", how="left")

    missing_prompts = merged[prompt_col].isna()
    if missing_prompts.any():
        print(
            f"WARNING {run_file.path.name}: {missing_prompts.sum()} ids missing from dataset parquet."
        )
        merged.loc[missing_prompts, prompt_col] = ""

    # Token-level slicing of the reasoning chain at each decile.
    decile_dicts = merged["thoughts"].apply(lambda s: family.decile_fn(s, tokenizer))
    deciles_df = pd.DataFrame(list(decile_dicts))

    out_df = pd.concat([merged, deciles_df], axis=1)

    output_path = (
        output_root
        / run_file.dataset
        / f"{run_file.dataset}_{run_file.model_slug}_run{run_value}_reasoning_deciles.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create decile-sliced reasoning traces for multiple datasets/models."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        nargs="+",
        default=list(DATASETS.keys()),
        help="Datasets to process (default: all).",
    )
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILIES.keys(),
        nargs="+",
        default=list(MODEL_FAMILIES.keys()),
        help="Model families to include (default: both).",
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/reasoning_traces",
        help="Root directory containing <dataset>/<dataset>_*_reasoning_chains.jsonl files.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/datasets",
        help="Root directory containing <dataset>.parquet files used for joins.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/deciles",
        help="Root directory where decile parquet files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir)
    data_root = Path(args.data_dir)
    output_root = Path(args.output_dir)

    tokenizer_cache: dict[str, AutoTokenizer] = {}
    dataset_cache: dict[str, pd.DataFrame] = {}

    families = set(args.model_family)
    run_files: list[RunFile] = []
    for dataset in args.dataset:
        run_files.extend(discover_runs(dataset, families, input_root))

    if not run_files:
        raise FileNotFoundError(
            f"No reasoning chain files found under {input_root} for datasets {sorted(args.dataset)}"
        )

    for run_file in run_files:
        if run_file.dataset not in dataset_cache:
            dataset_cache[run_file.dataset] = load_dataset(run_file.dataset, data_root)
        out_path = process_run(
            run_file,
            dataset_cache[run_file.dataset],
            tokenizer_cache,
            output_root,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
