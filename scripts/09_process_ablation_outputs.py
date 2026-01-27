from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils import DATASETS, MODEL_CONFIGS, compute_decile_flips, dataset_choice_labels

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGPROB_ROOT = REPO_ROOT / "outputs" / "logprobs"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "datasets"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_NAME = "processed_ablation_outputs.parquet"
QWEN_MODELS = tuple(key for key, cfg in MODEL_CONFIGS.items() if cfg.get("spec") == "qwen")


@dataclass(frozen=True)
class AblationLogprobFile:
    dataset: str
    model: str
    run: int
    decile: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate decile logprobs for ablation runs.")
    parser.add_argument(
        "--dataset",
        "--datasets",
        nargs="+",
        dest="datasets",
        choices=DATASETS.keys(),
        default=list(DATASETS.keys()),
        help="Datasets to include.",
    )
    parser.add_argument(
        "--model",
        "--models",
        nargs="+",
        dest="models",
        choices=QWEN_MODELS,
        default=list(QWEN_MODELS),
        help="Qwen models to include.",
    )
    parser.add_argument(
        "--logprob-root",
        type=Path,
        default=DEFAULT_LOGPROB_ROOT,
        help="Directory containing ablation decile logprob JSONL files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing dataset parquet files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Output directory for the consolidated dataframe.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Filename for the consolidated dataframe (extension determines format).",
    )
    return parser.parse_args()


def discover_ablation_logprob_files(
    datasets: Iterable[str],
    models: set[str],
    logprob_root: Path,
) -> list[AblationLogprobFile]:
    discovered: list[AblationLogprobFile] = []
    pattern = re.compile(
        r"^(?P<dataset>[^_]+)_(?P<model>.+)_run(?P<run>\d+)_decile(?P<decile>\d+)_logprobs_ablation\.jsonl$"
    )

    for dataset in datasets:
        dataset_dir = logprob_root / dataset
        if not dataset_dir.exists():
            continue

        for path in dataset_dir.glob(f"{dataset}_*_logprobs_ablation.jsonl"):
            match = pattern.match(path.name)
            if not match:
                continue

            model = match.group("model")
            if model not in models:
                continue

            discovered.append(
                AblationLogprobFile(
                    dataset=dataset,
                    model=model,
                    run=int(match.group("run")),
                    decile=int(match.group("decile")),
                    path=path,
                )
            )

    return sorted(discovered, key=lambda f: (f.dataset, f.model, f.run, f.decile, f.path.name))


def load_ablation_logprobs(files: list[AblationLogprobFile]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for entry in files:
        df = pd.read_json(entry.path, lines=True)
        if df.empty:
            continue
        if "id" not in df.columns:
            raise KeyError(f"{entry.path} missing required column 'id'.")

        df["dataset"] = entry.dataset
        df["model"] = entry.model
        df["run"] = int(entry.run)
        df["decile"] = int(entry.decile)
        df["id"] = df["id"].astype(str)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_dataset_tables(datasets: Iterable[str], data_root: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        path = data_root / f"{dataset}.parquet"
        if not path.exists():
            print(f"Warning: dataset file not found: {path}")
            continue
        df = pd.read_parquet(path)
        if "id" not in df.columns:
            raise KeyError(f"{path} missing required column 'id'.")
        df["id"] = df["id"].astype(str)
        if "answer" in df.columns:
            answers = df["answer"]
            df["answer"] = answers.where(answers.notna(), pd.NA).apply(
                lambda x: x if pd.isna(x) else str(x).strip().upper()
            )
        tables[dataset] = df
    return tables


def add_choice_columns(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, list[str]]:
    labels = dataset_choice_labels(dataset)
    records = df["choice_logprobs"].apply(lambda x: x if isinstance(x, dict) else {}).tolist()
    choice_logprob_frame = pd.DataFrame.from_records(records).reindex(columns=labels)
    choice_logprob_frame = choice_logprob_frame.apply(pd.to_numeric, errors="coerce")

    # Recover rows where everything became NA (e.g., non-numeric placeholders) before filling gaps.
    all_na_mask = choice_logprob_frame.isna().all(axis=1)
    df = df.copy()
    df["choice_logprobs_all_na"] = all_na_mask
    if all_na_mask.any():
        recovered_rows: list[dict[str, float | pd.NA]] = []
        for rec in records:
            recovered_rows.append(
                {
                    label: pd.to_numeric(rec.get(label), errors="coerce")
                    if isinstance(rec, dict)
                    else pd.NA
                    for label in labels
                }
            )
        recovered_frame = pd.DataFrame(recovered_rows).reindex(columns=labels)
        choice_logprob_frame.loc[all_na_mask] = recovered_frame.loc[all_na_mask]

    choice_logprob_frame = choice_logprob_frame.fillna(-np.inf)
    choice_logprob_frame.columns = [f"logprob_{label}" for label in labels]

    prob_frame = np.exp(choice_logprob_frame)
    prob_frame.columns = [f"prob_{label}" for label in labels]

    df = df.join(choice_logprob_frame)
    df = df.join(prob_frame)
    return df, prob_frame.columns.tolist()


def lookup_prob(row: pd.Series, column_prefix: str, label: str | float | None) -> float:
    if not isinstance(label, str) or not label:
        return np.nan
    key = f"{column_prefix}{label}"
    return row.get(key, np.nan)


def process_dataset_logprobs(
    df: pd.DataFrame,
    dataset: str,
    dataset_table: pd.DataFrame | None,
) -> pd.DataFrame:
    # Align indexes before expanding choice_logprobs so joins stay row-aligned after filtering.
    df = df.reset_index(drop=True)

    if dataset_table is not None:
        df = df.merge(dataset_table, on="id", how="left", validate="many_to_one")

    df, prob_columns = add_choice_columns(df, dataset)
    df["decile_prediction"] = df[prob_columns].idxmax(axis=1).str.removeprefix("prob_")
    df["decile_prediction_prob"] = df[prob_columns].max(axis=1)
    df["decile_choice_mass"] = df[prob_columns].sum(axis=1)
    invalid_logprobs = df["choice_logprobs_all_na"]

    labels = dataset_choice_labels(dataset)

    def _argmax_choice(rec: dict | None) -> str | pd.NA:
        if not isinstance(rec, dict):
            return pd.NA
        best_label: str | pd.NA = pd.NA
        best_lp = -np.inf
        for label in labels:
            lp = rec.get(label)
            try:
                lp = float(lp)
            except Exception:
                continue
            if lp > best_lp:
                best_lp = lp
                best_label = label
        return best_label

    missing_pred = df["decile_prediction"].isna() & ~invalid_logprobs
    if missing_pred.any():
        df.loc[missing_pred, "decile_prediction"] = df.loc[missing_pred, "choice_logprobs"].apply(_argmax_choice)

    if "answer" in df.columns:
        answers = df["answer"]
        df["answer"] = answers.where(answers.notna(), pd.NA).apply(
            lambda x: x if pd.isna(x) else str(x).strip().upper()
        )
        missing_answer = df["answer"].isna()
        missing_pred_na = df["decile_prediction"].isna()
        valid_acc = ~missing_answer & ~missing_pred_na
        df["decile_accuracy"] = df["decile_prediction"].eq(df["answer"]).where(valid_acc)
        df["decile_prediction_prob_correct_answer"] = df.apply(
            lambda row: lookup_prob(row, "prob_", row.get("answer")),
            axis=1,
        )
        df.loc[~valid_acc, "decile_prediction_prob_correct_answer"] = np.nan
    else:
        df["decile_accuracy"] = np.nan
        df["decile_prediction_prob_correct_answer"] = np.nan

    if invalid_logprobs.any():
        df.loc[invalid_logprobs, "decile_prediction"] = pd.NA
        df.loc[invalid_logprobs, "decile_prediction_prob"] = np.nan
        df.loc[invalid_logprobs, "decile_choice_mass"] = np.nan

    final_answer = (
        df[df["decile"] == 100]
        [["dataset", "model", "run", "id", "decile_prediction"]]
        .rename(columns={"decile_prediction": "decile_final_answer"})
    )
    df = df.merge(
        final_answer,
        on=["dataset", "model", "run", "id"],
        how="left",
        validate="many_to_one",
    )
    df["decile_prediction_prob_final_answer"] = df.apply(
        lambda row: lookup_prob(row, "prob_", row.get("decile_final_answer")),
        axis=1,
    )
    choice_mass = df["decile_choice_mass"]
    zero_mass = choice_mass.isna() | (choice_mass <= 0)
    df["pred_prob_cond"] = np.nan
    df.loc[~zero_mass, "pred_prob_cond"] = (
        df.loc[~zero_mass, "decile_prediction_prob_final_answer"] / choice_mass.loc[~zero_mass]
    )

    df_with_flips, flip_rates = compute_decile_flips(df, ["dataset", "model", "run"])
    df = df_with_flips.merge(
        flip_rates,
        on=["dataset", "model", "run", "decile"],
        how="left",
        validate="many_to_one",
    )

    return df


def save_dataframe(df: pd.DataFrame, results_dir: Path, output_name: str) -> Path:
    """Persist the dataframe to disk; format inferred from extension."""
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / output_name

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        if suffix not in {".parquet", ".pq"}:
            output_path = output_path.with_suffix(".parquet")
        df.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    datasets = args.datasets
    models = set(args.models)

    logprob_files = discover_ablation_logprob_files(datasets, models, args.logprob_root)
    logprob_df = load_ablation_logprobs(logprob_files)
    dataset_tables = load_dataset_tables(datasets, args.data_root)

    processed_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        df_dataset = logprob_df[logprob_df["dataset"] == dataset]
        if df_dataset.empty:
            continue
        processed_df = process_dataset_logprobs(df_dataset, dataset, dataset_tables.get(dataset))
        missing_logprobs = int(processed_df.get("choice_logprobs_all_na", pd.Series(False)).sum())
        unlabeled = int(processed_df["answer"].isna().sum()) if "answer" in processed_df.columns else 0
        if missing_logprobs or unlabeled:
            print(
                f"{dataset}: flagged {missing_logprobs} rows with missing choice_logprobs; "
                f"{unlabeled} unlabeled rows excluded from accuracy."
            )
        processed_df = processed_df.drop(columns=["choice_logprobs_all_na"], errors="ignore")
        processed_frames.append(processed_df)

    if not processed_frames:
        raise RuntimeError("No ablation logprob data found for the requested datasets/models.")

    combined_df = pd.concat(processed_frames, ignore_index=True)
    combined_df.sort_values(["dataset", "model", "run", "id", "decile"], inplace=True)

    output_path = save_dataframe(combined_df, args.results_dir, args.output_name)
    print(f"Wrote {len(combined_df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
