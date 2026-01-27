from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils import DATASETS, MODEL_CONFIGS, dataset_choice_labels

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESCUE_ROOT = REPO_ROOT / "outputs" / "rescue"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "datasets"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_NAME = "processed_rescue.parquet"
DEFAULT_SUMMARY_NAME = "rescue_summary.csv"


@dataclass(frozen=True)
class RescueFile:
    dataset: str
    target_model: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate rescue JSONL files and compute accuracies plus continuation deltas."
    )
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
        "--base-model",
        "--base-models",
        nargs="+",
        dest="base_models",
        choices=MODEL_CONFIGS.keys(),
        default=list(MODEL_CONFIGS.keys()),
        help="Base models to include (filtered within the rescue JSONL records).",
    )
    parser.add_argument(
        "--target-model",
        "--target-models",
        nargs="+",
        dest="target_models",
        choices=MODEL_CONFIGS.keys(),
        default=list(MODEL_CONFIGS.keys()),
        help="Target models to include.",
    )
    parser.add_argument(
        "--rescue-root",
        type=Path,
        default=DEFAULT_RESCUE_ROOT,
        help="Directory containing rescue JSONL files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing dataset parquet files with answers.",
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
        help="Filename for the row-level dataframe (extension determines format).",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default=DEFAULT_SUMMARY_NAME,
        help="Filename for the aggregated summary (extension determines format).",
    )
    return parser.parse_args()


def discover_rescue_files(
    datasets: Iterable[str],
    target_models: set[str],
    rescue_root: Path,
) -> list[RescueFile]:
    discovered: list[RescueFile] = []
    pattern = re.compile(r"^(?P<dataset>[^_]+)_(?P<target_model>.+)_rescue\.jsonl$")

    for dataset in datasets:
        dataset_dir = rescue_root / dataset
        if not dataset_dir.exists():
            continue

        for path in dataset_dir.glob(f"{dataset}_*_rescue.jsonl"):
            match = pattern.match(path.name)
            if not match:
                continue

            target_model = match.group("target_model")
            if target_models and target_model not in target_models:
                continue

            discovered.append(
                RescueFile(
                    dataset=dataset,
                    target_model=target_model,
                    path=path,
                )
            )

    return sorted(discovered, key=lambda f: (f.dataset, f.target_model, f.path.name))


def load_rescue_records(files: list[RescueFile], allowed_base_models: set[str]) -> pd.DataFrame:
    """Load rescue JSONL files."""
    frames: list[pd.DataFrame] = []
    required_cols = {
        "dataset",
        "base_model",
        "target_model",
        "run",
        "decile",
        "id",
        "baseline_choice_logprobs",
        "free_choice_logprobs",
        "free_continuation_length",
    }

    for entry in files:
        df = pd.read_json(entry.path, lines=True)
        if df.empty:
            continue

        missing = required_cols.difference(df.columns)
        if missing:
            raise KeyError(f"{entry.path} missing required columns: {', '.join(sorted(missing))}")

        df = df[df["base_model"].isin(allowed_base_models)]
        if df.empty:
            continue

        df["dataset"] = entry.dataset
        df["target_model"] = entry.target_model
        df["id"] = df["id"].astype(str)
        df["run"] = df["run"].astype(int)
        df["decile"] = df["decile"].astype(int)
        df["free_continuation_length"] = pd.to_numeric(df["free_continuation_length"], errors="coerce")
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_answers(datasets: Iterable[str], data_root: Path) -> dict[str, pd.Series]:
    answers: dict[str, pd.Series] = {}
    for dataset in datasets:
        path = data_root / f"{dataset}.parquet"
        if not path.exists():
            print(f"Warning: dataset file not found: {path}")
            continue
        df = pd.read_parquet(path, columns=["id", "answer"])
        df["id"] = df["id"].astype(str)
        ans = df["answer"]
        df["answer"] = ans.where(ans.notna(), pd.NA).apply(
            lambda x: x if pd.isna(x) else str(x).strip().upper()
        )
        answers[dataset] = df.set_index("id")["answer"]
    return answers


def add_choice_columns(
    df: pd.DataFrame,
    dataset: str,
    source_col: str,
    prefix: str,
) -> tuple[pd.DataFrame, list[str]]:
    labels = dataset_choice_labels(dataset)
    records = df[source_col].apply(lambda x: x if isinstance(x, dict) else {}).tolist()
    logprob_frame = pd.DataFrame.from_records(records).reindex(columns=labels)

    # Convert to numeric, but recover rows that become all-NA (observed for some deciles).
    logprob_frame = logprob_frame.apply(pd.to_numeric, errors="coerce")
    all_na_mask = logprob_frame.isna().all(axis=1)
    df = df.copy()
    df[f"{prefix}_choice_logprobs_all_na"] = all_na_mask
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
        logprob_frame.loc[all_na_mask] = recovered_frame.loc[all_na_mask]

    logprob_frame = logprob_frame.fillna(-np.inf)
    logprob_frame.columns = [f"{prefix}_logprob_{label}" for label in labels]

    prob_frame = np.exp(logprob_frame)
    prob_frame.columns = [f"{prefix}_prob_{label}" for label in labels]

    df = df.join(logprob_frame)
    df = df.join(prob_frame)
    return df, prob_frame.columns.tolist()


def lookup_prob(row: pd.Series, column_prefix: str, label: str | float | None) -> float:
    if not isinstance(label, str) or not label:
        return np.nan
    key = f"{column_prefix}{label}"
    return row.get(key, np.nan)


def process_rescue_dataset(
    df: pd.DataFrame,
    dataset: str,
    answers: pd.Series | None,
) -> pd.DataFrame:
    """Compute predictions/metrics for one dataset of rescue probes."""
    # Align indexes before expanding logprob dicts; filtering by dataset preserves
    # original indexes from the concatenated frame, which breaks join alignment.
    df = df.reset_index(drop=True)

    if answers is not None:
        df = df.merge(
            answers.rename("answer"),
            left_on="id",
            right_index=True,
            how="left",
            validate="many_to_one",
        )
    else:
        df["answer"] = pd.NA

    df, baseline_prob_cols = add_choice_columns(df, dataset, "baseline_choice_logprobs", "baseline")
    df, free_prob_cols = add_choice_columns(df, dataset, "free_choice_logprobs", "free")
    baseline_all_na = df["baseline_choice_logprobs_all_na"].astype(bool)
    free_all_na = df["free_choice_logprobs_all_na"].astype(bool)

    df["baseline_prediction"] = pd.NA
    mask_baseline = ~baseline_all_na
    if mask_baseline.any():
        df.loc[mask_baseline, "baseline_prediction"] = (
            df.loc[mask_baseline, baseline_prob_cols].idxmax(axis=1).str.removeprefix("baseline_prob_")
        )
    df["baseline_prediction_prob"] = df[baseline_prob_cols].max(axis=1)
    df["baseline_choice_mass"] = df[baseline_prob_cols].sum(axis=1)

    df["free_prediction"] = pd.NA
    mask_free = ~free_all_na
    if mask_free.any():
        df.loc[mask_free, "free_prediction"] = (
            df.loc[mask_free, free_prob_cols].idxmax(axis=1).str.removeprefix("free_prob_")
        )
    df["free_prediction_prob"] = df[free_prob_cols].max(axis=1)
    df["free_choice_mass"] = df[free_prob_cols].sum(axis=1)

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

    missing_baseline = df["baseline_prediction"].isna() & ~baseline_all_na
    if missing_baseline.any():
        df.loc[missing_baseline, "baseline_prediction"] = df.loc[missing_baseline, "baseline_choice_logprobs"].apply(
            _argmax_choice
        )
    missing_free = df["free_prediction"].isna() & ~free_all_na
    if missing_free.any():
        df.loc[missing_free, "free_prediction"] = df.loc[missing_free, "free_choice_logprobs"].apply(_argmax_choice)

    if baseline_all_na.any():
        df.loc[baseline_all_na, "baseline_prediction"] = pd.NA
        df.loc[baseline_all_na, "baseline_prediction_prob"] = np.nan
        df.loc[baseline_all_na, "baseline_choice_mass"] = np.nan
    if free_all_na.any():
        df.loc[free_all_na, "free_prediction"] = pd.NA
        df.loc[free_all_na, "free_prediction_prob"] = np.nan
        df.loc[free_all_na, "free_choice_mass"] = np.nan

    ans = df["answer"]
    df["answer"] = ans.where(ans.notna(), pd.NA).apply(lambda x: x if pd.isna(x) else str(x).strip().upper())
    missing_answer = df["answer"].isna()

    df["baseline_logprob_correct_answer"] = df.apply(
        lambda row: lookup_prob(row, "baseline_logprob_", row.get("answer")),
        axis=1,
    )
    df["free_logprob_correct_answer"] = df.apply(
        lambda row: lookup_prob(row, "free_logprob_", row.get("answer")),
        axis=1,
    )
    df["baseline_prob_correct_answer"] = df.apply(
        lambda row: lookup_prob(row, "baseline_prob_", row.get("answer")),
        axis=1,
    )
    df["free_prob_correct_answer"] = df.apply(
        lambda row: lookup_prob(row, "free_prob_", row.get("answer")),
        axis=1,
    )
    df.loc[missing_answer, ["baseline_logprob_correct_answer", "free_logprob_correct_answer"]] = np.nan
    df.loc[missing_answer, ["baseline_prob_correct_answer", "free_prob_correct_answer"]] = np.nan
    df.loc[baseline_all_na, ["baseline_logprob_correct_answer", "baseline_prob_correct_answer"]] = np.nan
    df.loc[free_all_na, ["free_logprob_correct_answer", "free_prob_correct_answer"]] = np.nan

    valid_baseline = ~missing_answer & ~df["baseline_prediction"].isna()
    valid_free = ~missing_answer & ~df["free_prediction"].isna()
    valid_both = valid_baseline & valid_free

    baseline_accuracy = pd.Series(pd.NA, index=df.index, dtype="boolean")
    free_accuracy = pd.Series(pd.NA, index=df.index, dtype="boolean")
    if valid_baseline.any():
        baseline_accuracy.loc[valid_baseline] = (
            df.loc[valid_baseline, "baseline_prediction"] == df.loc[valid_baseline, "answer"]
        )
    if valid_free.any():
        free_accuracy.loc[valid_free] = df.loc[valid_free, "free_prediction"] == df.loc[valid_free, "answer"]
    df["baseline_accuracy"] = baseline_accuracy
    df["free_accuracy"] = free_accuracy

    df["improved_by_continuation"] = pd.Series([pd.NA] * len(df), dtype="boolean")
    df["worsened_by_continuation"] = pd.Series([pd.NA] * len(df), dtype="boolean")
    df["accuracy_delta"] = pd.Series([np.nan] * len(df), dtype="float")
    df.loc[valid_both, "improved_by_continuation"] = (
        (~df.loc[valid_both, "baseline_accuracy"]) & df.loc[valid_both, "free_accuracy"]
    )
    df.loc[valid_both, "worsened_by_continuation"] = (
        df.loc[valid_both, "baseline_accuracy"] & (~df.loc[valid_both, "free_accuracy"])
    )
    df.loc[valid_both, "accuracy_delta"] = (
        df.loc[valid_both, "free_accuracy"].astype(float) - df.loc[valid_both, "baseline_accuracy"].astype(float)
    )

    df["prob_correct_delta"] = df["free_prob_correct_answer"] - df["baseline_prob_correct_answer"]
    df["logprob_correct_delta"] = df["free_logprob_correct_answer"] - df["baseline_logprob_correct_answer"]

    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(["dataset", "base_model", "target_model", "run", "decile"])
        .agg(
            n=("id", "count"),
            baseline_accuracy=("baseline_accuracy", "mean"),
            free_accuracy=("free_accuracy", "mean"),
            improve_rate=("improved_by_continuation", "mean"),
            worsen_rate=("worsened_by_continuation", "mean"),
            mean_baseline_prob_correct=("baseline_prob_correct_answer", "mean"),
            mean_free_prob_correct=("free_prob_correct_answer", "mean"),
            mean_free_continuation_length=("free_continuation_length", "mean"),
            accuracy_delta=("accuracy_delta", "mean"),
            prob_correct_delta=("prob_correct_delta", "mean"),
            logprob_correct_delta=("logprob_correct_delta", "mean"),
        )
        .reset_index()
    )
    return grouped


def save_dataframe(df: pd.DataFrame, results_dir: Path, output_name: str) -> Path:
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
    target_models = set(args.target_models)
    base_models = set(args.base_models)

    rescue_files = discover_rescue_files(datasets, target_models, args.rescue_root)
    rescue_df = load_rescue_records(rescue_files, base_models)
    if rescue_df.empty:
        raise RuntimeError("No rescue data found for the requested datasets/models.")

    answers = load_answers(datasets, args.data_root)

    processed_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        df_dataset = rescue_df[rescue_df["dataset"] == dataset]
        if df_dataset.empty:
            continue
        processed_df = process_rescue_dataset(df_dataset, dataset, answers.get(dataset))
        baseline_missing = int(processed_df.get("baseline_choice_logprobs_all_na", pd.Series(False)).sum())
        free_missing = int(processed_df.get("free_choice_logprobs_all_na", pd.Series(False)).sum())
        unlabeled = int(processed_df["answer"].isna().sum())
        if baseline_missing or free_missing or unlabeled:
            print(
                f"{dataset}: flagged {baseline_missing} rows with missing baseline logprobs, "
                f"{free_missing} with missing free logprobs; {unlabeled} unlabeled rows excluded from accuracy."
            )
        processed_df = processed_df.drop(
            columns=["baseline_choice_logprobs_all_na", "free_choice_logprobs_all_na"],
            errors="ignore",
        )
        processed_frames.append(processed_df)

    if not processed_frames:
        raise RuntimeError("No rescue data found after processing the requested filters.")

    combined_df = pd.concat(processed_frames, ignore_index=True)
    combined_df.sort_values(
        ["dataset", "base_model", "target_model", "run", "decile", "id"],
        inplace=True,
    )

    summary_df = build_summary(combined_df)

    output_path = save_dataframe(combined_df, args.results_dir, args.output_name)
    print(f"Wrote {len(combined_df):,} rows to {output_path}")

    if not summary_df.empty:
        summary_path = save_dataframe(summary_df, args.results_dir, args.summary_name)
        print(f"Wrote summary ({len(summary_df)} rows) to {summary_path}")
    else:
        print("No summary written (empty dataframe).")


if __name__ == "__main__":
    main()
