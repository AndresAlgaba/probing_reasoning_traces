import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar


DECILES = [20, 40, 60, 80]
BASE_MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b"]
TARGET_MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"]
DATASETS = ["gpqa", "mmlu"]

PROMOTIONS: dict[str, list[str]] = {
    "qwen3_4b": ["qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"],
    "qwen3_8b": ["qwen3_4b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"],
    "qwen3_14b": ["gpt_oss_20b", "gpt_oss_120b"],
    "gpt_oss_20b": ["gpt_oss_120b"],
}

def promotion_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for base, targets in PROMOTIONS.items():
        for tgt in targets:
            pairs.append((base, tgt))
    return pairs

def format_model_label(model: str) -> str:
    if model.startswith("qwen3_"):
        size = model.split("_", 1)[1]
        return f"Qwen3-{size.upper()}"
    if model.startswith("gpt_oss_"):
        size = model.split("_")[-1]
        return f"gpt-oss-{size}"
    return model.replace("_", "-")


def format_transfer_label(base: str, target: str) -> str:
    return f"{format_model_label(base)}$\\to${format_model_label(target)}"


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def load_rescue_data(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "results" / "processed_rescue.parquet"
    cols = [
        "dataset",
        "base_model",
        "target_model",
        "run",
        "decile",
        "id",
        "answer",
        "baseline_prediction",
        "free_prediction",
        "baseline_accuracy",
        "free_accuracy",
    ]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["decile"].isin(DECILES)]
    df = df[df["dataset"].isin(DATASETS)]
    df = df[df["base_model"].isin(BASE_MODELS)]
    df = df[df["target_model"].isin(TARGET_MODELS)]

    df["id"] = df["id"].astype(str)
    for col in ["answer", "baseline_prediction", "free_prediction"]:
        df[col] = df[col].astype("string").str.strip().str.upper()
    return df


def load_base_predictions(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "results" / "processed_outputs.parquet"
    cols = ["dataset", "model", "run", "decile", "id", "decile_prediction"]
    df = pd.read_parquet(path, columns=cols, filters=[("decile", "in", DECILES)])
    df = df[df["model"].isin(BASE_MODELS)]

    df = df.rename(columns={"model": "base_model", "decile_prediction": "base_decile_prediction"})
    df["id"] = df["id"].astype(str)
    df["base_decile_prediction"] = df["base_decile_prediction"].astype("string").str.strip().str.upper()
    return df


def attach_base_predictions(rescue_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    merged = rescue_df.merge(
        base_df,
        on=["dataset", "base_model", "run", "decile", "id"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.dropna(subset=["answer", "base_decile_prediction"])

    base_wrong_mask = merged["base_decile_prediction"] != merged["answer"]
    merged = merged[base_wrong_mask].copy()

    merged["baseline_anchor"] = merged["baseline_prediction"] == merged["base_decile_prediction"]
    merged["free_anchor"] = merged["free_prediction"] == merged["base_decile_prediction"]

    return merged


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for dataset in DATASETS:
        for base_model, target_model in promotion_pairs():
            subset = df[
                (df["dataset"] == dataset) &
                (df["base_model"] == base_model) &
                (df["target_model"] == target_model)
            ]

            subset = subset.dropna(subset=["baseline_accuracy", "free_accuracy"])
            if subset.empty:
                continue

            n = len(subset)

            base_acc = subset["baseline_accuracy"].astype(bool)
            free_acc = subset["free_accuracy"].astype(bool)
            anchor_base_series = subset["baseline_anchor"].astype(bool)
            anchor_free_series = subset["free_anchor"].astype(bool)

            rescue_base = base_acc.mean()
            rescue_free = free_acc.mean()
            anchor_base = anchor_base_series.mean()
            anchor_free = anchor_free_series.mean()

            delta_rescue = rescue_free - rescue_base
            delta_anchor = anchor_free - anchor_base

            base_correct = base_acc.to_numpy(dtype=bool)
            free_correct = free_acc.to_numpy(dtype=bool)

            n_01 = np.sum(base_correct & ~free_correct)
            n_10 = np.sum(~base_correct & free_correct)

            if n_01 + n_10 > 0:
                table = [[0, n_01], [n_10, 0]]
                if n_01 + n_10 < 25:
                    result = mcnemar(table, exact=True)
                else:
                    result = mcnemar(table, exact=False, correction=True)
                p_free_vs_base = result.pvalue

                if n_10 > n_01:
                    p_free_vs_base = p_free_vs_base / 2
                else:
                    p_free_vs_base = 1 - p_free_vs_base / 2
            else:
                p_free_vs_base = 1.0

            base_anchor = anchor_base_series.to_numpy(dtype=bool)
            free_anchor = anchor_free_series.to_numpy(dtype=bool)

            n_01_anch = np.sum(base_anchor & ~free_anchor)
            n_10_anch = np.sum(~base_anchor & free_anchor)

            if n_01_anch + n_10_anch > 0:
                table_anch = [[0, n_01_anch], [n_10_anch, 0]]
                if n_01_anch + n_10_anch < 25:
                    result_anch = mcnemar(table_anch, exact=True)
                else:
                    result_anch = mcnemar(table_anch, exact=False, correction=True)
                p_anchor_delta = result_anch.pvalue

                if n_01_anch > n_10_anch:
                    p_anchor_delta = p_anchor_delta / 2
                else:
                    p_anchor_delta = 1 - p_anchor_delta / 2
            else:
                p_anchor_delta = 1.0

            records.append({
                "dataset": dataset,
                "base_model": base_model,
                "target_model": target_model,
                "n": n,
                "rescue_base": rescue_base,
                "rescue_free": rescue_free,
                "delta_rescue": delta_rescue,
                "anchor_base": anchor_base,
                "anchor_free": anchor_free,
                "delta_anchor": delta_anchor,
                "p_delta": p_free_vs_base,
                "p_delta_anchor": p_anchor_delta,
            })

    return pd.DataFrame(records)


def format_latex_table(stats_df: pd.DataFrame) -> str:
    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Cross-model rescue and anchoring summary.} "
                 r"Mean rescue rate (probability target model answers correctly given base model's incorrect partial trace) "
                 r"and anchoring rate (probability target repeats base's wrong answer), averaged across deciles 20--80\% and pooled over 3 runs. "
                 r"$\Delta$ shows the change from base to free mode (positive $\Delta$ Rescue is better; negative $\Delta$ Anchor is better). "
                 r"$n$ is the number of base-model mistakes available for rescue. "
                 r"Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (one-sided McNemar's test).}")
    lines.append(r"\label{tab:rescue-summary}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrcccccc}")
    lines.append(r"\toprule")
    lines.append(r" & & & \multicolumn{3}{c}{Rescue Rate} & \multicolumn{3}{c}{Anchoring Rate} \\")
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}")
    lines.append(r"Dataset & Transfer & $n$ & Base & Free & $\Delta$ & Base & Free & $\Delta$ \\")
    lines.append(r"\midrule")

    current_dataset = None
    for _, row in stats_df.iterrows():
        dataset = row["dataset"]

        if dataset != current_dataset:
            if current_dataset is not None:
                lines.append(r"\midrule")
            dataset_label = "GPQA" if dataset == "gpqa" else "MMLU"
            current_dataset = dataset
        else:
            dataset_label = ""

        transfer_label = format_transfer_label(row["base_model"], row["target_model"])
        n_str = f"{int(row['n']):,}"

        delta_rescue_stars = significance_stars(row["p_delta"])
        delta_anchor_stars = significance_stars(row["p_delta_anchor"])

        rescue_base_str = f"{row['rescue_base'] * 100:.1f}\\%"
        rescue_free_str = f"{row['rescue_free'] * 100:.1f}\\%"

        delta_rescue_val = row["delta_rescue"] * 100
        if delta_rescue_val >= 0:
            delta_rescue_str = f"+{delta_rescue_val:.1f}\\%$^{{{delta_rescue_stars}}}$"
        else:
            delta_rescue_str = f"{delta_rescue_val:.1f}\\%$^{{{delta_rescue_stars}}}$"

        anchor_base_str = f"{row['anchor_base'] * 100:.1f}\\%"
        anchor_free_str = f"{row['anchor_free'] * 100:.1f}\\%"

        delta_anchor_val = row["delta_anchor"] * 100
        if delta_anchor_val >= 0:
            delta_anchor_str = f"+{delta_anchor_val:.1f}\\%$^{{{delta_anchor_stars}}}$"
        else:
            delta_anchor_str = f"{delta_anchor_val:.1f}\\%$^{{{delta_anchor_stars}}}$"

        row_str = (f"{dataset_label} & {transfer_label} & {n_str} & "
                   f"{rescue_base_str} & {rescue_free_str} & {delta_rescue_str} & "
                   f"{anchor_base_str} & {anchor_free_str} & {delta_anchor_str} \\\\")
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "results" / "tables" / "rescue_summary.tex"

    rescue_df = load_rescue_data(repo_root)
    base_df = load_base_predictions(repo_root)

    df = attach_base_predictions(rescue_df, base_df)

    stats_df = compute_summary_stats(df)

    latex_table = format_latex_table(stats_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    main()
