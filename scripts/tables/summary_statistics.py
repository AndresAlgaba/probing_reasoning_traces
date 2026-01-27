import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar


MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"]
DATASETS = ["gpqa", "mmlu"]
CONTROL_ORDER = ["junk", "cross", "shuffle"]
CONTROL_LABELS = {
    "junk": "Random",
    "cross": "Trace-swap",
    "shuffle": "Permutation",
}


def format_model_label(model: str) -> str:
    if model.startswith("qwen3_"):
        size = model.split("_", 1)[1]
        return f"Qwen3-{size.upper()}"
    if model.startswith("gpt_oss_"):
        size = model.split("_")[-1]
        return f"gpt-oss-{size}"
    return model.replace("_", "-")


def format_dataset_label(dataset: str) -> str:
    if dataset == "gpqa":
        return "GPQA"
    if dataset == "mmlu":
        return "MMLU"
    return dataset.upper()


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def run_mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[float, float]:
    a = correct_a.astype(bool)
    b = correct_b.astype(bool)

    n_01 = np.sum(a & ~b)
    n_10 = np.sum(~a & b)

    table = [[0, n_01], [n_10, 0]]

    if n_01 + n_10 == 0:
        return 0.0, 1.0

    if n_01 + n_10 < 25:
        result = mcnemar(table, exact=True)
    else:
        result = mcnemar(table, exact=False, correction=True)

    return result.statistic, result.pvalue


def load_baseline_data(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "results" / "processed_outputs.parquet"
    cols = ["dataset", "model", "run", "id", "decile", "decile_accuracy"]
    df = pd.read_parquet(
        path,
        columns=cols,
        filters=[("decile", "in", [0, 100])],
    )
    df = df[df["model"].isin(MODELS)]
    return df


def load_control_data(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "results" / "processed_controls.parquet"
    cols = ["dataset", "model", "run", "id", "decile", "control_type", "control_accuracy"]
    df = pd.read_parquet(
        path,
        columns=cols,
        filters=[("decile", "==", 100)],
    )
    df = df[df["model"].isin(MODELS)]
    df = df[df["control_type"].isin(CONTROL_ORDER)]
    return df


def compute_summary_stats(
    baseline_df: pd.DataFrame,
    control_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for dataset in DATASETS:
        for model in MODELS:
            base_subset = baseline_df[
                (baseline_df["dataset"] == dataset) &
                (baseline_df["model"] == model)
            ]

            if base_subset.empty:
                continue

            d0 = base_subset[base_subset["decile"] == 0].copy()
            d100 = base_subset[base_subset["decile"] == 100].copy()

            paired = d0.merge(
                d100,
                on=["dataset", "model", "run", "id"],
                suffixes=("_d0", "_d100"),
            )

            if paired.empty:
                continue

            acc_d0 = paired["decile_accuracy_d0"].mean()
            acc_d100 = paired["decile_accuracy_d100"].mean()
            gain = acc_d100 - acc_d0

            _, p_gain = run_mcnemar(
                paired["decile_accuracy_d0"].values,
                paired["decile_accuracy_d100"].values,
            )

            control_results = {}
            for control_type in CONTROL_ORDER:
                ctrl_subset = control_df[
                    (control_df["dataset"] == dataset) &
                    (control_df["model"] == model) &
                    (control_df["control_type"] == control_type)
                ]

                if ctrl_subset.empty:
                    control_results[control_type] = {
                        "diff": np.nan,
                        "p": np.nan,
                        "diff_vs_d0": np.nan,
                        "p_vs_d0": np.nan,
                    }
                    continue

                ctrl_paired = d100.merge(
                    ctrl_subset[["run", "id", "control_accuracy"]],
                    on=["run", "id"],
                )

                if ctrl_paired.empty:
                    control_results[control_type] = {
                        "diff": np.nan,
                        "p": np.nan,
                        "diff_vs_d0": np.nan,
                        "p_vs_d0": np.nan,
                    }
                    continue

                orig_acc_for_ctrl = ctrl_paired["decile_accuracy"].mean()
                ctrl_acc = ctrl_paired["control_accuracy"].mean()
                diff = orig_acc_for_ctrl - ctrl_acc

                _, p_ctrl = run_mcnemar(
                    ctrl_paired["decile_accuracy"].values,
                    ctrl_paired["control_accuracy"].values,
                )

                ctrl_vs_d0_paired = d0.merge(
                    ctrl_subset[["run", "id", "control_accuracy"]],
                    on=["run", "id"],
                )
                if not ctrl_vs_d0_paired.empty:
                    d0_acc_for_ctrl = ctrl_vs_d0_paired["decile_accuracy"].mean()
                    ctrl_acc_for_d0 = ctrl_vs_d0_paired["control_accuracy"].mean()
                    diff_vs_d0 = ctrl_acc_for_d0 - d0_acc_for_ctrl

                    _, p_ctrl_vs_d0 = run_mcnemar(
                        ctrl_vs_d0_paired["decile_accuracy"].values,
                        ctrl_vs_d0_paired["control_accuracy"].values,
                    )
                else:
                    diff_vs_d0 = np.nan
                    p_ctrl_vs_d0 = np.nan

                control_results[control_type] = {
                    "diff": diff,
                    "p": p_ctrl,
                    "diff_vs_d0": diff_vs_d0,
                    "p_vs_d0": p_ctrl_vs_d0,
                }

            record = {
                "dataset": dataset,
                "model": model,
                "acc_d0": acc_d0,
                "acc_d100": acc_d100,
                "gain": gain,
                "p_gain": p_gain,
                "n_questions": len(paired),
            }

            for ctrl in CONTROL_ORDER:
                record[f"diff_{ctrl}"] = control_results[ctrl]["diff"]
                record[f"p_{ctrl}"] = control_results[ctrl]["p"]
                record[f"diff_vs_d0_{ctrl}"] = control_results[ctrl]["diff_vs_d0"]
                record[f"p_vs_d0_{ctrl}"] = control_results[ctrl]["p_vs_d0"]

            records.append(record)

    return pd.DataFrame(records)


def format_latex_table(stats_df: pd.DataFrame) -> str:
    lines = []

    # Table header (landscape mode)
    lines.append(r"\begin{landscape}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Summary statistics and significance tests.} "
                 r"Gain shows accuracy improvement from decile 0 to 100. "
                 r"``Original vs Control'' columns show the accuracy advantage of original traces over each control at decile 100. "
                 r"``Control vs d=0'' columns show the accuracy change of each control relative to the no-reasoning baseline. "
                 r"$n$ is the number of question--run pairs (pooled across 3 runs). "
                 r"Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (McNemar's test).}")
    lines.append(r"\label{tab:summary-statistics}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrccccccccc}")
    lines.append(r"\toprule")

    lines.append(r" & & & \multicolumn{3}{c}{Accuracy} & \multicolumn{3}{c}{Original vs Control (d=100)} & \multicolumn{3}{c}{Control vs d=0} \\")
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}")
    lines.append(r"Dataset & Model & $n$ & d=0 & d=100 & Gain & Random & Trace-swap & Permutation & Random & Trace-swap & Permutation \\")
    lines.append(r"\midrule")

    current_dataset = None
    for _, row in stats_df.iterrows():
        dataset = row["dataset"]
        model = row["model"]

        if dataset != current_dataset:
            if current_dataset is not None:
                lines.append(r"\midrule")
            dataset_label = format_dataset_label(dataset)
            current_dataset = dataset
        else:
            dataset_label = ""

        model_label = format_model_label(model)

        n_questions = int(row["n_questions"])
        n_str = f"{n_questions:,}"

        acc_d0 = f"{row['acc_d0'] * 100:.1f}\\%"
        acc_d100 = f"{row['acc_d100'] * 100:.1f}\\%"

        gain_val = row["gain"] * 100
        gain_stars = significance_stars(row["p_gain"])
        gain_str = f"+{gain_val:.1f}\\%$^{{{gain_stars}}}$" if gain_val >= 0 else f"{gain_val:.1f}\\%$^{{{gain_stars}}}$"

        ctrl_strs = []
        for ctrl in CONTROL_ORDER:
            diff = row[f"diff_{ctrl}"]
            p = row[f"p_{ctrl}"]
            if pd.isna(diff):
                ctrl_strs.append("--")
            else:
                diff_val = diff * 100
                stars = significance_stars(p)
                if diff_val >= 0:
                    ctrl_strs.append(f"+{diff_val:.1f}\\%$^{{{stars}}}$")
                else:
                    ctrl_strs.append(f"{diff_val:.1f}\\%$^{{{stars}}}$")

        ctrl_vs_d0_strs = []
        for ctrl in CONTROL_ORDER:
            diff_vs_d0 = row[f"diff_vs_d0_{ctrl}"]
            p_vs_d0 = row[f"p_vs_d0_{ctrl}"]
            if pd.isna(diff_vs_d0):
                ctrl_vs_d0_strs.append("--")
            else:
                diff_val = diff_vs_d0 * 100
                stars = significance_stars(p_vs_d0) if not pd.isna(p_vs_d0) else ""
                if diff_val >= 0:
                    ctrl_vs_d0_strs.append(f"+{diff_val:.1f}\\%$^{{{stars}}}$")
                else:
                    ctrl_vs_d0_strs.append(f"$-${abs(diff_val):.1f}\\%$^{{{stars}}}$")

        row_str = f"{dataset_label} & {model_label} & {n_str} & {acc_d0} & {acc_d100} & {gain_str} & {' & '.join(ctrl_strs)} & {' & '.join(ctrl_vs_d0_strs)} \\\\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"\end{landscape}")

    return "\n".join(lines)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "results" / "tables" / "summary_statistics.tex"

    baseline_df = load_baseline_data(repo_root)
    control_df = load_control_data(repo_root)

    stats_df = compute_summary_stats(baseline_df, control_df)

    latex_table = format_latex_table(stats_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    main()
