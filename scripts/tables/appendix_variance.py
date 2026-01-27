import pandas as pd

cols = [
    "dataset", "model", "run", "decile",
    "decile_accuracy", "decile_choice_mass",
    "decile_prediction_prob_final_answer", "decile_flip_rate",
]
df = pd.read_parquet("../../results/processed_outputs.parquet", columns=cols)
df["decile_non_choice_mass"] = 1 - df["decile_choice_mass"]

run_agg = df.groupby(["dataset", "model", "run", "decile"]).agg({
    "decile_accuracy": "mean",
    "decile_prediction_prob_final_answer": "mean",
    "decile_non_choice_mass": "mean",
    "decile_flip_rate": "mean",
}).reset_index()

d100 = run_agg[run_agg["decile"] == 100]

summary = d100.groupby(["dataset", "model"]).agg({
    "decile_accuracy": ["mean", "std"],
    "decile_prediction_prob_final_answer": ["mean", "std"],
    "decile_non_choice_mass": ["mean", "std"],
    "decile_flip_rate": ["mean", "std"],
})


def fmt(mean: float, std: float) -> str:
    """Format as Mean ± SD in percentage points."""
    return f"{mean*100:.1f} ± {std*100:.1f}"


MODEL_NAMES = {
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_14b": "Qwen3-14B",
    "gpt_oss_20b": "gpt-oss-20b",
    "gpt_oss_120b": "gpt-oss-120b",
}

DATASET_NAMES = {
    "gpqa": "GPQA Diamond",
    "mmlu": "MMLU-Pro",
}

MODEL_ORDER = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"]

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{\textbf{Run-to-run variance at full reasoning (decile 100).} "
             r"Mean $\pm$ standard deviation across 3 independent runs for accuracy, "
             r"decision commitment (probability on final answer), non-choice probability, "
             r"and flip rate. All values in percentage points. Variance is generally small "
             r"(SD $<$ 1\% for most metrics), with slightly higher variance on GPQA Diamond "
             r"due to its smaller sample size (198 questions) and greater task difficulty.}")
lines.append(r"\label{tab:appendix_variance}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{llcccc}")
lines.append(r"\toprule")
lines.append(r"Dataset & Model & Accuracy & Decision Commit. & Non-choice Prob. & Flip Rate \\")
lines.append(r"\midrule")

for dataset in ["gpqa", "mmlu"]:
    for i, model in enumerate(MODEL_ORDER):
        row = summary.loc[(dataset, model)]
        acc = fmt(row[("decile_accuracy", "mean")], row[("decile_accuracy", "std")])
        commit = fmt(
            row[("decile_prediction_prob_final_answer", "mean")],
            row[("decile_prediction_prob_final_answer", "std")],
        )
        nonchoice = fmt(
            row[("decile_non_choice_mass", "mean")],
            row[("decile_non_choice_mass", "std")],
        )
        flip = fmt(row[("decile_flip_rate", "mean")], row[("decile_flip_rate", "std")])

        ds_label = DATASET_NAMES[dataset] if i == 0 else ""
        model_label = MODEL_NAMES[model]

        lines.append(f"{ds_label} & {model_label} & {acc} & {commit} & {nonchoice} & {flip} \\\\")

    if dataset == "gpqa":
        lines.append(r"\midrule")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

table_latex = "\n".join(lines)
print(table_latex)

output_path = "../../results/tables/appendix_variance.tex"
with open(output_path, "w") as f:
    f.write(table_latex)
print(f"\nSaved to {output_path}")
