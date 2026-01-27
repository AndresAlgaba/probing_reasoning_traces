import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path


plt.rcParams["font.family"] = "Arial"
plt.rcParams["legend.title_fontsize"] = 7
plt.rcParams["text.usetex"] = True


MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"]
COLORS = ["#0173b2", "#56b4e9", "#029e73", "#de8f05", "#d55e00"]
MODEL_COLOR_MAP = dict(zip(MODELS, COLORS))
N_QUANTILES = 5


def format_model_label(model: str) -> str:
    if model.startswith("qwen3_"):
        size = model.split("_", 1)[1]
        return f"Qwen3-{size.upper()}"
    return model.replace("_", "-")


def load_data(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "results" / "processed_outputs.parquet"
    cols = ["dataset", "model", "run", "id", "decile", "response_n_tokens", "decile_accuracy"]
    df = pd.read_parquet(path, columns=cols, filters=[("decile", "==", 100)])
    df = df[df["model"].isin(MODELS)]
    return df


def compute_quantile_accuracy(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    subset = df[df["dataset"] == dataset].copy()
    records = []

    for model in MODELS:
        model_df = subset[subset["model"] == model].copy()
        if model_df.empty:
            continue

        model_df["length_quantile"] = pd.qcut(
            model_df["response_n_tokens"],
            q=N_QUANTILES,
            labels=False,
            duplicates="drop",
        )

        grouped = model_df.groupby("length_quantile").agg(
            mean_accuracy=("decile_accuracy", "mean"),
            mean_length=("response_n_tokens", "mean"),
            count=("decile_accuracy", "size"),
        ).reset_index()

        grouped["model"] = model
        grouped["dataset"] = dataset
        records.append(grouped)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "results" / "figures" / "appendix_accuracy_length.png"

    df = load_data(repo_root)
    df_gpqa = df[df["dataset"] == "gpqa"]
    df_mmlu = df[df["dataset"] == "mmlu"]

    quant_gpqa = compute_quantile_accuracy(df, "gpqa")
    quant_mmlu = compute_quantile_accuracy(df, "mmlu")

    fig, axes = plt.subplot_mosaic(
        [
            ["A", "B"],
            ["C", "D"],
        ],
        figsize=(7.08, 5.0),
        dpi=300,
        gridspec_kw={"wspace": 0.15, "hspace": 0.35},
    )

    ax = axes["A"]

    violin_data_gpqa = []
    for model in MODELS:
        model_df = df_gpqa[df_gpqa["model"] == model]
        for val in model_df["response_n_tokens"]:
            violin_data_gpqa.append({"model": model, "length": val})
    violin_df_gpqa = pd.DataFrame(violin_data_gpqa)

    sns.violinplot(
        data=violin_df_gpqa,
        x="model",
        y="length",
        hue="model",
        order=MODELS,
        hue_order=MODELS,
        palette=COLORS,
        ax=ax,
        inner="box",
        linewidth=0.5,
        cut=0,
        legend=False,
    )

    ax.set_xlabel("", fontsize=7)
    ax.set_ylabel("Trace length (tokens)", fontsize=7)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(
        [format_model_label(m) for m in MODELS],
        fontsize=6,
        rotation=30,
        ha="right",
    )
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylim(0, None)
    sns.despine(ax=ax, left=False, bottom=False)

    ax.text(
        0.5, 1.06, r"\textbf{GPQA Diamond}",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=7,
        clip_on=False,
    )

    ax = axes["B"]

    violin_data_mmlu = []
    for model in MODELS:
        model_df = df_mmlu[df_mmlu["model"] == model]
        for val in model_df["response_n_tokens"]:
            violin_data_mmlu.append({"model": model, "length": val})
    violin_df_mmlu = pd.DataFrame(violin_data_mmlu)

    sns.violinplot(
        data=violin_df_mmlu,
        x="model",
        y="length",
        hue="model",
        order=MODELS,
        hue_order=MODELS,
        palette=COLORS,
        ax=ax,
        inner="box",
        linewidth=0.5,
        cut=0,
        legend=False,
    )

    ax.set_xlabel("", fontsize=7)
    ax.set_ylabel("", fontsize=7)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(
        [format_model_label(m) for m in MODELS],
        fontsize=6,
        rotation=30,
        ha="right",
    )
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylim(0, None)
    sns.despine(ax=ax, left=False, bottom=False)

    ax.text(
        0.5, 1.06, r"\textbf{MMLU-Pro}",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=7,
        clip_on=False,
    )

    ax = axes["C"]

    for model in MODELS:
        model_data = quant_gpqa[quant_gpqa["model"] == model]
        if model_data.empty:
            continue
        ax.plot(
            model_data["length_quantile"],
            model_data["mean_accuracy"],
            color=MODEL_COLOR_MAP[model],
            marker="o",
            markersize=4,
            linewidth=1.2,
        )

    ax.set_xlabel("Trace length quantile", fontsize=7)
    ax.set_ylabel("Accuracy", fontsize=7)
    ax.set_xticks(range(N_QUANTILES))
    ax.set_xticklabels(
        ["Shortest", "", "Medium", "", "Longest"],
        fontsize=6,
    )
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        [r"0$\%$", r"20$\%$", r"40$\%$", r"60$\%$", r"80$\%$", r"100$\%$"],
        fontsize=6,
    )
    sns.despine(ax=ax, left=False, bottom=False)

    ax = axes["D"]

    for model in MODELS:
        model_data = quant_mmlu[quant_mmlu["model"] == model]
        if model_data.empty:
            continue
        ax.plot(
            model_data["length_quantile"],
            model_data["mean_accuracy"],
            color=MODEL_COLOR_MAP[model],
            marker="o",
            markersize=4,
            linewidth=1.2,
        )

    ax.set_xlabel("Trace length quantile", fontsize=7)
    ax.set_ylabel("", fontsize=7)
    ax.set_xticks(range(N_QUANTILES))
    ax.set_xticklabels(
        ["Shortest", "", "Medium", "", "Longest"],
        fontsize=6,
    )
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8,1.0])
    ax.set_yticklabels(["", "", "", "", "", ""], fontsize=6)
    sns.despine(ax=ax, left=False, bottom=False)

    panel_letters = {
        "A": ("a", -0.12),
        "B": ("b", -0.04),
        "C": ("c", -0.12),
        "D": ("d", -0.04),
    }
    for pid, (letter, x_offset) in panel_letters.items():
        axes[pid].text(
            x_offset,
            1.05,
            f"\\textbf{{{letter}}}",
            transform=axes[pid].transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )

    legend_elements = [
        mlines.Line2D(
            [], [],
            color=MODEL_COLOR_MAP[m],
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=format_model_label(m),
        )
        for m in MODELS
    ]

    axes["C"].legend(
        handles=legend_elements,
        fontsize=6,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0,
        frameon=False,
    )

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
