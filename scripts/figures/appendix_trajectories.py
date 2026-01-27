import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path


plt.rcParams["font.family"] = "Arial"
plt.rcParams["legend.title_fontsize"] = 7
plt.rcParams["text.usetex"] = True


MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"]

OUTCOME_COLORS = {
    "stable_correct": "#029e73",
    "gained": "#56b4e9",
    "lost": "#de8f05",
    "stable_wrong": "#949494",
}

OUTCOME_LABELS = {
    "stable_correct": "Stable correct",
    "gained": r"Gained (wrong$\rightarrow$right)",
    "lost": r"Lost (right$\rightarrow$wrong)",
    "stable_wrong": "Stable wrong",
}


def format_model_label(model: str) -> str:
    if model.startswith("qwen3_"):
        size = model.split("_", 1)[1]
        return f"Qwen3-{size.upper()}"
    return model.replace("_", "-")


def load_data(repo_root: Path) -> pd.DataFrame:
    """Load accuracy data at decile 0 and 100, compute trajectory outcomes."""
    path = repo_root / "results" / "processed_outputs.parquet"
    cols = ["dataset", "model", "run", "id", "decile", "decile_accuracy"]
    df = pd.read_parquet(
        path,
        columns=cols,
        filters=[("decile", "in", [0, 100])],
    )
    df = df[df["model"].isin(MODELS)]

    d0 = df[df["decile"] == 0][["dataset", "model", "run", "id", "decile_accuracy"]].copy()
    d0 = d0.rename(columns={"decile_accuracy": "acc_0"})

    d100 = df[df["decile"] == 100][["dataset", "model", "run", "id", "decile_accuracy"]].copy()
    d100 = d100.rename(columns={"decile_accuracy": "acc_100"})

    merged = d0.merge(d100, on=["dataset", "model", "run", "id"])

    merged["stable_correct"] = (merged["acc_0"] == 1) & (merged["acc_100"] == 1)
    merged["gained"] = (merged["acc_0"] == 0) & (merged["acc_100"] == 1)
    merged["lost"] = (merged["acc_0"] == 1) & (merged["acc_100"] == 0)
    merged["stable_wrong"] = (merged["acc_0"] == 0) & (merged["acc_100"] == 0)

    return merged


def compute_outcome_rates(df: pd.DataFrame) -> pd.DataFrame:
    outcomes = ["stable_correct", "gained", "lost", "stable_wrong"]
    grouped = df.groupby(["dataset", "model"])[outcomes].mean().reset_index()
    return grouped


def plot_stacked_bars(ax, data: pd.DataFrame, dataset: str, show_ylabel: bool = True):
    subset = data[data["dataset"] == dataset].copy()
    subset = subset.set_index("model").reindex(MODELS)

    outcomes = ["stable_correct", "gained", "lost", "stable_wrong"]

    y_positions = range(len(MODELS))
    bar_height = 0.7

    left = [0] * len(MODELS)
    for outcome in outcomes:
        values = subset[outcome].values
        ax.barh(
            y_positions,
            values,
            height=bar_height,
            left=left,
            color=OUTCOME_COLORS[outcome],
            edgecolor="white",
            linewidth=0.5,
        )
        left = [l + v for l, v in zip(left, values)]

    cumsum = [0] * len(MODELS)
    for outcome in outcomes:
        values = subset[outcome].values
        for i, v in enumerate(values):
            if v >= 0.06:
                x_pos = cumsum[i] + v / 2
                ax.text(
                    x_pos,
                    i,
                    f"{v * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if outcome != "stable_wrong" else "black",
                    fontweight="medium",
                )
            cumsum[i] += v

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(MODELS) - 0.5)
    ax.set_yticks(y_positions)
    if show_ylabel:
        ax.set_yticklabels([format_model_label(m) for m in MODELS], fontsize=6)
    else:
        ax.set_yticklabels([])

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(
        [r"0$\%$", r"25$\%$", r"50$\%$", r"75$\%$", r"100$\%$"],
        fontsize=6,
    )
    ax.set_xlabel("Proportion of questions", fontsize=7)

    ax.tick_params(length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.invert_yaxis()


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "results" / "figures" / "appendix_trajectories.png"

    df = load_data(repo_root)
    outcome_rates = compute_outcome_rates(df)

    # Create figure with 1x2 layout
    fig, axes = plt.subplot_mosaic(
        [["A", "B"]],
        figsize=(7.08, 2.8),
        dpi=300,
        gridspec_kw={"wspace": 0.08},
    )

    plot_stacked_bars(axes["A"], outcome_rates, "gpqa", show_ylabel=True)
    axes["A"].set_title(r"\textbf{GPQA Diamond}", fontsize=7, pad=8)

    plot_stacked_bars(axes["B"], outcome_rates, "mmlu", show_ylabel=False)
    axes["B"].set_title(r"\textbf{MMLU-Pro}", fontsize=7, pad=8)

    axes["A"].text(
        -0.18,
        1.08,
        r"\textbf{a}",
        transform=axes["A"].transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        clip_on=False,
    )
    axes["B"].text(
        -0.02,
        1.08,
        r"\textbf{b}",
        transform=axes["B"].transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        clip_on=False,
    )

    legend_patches = [
        mpatches.Patch(color=OUTCOME_COLORS[k], label=OUTCOME_LABELS[k])
        for k in ["stable_correct", "gained", "lost", "stable_wrong"]
    ]

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        fontsize=6,
        frameon=False,
        handlelength=1.2,
        handleheight=0.8,
        columnspacing=1.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
