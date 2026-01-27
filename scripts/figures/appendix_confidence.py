import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns
from pathlib import Path


plt.rcParams["font.family"] = "Arial"
plt.rcParams["legend.title_fontsize"] = 7
plt.rcParams["text.usetex"] = True


MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"]
COLORS = ["#0173b2", "#56b4e9", "#029e73", "#de8f05", "#d55e00"]
MODEL_COLOR_MAP = dict(zip(MODELS, COLORS))

DECILES = list(range(0, 101, 10))


def format_model_label(model: str) -> str:
    if model.startswith("qwen3_"):
        size = model.split("_", 1)[1]
        return f"Qwen3-{size.upper()}"
    return model.replace("_", "-")


def load_data(repo_root: Path) -> pd.DataFrame:
    """Load prediction probabilities and accuracy."""
    path = repo_root / "results" / "processed_outputs.parquet"
    cols = ["dataset", "model", "run", "decile", "decile_prediction_prob", "decile_accuracy"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["model"].isin(MODELS)]
    df = df[df["decile"].isin(DECILES)]
    return df


def compute_confidence_by_correctness(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for dataset in ["gpqa", "mmlu"]:
        for model in MODELS:
            for decile in DECILES:
                subset = df[
                    (df["dataset"] == dataset) &
                    (df["model"] == model) &
                    (df["decile"] == decile)
                ]

                if subset.empty:
                    continue

                correct = subset[subset["decile_accuracy"] == 1]["decile_prediction_prob"]
                incorrect = subset[subset["decile_accuracy"] == 0]["decile_prediction_prob"]

                conf_correct = correct.mean() if len(correct) > 0 else float("nan")
                conf_incorrect = incorrect.mean() if len(incorrect) > 0 else float("nan")

                records.append({
                    "dataset": dataset,
                    "model": model,
                    "decile": decile,
                    "conf_correct": conf_correct,
                    "conf_incorrect": conf_incorrect,
                    "n_correct": len(correct),
                    "n_incorrect": len(incorrect),
                })

    result = pd.DataFrame(records)
    result["discrimination_gap"] = result["conf_correct"] - result["conf_incorrect"]
    return result


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "results" / "figures" / "appendix_confidence.png"

    df = load_data(repo_root)
    conf_df = compute_confidence_by_correctness(df)

    fig, axes = plt.subplot_mosaic(
        [["A", "B"], ["C", "D"]],
        figsize=(7.08, 5.5),
        dpi=300,
        gridspec_kw={"wspace": 0.15, "hspace": 0.35},
    )

    ax = axes["A"]

    for model in MODELS:
        subset = conf_df[(conf_df["dataset"] == "gpqa") & (conf_df["model"] == model)]
        ax.plot(
            subset["decile"],
            subset["conf_correct"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            linestyle="-",
            color=MODEL_COLOR_MAP[model],
        )
        ax.plot(
            subset["decile"],
            subset["conf_incorrect"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            linestyle="--",
            color=MODEL_COLOR_MAP[model],
            alpha=0.6,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("", fontsize=7)
    ax.set_ylabel("Mean confidence", fontsize=7)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["", "", "", "", "", ""], fontsize=6)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([r"0$\%$", r"20$\%$", r"40$\%$", r"60$\%$", r"80$\%$", r"100$\%$"], fontsize=6)
    ax.set_title(r"\textbf{GPQA Diamond}", fontsize=7, pad=6)
    sns.despine(ax=ax)

    line_correct = mlines.Line2D([], [], color="gray", linestyle="-", linewidth=1.2, label="Correct")
    line_incorrect = mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.2, alpha=0.6, label="Incorrect")
    ax.legend(handles=[line_correct, line_incorrect], fontsize=6, loc="lower right", frameon=False)

    ax = axes["B"]

    for model in MODELS:
        subset = conf_df[(conf_df["dataset"] == "mmlu") & (conf_df["model"] == model)]
        ax.plot(
            subset["decile"],
            subset["conf_correct"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            linestyle="-",
            color=MODEL_COLOR_MAP[model],
        )
        ax.plot(
            subset["decile"],
            subset["conf_incorrect"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            linestyle="--",
            color=MODEL_COLOR_MAP[model],
            alpha=0.6,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("", fontsize=7)
    ax.set_ylabel("", fontsize=7)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["", "", "", "", "", ""], fontsize=6)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["", "", "", "", "", ""], fontsize=6)
    ax.set_title(r"\textbf{MMLU-Pro}", fontsize=7, pad=6)
    sns.despine(ax=ax)

    ax = axes["C"]

    for model in MODELS:
        subset = conf_df[(conf_df["dataset"] == "gpqa") & (conf_df["model"] == model)]
        ax.plot(
            subset["decile"],
            subset["discrimination_gap"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=MODEL_COLOR_MAP[model],
            label=format_model_label(model),
        )

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 0.38)
    ax.set_xlabel("Reasoning decile", fontsize=7)
    ax.set_ylabel("Discrimination gap", fontsize=7)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([r"0$\%$", r"20$\%$", r"40$\%$", r"60$\%$", r"80$\%$", r"100$\%$"], fontsize=6)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels([r"0$\%$", r"10$\%$", r"20$\%$", r"30$\%$"], fontsize=6)
    sns.despine(ax=ax)

    legend_elements = [
        mlines.Line2D([], [], color=MODEL_COLOR_MAP[m], marker="o", markersize=3,
                      linewidth=1.2, label=format_model_label(m))
        for m in MODELS
    ]
    ax.legend(handles=legend_elements, fontsize=5.5, loc="upper right", frameon=False)

    ax = axes["D"]

    for model in MODELS:
        subset = conf_df[(conf_df["dataset"] == "mmlu") & (conf_df["model"] == model)]
        ax.plot(
            subset["decile"],
            subset["discrimination_gap"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=MODEL_COLOR_MAP[model],
            label=format_model_label(model),
        )

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 0.38)
    ax.set_xlabel("Reasoning decile", fontsize=7)
    ax.set_ylabel("", fontsize=7)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([r"0$\%$", r"20$\%$", r"40$\%$", r"60$\%$", r"80$\%$", r"100$\%$"], fontsize=6)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["", "", "", ""], fontsize=6)
    sns.despine(ax=ax)

    panel_labels = [("A", "a"), ("B", "b"), ("C", "c"), ("D", "d")]
    x_offsets = {"A": -0.12, "B": -0.04, "C": -0.12, "D": -0.04}

    for pid, label in panel_labels:
        axes[pid].text(
            x_offsets[pid], 1.06,
            f"\\textbf{{{label}}}",
            transform=axes[pid].transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
