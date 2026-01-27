import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


plt.rcParams["font.family"] = "Arial"
plt.rcParams["legend.title_fontsize"] = 7
plt.rcParams["text.usetex"] = True


MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"]


def format_model_label(model: str) -> str:
    if model.startswith("qwen3_"):
        size = model.split("_", 1)[1]
        return f"Qwen3-{size.upper()}"
    return model.replace("_", "-")


def format_category_label(category: str) -> str:
    return category.replace("_", " ").title()


def load_data(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "results" / "processed_outputs.parquet"
    cols = ["dataset", "model", "run", "id", "decile", "category", "decile_accuracy"]
    df = pd.read_parquet(
        path,
        columns=cols,
        filters=[("dataset", "==", "mmlu"), ("decile", "in", [0, 100])],
    )
    df = df[df["model"].isin(MODELS)]
    return df


def compute_category_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    d0 = (
        df[df["decile"] == 0]
        .groupby(["category", "model"])["decile_accuracy"]
        .mean()
        .unstack()
    )
    d100 = (
        df[df["decile"] == 100]
        .groupby(["category", "model"])["decile_accuracy"]
        .mean()
        .unstack()
    )
    gain = d100 - d0

    category_order = gain.mean(axis=1).sort_values(ascending=False).index.tolist()

    d100 = d100.reindex(index=category_order, columns=MODELS)
    gain = gain.reindex(index=category_order, columns=MODELS)

    return d100, gain, category_order


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "results" / "figures" / "appendix_mmlu_categories.png"

    df = load_data(repo_root)
    acc_100, gain, category_order = compute_category_stats(df)

    fig, axes = plt.subplot_mosaic(
        [["A", "B"]],
        figsize=(7.08, 4.5),
        dpi=300,
        gridspec_kw={"wspace": 0.02},
    )

    cmap_acc = sns.color_palette("mako", as_cmap=True)
    cmap_gain = sns.color_palette("rocket", as_cmap=True)

    ax = axes["A"]

    annot_acc = acc_100.map(lambda x: f"{x * 100:.0f}" if pd.notna(x) else "")

    sns.heatmap(
        acc_100,
        ax=ax,
        cmap=cmap_acc,
        vmin=0.3,
        vmax=1.0,
        annot=annot_acc,
        fmt="",
        annot_kws={"fontsize": 6},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "aspect": 30},
    )

    ax.set_xlabel("", fontsize=7)
    ax.set_ylabel("", fontsize=7)

    ax.set_yticks([i + 0.5 for i in range(len(category_order))])
    ax.set_yticklabels(
        [format_category_label(c) for c in category_order],
        fontsize=6,
        rotation=0,
    )

    ax.set_xticks([i + 0.5 for i in range(len(MODELS))])
    ax.set_xticklabels(
        [format_model_label(m) for m in MODELS],
        fontsize=6,
        rotation=45,
        ha="right",
    )

    ax.tick_params(length=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cbar.set_ticklabels(
        [r"30$\%$", r"40$\%$", r"50$\%$", r"60$\%$", r"70$\%$", r"80$\%$", r"90$\%$", r"100$\%$"]
    )

    ax.set_title(r"\textbf{Final accuracy (decile 100)}", fontsize=7, pad=8)

    ax = axes["B"]

    annot_gain = gain.map(lambda x: f"{x * 100:.0f}" if pd.notna(x) else "")

    sns.heatmap(
        gain,
        ax=ax,
        cmap=cmap_gain,
        vmin=0.0,
        vmax=0.65,
        annot=annot_gain,
        fmt="",
        annot_kws={"fontsize": 6},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "aspect": 30},
    )

    ax.set_xlabel("", fontsize=7)
    ax.set_ylabel("", fontsize=7)

    ax.set_yticks([i + 0.5 for i in range(len(category_order))])
    ax.set_yticklabels([], fontsize=6)

    ax.set_xticks([i + 0.5 for i in range(len(MODELS))])
    ax.set_xticklabels(
        [format_model_label(m) for m in MODELS],
        fontsize=6,
        rotation=45,
        ha="right",
    )

    ax.tick_params(length=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cbar.set_ticklabels(
        [r"0$\%$", r"10$\%$", r"20$\%$", r"30$\%$", r"40$\%$", r"50$\%$", r"60$\%$"]
    )

    ax.set_title(r"\textbf{Accuracy gain (decile 100 $-$ decile 0)}", fontsize=7, pad=8)

    axes["A"].text(
        -0.02,
        1.02,
        r"\textbf{a}",
        transform=axes["A"].transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        clip_on=False,
    )
    axes["B"].text(
        -0.02,
        1.02,
        r"\textbf{b}",
        transform=axes["B"].transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        clip_on=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
