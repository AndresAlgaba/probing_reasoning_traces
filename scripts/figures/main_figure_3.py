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
CONTROL_TYPES = ["original", "cross", "junk", "shuffle"]
DATASETS = ["gpqa", "mmlu"]
ROW_VARIANTS = [
    ("original", "original"),
    ("junk", "random"),
    ("cross", "swap"),
    ("shuffle", "shuffle"),
]


def format_model_label(model: str) -> str:
    if model.startswith("qwen3_"):
        size = model.split("_", 1)[1]
        return f"Qwen3-{size.upper()}"
    return model.replace("_", "-")


def load_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_path = repo_root / "results" / "processed_outputs.parquet"
    controls_path = repo_root / "results" / "processed_controls.parquet"

    base_cols = ["dataset", "model", "run", "decile", "decile_accuracy"]
    ctrl_cols = ["dataset", "model", "run", "decile", "control_type", "control_accuracy"]

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Missing baseline file: {processed_path} (run scripts/06_process_outputs.py first)."
        )
    if not controls_path.exists():
        raise FileNotFoundError(
            f"Missing control file: {controls_path} (run scripts/07_process_controls.py first)."
        )

    df_base = pd.read_parquet(processed_path, columns=base_cols)
    df_ctrl = pd.read_parquet(controls_path, columns=ctrl_cols)
    return df_base, df_ctrl


def build_decile_points(df_base: pd.DataFrame, decile: int) -> dict[str, dict[str, float]]:
    subset = df_base[df_base["decile"] == decile]
    if subset.empty:
        return {}
    grouped = subset.groupby(["dataset", "model"])["decile_accuracy"].mean()
    points: dict[str, dict[str, float]] = {}
    for (dataset, model), val in grouped.items():
        points.setdefault(dataset, {})[model] = float(val)
    return points


def build_baseline_points(df_base: pd.DataFrame) -> dict[str, dict[str, float]]:
    return build_decile_points(df_base, decile=0)


def build_final_points(df_base: pd.DataFrame) -> dict[str, dict[str, float]]:
    return build_decile_points(df_base, decile=100)


def prepare_variant_frame(
    df_base: pd.DataFrame,
    df_ctrl: pd.DataFrame,
    dataset: str,
    variant: str,
) -> pd.DataFrame:
    if variant == "original":
        subset = df_base[df_base["dataset"] == dataset].copy()
        subset = subset.rename(columns={"decile_accuracy": "accuracy"})
        subset["variant"] = "original"
    else:
        subset = df_ctrl[
            (df_ctrl["dataset"] == dataset) & (df_ctrl["control_type"] == variant)
        ].copy()
        subset = subset.rename(columns={"control_accuracy": "accuracy"})
        subset["variant"] = variant
    return subset


def configure_axis(ax, show_xlabel: bool, show_ylabel: bool, y_label: str = "Accuracy") -> None:
    ax.set(xlim=(0, 100), ylim=(0, 0.82))
    ax.set_xlabel("Reasoning deciles" if show_xlabel else "", fontsize=7)
    ax.set_ylabel(y_label if show_ylabel else "", fontsize=7)
    ax.set_xticks(
        ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=[r"0$\%$", "", r"20$\%$", "", r"40$\%$", "", r"60$\%$", "", r"80$\%$", "", r"100$\%$"] if show_xlabel else ["", "", "", "", "", "", "", "", "", "", ""],
        fontsize=7,
    )
    ax.set_yticks(
        [0, 0.2, 0.4, 0.6, 0.8],
        labels=[r"0$\%$", r"20$\%$", r"40$\%$", r"60$\%$", r"80$\%$"] if show_ylabel else ["", "", "", "", ""],
        fontsize=7,
    )
    ax.set_title("")
    sns.despine(ax=ax, left=False, bottom=False)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        f"\\textbf{{{label}}}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        clip_on=False,
    )


def plot_panel(ax, df: pd.DataFrame) -> None:
    sns.lineplot(
        data=df,
        x="decile",
        y="accuracy",
        hue="model",
        hue_order=MODELS,
        palette=COLORS,
        estimator="mean",
        errorbar=None,
        legend=False,
        ax=ax,
    )


def add_decile_dots(ax, points: dict[str, float], x_position: float) -> None:
    for model, color in zip(MODELS, COLORS):
        if model not in points:
            continue
        ax.scatter(
            x=[x_position],
            y=[points[model]],
            color=color,
            s=20,
            marker="o",
            edgecolor="black",
            linewidths=0.6,
            alpha=0.95,
            zorder=6,
        )


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "results" / "figures" / "main_figure_3.png"

    df_base, df_ctrl = load_data(repo_root)
    baseline_points = build_baseline_points(df_base)
    final_points = build_final_points(df_base)

    panel_ids = [
        chr(ord("A") + idx) for idx in range(len(DATASETS) * len(ROW_VARIANTS))
    ]
    mosaic_rows = [
        panel_ids[i * len(DATASETS) : (i + 1) * len(DATASETS)]
        for i in range(len(ROW_VARIANTS))
    ]

    Figure4, ax = plt.subplot_mosaic(
        mosaic_rows,
        figsize=(7.08, 6.69),
        dpi=300,
        gridspec_kw={"wspace": 0.1, "hspace": 0.3},
    )

    panels = []
    for row_idx, (variant, _) in enumerate(ROW_VARIANTS):
        for col_idx, dataset in enumerate(DATASETS):
            panels.append((mosaic_rows[row_idx][col_idx], dataset, variant))

    for panel_id, dataset, variant in panels:
        frame = prepare_variant_frame(df_base, df_ctrl, dataset, variant)
        if frame.empty:
            raise ValueError(f"No data available for dataset={dataset}, variant={variant}.")
        plot_panel(ax[panel_id], frame)
        if variant != "original":
            add_decile_dots(ax[panel_id], baseline_points.get(dataset, {}), x_position=2)
            add_decile_dots(ax[panel_id], final_points.get(dataset, {}), x_position=98)

    for row_idx, row_keys in enumerate(mosaic_rows):
        show_xlabel = row_idx == len(mosaic_rows) - 1
        for col_idx, panel_id in enumerate(row_keys):
            configure_axis(ax[panel_id], show_xlabel=show_xlabel, show_ylabel=(col_idx == 0))

    ax[mosaic_rows[0][0]].text(
        0.5,
        1.1,
        r"\textbf{GPQA Diamond}",
        transform=ax[mosaic_rows[0][0]].transAxes,
        ha="center",
        va="top",
        fontsize=7,
        clip_on=False,
    )
    ax[mosaic_rows[0][1]].text(
        0.5,
        1.1,
        r"\textbf{MMLU-Pro}",
        transform=ax[mosaic_rows[0][1]].transAxes,
        ha="center",
        va="top",
        fontsize=7,
        clip_on=False,
    )

    for idx, pid in enumerate(panel_ids):
        letter = chr(ord("a") + idx)
        x_offset = -0.14 if idx % len(DATASETS) == 0 else -0.06
        ax[pid].text(
            x_offset,
            1.05,
            f"\\textbf{{{letter}}}",
            transform=ax[pid].transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )

    for row_idx, (_, row_label) in enumerate(ROW_VARIANTS):
        add_panel_label(ax[mosaic_rows[row_idx][0]], row_label)

    legend_elements_qwen = [
        mlines.Line2D([], [], color=COLORS[i], label=format_model_label(MODELS[i]))
        for i in range(3)
    ]
    legend_elements_gpt = [
        mlines.Line2D([], [], color=COLORS[i], label=format_model_label(MODELS[i]))
        for i in range(3, len(MODELS))
    ]
    ax[mosaic_rows[0][1]].legend(
        handles=legend_elements_qwen + legend_elements_gpt,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(0.31, 0.47),
        borderaxespad=0,
        frameon=False,
        ncol=2,
        handlelength=1.5,
        columnspacing=1.5,
        labelspacing=0.6,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Figure4.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
