from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


plt.rcParams["font.family"] = "Arial"
plt.rcParams["legend.title_fontsize"] = 7
plt.rcParams["text.usetex"] = True


DECILES = [20, 40, 60, 80]
BASE_MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b"]
TARGET_MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"]
DATASETS = ["gpqa", "mmlu"]
def format_model_label(model: str) -> str:
    if model.startswith("qwen3_"):
        size = model.split("_")[1].upper()
        return f"Qwen3-{size}"
    return model.replace("_", "-")


MODEL_LABELS = {m: format_model_label(m) for m in TARGET_MODELS}


def load_rescue_data(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "results" / "processed_rescue.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing rescue file: {path} (run scripts/08_process_rescue.py first)."
        )

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
    if not path.exists():
        raise FileNotFoundError(
            f"Missing outputs file: {path} (run scripts/06_process_outputs.py first)."
        )

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
    merged = merged.dropna(subset=["baseline_accuracy", "free_accuracy"])

    base_wrong_mask = merged["base_decile_prediction"] != merged["answer"]
    merged = merged[base_wrong_mask].copy()
    if merged.empty:
        raise ValueError("No rows remain after filtering to base-model mispredictions.")
    return merged


def compute_rescue_rates(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    col = "baseline_accuracy" if mode == "base" else "free_accuracy"
    grouped = (
        df.groupby(["dataset", "decile", "base_model", "target_model"])[col]
        .mean()
        .reset_index()
        .rename(columns={col: "rescue_rate"})
    )
    grouped["mode"] = mode
    return grouped


def compute_anchor_rates(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    pred_col = "baseline_prediction" if mode == "base" else "free_prediction"
    frame = df.copy()
    frame[pred_col] = frame[pred_col].astype("string").str.strip().str.upper()
    frame["anchor_flag"] = frame[pred_col] == frame["base_decile_prediction"]

    grouped = (
        frame.groupby(["dataset", "decile", "base_model", "target_model"])["anchor_flag"]
        .mean()
        .reset_index()
        .rename(columns={"anchor_flag": "anchor_rate"})
    )
    grouped["mode"] = mode
    return grouped


def pivot_grid(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grid = (
        frame.pivot(index="base_model", columns="target_model", values=value_col)
        .reindex(index=BASE_MODELS, columns=TARGET_MODELS)
    )
    return grid.astype(float)


def compute_vmax_by_dataset(df: pd.DataFrame, value_col: str, floor: float, cap: float) -> dict[str, float]:
    vmax: dict[str, float] = {}
    for dataset in DATASETS:
        subset = df[df["dataset"] == dataset]
        if subset.empty or subset[value_col].dropna().empty:
            vmax[dataset] = cap
            continue
        val = float(subset[value_col].max())
        vmax[dataset] = min(cap, max(floor, val))
    return vmax


def plot_grid(
    df: pd.DataFrame,
    value_col: str,
    value_label: str,
    output_path: Path,
    vmax_by_dataset: dict[str, float],
    cmap: str = "mako",
) -> None:
    rows = [
        ("gpqa", "base"),
        ("gpqa", "free"),
        ("mmlu", "base"),
        ("mmlu", "free"),
    ]
    fig, axes = plt.subplots(len(rows), len(DECILES), figsize=(7.0, 6.4), dpi=300)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.9, bottom=0.07, wspace=0.02, hspace=-0.18)

    dataset_rows = {
        "gpqa": [0, 1],
        "mmlu": [2, 3],
    }
    right_edge = max(ax.get_position().x1 for ax in axes[0])
    cbar_width = 0.02
    cbar_pad = 0.01
    cbar_axes: dict[str, plt.Axes] = {}
    for dataset, row_indices in dataset_rows.items():
        top_ax = axes[row_indices[0], 0]
        bottom_ax = axes[row_indices[-1], 0]
        y0 = bottom_ax.get_position().y0
        y1 = top_ax.get_position().y1
        full_height = y1 - y0
        shrink = 0.8
        y_offset = full_height * (1 - shrink) / 2
        cbar_axes[dataset] = fig.add_axes(
            [right_edge + cbar_pad, y0 + y_offset, cbar_width, full_height * shrink]
        )
    colormap = sns.color_palette(cmap, as_cmap=True)
    colormap.set_bad("#f5f5f5")

    colorbar_drawn = {ds: False for ds in cbar_axes}
    for row_idx, (dataset, mode) in enumerate(rows):
        subset = df[(df["dataset"] == dataset) & (df["mode"] == mode)]
        for col_idx, decile in enumerate(DECILES):
            ax = axes[row_idx, col_idx]
            panel = subset[subset["decile"] == decile]
            grid = pivot_grid(panel, value_col)
            has_data = grid.notna().any().any()

            if has_data:
                annotations = grid.apply(
                    lambda col: col.map(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "")
                )
                vmax = vmax_by_dataset.get(dataset, 1.0)
                heat = sns.heatmap(
                    grid,
                    ax=ax,
                    cmap=colormap,
                    vmin=0.0,
                    vmax=vmax,
                    mask=grid.isna(),
                    annot=annotations,
                    fmt="",
                    annot_kws={"fontsize": 5},
                    linewidths=0.35,
                    linecolor="white",
                    square=True,
                    cbar=not colorbar_drawn.get(dataset, False),
                    cbar_ax=cbar_axes.get(dataset) if not colorbar_drawn.get(dataset, False) else None,
                )
                if not colorbar_drawn.get(dataset, False):
                    cbar = heat.collections[0].colorbar
                    cbar.set_label("", fontsize=7)
                    cbar.ax.tick_params(labelsize=6)
                    colorbar_drawn[dataset] = True
            else:
                ax.set_facecolor("#f5f5f5")
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    fontsize=6,
                    transform=ax.transAxes,
                )

            if row_idx == 0:
                ax.set_title(rf"${decile}\%$", fontsize=7)
            else:
                ax.set_title("")

            show_y_labels = col_idx == 0
            ax.set_ylabel("")
            ax.set_xlabel("")

            ax.set_xticks([i + 0.5 for i in range(len(TARGET_MODELS))])
            if row_idx == len(rows) - 1:
                ax.set_xticklabels(
                    [MODEL_LABELS[m] for m in TARGET_MODELS],
                    rotation=45,
                    ha="right",
                    fontsize=6,
                )
            else:
                ax.set_xticklabels([])

            ax.set_yticks([i + 0.5 for i in range(len(BASE_MODELS))])
            if show_y_labels:
                ax.set_yticklabels([MODEL_LABELS[m] for m in BASE_MODELS], rotation=0, fontsize=6)
            else:
                ax.set_yticklabels([])

            ax.tick_params(length=0)

    for row_idx, (_, mode) in enumerate(rows):
        ax = axes[row_idx, 0]
        ax.text(
            -0.18,
            1.02,
            rf"\textbf{{{mode}}}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    dataset_rows = {
        "gpqa": [0, 1],
        "mmlu": [2, 3],
    }
    dataset_labels = {
        "gpqa": r"\textbf{GPQA Diamond}",
        "mmlu": r"\textbf{MMLU-Pro}",
    }
    mid_col = len(DECILES) // 2
    for dataset, row_indices in dataset_rows.items():
        ax_mid = axes[row_indices[0], mid_col]
        y_offset = 1.05
        if dataset == "gpqa":
            y_offset = 1.2
        ax_mid.text(
            0,
            y_offset,
            dataset_labels.get(dataset, dataset.upper()),
            transform=ax_mid.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    left_margin = min(ax.get_position().x0 for ax in axes[:, 0]) - 0.08
    panel_letters = ["a", "b", "c", "d"]
    for row_idx, letter in enumerate(panel_letters):
        ax = axes[row_idx, 0]
        pos = ax.get_position()
        y_center = pos.y1 + 0.02
        fig.text(
            left_margin,
            y_center,
            rf"\textbf{{{letter}}}",
            ha="right",
            va="center",
            fontsize=7,
            fontweight="bold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    rescue_df = load_rescue_data(repo_root)
    base_df = load_base_predictions(repo_root)
    merged = attach_base_predictions(rescue_df, base_df)

    rescue_rates = pd.concat(
        [
            compute_rescue_rates(merged, mode="base"),
            compute_rescue_rates(merged, mode="free"),
        ],
        ignore_index=True,
    )
    anchor_rates = pd.concat(
        [
            compute_anchor_rates(merged, mode="base"),
            compute_anchor_rates(merged, mode="free"),
        ],
        ignore_index=True,
    )

    rescue_vmax = compute_vmax_by_dataset(rescue_rates, "rescue_rate", floor=0.12, cap=0.7)
    anchor_vmax = compute_vmax_by_dataset(anchor_rates, "anchor_rate", floor=0.15, cap=0.7)

    figures_dir = repo_root / "results" / "figures"
    plot_grid(
        rescue_rates,
        value_col="rescue_rate",
        value_label="Rescue rate (target correct)",
        output_path=figures_dir / "main_figure_4.png",
        vmax_by_dataset=rescue_vmax,
        cmap="mako",
    )
    plot_grid(
        anchor_rates,
        value_col="anchor_rate",
        value_label="Anchoring rate (matches base wrong answer)",
        output_path=figures_dir / "appendix_anchoring.png",
        vmax_by_dataset=anchor_vmax,
        cmap="rocket_r",
    )


if __name__ == "__main__":
    main()
