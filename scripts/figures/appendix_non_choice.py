import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from pathlib import Path


plt.rcParams["font.family"] = "Arial"
plt.rcParams["text.usetex"] = True
plt.rcParams["legend.title_fontsize"] = 7


DATASETS = ["gpqa", "mmlu"]
TOP_K = 5
DECILES = list(range(0, 101, 10))
MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"]
TREAT_MISSING_AS_ZERO = True


def format_model_label(model: str) -> str:
    if model.startswith("qwen3_"):
        size = model.split("_", maxsplit=1)[1].upper()
        return f"Qwen3-{size}"
    return model.replace("_", "-")


def to_records(top_non_choice):
    if top_non_choice is None:
        return []
    if isinstance(top_non_choice, float) and np.isnan(top_non_choice):
        return []
    if isinstance(top_non_choice, np.ndarray):
        return top_non_choice.tolist()
    if isinstance(top_non_choice, (list, tuple)):
        return list(top_non_choice)
    return []


def clean_token_repr(token: str) -> str:
    if not isinstance(token, str):
        return ""
    if len(token) >= 2 and token[0] == token[-1] == "'":
        token = token[1:-1]

    latex_replacements = {
        "#": r"\#",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "$": r"\$",
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }

    cleaned = []
    for ch in token:
        if not ch.isprintable():
            continue
        cleaned.append(latex_replacements.get(ch, ch))

    return "".join(cleaned)


def most_common_token(group: pd.DataFrame) -> str:
    counts = group["token_id"].value_counts()
    top_count = counts.iloc[0]
    candidate_ids = counts[counts == top_count].index.tolist()

    if len(candidate_ids) == 1:
        token_id = candidate_ids[0]
    else:
        means = (
            group[group["token_id"].isin(candidate_ids)]
            .groupby("token_id")["prob"]
            .mean()
            .sort_values(ascending=False)
        )
        token_id = means.index[0]

    token_reprs = group.loc[group["token_id"] == token_id, "token_repr"]
    mode_repr = token_reprs.mode()
    token_repr = mode_repr.iloc[0] if not mode_repr.empty else token_reprs.iloc[0]
    return clean_token_repr(token_repr)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in df.itertuples(index=False):
        entries = to_records(row.top_non_choice_logprobs)
        seen = set()
        for entry in entries:
            token_id = entry.get("token_id")
            logprob = entry.get("logprob")
            if logprob is None or pd.isna(logprob):
                continue
            seen.add(token_id)
            records.append(
                {
                    "dataset": row.dataset,
                    "model": row.model,
                    "decile": row.decile,
                    "token_id": token_id,
                    "token_repr": entry.get("token_repr", ""),
                    "prob": float(np.exp(logprob)),
                }
            )

    flat = pd.DataFrame(records)
    if flat.empty:
        return flat

    counts = (
        df.groupby(["model", "decile"])
        .size()
        .reset_index(name="n_examples")
    )

    token_means = (
        flat.groupby(["model", "decile", "token_id"])
        .agg(
            mean_prob=("prob", "mean"),
            token_repr=("token_repr", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            freq=("prob", "size"),
        )
        .reset_index()
    )

    if TREAT_MISSING_AS_ZERO:
        token_means = token_means.merge(counts, on=["model", "decile"], how="left")
        token_means["mean_prob"] = token_means["mean_prob"] * (token_means["freq"] / token_means["n_examples"])

    token_means = token_means.sort_values(
        by=["model", "decile", "mean_prob", "freq", "token_id"],
        ascending=[True, True, False, False, True],
    )
    top_tokens = (
        token_means.groupby(["model", "decile"])
        .head(TOP_K)
        .reset_index(drop=True)
    )
    top_tokens["rank"] = top_tokens.groupby(["model", "decile"]).cumcount() + 1

    top_tokens = top_tokens.rename(columns={"token_repr": "token_label"})
    return top_tokens[["model", "decile", "rank", "mean_prob", "token_label"]]


def main():
    repo_root = Path(__file__).resolve().parents[2]
    results_path = repo_root / "results" / "processed_outputs.parquet"
    output_path = repo_root / "results" / "figures" / "appendix_non_choice.png"
    cols = ["dataset", "model", "run", "decile", "top_non_choice_logprobs"]
    filters = None
    if DATASETS:
        filters = [("dataset", "in", DATASETS)]

    df = pd.read_parquet(
        results_path,
        columns=cols,
        filters=filters,
    )

    df = df[df["model"].isin(MODELS)]
    df = df[df["decile"].isin(DECILES)]

    summary = build_summary(df)
    if summary.empty:
        raise ValueError("No data available after filtering for dataset/models.")
    decile_order = DECILES
    rank_order = list(range(1, TOP_K + 1))

    vmax = summary["mean_prob"].max()
    vmin = 0.0

    fig, axes = plt.subplots(
        len(MODELS),
        1,
        figsize=(7.08, 8.0),
        dpi=300,
        sharex=True,
        constrained_layout=False,
    )

    cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        model_df = summary[summary["model"] == model]

        prob_grid = (
            model_df.pivot(index="rank", columns="decile", values="mean_prob")
            .reindex(index=rank_order, columns=decile_order)
        )
        token_grid = (
            model_df.pivot(index="rank", columns="decile", values="token_label")
            .reindex(index=rank_order, columns=decile_order)
            .fillna("")
            .map(clean_token_repr)
        )

        heat = sns.heatmap(
            prob_grid,
            ax=ax,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            linewidths=0.25,
            linecolor="white",
            cbar=(idx == 0),
            cbar_ax=cbar_ax if idx == 0 else None,
            annot=token_grid,
            fmt="",
            annot_kws={"fontsize": 5},
        )
        ax.set_ylabel("")

        ax.text(
            -0.02,
            1.02,
            format_model_label(model),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
        )
        ax.set_yticks([i + 0.5 for i in range(len(rank_order))])
        ax.set_yticklabels(["" for _ in rank_order])
        ax.set_xticks([i + 0.5 for i in range(len(decile_order))])
        ax.set_xticklabels([f"{d}$\\%$" for d in decile_order], rotation=0, fontsize=6)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=0)

        if idx < len(MODELS) - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Reasoning decile", fontsize=7)

    cbar = axes[0].collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    fig.subplots_adjust(hspace=0.25, right=0.86, left=0.18, top=0.98, bottom=0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
