import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.lines as mlines


plt.rcParams["font.family"] = "Arial"
plt.rcParams['legend.title_fontsize'] = 7
plt.rcParams["text.usetex"] = True


GPQA_CHOICES = ['A', 'B', 'C', 'D']
MMLU_CHOICES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

MODELS = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"]
COLORS = ['#0173b2', '#56b4e9', '#029e73', '#de8f05', '#d55e00']


def format_model_label(model: str) -> str:
    if model.startswith("qwen3_"):
        size = model.split("_", 1)[1]
        return f"Qwen3-{size.upper()}"
    return model.replace("_", "-")


cols = [
    "dataset",
    "model",
    "run",
    "id",
    "decile",
    "decile_accuracy",
    "decile_choice_mass",
    "decile_prediction_prob_final_answer",
    "decile_prediction_flip",
    "decile_flip_rate",
]
df_gpqa = pd.read_parquet("../../results/processed_outputs.parquet",
                          columns=cols,
                          filters=[("dataset", "==", "gpqa")])
df_mmlu = pd.read_parquet("../../results/processed_outputs.parquet",
                          columns=cols,
                          filters=[("dataset", "==", "mmlu")])

for df in (df_gpqa, df_mmlu):
    df["decile_non_choice_mass"] = 1 - df["decile_choice_mass"]

# Exclude the 0% reasoning decile for flip-rate visuals
df_gpqa_flip = df_gpqa[df_gpqa["decile"] > 0]
df_mmlu_flip = df_mmlu[df_mmlu["decile"] > 0]

Figure1, ax1 = plt.subplot_mosaic(
    [
        ['A', 'B',],
        ['C', 'D',],
        ['E', 'F',],
        ['G', 'H'],
    ],
    figsize=(7.08, 6.69),
    dpi=300,
    gridspec_kw={'wspace': 0.1, 'hspace': 0.2},
)

sns.lineplot(
    data=df_gpqa,
    x="decile",
    y="decile_accuracy",
    hue="model",
    hue_order=MODELS,
    palette=COLORS,
    estimator="mean",
    errorbar=None, # ("ci", 95),
    legend=False,
    ax=ax1['A'],
)

ax1["A"].set(xlim=(0, 100), ylim=(0, 0.82))
ax1["A"].set_xlabel("", fontsize=7)
ax1["A"].set_ylabel("Accuracy", fontsize=7,)
ax1["A"].set_xticks(
    ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=["", "", "", "", "", "", "", "", "", "", ""],
    fontsize=7,
)
ax1["A"].set_yticks(
    [0,0.2,0.4,0.6,0.8],
    labels=[r"0$\%$",r"20$\%$",r"40$\%$",r"60$\%$",r"80$\%$"],
    fontsize=7,
)
ax1["A"].set_title("")

sns.despine(ax=ax1["A"],left=False, bottom=False)

ax1["A"].text(
    0.5, 1.04, r"\textbf{GPQA Diamond}",
    transform=ax1["A"].transAxes,
    ha="center", va="top",
    fontsize=7,
    clip_on=False,
)

sns.lineplot(
    data=df_mmlu,
    x="decile",
    y="decile_accuracy",
    hue="model",
    hue_order=MODELS,
    palette=COLORS,
    estimator="mean",
    errorbar=None,
    legend=False,
    ax=ax1['B'],
)

ax1["B"].set(xlim=(0, 100), ylim=(0, 0.82))
ax1["B"].set_xlabel("", fontsize=7)
ax1["B"].set_ylabel("", fontsize=7, rotation=0)
ax1["B"].set_xticks(
    ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=["", "", "", "", "", "", "", "", "", "", ""],
    fontsize=7,
)
ax1["B"].set_yticks(
    [0,0.2,0.4,0.6,0.8],
    labels=["","","","",""],
    fontsize=7,
)
ax1["B"].set_title("")

sns.despine(ax=ax1["B"],left=False, bottom=False)

ax1["B"].text(
    0.5, 1.04, r"\textbf{MMLU-Pro}",
    transform=ax1["B"].transAxes,
    ha="center", va="top",
    fontsize=7,
    clip_on=False,
)

sns.lineplot(
    data=df_gpqa,
    x="decile",
    y="decile_prediction_prob_final_answer",
    hue="model",
    hue_order=MODELS,
    palette=COLORS,
    estimator="mean",
    errorbar=None,
    legend=False,
    ax=ax1['C'],
)

ax1["C"].set(xlim=(0, 100), ylim=(0, 1.00))
ax1["C"].set_xlabel("", fontsize=7)
ax1["C"].set_ylabel("p(final answer)", fontsize=7,)
ax1["C"].set_xticks(
    ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=["", "", "", "", "", "", "", "", "", "", ""],
    fontsize=7,
)
ax1["C"].set_yticks(
    [0,0.2,0.4,0.6,0.8,1.0],
    labels=[r"0$\%$",r"20$\%$",r"40$\%$",r"60$\%$",r"80$\%$",r"100$\%$"],
    fontsize=7,
)
ax1["C"].set_title("")

sns.despine(ax=ax1["C"],left=False, bottom=False)

sns.lineplot(
    data=df_mmlu,
    x="decile",
    y="decile_prediction_prob_final_answer",
    hue="model",
    hue_order=MODELS,
    palette=COLORS,
    estimator="mean",
    errorbar=None,
    legend=False,
    ax=ax1['D'],
)

ax1["D"].set(xlim=(0, 100), ylim=(0, 1.00))
ax1["D"].set_xlabel("", fontsize=7)
ax1["D"].set_ylabel("", fontsize=7, rotation=0)
ax1["D"].set_xticks(
    ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=["", "", "", "", "", "", "", "", "", "", ""],
    fontsize=7,
)
ax1["D"].set_yticks(
    [0,0.2,0.4,0.6,0.8,1.0],
    labels=["","","","","",""],
    fontsize=7,
)
ax1["D"].set_title("")

sns.despine(ax=ax1["D"],left=False, bottom=False)

sns.lineplot(
    data=df_gpqa,
    x="decile",
    y="decile_non_choice_mass",
    hue="model",
    hue_order=MODELS,
    palette=COLORS,
    estimator="mean",
    errorbar=None,
    legend=False,
    ax=ax1['E'],
)

ax1["E"].set(xlim=(0, 100), ylim=(0, 0.46))
ax1["E"].set_xlabel("", fontsize=7)
ax1["E"].set_ylabel("Non-choice prob.", fontsize=7,)
ax1["E"].set_xticks(
    ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=["", "", "", "", "", "", "", "", "", "", ""],
    fontsize=7,
)
ax1["E"].set_yticks(
    [0.0,0.1,0.2,0.3,0.4],
    labels=[r"0$\%$",r"10$\%$",r"20$\%$",r"30$\%$",r"40$\%$"],
    fontsize=7,
)
ax1["E"].set_title("")

sns.despine(ax=ax1["E"],left=False, bottom=False)

sns.lineplot(
    data=df_mmlu,
    x="decile",
    y="decile_non_choice_mass",
    hue="model",
    hue_order=MODELS,
    palette=COLORS,
    estimator="mean",
    errorbar=None,
    legend=False,
    ax=ax1['F'],
)

ax1["F"].set(xlim=(0, 100), ylim=(0, 0.46))
ax1["F"].set_xlabel("", fontsize=7)
ax1["F"].set_ylabel("", fontsize=7, rotation=0)
ax1["F"].set_xticks(
    ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=["", "", "", "", "", "", "", "", "", "", ""],
    fontsize=7,
)
ax1["F"].set_yticks(
    [0.0,0.1,0.2,0.3,0.4],
    labels=["","","","",""],
    fontsize=7,
)
ax1["F"].set_title("")

sns.despine(ax=ax1["F"],left=False, bottom=False)

sns.lineplot(
    data=df_gpqa_flip,
    x="decile",
    y="decile_flip_rate",
    hue="model",
    hue_order=MODELS,
    palette=COLORS,
    estimator="mean",
    errorbar=None,
    legend=False,
    ax=ax1['G'],
)

ax1["G"].set(xlim=(0, 100), ylim=(0, 0.33))
ax1["G"].set_xlabel("Reasoning deciles", fontsize=7)
ax1["G"].set_ylabel("Flip rate", fontsize=7,)
ax1["G"].set_xticks(
    ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=[r"0$\%$", "", r"20$\%$", "", r"40$\%$", "", r"60$\%$", "", r"80$\%$", "", r"100$\%$"],
    fontsize=7,
)
ax1["G"].set_yticks(
    [0.0,0.1,0.2,0.3],
    labels=[r"0$\%$",r"10$\%$",r"20$\%$",r"30$\%$"],
    fontsize=7,
)
ax1["G"].set_title("")

sns.despine(ax=ax1["G"],left=False, bottom=False)

sns.lineplot(
    data=df_mmlu_flip,
    x="decile",
    y="decile_flip_rate",
    hue="model",
    hue_order=MODELS,
    palette=COLORS,
    estimator="mean",
    errorbar=None,
    legend=False,
    ax=ax1['H'],
)

ax1["H"].set(xlim=(0, 100), ylim=(0, 0.33))
ax1["H"].set_xlabel("Reasoning deciles", fontsize=7)
ax1["H"].set_ylabel("", fontsize=7, rotation=0)
ax1["H"].set_xticks(
    ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=[r"0$\%$", "", r"20$\%$", "", r"40$\%$", "", r"60$\%$", "", r"80$\%$", "", r"100$\%$"],
    fontsize=7,
)
ax1["H"].set_yticks(
    [0.0,0.1,0.2,0.3],
    labels=["","","",""],
    fontsize=7,
)
ax1["H"].set_title("")

sns.despine(ax=ax1["H"],left=False, bottom=False)

panel_letters = {
    "A": ("a", -0.14),
    "B": ("b", -0.06),
    "C": ("c", -0.14),
    "D": ("d", -0.06),
    "E": ("e", -0.14),
    "F": ("f", -0.06),
    "G": ("g", -0.14),
    "H": ("h", -0.06),
}
for pid, (letter, x_offset) in panel_letters.items():
    ax1[pid].text(
        x_offset,
        1.05,
        f"\\textbf{{{letter}}}",
        transform=ax1[pid].transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        clip_on=False,
    )

legend_elements = [
    mlines.Line2D([], [], color=COLORS[i], label=format_model_label(MODELS[i]))
    for i in range(len(MODELS))
]

ax1['E'].legend(
    handles=legend_elements,
    fontsize=7,
    loc='upper left',
    bbox_to_anchor=(0.52, 1.0),
    borderaxespad=0,
    frameon=False,
)

Figure1.savefig("../../results/figures/main_figure_2.png", dpi=300, bbox_inches='tight')
