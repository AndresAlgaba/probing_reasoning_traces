from pathlib import Path
import numpy as np
import pandas as pd


RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "processed_outputs.parquet"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"
DATASETS = ("gpqa", "mmlu")
DECILE = 100


def safe_mean(series: pd.Series, mask: pd.Series) -> float:
    n = int(mask.sum())
    if n == 0:
        return np.nan
    return series[mask].mean()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset, model), g in df.groupby(["dataset", "model"]):
        resp = g["response"].fillna("")
        boxed = resp.str.contains(r"\\boxed", regex=True)
        p_a = (g["decile_prediction"] == "A").astype(float)
        acc = g["decile_accuracy"].astype(float)

        records.append(
            {
                "dataset": dataset,
                "model": model,
                "n": len(g),
                "boxed_frac": boxed.mean() if len(g) else np.nan,
                "p_A_boxed": safe_mean(p_a, boxed),
                "p_A_non": safe_mean(p_a, ~boxed),
                "acc_boxed": safe_mean(acc, boxed),
                "acc_non": safe_mean(acc, ~boxed),
                "boxed_n": int(boxed.sum()),
                "non_n": int((~boxed).sum()),
            }
        )

    return pd.DataFrame(records)


def format_percents(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["boxed_frac", "p_A_boxed", "p_A_non", "acc_boxed", "acc_non"]:
        out[col] = (out[col] * 100).round(1)
    return out


def pretty_print(df: pd.DataFrame) -> None:
    df_fmt = format_percents(df)
    for dataset, sub in df_fmt.sort_values(["dataset", "model"]).groupby("dataset"):
        print(f"\n{dataset.upper()}")
        for row in sub.itertuples(index=False):
            def fmt(x):
                return f"{x:.1f}%" if pd.notna(x) else "—"
            print(
                f"{row.model:12s} | "
                f"{fmt(row.boxed_frac):>6s} | "
                f"{fmt(row.p_A_boxed):>6s} | "
                f"{fmt(row.p_A_non):>6s} | "
                f"{fmt(row.acc_boxed):>6s} | "
                f"{fmt(row.acc_non):>6s} | "
                f"boxed_n={row.boxed_n:5d} | non_n={row.non_n:5d}"
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = [
        "dataset",
        "model",
        "decile",
        "decile_prediction",
        "decile_accuracy",
        "response",
    ]
    filters = [("dataset", "in", list(DATASETS)), ("decile", "==", DECILE)]

    df = pd.read_parquet(RESULTS_PATH, columns=cols, filters=filters)
    summary = summarize(df)

    summary.sort_values(["dataset", "model"]).to_csv(
        OUTPUT_DIR / "boxed_collapse_table.csv", index=False
    )
    format_percents(summary).sort_values(["dataset", "model"]).to_csv(
        OUTPUT_DIR / "boxed_collapse_table_formatted.csv", index=False
    )

    pretty_print(summary)


if __name__ == "__main__":
    main()
