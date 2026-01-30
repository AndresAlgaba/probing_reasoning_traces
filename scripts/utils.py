import functools
import math
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from transformers import AutoTokenizer

if TYPE_CHECKING:
    from vllm import LLM


RNG = np.random.default_rng(0)
BASELINE_DECILE = 0
DECILES = tuple(range(10, 101, 10))
ALL_DECILES = (BASELINE_DECILE,) + DECILES
GPT_OSS_ANALYSIS_START = "<|channel|>analysis<|message|>"
QWEN_THINK_START = "<think>\n"
MODEL_CONFIGS: dict[str, dict[str, str | int]] = {
    "qwen3_4b": {"model_id": "Qwen/Qwen3-4B", "spec": "qwen", "max_len": 32768},
    "qwen3_8b": {"model_id": "Qwen/Qwen3-8B", "spec": "qwen", "max_len": 32768},
    "qwen3_14b": {"model_id": "Qwen/Qwen3-14B", "spec": "qwen", "max_len": 32768},
    "gpt_oss_20b": {"model_id": "openai/gpt-oss-20b", "spec": "gpt-oss", "max_len": 131072},
    "gpt_oss_120b": {"model_id": "openai/gpt-oss-120b", "spec": "gpt-oss", "max_len": 131072},
}
DATASETS: dict[str, dict[str, str]] = {
    "gpqa": {"prompt_column": "question"},
    "mmlu": {"prompt_column": "prompt"},
}
DATASET_CHOICES: dict[str, tuple[str, ...]] = {
    "gpqa": ("A", "B", "C", "D"),
    "mmlu": tuple(chr(ord("A") + i) for i in range(10)),
}


def dataset_choice_labels(dataset: str) -> tuple[str, ...]:
    try:
        return DATASET_CHOICES[dataset]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset '{dataset}'.") from exc


@functools.lru_cache(maxsize=None)
def get_tokenizer(model_key: str) -> AutoTokenizer:
    """Load and cache a tokenizer for the given model key."""
    if model_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model key '{model_key}'.")
    return AutoTokenizer.from_pretrained(MODEL_CONFIGS[model_key]["model_id"])


def build_llm(model_key: str, *, enforce_eager: bool = False) -> "LLM":
    from vllm import LLM

    max_logprobs = 1000
    cfg = MODEL_CONFIGS[model_key]
    if cfg["spec"] == "qwen":
        return LLM(
            model=cfg["model_id"],
            max_model_len=cfg["max_len"],
            max_num_batched_tokens=cfg["max_len"],
            gpu_memory_utilization=0.95,
            enforce_eager=enforce_eager,
            max_logprobs=max_logprobs,
        )
    return LLM(
        model=cfg["model_id"],
        max_model_len=cfg["max_len"],
        max_num_seqs=8,
        max_num_batched_tokens=cfg["max_len"],
        gpu_memory_utilization=0.95,
        enforce_eager=enforce_eager,
        max_logprobs=max_logprobs,
    )


def clean_and_split_solution(df: pd.DataFrame) -> pd.DataFrame:
    if "solution" in df.columns:
        column = "solution"
    elif "response" in df.columns:
        column = "response"
    else:
        raise KeyError("Expected a 'solution' or 'response' column in the dataframe.")

    split = df[column].str.split("\n</think>\n\n", n=1, expand=True)

    split = split.reindex(columns=[0, 1], fill_value="")
    df["thoughts"] = split[0].fillna("")
    df["reply"] = split[1].fillna("")

    return df


def clean_and_split_gpt_oss_solution(df: pd.DataFrame) -> pd.DataFrame:
    if "solution" in df.columns:
        column = "solution"
    elif "response" in df.columns:
        column = "response"
    else:
        raise KeyError("Expected a 'solution' or 'response' column in the dataframe.")

    def _split(text: str) -> tuple[str, str]:
        if not isinstance(text, str):
            return "", ""

        raw = text.strip()
        analysis_section, reply_section = raw, ""

        if "assistantfinal" in raw:
            analysis_section, _, reply_section = raw.partition("assistantfinal")

        analysis_section = analysis_section.strip()
        if analysis_section.startswith("analysis"):
            analysis_section = analysis_section[len("analysis") :].lstrip()

        reply_section = reply_section.strip()

        analysis_section = (
            f"{GPT_OSS_ANALYSIS_START}{analysis_section}" if analysis_section else GPT_OSS_ANALYSIS_START
        )

        return analysis_section, reply_section

    split = df[column].apply(lambda s: pd.Series(_split(s), index=["thoughts", "reply"]))
    return pd.concat([df, split], axis=1)


def build_deciles(
    text: str,
    tokenizer: AutoTokenizer,
    *,
    skip_special_tokens: bool = True,
) -> dict:
    text = text or ""
    tokens = tokenizer.encode(text, add_special_tokens=False)

    deciles = {}
    if not tokens:
        for pct in DECILES:
            deciles[f"thoughts_decile_{pct:02d}"] = ""
        return deciles

    total_tokens = len(tokens)
    for pct in DECILES:
        cutoff = max(1, math.ceil(total_tokens * pct / 100))
        deciles[f"thoughts_decile_{pct:02d}"] = tokenizer.decode(
            tokens[:cutoff],
            skip_special_tokens=skip_special_tokens,
        )
    return deciles


def build_deciles_with_special_tokens(text: str, tokenizer: AutoTokenizer) -> dict:
    return build_deciles(text, tokenizer, skip_special_tokens=False)


def build_choice_tokens(dataset: str, tokenizer) -> tuple[list[int], dict[int, str]]:
    labels = ["A", "B", "C", "D"] if dataset == "gpqa" else [chr(ord("A") + i) for i in range(10)]

    label_to_token_ids: dict[str, set[int]] = {label: set() for label in labels}
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if not ids:
            continue
        token_id = ids[-1]
        label_to_token_ids[label].add(token_id)

    token_ids: list[int] = []
    id_to_label: dict[int, str] = {}
    seen: set[int] = set()
    for label in labels:
        for tid in sorted(label_to_token_ids[label]):
            if tid in seen:
                continue
            seen.add(tid)
            token_ids.append(tid)
            id_to_label[tid] = label

    return token_ids, id_to_label


def compute_choice_logprobs(
    first_pos_logprobs,
    id_to_label: dict[int, str],
) -> dict[str, float]:
    if not id_to_label:
        return {}

    labels = set(id_to_label.values())
    label_logprobs: dict[str, list[float]] = {label: [] for label in labels}

    if not first_pos_logprobs:
        return {label: math.nan for label in labels}

    for tid, label in id_to_label.items():
        if tid not in first_pos_logprobs:
            continue
        label_logprobs[label].append(float(first_pos_logprobs[tid].logprob))

    choice_logprobs: dict[str, float] = {}
    for label, logprobs in label_logprobs.items():
        if not logprobs:
            choice_logprobs[label] = math.nan
        elif len(logprobs) == 1:
            choice_logprobs[label] = logprobs[0]
        else:
            choice_logprobs[label] = float(np.logaddexp.reduce(logprobs))

    return choice_logprobs


def argmax_choice(choice_logprobs: dict[str, float]) -> str | None:
    best_label: str | None = None
    best_value: float | None = None
    for label, value in choice_logprobs.items():
        if value is None or math.isnan(value):
            continue
        if best_value is None or value > best_value:
            best_label = label
            best_value = value
    return best_label


def compute_decile_flips(df: pd.DataFrame, group_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sort_keys = group_keys + ["id", "decile"]
    df_sorted = df.sort_values(sort_keys).copy()

    df_sorted["decile_prediction_prev"] = (
        df_sorted.groupby(group_keys + ["id"])["decile_prediction"].shift()
    )
    df_sorted["decile_prediction_flip"] = (
        df_sorted["decile_prediction_prev"].notna()
        & (df_sorted["decile_prediction"] != df_sorted["decile_prediction_prev"])
    )

    flip_rates = (
        df_sorted.groupby(group_keys + ["decile"])["decile_prediction_flip"]
        .mean()
        .reset_index()
        .rename(columns={"decile_prediction_flip": "decile_flip_rate"})
    )

    return df_sorted, flip_rates


def ensure_isolated_inductor_cache(model_key: str, force: bool = False) -> Path | None:
    existing_env = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if existing_env and not force:
        existing_path = Path(existing_env)
        if model_key in existing_path.parts:
            existing_path.mkdir(parents=True, exist_ok=True)
            return existing_path
        base_root = existing_path
    else:
        base_root = Path(".cache/torchinductor")

    cache_dir = base_root / model_key / f"run_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    return cache_dir
