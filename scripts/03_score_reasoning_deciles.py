from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from vllm import SamplingParams

from prompts import (
    SYSTEM_PROMPT,
    EARLY_STOPPING_PROMPT_QWEN,
    EARLY_STOPPING_PROMPT_GPT_OSS,
    BASELINE_STOP_PROMPT_QWEN,
)
from utils import (
    ALL_DECILES,
    BASELINE_DECILE,
    MODEL_CONFIGS,
    DATASETS,
    GPT_OSS_ANALYSIS_START,
    QWEN_THINK_START,
    build_choice_tokens,
    build_llm,
    ensure_isolated_inductor_cache,
    compute_choice_logprobs,
)


NON_CHOICE_TOP_K = 10


@dataclass(frozen=True)
class DecileRun:
    """Parsed metadata for one decile parquet file."""

    dataset: str
    model_key: str
    model_family: str
    model_size: str
    run: int
    path: Path


def discover_decile_runs(
    dataset: str,
    allowed_models: set[str],
    input_root: Path,
) -> list[DecileRun]:
    """
    Locate reasoning-decile parquet files and extract model/run metadata.

    Expected filename shape (underscore-separated):
      <dataset>_<family_slug>_<size>_run<run>_reasoning_deciles.parquet
      e.g., gpqa_qwen3_14b_run2_reasoning_deciles.parquet
    Legacy hyphenated names are also accepted for backwards compatibility.
    """
    patterns = [
        # Current underscore-separated naming.
        re.compile(
            rf"^{dataset}_(?P<family_slug>.+?)_(?P<size>[^_]+)"
            r"_run(?P<run>\d+)_reasoning_deciles\.parquet$"
        ),
        # Legacy hyphenated naming.
        re.compile(
            rf"^{dataset}_(?P<family_slug>[\w-]+)-(?P<size>[^_]+)"
            r"_run(?P<run>\d+)_reasoning_deciles\.parquet$"
        ),
    ]

    dataset_dir = input_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {dataset_dir}")

    discovered: list[DecileRun] = []
    for path in dataset_dir.glob(f"{dataset}_*_reasoning_deciles.parquet"):
        match = None
        for pattern in patterns:
            match = pattern.match(path.name)
            if match:
                break
        if not match:
            continue

        family_slug = match.group("family_slug")
        model_family = family_slug.replace("-", "_")
        model_size = match.group("size")
        model_key = f"{model_family}_{model_size}"
        if model_key not in MODEL_CONFIGS:
            continue
        if allowed_models and model_key not in allowed_models:
            continue

        discovered.append(
            DecileRun(
                dataset=dataset,
                model_key=model_key,
                model_family=model_family,
                model_size=model_size,
                run=int(match.group("run")),
                path=path,
            )
        )

    return sorted(
        discovered,
        key=lambda r: (r.dataset, r.model_family, r.model_size, r.run, r.path.name),
    )


def extract_top_non_choice_logprobs(
    first_pos_logprobs,
    choice_token_ids: list[int],
    tokenizer,
    top_k: int = NON_CHOICE_TOP_K,
) -> list[dict[str, float | int | str]]:
    """Return ranked list of top_k non-choice tokens with details."""
    if not first_pos_logprobs:
        return []

    choice_ids = set(choice_token_ids)
    candidates: list[tuple[int, float]] = []
    for tid, logprob in first_pos_logprobs.items():
        if tid in choice_ids:
            continue
        candidates.append((tid, float(logprob.logprob)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    top_candidates = candidates[:top_k]

    result: list[dict[str, float | int | str]] = []
    for rank, (tid, logprob) in enumerate(top_candidates, start=1):
        token_str = tokenizer.decode([tid], skip_special_tokens=False)
        result.append(
            {
                "rank": rank,
                "token_id": tid,
                "token_repr": repr(token_str),
                "logprob": logprob,
            }
        )
    return result


def score_one_run(
    run: DecileRun,
    deciles: list[int],
    sampling_kwargs: dict,
    choice_cache: dict[tuple[str, str], tuple[list[int], dict[int, str]]],
    output_root: Path,
    llm,
    tokenizer,
) -> None:
    df = pd.read_parquet(run.path)
    prompt_column = DATASETS[run.dataset]["prompt_column"]

    if df.empty:
        raise ValueError(f"{run.path} is empty.")
    if prompt_column not in df.columns:
        raise KeyError(f"{run.path} missing expected prompt column '{prompt_column}'.")
    if "id" not in df.columns:
        raise KeyError(f"{run.path} missing required column 'id'.")

    # Baseline (decile 0) doesn't require a column - it uses empty reasoning.
    # Other deciles require the corresponding column to exist.
    available_deciles = [
        decile for decile in deciles
        if decile == BASELINE_DECILE or f"thoughts_decile_{decile:02d}" in df.columns
    ]
    if not available_deciles:
        print(f"Skipping {run.path}: none of the requested deciles are present.")
        return

    if "run" in df.columns:
        runs = pd.to_numeric(df["run"], errors="coerce").dropna().unique()
        if len(runs) == 0:
            run_value = run.run
        elif len(runs) == 1:
            run_value = int(runs[0])
        else:
            raise ValueError(f"{run.path}: multiple run ids found: {runs[:10]}")
    else:
        run_value = run.run

    num_rows = len(df)

    ids = df["id"].tolist()
    ids = [x.item() if hasattr(x, "item") else x for x in ids]  # normalize numpy/pandas scalars

    prompts = df[prompt_column].fillna("").astype(str).tolist()

    tokenizer_id = getattr(tokenizer, "name_or_path", None) or run.model_key
    cache_key = (run.dataset, tokenizer_id)
    if cache_key not in choice_cache:
        choice_cache[cache_key] = build_choice_tokens(run.dataset, tokenizer)
    choice_token_ids, id_to_label = choice_cache[cache_key]

    # Ensure logprobs covers all choice tokens, without constraining generation.
    base_logprobs = sampling_kwargs.get("logprobs")
    required_logprobs = len(choice_token_ids) + NON_CHOICE_TOP_K
    if base_logprobs is None:
        effective_logprobs = required_logprobs
    elif base_logprobs == -1:
        effective_logprobs = -1  # request all logprobs
    elif base_logprobs < required_logprobs:
        effective_logprobs = required_logprobs
    else:
        effective_logprobs = base_logprobs
    run_sampling_params = SamplingParams(**{**sampling_kwargs, "logprobs": effective_logprobs})

    messages_batch = []
    for user_prompt in prompts:
        messages_batch.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

    prompt_prefixes = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if len(prompt_prefixes) != num_rows:
        raise RuntimeError(
            f"{run.path}: unexpected prompt count (rows={num_rows}, prompts={len(prompt_prefixes)})."
        )

    prompt_prefix_token_ids = [
        tokenizer(prefix, add_special_tokens=False).input_ids for prefix in prompt_prefixes
    ]
    # Reserve margin so tokenizer-added specials do not overflow the context.
    special_tokens_margin = len(tokenizer("", add_special_tokens=True).input_ids) - len(
        tokenizer("", add_special_tokens=False).input_ids
    )
    special_tokens_margin = max(special_tokens_margin, 0) + 128  # safety buffer
    max_model_len = int(MODEL_CONFIGS[run.model_key]["max_len"])
    max_generation_tokens = int(run_sampling_params.max_tokens or 0)
    max_input_tokens = max_model_len - max_generation_tokens - special_tokens_margin
    if max_input_tokens <= 0:
        raise ValueError(
            f"{run.path}: invalid token budget (max_model_len={max_model_len}, max_tokens={max_generation_tokens}, margin={special_tokens_margin})."
        )

    # Determine thinking start marker for baseline (GPT-OSS and Qwen need explicit markers).
    model_spec = MODEL_CONFIGS[run.model_key]["spec"]
    if model_spec == "gpt-oss":
        baseline_text = GPT_OSS_ANALYSIS_START
    elif model_spec == "qwen":
        baseline_text = QWEN_THINK_START
    else:
        baseline_text = ""

    for decile in available_deciles:
        output_path = (
            output_root
            / run.dataset
            / f"{run.dataset}_{run.model_key}_run{run_value}_decile{decile:02d}_logprobs.jsonl"
        )
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as existing_f:
                existing_rows = sum(1 for _ in existing_f)
            if existing_rows == num_rows:
                print(f"Skipping existing {output_path} ({existing_rows} rows)")
                continue
            print(f"Rewriting {output_path}: found {existing_rows}/{num_rows} rows")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine stop prompt: baseline_0 for Qwen uses a minimal close tag,
        # other deciles use the early stopping prompt with explanatory text.
        if decile == BASELINE_DECILE and model_spec == "qwen":
            stop_prompt = BASELINE_STOP_PROMPT_QWEN
        elif model_spec == "qwen":
            stop_prompt = EARLY_STOPPING_PROMPT_QWEN
        else:
            stop_prompt = EARLY_STOPPING_PROMPT_GPT_OSS
        stop_prompt_token_ids = tokenizer(stop_prompt, add_special_tokens=False).input_ids

        # Baseline (decile 0) uses empty reasoning; other deciles use column data.
        if decile == BASELINE_DECILE:
            decile_texts = [baseline_text] * num_rows
        else:
            decile_col = f"thoughts_decile_{decile:02d}"
            decile_texts = [str(text or "") for text in df[decile_col].fillna("")]
            if len(decile_texts) != num_rows:
                raise RuntimeError(
                    f"{run.path}: decile column {decile_col} has {len(decile_texts)} rows (expected {num_rows})."
                )

        decile_token_ids_batch = [
            tokenizer(decile_text, add_special_tokens=False).input_ids for decile_text in decile_texts
        ]

        truncated_prompts = 0
        input_prompts = []
        for prefix, prefix_token_ids, decile_token_ids, decile_text in zip(
            prompt_prefixes, prompt_prefix_token_ids, decile_token_ids_batch, decile_texts
        ):
            allowed_decile_tokens = max_input_tokens - len(prefix_token_ids) - len(stop_prompt_token_ids)
            if allowed_decile_tokens < 0:
                raise ValueError(
                    f"{run.path}: prompt prefix exceeds model context window ({max_model_len} tokens)."
                )

            if len(decile_token_ids) > allowed_decile_tokens:
                truncated_prompts += 1
                decile_token_ids = decile_token_ids[: max(0, allowed_decile_tokens)]
                decile_text = tokenizer.decode(decile_token_ids, skip_special_tokens=False)

            input_prompts.append(prefix + decile_text + stop_prompt)

        if truncated_prompts:
            print(
                f"{run.path}: decile {decile:02d} truncated {truncated_prompts}/{num_rows} prompts to fit context window ({max_model_len} tokens)."
            )

        with output_path.open("w", encoding="utf-8") as out_f:
            outputs = llm.generate(
                input_prompts,
                sampling_params=run_sampling_params,
                use_tqdm=False,
            )

            if len(outputs) != num_rows:
                raise RuntimeError(
                    f"{run.path}: expected {num_rows} generations, got {len(outputs)}."
                )

            for id_val, output in zip(ids, outputs):
                if not output.outputs:
                    raise RuntimeError(f"{run.path}: no completion returned for id {id_val}.")
                completion = output.outputs[0]
                first_pos_logprobs = (
                    completion.logprobs[0] if completion.logprobs else None
                )
                choice_logprobs = compute_choice_logprobs(
                    first_pos_logprobs,
                    id_to_label,
                )
                top_non_choice_logprobs = extract_top_non_choice_logprobs(
                    first_pos_logprobs,
                    choice_token_ids,
                    tokenizer,
                    top_k=NON_CHOICE_TOP_K,
                )

                record = {
                    "dataset": run.dataset,
                    "model": run.model_key,
                    "run": run_value,
                    "decile": decile,
                    "id": id_val,
                    "choice_logprobs": choice_logprobs,
                    "top_non_choice_logprobs": top_non_choice_logprobs,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score injected reasoning deciles and store logprobs."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASETS.keys(),
        default=list(DATASETS.keys()),
        help="Datasets to process (default: all).",
    )
    parser.add_argument(
        "--model-name",
        nargs="+",
        choices=MODEL_CONFIGS.keys(),
        default=list(MODEL_CONFIGS.keys()),
        help="Model keys to evaluate (default: all).",
    )
    parser.add_argument(
        "--decile",
        nargs="+",
        type=int,
        default=list(ALL_DECILES),
        help="Deciles to score (default: 0 for baseline plus 10-100 for reasoning deciles).",
    )
    parser.add_argument(
        "--input-dir",
        default="data/deciles",
        help="Root directory that contains decile parquet files (per dataset).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/logprobs",
        help="Directory where JSONL logprob files will be written.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=1000,
        help="Number of logprobs to request per generated token (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed for reproducibility.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager mode to bypass torch.compile/TorchInductor (avoids shared cache corruption at the cost of speed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    deciles = sorted(set(args.decile))

    sampling_kwargs = dict(
        max_tokens=1,
        seed=args.seed,
        logprobs=-1 if args.logprobs <= 0 else args.logprobs,
        flat_logprobs=True,
    )

    allowed_models = set(args.model_name)
    runs: list[DecileRun] = []
    for dataset in args.dataset:
        runs.extend(discover_decile_runs(dataset, allowed_models, input_root))

    if not runs:
        raise FileNotFoundError(
            f"No decile parquet files found under {input_root} for datasets {sorted(args.dataset)}"
        )

    runs_by_model: dict[str, list[DecileRun]] = defaultdict(list)
    for run in runs:
        runs_by_model[run.model_key].append(run)

    choice_cache: dict[tuple[str, str], tuple[list[int], dict[int, str]]] = {}

    for model_key in sorted(runs_by_model):
        model_runs = sorted(
            runs_by_model[model_key],
            key=lambda r: (r.dataset, r.run, r.path.name),
        )
        cache_dir = ensure_isolated_inductor_cache(model_key)
        if cache_dir:
            print(f"Using TORCHINDUCTOR_CACHE_DIR={cache_dir}")

        llm = build_llm(model_key, enforce_eager=args.enforce_eager)
        tokenizer = llm.get_tokenizer()

        for run in model_runs:
            score_one_run(
                run,
                deciles=deciles,
                sampling_kwargs=sampling_kwargs,
                choice_cache=choice_cache,
                output_root=output_root,
                llm=llm,
                tokenizer=tokenizer,
            )

        del llm
        gc.collect()


if __name__ == "__main__":
    main()
