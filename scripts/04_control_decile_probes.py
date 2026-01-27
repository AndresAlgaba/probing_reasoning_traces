from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from vllm import SamplingParams

from prompts import (
    SYSTEM_PROMPT,
    EARLY_STOPPING_PROMPT_QWEN,
    EARLY_STOPPING_PROMPT_GPT_OSS,
)
from utils import (
    DECILES,
    MODEL_CONFIGS,
    DATASETS,
    GPT_OSS_ANALYSIS_START,
    QWEN_THINK_START,
    build_choice_tokens,
    build_llm,
    compute_choice_logprobs,
    ensure_isolated_inductor_cache,
)

CONTROL_SEED_OFFSETS = {"junk": 17, "cross": 31, "shuffle": 47}


@dataclass(frozen=True)
class DecileRun:
    dataset: str
    model_key: str
    run: int
    path: Path


def discover_decile_runs(
    dataset: str,
    allowed_models: set[str],
    input_root: Path,
) -> list[DecileRun]:
    """Locate decile parquet files for the dataset/models."""
    patterns = [
        # Underscore naming: mmlu_qwen3_14b_run2_reasoning_deciles.parquet
        re.compile(
            rf"^{dataset}_(?P<family_slug>.+?)_(?P<size>[^_]+)"
            r"_run(?P<run>\d+)_reasoning_deciles\.parquet$"
        ),
        # Legacy hyphenated.
        re.compile(
            rf"^{dataset}_(?P<family_slug>[\w-]+)-(?P<size>[^_]+)"
            r"_run(?P<run>\d+)_reasoning_deciles\.parquet$"
        ),
    ]

    dataset_dir = input_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {dataset_dir}")

    runs: list[DecileRun] = []
    for path in dataset_dir.glob(f"{dataset}_*_reasoning_deciles.parquet"):
        match = None
        for pattern in patterns:
            match = pattern.match(path.name)
            if match:
                break
        if not match:
            continue

        family_slug = match.group("family_slug").replace("-", "_")
        size = match.group("size")
        model_key = f"{family_slug}_{size}"
        if model_key not in MODEL_CONFIGS:
            continue
        if allowed_models and model_key not in allowed_models:
            continue

        runs.append(
            DecileRun(
                dataset=dataset,
                model_key=model_key,
                run=int(match.group("run")),
                path=path,
            )
        )

    return sorted(runs, key=lambda r: (r.dataset, r.model_key, r.run, r.path.name))


def build_junk_tokens(
    tokenizer, length: int, rng: np.random.Generator
) -> tuple[str, int]:
    """Generate junk text targeting a token count; returns (text, actual_token_count)."""
    if length <= 0:
        return "", 0
    # Cache whitelist per tokenizer to avoid cross-model contamination.
    cache: dict[str, list[int]]
    if not hasattr(build_junk_tokens, "_cache") or not isinstance(
        getattr(build_junk_tokens, "_cache"), dict
    ):
        build_junk_tokens._cache = {}  # type: ignore[attr-defined]
    cache = build_junk_tokens._cache  # type: ignore[attr-defined]

    tokenizer_key = getattr(tokenizer, "name_or_path", None)
    if tokenizer_key is None:
        tokenizer_key = id(tokenizer)
    tokenizer_key = str(tokenizer_key)

    if tokenizer_key not in cache:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            ids = list(range(len(tokenizer)))
        else:
            ids = list(range(vocab_size))
        special = set(getattr(tokenizer, "all_special_ids", []) or [])
        # Also filter out negative/None ids.
        allowed = [tid for tid in ids if tid is not None and tid >= 0 and tid not in special]
        cache[tokenizer_key] = allowed

    allowed_ids: list[int] = cache[tokenizer_key]
    if not allowed_ids:
        return "", 0
    # Strip any decoded substrings that could be re-tokenized as special tokens.
    special_markers = set(getattr(tokenizer, "all_special_tokens", []) or [])
    special_markers.update({"<|", "|>", "<think>", "</think>", "<|assistant|>", "<|user|>", "<|system|>"})
    special_markers.update(re.findall(r"<\|[^|]+?\|>", EARLY_STOPPING_PROMPT_GPT_OSS))
    special_markers.update(re.findall(r"<\|[^|]+?\|>", GPT_OSS_ANALYSIS_START))
    special_markers.update({EARLY_STOPPING_PROMPT_GPT_OSS, GPT_OSS_ANALYSIS_START})
    special_markers = [m for m in sorted(special_markers, key=len, reverse=True) if m]

    def strip_markers(text: str) -> str:
        cleaned = text
        for marker in special_markers:
            cleaned = cleaned.replace(marker, "")
        return cleaned

    junk_chunks: list[str] = []
    max_rounds = 6
    for _ in range(max_rounds):
        current = "".join(junk_chunks)
        current_len = len(tokenizer.encode(current, add_special_tokens=False))
        remaining = length - current_len
        if remaining <= 0:
            break
        sample_ids = rng.choice(allowed_ids, size=remaining, replace=True)
        chunk = tokenizer.decode(sample_ids, skip_special_tokens=True)
        chunk = strip_markers(chunk)
        if chunk:
            junk_chunks.append(chunk)

    junk = "".join(junk_chunks)
    encoded = tokenizer.encode(junk, add_special_tokens=False)
    if len(encoded) > length:
        encoded = encoded[:length]
        junk = tokenizer.decode(encoded, skip_special_tokens=True)
        encoded = tokenizer.encode(junk, add_special_tokens=False)

    actual_token_count = len(encoded)
    return junk, actual_token_count


def trim_tokens(tokens: list[int], target_length: int) -> list[int]:
    """Return tokens truncated to target_length (no padding)."""
    if target_length <= 0:
        return []
    return tokens[:target_length]


def shuffle_tokens(tokens: list[int], rng: np.random.Generator) -> list[int]:
    """Shuffle token order while preserving exact tokens and length."""
    if not tokens:
        return []
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    return shuffled


def write_lines_atomic(out_path: Path, lines: Iterable[str]) -> None:
    """Write JSONL safely by using a temp file then renaming atomically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out_f:
        for line in lines:
            out_f.write(line)
            if not line.endswith("\n"):
                out_f.write("\n")
    tmp_path.replace(out_path)


def resolve_logprobs(requested: int, choice_token_count: int) -> int:
    """Ensure logprobs covers all choice tokens unless user requests all."""
    if requested <= 0:
        return -1  # request all logprobs
    if requested < choice_token_count:
        return choice_token_count
    return requested


def ensure_thinking_prefix(
    prefix: str,
    payload: str,
    thinking_start: str,
) -> tuple[str, bool]:
    """
    Ensure the prompt contains a thinking-start marker before the payload.

    Returns the combined prompt (without the stop prompt) and a boolean flag
    indicating whether the marker had to be injected.
    """
    marker = thinking_start or ""
    payload = payload or ""
    if not marker:
        return prefix + payload, False

    marker_stripped = marker.strip()
    prefix_trimmed = prefix.rstrip()
    prefix_has = prefix_trimmed.endswith(marker) or (
        marker_stripped and prefix_trimmed.endswith(marker_stripped)
    )

    payload_clean = payload.lstrip()
    payload_has = payload_clean.startswith(marker) or (
        marker_stripped and payload_clean.startswith(marker_stripped)
    )

    inserted = not (prefix_has or payload_has)
    combined = prefix + (marker if inserted else "") + payload
    return combined, inserted


def score_controls_for_run(
    run: DecileRun,
    deciles: list[int],
    controls: set[str],
    choice_cache: dict[tuple[str, str], tuple[list[int], dict[int, str]]],
    output_root: Path,
    llm,
    tokenizer,
    seed: int,
    logprobs_request: int,
    skip_existing: bool,
    debug_prompts: bool,
    debug_prompts_limit: int,
) -> None:
    df = pd.read_parquet(run.path)
    if df.empty:
        raise ValueError(f"{run.path} is empty.")

    prompt_column = DATASETS[run.dataset]["prompt_column"]
    required_cols = {"id", prompt_column}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"{run.path} missing required columns: {sorted(missing_cols)}")

    df["id"] = df["id"].astype(str)
    available_deciles = [d for d in deciles if f"thoughts_decile_{d:02d}" in df.columns]
    if not available_deciles:
        print(f"Skipping {run.path}: none of the requested deciles present.")
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

    tokenizer_id = getattr(tokenizer, "name_or_path", None) or tokenizer.__class__.__name__
    cache_key = (run.dataset, tokenizer_id)
    if cache_key not in choice_cache:
        choice_cache[cache_key] = build_choice_tokens(run.dataset, tokenizer)
    choice_token_ids, id_to_label = choice_cache[cache_key]
    effective_logprobs = resolve_logprobs(logprobs_request, len(choice_token_ids))

    spec = MODEL_CONFIGS[run.model_key]["spec"]
    stop_prompt = EARLY_STOPPING_PROMPT_QWEN if spec == "qwen" else EARLY_STOPPING_PROMPT_GPT_OSS
    thinking_start = QWEN_THINK_START if spec == "qwen" else GPT_OSS_ANALYSIS_START

    prompts = df[prompt_column].fillna("").astype(str).tolist()
    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        for user_prompt in prompts
    ]
    prompt_prefixes = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if len(prompt_prefixes) != len(prompts):
        raise RuntimeError(
            f"{run.path}: unexpected prompt count (rows={len(prompts)}, prompts={len(prompt_prefixes)})."
        )
    ids = df["id"].tolist()
    num_rows = len(ids)

    sampling_params = SamplingParams(
        max_tokens=1,
        logprobs=effective_logprobs,
        flat_logprobs=True,
        seed=seed,
    )

    # Calculate token budget for context window checks.
    prompt_prefix_token_ids = [
        tokenizer(prefix, add_special_tokens=False).input_ids for prefix in prompt_prefixes
    ]
    stop_prompt_token_ids = tokenizer(stop_prompt, add_special_tokens=False).input_ids
    thinking_start_token_ids = tokenizer(thinking_start, add_special_tokens=False).input_ids
    # Reserve margin so tokenizer-added specials do not overflow the context.
    special_tokens_margin = len(tokenizer("", add_special_tokens=True).input_ids) - len(
        tokenizer("", add_special_tokens=False).input_ids
    )
    special_tokens_margin = max(special_tokens_margin, 0) + 128  # safety buffer
    max_model_len = int(MODEL_CONFIGS[run.model_key]["max_len"])
    max_generation_tokens = int(sampling_params.max_tokens or 0)
    max_input_tokens = max_model_len - max_generation_tokens - special_tokens_margin
    if max_input_tokens <= 0:
        raise ValueError(
            f"{run.path}: invalid token budget (max_model_len={max_model_len}, "
            f"max_tokens={max_generation_tokens}, margin={special_tokens_margin})."
        )

    controls_for_deciles = set(controls)

    # Prepare cross-example derangements per decile if needed.
    if "cross" in controls_for_deciles:
        duplicate_ids = df.loc[df["id"].duplicated(), "id"].unique().tolist()
        if duplicate_ids:
            examples = duplicate_ids[:5]
            raise ValueError(
                f"{run.path}: duplicate ids found (examples: {examples}, total={len(duplicate_ids)})."
            )

    for decile in available_deciles:
        control_targets: list[tuple[str, Path]] = []
        for control in controls_for_deciles:
            out_path = (
                output_root
                / run.dataset
                / f"{run.dataset}_{run.model_key}_run{run_value}_control_{control}_decile{decile:02d}_logprobs.jsonl"
            )
            if skip_existing and out_path.exists():
                print(f"Skipping existing {out_path}")
                continue
            control_targets.append((control, out_path))

        if not control_targets:
            continue

        decile_texts: list[str] | None = None
        decile_token_ids: list[list[int]] | None = None
        token_lengths: list[int] | None = None
        needs_texts = any(ctrl in {"junk", "cross", "shuffle"} for ctrl, _ in control_targets)
        if needs_texts:
            decile_col = f"thoughts_decile_{decile:02d}"
            decile_texts = df[decile_col].fillna("").astype(str).tolist()
            decile_token_ids = [
                tokenizer.encode(txt, add_special_tokens=False) for txt in decile_texts
            ]
            token_lengths = [len(token_ids) for token_ids in decile_token_ids]
        # Cross control: inject reasoning from a different example, truncated to match length.
        # For each example i, we sample source j uniformly from examples whose decile slice
        # has >= tokens than i's slice, then truncate j's slice to exactly match i's length.
        # This ensures: (1) coherent reasoning text (not junk), (2) for a different question,
        # (3) with identical token count to control for length effects.
        cross_source_row_indices: list[int] | None = None
        cross_source_ids: list[str] | None = None
        if any(ctrl == "cross" for ctrl, _ in control_targets):
            if decile_token_ids is None or token_lengths is None:
                raise RuntimeError("Tokenized decile texts unavailable for cross control.")
            cross_rng = np.random.default_rng(
                seed + run_value + decile + CONTROL_SEED_OFFSETS["cross"]
            )
            cross_source_row_indices = []
            # Sort examples by token length to efficiently find candidates with length >= target.
            token_length_array = np.fromiter(token_lengths, dtype=np.int32, count=num_rows)
            order = np.argsort(token_length_array, kind="stable")
            sorted_lengths = token_length_array[order]
            order_positions = np.empty(num_rows, dtype=np.int32)
            order_positions[order] = np.arange(num_rows, dtype=np.int32)

            for idx, target_length in enumerate(token_length_array):
                # Find position where examples with length >= target_length begin.
                pos = int(np.searchsorted(sorted_lengths, target_length, side="left"))
                suffix_size = num_rows - pos
                idx_order_pos = int(order_positions[idx])
                if suffix_size > 1 and pos <= idx_order_pos < num_rows:
                    # Sample uniformly from examples with length >= target, excluding self.
                    draw = int(cross_rng.integers(0, suffix_size - 1))
                    if draw >= idx_order_pos - pos:
                        draw += 1
                    source_idx = int(order[pos + draw])
                elif num_rows > 1:
                    # Fallback when no other example has sufficient length: sample any other row.
                    # The source will be truncated (possibly to fewer tokens than target).
                    candidate = int(cross_rng.integers(0, num_rows - 1))
                    if candidate >= idx:
                        candidate += 1
                    source_idx = candidate
                else:
                    source_idx = idx
                cross_source_row_indices.append(source_idx)
            cross_source_ids = [ids[idx] for idx in cross_source_row_indices]

        for control, out_path in control_targets:
            rng = None
            if control in {"junk", "shuffle"}:
                rng = np.random.default_rng(
                    seed + decile + run_value + CONTROL_SEED_OFFSETS.get(control, 0)
                )

            prompts_with_meta: list[tuple[str, dict]] = []
            thinking_insertions: list[bool] = []
            truncated_prompts = 0
            for idx, prefix in enumerate(prompt_prefixes):
                # Calculate allowed payload tokens for this prompt (context window check).
                prefix_token_count = len(prompt_prefix_token_ids[idx])
                # Account for thinking_start if it will be inserted by ensure_thinking_prefix.
                thinking_overhead = len(thinking_start_token_ids) if thinking_start else 0
                allowed_payload_tokens = (
                    max_input_tokens
                    - prefix_token_count
                    - len(stop_prompt_token_ids)
                    - thinking_overhead
                )
                # Clamp to zero if prefix alone exceeds budget (inject empty payload).
                allowed_payload_tokens = max(0, allowed_payload_tokens)

                if control == "junk":
                    if token_lengths is None:
                        raise RuntimeError("Token lengths unavailable for junk control.")
                    if rng is None:
                        raise RuntimeError("RNG unavailable for junk control.")
                    target_length = token_lengths[idx]
                    # Truncate target length if it exceeds context window budget.
                    if target_length > allowed_payload_tokens:
                        truncated_prompts += 1
                        target_length = allowed_payload_tokens
                    injected, actual_token_count = build_junk_tokens(
                        tokenizer, target_length, rng
                    )
                    meta = {
                        "junk_target_token_count": token_lengths[idx],
                        "junk_actual_token_count": actual_token_count,
                        "junk_context_truncated": target_length < token_lengths[idx],
                    }
                elif control == "cross":
                    # Inject reasoning from a different example, truncated to match token length.
                    if (
                        cross_source_row_indices is None
                        or cross_source_ids is None
                        or decile_token_ids is None
                    ):
                        raise RuntimeError("Cross mapping missing despite cross control being requested.")
                    source_row_index = cross_source_row_indices[idx]
                    target_length = len(decile_token_ids[idx])
                    source_tokens = decile_token_ids[source_row_index]
                    # Truncate source to exactly match target's token count for length control.
                    injected_tokens = trim_tokens(source_tokens, target_length=target_length)
                    # Further truncate if exceeds context window budget.
                    if len(injected_tokens) > allowed_payload_tokens:
                        truncated_prompts += 1
                        injected_tokens = injected_tokens[:allowed_payload_tokens]
                    injected = tokenizer.decode(
                        injected_tokens,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    meta = {
                        "cross_source_id": cross_source_ids[idx],
                        "cross_source_row_index": source_row_index,
                        "cross_source_token_count": len(source_tokens),
                        "cross_target_token_count": target_length,
                        "cross_injected_token_count": len(injected_tokens),
                        "cross_context_truncated": len(injected_tokens) < target_length,
                    }
                elif control == "shuffle":
                    # Shuffle the tokens from the same example's reasoning.
                    if decile_texts is None:
                        raise RuntimeError("Decile texts unavailable for shuffle control.")
                    if rng is None:
                        raise RuntimeError("RNG unavailable for shuffle control.")
                    # Strip the thinking start marker before tokenizing to avoid shuffling structural tokens.
                    raw_text = decile_texts[idx]
                    if thinking_start and raw_text.startswith(thinking_start):
                        raw_text = raw_text[len(thinking_start):]
                    content_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
                    original_length = len(content_tokens)
                    # Shuffle the tokens.
                    shuffled_tokens = shuffle_tokens(content_tokens, rng)
                    # Truncate if exceeds context window budget.
                    if len(shuffled_tokens) > allowed_payload_tokens:
                        truncated_prompts += 1
                        shuffled_tokens = shuffled_tokens[:allowed_payload_tokens]
                    injected = tokenizer.decode(
                        shuffled_tokens,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    meta = {
                        "shuffle_original_token_count": original_length,
                        "shuffle_injected_token_count": len(shuffled_tokens),
                        "shuffle_context_truncated": len(shuffled_tokens) < original_length,
                    }
                else:
                    raise ValueError(f"Unknown control type: {control}")

                combined, inserted = ensure_thinking_prefix(prefix, injected, thinking_start)
                prompts_with_meta.append((combined + stop_prompt, meta))
                thinking_insertions.append(inserted)

            if truncated_prompts:
                print(
                    f"{run.path}: control={control} decile={decile:02d} truncated "
                    f"{truncated_prompts}/{num_rows} prompts to fit context window ({max_model_len} tokens)."
                )

            prompts_batch = [p[0] for p in prompts_with_meta]

            if debug_prompts:
                limit = debug_prompts_limit if debug_prompts_limit is not None else 0
                limit = len(prompts_batch) if limit <= 0 else min(limit, len(prompts_batch))
                print(
                    f"[DEBUG] Control prompts for dataset={run.dataset} model={run.model_key} "
                    f"run={run_value} control={control} decile={decile} "
                    f"(showing {limit}/{len(prompts_batch)})"
                )
                for idx in range(limit):
                    status = "inserted" if thinking_insertions[idx] else "present"
                    print(f" id={ids[idx]} thinking_start={status} meta={prompts_with_meta[idx][1]}")
                    print(prompts_batch[idx])
                    print("-" * 40)

            outputs = llm.generate(
                prompts_batch,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            if len(outputs) != num_rows:
                raise RuntimeError(
                    f"{run.path}: expected {num_rows} generations, got {len(outputs)}."
                )

            def iter_lines() -> Iterable[str]:
                for example_id, output, meta in zip(ids, outputs, (p[1] for p in prompts_with_meta)):
                    if not output.outputs:
                        raise RuntimeError(
                            f"{run.path}: no completion returned for id {example_id}."
                        )
                    completion = output.outputs[0]
                    first_pos_logprobs = completion.logprobs[0] if completion.logprobs else None
                    choice_logprobs = compute_choice_logprobs(first_pos_logprobs, id_to_label)

                    record = {
                        "dataset": run.dataset,
                        "model": run.model_key,
                        "model_id": MODEL_CONFIGS[run.model_key]["model_id"],
                        "run": run_value,
                        "decile": decile,
                        "id": example_id,
                        "control_type": control,
                        "choice_logprobs": choice_logprobs,
                    }
                    record.update(meta)
                    yield json.dumps(record, ensure_ascii=False) + "\n"

            write_lines_atomic(out_path, iter_lines())

            print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score control prefixes for decile reasoning slices."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASETS.keys(),
        default=list(DATASETS.keys()),
        help="Datasets to process.",
    )
    parser.add_argument(
        "--model-name",
        nargs="+",
        choices=MODEL_CONFIGS.keys(),
        default=list(MODEL_CONFIGS.keys()),
        help="Model keys to evaluate.",
    )
    parser.add_argument(
        "--decile",
        nargs="+",
        type=int,
        default=list(DECILES),
        help="Deciles to score.",
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=["junk", "cross", "shuffle"],
        default=["junk", "cross", "shuffle"],
        help="Control types to run (baseline is now handled in 03_score_reasoning_deciles.py as decile 0).",
    )
    parser.add_argument(
        "--input-dir",
        default="data/deciles",
        help="Root directory containing decile parquet files (per dataset).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/logprobs_control",
        help="Directory where control JSONL files will be written.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=1000,
        help="Number of logprobs to request per generated token (default: 1000).",
    )
    parser.add_argument(
        "--debug-prompts",
        action="store_true",
        help="Print prompts sent to each generation call for inspection.",
    )
    parser.add_argument(
        "--debug-prompts-limit",
        type=int,
        default=3,
        help="Max number of prompts to print per run/control when --debug-prompts is enabled (<=0 prints all).",
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
        help="Force eager mode to bypass torch.compile/TorchInductor.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip control logprob files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    deciles = sorted(set(args.decile))
    controls = set(args.controls)
    allowed_models = set(args.model_name)

    runs: list[DecileRun] = []
    for dataset in args.dataset:
        runs.extend(discover_decile_runs(dataset, allowed_models, input_root))

    if not runs:
        raise FileNotFoundError(
            f"No decile parquet files found under {input_root} for datasets {sorted(args.dataset)}"
        )

    choice_cache: dict[tuple[str, str], tuple[list[int], dict[int, str]]] = {}

    for model_key in sorted({r.model_key for r in runs}):
        model_runs = sorted(
            [r for r in runs if r.model_key == model_key],
            key=lambda r: (r.dataset, r.run, r.path.name),
        )
        cache_dir = ensure_isolated_inductor_cache(model_key)
        if cache_dir:
            print(f"Using TORCHINDUCTOR_CACHE_DIR={cache_dir}")

        llm = build_llm(model_key, enforce_eager=args.enforce_eager)
        tokenizer = llm.get_tokenizer()

        for run in model_runs:
            score_controls_for_run(
                run,
                deciles=deciles,
                controls=controls,
                choice_cache=choice_cache,
                output_root=output_root,
                llm=llm,
                tokenizer=tokenizer,
                seed=args.seed,
                logprobs_request=args.logprobs,
                skip_existing=args.skip_existing,
                debug_prompts=args.debug_prompts,
                debug_prompts_limit=args.debug_prompts_limit,
            )

        del llm
        gc.collect()


if __name__ == "__main__":
    main()
