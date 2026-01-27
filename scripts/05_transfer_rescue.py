from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from vllm import SamplingParams

from prompts import (
    SYSTEM_PROMPT,
    EARLY_STOPPING_PROMPT_QWEN,
    EARLY_STOPPING_PROMPT_GPT_OSS,
)
from utils import (
    MODEL_CONFIGS,
    DATASETS,
    clean_and_split_solution,
    clean_and_split_gpt_oss_solution,
    build_choice_tokens,
    build_llm,
    ensure_isolated_inductor_cache,
    argmax_choice,
    compute_choice_logprobs,
    GPT_OSS_ANALYSIS_START,
)

PROMOTIONS: dict[str, list[str]] = {
    "qwen3_4b": ["qwen3_8b", "qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"],
    "qwen3_8b": ["qwen3_14b", "gpt_oss_20b", "gpt_oss_120b"],
    "qwen3_14b": ["gpt_oss_20b", "gpt_oss_120b"],
    "gpt_oss_20b": ["gpt_oss_120b"],
}
PROMOTION_TARGETS = sorted({m for targets in PROMOTIONS.values() for m in targets})

# Only process these deciles for rescue.
RESCUE_DECILES = (20, 40, 60, 80)


@dataclass(frozen=True)
class DecileLogprobFile:
    dataset: str
    model: str
    run: int
    decile: int
    path: Path


@dataclass(frozen=True)
class RescueExample:
    dataset: str
    base_model: str
    base_spec: str
    run: int
    decile: int
    example_id: str
    prompt: str
    decile_text: str


def parse_logprob_filename(path: Path) -> DecileLogprobFile | None:
    """Extract dataset/model/run/decile from a logprob JSONL filename."""
    pattern = re.compile(
        r"^(?P<dataset>[^_]+)_(?P<model>.+)_run(?P<run>\d+)_decile(?P<decile>\d+)_logprobs\.jsonl$"
    )
    match = pattern.match(path.name)
    if not match:
        return None
    return DecileLogprobFile(
        dataset=match.group("dataset"),
        model=match.group("model"),
        run=int(match.group("run")),
        decile=int(match.group("decile")),
        path=path,
    )


def load_choice_logprob_df(path: Path) -> pd.DataFrame:
    """Load a logprob JSONL file into a dataframe."""
    return pd.read_json(path, lines=True)


def load_answers(dataset: str, data_root: Path) -> pd.Series:
    """Map id -> answer (assumed to already be letter)."""
    parquet_path = data_root / f"{dataset}.parquet"
    df = pd.read_parquet(parquet_path, columns=["id", "answer"])
    df["id"] = df["id"].astype(str)
    return df.set_index("id")["answer"].astype(str).str.strip().str.upper()


def load_decile_parquet(dataset: str, model: str, run: int, decile_root: Path) -> pd.DataFrame:
    """Load decile parquet for a specific dataset/model/run."""
    path = decile_root / dataset / f"{dataset}_{model}_run{run}_reasoning_deciles.parquet"
    cols = ["id", "run", DATASETS[dataset]["prompt_column"]]
    cols.extend([f"thoughts_decile_{d:02d}" for d in RESCUE_DECILES])
    df = pd.read_parquet(path, columns=cols)
    df["id"] = df["id"].astype(str)
    return df.set_index("id")


def split_reasoning_and_answer(spec: str, text: str) -> str:
    """Return reasoning_text from a raw completion."""
    if not isinstance(text, str):
        return ""

    df = pd.DataFrame({"response": [text]})
    if spec == "qwen":
        split_df = clean_and_split_solution(df)
    else:
        split_df = clean_and_split_gpt_oss_solution(df)

    thoughts = split_df["thoughts"].iloc[0] if "thoughts" in split_df else ""
    if spec == "gpt-oss" and isinstance(thoughts, str) and thoughts.startswith(GPT_OSS_ANALYSIS_START):
        thoughts = thoughts[len(GPT_OSS_ANALYSIS_START) :]
    return str(thoughts) if isinstance(thoughts, str) else ""


def compute_free_max_tokens(
    prompt_token_len: int,
    decile_token_len: int,
    max_model_len: int,
    margin: int,
) -> int | None:
    """
    Return allowed generation tokens so prompt + decile + generated fits the model context.
    The prompt_token_len should already include any generation-prompt tokens.
    """
    prompt_len = prompt_token_len + decile_token_len
    remaining = max_model_len - prompt_len - margin
    if remaining <= 0:
        return None
    return remaining


def discover_logprob_files(
    datasets: list[str],
    base_models: set[str],
    deciles: set[int],
    logprob_root: Path,
) -> list[DecileLogprobFile]:
    """Locate relevant logprob JSONL files."""
    discovered: list[DecileLogprobFile] = []
    for dataset in datasets:
        dataset_dir = logprob_root / dataset
        if not dataset_dir.exists():
            continue
        for path in dataset_dir.glob(f"{dataset}_*_decile*_logprobs.jsonl"):
            parsed = parse_logprob_filename(path)
            if not parsed:
                continue
            if parsed.model not in base_models:
                continue
            if parsed.decile not in deciles:
                continue
            discovered.append(parsed)
    return sorted(discovered, key=lambda f: (f.dataset, f.model, f.run, f.decile, f.path.name))


def normalize_decile_text(decile_text: str, base_spec: str, target_spec: str) -> str:
    """
    Convert reasoning tags when transferring across model families.
    Currently only maps Qwen <think> tags into GPT-OSS analysis channel tokens.
    """
    if not decile_text or base_spec == target_spec:
        return decile_text or ""

    if base_spec == "qwen" and target_spec == "gpt-oss":
        # Qwen often emits "<think>\n", so strip any immediate whitespace after the tag.
        text = re.sub(r"\s*<think>\s*", GPT_OSS_ANALYSIS_START, decile_text, count=1)
        # Remove closing tag plus surrounding whitespace/newlines.
        text = re.sub(r"\s*</think>\s*", "", text)
        return text

    return decile_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer mispredicted deciles to stronger models and measure logprobs.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASETS.keys(),
        default=list(DATASETS.keys()),
        help="Datasets to process.",
    )
    parser.add_argument(
        "--base-model",
        nargs="+",
        choices=MODEL_CONFIGS.keys(),
        default=list(MODEL_CONFIGS.keys()),
        help="Base models whose decile logprobs will be used for rescue.",
    )
    parser.add_argument(
        "--target-model",
        nargs="+",
        choices=PROMOTION_TARGETS,
        default=PROMOTION_TARGETS,
        help="Promotion targets to run rescue against.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/datasets",
        help="Directory containing dataset parquet files with ground-truth answers.",
    )
    parser.add_argument(
        "--decile-dir",
        default="data/deciles",
        help="Directory containing decile parquet files produced by script 02.",
    )
    parser.add_argument(
        "--base-logprob-dir",
        default="outputs/logprobs",
        help="Directory containing base decile logprob JSONL files (from script 03).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rescue",
        help="Directory where rescue JSONL files will be written.",
    )
    parser.add_argument(
        "--context-margin",
        type=int,
        default=256,
        help="Reserved tokens to keep buffer under the model max length.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=1000,
        help="Number of logprobs to request per generated token (default: 1000).",
    )
    parser.add_argument(
        "--debug-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print prompts sent to each generation call for inspection (default: enabled).",
    )
    parser.add_argument(
        "--debug-prompts-limit",
        type=int,
        default=3,
        help="Max number of examples to print per dataset/target when --debug-prompts is enabled (<=0 prints all).",
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
        help="Skip examples already present in the output JSONL (append-only when enabled).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    decile_root = Path(args.decile_dir)
    logprob_root = Path(args.base_logprob_dir)
    output_root = Path(args.output_dir)

    rescue_deciles = set(RESCUE_DECILES)
    allowed_targets = set(args.target_model)

    answers_cache: dict[str, pd.Series] = {}
    decile_cache: dict[tuple[str, str, int], pd.DataFrame] = {}

    # Discover candidate base logprob files.
    base_models = set(args.base_model)
    files = discover_logprob_files(args.dataset, base_models, rescue_deciles, logprob_root)
    if not files:
        print(f"No matching logprob files under {logprob_root} for datasets {sorted(args.dataset)}; skipping.")
        return

    rescue_examples_by_target: dict[str, list[RescueExample]] = {}

    for f in files:
        if f.decile not in rescue_deciles:
            continue

        dataset = f.dataset
        base_model = f.model
        run = f.run
        decile = f.decile

        base_spec = MODEL_CONFIGS[base_model]["spec"]

        # Load answers.
        if dataset not in answers_cache:
            answers_cache[dataset] = load_answers(dataset, data_root)
        answers = answers_cache[dataset]

        # Load decile parquet for this base model/run.
        decile_key = (dataset, base_model, run)
        if decile_key not in decile_cache:
            decile_cache[decile_key] = load_decile_parquet(dataset, base_model, run, decile_root)
        decile_df = decile_cache[decile_key]
        prompt_col = DATASETS[dataset]["prompt_column"]

        # Load decile predictions and filter to mispreds.
        decile_df_logprobs = load_choice_logprob_df(f.path)
        for row in decile_df_logprobs.itertuples(index=False):
            example_id = str(row.id)
            answer = answers.get(example_id)
            if not isinstance(answer, str):
                continue
            base_pred = argmax_choice(row.choice_logprobs)
            if base_pred is None or base_pred == answer:
                continue  # skip correct decile predictions

            if example_id not in decile_df.index:
                continue

            decile_col = f"thoughts_decile_{decile:02d}"
            decile_text = decile_df.at[example_id, decile_col]
            prompt_text = decile_df.at[example_id, prompt_col]

            prompt = "" if pd.isna(prompt_text) else str(prompt_text)
            example = RescueExample(
                dataset=dataset,
                base_model=base_model,
                base_spec=base_spec,
                run=run,
                decile=decile,
                example_id=example_id,
                prompt=prompt,
                decile_text="" if pd.isna(decile_text) else str(decile_text),
            )

            for target_model in PROMOTIONS.get(base_model, []):
                if target_model not in allowed_targets:
                    continue
                rescue_examples_by_target.setdefault(target_model, []).append(example)

    # Filter out empty target buckets.
    rescue_examples_by_target = {
        k: v for k, v in rescue_examples_by_target.items() if v
    }
    if not rescue_examples_by_target:
        print("No mispred examples found for the requested deciles/models.")
        return

    def load_existing_keys(path: Path) -> set[tuple[str, int, int, str]]:
        """Return keys of already-processed examples in an output JSONL."""
        keys: set[tuple[str, int, int, str]] = set()
        if not path.exists():
            return keys
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # (base_model, run, decile, id) uniquely identifies an example for a target.
                run_val = rec.get("run", -1)
                decile_val = rec.get("decile", -1)
                id_val = rec.get("id")
                try:
                    run_val = int(run_val)
                    decile_val = int(decile_val)
                except (TypeError, ValueError):
                    continue
                if id_val is None:
                    continue
                keys.add(
                    (
                        str(rec.get("base_model")),
                        run_val,
                        decile_val,
                        str(id_val),
                    )
                )
        return keys

    # Process each target model.
    for target_model, examples in rescue_examples_by_target.items():
        print(f"Processing {len(examples)} examples for target model {target_model}")
        cache_dir = ensure_isolated_inductor_cache(target_model)
        if cache_dir:
            print(f"Using TORCHINDUCTOR_CACHE_DIR={cache_dir}")

        llm = build_llm(target_model, enforce_eager=args.enforce_eager)
        tokenizer = llm.get_tokenizer()
        target_spec = MODEL_CONFIGS[target_model]["spec"]
        stop_prompt = (
            EARLY_STOPPING_PROMPT_QWEN if target_spec == "qwen" else EARLY_STOPPING_PROMPT_GPT_OSS
        )

        choice_cache: dict[str, tuple[list[int], dict[int, str]]] = {}

        by_dataset: dict[str, list[RescueExample]] = {}
        for ex in examples:
            by_dataset.setdefault(ex.dataset, []).append(ex)

        for dataset, ds_examples in by_dataset.items():
            output_path = output_root / dataset / f"{dataset}_{target_model}_rescue.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if dataset not in choice_cache:
                choice_cache[dataset] = build_choice_tokens(dataset, tokenizer)
            choice_token_ids, id_to_label = choice_cache[dataset]

            baseline_logprobs = (
                -1 if args.logprobs <= 0 else max(args.logprobs, len(choice_token_ids))
            )
            baseline_params = SamplingParams(
                max_tokens=1,
                logprobs=baseline_logprobs,
                flat_logprobs=True,
                seed=args.seed,
            )

            existing_keys = load_existing_keys(output_path) if args.skip_existing else set()
            if args.skip_existing and existing_keys:
                print(f"Skipping {len(existing_keys)} already written records in {output_path}")

            ds_examples_filtered = [
                ex
                for ex in ds_examples
                if (ex.base_model, ex.run, ex.decile, ex.example_id) not in existing_keys
            ]
            if not ds_examples_filtered:
                print(f"No new examples for {dataset} -> {target_model}; skipping.")
                continue

            mode = "a" if args.skip_existing and output_path.exists() else "w"
            with output_path.open(mode, encoding="utf-8") as out_f:
                # Precompute prompts and params per example to enable batching.
                messages_batch = [
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": ex.prompt},
                    ]
                    for ex in ds_examples_filtered
                ]
                prompt_prefixes = tokenizer.apply_chat_template(
                    messages_batch,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                prompt_token_ids = tokenizer.apply_chat_template(
                    messages_batch,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                num_examples = len(ds_examples_filtered)
                if len(prompt_prefixes) != num_examples or len(prompt_token_ids) != num_examples:
                    raise RuntimeError(
                        f"{dataset} -> {target_model}: chat template batch size mismatch "
                        f"(expected {num_examples}, got {len(prompt_prefixes)} prefixes and {len(prompt_token_ids)} token batches)."
                    )

                normalized_decile_texts = [
                    normalize_decile_text(ex.decile_text, ex.base_spec, target_spec)
                    for ex in ds_examples_filtered
                ]
                decile_token_lengths = [
                    len(tokenizer.encode(txt, add_special_tokens=False)) for txt in normalized_decile_texts
                ]
                prompt_token_lengths = [len(toks) for toks in prompt_token_ids]

                prepared_examples: list[dict] = []
                for idx, ex in enumerate(ds_examples_filtered):
                    prompt_prefix = prompt_prefixes[idx]
                    normalized_decile_text = normalized_decile_texts[idx]
                    max_tokens = compute_free_max_tokens(
                        prompt_token_lengths[idx],
                        decile_token_lengths[idx],
                        MODEL_CONFIGS[target_model]["max_len"],
                        args.context_margin,
                    )
                    if max_tokens is None:
                        print(
                            f"Skipping id {ex.example_id} for {dataset} -> {target_model}: prompt exceeds context."
                        )
                        continue
                    prepared_examples.append(
                        {
                            "ex": ex,
                            "prompt_prefix": prompt_prefix,
                            "normalized_decile_text": normalized_decile_text,
                            "baseline_prompt": prompt_prefix + normalized_decile_text + stop_prompt,
                            "free_start_prompt": prompt_prefix + normalized_decile_text,
                            "max_tokens": max_tokens,
                        }
                    )

                # Batch baseline measurement: decile slice + stop prompt.
                if not prepared_examples:
                    print(f"No context-fitting examples for {dataset} -> {target_model}; skipping.")
                    continue

                print(f"  {dataset} -> {target_model}: {len(prepared_examples)} examples")

                baseline_prompts = [p["baseline_prompt"] for p in prepared_examples]
                baseline_outputs = llm.generate(
                    baseline_prompts,
                    sampling_params=baseline_params,
                    use_tqdm=True,
                )
                if len(baseline_outputs) != len(prepared_examples):
                    raise RuntimeError(
                        f"{dataset} -> {target_model}: expected {len(prepared_examples)} baseline generations, got {len(baseline_outputs)}."
                    )
                for p, out in zip(prepared_examples, baseline_outputs):
                    if not out.outputs:
                        raise RuntimeError(
                            f"{dataset} -> {target_model}: no baseline completion returned for id {p['ex'].example_id}."
                        )
                    completion = out.outputs[0]
                    first_pos_logprobs = completion.logprobs[0] if completion.logprobs else None
                    baseline_choice_logprobs = compute_choice_logprobs(
                        first_pos_logprobs,
                        id_to_label,
                    )
                    p["baseline_choice_logprobs"] = baseline_choice_logprobs

                # Batch free run continuation (no stop prompt) with per-prompt SamplingParams.
                free_prompts = [p["free_start_prompt"] for p in prepared_examples]
                free_params_list = [
                    SamplingParams(
                        max_tokens=p["max_tokens"],
                        seed=args.seed,
                    )
                    for p in prepared_examples
                ]
                free_outputs = llm.generate(
                    free_prompts,
                    sampling_params=free_params_list,
                    use_tqdm=True,
                )
                if len(free_outputs) != len(prepared_examples):
                    raise RuntimeError(
                        f"{dataset} -> {target_model}: expected {len(prepared_examples)} free generations, got {len(free_outputs)}."
                    )
                for p, out in zip(prepared_examples, free_outputs):
                    if not out.outputs:
                        raise RuntimeError(
                            f"{dataset} -> {target_model}: no free completion returned for id {p['ex'].example_id}."
                        )
                    completion = out.outputs[0]
                    free_text = completion.text or ""
                    free_reasoning_extra = split_reasoning_and_answer(target_spec, free_text)
                    combined_reasoning = p["normalized_decile_text"] + free_reasoning_extra
                    p["free_reasoning_prompt"] = p["prompt_prefix"] + combined_reasoning + stop_prompt
                    p["free_continuation_length"] = len(tokenizer.encode(free_reasoning_extra, add_special_tokens=False))

                if args.debug_prompts:
                    limit = args.debug_prompts_limit if args.debug_prompts_limit is not None else 0
                    limit = len(prepared_examples) if limit <= 0 else limit
                    debug_examples = prepared_examples[:limit]
                    if debug_examples:
                        print(
                            f"[DEBUG] Prompts for dataset={dataset}, target_model={target_model} "
                            f"(showing {len(debug_examples)}/{len(prepared_examples)})"
                        )
                        for p in debug_examples:
                            ex = p["ex"]
                            print(
                                f" id={ex.example_id} base_model={ex.base_model} run={ex.run} decile={ex.decile}"
                            )
                            print("  baseline_prompt:")
                            print(p["baseline_prompt"])
                            print("  free_start_prompt:")
                            print(p["free_start_prompt"])
                            print("  free_reasoning_prompt:")
                            print(p["free_reasoning_prompt"])
                            print("-" * 60)

                # Batch free reasoning measurement: reasoning + stop prompt -> 1 token logprobs.
                free_measure_prompts = [p["free_reasoning_prompt"] for p in prepared_examples]
                free_measure_outputs = llm.generate(
                    free_measure_prompts,
                    sampling_params=baseline_params,
                    use_tqdm=True,
                )
                if len(free_measure_outputs) != len(prepared_examples):
                    raise RuntimeError(
                        f"{dataset} -> {target_model}: expected {len(prepared_examples)} free-measure generations, got {len(free_measure_outputs)}."
                    )
                for p, out in zip(prepared_examples, free_measure_outputs):
                    if not out.outputs:
                        raise RuntimeError(
                            f"{dataset} -> {target_model}: no free-measure completion returned for id {p['ex'].example_id}."
                        )
                    completion = out.outputs[0]
                    first_pos_logprobs = completion.logprobs[0] if completion.logprobs else None
                    free_choice_logprobs = compute_choice_logprobs(
                        first_pos_logprobs,
                        id_to_label,
                    )
                    p["free_choice_logprobs"] = free_choice_logprobs

                for p in prepared_examples:
                    ex = p["ex"]
                    record = {
                        "dataset": ex.dataset,
                        "base_model": ex.base_model,
                        "target_model": target_model,
                        "run": ex.run,
                        "decile": ex.decile,
                        "id": ex.example_id,
                        "baseline_choice_logprobs": p["baseline_choice_logprobs"],
                        "free_choice_logprobs": p["free_choice_logprobs"],
                        "free_continuation_length": p["free_continuation_length"],
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            print(f"Wrote {output_path}")
        del llm
        gc.collect()


if __name__ == "__main__":
    main()
