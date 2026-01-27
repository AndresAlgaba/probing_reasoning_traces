import argparse
import json
import os
import pandas as pd

from vllm import SamplingParams

from prompts import SYSTEM_PROMPT
from utils import MODEL_CONFIGS, DATASETS, build_llm


def generate_reasoning_traces(
    model_name: str,
    dataset: str,
    data_dir: str,
    output_dir: str,
    run_id: int,
) -> str:
    model_cfg = MODEL_CONFIGS[model_name]
    dataset_cfg = DATASETS[dataset]

    llm = build_llm(model_name)
    if model_cfg["spec"] == "qwen":
        sampling_params = SamplingParams(max_tokens=28000)
    else:
        sampling_params = SamplingParams(max_tokens=126000)
    tokenizer = llm.get_tokenizer()

    df = pd.read_parquet(os.path.join(data_dir, f"{dataset}.parquet"))

    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": getattr(row, dataset_cfg["prompt_column"])},
        ]
        for row in df.itertuples(index=False)
    ]

    prompts = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)
    output_path = os.path.join(
        output_dir,
        dataset,
        f"{dataset}_{model_name}_run{run_id}_reasoning_chains.jsonl",
    )

    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
    )

    with open(output_path, "w", encoding="utf-8") as out_f:
        for row, out in zip(df.itertuples(index=False), outputs):
            record = {
                "run": run_id,
                "dataset": dataset,
                "model": model_name,
                "id": row.id,
                "response": out.outputs[0].text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reasoning traces for a specific model/dataset/run."
    )
    parser.add_argument(
        "--model-name",
        choices=MODEL_CONFIGS.keys(),
        required=True,
        help="Logical name of the model to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        required=True,
        help="Dataset to run against (gpqa or mmlu).",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Run identifier used to suffix the output file name.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/datasets",
        help="Directory containing the parquet datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reasoning_traces",
        help="Directory where reasoning traces will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = generate_reasoning_traces(
        model_name=args.model_name,
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(output_path)


if __name__ == "__main__":
    main()
