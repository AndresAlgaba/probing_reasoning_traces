SYSTEM_PROMPT = "Solve the following problem. Please make sure that your response only consists of a single letter corresponding to the correct answer choice. Do not include anything else in your final response."

EARLY_STOPPING_PROMPT_QWEN = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

BASELINE_STOP_PROMPT_QWEN = "\n</think>\n\n"

EARLY_STOPPING_PROMPT_GPT_OSS = "<|end|><|start|>assistant<|channel|>final<|message|>"
