# Training Data

**GSM8K training split — the exact prompt pool used for GRPO/ZSBR training in the paper.**

| File | Content |
|---|---|
| `gsm8k_train_prompts.json` | 7473 training problems (the full pool; the paper's main runs train on ZSBR-selected subsets of it) |

## Format

JSON array; each entry:

| Field | Description |
|---|---|
| `idx` | Pool index (0–7472), the identity used by the scheduler and all result logs |
| `question` | Original GSM8K question text |
| `prompt` | The exact prompt fed to the model: `"Solve the following math problem step by step. Give your final answer as a number.\n\nProblem: {question}"` |
| `answer` | Normalized ground-truth numeric answer (used by the rule-based reward) |

## Source and derivation

- Source: HuggingFace `openai/gsm8k` (config `main`, split `train`), MIT license, Cobbe et al. 2021.
- The `answer` field is extracted from the part of the original solution after `####`, with commas/currency signs stripped and numbers normalized — identical to the reward function used in training.
