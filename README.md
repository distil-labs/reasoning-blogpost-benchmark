# Reasoning SLM benchmark

Data, configs and predictions behind the distil labs blog post on reasoning small language models. Two tasks, one student (Qwen3.5-4B), each trained twice on the same synthetic data: once with reasoning (`enable_thinking: true`) and once without.

| Qwen3.5-4B, 100 test cases each | invoice | flight |
|---|---|---|
| Base, thinking off | 53 | 13 |
| Base, thinking on | 37 (63 cut off at 2,048 tokens) | 0 (100 cut off at 2,048 tokens) |
| Fine-tuned, thinking off | 84 | 45 |
| Fine-tuned, thinking on | **98** (272.5 median reasoning tokens) | **95** (208.5 median reasoning tokens) |
| Teacher | GLM 5.3 high-thinking | Kimi K3 max-thinking: 100 (thinking off: 17) |

Score = exact match of the JSON answer. Reproduce it with `python score.py results/*/*.jsonl`.

- **invoice**: pay or hold a vendor invoice against its purchase order and goods receipt. Output `{"decision": "approve|hold_no_po|hold_quantity|hold_price|hold_total"}`.
- **flight**: check every connection of a 3 or 4 leg itinerary given local departure times, durations, UTC offsets and minimum connection times. Output `{"valid": ..., "problem": "short_connection|long_layover"|null, "leg": ...}`.

## Layout

```
<task>/seed/                           what we uploaded: train.jsonl (40 examples with reasoning), test.jsonl (100), config.yaml, job_description.json
<task>/training-data/                  the synthetic training set both models were trained on (seed + ~4,000 generated rows)
<task>/training-config-no-thinking.yaml  training-data/config.yaml with enable_thinking: false
results/<task>/                        prediction files for every model in the table
score.py                               scoring script
```

The reasoning style is set in `synthetic_data_generation_instructions` in `job_description.json`, which reaches every prompt that writes reasoning, and by the `reasoning_content` of the seed examples.

## Reproduce with the distil CLI

Install and log in: see [docs.distillabs.ai](https://docs.distillabs.ai). Commands for `flight`; `invoice` is the same.

```bash
# 1. upload the seed data (prints the seed dataset id)
distil seed-dataset create --data flight/seed

# 2. optional: check the teacher on the test set
distil teacher-evaluation create-from-seed-dataset <seed-dataset-id>

# 3. generate the synthetic training data (prints the training dataset id)
distil training-dataset create-from-seed-dataset <seed-dataset-id>
#    or skip generation and upload ours
distil training-dataset create --data flight/training-data

# 4. train the student with and without reasoning on the same data
distil slm create-from-training-dataset <training-dataset-id>
distil slm create-from-training-dataset --config flight/training-config-no-thinking.yaml <training-dataset-id>

# 5. score the fine-tuned model
distil slm status <slm-id>
distil slm download-predictions --file-name flight-thinking.jsonl <slm-id>
pip install transformers
python score.py flight-thinking.jsonl
```

Generation and scoring run at non-zero temperature, so a rerun gives close but not identical numbers.

## Models

| | thinking on | thinking off |
|---|---|---|
| invoice | [distil-qwen3.5-4b-invoice-decision](https://huggingface.co/distil-labs/distil-qwen3.5-4b-invoice-decision) | [distil-qwen3.5-4b-invoice-decision-no-thinking](https://huggingface.co/distil-labs/distil-qwen3.5-4b-invoice-decision-no-thinking) |
| flight | [distil-qwen3.5-4b-flight-connection-check](https://huggingface.co/distil-labs/distil-qwen3.5-4b-flight-connection-check) | [distil-qwen3.5-4b-flight-connection-check-no-thinking](https://huggingface.co/distil-labs/distil-qwen3.5-4b-flight-connection-check-no-thinking) |

All data is synthetic. License: Apache 2.0.
