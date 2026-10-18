# Experiment commands

Run these commands from the repository root after installing `requirements.txt`.

## RQ6: paired graph vectors

CPG and HCPG inputs can each be a DOT/JSON file or a directory of graph files. Graphs are paired by `graph_id`. Provide one CPG and one HCPG for every ID, with matching original dictionary entries and labels. Their `metadata.representation` values are `CPG` and `HCPG`, respectively.

```bash
python -m experiments.rq6 --cpg preprocess/dots-cpg --hcpg preprocess/dots-hcpg --output output/rq6/D1 --dataset D1 --codebert microsoft/codebert-base --max-length 128 --batch-size 32 --device cpu
```

The output directory must be new or empty. Run each dataset with its own inputs, dataset name and output directory.

The dataset IDs are `D1` for FFmpeg+Qemu, `D2` for Chrome+Debian, `D3` for Big-Vul and `D4` for PrimeVul.

Both graphs use the same frozen CodeBERT instance in evaluation mode. Each physical node contributes its `code` text through the same tokenizer and CLS-vector encoder used by `preprocess/dot_embedding.py`. Mean pooling over the node vectors produces one vector per graph; cosine similarity compares the paired graph vectors:

```text
z_G = (1 / |V_G|) sum_v CodeBERT_CLS(code(v))
similarity(CPG,HCPG) = dot(z_CPG,z_HCPG) / (norm(z_CPG) norm(z_HCPG))
```

| Option | Default | Use |
|---|---|---|
| `--codebert` | `microsoft/codebert-base` | Hugging Face model ID or local model directory. |
| `--model-version` | Model default | Model commit, tag or branch. |
| `--max-length` | `128` | Token limit for each node's text. |
| `--batch-size` | `32` | Node-text encoding batch size. |
| `--device` | `cpu` | PyTorch device, such as `cpu` or `cuda`. |
| `--local-files-only` | Disabled | Load files from the local cache or directory. |

| File | Contents |
|---|---|
| `rq6_results.json` | Dataset and encoder settings, paired scores, graph inputs and `statistics`. |
| `rq6_pairs.csv` | `graph_id`, `cpg_nodes`, `hcpg_nodes`, `cosine_similarity`, `cosine_similarity_x100`. |

The JSON `statistics` object contains `min`, `max`, `mean` and the corresponding `min_x100`, `max_x100`, `mean_x100` fields. The `pairs` array contains each graph pair's score.

## LLM function classification

The six model configurations use the same two-example prompt in `experiments/prompts/two_shot.txt` and `temperature=0`. Each selected function receives one individual request. The required response is a JSON object with `label`, `confidence`, `reason` and `key_snippets`.

| Model key | Model |
|---|---|
| `gpt4o_mini` | GPT-4o mini |
| `claude_sonnet4` | Claude Sonnet 4 |
| `gemini25_pro` | Gemini 2.5 Pro |
| `deepseek_coder67b` | DeepSeek-Coder 6.7B |
| `llama31_8b` | Llama 3.1 8B |
| `qwen25_coder7b` | Qwen2.5-Coder 7B |

### 1. Configure the models and test functions

Copy `experiments/llm_protocol.json` to `data/llm_protocol.json` and fill the model settings:

| Field | Value |
|---|---|
| `model_id` | Exact model identifier sent to the configured deployment. |
| `endpoint` | Complete request URL. |
| `api_key_env` | Environment variable holding the deployment's API key; `null` for an unauthenticated chat deployment. |
| `api_version` | Anthropic message API version; `null` for the other providers. |
| `context_window` | Total input and output token capacity. |
| `input_token_limit` | Maximum input token count. |
| `max_output_tokens` | Output token budget for each request. |
| `tokenizer.id`, `tokenizer.version` | Tokenizer and version used to count the complete request. |
| `output_token_field` | `max_tokens` or `max_completion_tokens` for the output budget on chat endpoints. |

Set the environment variables named by `api_key_env` before sending requests. Keep `temperature` at `0`.

Use a JSON array or JSONL test file. Each record has a stable `sample_id`, `id` or `idx`, function text in `func` or `code`, and an integer binary label in `target` or `label`. Records with a `split` field identify the test partition; `--split test` declares the partition for an already separated test file.

### 2. Export the full requests and count tokens

```bash
python -m experiments.llm token-inputs --test data/PrimeVul_test.jsonl --split test --config data/llm_protocol.json --output output/llm/token_inputs.json
```

`token_inputs.json` contains the rendered prompt and each provider's request body for every test ID. Count each complete request with the provider's token counter or the configured model tokenizer, including message formatting, the two examples and the target function.

Write `output/llm/token_counts.json` with this structure:

| JSON field | Contents |
|---|---|
| `schema_version` | Integer `1`. |
| `config_sha256`, `prompt_sha256` | Values from `token_inputs.json`. |
| `models.MODEL_KEY.model_sha256` | Model-setting hash from the exported token inputs. |
| `models.MODEL_KEY.tokenizer` | Matching `id` and `version`. |
| `models.MODEL_KEY.method` | `provider` or `tokenizer`. |
| `models.MODEL_KEY.includes_request_overhead` | Boolean `true`. |
| `records` | One record for every test function. |
| `records[].sample_id` | Stable test ID. |
| `records[].code_sha256` | Exported function-text hash. |
| `records[].rendered_prompt_sha256` | Exported prompt hash for the function. |
| `records[].input_tokens` | Object mapping all six model keys to their exact integer counts. |

### 3. Select the common input IDs

```bash
python -m experiments.llm prepare --test data/PrimeVul_test.jsonl --split test --config data/llm_protocol.json --token-counts output/llm/token_counts.json --output output/llm/manifest.json
```

A function is selected when all six models satisfy both conditions:

```text
input_tokens <= input_token_limit
input_tokens + max_output_tokens <= context_window
```

The manifest stores `selected_ids`, their `samples`, excluded IDs, prompt and configuration hashes, and all token counts. Every model and SP4CPG uses this same ID set.

### 4. Request the six models

```bash
python -m experiments.llm run --manifest output/llm/manifest.json --model gpt4o_mini --config data/llm_protocol.json --output-dir output/llm/gpt4o_mini
python -m experiments.llm run --manifest output/llm/manifest.json --model claude_sonnet4 --config data/llm_protocol.json --output-dir output/llm/claude_sonnet4
python -m experiments.llm run --manifest output/llm/manifest.json --model gemini25_pro --config data/llm_protocol.json --output-dir output/llm/gemini25_pro
python -m experiments.llm run --manifest output/llm/manifest.json --model deepseek_coder67b --config data/llm_protocol.json --output-dir output/llm/deepseek_coder67b
python -m experiments.llm run --manifest output/llm/manifest.json --model llama31_8b --config data/llm_protocol.json --output-dir output/llm/llama31_8b
python -m experiments.llm run --manifest output/llm/manifest.json --model qwen25_coder7b --config data/llm_protocol.json --output-dir output/llm/qwen25_coder7b
```

`--timeout` sets the request timeout in seconds; its default is `120`. Each model directory receives:

| File | Contents |
|---|---|
| `MODEL_KEY.attempts.jsonl` | Attempt IDs, request hashes, response text, statuses and errors. |
| `MODEL_KEY.predictions.jsonl` | Parsed predictions and request statuses. |
| `run.json` | Manifest, configuration and prompt hashes, plus the request timeout. |

Failed requests and invalid responses stay in the output records. Evaluation requires one successful prediction for each selected ID.

### 5. Predict the same functions with SP4CPG

```bash
python -m experiments.sp4cpg_predict --manifest output/llm/manifest.json --data preprocess/hcpg_dataset.pkl --checkpoint logs/HGCN_0.0001_0.4_best_model.pt --output output/llm/sp4cpg.predictions.jsonl --batch-size 64 --device cpu --hop-size 3
```

The command loads the trained TAF-HGCN state dictionary and the encoded `GraphData` list. `--hop-size` supplies the model's per-layer hop size. Model widths, layer count and type-embedding width come from the checkpoint parameters.

CPG export uses an independent working directory for each function and records the function text's SHA256. Temporary source and Joern files are removed after the task. Prediction checks the graph's source hash against the test manifest.

Use `--id-map data/graph_ids.json` to map manifest sample IDs to graph IDs. The mapping is a JSON object whose keys are dataset IDs and whose values are `GraphData.sample_id` strings.

SP4CPG predictions contain `sample_id`, `code_sha256` and integer `prediction` (`0` or `1`). They use the same IDs and code hashes as the manifest.

### 6. Evaluate all predictions

```bash
python -m experiments.llm evaluate --manifest output/llm/manifest.json --predictions gpt4o_mini=output/llm/gpt4o_mini/gpt4o_mini.predictions.jsonl --predictions claude_sonnet4=output/llm/claude_sonnet4/claude_sonnet4.predictions.jsonl --predictions gemini25_pro=output/llm/gemini25_pro/gemini25_pro.predictions.jsonl --predictions deepseek_coder67b=output/llm/deepseek_coder67b/deepseek_coder67b.predictions.jsonl --predictions llama31_8b=output/llm/llama31_8b/llama31_8b.predictions.jsonl --predictions qwen25_coder7b=output/llm/qwen25_coder7b/qwen25_coder7b.predictions.jsonl --predictions sp4cpg=output/llm/sp4cpg.predictions.jsonl --output output/llm/metrics.json
```

The evaluator checks sample IDs and code hashes, then computes each model's confusion counts, precision, recall and F1 against the labels in the manifest.
