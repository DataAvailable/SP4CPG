# Reproducibility and result provenance

## What this update establishes

The executable implementation, numerical/structural tests and CLI checks are
distinct from reproducing the historical full-corpus experiments. See
[local validation](VALIDATION.md) for the actual environment and checks performed
for this update. [paper_reported.json](../results/paper_reported.json) contains
author-reported manuscript summaries, not outputs of the new test suite.
The checker computes its metrics directly from the supplied graph pairs.

Historical reproduction still requires the original split manifests, all
CPG/HCPG pairs, original Joern and encoding settings, trained checkpoints and
per-seed predictions/logs. The revision package supplied aggregate results,
not those raw artifacts. In particular, this update cannot independently certify
all 54,142 reported graph-pair checks or reproduce published F1/efficiency values.
Recheck multi-hop learning and graph-depth statistics using the explicit current
operators/counting conventions; do not assume unchanged scores after code changes.

## Environment and chosen details

The manuscript names torch 2.1.0, torch_geometric 2.3.1, transformers 4.40.0 and
an RTX 4090. `requirements.txt` records those library versions; the validation
report states the actual local test environment separately. Joern's exact
historical version was not supplied; export records the version provided for
each new run, with unavailable historical details left unspecified.

The paper specifies Adam, lr=1e-4, batch=64, dropout=0.4, up to 500 epochs,
patience=100; its parameter sweeps identify 3 layers, hidden width 512 and K=3.
`configs/paper.json` records these values. The current implementation explicitly
chooses type-embedding width 64 and a linear two-class classifier, and retains
legacy defaults of weight decay 1e-5 and validation-accuracy checkpoint selection.
The historical values of these implementation details are not established by
the manuscript. Override them explicitly and retain the resulting run metadata.

CodeBERT now uses each canonical node's `code` text, CLS token features,
configurable truncation (default 128) and a frozen encoder in eval mode.
These are recorded new-run choices; the exact historical checkpoint revision,
node serialization and token settings were not supplied. Pin `--revision`, or
supply a local checkpoint and record its provenance. Both graph forms must use
the same encoder configuration. Encoding failures produce explicit reports;
`--on-error skip` is an opt-in NEW dataset change, which requires a matching
CPG/HCPG sample manifest and cannot be called an unchanged paper split.

## Sample splits, metrics and checkpoint integrity

`scripts/make_split.py` creates a new seeded 8:1:1 split over sorted stable IDs.
The split seed is separate from training seeds. A supplied historical manifest
uses `train`, `validation`, `test` ID lists and `membership_sha256` over sorted
`sample_id`, binary `label`, and `original_node_count` records. The three sets
must be disjoint and cover the complete dataset. Compare CPG/HCPG membership
fingerprints, labels, original graph sizes and encoder metadata before training.

The four controlled configurations train on the complete shared training split
and evaluate on identical test IDs; the >500 restriction applies only to the
reported test subset, based on ORIGINAL CPG size. Per-run precision/recall/F1
use binary classification with zero-division=0 for nonempty samples; empty size
bins are N/A. Aggregate P/R/F1 separately across seeds 2025/2026/2027. Arithmetic
mean F1 is not reconstructed from mean precision and mean recall. Sample
standard deviation is computed only from actual supplied/measured run values.

No checkpoint selection or stopping uses test scores. Checkpoints include model
configuration, schema, graph representation, dataset and split fingerprints,
training seed and best epoch. Run directories are separate for every model,
graph form and seed, preventing accidental result/weight replacement. Eval-only
checks require the same requested model, encoding/data, graph representation and
split. Inference on new samples validates model/feature metadata without
requiring the new sample membership to equal the training membership.

Training time covers the epoch loop including validation and best-checkpoint
writes, excluding preprocessing/model setup. Test time covers the complete-test
forward pass, transfers and prediction materialization, excluding metric
aggregation. CUDA timing synchronizes at measurement boundaries; peak allocated
bytes include the resident model. CPU VRAM is N/A. These explicit new-run
definitions do not recover unrecorded historical timing boundaries.

## Direct dependency validation (RQ7)

The reference comes independently from the original graph and verified rule
preconditions. Only completely removed pass-through nodes are contracted in
CFG; AST adjacency never infers a CFG transition. DDG compares direct original
definition-use associations, not arbitrary chain closure. Carried/merged roles
retain original logical endpoint identities. Keys include original labels,
entity bindings, control context and attributes; missing output information
cannot shrink a comparison key or comparison domain.

The decoded side uses only active output binary edges/hyperedges and checked
dictionary mappings. Every physical pair needs a valid binding witness; global
construction-history records cannot supply absent output edges. The checker
also validates original attributes, carrier contents, occurrence associations,
metadata sharing, explicit constants and CFG bypass path provenance.

For each graph and kind, TP is set intersection, FP decoded minus reference,
and FN reference minus decoded. Dataset micro P/R sums per-graph counts first;
zero denominators are N/A (`null`), never 100%. Exact matching additionally
requires identity/binding/content checks and equal positive-length reachability,
including cycles, self-reachability and previously unreachable pairs. Equal
logical edge sets certify equal reachability; otherwise closures from every
logical anchor are checked, without query sampling. The CLI fails when any graph
is missing, malformed, duplicated or nonmatching, while retaining its report.
The final manuscript scope is all 54,142 test pairs; the earlier U-exclusion
layout is superseded. That count is a reported historical scope, not a hardcoded
denominator or expected success target for a new run.

## LLM comparison

`preprocess/prompt.txt` is the full Appendix-B prompt. Both demonstrations are
vulnerable PrimeVul functions: spice index 71 (CVE-2013-4130), openssl index 0
(CVE-2013-6449), from https://huggingface.co/datasets/colin/PrimeVul.
The selected protocol uses two examples, no task-specific fine-tuning,
temperature=0 and one provider call per eligible target/model.

Prepare a JSON list of `sample_id`, `label`, `func`; a complete-test split; actual
provider settings keyed by model alias; and FULL rendered prompt token counts:

```json
{
  "model-alias": {
    "model_id": "actual-provider-model-id",
    "version": "actual-pinned-provider-version",
    "input_token_limit": 8192,
    "max_output_tokens": 512
  }
}
```

These budgets are format examples, not historical settings. `input_token_limit`
is the actual input budget remaining after reserving output and provider
overhead. Token-count JSON is `{ "sample-id": { "model-alias": 123 } }`, using
the actual tokenizer for each full prompt, including demonstrations. The original
API versions, token limits, filter counts and raw responses were not supplied.

```bash
python scripts/llm_evaluation.py prepare --functions functions.json --split runs/split.json --limits actual_limits.json --token-counts actual_full_prompt_counts.json --output runs/llm_requests
python scripts/llm_evaluation.py collect --manifest runs/llm_requests/manifest.json --responses provider_responses.jsonl --sp4cpg-predictions runs/taf_hcpg_2025/predictions.json --sp4cpg-summary runs/taf_hcpg_2025/summary.json --output runs/llm_comparison.json
```

The prepare step filters to common eligible IDs across every selected model and
records exclusion reasons. `requests.jsonl` contains one prompt per function
and model, without target labels. Execute those requests with the recorded
provider model/version; no credentials or provider SDK is assumed here. Preserve
the provider's raw response once per request in JSONL rows containing `model`,
`sample_id`, `model_id`, `version`, `status` (`ok` or an error state), `response`
(raw JSON text or object), and optional `error`.

Import requires integer label 0/1, finite confidence in [0,1], up to three
reasons/snippets and the requested provider identity. Duplicates, missing/error
responses and parse errors are recorded and prevent comparison metrics. It
does not silently select a different successful subset per model or retry until
a desired answer appears. SP4CPG predictions are aligned to the same eligible
IDs with matching labels, the same split hash and matching prediction-file hash.
These comparisons concern the selected function-text
prompting configurations; graph-conditioned/fine-tuned LLM accuracy, token
savings, cost and whole-project scaling are not established by them.

## Client migration

The Windows client shares `models.factory` with training, reconstructs the
architecture from checkpoint metadata and preserves sample IDs. The default
architecture is TAF-HGCN; `HGCN` remains explicitly the legacy architecture.
Choose a schema-v2 `.pt` dataset and versioned checkpoint. For training, provide
the split JSON in the added configuration field. Old raw state dictionaries and
unversioned `.pkl` inputs are rejected with rebuild/retrain guidance; renaming
an old file does not migrate it. Existing named HGCN configurations retain that
architecture but need compatible versioned data/checkpoints for the new pipeline.

Lexical fallback remains opt-in and is explicitly marked. Its outputs are excluded
from model metrics. The client reads the current run's structured summary and
does not import metrics from an unrelated older log file. Static-evidence fusion
in the client remains a separate workflow, not the paper's RQ7 dependency checker.
