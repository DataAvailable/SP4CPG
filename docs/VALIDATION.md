# Validation of this code update

Date: October 16, 2026. Base repository commit:
`cfc9b9b9bd546b82230912f6015e819ddb2b0809`.

## Local environment and actual checks

- Windows, Python 3.10.9, CPU-only torch 1.12.1.
- torch_geometric 2.3.1, transformers 4.30.0, NumPy 1.26.4,
  pytest 7.1.2, pydot 3.0.4.
- `python -m pytest -q`: **109 passed**. One upstream torch 1.12
  `scatter_reduce` beta-API warning occurred while exercising the generic GNNs.
- `python -m compileall -q main.py sp4cpg models scripts preprocess client/sp4cpg_client`:
  passed.
- `git diff --check`: passed.

The local environment is not the paper's torch/transformers/GPU environment.
The added CI workflow targets the reference torch 2.1.0/transformers 4.40.0
versions on CPU; the local checks above do not claim that CI has run.

## Test scope

Tests cover conditional pruning, immutable original dependencies, declaration
identity, source constants, dictionary/active-membership checks, lossless
DOT/JSON round trips, deterministic grouping, parallel labels and original
relation identities, sequential AST reconnect provenance, and malformed inputs.
Mutation tests deliberately remove edges, assign invalid retained/merged roles, change
bindings/labels/context/content, omit original roles and try construction-history
substitution. The independent checker must reject those cases.

Model tests compare hand-computed directed weighted normalization, source
cardinality, six-kind softmax, zero degrees/no self-channel, 1…K hop means and
undirected HGNN. They also check independent graph batching, gradients at tiny
positive weights, and finite outputs for all eight registry architectures.
Numerically sensitive normalization/aggregation uses float64 intermediates and
returns the input feature dtype, preserving the normalized operator. Performance
and memory must be measured anew with this implementation.

Integration tests train both revision models on small in-memory test graphs,
reload checkpoints and compare eval-only predictions, run the full
four-configuration CLI on shared IDs, and exercise client inference (including
unlabeled samples). They check that re-evaluating the same checkpoint with a
different evaluation seed cannot become an independent training-seed result.
Split/feature fingerprints, strict classification values, common LLM filtering,
provider-error reporting, client command paths and generated-ID alignment are
also covered. These fixture scores are never published as experimental results.

Joern command tests use controlled command fixtures: safe argv, source IDs/labels,
selection ambiguity, invalid records, failure reporting and exactly one selected
method are checked without invoking Joern. CodeBERT state/token handling is
tested with a small injected encoder; no pretrained checkpoint download or full
corpus encoding was performed. Qt command construction is tested without a live
window; the native GUI and actual static-analysis toolchain were not exercised.

## Existing repository DOT integration

The repository's existing `figures/CVE-2022-0407-CPG.dot` was read from the base
commit, transformed, and checked against the original using the actual CLIs.
It contains 112 original nodes; the conservative transformation produced 83
physical nodes and 212 active relations. The independent checker reported:

| Relation | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFG | 61 | 0 | 0 | 1.0 | 1.0 |
| DDG | 86 | 0 | 0 | 1.0 | 1.0 |

Identity/content/binding checks, structural relations and positive-length
reachability passed. Serial and two-process construction yielded equal graph
objects; directory statistics also completed with the construction report
present. This artifact illustrates the repaired case-study condition, as noted
in the revised manuscript. It is one existing representation example, not the
vulnerable classifier input and not a corpus benchmark.

## What remains unmeasured

No full dataset training, RTX 4090 timing/VRAM measurement, real Joern export,
provider LLM API evaluation, or historical 54,142-pair graph validation was run.
The raw historical artifacts needed to establish those results were not supplied.
Paper-reported summaries remain explicitly separate in
[`results/paper_reported.json`](../results/paper_reported.json).
