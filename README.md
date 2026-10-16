# SP4CPG

SP4CPG combines a **Hyper Code Property Graph (HCPG)** representation with a
**type-aware flow-sensitive hypergraph convolutional network (TAF-HGCN)** for
function-level C/C++ vulnerability classification.

HCPG construction is **dependency-preserving under the applicable pruning
preconditions**: the claim concerns CFG/DDG information encoded in the input
CPG for represented program elements. It does not establish full program
behavioral equivalence or recover dependencies absent from Joern's output.
Construction runs independently on each exported function graph.

This revision synchronizes the public implementation with the revised paper's
construction, directed/type-aware model, and evaluation protocols. Historical
unversioned datasets and weights are not interchangeable with the new schema.
See [revision mapping](docs/REVISION_MAPPING.md), [graph schema](docs/GRAPH_SCHEMA.md),
and [reproducibility and result provenance](docs/REPRODUCIBILITY.md).

## Install

Use Python 3.10 in a dedicated environment. The reference dependencies record
the paper's torch 2.1.0, torch_geometric 2.3.1 and transformers 4.40.0. Install
the appropriate PyTorch build for your CPU/CUDA environment before resolving
the remaining requirements.

```bash
git clone https://github.com/DataAvailable/SP4CPG.git
cd SP4CPG
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Install [Joern](https://docs.joern.io/installation/) separately for source-to-CPG
export. Graph construction/validation from existing DOT/JSON needs no Joern
process. Encoding needs a CodeBERT checkpoint; supply a pinned model/tokenizer
revision, or use a local checkpoint with `--local-files-only`.

## Prepare graphs

Supply JSON or JSONL records with `sample_id`, `func` and binary `target` (or
`label`). Keep the same stable IDs throughout export, construction and learning.
Inputs lacking IDs receive explicitly marked generated IDs; those are a new
dataset mapping, not recovered historical experiment IDs.

```json
[{"sample_id": "project:commit:function", "func": "int f(void) { return 0; }", "target": 0}]
```

From the repository root:

```bash
python preprocess/cpg_generate.py --input data/functions.json --output runs/export
python preprocess/hcpg_generate.py --input runs/export/graphs --output runs/hcpg
python scripts/validate_dependencies.py --original runs/export/graphs --transformed runs/hcpg --output runs/dependency_checks.json
python scripts/graph_statistics.py --input runs/hcpg --output runs/hcpg_statistics.json
```

The exporter records commands, source hashes, chosen method DOT and failures.
If several non-global methods remain, select one explicitly with `--method-dot`
or `--method-map`; the exporter does not silently choose the first method.
Parallel wrappers use the same construction/export cores as their serial entrypoints.
Construction defaults to lossless schema-v2 JSON; use `--format dot` for DOT
interchange with embedded dictionary/binding metadata.

**Pruning is conservative.** Ordinary Joern DOT often lacks the facts needed to
justify deleting output calls or merging declarations. Unresolved candidates
remain. Add only program-analysis-derived safety/identity facts described in
[the schema](docs/GRAPH_SCHEMA.md); never derive them from vulnerability labels.
This policy does not imply the historical graph-reduction percentages will be
reproduced on newly imported, less richly annotated graphs.

## Embed and fix a shared split

Run the same node-input construction and CodeBERT revision for both graph forms:

```bash
python preprocess/dot_embedding.py --input runs/export/graphs --representation CPG --labels runs/export/labels.json --revision MODEL_COMMIT --output runs/cpg.pt
python preprocess/dot_embedding.py --input runs/hcpg --representation HCPG --labels runs/export/labels.json --revision MODEL_COMMIT --output runs/hcpg.pt
python scripts/make_split.py --data runs/cpg.pt --output runs/split.json --seed 2025
python scripts/make_split.py --data runs/hcpg.pt --output runs/split.json --validate-only
```

`MODEL_COMMIT` must be replaced with the actual checkpoint revision. The split
command creates a **new** 8:1:1 split. To reproduce a historical run, supply its
actual ID manifest instead. Label, ID or original-CPG-size differences invalidate
the shared manifest. Parse/encoding failures are reported; the default fails the
run rather than converting failed graphs to ordinary empty samples.

## Train and evaluate

```bash
python main.py --config configs/paper.json --model TAF-HGCN --data runs/hcpg.pt --split runs/split.json --output runs/taf_hcpg_2025
python scripts/run_matched.py --cpg-data runs/cpg.pt --hcpg-data runs/hcpg.pt --split runs/split.json --output runs/matched
```

`run_matched.py` runs CPG/HCPG × HGNN/TAF-HGCN for training seeds 2025, 2026,
2027. All configurations reuse the same split and test records. HGNN is the
standard normalized **undirected**, unit-weight encoder without node/relation
type embeddings; each input relation has its own incidence column, even when
memberships coincide. It creates no feature-based hyperedges.

TAF-HGCN uses source/target incidence, learned six-kind relation weights and
17 node-kind embeddings. Each layer averages 1…K propagation steps, independently
of trainable layer count, with no extra zero-hop/residual channel and mean graph
pooling. Use `--ablation no-tae` or `--ablation no-fshr` to remove type-aware
learning or use unordered incidence on the **same** HCPG, respectively.

Every run saves configuration, dataset/split fingerprints, the best checkpoint,
per-epoch validation metrics, per-sample predictions and a structured summary.
Evaluation includes the complete test set, the subset with **more than 500
original CPG nodes**, and size bins. Means aggregate each seed's P/R/F1 separately.
Existing output directories are not silently overwritten.

The registry also retains `GCN`, `GAT`, `GIN`, `GraphSAGE`, `GGNN`, and explicit
legacy architecture `HGCN`. Binary GNNs receive witnessed binary node-to-node
edges; the hypergraph models receive independent node/relation incidence indices.
These generic GNN implementations are not reimplementations of the paper's six
named external vulnerability detectors.
The existing GCN/GAT/GIN/GraphSAGE implementations are fixed at two layers;
pass `--layers 2` when selecting them.

## Other revision experiments

```bash
python scripts/representation_consistency.py --cpg-data runs/cpg.pt --hcpg-data runs/hcpg.pt --split runs/split.json --output runs/consistency.json
python scripts/llm_evaluation.py --help
```

RQ6 compares mean-pooled CodeBERT node features by cosine similarity, without
TAF-HGCN propagation or classifier outputs. It measures representational
consistency, not dependency correctness. RQ7 is the separate direct dependency
checker. The LLM workflow prepares the exact Appendix-B two-shot prompts with
temperature 0 and imports one response per function/model on common eligible
test IDs; [protocol details](docs/REPRODUCIBILITY.md#llm-comparison) describe
provider versioning, token budgets and error handling.

## Results and limits

[Paper-reported results](results/paper_reported.json) are reference summaries
with explicit provenance. This code update does not independently reproduce
the paper's full-corpus scores, 54,142 graph-pair checks or efficiency claims.
No historical simulated layout counts are used as measured results or test
expectations. Fresh measurements belong in separate run directories.

Full historical reproduction needs original split IDs, graph pairs, checkpoint
and encoding details, and per-run logs/predictions. Multi-hop/depth statistics
must be checked with this implementation's explicit conventions.

The Windows client uses the same model factory and checkpoint metadata. Choose
the versioned `.pt` dataset; training additionally requires the split JSON.
Legacy raw weights/unversioned datasets require rebuilding/retraining, and demo
fallbacks never produce model-comparison metrics. See [client migration](docs/REPRODUCIBILITY.md#client-migration).

The project is distributed under the [MIT license](LICENSE).
