# SP4CPG

SP4CPG constructs dependency-preserving Hyper Code Property Graphs (HCPGs) and learns vulnerability classifiers with a type-aware flow-sensitive hypergraph convolutional network (TAF-HGCN).

## Installation

Use Python 3.10. The environment includes PyTorch 2.1.0, PyTorch Geometric 2.3.1, Transformers 4.40.0 and NumPy 1.26.4.

```bash
git clone https://github.com/DataAvailable/SP4CPG.git
cd SP4CPG
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install one PyTorch build:

```bash
# CPU
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

```bash
# NVIDIA GPU with CUDA 12.1
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v210) list additional hardware options. The [PyG installation instructions](https://github.com/pyg-team/pytorch_geometric/tree/2.3.1#installation) describe its packages.

Alternatively, create the Conda environment:

```bash
conda env create -f environment.yml
conda activate sp4cpg
```

Install [Joern](https://docs.joern.io/installation/) and make `joern-parse` and `joern-export` available on `PATH` for CPG generation.

## Graph and training pipeline

Run commands from the repository root except for the CPG export command below.

### 1. Generate CPGs

Place the PrimeVul JSONL input in `data/PrimeVul.json`. Each line contains a JSON object with `func` for the C/C++ function text and `target` for its integer label, `0` or `1`.

```bash
cd preprocess
python cpg_generate.py --dataset PrimeVul.json
cd ..
```

The exported files are named `preprocess/dots-cpg/ID-cpg-LABEL.dot`.

### 2. Construct HCPGs

```bash
python preprocess/hcpg_generate.py --input preprocess/dots-cpg --output preprocess/dots-hcpg
```

For parallel construction:

```bash
python preprocess/hcpg_generate_parallel.py --input preprocess/dots-cpg --output preprocess/dots-hcpg --workers 4
```

Both commands accept a graph file or a directory. DOT is the default output; use `--format json` for JSON files. Directory output includes `construction_report.json` with ordered file statuses. A CPG named `ID-cpg-LABEL.dot` produces `ID-hcpg-LABEL.dot`.

### 3. Encode graph nodes

```bash
python preprocess/dot_embedding.py --input preprocess/dots-hcpg --output preprocess/hcpg_dataset.pkl --codebert microsoft/codebert-base --max-length 128 --batch-size 32 --device cpu
```

`CodeGraphEncoder` freezes CodeBERT, runs it in evaluation mode, and encodes each physical node's `code` text with its CLS vector. The saved file contains a list of `GraphData` samples.

| Option | Use |
|---|---|
| `--labels data/labels.json` | Supply a JSON mapping from `graph_id` to integer `0` or `1`. |
| `--codebert` | Select a Hugging Face model ID or a local model directory. |
| `--model-version` | Select a model commit, tag or branch. |
| `--max-length` | Set the tokenizer's per-node token limit; default `128`. |
| `--batch-size` | Set the node-text encoding batch size; default `32`. |
| `--device cuda` | Encode with a CUDA device. |
| `--local-files-only` | Load model files from the local cache or directory. |
| `--overwrite` | Replace an existing dataset output file. |

Labels can also come from graph metadata or the `ID-cpg-LABEL` / `ID-hcpg-LABEL` filename. Supplied labels must agree.

### 4. Train TAF-HGCN

```bash
python main.py --model HGCN --batch 64 --lr 1e-4 --weight_decay 1e-5 --dropout 0.4 --epoch 500 --patience 100
```

`main.py` loads `preprocess/hcpg_dataset.pkl`, creates an 80/10/10 training/validation/test split with seed `2025`, and trains with Adam and cross-entropy. Early stopping selects validation accuracy. The `HGCN` model name selects `TAFHGCN`.

Logs and checkpoints are written to `logs/`. Checkpoints use the name `MODEL_LR_DROPOUT_best_model.pt`.

```bash
tensorboard --logdir logs
```

## HCPG construction

The construction keeps original node identities and logical dependency endpoints in a dictionary. Physical nodes hold original `entries`; carried contents retain their original code, attributes and expression association. Every relation stores its original endpoint bindings, type, direction, raw label, entity and control context.

1. **Redundant assignment nodes.** A `BLOCK → assignment → alloc` AST pattern is eligible when the assignment's text is retained in both related expressions and the allocation carrier is unambiguous. Assignment and allocation roles are `CALL` attributes. AST paths are bypassed within AST. Incident CFG, CDG and DDG relations move to the content carrier with their original logical bindings.
2. **Comments and constant-output calls.** Dependency-free comments can be removed. A debugging call requires known constant output, an unused return value, output-only effects, valid arguments, no vulnerability-relevant role and a safe CFG bypass. Buffer-writing calls and `%n` formats are retained. CFG bypasses require the same known control context.
3. **AST leaves.** A `LITERAL` or `IDENTIFIER` must have one original AST parent and no independent original dependency or binding. Its complete token and occurrence association must survive in the parent or a recorded content carrier.
4. **Declarations and repeated metadata.** Nodes share a fresh representative only with a common known kind, type and function context. Declarations retain separate original names and scopes; type and return metadata retain separate occurrences. Redirected edges keep their original logical ports.

Unknown conditions retain the node. Explicit call facts use `attributes.pruning.rule2`; identity fields use `declared_type`, `function_context`, `name`, `scope` and `occurrence`.

| Call fact | Required value |
|---|---|
| `call_name` | `printf`, `fprintf`, `puts` or `putchar`, agreeing with the recorded call name. |
| `format` | Resolved constant text; literal `%%` is permitted. |
| `constant_output`, `result_unused`, `output_only` | `true`. |
| `arguments_valid`, `control_context_preserved` | `true`. |
| `vulnerability_relevant` | `false`. |

Calls with independent CDG/DDG edges are retained. A CFG-connected call also needs one incoming and one outgoing CFG relation with the same known context. Original edge annotations remain attached to the bypassed path.

CFG and DDG edges are packed separately in **MM → SM → MS** order. Candidates follow original-ID order, require an existing witness for every physical source–target pair, and keep different original ports distinguishable. CFG grouping uses recorded contexts or explicit CDG controller/branch records. DDG grouping retains all raw labels and available `explicit_constant` annotations. Parallel relations retain distinct IDs and bindings.

| Mode | Group construction |
|---|---|
| MM | At least two sources and two targets; incoming target columns have the same source set and matching source-binding/label patterns. |
| SM | One original source entry with edges to at least two physical targets. |
| MS | At least two physical sources with edges to one original target entry. |

Source and target sets are disjoint within each accepted hyperedge. Each group consumes its own edges; the next candidate is built from the remaining pool. Unmatched edges remain binary.

The JSON schema contains `schema_version`, `graph_id`, `nodes`, `relations`, `dictionary` and `metadata`. DOT output embeds the complete JSON graph. `GraphData` stores directed memberships in `hyperedge_index` and `incidence_role`, binary witnesses in `edge_index`, and the full graph in `canonical_json`. Node and relation index spaces receive independent batch offsets.

## TAF-HGCN

The node-type vocabulary has 17 categories:

```text
FILE, CLASS, METHOD, BLOCK, MODIFIER, CONTROL_STRUCTURE, LITERAL,
PARAM, LOCAL, IDENTIFIER, METHOD_RETURN, CALL, RETURN, JUMP_TARGET,
FIELD_IDENTIFIER, TYPE, TYPE_REF
```

The six relation categories are `AST`, `CFG`, `CDG`, `DDG`, `CTRL` and `DATA`. Trainable node-type embeddings augment the frozen CodeBERT features. Each layer scores the six relation-type embeddings and normalizes their weights with softmax.

For source and target incidence matrices `H_S` and `H_T`, weighted source and target degrees form `D_S` and `D_T`; `D_E` contains each relation's source cardinality:

```text
d_S(v) = sum_e H_S(v,e) w(e)
d_T(v) = sum_e H_T(v,e) w(e)
D_E(e,e) = |S_e|
P_l = D_T^(-1/2) H_T W_l D_E^(-1) H_S^T D_S^(-1/2)
X_(l+1) = ReLU((1/K) sum_(q=1..K) P_l^q X_l Phi_l)
```

Propagation follows **source → relation → target**. Zero-degree factors produce zero rows. Hop size `K` controls the per-layer average over `1..K` steps; `num_layers` counts trainable layers. After the layers, graph mean pooling feeds a two-class classifier.

## Experiments

Run paired CPG/HCPG vector comparisons with:

```bash
python -m experiments.rq6 --cpg preprocess/dots-cpg --hcpg preprocess/dots-hcpg --output output/rq6/D4 --dataset D4
```

The [experiment commands](docs/experiments.md) describe RQ6 options and the LLM token-input, preparation, request and evaluation workflow. LLM settings are stored in `experiments/llm_protocol.json`; the shared two-example prompt is `experiments/prompts/two_shot.txt`.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Project files

| Path | Purpose |
|---|---|
| `preprocess/graph_core.py` | Graph dictionaries, pruning, bindings and hyperedge construction. |
| `preprocess/hcpg_generate.py` | File and directory graph construction. |
| `preprocess/hcpg_generate_parallel.py` | Parallel graph construction. |
| `preprocess/graph_data.py` | Directed PyG data and graph serialization. |
| `preprocess/parse_dot.py` | Graph input and frozen CodeBERT encoding. |
| `preprocess/dot_embedding.py` | Labeled graph dataset encoding. |
| `models/hgcn_models.py` | TAF-HGCN layers and classifier. |
| `models/hgnn_models.py` | HGCN and TAFHGCN exports. |
| `main.py` | Training and evaluation. |
| `experiments/` | Paired graph-vector and LLM commands. |
| `tests/` | Graph, data, model, pipeline and experiment tests. |
