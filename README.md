# SP4CPG: Enhancing Graph-based Vulnerability Detection via Semantics-Preserving Graph Pruning

A novel vulnerability detection framework with two components: (1) hyper code property graph (HCPG) construction, which reduces redundancy through semantics-preserving graph pruning and hyperedge creation; and (2) the type-aware flow-sensitive hypergraph convolutional network (TAF-HGCN), which combines node type-aware embeddings with flow-sensitive hyperedge learning to model high-order semantic relationships.

![GitHub Logo](https://github.com/DataAvailable/SP4CPG/blob/main/figures/framework.png)

## 🚀 Features

- **Hyper Code Property Graph Generation**: Automated HCPG generation
- **Advanced Graph Pruning**: Intelligent semantics-preserving graph pruning
- **High-order Representation**: Control and Data hyperedges generation

## 📋 Requirements

### System Dependencies
- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- [Joern](https://joern.io/) - Static analysis platform for code property graphs

### Python Dependencies
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install transformers
pip install scikit-learn
pip install tensorboard
pip install tqdm
pip install tabulate
pip install pydot
pip install numpy
```

## 🏗️ Project Structure

```
├── main.py                     # Main training and evaluation script
├── models/
│   ├── gnn_models.py          # Standard GNN model implementations
│   └── hgnn_models.py         # Hypergraph neural network models
├── preprocess/
│   ├── cpg_generate.py        # CPG generation from source code
│   ├── graph_dataset.py       # Dataset creation and graph preprocessing
│   ├── process_dot.py         # DOT file parsing and feature extraction
│   ├── dots/                  # Generated CPG DOT files
│   ├── source/                # Source code files
│   ├── workspace/             # Joern workspace
├── logs/                      # Training logs and outputs
├── joern-cli/                 # Joern source
└── data/
    └── function.json          # Input dataset (code functions with labels)
```

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <https://github.com/DataAvailable/SP4CPG>
cd SP4CPG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Joern
Follow the [Joern installation guide](https://docs.joern.io/installation/) to install Joern for CPG generation.

### 4. Prepare Dataset
Place your dataset in JSON format at `data/function.json` with the following structure:
```json
[
    {
        "func": "int vulnerable_function() { ... }",
        "target": 1
    },
    {
        "func": "int safe_function() { ... }",
        "target": 0
    }
]
```

## 🔧 Usage

### Step 1: Generate Code Property Graphs
```bash
cd preprocess
python cpg_generate.py
```
This will:
- Parse the JSON dataset
- Generate C source files
- Use Joern to create CPGs
- Output DOT files to `preprocess/dots/`

### Step 2: Create HCPG Dataset
```bash
cd preprocess
python graph_dataset.py
```
This will:
- Process DOT files
- Apply graph pruning strategies
- Generate control and data hyperedges
- Get HCPG
- Save the embedding as `pruned_cpg_dataset.pkl`

### Step 3: Train Models
```bash
# Train with GCN
python main.py --model GCN --batch 128 --lr 1e-4 --dropout 0.4 --epoch 500

# Train with GAT
python main.py --model GAT --batch 64 --lr 5e-5 --dropout 0.3 --epoch 300

# Train with GGNN
python main.py --model GGNN --batch 128 --lr 1e-4 --dropout 0.4 --epoch 500

# Train with Hypergraph GCN
python main.py --model HGCN --batch 64 --lr 1e-4 --dropout 0.5 --epoch 400
```

### Command Line Arguments
- `--model`: Model architecture (GCN, GAT, GIN, GraphSAGE, GGNN, HGCN)
- `--batch`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for optimization (default: 1e-5)
- `--dropout`: Dropout rate (default: 0.4)
- `--epoch`: Maximum number of epochs (default: 500)
- `--patience`: Early stopping patience (default: 100)

## 🧠 Model Architectures

### Standard GNN Models
- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network with multi-head attention
- **GIN**: Graph Isomorphism Network
- **GraphSAGE**: GraphSAGE with sampling
- **GGNN**: Gated Graph Neural Network

### Hypergraph Models
- **HGCN**: Hypergraph Convolutional Network with advanced features:
  - Multi-scale pooling (mean + max)
  - Residual connections
  - Layer normalization
  - Edge dropout regularization

## 📊 Graph Preprocessing

The framework implements sophisticated graph preprocessing strategies:

### 1. Intermediate Node Pruning
- Removes redundant assignment nodes
- Eliminates print statement nodes
- Reconnects parent-child relationships

### 2. AST Leaf Node Processing
- Removes LITERAL and IDENTIFIER leaf nodes
- Merges nodes with identical data types (LOCAL, PARAM)
- Consolidates duplicate TYPE_REF and METHOD_RETURN nodes

### 3. Control Hyperedge Generation

### 4. Data Hyperedge Generation

### 5. Feature Engineering
- Type-aware embeddings for semantic code representation
- Node type embeddings
- Edge type embeddings
- Multi-modal feature fusion

## 📋 Output and Logging

### TensorBoard Visualization
```bash
tensorboard --logdir=logs
```

### Log Files
Training logs are saved in `logs/train_log_YYYY-MM-DD_HH-MM.txt` containing:
- Device information
- Hyperparameter settings
- Epoch-wise training metrics
- Final evaluation results

### Model Checkpoints
Best models are saved as `logs/{MODEL}_{LR}_{DROPOUT}_best_model.pt`

## 🔍 Example Results

```
========== Evaluation Results ==========
┌─────────────────────────────┬─────────┐
│ Metric                      │ Value   │
├─────────────────────────────┼─────────┤
│ Test Accuracy               │ 0.8542  │
│ Precision                   │ 0.8234  │
│ Recall                      │ 0.8765  │
│ F1 Score                    │ 0.8491  │
│ False Positive Rate (FPR)   │ 0.1456  │
│ False Negative Rate (FNR)   │ 0.1235  │
│ AUC                         │ 0.9123  │
│ Test Evaluation Time (s)    │ 2.34    │
└─────────────────────────────┴─────────┘
```

## 🚨 Important Notes

### Memory Considerations
- **Browser Storage Limitation**: The framework avoids localStorage/sessionStorage APIs for compatibility
- **GPU Memory**: Large graphs may require batch size adjustment for GPU memory constraints

### Dataset Requirements
- Input code should be valid C/C++ functions
- Labels should be binary (0 for safe, 1 for vulnerable)
- Minimum dataset size: 1000+ samples recommended

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
```
