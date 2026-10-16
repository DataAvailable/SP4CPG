
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os

APP_NAME = "SP4CPG HGCN Security Workbench"
CONFIG_DIR = Path(os.getenv("APPDATA", str(Path.home()))) / "SP4CPGHGCNClient"
CONFIG_FILE = CONFIG_DIR / "client_config.json"


@dataclass
class ClientConfig:
    repo_root: str = ""
    python_exe: str = "python"
    dataset_name: str = "function.json"
    output_dir: str = ""

    # HGCN settings
    task_mode: str = "predict"  # train | predict | full
    hgcn_model_name: str = "TAF-HGCN"
    hgcn_checkpoint: str = ""
    hgcn_epochs: int = 500
    hgcn_batch: int = 64
    hgcn_lr: float = 1e-4
    hgcn_patience: int = 100
    allow_fallback: bool = False
    hgcn_split_manifest: str = ""

    # Optional graph artifacts from original SP4CPG pipeline
    run_cpg: bool = False
    run_hcpg: bool = False
    run_embedding: bool = False

    # MEDE / static analysis settings
    enable_MEDE: bool = True
    MEDE_home: str = ""
    bitcode_path: str = ""
    MEDE_command: str = ""
    static_result_path: str = ""

    # Advanced/static semantic configuration placeholders
    llm_config_path: str = ""
    skills_home: str = ""
    joern_home: str = ""

    @property
    def repo(self) -> Path:
        return Path(self.repo_root).expanduser().resolve() if self.repo_root else Path.cwd()

    @property
    def preprocess_dir(self) -> Path:
        return self.repo / "preprocess"

    @property
    def data_path(self) -> Path:
        p = Path(self.dataset_name)
        if p.is_absolute():
            return p
        return self.repo / "data" / self.dataset_name


    @property
    def hgcn_dataset_path(self) -> Path:
        """Use explicit embedded input, or the revision's default versioned dataset."""
        p = self.data_path
        if p.suffix.lower() in {".pkl", ".pt", ".pth"}:
            return p
        return self.preprocess_dir / "hcpg_dataset.pt"

    @property
    def logs_dir(self) -> Path:
        return self.repo / "logs"

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir).expanduser().resolve()
        return self.repo / "client_outputs" / "hgcn_workflow"


def load_config() -> ClientConfig:
    if not CONFIG_FILE.exists():
        return ClientConfig()
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Backward compatibility with earlier CodeBERT client configs.
        alias = {
            "codebert_model_name": "hgcn_model_name",
            "codebert_checkpoint": "hgcn_checkpoint",
            "codebert_epochs": "hgcn_epochs",
            "codebert_batch": "hgcn_batch",
            "codebert_lr": "hgcn_lr",
            "codebert_max_length": "hgcn_patience",
        }
        for old_key, new_key in alias.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]
        if data.get("hgcn_model_name") not in {"TAF-HGCN", "HGNN", "GCN", "GAT", "GIN", "GraphSAGE", "GGNN", "HGCN"}:
            data["hgcn_model_name"] = "TAF-HGCN"
        allowed = {field.name for field in ClientConfig.__dataclass_fields__.values()}
        return ClientConfig(**{k: v for k, v in data.items() if k in allowed})
    except Exception:
        return ClientConfig()


def save_config(config: ClientConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
