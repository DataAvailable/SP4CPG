"""One architecture registry for training and client inference."""
from __future__ import annotations

MODEL_NAMES = ("TAF-HGCN", "HGNN", "HGCN", "GCN", "GAT", "GIN", "GraphSAGE", "GGNN")


def build_model(name, **config):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model {name!r}; choose from {MODEL_NAMES}")
    if name in {"GCN", "GAT", "GIN", "GraphSAGE"} and config.get("num_layers", 2) != 2:
        raise ValueError(f"{name} is the existing fixed two-layer generic encoder")
    if name == "TAF-HGCN":
        from models.taf_hgcn import TAFHGCN
        return TAFHGCN(**config)
    if name == "HGNN":
        from models.standard_hgnn import HGNN
        return HGNN(**{k: v for k, v in config.items() if k not in ("type_dim", "use_types", "directed")})
    common = {k: config[k] for k in ("dropout", "in_channels", "hidden_channels") if k in config}
    if name == "HGCN":
        import copy
        from models.hgcn_models import HGCN

        class LegacyHGCN(HGCN):
            # Preserve historical state_dict parameter names, but supply incidence explicitly.
            def forward(self, data):
                legacy_input = copy.copy(data)
                legacy_input.edge_index = data.hyperedge_index
                return super().forward(legacy_input)

        return LegacyHGCN(**common, num_layers=config.get("num_layers", 3))
    from models import gnn_models
    if name == "GGNN":
        return gnn_models.GGNN(dropout=common.get("dropout", 0.4),
                               in_channels=common["in_channels"], out_channels=common["hidden_channels"],
                               num_layers=config.get("num_layers", 3))
    return getattr(gnn_models, name)(**common)
