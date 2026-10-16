"""Train/evaluate versioned CPG/HCPG datasets with explicit, shared sample splits."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from models.factory import MODEL_NAMES, build_model
from sp4cpg.experiment import (classification_metrics, evaluate_scopes, file_sha256, load_torch,
                                membership_fingerprint, read_dataset, sample_records, validate_split, write_json)


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    defaults = {}
    if known.config:
        defaults = json.loads(Path(known.config).read_text(encoding="utf-8"))
        defaults.pop("notes", None)
    p = argparse.ArgumentParser(description=__doc__, parents=[pre])
    p.add_argument("--data", required=True, help="Version 2 embedded .pt dataset")
    p.add_argument("--split", required=True, help="Explicit shared sample-ID manifest")
    p.add_argument("--output", required=True, help="New run directory")
    p.add_argument("--model", choices=MODEL_NAMES, default="TAF-HGCN")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--epoch", type=int, default=500)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--type-dim", type=int, default=64)
    p.add_argument("--hop-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=2025, help="Training seed, not split seed")
    p.add_argument("--ablation", choices=("none", "no-tae", "no-fshr"), default="none")
    p.add_argument("--selection-metric", choices=("accuracy", "f1"), default="accuracy")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", help="Versioned checkpoint required for --eval-only")
    unknown = set(defaults) - {a.dest for a in p._actions}
    if unknown:
        p.error(f"Unknown config keys: {sorted(unknown)}")
    p.set_defaults(**defaults)
    args = p.parse_args(argv)
    if min(args.batch, args.epoch, args.patience, args.layers, args.hidden, args.type_dim, args.hop_size) < 1:
        p.error("Batch, epoch, patience and model dimensions/hops must be positive")
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0:
        p.error("Invalid dropout, learning rate or weight decay")
    if args.ablation != "none" and args.model != "TAF-HGCN":
        p.error("TAE/FSHR ablations apply only to TAF-HGCN")
    if args.model in {"GCN", "GAT", "GIN", "GraphSAGE"} and args.layers != 2:
        p.error("The existing generic GCN/GAT/GIN/GraphSAGE encoders have two layers; pass --layers 2")
    if args.eval_only != bool(args.checkpoint):
        p.error("Use --eval-only and --checkpoint together")
    return args


def synchronize(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def predict(model, loader, device):
    import torch
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            if logits.shape != (batch.num_graphs, 2) or not torch.isfinite(logits).all():
                raise ValueError("Model must return finite [num_graphs,2] outputs")
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            predictions = logits.argmax(dim=1).cpu().tolist()
            labels = batch.y.view(-1).cpu().tolist()
            counts = batch.original_node_count.view(-1).cpu().tolist()
            for sample_id, label, prediction, probability, count in zip(
                    batch.sample_id, labels, predictions, probabilities, counts):
                rows.append({"sample_id": sample_id, "label": int(label), "prediction": int(prediction),
                             "probability": probability, "original_node_count": int(count)})
    return rows


def run(args):
    import numpy as np
    import torch
    import torch_geometric
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(args.deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = args.deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    dataset = read_dataset(args.data)
    records = sample_records(dataset["samples"])
    manifest = json.loads(Path(args.split).read_text(encoding="utf-8"))
    splits = validate_split(manifest, records)
    lookup = {sample.sample_id: sample for sample in dataset["samples"]}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader([lookup[i] for i in ids], batch_size=args.batch,
                               shuffle=name == "train", generator=generator if name == "train" else None,
                               num_workers=0) for name, ids in splits.items()}
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_config = {"in_channels": int(dataset["samples"][0].x.size(1)), "hidden_channels": args.hidden,
                    "num_classes": 2, "num_layers": args.layers, "dropout": args.dropout}
    if args.model == "TAF-HGCN":
        model_config.update(hop_size=args.hop_size, type_dim=args.type_dim,
                            use_types=args.ablation != "no-tae", directed=args.ablation != "no-fshr")
    elif args.model == "HGNN":
        model_config["hop_size"] = 1
    training_config = {k: getattr(args, k) for k in ("batch", "lr", "weight_decay", "epoch", "patience",
                                                    "selection_metric", "seed", "deterministic", "threads")}
    metadata = {"schema_version": 2, "result_kind": "measured_evaluation" if args.eval_only else "measured_run", "model": args.model,
                "representation": dataset["representation"], "seed": args.seed, "ablation": args.ablation,
                "model_config": model_config, "training_config": training_config,
                "membership_sha256": membership_fingerprint(records), "split_sha256": file_sha256(args.split),
                "dataset_sha256": file_sha256(args.data), "dataset_metadata": dataset.get("metadata", {}),
                "split_source": manifest.get("source", "supplied_manifest"),
                "historical_paper_split": manifest.get("historical_paper_split", False),
                "environment": {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    model = build_model(args.model, **model_config).to(device)
    write_json(output / "run_config.json", metadata)
    write_json(output / "split.json", manifest)
    checkpoint_path = output / "best_model.pt"
    epochs, training_seconds = [], None
    if args.eval_only:
        checkpoint = load_torch(args.checkpoint, device)
        if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 2:
            raise ValueError("Checkpoint lacks revision model/schema metadata; legacy weights require explicit migration")
        for key in ("model", "model_config", "representation", "membership_sha256", "split_sha256", "dataset_sha256"):
            if checkpoint.get(key) != metadata[key]:
                raise ValueError(f"Checkpoint and requested evaluation differ in {key}")
        model.load_state_dict(checkpoint["state_dict"])
        metadata["evaluation_seed"] = args.seed
        metadata["seed"] = checkpoint["seed"]
        metadata["training_config"] = checkpoint["training_config"]
        metadata["evaluated_checkpoint_sha256"] = file_sha256(args.checkpoint)
        metadata["best_epoch"] = checkpoint.get("best_epoch")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_score, stale = float("-inf"), 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        start = time.perf_counter()
        for epoch in range(1, args.epoch + 1):
            model.train()
            loss_sum, count = 0.0, 0
            for batch in loaders["train"]:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = F.cross_entropy(logits, batch.y.view(-1).long())
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * batch.num_graphs
                count += batch.num_graphs
            val = classification_metrics(predict(model, loaders["validation"], device))
            score = val[args.selection_metric]
            epochs.append({"epoch": epoch, "loss": loss_sum / count, "validation": val})
            print(f"Epoch {epoch}: loss={loss_sum / count:.6f} val_{args.selection_metric}={score:.6f}", flush=True)
            if score > best_score:
                best_score, stale = score, 0
                metadata["best_epoch"] = epoch
                torch.save({**metadata, "state_dict": model.state_dict()}, checkpoint_path)
            else:
                stale += 1
                if stale >= args.patience:
                    break
        synchronize(device)
        training_seconds = time.perf_counter() - start
        metadata["training_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        model.load_state_dict(load_torch(checkpoint_path, device)["state_dict"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    rows = predict(model, loaders["test"], device)
    synchronize(device)
    test_seconds = time.perf_counter() - start
    metadata.update({"evaluation": evaluate_scopes(rows), "training_seconds": training_seconds,
                     "testing_seconds": test_seconds,
                     "testing_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                     "timing_scope": {"training": "epoch loop including validation and best-checkpoint writes; excludes preprocessing/model setup",
                                      "testing": "complete-test forward, device transfers and prediction materialization; excludes metric aggregation",
                                      "memory": "CUDA peak allocated bytes including resident model; CPU is null"}})
    write_json(output / "epochs.json", epochs)
    write_json(output / "predictions.json", rows)
    metadata["predictions_sha256"] = file_sha256(output / "predictions.json")
    write_json(output / "run_config.json", {k: v for k, v in metadata.items() if k != "evaluation"})
    write_json(output / "summary.json", metadata)
    print(json.dumps({"output": str(output), "evaluation": metadata["evaluation"]}, ensure_ascii=False), flush=True)
    return metadata


if __name__ == "__main__":
    run(parse_args())
