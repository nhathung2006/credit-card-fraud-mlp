"""Evaluate a checkpoint on the held-out internal test or validation split."""

import argparse
import math

from sklearn.metrics import recall_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from .configuration import add_common_arguments, parse_options, validation_split_options
from .data import embedding_sizes, read_data_splits, training_data_digest, transform
from .model import FraudMLP
from .runtime import predict_logits, resolve_device


def print_metrics(args, model, checkpoint, device):
    split_options = checkpoint.get("validation_split")
    legacy_checkpoint = split_options is None
    if legacy_checkpoint:
        split_options = validation_split_options(
            args.training_config.get("seed"), args.training_config.get("validation_fraction")
        )
    has_test = split_options.get("test_fraction", 0) > 0
    selected_split = args.split
    if selected_split == "auto":
        selected_split = "test" if has_test else "validation"
    if selected_split == "test" and not has_test:
        raise ValueError(
            "This checkpoint has no held-out internal test split. "
            "Train a new model with training.test_fraction = 0.15."
        )

    training_path = args.data_dir / "train.csv"
    saved_digest = checkpoint.get("training_data_sha256")
    if saved_digest is not None and training_data_digest(training_path) != saved_digest:
        raise ValueError("train.csv differs from the file used to train this checkpoint.")
    splits = read_data_splits(training_path, **split_options)
    frame, labels = splits[selected_split]
    dense, categorical = transform(frame, checkpoint["preprocessor"])
    dataset = TensorDataset(torch.from_numpy(dense), torch.from_numpy(categorical))
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=device.type == "cuda")
    logits = predict_logits(model, loader, device)
    if not logits.isfinite().all():
        raise ValueError("The model returned non-finite logits.")

    auc = float(roc_auc_score(labels, logits.numpy()))
    # Older checkpoints lack split metadata. Reconcile their saved validation AUC.
    if legacy_checkpoint and not math.isclose(
        auc, checkpoint["val_roc_auc"], rel_tol=0, abs_tol=1e-5
    ):
        raise ValueError(
            "Validation ROC-AUC differs from the checkpoint. "
            "Use the original train.csv, training.seed and training.validation_fraction."
        )

    predicted = logits.sigmoid().numpy() >= args.threshold
    label = "Internal test" if selected_split == "test" else "Validation"
    print(f"{label} metrics | Rows: {len(labels):,} | Threshold: {args.threshold:g}")
    print(f"Score (ROC-AUC): {auc:.6f}")
    print(f"Recall: {recall_score(labels, predicted, zero_division=0):.6f}")


def probability_threshold(value):
    threshold = float(value)
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("Expected a probability between 0 and 1.")
    return threshold


def add_evaluation_arguments(parser, *, default_split):
    parser.add_argument("--threshold", type=probability_threshold, default=0.5,
                        help="Classification threshold for recall (default: 0.5)")
    parser.add_argument("--split", choices=["auto", "test", "validation"], default=default_split,
                        help="Labeled split to evaluate (auto selects the checkpoint's held-out split)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    add_evaluation_arguments(parser, default_split="auto")
    parser.set_defaults(metrics=True)
    args = parse_options(parser, training=False, prediction=False)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FraudMLP(
        embedding_sizes(checkpoint["preprocessor"]), **checkpoint.get("model_options", {})
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print_metrics(args, model, checkpoint, device)


if __name__ == "__main__":
    main()
