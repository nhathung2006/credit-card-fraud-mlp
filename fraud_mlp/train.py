"""Train the MLP and write the checkpoint with the highest validation ROC-AUC."""

import argparse
import math

import numpy as np
from sklearn.metrics import recall_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .configuration import add_common_arguments, parse_options, positive_int
from .data import embedding_sizes, fit_preprocessor, read_data_splits, training_data_digest, transform
from .model import FraudMLP
from .runtime import atomic_output, predict_logits, resolve_device


def train(args):
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    training_path = args.data_dir / "train.csv"
    source_digest = training_data_digest(training_path)
    splits = read_data_splits(training_path, **args.validation_split)
    train_frame, train_labels = splits["train"]
    val_frame, val_labels = splits["validation"]

    preprocessor = fit_preprocessor(train_frame)
    train_dense, train_categorical = transform(train_frame, preprocessor)
    val_dense, val_categorical = transform(val_frame, preprocessor)

    training = TensorDataset(
        torch.from_numpy(train_dense),
        torch.from_numpy(train_categorical),
        torch.from_numpy(train_labels.to_numpy(dtype=np.float32)),
    )
    validation = TensorDataset(torch.from_numpy(val_dense), torch.from_numpy(val_categorical))
    loader_options = {"batch_size": args.batch_size, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(training, shuffle=True, **loader_options)
    val_loader = DataLoader(validation, shuffle=False, **loader_options)

    model = FraudMLP(embedding_sizes(preprocessor), **args.model_options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_function = nn.BCEWithLogitsLoss()
    best_auc = -math.inf
    best_epoch = 0
    best_weights = {}
    counts = " | ".join(f"{name.title()}: {len(frame):,}" for name, (frame, _) in splits.items())
    print(f"Device: {device} | {counts}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for dense, categorical, targets in train_loader:
            dense = dense.to(device, non_blocking=True)
            categorical = categorical.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(dense, categorical), targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * targets.numel()

        val_logits = predict_logits(model, val_loader, device)
        auc = float(roc_auc_score(val_labels, val_logits.numpy()))
        recall = recall_score(val_labels, val_logits.sigmoid().numpy() >= 0.5, zero_division=0)
        mean_loss = loss_sum.item() / len(training)
        print(f"Epoch {epoch:02d} | Train loss: {mean_loss:.5f} | Val ROC-AUC: {auc:.6f}"
              f" | Val Recall (0.5): {recall:.6f}",
              flush=True)
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_weights = {name: tensor.detach().cpu().clone()
                            for name, tensor in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    checkpoint = {
        "model_state_dict": best_weights,
        "model_options": args.model_options,
        "preprocessor": preprocessor,
        "validation_split": args.validation_split,
        "training_data_sha256": source_digest,
        "epoch": best_epoch,
        "val_roc_auc": best_auc,
    }
    with atomic_output(args.checkpoint) as temporary:
        torch.save(checkpoint, temporary)
    print(f"Saved {args.checkpoint} | Epoch {best_epoch} | Val ROC-AUC: {best_auc:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--patience", type=positive_int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    train(parse_options(parser, training=True))


if __name__ == "__main__":
    main()
