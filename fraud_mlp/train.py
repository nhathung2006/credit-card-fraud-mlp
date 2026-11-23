"""Train the MLP and write the checkpoint with the highest validation ROC-AUC."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
import torch
from torch import nn

from .configuration import add_common_arguments, parse_options, positive_int
from .data import (
    ID_COLUMN,
    TARGET_COLUMN,
    fit_transform_train,
    read_data_splits,
    training_data_digest,
    transform,
)
from .model import build_model
from .runtime import TensorBatchLoader, atomic_output, predict_logits, resolve_device


def train(args):
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    training_path = args.data_dir / "train.csv"
    source_digest = training_data_digest(training_path)
    splits = read_data_splits(training_path, **args.validation_split)
    train_frame, train_labels = splits["train"]
    val_frame, val_labels = splits["validation"]

    preprocessor, train_dense, train_categorical = fit_transform_train(
        train_frame, train_labels, args.preprocessing_options, seed=args.seed
    )
    val_dense, val_categorical = transform(val_frame, preprocessor)

    train_loader = TensorBatchLoader(
        torch.from_numpy(train_dense),
        torch.from_numpy(train_categorical),
        torch.from_numpy(train_labels.to_numpy(dtype=np.float32)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = TensorBatchLoader(
        torch.from_numpy(val_dense),
        torch.from_numpy(val_categorical),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(preprocessor, args.model_options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=args.min_lr
        )

    loss_function = nn.BCEWithLogitsLoss()
    best_auc = -math.inf
    best_recall = 0.0
    best_epoch = 0
    best_weights = {}
    best_val_probs = None
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
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.detach() * targets.numel()

        val_logits = predict_logits(model, val_loader, device)
        val_probs = val_logits.sigmoid().numpy()
        auc = float(roc_auc_score(val_labels, val_logits.numpy()))
        recall = float(recall_score(val_labels, val_probs >= 0.5, zero_division=0))
        mean_loss = loss_sum.item() / train_loader.size
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} | LR: {current_lr:.6g} | Train loss: {mean_loss:.5f}"
            f" | Val ROC-AUC: {auc:.6f} | Val Recall (0.5): {recall:.6f}",
            flush=True,
        )
        if auc > best_auc:
            best_auc = auc
            best_recall = recall
            best_epoch = epoch
            best_weights = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_val_probs = val_probs.copy()

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(auc)
            else:
                scheduler.step()

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
        "val_recall_05": best_recall,
    }
    with atomic_output(args.checkpoint) as temporary:
        torch.save(checkpoint, temporary)
    print(
        f"Saved {args.checkpoint} | Epoch {best_epoch} | Val ROC-AUC: {best_auc:.6f}"
        f" | Val Recall (0.5): {best_recall:.6f}"
    )

    if args.save_val_predictions is not None:
        val_output = pd.DataFrame({
            ID_COLUMN: val_frame[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: val_labels.to_numpy(),
            "pred_prob": best_val_probs,
        })
        with atomic_output(args.save_val_predictions) as temporary:
            val_output.to_csv(temporary, index=False)
        print(f"Saved validation predictions to {args.save_val_predictions} | Rows: {len(val_output):,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--patience", type=positive_int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--save-val-predictions",
        type=Path,
        help="Optional CSV path to write best-epoch validation predictions with id",
    )
    train(parse_options(parser, training=True))


if __name__ == "__main__":
    main()
