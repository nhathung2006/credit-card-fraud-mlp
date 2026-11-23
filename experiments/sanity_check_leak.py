"""Kiểm tra rò rỉ nhãn (đảo ngẫu nhiên is_fraud) cho quy trình đặc trưng Tầng B."""

from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fraud_mlp.data import fit_transform_train, read_data_splits, transform
from fraud_mlp.model import build_model
from fraud_mlp.runtime import TensorBatchLoader, predict_logits


def main():
    cfg_path = PROJECT_ROOT / "experiments" / "exp09_resmlp_ple_tierB_pairs.toml"
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)

    data_path = PROJECT_ROOT / "dataset" / "playground-series-s6e11" / "train.csv"
    splits = read_data_splits(data_path, validation_fraction=0.2, seed=42)
    train_frame, train_labels = splits["train"]
    val_frame, val_labels = splits["validation"]

    rng = np.random.default_rng(12345)
    y_train_shuf = pd.Series(rng.permutation(train_labels.to_numpy()), index=train_labels.index)
    y_val_shuf = pd.Series(rng.permutation(val_labels.to_numpy()), index=val_labels.index)

    state, tr_dense, tr_cat = fit_transform_train(
        train_frame, y_train_shuf, cfg["preprocessing"], seed=42
    )
    va_dense, va_cat = transform(val_frame, state)
    print(f"Dense feature dim: {tr_dense.shape[1]} | Train rows: {len(tr_dense):,} | Val rows: {len(va_dense):,}")

    # Kiểm tra AUC đơn biến lớn nhất của từng cột Tầng B trên nhãn đảo ngẫu nhiên
    n_te = len(state["tier_b_state"]["mean"])
    te_val_block = va_dense[:, -n_te:]
    univariate_aucs = [roc_auc_score(y_val_shuf, te_val_block[:, j]) for j in range(n_te)]
    print(
        f"Tier-B univariate AUCs on shuffled labels | min={min(univariate_aucs):.4f}"
        f" | mean={np.mean(univariate_aucs):.4f} | max={max(univariate_aucs):.4f}"
    )

    torch.manual_seed(42)
    model_opts = {
        "architecture": "resmlp",
        "embedding_dim": 4,
        "use_embeddings": False,
        "wide_skip": True,
        "hidden_sizes": [256],
        "dropout": 0.3,
        "norm": "layer",
        "activation": "silu",
        "residual_blocks": 3,
        "residual_dim": 256,
    }
    model = build_model(state, model_opts)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    tr_loader = TensorBatchLoader(
        torch.from_numpy(tr_dense),
        torch.from_numpy(tr_cat),
        torch.from_numpy(y_train_shuf.to_numpy(dtype=np.float32)),
        batch_size=1024,
        shuffle=True,
    )
    va_loader = TensorBatchLoader(
        torch.from_numpy(va_dense),
        torch.from_numpy(va_cat),
        batch_size=1024,
        shuffle=False,
    )

    for epoch in range(1, 4):
        model.train()
        for d, c, y in tr_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(d, c), y)
            loss.backward()
            opt.step()
        logits = predict_logits(model, va_loader, torch.device("cpu"))
        auc = roc_auc_score(y_val_shuf, logits.numpy())
        print(f"[Sanity Check] Epoch {epoch:02d} | Shuffled Val ROC-AUC: {auc:.6f}", flush=True)
        assert abs(auc - 0.5) < 0.015, f"Potential leakage detected! Shuffled AUC={auc:.6f}"

    print("SANITY CHECK PASSED: Shuffled Val ROC-AUC is ~0.500 (zero target leakage).")


if __name__ == "__main__":
    main()
