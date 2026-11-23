"""Kiểm tra rò rỉ nhãn (đảo ngẫu nhiên is_fraud) cho nhánh GLM thưa, WoE lồng và LightGBM."""

from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fraud_mlp.data import CATEGORICAL_COLUMNS, read_data_splits
from experiments.feature_pipeline import (
    build_tier_a_and_digit_frame,
    fit_predict_nested_sparse_glm,
    fit_transform_nested_te,
)


def main():
    data_path = PROJECT_ROOT / "dataset" / "playground-series-s6e11" / "train.csv"
    splits = read_data_splits(data_path, validation_fraction=0.2, seed=42)
    tr_raw, y_tr_true = splits["train"]
    va_raw, y_va_true = splits["validation"]

    rng = np.random.default_rng(2026)
    y_tr = rng.permutation(y_tr_true.to_numpy(dtype=np.int64))
    y_va = rng.permutation(y_va_true.to_numpy(dtype=np.int64))

    # 1. Kiểm tra rò rỉ nhánh GLM thưa
    glm_oof, [glm_va] = fit_predict_nested_sparse_glm(
        tr_raw, y_tr, [va_raw], C=0.2, min_frequency=3, n_knots=10, n_folds=3, seed=42
    )
    auc_glm_oof = roc_auc_score(y_tr, glm_oof)
    auc_glm_va = roc_auc_score(y_va, glm_va)
    print(f"[Sanity] Sparse GLM (Tier B) | Shuffled OOF AUC: {auc_glm_oof:.5f} | Shuffled Val AUC: {auc_glm_va:.5f}")
    assert abs(auc_glm_oof - 0.5) < 0.015 and abs(auc_glm_va - 0.5) < 0.015

    # 2. Kiểm tra rò rỉ nhánh WoE lồng
    te_tr, [te_va] = fit_transform_nested_te(tr_raw, y_tr, [va_raw], smoothings=(5.0, 30.0), n_folds=3, seed=42)
    va_aucs = [roc_auc_score(y_va, te_va[c].to_numpy()) for c in te_va.columns if c.startswith("te_")]
    print(f"[Sanity] Nested TE | Shuffled Val AUC min={min(va_aucs):.4f} mean={np.mean(va_aucs):.4f} max={max(va_aucs):.4f}")
    assert abs(np.mean(va_aucs) - 0.5) < 0.01

    # 3. Kiểm tra rò rỉ LightGBM trên nền GLM
    base_tr = build_tier_a_and_digit_frame(tr_raw)
    base_va = build_tier_a_and_digit_frame(va_raw)
    X_tr = pd.concat([base_tr, te_tr], axis=1)
    X_va = pd.concat([base_va, te_va], axis=1)
    for col in CATEGORICAL_COLUMNS:
        X_tr[col] = X_tr[col].astype("category")
        X_va[col] = pd.Categorical(X_va[col], categories=X_tr[col].cat.categories)

    dtr = lgb.Dataset(X_tr, label=y_tr, init_score=glm_oof, categorical_feature=list(CATEGORICAL_COLUMNS))
    dva = lgb.Dataset(X_va, label=y_va, init_score=glm_va, reference=dtr, categorical_feature=list(CATEGORICAL_COLUMNS))
    bst = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 6,
            "min_child_samples": 100,
            "verbosity": -1,
            "n_jobs": -1,
            "seed": 42,
        },
        dtr,
        num_boost_round=100,
        valid_sets=[dva],
    )
    pred_va = glm_va + bst.predict(X_va, raw_score=True)
    auc_lgb = roc_auc_score(y_va, pred_va)
    print(f"[Sanity] GLM-Offset Shallow LightGBM | Shuffled Val AUC: {auc_lgb:.5f}")
    assert abs(auc_lgb - 0.5) < 0.015
    print("ALL SANITY CHECKS PASSED: ZERO LEAKAGE CONFIRMED.")


if __name__ == "__main__":
    main()
