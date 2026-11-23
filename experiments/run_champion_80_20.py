"""Đối chứng các mô hình học trực tiếp (cây nông/sâu, ResMLP v2 và tổ hợp) trên chia 80/20 (seed=42)."""

from __future__ import annotations

from pathlib import Path
import sys
import time

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fraud_mlp.data import CATEGORICAL_COLUMNS, ID_COLUMN, TARGET_COLUMN, read_data_splits
from fraud_mlp.runtime import TensorBatchLoader, atomic_output
from experiments.feature_pipeline import (
    build_tier_a_and_digit_frame,
    fit_transform_group_aggs,
    fit_transform_nested_te,
)

ARTIFACTS = PROJECT_ROOT / "artifacts"
SUMMARY_CSV = ARTIFACTS / "results_summary.csv"


def safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def save_result(
    name: str,
    val_ids: np.ndarray,
    y_val: np.ndarray,
    val_logits: np.ndarray,
    test_ids: np.ndarray,
    sample_ids: pd.Index,
    test_logits: np.ndarray,
    *,
    best_iter: int,
    elapsed: float,
    note: str = "",
):
    val_probs = expit(val_logits)
    test_probs = expit(test_logits)
    auc = float(roc_auc_score(y_val, val_logits))
    rec = float(recall_score(y_val, val_probs >= 0.5, zero_division=0))

    val_path = ARTIFACTS / f"val_{name}.csv"
    with atomic_output(val_path) as tmp:
        pd.DataFrame({
            ID_COLUMN: val_ids,
            TARGET_COLUMN: y_val,
            "pred_prob": val_probs,
        }).to_csv(tmp, index=False)

    sub_path = ARTIFACTS / f"sub_{name}.csv"
    sub_df = pd.DataFrame({ID_COLUMN: sample_ids})
    sub_df[TARGET_COLUMN] = pd.Series(test_probs, index=test_ids).reindex(sample_ids).to_numpy()
    with atomic_output(sub_path) as tmp:
        sub_df.to_csv(tmp, index=False)

    row = {
        "experiment": name,
        "config": note or name,
        "checkpoint": f"artifacts/sub_{name}.csv",
        "val_predictions": f"artifacts/val_{name}.csv",
        "best_epoch": int(best_iter),
        "val_roc_auc": auc,
        "val_recall_05": rec,
        "elapsed_sec": round(elapsed, 1),
    }
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        df = df[df["experiment"] != name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    with atomic_output(SUMMARY_CSV) as tmp:
        df.to_csv(tmp, index=False)

    print(
        f"[{name}] Val ROC-AUC: {auc:.6f} | Recall@0.5: {rec:.6f} | Best iter: {best_iter} | Time: {elapsed:.1f}s",
        flush=True,
    )
    return auc, val_logits, test_logits


class DirectResMLP(nn.Module):
    """Mạng Wide & Deep Residual MLP có Pre-LayerNorm và Input Dropout."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, n_blocks: int = 3, dropout: float = 0.35, input_dropout: float = 0.15):
        super().__init__()
        self.in_drop = nn.Dropout(input_dropout)
        self.wide = nn.Linear(in_dim, 1)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xd = self.in_drop(x)
        h = self.in_proj(xd)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h).squeeze(1) + self.wide(xd).squeeze(1)


def train_direct_resmlp(
    X_tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    X_va_df: pd.DataFrame,
    y_va: np.ndarray,
    X_te_df: pd.DataFrame,
    *,
    seed: int = 42,
    epochs: int = 22,
    patience: int = 5,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, int]:
    num_cols = [c for c in X_tr_df.columns if c not in CATEGORICAL_COLUMNS]
    tr_num = X_tr_df[num_cols].to_numpy(dtype=np.float32)
    va_num = X_va_df[num_cols].to_numpy(dtype=np.float32)
    te_num = X_te_df[num_cols].to_numpy(dtype=np.float32)

    mean = np.nanmean(tr_num, axis=0)
    std = np.nanstd(tr_num, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    tr_num = np.clip((np.nan_to_num(tr_num, nan=0.0) - mean) / std, -8.0, 8.0).astype(np.float32)
    va_num = np.clip((np.nan_to_num(va_num, nan=0.0) - mean) / std, -8.0, 8.0).astype(np.float32)
    te_num = np.clip((np.nan_to_num(te_num, nan=0.0) - mean) / std, -8.0, 8.0).astype(np.float32)

    oh_tr_list, oh_va_list, oh_te_list = [], [], []
    for col in CATEGORICAL_COLUMNS:
        cats = X_tr_df[col].cat.categories
        k = len(cats)
        for src, dst in [(X_tr_df[col], oh_tr_list), (X_va_df[col], oh_va_list), (X_te_df[col], oh_te_list)]:
            codes = src.cat.codes.to_numpy()
            oh = np.zeros((len(codes), k), dtype=np.float32)
            valid = codes >= 0
            oh[np.arange(len(codes))[valid], codes[valid]] = 1.0
            dst.append(oh)

    X_tr = np.concatenate([tr_num, *oh_tr_list], axis=1)
    X_va = np.concatenate([va_num, *oh_va_list], axis=1)
    X_te = np.concatenate([te_num, *oh_te_list], axis=1)

    torch.manual_seed(seed)
    model = DirectResMLP(X_tr.shape[1], hidden_dim=256, n_blocks=3, dropout=0.35, input_dropout=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    tr_loader = TensorBatchLoader(
        torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)), batch_size=batch_size, shuffle=True
    )
    va_loader = TensorBatchLoader(torch.from_numpy(X_va), batch_size=batch_size, shuffle=False)
    te_loader = TensorBatchLoader(torch.from_numpy(X_te), batch_size=batch_size, shuffle=False)

    top_k: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    best_auc = -1.0
    best_epoch = 0

    for ep in range(1, epochs + 1):
        model.train()
        for (xb, yb) in tr_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        va_preds = []
        with torch.inference_mode():
            for (xb,) in va_loader:
                va_preds.append(model(xb).numpy())
        va_logits = np.concatenate(va_preds)
        auc = float(roc_auc_score(y_va, va_logits))
        print(f"  [ResMLP-Direct-v2] Epoch {ep:02d} | Val ROC-AUC: {auc:.6f}", flush=True)

        state_copy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top_k.append((auc, ep, state_copy))
        top_k.sort(key=lambda t: t[0], reverse=True)
        top_k = top_k[:3]

        if auc > best_auc:
            best_auc = auc
            best_epoch = ep
        if ep - best_epoch >= patience:
            break

    # Trung bình logit của 3 epoch tốt nhất
    va_ens_list, te_ens_list = [], []
    for _, _, st in top_k:
        model.load_state_dict(st)
        model.eval()
        with torch.inference_mode():
            va_ens_list.append(np.concatenate([model(xb).numpy() for (xb,) in va_loader]))
            te_ens_list.append(np.concatenate([model(xb).numpy() for (xb,) in te_loader]))

    va_snap = np.mean(va_ens_list, axis=0)
    te_snap = np.mean(te_ens_list, axis=0)
    print(f"  [ResMLP-Direct-v2] Top-3 Snapshot AUC: {roc_auc_score(y_va, va_snap):.6f} (best single ep={best_epoch}: {best_auc:.6f})", flush=True)
    return va_snap, te_snap, best_epoch


def optimize_logit_weights_nested(y_val: np.ndarray, val_dict: dict[str, np.ndarray], seed: int = 42):
    names = list(val_dict.keys())
    M = np.column_stack([val_dict[k] for k in names])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    nested_oof = np.zeros(len(y_val), dtype=np.float64)
    fold_weights = []

    def fit_weights(X_sub, y_sub):
        k = X_sub.shape[1]
        def obj(params):
            w = np.exp(params[:k])
            w = w / w.sum()
            b = params[k]
            z = X_sub @ w + b
            loss = np.mean(np.logaddexp(0.0, z) - y_sub * z)
            return loss + 5e-4 * np.sum((w - 1.0 / k) ** 2)
        res = minimize(obj, np.zeros(k + 1), method="BFGS")
        w = np.exp(res.x[:k])
        return w / w.sum()

    for tr_i, va_i in skf.split(M, y_val):
        w = fit_weights(M[tr_i], y_val[tr_i])
        fold_weights.append(w)
        nested_oof[va_i] = M[va_i] @ w

    mean_w = np.mean(fold_weights, axis=0)
    mean_w = mean_w / mean_w.sum()
    return names, mean_w, float(roc_auc_score(y_val, nested_oof))


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    probe = ARTIFACTS / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    data_dir = PROJECT_ROOT / "dataset" / "playground-series-s6e11"
    splits = read_data_splits(data_dir / "train.csv", validation_fraction=0.2, seed=42)
    tr_raw, y_tr_s = splits["train"]
    va_raw, y_va_s = splits["validation"]
    y_tr = y_tr_s.to_numpy(dtype=np.int64)
    y_va = y_va_s.to_numpy(dtype=np.int64)
    val_ids = va_raw[ID_COLUMN].to_numpy()

    te_raw = pd.read_csv(data_dir / "test.csv")
    test_ids = te_raw[ID_COLUMN].to_numpy()
    sample_ids = pd.Index(pd.read_csv(data_dir / "sample_submission.csv", usecols=[ID_COLUMN])[ID_COLUMN])

    # 0. Kiểm tra rò rỉ nhãn (Label-Shuffle) trên ma trận đặc trưng v2
    print("\n=== 0. Label-Shuffle Sanity Check (v2 Features + Exact Pairs) ===", flush=True)
    rng = np.random.default_rng(777)
    y_tr_shuf = rng.permutation(y_tr)
    y_va_shuf = rng.permutation(y_va)
    _, [te_va_shuf] = fit_transform_nested_te(tr_raw, y_tr_shuf, [va_raw], smoothings=(5.0, 30.0), n_folds=3, seed=42)
    shuf_aucs = [roc_auc_score(y_va_shuf, te_va_shuf[c].to_numpy()) for c in te_va_shuf.columns if c.startswith("te_")]
    print(f"  Shuffled v2 TE Val AUC | min={min(shuf_aucs):.4f} mean={np.mean(shuf_aucs):.4f} max={max(shuf_aucs):.4f}", flush=True)
    assert abs(np.mean(shuf_aucs) - 0.5) < 0.01, "Leak detected in v2 TE!"

    # Tạo bộ đặc trưng v2 (Tầng A, chữ số, thống kê nhóm, WoE lồng, cặp cột)
    print("\n=== Building v2 Feature Matrix ===", flush=True)
    t_feat = time.perf_counter()
    base_tr = build_tier_a_and_digit_frame(tr_raw)
    base_va = build_tier_a_and_digit_frame(va_raw)
    base_te = build_tier_a_and_digit_frame(te_raw)

    grp_tr, [grp_va, grp_te] = fit_transform_group_aggs(tr_raw, [va_raw, te_raw])
    te_tr, [te_va, te_te] = fit_transform_nested_te(
        tr_raw, y_tr, [va_raw, te_raw], smoothings=(5.0, 30.0), n_folds=5, seed=42
    )

    X_tr = pd.concat([base_tr, grp_tr, te_tr], axis=1)
    X_va = pd.concat([base_va, grp_va, te_va], axis=1)
    X_te = pd.concat([base_te, grp_te, te_te], axis=1)

    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_te.copy()
    for col in CATEGORICAL_COLUMNS:
        X_tr_cb[col] = X_tr_cb[col].astype(str)
        X_va_cb[col] = X_va_cb[col].astype(str)
        X_te_cb[col] = X_te_cb[col].astype(str)

        X_tr[col] = X_tr[col].astype("category")
        cats = X_tr[col].cat.categories
        X_va[col] = pd.Categorical(X_va[col], categories=cats)
        X_te[col] = pd.Categorical(X_te[col], categories=cats)

    print(f"v2 Feature matrix ready in {time.perf_counter() - t_feat:.1f}s | Shape: {X_tr.shape}", flush=True)

    models_val: dict[str, np.ndarray] = {}
    models_test: dict[str, np.ndarray] = {}

    # 1. LightGBM nông học trực tiếp (6 lá)
    print("\n=== 1. Shallow LightGBM Direct v2 (num_leaves=6) ===", flush=True)
    t0 = time.perf_counter()
    dtr_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=list(CATEGORICAL_COLUMNS), free_raw_data=False)
    dva_lgb = lgb.Dataset(X_va, label=y_va, reference=dtr_lgb, categorical_feature=list(CATEGORICAL_COLUMNS), free_raw_data=False)
    bst1 = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.03,
            "num_leaves": 6,
            "max_depth": 4,
            "min_child_samples": 120,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.35,
            "reg_alpha": 5.0,
            "reg_lambda": 25.0,
            "n_jobs": -1,
            "verbosity": -1,
            "seed": 42,
        },
        dtr_lgb,
        num_boost_round=5000,
        valid_sets=[dva_lgb],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)],
    )
    va_l = bst1.predict(X_va, raw_score=True, num_iteration=bst1.best_iteration)
    te_l = bst1.predict(X_te, raw_score=True, num_iteration=bst1.best_iteration)
    _, models_val["lgbm_shallow_v2"], models_test["lgbm_shallow_v2"] = save_result(
        "c1_lgbm_shallow_v2", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=bst1.best_iteration, elapsed=time.perf_counter() - t0,
        note="Shallow LightGBM Direct v2 (num_leaves=6, colsample=0.35)"
    )

    # 2. LightGBM trung bình (15 lá, dùng để đối chứng)
    print("\n=== 2. Medium LightGBM Direct v2 (num_leaves=15) ===", flush=True)
    t0 = time.perf_counter()
    bst2 = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.025,
            "num_leaves": 15,
            "max_depth": 6,
            "min_child_samples": 200,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.30,
            "reg_alpha": 8.0,
            "reg_lambda": 40.0,
            "n_jobs": -1,
            "verbosity": -1,
            "seed": 102,
        },
        dtr_lgb,
        num_boost_round=5000,
        valid_sets=[dva_lgb],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)],
    )
    va_l = bst2.predict(X_va, raw_score=True, num_iteration=bst2.best_iteration)
    te_l = bst2.predict(X_te, raw_score=True, num_iteration=bst2.best_iteration)
    _, models_val["lgbm_medium_v2"], models_test["lgbm_medium_v2"] = save_result(
        "c2_lgbm_medium_v2", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=bst2.best_iteration, elapsed=time.perf_counter() - t0,
        note="Medium LightGBM Direct v2 (num_leaves=15, reg_lambda=40)"
    )

    # 3. XGBoost nông học trực tiếp (độ sâu 3)
    print("\n=== 3. Shallow XGBoost Direct v2 (max_depth=3) ===", flush=True)
    t0 = time.perf_counter()
    dtr_xgb = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dva_xgb = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    dte_xgb = xgb.DMatrix(X_te, enable_categorical=True)
    bst_x3 = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 60.0,
            "subsample": 0.8,
            "colsample_bytree": 0.35,
            "reg_alpha": 5.0,
            "reg_lambda": 25.0,
            "nthread": -1,
            "seed": 42,
        },
        dtr_xgb,
        num_boost_round=5000,
        evals=[(dva_xgb, "val")],
        early_stopping_rounds=150,
        verbose_eval=500,
    )
    va_l = bst_x3.predict(dva_xgb, output_margin=True, iteration_range=(0, bst_x3.best_iteration + 1))
    te_l = bst_x3.predict(dte_xgb, output_margin=True, iteration_range=(0, bst_x3.best_iteration + 1))
    _, models_val["xgb_depth3_v2"], models_test["xgb_depth3_v2"] = save_result(
        "c3_xgb_depth3_v2", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=bst_x3.best_iteration, elapsed=time.perf_counter() - t0,
        note="Shallow XGBoost Direct v2 (max_depth=3)"
    )

    # 4. XGBoost sâu hơn (độ sâu 5, dùng để đối chứng)
    print("\n=== 4. Medium XGBoost Direct v2 (max_depth=5) ===", flush=True)
    t0 = time.perf_counter()
    bst_x5 = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "learning_rate": 0.025,
            "max_depth": 5,
            "min_child_weight": 100.0,
            "subsample": 0.8,
            "colsample_bytree": 0.30,
            "reg_alpha": 8.0,
            "reg_lambda": 40.0,
            "nthread": -1,
            "seed": 102,
        },
        dtr_xgb,
        num_boost_round=5000,
        evals=[(dva_xgb, "val")],
        early_stopping_rounds=150,
        verbose_eval=500,
    )
    va_l = bst_x5.predict(dva_xgb, output_margin=True, iteration_range=(0, bst_x5.best_iteration + 1))
    te_l = bst_x5.predict(dte_xgb, output_margin=True, iteration_range=(0, bst_x5.best_iteration + 1))
    _, models_val["xgb_depth5_v2"], models_test["xgb_depth5_v2"] = save_result(
        "c4_xgb_depth5_v2", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=bst_x5.best_iteration, elapsed=time.perf_counter() - t0,
        note="Medium XGBoost Direct v2 (max_depth=5)"
    )

    # 5. CatBoost đối xứng học trực tiếp (độ sâu 4)
    print("\n=== 5. CatBoost Direct v2 (depth=4, fast one-hot cats) ===", flush=True)
    t0 = time.perf_counter()
    cat_idx = [X_tr_cb.columns.get_loc(c) for c in CATEGORICAL_COLUMNS]
    tr_pool = Pool(X_tr_cb, label=y_tr, cat_features=cat_idx)
    va_pool = Pool(X_va_cb, label=y_va, cat_features=cat_idx)
    te_pool = Pool(X_te_cb, cat_features=cat_idx)
    cb4 = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.04,
        depth=4,
        l2_leaf_reg=15.0,
        one_hot_max_size=16,
        rsm=0.4,
        eval_metric="AUC",
        random_seed=42,
        thread_count=-1,
        verbose=500,
    )
    cb4.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=150, use_best_model=True)
    va_l = cb4.predict(va_pool, prediction_type="RawFormulaVal")
    te_l = cb4.predict(te_pool, prediction_type="RawFormulaVal")
    _, models_val["catboost_d4_v2"], models_test["catboost_d4_v2"] = save_result(
        "c5_catboost_d4_v2", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=cb4.get_best_iteration(), elapsed=time.perf_counter() - t0,
        note="CatBoost Direct v2 (depth=4, one_hot_max_size=16, rsm=0.4)"
    )

    # 6. Wide & Deep Residual MLP v2 (lấy trung bình 3 epoch tốt nhất)
    print("\n=== 6. Wide & Deep Residual MLP Direct v2 ===", flush=True)
    t0 = time.perf_counter()
    va_l, te_l, best_ep = train_direct_resmlp(X_tr, y_tr, X_va, y_va, X_te, seed=42)
    _, models_val["resmlp_direct_v2"], models_test["resmlp_direct_v2"] = save_result(
        "c6_resmlp_direct_v2", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=best_ep, elapsed=time.perf_counter() - t0,
        note="Wide&Deep Residual MLP Direct v2 (InputDrop 0.15 + Top-3 Snapshots)"
    )

    # Ghép thêm exp09 (PLE + ResMLP) nếu có sẵn
    exp09_val_p = ARTIFACTS / "val_exp09_resmlp_ple_tierB_pairs.csv"
    exp09_sub_p = ARTIFACTS / "sub_exp09_resmlp_ple_tierB_pairs.csv"
    if exp09_val_p.exists() and exp09_sub_p.exists():
        v09 = pd.read_csv(exp09_val_p)
        s09 = pd.read_csv(exp09_sub_p)
        if np.array_equal(v09[ID_COLUMN].to_numpy(), val_ids):
            models_val["resmlp_ple_exp09"] = safe_logit(v09["pred_prob"].to_numpy())
            s09_re = pd.Series(s09[TARGET_COLUMN].to_numpy(), index=s09[ID_COLUMN]).reindex(test_ids).to_numpy()
            models_test["resmlp_ple_exp09"] = safe_logit(s09_re)

    # 7. Tổ hợp 7 mô hình tối ưu bằng Nested CV
    print("\n=== 7. Multi-Family Logit Ensemble (Nested-CV Verified) ===", flush=True)
    t0 = time.perf_counter()
    names, weights, nested_auc = optimize_logit_weights_nested(y_va, models_val, seed=42)
    print("Learned Nested-CV Logit Blend Weights:", flush=True)
    for n, w in zip(names, weights):
        print(f"  {n:22s}: {w:.4f} (single Val AUC={roc_auc_score(y_va, models_val[n]):.6f})", flush=True)
    print(f"  --> Strictly Nested-CV Val AUC: {nested_auc:.6f}", flush=True)

    ens_va = sum(w * models_val[n] for n, w in zip(names, weights))
    ens_te = sum(w * models_test[n] for n, w in zip(names, weights))
    save_result(
        "c7_ensemble_champion_80_20", val_ids, y_va, ens_va, test_ids, sample_ids, ens_te,
        best_iter=len(names), elapsed=time.perf_counter() - t0,
        note=f"Nested-CV Logit Blend ({len(names)} models, nested_auc={nested_auc:.6f})"
    )

    print("\n=== TOP 12 MODELS ON 80/20 VALIDATION ===", flush=True)
    df = pd.read_csv(SUMMARY_CSV).sort_values("val_roc_auc", ascending=False).head(12)
    print(df[["experiment", "best_epoch", "val_roc_auc", "val_recall_05", "elapsed_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
