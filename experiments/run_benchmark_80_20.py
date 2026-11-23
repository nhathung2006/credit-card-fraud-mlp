"""Đối chứng các họ mô hình trên tập chia khóa 80/20 (seed=42): GLM thưa, cây nông và ResMLP."""

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
    fit_predict_nested_sparse_glm,
    fit_transform_nested_te,
)

ARTIFACTS = PROJECT_ROOT / "artifacts"
SUMMARY_CSV = ARTIFACTS / "results_summary.csv"


def safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def save_val_and_test(
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


class ResidualOnGLMNet(nn.Module):
    """Mạng Residual MLP học phần sai số phi tuyến cộng thêm vào logit nền của GLM."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, n_blocks: int = 3, dropout: float = 0.3, input_dropout: float = 0.15):
        super().__init__()
        self.in_drop = nn.Dropout(input_dropout)
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
        # Hệ số khuếch đại học được cho logit nền GLM
        self.glm_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, glm_offset: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(self.in_drop(x))
        for blk in self.blocks:
            h = h + blk(h)
        delta = self.head(h).squeeze(1)
        return self.glm_scale * glm_offset + delta


def train_resmlp_on_glm(
    X_tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    glm_tr: np.ndarray,
    X_va_df: pd.DataFrame,
    y_va: np.ndarray,
    glm_va: np.ndarray,
    X_te_df: pd.DataFrame,
    glm_te: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 25,
    patience: int = 5,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Huấn luyện Residual MLP trên đặc trưng chuẩn hóa, One-Hot và logit nền GLM."""
    num_cols = [c for c in X_tr_df.columns if c not in CATEGORICAL_COLUMNS]
    tr_num = X_tr_df[num_cols].to_numpy(dtype=np.float32)
    va_num = X_va_df[num_cols].to_numpy(dtype=np.float32)
    te_num = X_te_df[num_cols].to_numpy(dtype=np.float32)

    mean = np.nanmean(tr_num, axis=0)
    std = np.nanstd(tr_num, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    tr_num = np.clip((np.nan_to_num(tr_num, nan=0.0) - mean) / std, -10.0, 10.0).astype(np.float32)
    va_num = np.clip((np.nan_to_num(va_num, nan=0.0) - mean) / std, -10.0, 10.0).astype(np.float32)
    te_num = np.clip((np.nan_to_num(te_num, nan=0.0) - mean) / std, -10.0, 10.0).astype(np.float32)

    # Mã hóa One-Hot cho 5 cột phân loại
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
    model = ResidualOnGLMNet(X_tr.shape[1], hidden_dim=256, n_blocks=3, dropout=0.35, input_dropout=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    tr_loader = TensorBatchLoader(
        torch.from_numpy(X_tr),
        torch.from_numpy(glm_tr.astype(np.float32)),
        torch.from_numpy(y_tr.astype(np.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    va_loader = TensorBatchLoader(
        torch.from_numpy(X_va),
        torch.from_numpy(glm_va.astype(np.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    te_loader = TensorBatchLoader(
        torch.from_numpy(X_te),
        torch.from_numpy(glm_te.astype(np.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    best_auc = -1.0
    best_epoch = 0
    best_state = None
    best_va_logits = None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, gb, yb in tr_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, gb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        va_preds = []
        with torch.inference_mode():
            for xb, gb in va_loader:
                va_preds.append(model(xb, gb).numpy())
        va_logits = np.concatenate(va_preds)
        auc = float(roc_auc_score(y_va, va_logits))
        print(f"  [ResMLP-GLM] Epoch {ep:02d} | Val ROC-AUC: {auc:.6f} | glm_scale={model.glm_scale.item():.3f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_va_logits = va_logits.copy()
        if ep - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    te_preds = []
    with torch.inference_mode():
        for xb, gb in te_loader:
            te_preds.append(model(xb, gb).numpy())
    te_logits = np.concatenate(te_preds)
    return best_va_logits, te_logits, best_epoch


def optimize_logit_weights_nested(y_val: np.ndarray, val_dict: dict[str, np.ndarray], seed: int = 42):
    """Tối ưu trọng số cộng logit bằng kiểm chứng chéo lồng 5-Fold trên tập validation."""
    names = list(val_dict.keys())
    M = np.column_stack([val_dict[k] for k in names])
    # Giữ nguyên thang logit để trọng số tổ hợp dễ diễn giải
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    nested_oof = np.zeros(len(y_val), dtype=np.float64)
    fold_weights = []

    def fit_weights(X_sub, y_sub):
        # Tối ưu log-loss với trọng số cộng dương (simplex)
        k = X_sub.shape[1]
        def obj(params):
            w = np.exp(params[:k])
            w = w / w.sum()
            b = params[k]
            z = X_sub @ w + b
            # Hàm mất mát cross-entropy ổn định số học
            loss = np.mean(np.logaddexp(0.0, z) - y_sub * z)
            return loss + 1e-4 * np.sum((w - 1.0 / k) ** 2)
        res = minimize(obj, np.zeros(k + 1), method="BFGS")
        w = np.exp(res.x[:k])
        return w / w.sum()

    for tr_i, va_i in skf.split(M, y_val):
        w = fit_weights(M[tr_i], y_val[tr_i])
        fold_weights.append(w)
        nested_oof[va_i] = M[va_i] @ w

    mean_w = np.mean(fold_weights, axis=0)
    mean_w = mean_w / mean_w.sum()
    nested_auc = float(roc_auc_score(y_val, nested_oof))
    return names, mean_w, nested_auc


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

    models_val: dict[str, np.ndarray] = {}
    models_test: dict[str, np.ndarray] = {}

    # 1A. GLM thưa Tầng A (không dùng chữ số lẻ)
    print("\n=== 1A. Sparse Spline GLM (Tier A Clean) ===", flush=True)
    t0 = time.perf_counter()
    glmA_oof, [glmA_va, glmA_te] = fit_predict_nested_sparse_glm(
        tr_raw, y_tr, [va_raw, te_raw], C=0.25, min_frequency=3, n_knots=12, n_folds=5, seed=42, tier_a_only=True
    )
    print(f"  Inner 5-Fold Train OOF AUC (Tier A GLM): {roc_auc_score(y_tr, glmA_oof):.6f}", flush=True)
    save_val_and_test(
        "m1a_sparse_glm_tierA", val_ids, y_va, glmA_va, test_ids, sample_ids, glmA_te,
        best_iter=1, elapsed=time.perf_counter() - t0, note="Sparse Spline+Discrete GLM (Tier A Clean, C=0.25)"
    )

    # 1B. GLM thưa Tầng B (chọn C bằng OOF nội bộ)
    print("\n=== 1B. Joint Sparse Spline + Exact/Digit GLM (Tier B) ===", flush=True)
    t0 = time.perf_counter()
    best_C, best_oof_auc = None, -1.0
    best_glmB = None
    for C_val in (0.08, 0.20, 0.45):
        oof_c, others_c = fit_predict_nested_sparse_glm(
            tr_raw, y_tr, [va_raw, te_raw], C=C_val, min_frequency=3, n_knots=12, n_folds=5, seed=42, tier_a_only=False
        )
        oof_auc = float(roc_auc_score(y_tr, oof_c))
        print(f"  C={C_val:.2f} | Inner 5-Fold Train OOF AUC: {oof_auc:.6f}", flush=True)
        if oof_auc > best_oof_auc:
            best_oof_auc = oof_auc
            best_C = C_val
            best_glmB = (oof_c, others_c[0], others_c[1])
    glmB_oof, glmB_va, glmB_te = best_glmB
    _, v_l, t_l = save_val_and_test(
        "m1b_sparse_glm_tierB", val_ids, y_va, glmB_va, test_ids, sample_ids, glmB_te,
        best_iter=1, elapsed=time.perf_counter() - t0, note=f"Joint Sparse Spline+Exact/Digit GLM (C={best_C})"
    )
    models_val["sparse_glm_tierB"] = v_l
    models_test["sparse_glm_tierB"] = t_l

    # Tạo bảng đặc trưng dùng chung (Tầng A + chữ số + WoE lồng + tần suất)
    print("\n=== Building Shared Feature Matrix (Tier A + Digits + Nested Multi-Smoothing TE) ===", flush=True)
    t_feat = time.perf_counter()
    base_tr = build_tier_a_and_digit_frame(tr_raw)
    base_va = build_tier_a_and_digit_frame(va_raw)
    base_te = build_tier_a_and_digit_frame(te_raw)
    te_tr, [te_va, te_te] = fit_transform_nested_te(
        tr_raw, y_tr, [va_raw, te_raw], smoothings=(5.0, 30.0), n_folds=5, seed=42
    )
    X_tr = pd.concat([base_tr, te_tr], axis=1)
    X_va = pd.concat([base_va, te_va], axis=1)
    X_te = pd.concat([base_te, te_te], axis=1)

    # Lưu bản sao chuỗi cho CatBoost trước khi đổi sang kiểu category
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

    print(f"Feature matrix ready in {time.perf_counter() - t_feat:.1f}s | Shape: {X_tr.shape}", flush=True)

    # 2A. LightGBM nông học trực tiếp (6 lá)
    print("\n=== 2A. Shallow LightGBM Direct (num_leaves=6) ===", flush=True)
    t0 = time.perf_counter()
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 6,
        "max_depth": 4,
        "min_child_samples": 120,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.45,
        "reg_alpha": 5.0,
        "reg_lambda": 25.0,
        "n_jobs": -1,
        "verbosity": -1,
        "seed": 42,
    }
    dtr_direct = lgb.Dataset(X_tr, label=y_tr, categorical_feature=list(CATEGORICAL_COLUMNS))
    dva_direct = lgb.Dataset(X_va, label=y_va, reference=dtr_direct, categorical_feature=list(CATEGORICAL_COLUMNS))
    bst_direct = lgb.train(
        lgb_params,
        dtr_direct,
        num_boost_round=4000,
        valid_sets=[dva_direct],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)],
    )
    va_l = bst_direct.predict(X_va, raw_score=True, num_iteration=bst_direct.best_iteration)
    te_l = bst_direct.predict(X_te, raw_score=True, num_iteration=bst_direct.best_iteration)
    _, v_l, t_l = save_val_and_test(
        "m2a_lgbm_shallow_direct", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=bst_direct.best_iteration, elapsed=time.perf_counter() - t0,
        note="Shallow LightGBM Direct (num_leaves=6, Tier A+B Nested TE)"
    )
    models_val["lgbm_direct"] = v_l
    models_test["lgbm_direct"] = t_l

    # 2B. LightGBM nông trên nền GLM thưa
    print("\n=== 2B. Shallow LightGBM + GLM-Offset (num_leaves=6) ===", flush=True)
    t0 = time.perf_counter()
    lgb_off_params = dict(lgb_params, learning_rate=0.02, num_leaves=6, colsample_bytree=0.4, reg_lambda=30.0)
    dtr_off = lgb.Dataset(X_tr, label=y_tr, init_score=glmB_oof, categorical_feature=list(CATEGORICAL_COLUMNS))
    dva_off = lgb.Dataset(X_va, label=y_va, init_score=glmB_va, reference=dtr_off, categorical_feature=list(CATEGORICAL_COLUMNS))
    bst_off = lgb.train(
        lgb_off_params,
        dtr_off,
        num_boost_round=4000,
        valid_sets=[dva_off],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)],
    )
    # LightGBM không tự cộng init_score khi dự đoán nên cần cộng tường minh
    va_l = glmB_va + bst_off.predict(X_va, raw_score=True, num_iteration=bst_off.best_iteration)
    te_l = glmB_te + bst_off.predict(X_te, raw_score=True, num_iteration=bst_off.best_iteration)
    _, v_l, t_l = save_val_and_test(
        "m2b_lgbm_shallow_glmoffset", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=bst_off.best_iteration, elapsed=time.perf_counter() - t0,
        note="Shallow LightGBM + GLM-Offset (num_leaves=6)"
    )
    models_val["lgbm_glmoffset"] = v_l
    models_test["lgbm_glmoffset"] = t_l

    # 3. XGBoost nông trên nền GLM (độ sâu 3)
    print("\n=== 3. Shallow XGBoost + GLM-Offset (max_depth=3) ===", flush=True)
    t0 = time.perf_counter()
    dtr_xgb = xgb.DMatrix(X_tr, label=y_tr, base_margin=glmB_oof, enable_categorical=True)
    dva_xgb = xgb.DMatrix(X_va, label=y_va, base_margin=glmB_va, enable_categorical=True)
    dte_xgb = xgb.DMatrix(X_te, base_margin=glmB_te, enable_categorical=True)
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.025,
        "max_depth": 3,
        "min_child_weight": 50.0,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        "reg_alpha": 5.0,
        "reg_lambda": 30.0,
        "nthread": -1,
        "seed": 42,
    }
    bst_xgb = xgb.train(
        xgb_params,
        dtr_xgb,
        num_boost_round=4000,
        evals=[(dva_xgb, "val")],
        early_stopping_rounds=150,
        verbose_eval=500,
    )
    # XGBoost tự động cộng sẵn base_margin trong DMatrix
    va_l = bst_xgb.predict(dva_xgb, output_margin=True, iteration_range=(0, bst_xgb.best_iteration + 1))
    te_l = bst_xgb.predict(dte_xgb, output_margin=True, iteration_range=(0, bst_xgb.best_iteration + 1))
    _, v_l, t_l = save_val_and_test(
        "m3_xgb_shallow_glmoffset", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=bst_xgb.best_iteration, elapsed=time.perf_counter() - t0,
        note="Shallow XGBoost + GLM-Offset (max_depth=3)"
    )
    models_val["xgb_glmoffset"] = v_l
    models_test["xgb_glmoffset"] = t_l

    # 4. CatBoost đối xứng trên nền GLM (độ sâu 4)
    print("\n=== 4. Shallow CatBoost + GLM-Offset (depth=4) ===", flush=True)
    t0 = time.perf_counter()
    cat_idx = [X_tr_cb.columns.get_loc(c) for c in CATEGORICAL_COLUMNS]
    tr_pool = Pool(X_tr_cb, label=y_tr, baseline=glmB_oof, cat_features=cat_idx)
    va_pool = Pool(X_va_cb, label=y_va, baseline=glmB_va, cat_features=cat_idx)
    te_pool = Pool(X_te_cb, baseline=glmB_te, cat_features=cat_idx)
    cb = CatBoostClassifier(
        iterations=3500,
        learning_rate=0.035,
        depth=4,
        l2_leaf_reg=20.0,
        eval_metric="AUC",
        random_seed=42,
        thread_count=-1,
        verbose=500,
    )
    cb.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=150, use_best_model=True)
    # CatBoost tự động cộng sẵn baseline trong Pool
    va_l = cb.predict(va_pool, prediction_type="RawFormulaVal")
    te_l = cb.predict(te_pool, prediction_type="RawFormulaVal")
    _, v_l, t_l = save_val_and_test(
        "m4_catboost_shallow_glmoffset", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=cb.get_best_iteration(), elapsed=time.perf_counter() - t0,
        note="Shallow CatBoost + GLM-Offset (depth=4)"
    )
    models_val["catboost_glmoffset"] = v_l
    models_test["catboost_glmoffset"] = t_l

    # 5. Residual MLP trên nền GLM + Tầng B
    print("\n=== 5. Residual MLP on GLM-Offset + Tier B ===", flush=True)
    t0 = time.perf_counter()
    va_l, te_l, best_ep = train_resmlp_on_glm(X_tr, y_tr, glmB_oof, X_va, y_va, glmB_va, X_te, glmB_te, seed=42)
    _, v_l, t_l = save_val_and_test(
        "m5_resmlp_glmoffset", val_ids, y_va, va_l, test_ids, sample_ids, te_l,
        best_iter=best_ep, elapsed=time.perf_counter() - t0,
        note="Residual MLP (InputDrop 0.15 + SiLU + GLM-Offset)"
    )
    models_val["resmlp_glmoffset"] = v_l
    models_test["resmlp_glmoffset"] = t_l

    # Nạp thêm exp09_resmlp_ple_tierB_pairs nếu đã chạy trước đó
    exp09_val_p = ARTIFACTS / "val_exp09_resmlp_ple_tierB_pairs.csv"
    exp09_sub_p = ARTIFACTS / "sub_exp09_resmlp_ple_tierB_pairs.csv"
    if exp09_val_p.exists() and exp09_sub_p.exists():
        v09 = pd.read_csv(exp09_val_p)
        s09 = pd.read_csv(exp09_sub_p)
        if np.array_equal(v09[ID_COLUMN].to_numpy(), val_ids):
            models_val["resmlp_ple_exp09"] = safe_logit(v09["pred_prob"].to_numpy())
            s09_re = pd.Series(s09[TARGET_COLUMN].to_numpy(), index=s09[ID_COLUMN]).reindex(test_ids).to_numpy()
            models_test["resmlp_ple_exp09"] = safe_logit(s09_re)

    # 6. Tổ hợp nhiều họ mô hình (kiểm chứng bằng Nested CV)
    print("\n=== 6. Multi-Family Logit Ensemble (Nested-CV Verified) ===", flush=True)
    t0 = time.perf_counter()
    names, weights, nested_auc = optimize_logit_weights_nested(y_va, models_val, seed=42)
    print("Learned Nested-CV Logit Blend Weights:", flush=True)
    for n, w in zip(names, weights):
        print(f"  {n:24s}: {w:.4f} (single Val AUC={roc_auc_score(y_va, models_val[n]):.6f})", flush=True)
    print(f"  --> Strictly Nested-CV Val AUC: {nested_auc:.6f}", flush=True)

    ens_va = sum(w * models_val[n] for n, w in zip(names, weights))
    ens_te = sum(w * models_test[n] for n, w in zip(names, weights))
    # Lưu kết quả tổ hợp và đối chiếu trung bình đều
    top_families = ["lgbm_glmoffset", "xgb_glmoffset", "catboost_glmoffset", "resmlp_glmoffset"]
    eq_va = np.mean([models_val[k] for k in top_families], axis=0)
    eq_te = np.mean([models_test[k] for k in top_families], axis=0)
    print(f"  --> Simple Equal-Weight 4-Family Logit Blend Val AUC: {roc_auc_score(y_va, eq_va):.6f}", flush=True)

    save_val_and_test(
        "m6_ensemble_logit_80_20", val_ids, y_va, ens_va, test_ids, sample_ids, ens_te,
        best_iter=len(names), elapsed=time.perf_counter() - t0,
        note=f"Nested-CV Logit Blend ({len(names)} models, nested_auc={nested_auc:.6f})"
    )
    save_val_and_test(
        "m6b_ensemble_equal4_80_20", val_ids, y_va, eq_va, test_ids, sample_ids, eq_te,
        best_iter=4, elapsed=0.5,
        note="Equal-Weight 4-Family Logit Blend (LGBM+XGB+CB+ResMLP)"
    )

    print("\n=== FINAL 80/20 BENCHMARK SUMMARY ===", flush=True)
    df = pd.read_csv(SUMMARY_CSV).sort_values("val_roc_auc", ascending=False)
    print(df[["experiment", "best_epoch", "val_roc_auc", "val_recall_05", "elapsed_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
