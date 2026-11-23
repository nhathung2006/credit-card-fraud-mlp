"""Quy trình kiểm chứng chéo 5-Fold (seed=42) cho 5 họ mô hình trực tiếp (LightGBM, XGBoost, CatBoost, ResMLP)."""

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

from fraud_mlp.data import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)
from fraud_mlp.runtime import TensorBatchLoader, atomic_output
from experiments.feature_pipeline import FLOAT_COLUMNS, _leading_digits

ARTIFACTS = PROJECT_ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "final_5fold"


def safe_logit(p: np.ndarray | float) -> np.ndarray | float:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def _extract_compact_v3_keys(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    keys: dict[str, np.ndarray] = {}
    freqs: dict[str, np.ndarray] = {}
    for col in NUMERIC_COLUMNS:
        c100 = np.rint(df[col].to_numpy(dtype=np.float64) * 100.0).astype(np.int64)
        keys[f"{col}__exact"] = c100
        freqs[f"{col}__freq"] = c100
    for col in FLOAT_COLUMNS:
        c100 = keys[f"{col}__exact"]
        ip = np.floor_divide(c100, 100)
        keys[f"{col}__int"] = ip
        keys[f"{col}__r1"] = np.floor_divide(c100, 10)
        keys[f"{col}__dec2"] = np.mod(c100, 100)
        keys[f"{col}__cent1"] = np.mod(c100, 10)
        keys[f"{col}__cent2"] = np.mod(np.floor_divide(c100, 10), 10)
        keys[f"{col}__unit"] = np.mod(ip, 10)
        keys[f"{col}__tens"] = np.mod(np.floor_divide(ip, 10), 10)
        keys[f"{col}__lead1"] = _leading_digits(ip, 1)
        keys[f"{col}__lead2"] = _leading_digits(ip, 2)
        freqs[f"{col}__int_freq"] = ip
        if col in ("amount_usd", "account_balance_usd"):
            keys[f"{col}__int_mod100"] = np.mod(ip, 100)
    return keys, freqs


def _lookup_sorted(k_arr: np.ndarray, uniq: np.ndarray, vals: np.ndarray, default: float = 0.0) -> np.ndarray:
    idx = np.searchsorted(uniq, k_arr)
    idx_s = np.minimum(idx, uniq.size - 1)
    return np.where(uniq[idx_s] == k_arr, vals[idx_s], default).astype(np.float32)


def build_base_v3(df: pd.DataFrame) -> pd.DataFrame:
    out = df[NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS].copy()
    for c in NUMERIC_COLUMNS:
        out[c] = out[c].astype(np.float32)
    for c in BINARY_COLUMNS:
        out[c] = out[c].astype(np.int8)
    for c in CATEGORICAL_COLUMNS:
        out[c] = out[c].astype("string").fillna("__MISSING__")
    amt = df["amount_usd"].to_numpy(dtype=np.float64)
    bal = df["account_balance_usd"].to_numpy(dtype=np.float64)
    hour = df["time_of_day_hour"].to_numpy(dtype=np.float64)
    dow = df["day_of_week"].to_numpy(dtype=np.float64)
    out["n_risk_flags"] = out[BINARY_COLUMNS].sum(axis=1).astype(np.int8)
    out["log_amt_minus_log_bal"] = (np.log1p(np.maximum(amt, 0.0)) - np.log1p(np.maximum(bal, 0.0))).astype(np.float32)
    out["amount_to_balance"] = (amt / (np.maximum(bal, 0.0) + 1.0)).astype(np.float32)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)
    return out


def build_fold_v3_features(
    base_all: pd.DataFrame,
    base_test: pd.DataFrame,
    all_keys: dict[str, np.ndarray],
    all_freqs: dict[str, np.ndarray],
    test_keys: dict[str, np.ndarray],
    test_freqs: dict[str, np.ndarray],
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tạo bộ đặc trưng v3 riêng cho từng fold huấn luyện (không rò rỉ nhãn)."""
    y_tr = y_all[tr_idx]
    n = len(y_tr)
    tot_y = float(y_tr.sum())
    prior = tot_y / n
    prior_l = float(safe_logit(prior))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_folds = [idx for _, idx in skf.split(np.zeros(n), y_tr)]

    tr_extra, va_extra, te_extra = {}, {}, {}

    for name, full_k in all_keys.items():
        k_tr = full_k[tr_idx]
        k_va = full_k[va_idx]
        k_te = test_keys[name]

        uniq, inv, tot_cnt = np.unique(k_tr, return_inverse=True, return_counts=True)
        tot_sum = np.bincount(inv, weights=y_tr, minlength=uniq.size)
        m = 5.0 if uniq.size < 2000 else 15.0

        oof = np.zeros(n, dtype=np.float32)
        for vi in inner_folds:
            fi = inv[vi]
            fc = np.bincount(fi, minlength=uniq.size)
            fs = np.bincount(fi, weights=y_tr[vi], minlength=uniq.size)
            oc = tot_cnt[fi] - fc[fi]
            os_ = tot_sum[fi] - fs[fi]
            fp = (tot_y - float(y_tr[vi].sum())) / (n - len(vi))
            p = (os_ + m * fp) / (oc + m)
            oof[vi] = (safe_logit(p) - safe_logit(fp)).astype(np.float32)

        full_woe = (safe_logit((tot_sum + m * prior) / (tot_cnt + m)) - prior_l).astype(np.float32)
        cname = f"te__{name}"
        tr_extra[cname] = oof
        va_extra[cname] = _lookup_sorted(k_va, uniq, full_woe, 0.0)
        te_extra[cname] = _lookup_sorted(k_te, uniq, full_woe, 0.0)

    for name, full_k in all_freqs.items():
        k_tr = full_k[tr_idx]
        k_va = full_k[va_idx]
        k_te = test_freqs[name]
        uniq, inv, cnt = np.unique(k_tr, return_inverse=True, return_counts=True)
        lc = np.log1p(cnt.astype(np.float64)).astype(np.float32)
        tr_extra[name] = lc[inv]
        va_extra[name] = _lookup_sorted(k_va, uniq, lc, 0.0)
        te_extra[name] = _lookup_sorted(k_te, uniq, lc, 0.0)

    X_tr = pd.concat([base_all.iloc[tr_idx].reset_index(drop=True), pd.DataFrame(tr_extra)], axis=1)
    X_va = pd.concat([base_all.iloc[va_idx].reset_index(drop=True), pd.DataFrame(va_extra)], axis=1)
    X_te = pd.concat([base_test.reset_index(drop=True), pd.DataFrame(te_extra)], axis=1)
    return X_tr, X_va, X_te


class WideDeepResMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, n_blocks: int = 3, dropout: float = 0.32, input_dropout: float = 0.12):
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


def _ple_encode_col(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    left = edges[:-1]
    width = np.maximum(edges[1:] - left, 1e-12)
    return np.clip((values[:, None] - left[None, :]) / width[None, :], 0.0, 1.0).astype(np.float32)


def train_resmlp_v3_fold(
    X_tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    X_va_df: pd.DataFrame,
    y_va: np.ndarray,
    X_te_df: pd.DataFrame,
    *,
    seed: int,
    ple_bins: int = 16,
    epochs: int = 18,
    patience: int = 4,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    num_cols = [c for c in X_tr_df.columns if c not in CATEGORICAL_COLUMNS]
    tr_num = X_tr_df[num_cols].to_numpy(dtype=np.float32)
    va_num = X_va_df[num_cols].to_numpy(dtype=np.float32)
    te_num = X_te_df[num_cols].to_numpy(dtype=np.float32)

    mean = np.nanmean(tr_num, axis=0)
    std = np.nanstd(tr_num, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    tr_s = np.clip((np.nan_to_num(tr_num, nan=0.0) - mean) / std, -8.0, 8.0).astype(np.float32)
    va_s = np.clip((np.nan_to_num(va_num, nan=0.0) - mean) / std, -8.0, 8.0).astype(np.float32)
    te_s = np.clip((np.nan_to_num(te_num, nan=0.0) - mean) / std, -8.0, 8.0).astype(np.float32)

    # Mã hóa tuyến tính từng khúc PLE-16 trên 13 cột số gốc (fit trên tập train)
    ple_tr_parts, ple_va_parts, ple_te_parts = [], [], []
    q = np.linspace(0.0, 1.0, ple_bins + 1)
    for col in NUMERIC_COLUMNS:
        v_tr = X_tr_df[col].to_numpy(dtype=np.float64)
        edges = np.unique(np.quantile(v_tr, q))
        if edges.size >= 2:
            ple_tr_parts.append(_ple_encode_col(v_tr, edges))
            ple_va_parts.append(_ple_encode_col(X_va_df[col].to_numpy(dtype=np.float64), edges))
            ple_te_parts.append(_ple_encode_col(X_te_df[col].to_numpy(dtype=np.float64), edges))

    # Mã hóa One-Hot cho 5 cột phân loại
    oh_tr_parts, oh_va_parts, oh_te_parts = [], [], []
    for col in CATEGORICAL_COLUMNS:
        cats = X_tr_df[col].cat.categories
        k = len(cats)
        for src, dst in [(X_tr_df[col], oh_tr_parts), (X_va_df[col], oh_va_parts), (X_te_df[col], oh_te_parts)]:
            codes = src.cat.codes.to_numpy()
            oh = np.zeros((len(codes), k), dtype=np.float32)
            valid = codes >= 0
            oh[np.arange(len(codes))[valid], codes[valid]] = 1.0
            dst.append(oh)

    X_tr = np.concatenate([tr_s, *ple_tr_parts, *oh_tr_parts], axis=1)
    X_va = np.concatenate([va_s, *ple_va_parts, *oh_va_parts], axis=1)
    X_te = np.concatenate([te_s, *ple_te_parts, *oh_te_parts], axis=1)

    torch.manual_seed(seed)
    model = WideDeepResMLP(X_tr.shape[1], hidden_dim=256, n_blocks=3, dropout=0.32, input_dropout=0.12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=2e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    tr_loader = TensorBatchLoader(
        torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)), batch_size=batch_size, shuffle=True
    )
    va_loader = TensorBatchLoader(torch.from_numpy(X_va), batch_size=batch_size, shuffle=False)
    te_loader = TensorBatchLoader(torch.from_numpy(X_te), batch_size=batch_size, shuffle=False)

    top_k: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    best_auc = -1.0
    best_ep = 0

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
        with torch.inference_mode():
            va_logits = np.concatenate([model(xb).numpy() for (xb,) in va_loader])
        auc = float(roc_auc_score(y_va, va_logits))
        state_copy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top_k.append((auc, ep, state_copy))
        top_k.sort(key=lambda t: t[0], reverse=True)
        top_k = top_k[:3]
        if auc > best_auc:
            best_auc = auc
            best_ep = ep
        if ep - best_ep >= patience:
            break

    va_snaps, te_snaps = [], []
    for _, _, st in top_k:
        model.load_state_dict(st)
        model.eval()
        with torch.inference_mode():
            va_snaps.append(np.concatenate([model(xb).numpy() for (xb,) in va_loader]))
            te_snaps.append(np.concatenate([model(xb).numpy() for (xb,) in te_loader]))
    return np.mean(va_snaps, axis=0).astype(np.float32), np.mean(te_snaps, axis=0).astype(np.float32)


def optimize_logit_weights_nested(y: np.ndarray, oof_dict: dict[str, np.ndarray], seed: int = 42):
    names = list(oof_dict.keys())
    M = np.column_stack([oof_dict[k] for k in names])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    nested_oof = np.zeros(len(y), dtype=np.float64)
    fold_weights = []

    def fit_w(X_sub, y_sub):
        k = X_sub.shape[1]
        def obj(params):
            w = np.exp(params[:k])
            w = w / w.sum()
            b = params[k]
            z = X_sub @ w + b
            return float(np.mean(np.logaddexp(0.0, z) - y_sub * z) + 5e-4 * np.sum((w - 1.0 / k) ** 2))
        res = minimize(obj, np.zeros(k + 1), method="BFGS")
        w = np.exp(res.x[:k])
        return w / w.sum()

    for tr_i, va_i in skf.split(M, y):
        w = fit_w(M[tr_i], y[tr_i])
        fold_weights.append(w)
        nested_oof[va_i] = M[va_i] @ w

    mean_w = np.mean(fold_weights, axis=0)
    mean_w = mean_w / mean_w.sum()
    return names, mean_w, float(roc_auc_score(y, nested_oof)), nested_oof


def save_submission(path: Path, test_ids: np.ndarray, sample_ids: pd.Index, test_logits: np.ndarray):
    probs = expit(test_logits)
    sub = pd.DataFrame({ID_COLUMN: sample_ids})
    sub[TARGET_COLUMN] = pd.Series(probs, index=test_ids).reindex(sample_ids).to_numpy()
    with atomic_output(path) as tmp:
        sub.to_csv(tmp, index=False)
    print(f"Saved {path} | Predictions: {len(sub):,} | mean_prob={probs.mean():.5f}", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe = OUT_DIR / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    t_all = time.perf_counter()
    data_dir = PROJECT_ROOT / "dataset" / "playground-series-s6e11"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_ids = pd.Index(pd.read_csv(data_dir / "sample_submission.csv", usecols=[ID_COLUMN])[ID_COLUMN])

    y_all = train_df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    train_ids = train_df[ID_COLUMN].to_numpy()
    test_ids = test_df[ID_COLUMN].to_numpy()

    print(f"Loaded train={len(train_df):,}, test={len(test_df):,}. Precomputing row-wise base & int64 keys...", flush=True)
    base_all = build_base_v3(train_df)
    base_test = build_base_v3(test_df)
    all_keys, all_freqs = _extract_compact_v3_keys(train_df)
    test_keys, test_freqs = _extract_compact_v3_keys(test_df)

    n_splits = 5
    outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    outer_folds = list(outer_skf.split(np.zeros(len(y_all)), y_all))

    model_names = [
        "lgbm_v3_l4",
        "lgbm_v3_l6_et",
        "xgb_v3_d3",
        "catboost_v3_d4",
        "resmlp_v3_ple",
    ]
    oof_logits = {m: np.zeros(len(y_all), dtype=np.float64) for m in model_names}
    test_logits = {m: np.zeros(len(test_df), dtype=np.float64) for m in model_names}
    fold_scores = {m: [] for m in model_names}

    for fold_id, (tr_idx, va_idx) in enumerate(outer_folds):
        t_f = time.perf_counter()
        print(f"\n{'=' * 72}\nOUTER FOLD {fold_id + 1}/{n_splits} (Train={len(tr_idx):,} | Val={len(va_idx):,})\n{'=' * 72}", flush=True)

        # Bỏ qua fold nếu cả 5 mô hình đã lưu đủ tệp cache
        fold_cache = {m: OUT_DIR / f"fold{fold_id}_{m}.npz" for m in model_names}
        if all(p.exists() for p in fold_cache.values()):
            for m in model_names:
                d = np.load(fold_cache[m])
                oof_logits[m][va_idx] = d["val"]
                test_logits[m] += d["test"] / n_splits
                sc = float(roc_auc_score(y_all[va_idx], d["val"]))
                fold_scores[m].append(sc)
                print(f"  [Cached] {m:16s} | Fold {fold_id+1} AUC: {sc:.6f}", flush=True)
            continue

        X_tr, X_va, X_te = build_fold_v3_features(
            base_all, base_test, all_keys, all_freqs, test_keys, test_freqs, y_all, tr_idx, va_idx, seed=42 + fold_id * 17
        )
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        # Bản sao chuỗi cho CatBoost
        X_tr_cb, X_va_cb, X_te_cb = X_tr.copy(), X_va.copy(), X_te.copy()
        for c in CATEGORICAL_COLUMNS:
            X_tr_cb[c] = X_tr_cb[c].astype(str)
            X_va_cb[c] = X_va_cb[c].astype(str)
            X_te_cb[c] = X_te_cb[c].astype(str)

            X_tr[c] = X_tr[c].astype("category")
            cats = X_tr[c].cat.categories
            X_va[c] = pd.Categorical(X_va[c], categories=cats)
            X_te[c] = pd.Categorical(X_te[c], categories=cats)
        print(f"  Fold {fold_id+1} features built in {time.perf_counter()-t_f:.1f}s | shape={X_tr.shape}", flush=True)

        # 1. LightGBM 4 lá (max_bin=1023)
        if not fold_cache["lgbm_v3_l4"].exists():
            t0 = time.perf_counter()
            dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=list(CATEGORICAL_COLUMNS), params={"max_bin": 1023})
            dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=list(CATEGORICAL_COLUMNS))
            bst = lgb.train(
                dict(
                    objective="binary", metric="auc", learning_rate=0.03,
                    num_leaves=4, max_depth=2, min_child_samples=50,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.65,
                    reg_alpha=0.0, reg_lambda=2.0, max_bin=1023,
                    n_jobs=-1, verbosity=-1, seed=42 + fold_id,
                ),
                dtr, num_boost_round=5000, valid_sets=[dva],
                callbacks=[lgb.early_stopping(150, verbose=False)],
            )
            va_l = bst.predict(X_va, raw_score=True, num_iteration=bst.best_iteration).astype(np.float32)
            te_l = bst.predict(X_te, raw_score=True, num_iteration=bst.best_iteration).astype(np.float32)
            np.savez_compressed(fold_cache["lgbm_v3_l4"], val=va_l, test=te_l)
            print(f"  lgbm_v3_l4       | Fold {fold_id+1} AUC: {roc_auc_score(y_va, va_l):.6f} | iter={bst.best_iteration} | {time.perf_counter()-t0:.1f}s", flush=True)

        # 2. LightGBM 6 lá Extra-Trees (max_bin=2047)
        if not fold_cache["lgbm_v3_l6_et"].exists():
            t0 = time.perf_counter()
            dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=list(CATEGORICAL_COLUMNS), params={"max_bin": 2047})
            dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=list(CATEGORICAL_COLUMNS))
            bst = lgb.train(
                dict(
                    objective="binary", metric="auc", learning_rate=0.025,
                    num_leaves=6, max_depth=3, min_child_samples=80, extra_trees=True,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.55,
                    reg_alpha=1.0, reg_lambda=5.0, max_bin=2047,
                    n_jobs=-1, verbosity=-1, seed=142 + fold_id,
                ),
                dtr, num_boost_round=6000, valid_sets=[dva],
                callbacks=[lgb.early_stopping(150, verbose=False)],
            )
            va_l = bst.predict(X_va, raw_score=True, num_iteration=bst.best_iteration).astype(np.float32)
            te_l = bst.predict(X_te, raw_score=True, num_iteration=bst.best_iteration).astype(np.float32)
            np.savez_compressed(fold_cache["lgbm_v3_l6_et"], val=va_l, test=te_l)
            print(f"  lgbm_v3_l6_et    | Fold {fold_id+1} AUC: {roc_auc_score(y_va, va_l):.6f} | iter={bst.best_iteration} | {time.perf_counter()-t0:.1f}s", flush=True)

        # 3. XGBoost độ sâu 3 (max_bin=1024)
        if not fold_cache["xgb_v3_d3"].exists():
            t0 = time.perf_counter()
            dtr_x = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
            dva_x = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
            dte_x = xgb.DMatrix(X_te, enable_categorical=True)
            bst_x = xgb.train(
                dict(
                    objective="binary:logistic", eval_metric="auc", tree_method="hist",
                    max_bin=1024, learning_rate=0.03, max_depth=3, min_child_weight=35.0,
                    subsample=0.8, colsample_bytree=0.65, reg_alpha=0.5, reg_lambda=3.0,
                    nthread=-1, seed=42 + fold_id,
                ),
                dtr_x, num_boost_round=5000, evals=[(dva_x, "val")],
                early_stopping_rounds=150, verbose_eval=False,
            )
            va_l = bst_x.predict(dva_x, output_margin=True, iteration_range=(0, bst_x.best_iteration + 1)).astype(np.float32)
            te_l = bst_x.predict(dte_x, output_margin=True, iteration_range=(0, bst_x.best_iteration + 1)).astype(np.float32)
            np.savez_compressed(fold_cache["xgb_v3_d3"], val=va_l, test=te_l)
            print(f"  xgb_v3_d3        | Fold {fold_id+1} AUC: {roc_auc_score(y_va, va_l):.6f} | iter={bst_x.best_iteration} | {time.perf_counter()-t0:.1f}s", flush=True)

        # 4. CatBoost đối xứng độ sâu 4
        if not fold_cache["catboost_v3_d4"].exists():
            t0 = time.perf_counter()
            cat_idx = [X_tr_cb.columns.get_loc(c) for c in CATEGORICAL_COLUMNS]
            tr_pool = Pool(X_tr_cb, label=y_tr, cat_features=cat_idx)
            va_pool = Pool(X_va_cb, label=y_va, cat_features=cat_idx)
            te_pool = Pool(X_te_cb, cat_features=cat_idx)
            cb = CatBoostClassifier(
                iterations=4500, learning_rate=0.04, depth=4, l2_leaf_reg=8.0,
                one_hot_max_size=16, rsm=0.6, eval_metric="AUC",
                random_seed=42 + fold_id, thread_count=-1, verbose=False,
            )
            cb.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=150, use_best_model=True)
            va_l = cb.predict(va_pool, prediction_type="RawFormulaVal").astype(np.float32)
            te_l = cb.predict(te_pool, prediction_type="RawFormulaVal").astype(np.float32)
            np.savez_compressed(fold_cache["catboost_v3_d4"], val=va_l, test=te_l)
            print(f"  catboost_v3_d4   | Fold {fold_id+1} AUC: {roc_auc_score(y_va, va_l):.6f} | iter={cb.get_best_iteration()} | {time.perf_counter()-t0:.1f}s", flush=True)

        # 5. Wide & Deep PLE Residual MLP (trung bình 3 epoch tốt nhất)
        if not fold_cache["resmlp_v3_ple"].exists():
            t0 = time.perf_counter()
            va_l, te_l = train_resmlp_v3_fold(X_tr, y_tr, X_va, y_va, X_te, seed=42 + fold_id)
            np.savez_compressed(fold_cache["resmlp_v3_ple"], val=va_l, test=te_l)
            print(f"  resmlp_v3_ple    | Fold {fold_id+1} AUC: {roc_auc_score(y_va, va_l):.6f} | {time.perf_counter()-t0:.1f}s", flush=True)

        for m in model_names:
            d = np.load(fold_cache[m])
            oof_logits[m][va_idx] = d["val"]
            test_logits[m] += d["test"] / n_splits
            fold_scores[m].append(float(roc_auc_score(y_va, d["val"])))

    # Tổng hợp OOF 5-Fold (699.960 dòng) và trộn trọng số Nested CV
    print(f"\n{'=' * 72}\nFINAL 5-FOLD STRATIFIED CV SUMMARY (ALL {len(y_all):,} ROWS)\n{'=' * 72}", flush=True)
    for m in model_names:
        full_auc = float(roc_auc_score(y_all, oof_logits[m]))
        rec05 = float(recall_score(y_all, expit(oof_logits[m]) >= 0.5, zero_division=0))
        mean_f = float(np.mean(fold_scores[m]))
        std_f = float(np.std(fold_scores[m]))
        print(
            f"  {m:16s} | OOF AUC: {full_auc:.6f} | Fold Mean±Std: {mean_f:.6f} ± {std_f:.6f} | Recall@0.5: {rec05:.6f}",
            flush=True,
        )
        # Lưu tệp OOF có cột id cho từng mô hình
        with atomic_output(OUT_DIR / f"oof_5fold_{m}.csv") as tmp:
            pd.DataFrame({
                ID_COLUMN: train_ids,
                TARGET_COLUMN: y_all,
                "pred_prob": expit(oof_logits[m]),
            }).to_csv(tmp, index=False)

    names, weights, nested_auc, nested_oof = optimize_logit_weights_nested(y_all, oof_logits, seed=42)
    print("\nLearned Nested-CV Logit Blend Weights (over 699,960 OOF rows):", flush=True)
    for n_m, w in zip(names, weights):
        print(f"  {n_m:16s}: {w:.4f}", flush=True)

    ens_oof = sum(w * oof_logits[n_m] for n_m, w in zip(names, weights))
    ens_test = sum(w * test_logits[n_m] for n_m, w in zip(names, weights))
    ens_auc = float(roc_auc_score(y_all, ens_oof))
    ens_rec = float(recall_score(y_all, expit(ens_oof) >= 0.5, zero_division=0))
    print(
        f"\n>>> FINAL 5-FOLD ENSEMBLE | Strictly Nested-CV OOF AUC: {nested_auc:.6f}"
        f" | Full OOF AUC: {ens_auc:.6f} | Recall@0.5: {ens_rec:.6f}",
        flush=True,
    )

    with atomic_output(OUT_DIR / "oof_5fold_ensemble.csv") as tmp:
        pd.DataFrame({
            ID_COLUMN: train_ids,
            TARGET_COLUMN: y_all,
            "pred_prob": expit(ens_oof),
        }).to_csv(tmp, index=False)

    save_submission(ARTIFACTS / "sub_final_5fold_ensemble.csv", test_ids, sample_ids, ens_test)
    save_submission(ARTIFACTS / "sub_final_5fold_resmlp.csv", test_ids, sample_ids, test_logits["resmlp_v3_ple"])
    save_submission(ARTIFACTS / "sub_final_5fold_lgbm_l4.csv", test_ids, sample_ids, test_logits["lgbm_v3_l4"])
    print(f"\nTotal 5-Fold pipeline elapsed: {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
