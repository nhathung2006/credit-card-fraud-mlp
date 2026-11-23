"""Tổ hợp cuối trên kiểm chứng chéo 5-Fold (Nested CV) ghép các mô hình cục bộ và OOF từ Kaggle.

Các bước chính:
  1. Căn chuẩn trung bình/độ lệch chuẩn logit từng fold (không dùng nhãn).
  2. Lọc giữ các mô hình đạt OOF 5-Fold >= 0.9450.
  3. Ghép 75% bản học lại 100% dữ liệu với 25% trung bình 5-Fold cho nhánh cây/GLM.
  4. Tối ưu trọng số cộng dương có điều chuẩn L2 bằng Nested 5-Fold CV.
  5. Hiệu chuẩn vùng merchant_risk_score < 13.1 nếu cả 5 fold đều cải thiện.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fraud_mlp.data import ID_COLUMN, TARGET_COLUMN
from fraud_mlp.runtime import atomic_output

ARTIFACTS = PROJECT_ROOT / "artifacts"
FINAL_DIR = ARTIFACTS / "final_5fold"

V3_MODELS = ["lgbm_v3_l4", "lgbm_v3_l6_et", "xgb_v3_d3", "catboost_v3_d4", "resmlp_v3_ple"]
V4_MODELS = ["v4_glm", "v4_lgb_l3_glm", "v4_lgb_l4_glm", "v4_lgb_l5_glm", "v4_cb_d4_glm", "v4_xgb_l4_glm"]
TREE_GLM_SET = set(V3_MODELS[:4] + V4_MODELS)


def safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def find_file(names: list[str]) -> Path | None:
    for d in (FINAL_DIR, PROJECT_ROOT.parent, ARTIFACTS):
        for n in names:
            p = d / n
            if p.exists() and p.stat().st_size > 1_000_000:
                return p
    return None


def align_fold_logits(
    oof: np.ndarray,
    test: np.ndarray,
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Căn chuẩn phân phối logit từng fold không giám sát (không dùng nhãn)."""
    g_mu = float(oof.mean())
    g_sd = max(float(oof.std()), 1e-6)
    al_oof = np.zeros_like(oof, dtype=np.float64)
    for _, va_i in outer_folds:
        v = oof[va_i]
        al_oof[va_i] = (v - float(v.mean())) / max(float(v.std()), 1e-6) * g_sd + g_mu
    al_test = (test - float(test.mean())) / max(float(test.std()), 1e-6) * g_sd + g_mu
    return al_oof, al_test


def fit_simplex_weights(X: np.ndarray, y: np.ndarray, reg: float = 2e-4) -> np.ndarray:
    k = X.shape[1]

    def obj(params):
        w = np.exp(params[:k] - np.max(params[:k]))
        w /= w.sum()
        z = np.exp(params[k]) * (X @ w) + params[k + 1]
        return float(np.mean(np.logaddexp(0.0, z) - y * z) + reg * np.sum((w - 1.0 / k) ** 2))

    res = minimize(obj, np.zeros(k + 2), method="BFGS")
    w = np.exp(res.x[:k] - np.max(res.x[:k]))
    return w / w.sum()


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    probe = ARTIFACTS / ".write_probe_champ"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    data_dir = PROJECT_ROOT / "dataset" / "playground-series-s6e11"
    train_df = pd.read_csv(data_dir / "train.csv", usecols=[ID_COLUMN, TARGET_COLUMN, "merchant_risk_score"])
    test_df = pd.read_csv(data_dir / "test.csv", usecols=[ID_COLUMN, "merchant_risk_score"])
    sample_ids = pd.Index(pd.read_csv(data_dir / "sample_submission.csv", usecols=[ID_COLUMN])[ID_COLUMN])

    y_all = train_df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    train_ids = train_df[ID_COLUMN].to_numpy()
    test_ids = test_df[ID_COLUMN].to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_folds = list(skf.split(np.zeros(len(y_all)), y_all))

    raw_oof: dict[str, np.ndarray] = {}
    raw_test: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    # 1. Tải mô hình cục bộ v3, v4 và bản học lại 100% dữ liệu
    for m in [*V3_MODELS, *V4_MODELS]:
        if all((FINAL_DIR / f"fold{f}_{m}.npz").exists() for f in range(5)):
            o = np.zeros(len(y_all), dtype=np.float64)
            t = np.zeros(len(test_df), dtype=np.float64)
            for f_id, (_, va_i) in enumerate(outer_folds):
                d = np.load(FINAL_DIR / f"fold{f_id}_{m}.npz")
                o[va_i] = d["val"].astype(np.float64)
                t += d["test"].astype(np.float64) / 5.0
            refit_p = FINAL_DIR / f"refit_full_{m}.npz"
            if refit_p.exists():
                t = 0.75 * np.load(refit_p)["test"].astype(np.float64) + 0.25 * t
            raw_oof[m] = o
            raw_test[m] = t
            counts[m] = 5

    # 2. Tải OOF/test của TabM + Swap-Noise DAE
    tabm_o = find_file(["oof_kaggle_tabm_5fold.csv"])
    tabm_s = find_file(["sub_kaggle_tabm_5fold.csv"])
    if tabm_o and tabm_s:
        df_o = pd.read_csv(tabm_o)
        df_s = pd.read_csv(tabm_s)
        assert np.array_equal(df_o[ID_COLUMN].to_numpy(), train_ids)
        raw_oof["kaggle_tabm_dae"] = 0.5 * df_o["logit_core"].to_numpy(dtype=np.float64) + 0.5 * df_o["logit_dae"].to_numpy(dtype=np.float64)
        tp = pd.Series(df_s[TARGET_COLUMN].to_numpy(), index=df_s[ID_COLUMN]).reindex(test_ids).to_numpy()
        assert not np.isnan(tp).any()
        raw_test["kaggle_tabm_dae"] = safe_logit(tp)
        counts["kaggle_tabm_dae"] = 10

    # 3. Tải OOF/test của ModernNCA và FT-Transformer
    s2_o = find_file(["oof_kaggle_sota2025_v2_5fold.csv"])
    s2_d = find_file(["sub_kaggle_sota2025_v2_details.csv"])
    if s2_o and s2_d:
        df_o = pd.read_csv(s2_o)
        df_d = pd.read_csv(s2_d)
        assert np.array_equal(df_o[ID_COLUMN].to_numpy(), train_ids)
        for col in [c for c in df_o.columns if c.startswith("logit_") and c in df_d.columns]:
            mn = f"k2_{col.replace('logit_', '')}"
            raw_oof[mn] = df_o[col].to_numpy(dtype=np.float64)
            tv = pd.Series(df_d[col].to_numpy(dtype=np.float64), index=df_d[ID_COLUMN]).reindex(test_ids).to_numpy()
            assert not np.isnan(tv).any()
            raw_test[mn] = tv
            counts[mn] = 5

    # 4. Tải thêm RealMLP (use_ls=False) nếu có tệp
    s3_o = find_file(["oof_kaggle_realmlp_clean_5fold.csv"])
    s3_d = find_file(["sub_kaggle_realmlp_clean_details.csv", "sub_kaggle_realmlp_clean_5fold.csv"])
    if s3_o and s3_d:
        df_o = pd.read_csv(s3_o)
        df_d = pd.read_csv(s3_d)
        assert np.array_equal(df_o[ID_COLUMN].to_numpy(), train_ids)
        o_col = "logit_realmlp" if "logit_realmlp" in df_o.columns else "pred_prob"
        d_col = "logit_realmlp" if "logit_realmlp" in df_d.columns else TARGET_COLUMN
        raw_oof["k3_realmlp_clean"] = df_o[o_col].to_numpy(dtype=np.float64) if o_col.startswith("logit_") else safe_logit(df_o[o_col].to_numpy(dtype=np.float64))
        tv = pd.Series(df_d[d_col].to_numpy(dtype=np.float64), index=df_d[ID_COLUMN]).reindex(test_ids).to_numpy()
        raw_test["k3_realmlp_clean"] = tv if d_col.startswith("logit_") else safe_logit(tv)
        counts["k3_realmlp_clean"] = 5

    # Căn thang không giám sát giữa 5 fold và lọc mô hình đạt OOF >= 0.9450
    kept_names: list[str] = []
    al_oof: dict[str, np.ndarray] = {}
    al_test: dict[str, np.ndarray] = {}
    print("=== QUALITY GATE & FOLD ALIGNMENT ===", flush=True)
    for m in raw_oof:
        ao, at = align_fold_logits(raw_oof[m], raw_test[m], outer_folds)
        auc_raw = roc_auc_score(y_all, raw_oof[m])
        auc_al = roc_auc_score(y_all, ao)
        status = "KEEP" if auc_al >= 0.9450 else "DROP (<0.9450)"
        print(f"  {m:20s} ({counts[m]:2d}m) | raw={auc_raw:.6f} -> aligned={auc_al:.6f} | {status}", flush=True)
        if auc_al >= 0.9450:
            kept_names.append(m)
            al_oof[m] = ao
            al_test[m] = at

    total_m = sum(counts[m] for m in kept_names)
    M_oof = np.column_stack([al_oof[m] for m in kept_names])
    M_test = np.column_stack([al_test[m] for m in kept_names])

    oof_ens = np.zeros(len(y_all), dtype=np.float64)
    test_ens = np.zeros(len(test_df), dtype=np.float64)
    w_folds = []
    for tr_i, va_i in outer_folds:
        w = fit_simplex_weights(M_oof[tr_i], y_all[tr_i], reg=2e-4)
        w_folds.append(w)
        oof_ens[va_i] = M_oof[va_i] @ w
        test_ens += (M_test @ w) / 5.0

    mean_w = np.mean(w_folds, axis=0)
    mean_w /= mean_w.sum()
    dl_share = sum(w for m, w in zip(kept_names, mean_w) if m not in TREE_GLM_SET)
    auc_raw_ens = roc_auc_score(y_all, oof_ens)

    print(f"\n=== FINAL CHAMPION ({len(kept_names)} elite families = {total_m} models, DL share={dl_share*100:.1f}%) ===", flush=True)
    for m, w in zip(kept_names, mean_w):
        print(f"  {m:20s}: {w:.4f}")
    print(f"  --> Strictly Nested-CV OOF AUC (pre-zone1): {auc_raw_ens:.6f}", flush=True)

    # Kiểm tra hiệu chuẩn vùng merchant_risk_score < 13.1 trên từng fold
    tr_zone = (train_df["merchant_risk_score"] < 13.1).to_numpy()
    te_zone = (test_df["merchant_risk_score"] < 13.1).to_numpy()
    oof_p = expit(oof_ens)
    test_p = expit(test_ens)
    oof_pz = np.where(tr_zone, oof_p * 0.01, oof_p)
    test_pz = np.where(te_zone, test_p * 0.01, test_p)

    folds_up = 0
    for f_id, (_, va_i) in enumerate(outer_folds):
        a0 = roc_auc_score(y_all[va_i], oof_p[va_i])
        a1 = roc_auc_score(y_all[va_i], oof_pz[va_i])
        if a1 > a0:
            folds_up += 1
        print(f"  Fold {f_id+1} zone1: {a0:.6f} -> {a1:.6f} ({a1 - a0:+.6f})", flush=True)

    use_oof_p = oof_pz if folds_up == 5 else oof_p
    use_test_p = test_pz if folds_up == 5 else test_p
    final_auc = roc_auc_score(y_all, use_oof_p)
    final_rec = recall_score(y_all, use_oof_p >= 0.5, zero_division=0)

    out_sub = ARTIFACTS / f"sub_final_champion_{total_m}models_aligned.csv"
    out_oof = FINAL_DIR / f"oof_final_champion_{total_m}models_aligned.csv"

    with atomic_output(out_oof) as tmp:
        pd.DataFrame({ID_COLUMN: train_ids, TARGET_COLUMN: y_all, "pred_prob": use_oof_p}).to_csv(tmp, index=False)

    sub_df = pd.DataFrame({ID_COLUMN: sample_ids})
    sub_df[TARGET_COLUMN] = pd.Series(use_test_p, index=test_ids).reindex(sample_ids).to_numpy()
    assert len(sub_df) == len(sample_ids) and not sub_df[TARGET_COLUMN].isna().any()
    with atomic_output(out_sub) as tmp:
        sub_df.to_csv(tmp, index=False)

    print(
        f"\n>>> FINAL CHAMPION ({total_m} MODELS, FOLD-ALIGNED) | OOF AUC: {final_auc:.6f}"
        f" | Recall@0.5: {final_rec:.6f} | Zone1: {folds_up==5} | {time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    print(f"Saved: {out_sub} ({len(sub_df):,} rows)")
    print(f"Saved: {out_oof} ({len(y_all):,} rows)")
    print("First 5 rows:\n", sub_df.head(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
