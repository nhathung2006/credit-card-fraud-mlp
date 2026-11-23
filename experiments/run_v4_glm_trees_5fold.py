"""Quy trình 5-Fold nhánh nền GLM và cây nông học phần dư (v4_glm, LightGBM 3-5 lá, CatBoost, XGBoost)."""

from __future__ import annotations

from pathlib import Path
import sys
import time

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fraud_mlp.data import BINARY_COLUMNS, CATEGORICAL_COLUMNS, ID_COLUMN, NUMERIC_COLUMNS, TARGET_COLUMN
from fraud_mlp.runtime import atomic_output
from experiments.feature_pipeline import FLOAT_COLUMNS, _leading_digits

ARTIFACTS = PROJECT_ROOT / "artifacts"
FINAL_DIR = ARTIFACTS / "final_5fold"

V4_MODELS = [
    "v4_glm",
    "v4_lgb_l3_glm",
    "v4_lgb_l4_glm",
    "v4_lgb_l5_glm",
    "v4_cb_d4_glm",
    "v4_xgb_l4_glm",
]


def safe_logit(p: np.ndarray | float) -> np.ndarray | float:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def build_plain_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS].copy()
    for c in NUMERIC_COLUMNS:
        out[c] = out[c].astype(np.float32)
    for c in BINARY_COLUMNS:
        out[c] = out[c].astype(np.int8)
    for c in CATEGORICAL_COLUMNS:
        out[c] = out[c].astype("string")
    for c in FLOAT_COLUMNS:
        c100 = np.rint(df[c].to_numpy(dtype=np.float64) * 100.0).astype(np.int64)
        ip = np.floor_divide(c100, 100)
        out[f"{c}__dec2_plain"] = np.mod(c100, 100).astype(np.int16)
        out[f"{c}__cent1_plain"] = np.mod(c100, 10).astype(np.int8)
        out[f"{c}__cent2_plain"] = np.mod(np.floor_divide(c100, 10), 10).astype(np.int8)
        out[f"{c}__unit_plain"] = np.mod(ip, 10).astype(np.int8)
        out[f"{c}__tens_plain"] = np.mod(np.floor_divide(ip, 10), 10).astype(np.int8)
        out[f"{c}__lead1_plain"] = _leading_digits(ip, 1).astype(np.int8)
    return out


def extract_encoding_keys(df: pd.DataFrame) -> dict[str, np.ndarray]:
    keys: dict[str, np.ndarray] = {}
    for c in NUMERIC_COLUMNS:
        c100 = np.rint(df[c].to_numpy(dtype=np.float64) * 100.0).astype(np.int64)
        keys[f"{c}__exact"] = c100
    for c in FLOAT_COLUMNS:
        c100 = keys[f"{c}__exact"]
        ip = np.floor_divide(c100, 100)
        keys[f"{c}__int"] = ip
        keys[f"{c}__r1"] = np.floor_divide(c100, 10)
        keys[f"{c}__dec2"] = np.mod(c100, 100)
        keys[f"{c}__cent1"] = np.mod(c100, 10)
        keys[f"{c}__cent2"] = np.mod(np.floor_divide(c100, 10), 10)
        keys[f"{c}__unit"] = np.mod(ip, 10)
        keys[f"{c}__tens"] = np.mod(np.floor_divide(ip, 10), 10)
        keys[f"{c}__lead1"] = _leading_digits(ip, 1)
        keys[f"{c}__lead2"] = _leading_digits(ip, 2)
        if c in ("amount_usd", "account_balance_usd"):
            keys[f"{c}__int_mod100"] = np.mod(ip, 100)
    return keys


def build_fold_v4_data(
    tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    fold_id: int,
    seeds: tuple[int, ...] = (42, 102, 202),
    c_glm: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    tr_k = extract_encoding_keys(tr_df)
    va_k = extract_encoding_keys(va_df)
    te_k = extract_encoding_keys(te_df)

    n = len(y_tr)
    tot_y = float(y_tr.sum())
    prior = tot_y / n
    prior_l = float(safe_logit(prior))

    inner_fold_sets = [
        [vi for _, vi in StratifiedKFold(n_splits=5, shuffle=True, random_state=sd + fold_id).split(np.zeros(n), y_tr)]
        for sd in seeds
    ]

    te_tr_cols: dict[str, np.ndarray] = {}
    te_va_cols: dict[str, np.ndarray] = {}
    te_te_cols: dict[str, np.ndarray] = {}

    for name, k_tr in tr_k.items():
        uniq, inv, tot_cnt = np.unique(k_tr, return_inverse=True, return_counts=True)
        tot_sum = np.bincount(inv, weights=y_tr, minlength=uniq.size)
        m = 5.0 if uniq.size < 2000 else 15.0

        oof_acc = np.zeros(n, dtype=np.float64)
        for f_set in inner_fold_sets:
            for vi in f_set:
                fi = inv[vi]
                fc = np.bincount(fi, minlength=uniq.size)
                fs = np.bincount(fi, weights=y_tr[vi], minlength=uniq.size)
                oc = tot_cnt[fi] - fc[fi]
                os_ = tot_sum[fi] - fs[fi]
                fp = (tot_y - float(y_tr[vi].sum())) / (n - len(vi))
                p = (os_ + m * fp) / (oc + m)
                oof_acc[vi] += (safe_logit(p) - safe_logit(fp)) / len(seeds)
        te_tr_cols[f"te__{name}"] = oof_acc.astype(np.float32)

        full_woe = (safe_logit((tot_sum + m * prior) / (tot_cnt + m)) - prior_l).astype(np.float32)

        for dest, src_k in ((te_va_cols, va_k[name]), (te_te_cols, te_k[name])):
            idx = np.searchsorted(uniq, src_k)
            idx_s = np.minimum(idx, uniq.size - 1)
            dest[f"te__{name}"] = np.where(uniq[idx_s] == src_k, full_woe[idx_s], 0.0).astype(np.float32)

    X_glm_tr = pd.concat([
        pd.DataFrame(te_tr_cols),
        tr_df[BINARY_COLUMNS].astype(np.float32),
        pd.get_dummies(tr_df[CATEGORICAL_COLUMNS], dtype=np.float32),
    ], axis=1)
    X_glm_va = pd.concat([
        pd.DataFrame(te_va_cols),
        va_df[BINARY_COLUMNS].astype(np.float32),
        pd.get_dummies(va_df[CATEGORICAL_COLUMNS], dtype=np.float32),
    ], axis=1).reindex(columns=X_glm_tr.columns, fill_value=0.0)
    X_glm_te = pd.concat([
        pd.DataFrame(te_te_cols),
        te_df[BINARY_COLUMNS].astype(np.float32),
        pd.get_dummies(te_df[CATEGORICAL_COLUMNS], dtype=np.float32),
    ], axis=1).reindex(columns=X_glm_tr.columns, fill_value=0.0)

    glm_oof = np.zeros(n, dtype=np.float32)
    X_np = X_glm_tr.to_numpy()
    for vi in inner_fold_sets[0]:
        ti_mask = np.ones(n, dtype=bool)
        ti_mask[vi] = False
        clf = LogisticRegression(C=c_glm, max_iter=200)
        clf.fit(X_np[ti_mask], y_tr[ti_mask])
        glm_oof[vi] = clf.decision_function(X_np[vi])

    clf_full = LogisticRegression(C=c_glm, max_iter=200)
    clf_full.fit(X_np, y_tr)
    glm_va = clf_full.decision_function(X_glm_va.to_numpy()).astype(np.float32)
    glm_te = clf_full.decision_function(X_glm_te.to_numpy()).astype(np.float32)

    X_tr = pd.concat([build_plain_features(tr_df), pd.DataFrame(te_tr_cols)], axis=1)
    X_va = pd.concat([build_plain_features(va_df), pd.DataFrame(te_va_cols)], axis=1)
    X_te = pd.concat([build_plain_features(te_df), pd.DataFrame(te_te_cols)], axis=1)
    for c in CATEGORICAL_COLUMNS:
        X_tr[c] = X_tr[c].astype("category")
        cats = X_tr[c].cat.categories
        X_va[c] = pd.Categorical(X_va[c], categories=cats)
        X_te[c] = pd.Categorical(X_te[c], categories=cats)

    return X_tr, X_va, X_te, glm_oof, glm_va, glm_te


def save_fold_cache(path: Path, val_logits: np.ndarray, test_logits: np.ndarray) -> None:
    with atomic_output(path) as tmp:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, val=val_logits.astype(np.float32), test=test_logits.astype(np.float32))


def assert_sane_logits(name: str, val_l: np.ndarray, test_l: np.ndarray, glm_va: np.ndarray, glm_te: np.ndarray) -> None:
    assert np.isfinite(val_l).all() and np.isfinite(test_l).all(), f"{name}: non-finite logits!"
    assert abs(float(np.mean(val_l)) - float(np.mean(glm_va))) < 0.8, f"{name}: val logit mean drift!"
    assert abs(float(np.mean(test_l)) - float(np.mean(glm_te))) < 0.8, f"{name}: test logit mean drift!"


def main() -> None:
    t_start = time.perf_counter()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    probe = FINAL_DIR / ".write_probe_v4"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    data_dir = PROJECT_ROOT / "dataset" / "playground-series-s6e11"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    y_all = train_df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    train_ids = train_df[ID_COLUMN].to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_folds = list(skf.split(np.zeros(len(y_all)), y_all))

    oof_store = {m: np.zeros(len(y_all), dtype=np.float64) for m in V4_MODELS}
    test_store = {m: np.zeros(len(test_df), dtype=np.float64) for m in V4_MODELS}

    for fold_id, (tr_idx, va_idx) in enumerate(outer_folds):
        t_fold = time.perf_counter()
        missing = [m for m in V4_MODELS if not (FINAL_DIR / f"fold{fold_id}_{m}.npz").exists()]
        if not missing:
            print(f"\n[Fold {fold_id + 1}/5] All 6 v4 caches present, loading...", flush=True)
            for m in V4_MODELS:
                d = np.load(FINAL_DIR / f"fold{fold_id}_{m}.npz")
                oof_store[m][va_idx] = d["val"].astype(np.float64)
                test_store[m] += d["test"].astype(np.float64) / 5.0
            continue

        print(f"\n==================== FOLD {fold_id + 1}/5 (missing: {missing}) ====================", flush=True)
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        t0 = time.perf_counter()
        X_tr, X_va, X_te, glm_oof, glm_va, glm_te = build_fold_v4_data(
            tr_df, y_tr, va_df, test_df, fold_id=fold_id, seeds=(42, 102, 202), c_glm=0.2
        )
        print(
            f"  [Data+GLM] {X_tr.shape[1]} cols | GLM Inner OOF: {roc_auc_score(y_tr, glm_oof):.6f}"
            f" | Fold {fold_id+1} Val: {roc_auc_score(y_va, glm_va):.6f} | {time.perf_counter()-t0:.1f}s",
            flush=True,
        )

        # 1. Hồi quy Logistic nền (v4_glm)
        p_glm = FINAL_DIR / f"fold{fold_id}_v4_glm.npz"
        if not p_glm.exists():
            save_fold_cache(p_glm, glm_va, glm_te)

        # 2-4. LightGBM 3, 4 và 5 lá học phần dư từ GLM
        lgb_specs = [
            ("v4_lgb_l3_glm", 3, 2, 42 + fold_id),
            ("v4_lgb_l4_glm", 4, 3, 142 + fold_id),
            ("v4_lgb_l5_glm", 5, 3, 242 + fold_id),
        ]
        if any(not (FINAL_DIR / f"fold{fold_id}_{m}.npz").exists() for m, _, _, _ in lgb_specs):
            dtr = lgb.Dataset(
                X_tr, label=y_tr, init_score=glm_oof,
                categorical_feature=list(CATEGORICAL_COLUMNS), params={"max_bin": 1023}, free_raw_data=False
            )
            dva = lgb.Dataset(
                X_va, label=y_va, init_score=glm_va, reference=dtr,
                categorical_feature=list(CATEGORICAL_COLUMNS), free_raw_data=False
            )
            for m_name, leaves, max_d, seed in lgb_specs:
                cache_p = FINAL_DIR / f"fold{fold_id}_{m_name}.npz"
                if cache_p.exists():
                    continue
                t1 = time.perf_counter()
                bst = lgb.train(
                    dict(
                        objective="binary", metric="auc", learning_rate=0.03,
                        num_leaves=leaves, max_depth=max_d, min_child_samples=80,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                        reg_alpha=0.5, reg_lambda=5.0, max_bin=1023,
                        n_jobs=-1, verbosity=-1, seed=seed,
                    ),
                    dtr, num_boost_round=4000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(120, verbose=False)],
                )
                va_l = glm_va + bst.predict(X_va, raw_score=True, num_iteration=bst.best_iteration)
                te_l = glm_te + bst.predict(X_te, raw_score=True, num_iteration=bst.best_iteration)
                assert_sane_logits(m_name, va_l, te_l, glm_va, glm_te)
                save_fold_cache(cache_p, va_l, te_l)
                print(
                    f"  [{m_name:14s}] Val AUC: {roc_auc_score(y_va, va_l):.6f}"
                    f" | iter={bst.best_iteration:4d} | {time.perf_counter()-t1:.1f}s",
                    flush=True,
                )

        # 5. CatBoost đối xứng độ sâu 4 học phần dư từ GLM
        p_cb = FINAL_DIR / f"fold{fold_id}_v4_cb_d4_glm.npz"
        if not p_cb.exists():
            t1 = time.perf_counter()
            X_tr_cb = X_tr.copy()
            X_va_cb = X_va.copy()
            X_te_cb = X_te.copy()
            for c in CATEGORICAL_COLUMNS:
                X_tr_cb[c] = X_tr_cb[c].astype(str)
                X_va_cb[c] = X_va_cb[c].astype(str)
                X_te_cb[c] = X_te_cb[c].astype(str)
            ptr = Pool(X_tr_cb, label=y_tr, baseline=glm_oof, cat_features=list(CATEGORICAL_COLUMNS))
            pva = Pool(X_va_cb, label=y_va, baseline=glm_va, cat_features=list(CATEGORICAL_COLUMNS))
            pte = Pool(X_te_cb, baseline=glm_te, cat_features=list(CATEGORICAL_COLUMNS))
            cb = CatBoostClassifier(
                iterations=4000, learning_rate=0.03, depth=4, l2_leaf_reg=15.0,
                rsm=0.5, one_hot_max_size=16, eval_metric="AUC", random_seed=342 + fold_id,
                thread_count=-1, allow_writing_files=False, verbose=False,
            )
            cb.fit(ptr, eval_set=pva, early_stopping_rounds=150, use_best_model=True)
            va_l = cb.predict(pva, prediction_type="RawFormulaVal")
            te_l = cb.predict(pte, prediction_type="RawFormulaVal")
            assert_sane_logits("v4_cb_d4_glm", va_l, te_l, glm_va, glm_te)
            save_fold_cache(p_cb, va_l, te_l)
            print(
                f"  [{'v4_cb_d4_glm':14s}] Val AUC: {roc_auc_score(y_va, va_l):.6f}"
                f" | iter={cb.get_best_iteration():4d} | {time.perf_counter()-t1:.1f}s",
                flush=True,
            )

        # 6. XGBoost 4 lá học phần dư từ GLM
        p_xgb = FINAL_DIR / f"fold{fold_id}_v4_xgb_l4_glm.npz"
        if not p_xgb.exists():
            t1 = time.perf_counter()
            dtr_x = xgb.DMatrix(X_tr, label=y_tr, base_margin=glm_oof, enable_categorical=True)
            dva_x = xgb.DMatrix(X_va, label=y_va, base_margin=glm_va, enable_categorical=True)
            dte_x = xgb.DMatrix(X_te, base_margin=glm_te, enable_categorical=True)
            bst_x = xgb.train(
                dict(
                    objective="binary:logistic", eval_metric="auc", tree_method="hist",
                    grow_policy="lossguide", max_leaves=4, max_depth=0, max_bin=1024,
                    learning_rate=0.025, min_child_weight=50.0, subsample=0.8,
                    colsample_bytree=0.45, reg_alpha=1.0, reg_lambda=10.0,
                    nthread=-1, seed=442 + fold_id,
                ),
                dtr_x, num_boost_round=4000, evals=[(dva_x, "val")],
                early_stopping_rounds=120, verbose_eval=False,
            )
            va_l = bst_x.predict(dva_x, output_margin=True, iteration_range=(0, bst_x.best_iteration + 1))
            te_l = bst_x.predict(dte_x, output_margin=True, iteration_range=(0, bst_x.best_iteration + 1))
            assert_sane_logits("v4_xgb_l4_glm", va_l, te_l, glm_va, glm_te)
            save_fold_cache(p_xgb, va_l, te_l)
            print(
                f"  [{'v4_xgb_l4_glm':14s}] Val AUC: {roc_auc_score(y_va, va_l):.6f}"
                f" | iter={bst_x.best_iteration:4d} | {time.perf_counter()-t1:.1f}s",
                flush=True,
            )

        for m in V4_MODELS:
            d = np.load(FINAL_DIR / f"fold{fold_id}_{m}.npz")
            oof_store[m][va_idx] = d["val"].astype(np.float64)
            test_store[m] += d["test"].astype(np.float64) / 5.0
        print(f"  [Fold {fold_id+1} done in {time.perf_counter()-t_fold:.1f}s]", flush=True)

    print("\n================ V4 5-FOLD GLOBAL OOF SUMMARY (ALL 699,960 ROWS) ================", flush=True)
    for m in V4_MODELS:
        g_auc = roc_auc_score(y_all, oof_store[m])
        f_aucs = [roc_auc_score(y_all[va], oof_store[m][va]) for _, va in outer_folds]
        print(f"  {m:16s} | Global OOF: {g_auc:.6f} | Mean±Std: {np.mean(f_aucs):.6f} ± {np.std(f_aucs):.6f}", flush=True)
        out_oof = FINAL_DIR / f"oof_5fold_{m}.csv"
        with atomic_output(out_oof) as tmp:
            pd.DataFrame({ID_COLUMN: train_ids, TARGET_COLUMN: y_all, "pred_prob": expit(oof_store[m])}).to_csv(tmp, index=False)

    print(f"\nALL V4 5-FOLD MODELS COMPLETE in {(time.perf_counter()-t_start)/60:.1f} min!", flush=True)


if __name__ == "__main__":
    main()
