"""Huấn luyện lại 10 cấu hình cây và GLM trên 100% train.csv (699.960 dòng) theo số vòng dừng từ 5-Fold CV."""

from __future__ import annotations

from pathlib import Path
import sys
import time

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fraud_mlp.data import BINARY_COLUMNS, CATEGORICAL_COLUMNS, TARGET_COLUMN
from fraud_mlp.runtime import atomic_output
from experiments.run_final_5fold import _extract_compact_v3_keys, _lookup_sorted, build_base_v3
from experiments.run_v4_glm_trees_5fold import build_plain_features, extract_encoding_keys

FINAL_DIR = PROJECT_ROOT / "artifacts" / "final_5fold"


def safe_logit(p: np.ndarray | float) -> np.ndarray | float:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def verify_and_save(name: str, test_l: np.ndarray) -> None:
    assert np.isfinite(test_l).all(), f"{name}: non-finite test logits!"
    cv5_te = np.mean([np.load(FINAL_DIR / f"fold{f}_{name}.npz")["test"].astype(np.float64) for f in range(5)], axis=0)
    corr = float(np.corrcoef(test_l, cv5_te)[0, 1])
    diff = float(np.mean(test_l) - np.mean(cv5_te))
    print(f"  [Verified {name:16s}] mean={test_l.mean():+.3f} (vs 5fold {cv5_te.mean():+.3f}, diff={diff:+.3f}) | corr_5fold={corr:.5f}", flush=True)
    assert corr > 0.985 and abs(diff) < 0.45, f"{name}: suspicious drift corr={corr:.5f}, diff={diff:+.3f}!"
    with atomic_output(FINAL_DIR / f"refit_full_{name}.npz") as tmp:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, test=test_l.astype(np.float32))


def build_full_v4_data(train_df: pd.DataFrame, y_all: np.ndarray, test_df: pd.DataFrame):
    tr_k = extract_encoding_keys(train_df)
    te_k = extract_encoding_keys(test_df)
    n = len(y_all)
    tot_y = float(y_all.sum())
    prior = tot_y / n
    prior_l = float(safe_logit(prior))

    # Tính WoE lồng 5-Fold (3 hạt giống) trên toàn bộ train và WoE đầy đủ cho test
    seeds = (42, 102, 202)
    fold_sets = [
        [vi for _, vi in StratifiedKFold(n_splits=5, shuffle=True, random_state=sd).split(np.zeros(n), y_all)]
        for sd in seeds
    ]
    te_tr_cols, te_te_cols = {}, {}
    for name, k_tr in tr_k.items():
        uniq, inv, tot_cnt = np.unique(k_tr, return_inverse=True, return_counts=True)
        tot_sum = np.bincount(inv, weights=y_all, minlength=uniq.size)
        m = 5.0 if uniq.size < 2000 else 15.0
        oof_acc = np.zeros(n, dtype=np.float64)
        for f_set in fold_sets:
            for vi in f_set:
                fi = inv[vi]
                fc = np.bincount(fi, minlength=uniq.size)
                fs = np.bincount(fi, weights=y_all[vi], minlength=uniq.size)
                oc = tot_cnt[fi] - fc[fi]
                os_ = tot_sum[fi] - fs[fi]
                fp = (tot_y - float(y_all[vi].sum())) / (n - len(vi))
                p = (os_ + m * fp) / (oc + m)
                oof_acc[vi] += (safe_logit(p) - safe_logit(fp)) / len(seeds)
        te_tr_cols[f"te__{name}"] = oof_acc.astype(np.float32)

        full_woe = (safe_logit((tot_sum + m * prior) / (tot_cnt + m)) - prior_l).astype(np.float32)
        te_te_cols[f"te__{name}"] = _lookup_sorted(te_k[name], uniq, full_woe, 0.0)

    X_glm_tr = pd.concat([
        pd.DataFrame(te_tr_cols),
        train_df[BINARY_COLUMNS].astype(np.float32),
        pd.get_dummies(train_df[CATEGORICAL_COLUMNS], dtype=np.float32),
    ], axis=1)
    X_glm_te = pd.concat([
        pd.DataFrame(te_te_cols),
        test_df[BINARY_COLUMNS].astype(np.float32),
        pd.get_dummies(test_df[CATEGORICAL_COLUMNS], dtype=np.float32),
    ], axis=1).reindex(columns=X_glm_tr.columns, fill_value=0.0)

    X_np = X_glm_tr.to_numpy()
    glm_oof = np.zeros(n, dtype=np.float32)
    for vi in fold_sets[0]:
        ti_mask = np.ones(n, dtype=bool)
        ti_mask[vi] = False
        clf = LogisticRegression(C=0.2, max_iter=200)
        clf.fit(X_np[ti_mask], y_all[ti_mask])
        glm_oof[vi] = clf.decision_function(X_np[vi])

    clf_full = LogisticRegression(C=0.2, max_iter=200)
    clf_full.fit(X_np, y_all)
    glm_te = clf_full.decision_function(X_glm_te.to_numpy()).astype(np.float32)

    X_tr = pd.concat([build_plain_features(train_df), pd.DataFrame(te_tr_cols)], axis=1)
    X_te = pd.concat([build_plain_features(test_df), pd.DataFrame(te_te_cols)], axis=1)
    for c in CATEGORICAL_COLUMNS:
        X_tr[c] = X_tr[c].astype("category")
        X_te[c] = pd.Categorical(X_te[c], categories=X_tr[c].cat.categories)
    return X_tr, X_te, glm_oof, glm_te


def build_full_v3_data(train_df: pd.DataFrame, y_all: np.ndarray, test_df: pd.DataFrame):
    base_tr = build_base_v3(train_df)
    base_te = build_base_v3(test_df)
    all_keys, all_freqs = _extract_compact_v3_keys(train_df)
    te_keys, te_freqs = _extract_compact_v3_keys(test_df)

    n = len(y_all)
    tot_y = float(y_all.sum())
    prior = tot_y / n
    prior_l = float(safe_logit(prior))
    seeds = (42, 102, 202)
    fold_sets = [
        [vi for _, vi in StratifiedKFold(n_splits=5, shuffle=True, random_state=sd).split(np.zeros(n), y_all)]
        for sd in seeds
    ]
    tr_ex, te_ex = {}, {}
    for name, k_tr in all_keys.items():
        uniq, inv, tot_cnt = np.unique(k_tr, return_inverse=True, return_counts=True)
        tot_sum = np.bincount(inv, weights=y_all, minlength=uniq.size)
        m = 5.0 if uniq.size < 2000 else 15.0
        oof_acc = np.zeros(n, dtype=np.float64)
        for f_set in fold_sets:
            for vi in f_set:
                fi = inv[vi]
                fc = np.bincount(fi, minlength=uniq.size)
                fs = np.bincount(fi, weights=y_all[vi], minlength=uniq.size)
                oc = tot_cnt[fi] - fc[fi]
                os_ = tot_sum[fi] - fs[fi]
                fp = (tot_y - float(y_all[vi].sum())) / (n - len(vi))
                p = (os_ + m * fp) / (oc + m)
                oof_acc[vi] += (safe_logit(p) - safe_logit(fp)) / len(seeds)
        full_woe = (safe_logit((tot_sum + m * prior) / (tot_cnt + m)) - prior_l).astype(np.float32)
        tr_ex[f"te__{name}"] = oof_acc.astype(np.float32)
        te_ex[f"te__{name}"] = _lookup_sorted(te_keys[name], uniq, full_woe, 0.0)

    for name, k_tr in all_freqs.items():
        uniq, inv, cnt = np.unique(k_tr, return_inverse=True, return_counts=True)
        lc = np.log1p(cnt.astype(np.float64)).astype(np.float32)
        tr_ex[name] = lc[inv]
        te_ex[name] = _lookup_sorted(te_freqs[name], uniq, lc, 0.0)

    X_tr = pd.concat([base_tr.reset_index(drop=True), pd.DataFrame(tr_ex)], axis=1)
    X_te = pd.concat([base_te.reset_index(drop=True), pd.DataFrame(te_ex)], axis=1)
    for c in CATEGORICAL_COLUMNS:
        X_tr[c] = X_tr[c].astype("category")
        X_te[c] = pd.Categorical(X_te[c], categories=X_tr[c].cat.categories)
    return X_tr, X_te


def main() -> None:
    t_all = time.perf_counter()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    probe = FINAL_DIR / ".write_probe_refit"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()

    data_dir = PROJECT_ROOT / "dataset" / "playground-series-s6e11"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    y_all = train_df[TARGET_COLUMN].to_numpy(dtype=np.int64)

    # Phần 1: Học lại nhánh v4 nền GLM trên 100% train.csv
    v4_names = ["v4_glm", "v4_lgb_l3_glm", "v4_lgb_l4_glm", "v4_lgb_l5_glm", "v4_cb_d4_glm", "v4_xgb_l4_glm"]
    if any(not (FINAL_DIR / f"refit_full_{m}.npz").exists() for m in v4_names):
        print("\n=== PART 1: FULL-DATA REFIT V4 SUITE (699,960 rows) ===", flush=True)
        t0 = time.perf_counter()
        X_tr, X_te, glm_oof, glm_te = build_full_v4_data(train_df, y_all, test_df)
        print(f"  Built full v4 features ({X_tr.shape}) | GLM OOF: {roc_auc_score(y_all, glm_oof):.6f} | {time.perf_counter()-t0:.1f}s", flush=True)

        if not (FINAL_DIR / "refit_full_v4_glm.npz").exists():
            verify_and_save("v4_glm", glm_te.astype(np.float64))

        dtr = lgb.Dataset(X_tr, label=y_all, init_score=glm_oof, categorical_feature=list(CATEGORICAL_COLUMNS), params={"max_bin": 1023}, free_raw_data=False)
        for m_name, leaves, max_d, n_rounds in [
            ("v4_lgb_l3_glm", 3, 2, 750),
            ("v4_lgb_l4_glm", 4, 3, 565),
            ("v4_lgb_l5_glm", 5, 3, 540),
        ]:
            if (FINAL_DIR / f"refit_full_{m_name}.npz").exists():
                continue
            t1 = time.perf_counter()
            acc = np.zeros(len(test_df), dtype=np.float64)
            for sd in (42, 142, 242):
                bst = lgb.train(
                    dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=leaves, max_depth=max_d,
                         min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                         reg_alpha=0.5, reg_lambda=5.0, max_bin=1023, n_jobs=-1, verbosity=-1, seed=sd),
                    dtr, num_boost_round=n_rounds,
                )
                acc += (glm_te + bst.predict(X_te, raw_score=True)) / 3.0
            print(f"  Trained 3-seed {m_name} ({n_rounds} rounds) in {time.perf_counter()-t1:.1f}s", flush=True)
            verify_and_save(m_name, acc)

        if not (FINAL_DIR / "refit_full_v4_cb_d4_glm.npz").exists():
            t1 = time.perf_counter()
            X_tr_cb, X_te_cb = X_tr.copy(), X_te.copy()
            for c in CATEGORICAL_COLUMNS:
                X_tr_cb[c] = X_tr_cb[c].astype(str)
                X_te_cb[c] = X_te_cb[c].astype(str)
            ptr = Pool(X_tr_cb, label=y_all, baseline=glm_oof, cat_features=list(CATEGORICAL_COLUMNS))
            pte = Pool(X_te_cb, baseline=glm_te, cat_features=list(CATEGORICAL_COLUMNS))
            acc = np.zeros(len(test_df), dtype=np.float64)
            for sd in (342, 442):
                cb = CatBoostClassifier(
                    iterations=1075, learning_rate=0.03, depth=4, l2_leaf_reg=15.0,
                    rsm=0.5, one_hot_max_size=16, eval_metric="AUC", random_seed=sd,
                    thread_count=-1, allow_writing_files=False, verbose=False,
                )
                cb.fit(ptr)
                acc += cb.predict(pte, prediction_type="RawFormulaVal") / 2.0
            print(f"  Trained 2-seed v4_cb_d4_glm (1075 rounds) in {time.perf_counter()-t1:.1f}s", flush=True)
            verify_and_save("v4_cb_d4_glm", acc)

        if not (FINAL_DIR / "refit_full_v4_xgb_l4_glm.npz").exists():
            t1 = time.perf_counter()
            dtr_x = xgb.DMatrix(X_tr, label=y_all, base_margin=glm_oof, enable_categorical=True)
            dte_x = xgb.DMatrix(X_te, base_margin=glm_te, enable_categorical=True)
            acc = np.zeros(len(test_df), dtype=np.float64)
            for sd in (442, 542):
                bst_x = xgb.train(
                    dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                         grow_policy="lossguide", max_leaves=4, max_depth=0, max_bin=1024,
                         learning_rate=0.025, min_child_weight=50.0, subsample=0.8,
                         colsample_bytree=0.45, reg_alpha=1.0, reg_lambda=10.0, nthread=-1, seed=sd),
                    dtr_x, num_boost_round=660,
                )
                acc += bst_x.predict(dte_x, output_margin=True) / 2.0
            print(f"  Trained 2-seed v4_xgb_l4_glm (660 rounds) in {time.perf_counter()-t1:.1f}s", flush=True)
            verify_and_save("v4_xgb_l4_glm", acc)

    # Phần 2: Học lại nhánh cây trực tiếp v3 trên 100% train.csv
    v3_trees = ["lgbm_v3_l4", "lgbm_v3_l6_et", "xgb_v3_d3", "catboost_v3_d4"]
    if any(not (FINAL_DIR / f"refit_full_{m}.npz").exists() for m in v3_trees):
        print("\n=== PART 2: FULL-DATA REFIT V3 TREE SUITE (699,960 rows) ===", flush=True)
        t0 = time.perf_counter()
        X_tr3, X_te3 = build_full_v3_data(train_df, y_all, test_df)
        print(f"  Built full v3 features ({X_tr3.shape}) in {time.perf_counter()-t0:.1f}s", flush=True)

        if not (FINAL_DIR / "refit_full_lgbm_v3_l4.npz").exists():
            t1 = time.perf_counter()
            dtr3 = lgb.Dataset(X_tr3, label=y_all, categorical_feature=list(CATEGORICAL_COLUMNS), params={"max_bin": 1023}, free_raw_data=False)
            acc = np.zeros(len(test_df), dtype=np.float64)
            for sd in (42, 142):
                bst = lgb.train(
                    dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=4, max_depth=2,
                         min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.65,
                         reg_alpha=0.0, reg_lambda=2.0, max_bin=1023, n_jobs=-1, verbosity=-1, seed=sd),
                    dtr3, num_boost_round=2400,
                )
                acc += bst.predict(X_te3, raw_score=True) / 2.0
            print(f"  Trained 2-seed lgbm_v3_l4 (2400 rounds) in {time.perf_counter()-t1:.1f}s", flush=True)
            verify_and_save("lgbm_v3_l4", acc)

        if not (FINAL_DIR / "refit_full_lgbm_v3_l6_et.npz").exists():
            t1 = time.perf_counter()
            dtr3e = lgb.Dataset(X_tr3, label=y_all, categorical_feature=list(CATEGORICAL_COLUMNS), params={"max_bin": 2047}, free_raw_data=False)
            acc = np.zeros(len(test_df), dtype=np.float64)
            for sd in (142, 242):
                bst = lgb.train(
                    dict(objective="binary", metric="auc", learning_rate=0.025, num_leaves=6, max_depth=3,
                         min_child_samples=80, extra_trees=True, subsample=0.8, subsample_freq=1, colsample_bytree=0.55,
                         reg_alpha=1.0, reg_lambda=5.0, max_bin=2047, n_jobs=-1, verbosity=-1, seed=sd),
                    dtr3e, num_boost_round=3000,
                )
                acc += bst.predict(X_te3, raw_score=True) / 2.0
            print(f"  Trained 2-seed lgbm_v3_l6_et (3000 rounds) in {time.perf_counter()-t1:.1f}s", flush=True)
            verify_and_save("lgbm_v3_l6_et", acc)

        if not (FINAL_DIR / "refit_full_xgb_v3_d3.npz").exists():
            t1 = time.perf_counter()
            dtr_x3 = xgb.DMatrix(X_tr3, label=y_all, enable_categorical=True)
            dte_x3 = xgb.DMatrix(X_te3, enable_categorical=True)
            acc = np.zeros(len(test_df), dtype=np.float64)
            for sd in (42, 142):
                bst_x = xgb.train(
                    dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                         max_bin=1024, learning_rate=0.03, max_depth=3, min_child_weight=35.0,
                         subsample=0.8, colsample_bytree=0.65, reg_alpha=0.5, reg_lambda=3.0,
                         nthread=-1, seed=sd),
                    dtr_x3, num_boost_round=1800,
                )
                acc += bst_x.predict(dte_x3, output_margin=True) / 2.0
            print(f"  Trained 2-seed xgb_v3_d3 (1800 rounds) in {time.perf_counter()-t1:.1f}s", flush=True)
            verify_and_save("xgb_v3_d3", acc)

        if not (FINAL_DIR / "refit_full_catboost_v3_d4.npz").exists():
            t1 = time.perf_counter()
            X_tr_cb3, X_te_cb3 = X_tr3.copy(), X_te3.copy()
            for c in CATEGORICAL_COLUMNS:
                X_tr_cb3[c] = X_tr_cb3[c].astype(str)
                X_te_cb3[c] = X_te_cb3[c].astype(str)
            cat_idx = [X_tr_cb3.columns.get_loc(c) for c in CATEGORICAL_COLUMNS]
            ptr3 = Pool(X_tr_cb3, label=y_all, cat_features=cat_idx)
            pte3 = Pool(X_te_cb3, cat_features=cat_idx)
            acc = np.zeros(len(test_df), dtype=np.float64)
            for sd in (42, 142):
                cb = CatBoostClassifier(
                    iterations=2400, learning_rate=0.04, depth=4, l2_leaf_reg=8.0,
                    one_hot_max_size=16, rsm=0.6, eval_metric="AUC", random_seed=sd,
                    thread_count=-1, allow_writing_files=False, verbose=False,
                )
                cb.fit(ptr3)
                acc += cb.predict(pte3, prediction_type="RawFormulaVal") / 2.0
            print(f"  Trained 2-seed catboost_v3_d4 (2400 rounds) in {time.perf_counter()-t1:.1f}s", flush=True)
            verify_and_save("catboost_v3_d4", acc)

    print(f"\nALL 10 FULL-DATA REFIT FAMILIES COMPLETE in {(time.perf_counter()-t_all)/60:.1f} min!", flush=True)


if __name__ == "__main__":
    main()
