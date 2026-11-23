"""Hàm tạo đặc trưng dùng chung (Tầng A, Tầng B WoE lồng 5-Fold và GLM thưa) cho S6E11."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from fraud_mlp.data import BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


FLOAT_COLUMNS = (
    "amount_usd",
    "account_balance_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "velocity_score",
    "merchant_risk_score",
)

DISCRETE_NUMERIC_COLUMNS = (
    "customer_age",
    "card_age_months",
    "time_of_day_hour",
    "day_of_week",
    "txn_count_last_24h",
    "prior_disputes",
    "cvv_retry_count",
)


def _safe_logit(p: np.ndarray | float) -> np.ndarray | float:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def _leading_digits(int_arr: np.ndarray, k: int) -> np.ndarray:
    """Lấy k chữ số đầu của mảng số nguyên không âm theo dạng vector."""
    vals = np.maximum(int_arr, 0).astype(np.int64)
    out = np.zeros_like(vals)
    pos = vals > 0
    if not pos.any():
        return out
    v = vals[pos]
    digits = np.floor(np.log10(v.astype(np.float64))).astype(np.int64) + 1
    shift = np.maximum(digits - k, 0)
    # Dùng lũy thừa nguyên của 10 để tránh sai số dấu phẩy động
    pow10 = np.power(10, np.minimum(shift, 12), dtype=np.int64)
    out[pos] = v // pow10
    return out


def build_tier_a_and_digit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo đặc trưng theo từng dòng (Tầng A và các chữ số phần lẻ), không dùng nhãn."""
    out = pd.DataFrame(index=df.index)

    # Các cột số gốc
    for col in NUMERIC_COLUMNS:
        out[col] = df[col].astype(np.float32)

    # Các cờ nhị phân gốc
    bin_mat = np.column_stack([df[c].astype(np.int8).to_numpy() for c in BINARY_COLUMNS])
    for i, col in enumerate(BINARY_COLUMNS):
        out[col] = bin_mat[:, i]

    # Giữ các cột phân loại ở kiểu category
    for col in CATEGORICAL_COLUMNS:
        out[col] = df[col].astype("string").fillna("__MISSING__")

    # Đặc trưng nghiệp vụ Tầng A
    amt = df["amount_usd"].to_numpy(dtype=np.float64)
    bal = df["account_balance_usd"].to_numpy(dtype=np.float64)
    hrs = df["hours_since_last_txn"].to_numpy(dtype=np.float64)
    dist = df["distance_from_home_km"].to_numpy(dtype=np.float64)
    vel = df["velocity_score"].to_numpy(dtype=np.float64)
    risk = df["merchant_risk_score"].to_numpy(dtype=np.float64)
    txn24 = df["txn_count_last_24h"].to_numpy(dtype=np.float64)
    hour = df["time_of_day_hour"].to_numpy(dtype=np.float64)
    dow = df["day_of_week"].to_numpy(dtype=np.float64)

    log_amt = np.log1p(np.maximum(amt, 0.0))
    log_bal = np.log1p(np.maximum(bal, 0.0))
    log_hrs = np.log1p(np.maximum(hrs, 0.0))
    log_dist = np.log1p(np.maximum(dist, 0.0))

    out["n_risk_flags"] = bin_mat.sum(axis=1).astype(np.int8)
    out["log_amount"] = log_amt.astype(np.float32)
    out["log_balance"] = log_bal.astype(np.float32)
    out["log_hours"] = log_hrs.astype(np.float32)
    out["log_distance"] = log_dist.astype(np.float32)
    out["amount_to_balance_ratio"] = (amt / (np.maximum(bal, 0.0) + 1.0)).astype(np.float32)
    out["log_amount_to_balance"] = (log_amt - log_bal).astype(np.float32)
    out["amount_gt_balance"] = (amt > bal).astype(np.int8)
    out["distance_per_hour"] = (dist / (np.maximum(hrs, 0.0) + 0.1)).astype(np.float32)
    out["amount_per_txn_24h"] = (amt / (np.maximum(txn24, 0.0) + 1.0)).astype(np.float32)
    out["risk_x_log_amount"] = (risk * log_amt).astype(np.float32)
    out["velocity_x_txn"] = (vel * txn24).astype(np.float32)
    out["amount_x_velocity"] = (log_amt * vel).astype(np.float32)
    out["risk_x_velocity"] = (risk * vel).astype(np.float32)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)
    out["is_night"] = ((hour <= 5) | (hour >= 23)).astype(np.int8)
    out["cvv_retry_flag"] = (df["cvv_retry_count"].to_numpy() > 0).astype(np.int8)
    out["prior_disputes_flag"] = (df["prior_disputes"].to_numpy() > 0).astype(np.int8)

    # Các cột chữ số phần lẻ dạng số nguyên cho mô hình cây
    for col in FLOAT_COLUMNS:
        c100 = np.rint(df[col].to_numpy(dtype=np.float64) * 100.0).astype(np.int64)
        int_part = np.floor_divide(c100, 100)
        dec2 = np.mod(c100, 100)
        out[f"{col}__int"] = int_part.astype(np.int32)
        out[f"{col}__dec2"] = dec2.astype(np.int16)
        out[f"{col}__cent1"] = np.mod(c100, 10).astype(np.int8)
        out[f"{col}__cent2"] = np.mod(np.floor_divide(c100, 10), 10).astype(np.int8)
        out[f"{col}__unit"] = np.mod(int_part, 10).astype(np.int8)
        out[f"{col}__tens"] = np.mod(np.floor_divide(int_part, 10), 10).astype(np.int8)
        out[f"{col}__lead1"] = _leading_digits(int_part, 1).astype(np.int8)
        out[f"{col}__lead2"] = _leading_digits(int_part, 2).astype(np.int16)
        out[f"{col}__is_round_00"] = (dec2 == 0).astype(np.int8)
        if col in ("amount_usd", "account_balance_usd"):
            out[f"{col}__int_mod100"] = np.mod(int_part, 100).astype(np.int16)
            out[f"{col}__n_digits"] = np.where(
                int_part > 0, np.floor(np.log10(np.maximum(int_part, 1))).astype(np.int8) + 1, 0
            ).astype(np.int8)

    return out


def extract_te_and_freq_keys(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Tạo các khóa số nguyên int64 phục vụ mã hóa WoE lồng 5-Fold và đếm tần suất."""
    te_keys: dict[str, np.ndarray] = {}
    freq_keys: dict[str, np.ndarray] = {}
    c100_map: dict[str, np.ndarray] = {}

    for col in NUMERIC_COLUMNS:
        arr = df[col].to_numpy(dtype=np.float64)
        c100 = np.rint(arr * 100.0).astype(np.int64)
        c100_map[col] = c100
        te_keys[f"{col}__exact"] = c100
        freq_keys[f"{col}__exact_freq"] = c100

    for col in FLOAT_COLUMNS:
        c100 = c100_map[col]
        int_part = np.floor_divide(c100, 100)
        te_keys[f"{col}__int"] = int_part
        te_keys[f"{col}__r1"] = np.floor_divide(c100, 10)
        te_keys[f"{col}__dec2"] = np.mod(c100, 100)
        te_keys[f"{col}__cent1"] = np.mod(c100, 10)
        te_keys[f"{col}__cent2"] = np.mod(np.floor_divide(c100, 10), 10)
        te_keys[f"{col}__unit"] = np.mod(int_part, 10)
        te_keys[f"{col}__tens"] = np.mod(np.floor_divide(int_part, 10), 10)
        te_keys[f"{col}__lead1"] = _leading_digits(int_part, 1)
        te_keys[f"{col}__lead2"] = _leading_digits(int_part, 2)
        te_keys[f"{col}__int_div10"] = np.floor_divide(int_part, 10)
        freq_keys[f"{col}__int_freq"] = int_part
        freq_keys[f"{col}__r1_freq"] = np.floor_divide(c100, 10)
        if col in ("amount_usd", "account_balance_usd"):
            te_keys[f"{col}__int_mod100"] = np.mod(int_part, 100)
            te_keys[f"{col}__int_div100"] = np.floor_divide(int_part, 100)

    # Mã hóa cột phân loại bằng hàm băm ổn định giữa các tập chia
    cat_codes: dict[str, np.ndarray] = {}
    for col in CATEGORICAL_COLUMNS:
        s = df[col].astype("string").fillna("__MISSING__")
        cat_codes[col] = pd.util.hash_array(s.to_numpy(dtype=object), categorize=False).astype(np.int64) & 0x7FFFFFFF

    bin_mat = np.column_stack([df[c].astype(np.int64).to_numpy() for c in BINARY_COLUMNS])
    bin_map = {col: bin_mat[:, i] for i, col in enumerate(BINARY_COLUMNS)}
    n_flags = bin_mat.sum(axis=1)
    risk_int = np.floor_divide(c100_map["merchant_risk_score"], 100)
    vel_int = np.floor_divide(c100_map["velocity_score"], 100)
    age_int = np.floor_divide(c100_map["customer_age"], 100)
    cvv_int = np.floor_divide(c100_map["cvv_retry_count"], 100)
    amt_int = np.floor_divide(c100_map["amount_usd"], 100)
    hrs_int = np.floor_divide(c100_map["hours_since_last_txn"], 100)

    def combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * 1000003 + b

    pairs = [
        ("mc_x_auth", cat_codes["merchant_category"], cat_codes["auth_method"]),
        ("mc_x_new", cat_codes["merchant_category"], bin_map["is_new_merchant"]),
        ("auth_x_cvv", cat_codes["auth_method"], cvv_int),
        ("auth_x_flags", cat_codes["auth_method"], n_flags),
        ("mc_x_flags", cat_codes["merchant_category"], n_flags),
        ("new_x_cvv", bin_map["is_new_merchant"], cvv_int),
        ("new_x_foreign", bin_map["is_new_merchant"], bin_map["is_foreign_transaction"]),
        ("vpn_x_ipmis", bin_map["used_vpn"], bin_map["ip_country_mismatch"]),
        ("risk_int_x_vel_int", risk_int, vel_int),
        ("risk_int_x_flags", risk_int, n_flags),
        ("vel_int_x_flags", vel_int, n_flags),
        ("age_x_auth", age_int, cat_codes["auth_method"]),
        ("age_x_flags", age_int, n_flags),
        ("hrs_int_x_vel_int", hrs_int, vel_int),
        ("amt_div10_x_mc", np.floor_divide(amt_int, 10), cat_codes["merchant_category"]),
        ("chan_x_dev", cat_codes["channel"], cat_codes["device_type"]),
        # Các cặp cột giá trị lặp lại chính xác
        ("exact_amt_x_bal", c100_map["amount_usd"], c100_map["account_balance_usd"]),
        ("exact_mc_x_risk", cat_codes["merchant_category"], c100_map["merchant_risk_score"]),
        ("exact_vel_x_txn24", c100_map["velocity_score"], np.floor_divide(c100_map["txn_count_last_24h"], 100)),
        ("exact_auth_x_risk_int", cat_codes["auth_method"], risk_int),
        ("exact_auth_x_vel_int", cat_codes["auth_method"], vel_int),
        ("exact_cvv_x_risk_int", cvv_int, risk_int),
        ("exact_cvv_x_vel_int", cvv_int, vel_int),
        ("exact_new_x_risk_int", bin_map["is_new_merchant"], risk_int),
        ("exact_new_x_vel_int", bin_map["is_new_merchant"], vel_int),
        ("exact_age_x_cardage", age_int, np.floor_divide(c100_map["card_age_months"], 100)),
        ("exact_age_x_risk_int", age_int, risk_int),
        ("exact_hrs_r1_x_dist_int", np.floor_divide(c100_map["hours_since_last_txn"], 10), np.floor_divide(c100_map["distance_from_home_km"], 100)),
        # Tổ hợp 3 yếu tố rủi ro (cờ rủi ro, xác thực, CVV)
        ("triple_auth_new_cvv", combine(cat_codes["auth_method"], bin_map["is_new_merchant"]), cvv_int),
        ("triple_mc_auth_new", combine(cat_codes["merchant_category"], cat_codes["auth_method"]), bin_map["is_new_merchant"]),
    ]
    for name, a, b in pairs:
        te_keys[f"pair__{name}"] = combine(a, b)
        if name.startswith("exact_"):
            freq_keys[f"freq__{name}"] = combine(a, b)

    return te_keys, freq_keys


def fit_transform_group_aggs(
    train_df: pd.DataFrame,
    other_dfs: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Tính độ lệch so với trung bình và độ lệch chuẩn theo nhóm trên tập train (không dùng nhãn)."""
    specs = [
        ("merchant_category", "amount_usd"),
        ("merchant_category", "merchant_risk_score"),
        ("merchant_category", "velocity_score"),
        ("auth_method", "amount_usd"),
        ("auth_method", "velocity_score"),
        ("auth_method", "merchant_risk_score"),
        ("txn_count_last_24h", "velocity_score"),
        ("txn_count_last_24h", "hours_since_last_txn"),
        ("customer_age", "amount_usd"),
        ("customer_age", "account_balance_usd"),
        ("card_type", "amount_usd"),
        ("channel", "amount_usd"),
    ]
    tr_cols: dict[str, np.ndarray] = {}
    other_cols: list[dict[str, np.ndarray]] = [{} for _ in other_dfs]

    for gcol, vcol in specs:
        stats = train_df.groupby(gcol, observed=False)[vcol].agg(["mean", "std"])
        global_mean = float(train_df[vcol].mean())
        global_std = float(train_df[vcol].std()) or 1.0
        mean_map = stats["mean"]
        std_map = stats["std"].fillna(global_std).replace(0.0, global_std)

        tr_m = train_df[gcol].map(mean_map).fillna(global_mean).to_numpy(dtype=np.float64)
        tr_s = train_df[gcol].map(std_map).fillna(global_std).to_numpy(dtype=np.float64)
        tr_v = train_df[vcol].to_numpy(dtype=np.float64)
        tr_cols[f"grp_diff__{vcol}__by__{gcol}"] = (tr_v - tr_m).astype(np.float32)
        tr_cols[f"grp_z__{vcol}__by__{gcol}"] = ((tr_v - tr_m) / tr_s).astype(np.float32)

        for i, df in enumerate(other_dfs):
            o_m = df[gcol].map(mean_map).fillna(global_mean).to_numpy(dtype=np.float64)
            o_s = df[gcol].map(std_map).fillna(global_std).to_numpy(dtype=np.float64)
            o_v = df[vcol].to_numpy(dtype=np.float64)
            other_cols[i][f"grp_diff__{vcol}__by__{gcol}"] = (o_v - o_m).astype(np.float32)
            other_cols[i][f"grp_z__{vcol}__by__{gcol}"] = ((o_v - o_m) / o_s).astype(np.float32)

    return (
        pd.DataFrame(tr_cols, index=train_df.index),
        [pd.DataFrame(cols, index=df.index) for cols, df in zip(other_cols, other_dfs)],
    )


def _lookup_sorted(keys_arr: np.ndarray, sorted_keys: np.ndarray, sorted_vals: np.ndarray, default_val: float = 0.0) -> np.ndarray:
    if sorted_keys.size == 0:
        return np.full(len(keys_arr), default_val, dtype=np.float32)
    idx = np.searchsorted(sorted_keys, keys_arr)
    idx_safe = np.minimum(idx, sorted_keys.size - 1)
    matched = sorted_keys[idx_safe] == keys_arr
    return np.where(matched, sorted_vals[idx_safe], default_val).astype(np.float32)


def fit_transform_nested_te(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    other_dfs: list[pd.DataFrame],
    *,
    smoothings: tuple[float, ...] = (5.0, 30.0),
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Mã hóa WoE lồng 5-Fold và đếm tần suất theo từng fold, tránh rò rỉ nhãn."""
    y = np.asarray(y_train, dtype=np.float64)
    n = len(y)
    total_y = float(y.sum())
    prior = total_y / max(n, 1)
    prior_logit = float(_safe_logit(prior))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_indices = [val_idx for _, val_idx in skf.split(np.zeros(n), y.astype(np.int64))]

    tr_te_keys, tr_freq_keys = extract_te_and_freq_keys(train_df)
    other_keys = [extract_te_and_freq_keys(df) for df in other_dfs]

    tr_cols: dict[str, np.ndarray] = {}
    other_cols: list[dict[str, np.ndarray]] = [{} for _ in other_dfs]

    for name, k in tr_te_keys.items():
        uniq, inv, total_cnt = np.unique(k, return_inverse=True, return_counts=True)
        total_sum = np.bincount(inv, weights=y, minlength=uniq.size)

        # Trừ thống kê của chính fold hiện tại để tránh rò rỉ
        fold_stats = []
        for val_idx in fold_indices:
            fold_inv = inv[val_idx]
            fold_cnt = np.bincount(fold_inv, minlength=uniq.size)
            fold_sum = np.bincount(fold_inv, weights=y[val_idx], minlength=uniq.size)
            oof_cnt = total_cnt[fold_inv] - fold_cnt[fold_inv]
            oof_sum = total_sum[fold_inv] - fold_sum[fold_inv]
            fold_n = n - len(val_idx)
            fold_prior = (total_y - float(y[val_idx].sum())) / max(fold_n, 1)
            fold_prior_logit = float(_safe_logit(fold_prior))
            fold_stats.append((val_idx, oof_cnt, oof_sum, fold_prior, fold_prior_logit))

        for m in smoothings:
            col_name = f"te_{int(m)}__{name}"
            oof_vals = np.zeros(n, dtype=np.float32)
            for val_idx, oof_cnt, oof_sum, fold_prior, fold_prior_logit in fold_stats:
                p = (oof_sum + m * fold_prior) / (oof_cnt + m)
                oof_vals[val_idx] = (_safe_logit(p) - fold_prior_logit).astype(np.float32)
            tr_cols[col_name] = oof_vals

            full_p = (total_sum + m * prior) / (total_cnt + m)
            full_woe = (_safe_logit(full_p) - prior_logit).astype(np.float32)
            for i, (o_te_keys, _) in enumerate(other_keys):
                other_cols[i][col_name] = _lookup_sorted(o_te_keys[name], uniq, full_woe, default_val=0.0)

    for name, k in tr_freq_keys.items():
        uniq, inv, counts = np.unique(k, return_inverse=True, return_counts=True)
        log_counts = np.log1p(counts.astype(np.float64)).astype(np.float32)
        tr_cols[name] = log_counts[inv]
        for i, (_, o_freq_keys) in enumerate(other_keys):
            other_cols[i][name] = _lookup_sorted(o_freq_keys[name], uniq, log_counts, default_val=0.0)

    return (
        pd.DataFrame(tr_cols, index=train_df.index),
        [pd.DataFrame(cols, index=df.index) for cols, df in zip(other_cols, other_dfs)],
    )


def _build_sparse_glm_inputs(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Tạo ma trận liên tục (cho Spline) và bảng rời rạc (cho One-Hot thưa) của mô hình GLM."""
    amt = df["amount_usd"].to_numpy(dtype=np.float64)
    bal = df["account_balance_usd"].to_numpy(dtype=np.float64)
    hrs = df["hours_since_last_txn"].to_numpy(dtype=np.float64)
    dist = df["distance_from_home_km"].to_numpy(dtype=np.float64)
    vel = df["velocity_score"].to_numpy(dtype=np.float64)
    risk = df["merchant_risk_score"].to_numpy(dtype=np.float64)
    hour = df["time_of_day_hour"].to_numpy(dtype=np.float64)
    dow = df["day_of_week"].to_numpy(dtype=np.float64)

    log_amt = np.log1p(np.maximum(amt, 0.0))
    log_bal = np.log1p(np.maximum(bal, 0.0))
    log_hrs = np.log1p(np.maximum(hrs, 0.0))
    log_dist = np.log1p(np.maximum(dist, 0.0))

    cont = np.column_stack([
        amt, bal, hrs, dist, vel, risk,
        log_amt, log_bal, log_hrs, log_dist,
        amt / (np.maximum(bal, 0.0) + 1.0),
        log_amt - log_bal,
        dist / (np.maximum(hrs, 0.0) + 0.1),
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        np.sin(2.0 * np.pi * dow / 7.0),
        np.cos(2.0 * np.pi * dow / 7.0),
    ])

    tok = {}
    for col in CATEGORICAL_COLUMNS:
        tok[col] = df[col].astype("string").fillna("__M__").to_numpy(dtype=object)
    for col in BINARY_COLUMNS:
        tok[col] = df[col].astype(np.int8).to_numpy()
    bin_sum = np.column_stack([df[c].astype(np.int8).to_numpy() for c in BINARY_COLUMNS]).sum(axis=1)
    tok["n_risk_flags"] = bin_sum
    tok["amount_gt_balance"] = (amt > bal).astype(np.int8)

    for col in DISCRETE_NUMERIC_COLUMNS:
        tok[col] = np.rint(df[col].to_numpy(dtype=np.float64)).astype(np.int32)

    # Mã hóa giá trị chính xác, phần nguyên và chữ số lẻ cho cột số thực
    for col in FLOAT_COLUMNS:
        c100 = np.rint(df[col].to_numpy(dtype=np.float64) * 100.0).astype(np.int64)
        int_part = np.floor_divide(c100, 100)
        tok[f"{col}__exact"] = c100
        tok[f"{col}__int"] = int_part
        tok[f"{col}__r1"] = np.floor_divide(c100, 10)
        tok[f"{col}__dec2"] = np.mod(c100, 100)
        tok[f"{col}__cent1"] = np.mod(c100, 10)
        tok[f"{col}__cent2"] = np.mod(np.floor_divide(c100, 10), 10)
        tok[f"{col}__unit"] = np.mod(int_part, 10)
        tok[f"{col}__tens"] = np.mod(np.floor_divide(int_part, 10), 10)
        tok[f"{col}__lead1"] = _leading_digits(int_part, 1)
        tok[f"{col}__lead2"] = _leading_digits(int_part, 2)
        if col in ("amount_usd", "account_balance_usd"):
            tok[f"{col}__int_mod100"] = np.mod(int_part, 100)
            tok[f"{col}__int_div10"] = np.floor_divide(int_part, 10)

    # Ghép cặp các cột rời rạc ít giá trị
    mc = df["merchant_category"].astype("string").to_numpy()
    auth = df["auth_method"].astype("string").to_numpy()
    chan = df["channel"].astype("string").to_numpy()
    dev = df["device_type"].astype("string").to_numpy()
    new_m = df["is_new_merchant"].astype(np.int8).astype(str).to_numpy()
    for_m = df["is_foreign_transaction"].astype(np.int8).astype(str).to_numpy()
    cvv = tok["cvv_retry_count"].astype(str)
    flags = bin_sum.astype(str)
    tok["p_mc_auth"] = np.char.add(np.char.add(mc.astype(str), "|"), auth.astype(str))
    tok["p_mc_new"] = np.char.add(np.char.add(mc.astype(str), "|"), new_m)
    tok["p_auth_cvv"] = np.char.add(np.char.add(auth.astype(str), "|"), cvv)
    tok["p_auth_flags"] = np.char.add(np.char.add(auth.astype(str), "|"), flags)
    tok["p_mc_flags"] = np.char.add(np.char.add(mc.astype(str), "|"), flags)
    tok["p_new_for"] = np.char.add(np.char.add(new_m, "|"), for_m)
    tok["p_chan_dev"] = np.char.add(np.char.add(chan.astype(str), "|"), dev.astype(str))

    return cont, pd.DataFrame(tok, index=df.index)


def _fit_one_sparse_glm(
    tr_cont: np.ndarray,
    tr_tok: pd.DataFrame,
    y_tr: np.ndarray,
    eval_list: list[tuple[np.ndarray, pd.DataFrame]],
    *,
    C: float = 0.25,
    min_frequency: int = 3,
    n_knots: int = 12,
    tier_a_only: bool = False,
) -> list[np.ndarray]:
    """Huấn luyện một mô hình hồi quy Logistic L2 trên đặc trưng Spline + One-Hot thưa."""
    scaler = StandardScaler()
    tr_cont_s = scaler.fit_transform(tr_cont)
    spline = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
    tr_spline = sparse.csr_matrix(spline.fit_transform(tr_cont_s))

    if tier_a_only:
        keep_cols = list(CATEGORICAL_COLUMNS) + list(BINARY_COLUMNS) + [
            "n_risk_flags", "amount_gt_balance", *DISCRETE_NUMERIC_COLUMNS,
            "p_mc_auth", "p_mc_new", "p_auth_cvv", "p_auth_flags", "p_mc_flags", "p_new_for", "p_chan_dev",
        ]
        tr_tok_use = tr_tok[keep_cols]
    else:
        tr_tok_use = tr_tok

    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency, sparse_output=True)
    tr_oh = ohe.fit_transform(tr_tok_use)
    X_tr = sparse.hstack([tr_spline, tr_oh], format="csr", dtype=np.float32)

    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=300,
        tol=1e-4,
    )
    clf.fit(X_tr, y_tr.astype(np.int64))

    logits_out = []
    for e_cont, e_tok in eval_list:
        e_cont_s = scaler.transform(e_cont)
        e_spline = sparse.csr_matrix(spline.transform(e_cont_s))
        e_tok_use = e_tok[tr_tok_use.columns] if tier_a_only else e_tok
        e_oh = ohe.transform(e_tok_use)
        X_e = sparse.hstack([e_spline, e_oh], format="csr", dtype=np.float32)
        logits_out.append(clf.decision_function(X_e).astype(np.float32))
    return logits_out


def fit_predict_nested_sparse_glm(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    other_dfs: list[pd.DataFrame],
    *,
    C: float = 0.25,
    min_frequency: int = 3,
    n_knots: int = 12,
    n_folds: int = 5,
    seed: int = 42,
    tier_a_only: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sinh logit OOF lồng cho tập train và logit toàn tập train cho tập validation/test."""
    y = np.asarray(y_train, dtype=np.int64)
    tr_cont, tr_tok = _build_sparse_glm_inputs(train_df)
    other_inputs = [_build_sparse_glm_inputs(df) for df in other_dfs]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_logits = np.zeros(len(train_df), dtype=np.float32)

    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        va_logit = _fit_one_sparse_glm(
            tr_cont[tr_idx],
            tr_tok.iloc[tr_idx],
            y[tr_idx],
            [(tr_cont[va_idx], tr_tok.iloc[va_idx])],
            C=C,
            min_frequency=min_frequency,
            n_knots=n_knots,
            tier_a_only=tier_a_only,
        )[0]
        oof_logits[va_idx] = va_logit

    other_logits = _fit_one_sparse_glm(
        tr_cont,
        tr_tok,
        y,
        other_inputs,
        C=C,
        min_frequency=min_frequency,
        n_knots=n_knots,
        tier_a_only=tier_a_only,
    )
    return oof_logits, other_logits
