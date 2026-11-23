"""Feature schema and preprocessing fitted exclusively on training rows."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch


ID_COLUMN = "id"
TARGET_COLUMN = "is_fraud"
NUMERIC_COLUMNS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]
BINARY_COLUMNS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
CATEGORICAL_COLUMNS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]
DENSE_FEATURE_COUNT = len(NUMERIC_COLUMNS) + len(BINARY_COLUMNS)
BINARY_VALUES = {"true": 1, "false": 0, "1": 1, "0": 0}
EXTRA_FEATURE_NAMES = (
    "amount_to_balance_ratio",
    "log_amount_to_balance",
    "distance_per_hour",
    "amount_per_txn_24h",
    "risk_x_log_amount",
    "velocity_x_txn",
    "mismatch_count",
    "is_night",
    "balance_after_txn",
    "amount_x_velocity",
    "cvv_retry_flag",
    "prior_disputes_flag",
    "amount_gt_balance",
    "log_amount",
    "log_balance",
    "log_hours",
    "log_distance",
)

FLOAT_DIGIT_COLUMNS = (
    "amount_usd",
    "account_balance_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "velocity_score",
    "merchant_risk_score",
)


def normalize_preprocessing_options(options=None):
    if not options:
        return {
            "log1p_columns": [],
            "cyclic_columns": {},
            "extra_features": [],
            "ple_bins": 0,
            "onehot_categoricals": False,
            "discrete_onehot_columns": [],
            "tier_b_te": False,
            "te_smoothing": 20.0,
            "te_folds": 5,
            "te_pairs": False,
        }

    log1p_columns = list(options.get("log1p_columns", []))
    cyclic_raw = options.get("cyclic_columns", {})
    extra_features = list(options.get("extra_features", []))
    ple_bins = int(options.get("ple_bins", 0))
    onehot_categoricals = bool(options.get("onehot_categoricals", False))
    discrete_onehot_columns = list(options.get("discrete_onehot_columns", []))
    tier_b_te = bool(options.get("tier_b_te", False))
    te_smoothing = float(options.get("te_smoothing", 20.0))
    te_folds = int(options.get("te_folds", 5))
    te_pairs = bool(options.get("te_pairs", False))

    unknown_extra = [name for name in extra_features if name not in EXTRA_FEATURE_NAMES]
    if unknown_extra:
        raise ValueError(f"Unknown extra features: {', '.join(unknown_extra)}")

    allowed_log1p = set(NUMERIC_COLUMNS).union(extra_features)
    unknown_log1p = [name for name in log1p_columns if name not in allowed_log1p]
    if unknown_log1p:
        raise ValueError(f"Unknown log1p columns: {', '.join(unknown_log1p)}")

    unknown_discrete = [name for name in discrete_onehot_columns if name not in NUMERIC_COLUMNS]
    if unknown_discrete:
        raise ValueError(f"Unknown discrete_onehot_columns: {', '.join(unknown_discrete)}")

    if not isinstance(cyclic_raw, dict):
        raise ValueError("preprocessing.cyclic_columns must be a mapping of column to period.")
    cyclic_columns = {}
    for column, period in cyclic_raw.items():
        if column not in NUMERIC_COLUMNS:
            raise ValueError(f"Unknown cyclic column: {column}")
        if type(period) not in (int, float) or not np.isfinite(period) or period <= 0:
            raise ValueError(f"Cyclic period for {column} must be a positive finite number.")
        cyclic_columns[column] = float(period)

    overlap = set(log1p_columns).intersection(cyclic_columns)
    if overlap:
        raise ValueError(f"Columns cannot be both log1p and cyclic: {', '.join(sorted(overlap))}")

    if ple_bins < 0 or ple_bins == 1:
        raise ValueError("preprocessing.ple_bins must be 0 (disabled) or >= 2.")
    if te_smoothing <= 0 or not np.isfinite(te_smoothing):
        raise ValueError("preprocessing.te_smoothing must be a positive finite number.")
    if te_folds < 2:
        raise ValueError("preprocessing.te_folds must be at least 2.")

    return {
        "log1p_columns": log1p_columns,
        "cyclic_columns": cyclic_columns,
        "extra_features": extra_features,
        "ple_bins": ple_bins,
        "onehot_categoricals": onehot_categoricals,
        "discrete_onehot_columns": discrete_onehot_columns,
        "tier_b_te": tier_b_te,
        "te_smoothing": te_smoothing,
        "te_folds": te_folds,
        "te_pairs": te_pairs,
    }


def _parse_binary(frame):
    binary = frame[BINARY_COLUMNS].apply(
        lambda column: column.astype("string").str.strip().str.lower().map(BINARY_VALUES)
    )
    invalid = binary.columns[binary.isna().any()].tolist()
    if invalid:
        raise ValueError(f"Expected True/False or 0/1 in: {', '.join(invalid)}")
    return binary.to_numpy(dtype=np.float32)


def _compute_extra_features(frame, binary_array, extra_features):
    if not extra_features:
        return {}
    num = {
        col: frame[col].to_numpy(dtype=np.float64, na_value=np.nan)
        for col in NUMERIC_COLUMNS
    }
    computed = {}
    for name in extra_features:
        if name == "amount_to_balance_ratio":
            computed[name] = num["amount_usd"] / (np.maximum(num["account_balance_usd"], 0.0) + 1.0)
        elif name == "log_amount_to_balance":
            computed[name] = (
                np.log1p(np.maximum(num["amount_usd"], 0.0))
                - np.log1p(np.maximum(num["account_balance_usd"], 0.0))
            )
        elif name == "distance_per_hour":
            computed[name] = num["distance_from_home_km"] / (np.maximum(num["hours_since_last_txn"], 0.0) + 0.1)
        elif name == "amount_per_txn_24h":
            computed[name] = num["amount_usd"] / (np.maximum(num["txn_count_last_24h"], 0.0) + 1.0)
        elif name == "risk_x_log_amount":
            computed[name] = num["merchant_risk_score"] * np.log1p(np.maximum(num["amount_usd"], 0.0))
        elif name == "velocity_x_txn":
            computed[name] = num["velocity_score"] * num["txn_count_last_24h"]
        elif name == "mismatch_count":
            computed[name] = binary_array.sum(axis=1, dtype=np.float64)
        elif name == "is_night":
            hour = num["time_of_day_hour"]
            computed[name] = np.where(np.isfinite(hour), ((hour <= 5) | (hour >= 23)).astype(np.float64), np.nan)
        elif name == "balance_after_txn":
            computed[name] = num["account_balance_usd"] - num["amount_usd"]
        elif name == "amount_x_velocity":
            computed[name] = np.log1p(np.maximum(num["amount_usd"], 0.0)) * num["velocity_score"]
        elif name == "cvv_retry_flag":
            val = num["cvv_retry_count"]
            computed[name] = np.where(np.isfinite(val), (val > 0).astype(np.float64), np.nan)
        elif name == "prior_disputes_flag":
            val = num["prior_disputes"]
            computed[name] = np.where(np.isfinite(val), (val > 0).astype(np.float64), np.nan)
        elif name == "amount_gt_balance":
            computed[name] = (num["amount_usd"] > num["account_balance_usd"]).astype(np.float64)
        elif name == "log_amount":
            computed[name] = np.log1p(np.maximum(num["amount_usd"], 0.0))
        elif name == "log_balance":
            computed[name] = np.log1p(np.maximum(num["account_balance_usd"], 0.0))
        elif name == "log_hours":
            computed[name] = np.log1p(np.maximum(num["hours_since_last_txn"], 0.0))
        elif name == "log_distance":
            computed[name] = np.log1p(np.maximum(num["distance_from_home_km"], 0.0))
    return computed


def _extract_scaled_matrix(frame, binary_array, prep):
    cyclic_set = set(prep["cyclic_columns"])
    scaled_cols = [col for col in NUMERIC_COLUMNS if col not in cyclic_set]
    extra_map = _compute_extra_features(frame, binary_array, prep["extra_features"])

    columns_data = []
    log1p_set = set(prep["log1p_columns"])
    for col in scaled_cols:
        arr = frame[col].to_numpy(dtype=np.float64, na_value=np.nan)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        if col in log1p_set:
            finite = arr[np.isfinite(arr)]
            if finite.size and (finite < 0).any():
                raise ValueError(f"Cannot apply log1p to negative values in {col}.")
            arr = np.where(np.isfinite(arr), np.log1p(np.where(np.isfinite(arr), arr, 0.0)), np.nan)
        columns_data.append(arr)

    for name in prep["extra_features"]:
        arr = extra_map[name]
        arr = np.where(np.isfinite(arr), arr, np.nan)
        if name in log1p_set:
            finite = arr[np.isfinite(arr)]
            if finite.size and (finite < 0).any():
                raise ValueError(f"Cannot apply log1p to negative values in {name}.")
            arr = np.where(np.isfinite(arr), np.log1p(np.where(np.isfinite(arr), arr, 0.0)), np.nan)
        columns_data.append(arr)

    return np.column_stack(columns_data) if columns_data else np.empty((len(frame), 0), dtype=np.float64)


def _has_custom_preprocessing(prep):
    return bool(
        prep["log1p_columns"]
        or prep["cyclic_columns"]
        or prep["extra_features"]
        or prep["ple_bins"]
        or prep["onehot_categoricals"]
        or prep["discrete_onehot_columns"]
        or prep["tier_b_te"]
    )


def _ple_transform_column(values, edges):
    """Piecewise Linear Encoding (Gorishniy et al., NeurIPS 2022) for one numeric column."""
    edges_arr = np.asarray(edges, dtype=np.float64)
    if edges_arr.size < 2:
        return np.empty((len(values), 0), dtype=np.float32)
    left = edges_arr[:-1]
    width = np.maximum(edges_arr[1:] - left, 1e-12)
    encoded = (values[:, None] - left[None, :]) / width[None, :]
    return np.clip(encoded, 0.0, 1.0).astype(np.float32)


def _extract_tier_b_keys(frame, vocabularies, *, include_pairs=False):
    """Build deterministic int64 key arrays for Per-Value, Digit, and Frequency encoding."""
    keys = {}
    freq_keys = {}
    scaled_int = {}

    for col in NUMERIC_COLUMNS:
        arr = frame[col].to_numpy(dtype=np.float64, na_value=np.nan)
        arr = np.where(np.isfinite(arr), arr, -999999.0)
        c100 = np.rint(arr * 100.0).astype(np.int64)
        scaled_int[col] = c100
        keys[f"{col}__exact"] = c100

    for col in FLOAT_DIGIT_COLUMNS:
        c100 = scaled_int[col]
        int_part = np.floor_divide(c100, 100)
        keys[f"{col}__int"] = int_part
        keys[f"{col}__r1"] = np.floor_divide(c100, 10)
        keys[f"{col}__dec2"] = np.mod(c100, 100)
        keys[f"{col}__cent1"] = np.mod(c100, 10)
        keys[f"{col}__cent2"] = np.mod(np.floor_divide(c100, 10), 10)
        keys[f"{col}__unit"] = np.mod(int_part, 10)
        keys[f"{col}__tens"] = np.mod(np.floor_divide(int_part, 10), 10)
        if col in ("amount_usd", "account_balance_usd"):
            keys[f"{col}__int_mod100"] = np.mod(int_part, 100)
        freq_keys[f"{col}__exact_freq"] = c100
        freq_keys[f"{col}__int_freq"] = int_part

    for col in ("customer_age", "card_age_months"):
        freq_keys[f"{col}__exact_freq"] = scaled_int[col]

    if include_pairs:
        cat_ids = {
            col: frame[col].astype("string").map(vocabularies[col]).fillna(0).to_numpy(dtype=np.int64)
            for col in CATEGORICAL_COLUMNS
        }
        bin_mat = _parse_binary(frame).astype(np.int64)
        bin_map = {col: bin_mat[:, i] for i, col in enumerate(BINARY_COLUMNS)}
        n_flags = bin_mat.sum(axis=1)
        pairs = [
            ("mc_x_auth", cat_ids["merchant_category"], cat_ids["auth_method"]),
            ("mc_x_new", cat_ids["merchant_category"], bin_map["is_new_merchant"]),
            ("auth_x_cvv", cat_ids["auth_method"], np.floor_divide(scaled_int["cvv_retry_count"], 100)),
            ("auth_x_flags", cat_ids["auth_method"], n_flags),
            ("mc_x_flags", cat_ids["merchant_category"], n_flags),
            ("risk_int_x_vel_int", np.floor_divide(scaled_int["merchant_risk_score"], 100), np.floor_divide(scaled_int["velocity_score"], 100)),
            ("age_x_auth", np.floor_divide(scaled_int["customer_age"], 100), cat_ids["auth_method"]),
            ("chan_x_dev", cat_ids["channel"], cat_ids["device_type"]),
            ("new_x_foreign", bin_map["is_new_merchant"], bin_map["is_foreign_transaction"]),
            ("vpn_x_ipmis", bin_map["used_vpn"], bin_map["ip_country_mismatch"]),
        ]
        for pname, a, b in pairs:
            keys[f"pair__{pname}"] = a * 100003 + b

    return keys, freq_keys


def _logit(p):
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _lookup_sorted(keys_arr, sorted_keys, sorted_vals, default_val=0.0):
    if sorted_keys.size == 0:
        return np.full(len(keys_arr), default_val, dtype=np.float64)
    idx = np.searchsorted(sorted_keys, keys_arr)
    idx_safe = np.minimum(idx, sorted_keys.size - 1)
    matched = sorted_keys[idx_safe] == keys_arr
    return np.where(matched, sorted_vals[idx_safe], default_val)


def _fit_nested_te_and_freq(frame, labels, vocabularies, prep, seed):
    y = labels.to_numpy(dtype=np.float64)
    n = len(y)
    total_y = float(y.sum())
    prior = total_y / max(n, 1)
    prior_logit = float(_logit(prior))
    m = float(prep["te_smoothing"])

    te_keys_map, freq_keys_map = _extract_tier_b_keys(
        frame, vocabularies, include_pairs=prep["te_pairs"]
    )

    n_splits = min(prep["te_folds"], int(min(y.sum(), n - y.sum())))
    if n_splits < 2:
        n_splits = 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = [val_idx for _, val_idx in skf.split(np.zeros(n), y.astype(np.int64))]

    oof_cols = []
    te_tables = {}
    te_names = list(te_keys_map.keys())
    for name in te_names:
        k = te_keys_map[name]
        uniq, inv, total_cnt = np.unique(k, return_inverse=True, return_counts=True)
        total_sum = np.bincount(inv, weights=y, minlength=uniq.size)

        oof_vals = np.zeros(n, dtype=np.float64)
        for val_idx in fold_indices:
            fold_inv = inv[val_idx]
            fold_cnt = np.bincount(fold_inv, minlength=uniq.size)
            fold_sum = np.bincount(fold_inv, weights=y[val_idx], minlength=uniq.size)
            oof_cnt = total_cnt[fold_inv] - fold_cnt[fold_inv]
            oof_sum = total_sum[fold_inv] - fold_sum[fold_inv]
            fold_n = n - len(val_idx)
            fold_prior = (total_y - float(y[val_idx].sum())) / max(fold_n, 1)
            fold_prior_logit = float(_logit(fold_prior))
            p = (oof_sum + m * fold_prior) / (oof_cnt + m)
            oof_vals[val_idx] = _logit(p) - fold_prior_logit

        full_p = (total_sum + m * prior) / (total_cnt + m)
        full_woe = (_logit(full_p) - prior_logit).astype(np.float32)
        te_tables[name] = (
            torch.from_numpy(uniq.astype(np.int64)),
            torch.from_numpy(full_woe),
        )
        oof_cols.append(oof_vals)

    freq_tables = {}
    freq_names = list(freq_keys_map.keys())
    for name in freq_names:
        k = freq_keys_map[name]
        uniq, inv, counts = np.unique(k, return_inverse=True, return_counts=True)
        log_counts = np.log1p(counts.astype(np.float64))
        oof_cols.append(log_counts[inv])
        freq_tables[name] = (
            torch.from_numpy(uniq.astype(np.int64)),
            torch.from_numpy(log_counts.astype(np.float32)),
        )

    mat = np.column_stack(oof_cols)
    te_mean = mat.mean(axis=0)
    te_scale = mat.std(axis=0, ddof=0)
    te_scale = np.where(te_scale == 0.0, 1.0, te_scale)
    standardized_oof = ((mat - te_mean) / te_scale).astype(np.float32)

    te_state = {
        "te_names": te_names,
        "freq_names": freq_names,
        "te_tables": te_tables,
        "freq_tables": freq_tables,
        "mean": te_mean.tolist(),
        "scale": te_scale.tolist(),
    }
    return te_state, standardized_oof


def _transform_tier_b(frame, state):
    prep = state["preprocessing"]
    te_state = state["tier_b_state"]
    te_keys_map, freq_keys_map = _extract_tier_b_keys(
        frame, state["vocabularies"], include_pairs=prep["te_pairs"]
    )
    cols = []
    for name in te_state["te_names"]:
        sk, sv = te_state["te_tables"][name]
        cols.append(_lookup_sorted(te_keys_map[name], sk.numpy(), sv.numpy(), default_val=0.0))
    for name in te_state["freq_names"]:
        sk, sv = te_state["freq_tables"][name]
        cols.append(_lookup_sorted(freq_keys_map[name], sk.numpy(), sv.numpy(), default_val=0.0))
    mat = np.column_stack(cols)
    mean = np.asarray(te_state["mean"], dtype=np.float64)
    scale = np.asarray(te_state["scale"], dtype=np.float64)
    return ((mat - mean) / scale).astype(np.float32)


def training_data_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_data_splits(path, *, validation_fraction, seed, test_fraction=0):
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame:
        raise ValueError(f"train.csv is missing {TARGET_COLUMN}.")

    labels = frame.pop(TARGET_COLUMN)
    if not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("is_fraud must contain both binary classes (0 and 1).")

    test_split = None
    if test_fraction:
        test_count = round(len(frame) * test_fraction)
        frame, test_frame, labels, test_labels = train_test_split(
            frame, labels, test_size=test_count, stratify=labels, random_state=seed
        )
        test_split = (test_frame, test_labels)
        # Use the original row count for the validation fraction.
        validation_size = round((len(frame) + len(test_frame)) * validation_fraction)
    else:
        # Preserve the original two-way split for older checkpoints.
        validation_size = validation_fraction

    train_frame, val_frame, train_labels, val_labels = train_test_split(
        frame, labels, test_size=validation_size, stratify=labels, random_state=seed
    )
    splits = {"train": (train_frame, train_labels), "validation": (val_frame, val_labels)}
    if test_split is not None:
        splits["test"] = test_split
    if any(split_labels.nunique() != 2 for _, split_labels in splits.values()):
        raise ValueError("Each split needs both binary classes.")
    return splits


def read_training_split(path, *, validation_fraction, seed):
    splits = read_data_splits(path, validation_fraction=validation_fraction, seed=seed)
    train_frame, train_labels = splits["train"]
    val_frame, val_labels = splits["validation"]
    return train_frame, val_frame, train_labels, val_labels


def validate_features(frame):
    required = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(sorted(missing))}")


def _build_base_preprocessor(frame, prep):
    if not _has_custom_preprocessing(prep):
        numeric = frame[NUMERIC_COLUMNS].astype("float64").replace([np.inf, -np.inf], np.nan)
        mean = numeric.mean().fillna(0.0)
        scale = numeric.std(ddof=0).fillna(0.0).replace(0.0, 1.0)
        state = {"mean": mean.tolist(), "scale": scale.tolist(), "vocabularies": {}}
    else:
        binary_array = _parse_binary(frame)
        matrix = _extract_scaled_matrix(frame, binary_array, prep)
        means = []
        scales = []
        ple_edges = []
        for idx in range(matrix.shape[1]):
            col = matrix[:, idx]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                means.append(0.0)
                scales.append(1.0)
                if prep["ple_bins"] > 0:
                    ple_edges.append([])
            else:
                m = float(finite.mean())
                s = float(finite.std(ddof=0))
                means.append(m)
                scales.append(s if s != 0.0 else 1.0)
                if prep["ple_bins"] > 0:
                    quantiles = np.linspace(0.0, 1.0, prep["ple_bins"] + 1)
                    edges = np.unique(np.quantile(finite, quantiles)).tolist()
                    ple_edges.append(edges if len(edges) >= 2 else [])

        cyclic_fill = {}
        for col in NUMERIC_COLUMNS:
            if col in prep["cyclic_columns"]:
                arr = frame[col].to_numpy(dtype=np.float64, na_value=np.nan)
                finite = arr[np.isfinite(arr)]
                cyclic_fill[col] = float(finite.mean()) if finite.size else 0.0

        discrete_vocabs = {}
        for col in prep["discrete_onehot_columns"]:
            arr = frame[col].to_numpy(dtype=np.float64, na_value=np.nan)
            finite = arr[np.isfinite(arr)]
            vals = sorted(np.unique(np.rint(finite).astype(np.int64)).tolist())
            discrete_vocabs[col] = vals

        state = {
            "mean": means,
            "scale": scales,
            "vocabularies": {},
            "preprocessing": prep,
            "cyclic_fill": cyclic_fill,
        }
        if prep["ple_bins"] > 0:
            state["ple_edges"] = ple_edges
        if discrete_vocabs:
            state["discrete_vocabs"] = discrete_vocabs

    # Index 0 is reserved for missing or unseen categories.
    vocabularies = {}
    for column in CATEGORICAL_COLUMNS:
        values = sorted(frame[column].dropna().astype("string").unique().tolist())
        vocabularies[column] = {value: index for index, value in enumerate(values, start=1)}
    state["vocabularies"] = vocabularies
    return state


def fit_preprocessor(frame, preprocessing=None):
    validate_features(frame)
    prep = normalize_preprocessing_options(preprocessing)
    if prep["tier_b_te"]:
        raise ValueError("tier_b_te requires labels; call fit_transform_train() instead.")
    return _build_base_preprocessor(frame, prep)


def _transform_base(frame, state):
    binary_array = _parse_binary(frame)
    prep = normalize_preprocessing_options(state.get("preprocessing"))
    encoded = [
        frame[column].astype("string").map(state["vocabularies"][column]).fillna(0).to_numpy(
            dtype=np.int64
        )
        for column in CATEGORICAL_COLUMNS
    ]
    categorical = np.column_stack(encoded)

    if not _has_custom_preprocessing(prep):
        numeric = frame[NUMERIC_COLUMNS].to_numpy(dtype=np.float64, na_value=np.nan)
        mean = np.asarray(state["mean"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        numeric = np.where(np.isfinite(numeric), numeric, mean)
        standardized = ((numeric - mean) / scale).astype(np.float32)
        return [standardized, binary_array], categorical

    matrix = _extract_scaled_matrix(frame, binary_array, prep)
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, mean)
    standardized = ((matrix - mean) / scale).astype(np.float32)
    dense_parts = [standardized]

    if prep["ple_bins"] > 0:
        ple_edges = state.get("ple_edges", [])
        ple_parts = [
            _ple_transform_column(matrix[:, idx], edges)
            for idx, edges in enumerate(ple_edges)
            if len(edges) >= 2
        ]
        if ple_parts:
            dense_parts.append(np.concatenate(ple_parts, axis=1))

    cyclic_cols = prep["cyclic_columns"]
    if cyclic_cols:
        cyclic_fill = state.get("cyclic_fill", {})
        cyclic_arrays = []
        for col in NUMERIC_COLUMNS:
            if col in cyclic_cols:
                period = cyclic_cols[col]
                arr = frame[col].to_numpy(dtype=np.float64, na_value=np.nan)
                fill_value = cyclic_fill.get(col, 0.0)
                arr = np.where(np.isfinite(arr), arr, fill_value)
                angle = (2.0 * np.pi / period) * arr
                cyclic_arrays.append(np.sin(angle).astype(np.float32))
                cyclic_arrays.append(np.cos(angle).astype(np.float32))
        dense_parts.append(np.column_stack(cyclic_arrays))

    if prep["onehot_categoricals"]:
        oh_parts = []
        for col_idx, column in enumerate(CATEGORICAL_COLUMNS):
            num_classes = len(state["vocabularies"][column]) + 1
            oh = np.zeros((len(frame), num_classes), dtype=np.float32)
            oh[np.arange(len(frame)), categorical[:, col_idx]] = 1.0
            oh_parts.append(oh)
        dense_parts.append(np.concatenate(oh_parts, axis=1))

    if prep["discrete_onehot_columns"]:
        discrete_vocabs = state.get("discrete_vocabs", {})
        doh_parts = []
        for col in prep["discrete_onehot_columns"]:
            vals = discrete_vocabs[col]
            val_to_idx = {v: i + 1 for i, v in enumerate(vals)}
            arr = frame[col].to_numpy(dtype=np.float64, na_value=np.nan)
            rounded = np.where(np.isfinite(arr), np.rint(arr), -999999).astype(np.int64)
            mapped = pd.Series(rounded).map(val_to_idx).fillna(0).to_numpy(dtype=np.int64)
            oh = np.zeros((len(frame), len(vals) + 1), dtype=np.float32)
            oh[np.arange(len(frame)), mapped] = 1.0
            doh_parts.append(oh)
        if doh_parts:
            dense_parts.append(np.concatenate(doh_parts, axis=1))

    dense_parts.append(binary_array)
    return dense_parts, categorical


def fit_transform_train(frame, labels, preprocessing=None, *, seed=42):
    validate_features(frame)
    prep = normalize_preprocessing_options(preprocessing)
    state = _build_base_preprocessor(frame, prep)
    dense_parts, categorical = _transform_base(frame, state)
    if prep["tier_b_te"]:
        te_state, oof_tier_b = _fit_nested_te_and_freq(
            frame, labels, state["vocabularies"], prep, seed=seed
        )
        state["tier_b_state"] = te_state
        dense_parts.append(oof_tier_b)
    return state, np.concatenate(dense_parts, axis=1), categorical


def transform(frame, state):
    validate_features(frame)
    dense_parts, categorical = _transform_base(frame, state)
    prep = normalize_preprocessing_options(state.get("preprocessing"))
    if prep["tier_b_te"]:
        dense_parts.append(_transform_tier_b(frame, state))
    return np.concatenate(dense_parts, axis=1), categorical


def dense_feature_count(state):
    prep = normalize_preprocessing_options(state.get("preprocessing"))
    if not _has_custom_preprocessing(prep):
        return DENSE_FEATURE_COUNT
    total = len(state["mean"]) + 2 * len(prep["cyclic_columns"]) + len(BINARY_COLUMNS)
    if prep["ple_bins"] > 0:
        total += sum(max(0, len(edges) - 1) for edges in state.get("ple_edges", []))
    if prep["onehot_categoricals"]:
        total += sum(len(state["vocabularies"][col]) + 1 for col in CATEGORICAL_COLUMNS)
    if prep["discrete_onehot_columns"]:
        discrete_vocabs = state.get("discrete_vocabs", {})
        total += sum(len(discrete_vocabs[col]) + 1 for col in prep["discrete_onehot_columns"])
    if prep["tier_b_te"]:
        total += len(state["tier_b_state"]["mean"])
    return total


def embedding_sizes(state):
    return [len(state["vocabularies"][column]) + 1 for column in CATEGORICAL_COLUMNS]
