"""Feature schema and preprocessing fitted exclusively on training rows."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


def fit_preprocessor(frame):
    validate_features(frame)
    numeric = frame[NUMERIC_COLUMNS].astype("float64").replace([np.inf, -np.inf], np.nan)
    mean = numeric.mean().fillna(0.0)
    scale = numeric.std(ddof=0).fillna(0.0).replace(0.0, 1.0)

    # Index 0 is reserved for missing or unseen categories.
    vocabularies = {}
    for column in CATEGORICAL_COLUMNS:
        values = sorted(frame[column].dropna().astype("string").unique().tolist())
        vocabularies[column] = {value: index for index, value in enumerate(values, start=1)}

    return {"mean": mean.tolist(), "scale": scale.tolist(), "vocabularies": vocabularies}


def transform(frame, state):
    validate_features(frame)
    numeric = frame[NUMERIC_COLUMNS].to_numpy(dtype=np.float64, na_value=np.nan)
    mean = np.asarray(state["mean"])
    numeric = np.where(np.isfinite(numeric), numeric, mean)
    standardized = ((numeric - mean) / state["scale"]).astype(np.float32)

    binary = frame[BINARY_COLUMNS].apply(
        lambda column: column.astype("string").str.strip().str.lower().map(BINARY_VALUES)
    )
    invalid = binary.columns[binary.isna().any()].tolist()
    if invalid:
        raise ValueError(f"Expected True/False or 0/1 in: {', '.join(invalid)}")

    encoded = [
        frame[column].astype("string").map(state["vocabularies"][column]).fillna(0).to_numpy(
            dtype=np.int64
        )
        for column in CATEGORICAL_COLUMNS
    ]
    dense = np.concatenate((standardized, binary.to_numpy(dtype=np.float32)), axis=1)
    return dense, np.column_stack(encoded)


def embedding_sizes(state):
    return [len(state["vocabularies"][column]) + 1 for column in CATEGORICAL_COLUMNS]
