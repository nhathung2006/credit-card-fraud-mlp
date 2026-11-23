import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from uuid import uuid4

import numpy as np
import pandas as pd


def example_frame(size):
    rows = np.arange(size)
    return pd.DataFrame({
        "id": rows,
        "amount_usd": 20.0 + 50.0 * (rows % 2),
        "hours_since_last_txn": 4.52,
        "txn_count_last_24h": 4,
        "distance_from_home_km": 17.87,
        "card_age_months": 37,
        "customer_age": 45,
        "account_balance_usd": 1483.49,
        "cvv_retry_count": 0,
        "velocity_score": 15.4,
        "time_of_day_hour": 21,
        "day_of_week": 3,
        "merchant_risk_score": 22.1,
        "prior_disputes": 0,
        "merchant_category": "Fuel",
        "card_type": "Visa",
        "auth_method": "OTP",
        "channel": "Online",
        "device_type": "iPhone",
        "is_foreign_transaction": False,
        "is_new_merchant": False,
        "used_vpn": False,
        "ip_country_mismatch": False,
        "billing_shipping_mismatch": False,
        "is_ai_generated_scam_attempt": False,
    })


class PipelineTest(unittest.TestCase):
    def test_stratified_splits_are_disjoint_complete_and_repeatable(self):
        from fraud_mlp.data import read_data_splits

        folder = self.make_data_folder()
        splits = read_data_splits(
            folder / "train.csv", validation_fraction=0.15, test_fraction=0.15, seed=42
        )
        self.assertEqual([len(frame) for frame, _ in splits.values()], [56, 12, 12])
        groups = [set(frame["id"]) for frame, _ in splits.values()]
        self.assertEqual(set.union(*groups), set(range(80)))
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        for frame, labels in splits.values():
            self.assertEqual(frame.index.tolist(), labels.index.tolist())
            self.assertEqual(labels.mean(), 0.5)
        repeated = read_data_splits(
            folder / "train.csv", validation_fraction=0.15, test_fraction=0.15, seed=42
        )
        for name in splits:
            self.assertEqual(splits[name][0]["id"].tolist(), repeated[name][0]["id"].tolist())

    def test_transform_reuses_training_statistics_and_maps_unseen_categories(self):
        from fraud_mlp.data import fit_preprocessor, transform

        train = example_frame(4)
        held_out = example_frame(2)
        held_out["amount_usd"] = [95.0, np.nan]
        held_out["merchant_category"] = ["New Category", "Fuel"]
        held_out["used_vpn"] = [True, False]

        dense, categorical = transform(held_out, fit_preprocessor(train))

        self.assertEqual(dense.shape, (2, 19))
        self.assertEqual(categorical.shape, (2, 5))
        np.testing.assert_allclose(dense[:, 0], [2.0, 0.0])
        np.testing.assert_array_equal(dense[:, 15], [1.0, 0.0])
        np.testing.assert_array_equal(categorical[:, 0], [0, 1])

    def make_data_folder(self):
        project = Path(__file__).resolve().parents[1]
        folder = project / f".test-{uuid4().hex}"
        folder.mkdir()
        self.addCleanup(shutil.rmtree, folder)

        train = example_frame(80)
        train["is_fraud"] = (train["id"] % 2).astype(float)
        train.to_csv(folder / "train.csv", index=False)
        test = example_frame(3)
        test["id"] = [100, 101, 102]
        test.to_csv(folder / "test.csv", index=False)
        pd.DataFrame({"id": [102, 100, 101], "is_fraud": 0.5}).to_csv(
            folder / "sample_submission.csv", index=False
        )
        return folder

    def run_cli(self, module, *flags):
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        result = subprocess.run(
            [sys.executable, "-B", "-m", f"fraud_mlp.{module}", *flags],
            cwd=Path(__file__).resolve().parents[1], env=env,
            text=True, capture_output=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_cli_trains_and_writes_probabilities_in_sample_order(self):
        folder = self.make_data_folder()
        checkpoint = folder / "artifacts" / "best.pt"
        output = folder / "artifacts" / "submission.csv"
        common = ["--data-dir", str(folder), "--device", "cpu",
                  "--checkpoint", str(checkpoint), "--batch-size", "16"]
        self.run_cli("train", *common, "--epochs", "1")
        self.run_cli("predict", *common, "--output", str(output))

        submission = pd.read_csv(output)
        self.assertEqual(list(submission.columns), ["id", "is_fraud"])
        self.assertEqual(submission["id"].tolist(), [102, 100, 101])
        self.assertTrue(submission["is_fraud"].between(0, 1).all())
        self.assertGreater(submission["is_fraud"].nunique(), 1)

    def test_custom_config_and_cli_override_with_checkpoint_architecture(self):
        import torch

        folder = self.make_data_folder()
        config = folder / "experiment.toml"
        settings = """[paths]
data_dir = "."
checkpoint = "run/model.pt"
output = "run/results.csv"

[runtime]
device = "cpu"
batch_size = 16

[training]
epochs = 2
patience = 2
lr = 0.001
seed = 42
validation_fraction = 0.15
test_fraction = 0.15

[model]
embedding_dim = 2
hidden_sizes = [12, 8]
dropout = 0.1
"""
        config.write_text(settings, encoding="utf-8")
        stdout = self.run_cli("train", "--config", str(config), "--epochs", "1")
        self.assertIn("Train: 56 | Validation: 12 | Test: 12", stdout)
        self.assertIn("Epoch 01", stdout)
        self.assertNotIn("Epoch 02", stdout)
        checkpoint = torch.load(folder / "run/model.pt", weights_only=True)
        weights = checkpoint["model_state_dict"]
        self.assertEqual(tuple(weights["classifier.0.weight"].shape), (12, 29))
        self.assertEqual(tuple(weights["classifier.3.weight"].shape), (8, 12))

        changed_settings = settings.replace("[12, 8]", "[64, 32]").replace("0.15", "0.2")
        config.write_text(changed_settings, encoding="utf-8")
        stdout = self.run_cli("predict", "--config", str(config), "--metrics", "--threshold", "0")
        self.assertIn("Internal test metrics | Rows: 12 | Threshold: 0", stdout)
        self.assertNotIn("Accuracy:", stdout)
        self.assertNotIn("Precision:", stdout)
        self.assertIn("Recall: 1.000000", stdout)
        stdout = self.run_cli("evaluate", "--config", str(config), "--threshold", "0")
        self.assertIn("Internal test metrics | Rows: 12 | Threshold: 0", stdout)
        self.assertIn("Recall: 1.000000", stdout)
        self.assertNotIn("Accuracy:", stdout)
        self.assertNotIn("Precision:", stdout)
        submission = pd.read_csv(folder / "run/results.csv")
        self.assertEqual(submission["id"].tolist(), [102, 100, 101])
        self.assertTrue(submission["is_fraud"].between(0, 1).all())

    def test_log1p_cyclic_resmlp_and_val_predictions(self):
        folder = self.make_data_folder()
        config = folder / "resmlp.toml"
        val_preds = folder / "run/val_preds.csv"
        settings = """[paths]
data_dir = "."
checkpoint = "run/resmlp.pt"
output = "run/results.csv"

[runtime]
device = "cpu"
batch_size = 16

[training]
epochs = 2
patience = 2
lr = 0.001
seed = 42
validation_fraction = 0.2
test_fraction = 0.0
scheduler = "cosine"

[model]
architecture = "resmlp"
embedding_dim = 4
hidden_sizes = [16]
dropout = 0.1
norm = "layer"
activation = "silu"
residual_blocks = 2
residual_dim = 16

[preprocessing]
log1p_columns = ["amount_usd", "account_balance_usd"]
extra_features = ["amount_to_balance_ratio", "mismatch_count"]

[preprocessing.cyclic_columns]
time_of_day_hour = 24.0
"""
        config.write_text(settings, encoding="utf-8")
        self.run_cli("train", "--config", str(config), "--save-val-predictions", str(val_preds))
        self.assertTrue(val_preds.exists())
        val_df = pd.read_csv(val_preds)
        self.assertEqual(list(val_df.columns), ["id", "is_fraud", "pred_prob"])
        self.assertEqual(len(val_df), 16)
        self.assertTrue(val_df["pred_prob"].between(0, 1).all())

        eval_preds = folder / "run/eval_preds.csv"
        self.run_cli("evaluate", "--config", str(config), "--save-predictions", str(eval_preds))
        eval_df = pd.read_csv(eval_preds)
        np.testing.assert_allclose(val_df["pred_prob"].to_numpy(), eval_df["pred_prob"].to_numpy(), rtol=1e-5, atol=1e-6)

        self.run_cli("predict", "--config", str(config))
        submission = pd.read_csv(folder / "run/results.csv")
        self.assertEqual(submission["id"].tolist(), [102, 100, 101])
        self.assertTrue(submission["is_fraud"].between(0, 1).all())

    def test_ple_onehot_wide_skip_and_nested_tier_b_te(self):
        folder = self.make_data_folder()
        config = folder / "tier_b.toml"
        settings = """[paths]
data_dir = "."
checkpoint = "run/tier_b.pt"
output = "run/results.csv"

[runtime]
device = "cpu"
batch_size = 16

[training]
epochs = 2
patience = 2
lr = 0.001
seed = 42
validation_fraction = 0.2
test_fraction = 0.0
scheduler = "cosine"
weight_decay = 0.001

[model]
architecture = "resmlp"
embedding_dim = 4
use_embeddings = false
wide_skip = true
hidden_sizes = [16]
dropout = 0.2
norm = "layer"
activation = "silu"
residual_blocks = 2
residual_dim = 16

[preprocessing]
ple_bins = 4
onehot_categoricals = true
discrete_onehot_columns = ["customer_age", "time_of_day_hour"]
extra_features = ["mismatch_count", "amount_to_balance_ratio", "log_amount_to_balance", "amount_gt_balance"]
tier_b_te = true
te_smoothing = 10.0
te_folds = 3
te_pairs = true

[preprocessing.cyclic_columns]
time_of_day_hour = 24.0
"""
        config.write_text(settings, encoding="utf-8")
        self.run_cli("train", "--config", str(config))
        self.run_cli("evaluate", "--config", str(config))
        self.run_cli("predict", "--config", str(config))
        submission = pd.read_csv(folder / "run/results.csv")
        self.assertEqual(submission["id"].tolist(), [102, 100, 101])
        self.assertTrue(submission["is_fraud"].between(0, 1).all())
