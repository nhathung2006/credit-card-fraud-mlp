"""Run inference and export a Kaggle submission in the sample's ID order."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .configuration import add_common_arguments, parse_options
from .data import ID_COLUMN, TARGET_COLUMN, embedding_sizes, transform
from .evaluate import add_evaluation_arguments, print_metrics
from .model import FraudMLP
from .runtime import atomic_output, predict_logits, resolve_device


def predict(args):
    device = resolve_device(args.device)
    frame = pd.read_csv(args.data_dir / "test.csv")
    submission = pd.read_csv(args.data_dir / "sample_submission.csv", usecols=[ID_COLUMN])
    if ID_COLUMN not in frame:
        raise ValueError(f"test.csv is missing {ID_COLUMN}.")
    test_ids = pd.Index(frame[ID_COLUMN])
    sample_ids = pd.Index(submission[ID_COLUMN])
    if (
        not len(test_ids)
        or len(test_ids) != len(sample_ids)
        or test_ids.hasnans
        or sample_ids.hasnans
        or not test_ids.is_unique
        or not sample_ids.is_unique
        or not sample_ids.isin(test_ids).all()
    ):
        raise ValueError("test.csv and sample_submission.csv must have the same unique IDs.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    preprocessor = checkpoint["preprocessor"]
    dense, categorical = transform(frame, preprocessor)
    model = FraudMLP(embedding_sizes(preprocessor), **checkpoint.get("model_options", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if args.metrics:
        print_metrics(args, model, checkpoint, device)

    dataset = TensorDataset(torch.from_numpy(dense), torch.from_numpy(categorical))
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=device.type == "cuda")
    probabilities = predict_logits(model, loader, device).sigmoid().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("The model returned non-finite probabilities.")

    submission[TARGET_COLUMN] = pd.Series(probabilities, index=test_ids).reindex(sample_ids).to_numpy()
    with atomic_output(args.output) as temporary:
        submission.to_csv(temporary, index=False)
    print(f"Saved {args.output} | Device: {device} | Predictions: {len(submission):,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--output", type=Path, help="Path to the submission CSV")
    parser.add_argument("--metrics", action="store_true",
                        help="Print recall on the checkpoint's held-out labeled split")
    add_evaluation_arguments(parser, default_split="auto")
    predict(parse_options(parser, training=False))


if __name__ == "__main__":
    main()
