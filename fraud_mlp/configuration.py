"""Load TOML settings for training and prediction, with CLI overrides."""

import math
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"
DEVICES = ("auto", "cpu", "cuda")


def positive_int(value):
    number = int(value) if isinstance(value, str) else value
    if type(number) is not int or number < 1:
        raise ValueError("Expected a positive integer.")
    return number


def add_common_arguments(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="TOML configuration file (default: the project's config.toml)")
    parser.add_argument("--data-dir", type=Path, help="Directory containing the Kaggle CSV files")
    parser.add_argument("--checkpoint", type=Path, help="Path to the model checkpoint")
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--device", choices=DEVICES)


def parse_options(parser, *, training, prediction=True):
    args = parser.parse_args()
    try:
        with args.config.open("rb") as stream:
            config = tomllib.load(stream)

        paths = ["data_dir", "checkpoint"]
        if not training and prediction:
            paths.append("output")
        base_dir = args.config.resolve().parent
        for name in paths:
            if getattr(args, name) is None:
                setattr(args, name, (base_dir / config["paths"][name]).resolve())

        if args.device is None:
            args.device = config["runtime"]["device"]
        if args.device not in DEVICES:
            raise ValueError("runtime.device must be auto, cpu or cuda.")
        if args.batch_size is None:
            args.batch_size = positive_int(config["runtime"]["batch_size"])

        if training:
            apply_training_options(args, config["training"], config["model"])
        elif args.metrics:
            args.training_config = config.get("training", {})
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(f"Configuration {args.config}: {error}")
    return args


def validation_split_options(seed, fraction, test_fraction=0):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("training.seed must be an integer from 0 through 2**32 - 1.")
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError("training.validation_fraction must be between 0 and 1, excluding both.")
    if type(test_fraction) not in (int, float) or not 0 <= test_fraction < 1:
        raise ValueError("training.test_fraction must be at least 0 and less than 1.")
    if fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")
    return {"seed": seed, "validation_fraction": fraction, "test_fraction": test_fraction}


def apply_training_options(args, training, model):
    for name in ("epochs", "patience"):
        if getattr(args, name) is None:
            setattr(args, name, positive_int(training[name]))
    if args.lr is None:
        args.lr = training["lr"]
    if type(args.lr) not in (int, float) or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("training.lr must be a positive finite number.")
    if args.seed is None:
        args.seed = training["seed"]
    args.validation_split = validation_split_options(
        args.seed, training["validation_fraction"], training.get("test_fraction", 0)
    )

    hidden_sizes = model["hidden_sizes"]
    if not isinstance(hidden_sizes, list) or not hidden_sizes:
        raise ValueError("model.hidden_sizes must be a nonempty array of positive integers.")
    dropout = model["dropout"]
    if type(dropout) not in (int, float) or not 0 <= dropout < 1:
        raise ValueError("model.dropout must be at least 0 and less than 1.")
    args.model_options = {
        "embedding_dim": positive_int(model["embedding_dim"]),
        "hidden_sizes": [positive_int(size) for size in hidden_sizes],
        "dropout": dropout,
    }
