"""Load TOML settings for training and prediction, with CLI overrides."""

import math
from pathlib import Path
import tomllib

from .data import normalize_preprocessing_options


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"
DEVICES = ("auto", "cpu", "cuda")
ARCHITECTURES = ("mlp", "resmlp")
NORMS = ("none", "batch", "layer")
ACTIVATIONS = ("relu", "silu", "gelu", "leaky_relu")
SCHEDULERS = ("none", "cosine", "plateau")


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
            apply_training_options(
                args, config["training"], config["model"], config.get("preprocessing")
            )
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


def apply_training_options(args, training, model, preprocessing=None):
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

    weight_decay = training.get("weight_decay", 0.01)
    if type(weight_decay) not in (int, float) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("training.weight_decay must be a non-negative finite number.")
    args.weight_decay = float(weight_decay)

    scheduler = training.get("scheduler", "none")
    if scheduler not in SCHEDULERS:
        raise ValueError(f"training.scheduler must be one of {', '.join(SCHEDULERS)}.")
    args.scheduler = scheduler

    min_lr = training.get("min_lr", 1e-6)
    if type(min_lr) not in (int, float) or not math.isfinite(min_lr) or min_lr < 0:
        raise ValueError("training.min_lr must be a non-negative finite number.")
    args.min_lr = float(min_lr)

    grad_clip = training.get("grad_clip", 0.0)
    if type(grad_clip) not in (int, float) or not math.isfinite(grad_clip) or grad_clip < 0:
        raise ValueError("training.grad_clip must be a non-negative finite number.")
    args.grad_clip = float(grad_clip)

    hidden_sizes = model["hidden_sizes"]
    if not isinstance(hidden_sizes, list) or not hidden_sizes:
        raise ValueError("model.hidden_sizes must be a nonempty array of positive integers.")
    dropout = model["dropout"]
    if type(dropout) not in (int, float) or not 0 <= dropout < 1:
        raise ValueError("model.dropout must be at least 0 and less than 1.")

    model_options = {
        "embedding_dim": positive_int(model["embedding_dim"]),
        "hidden_sizes": [positive_int(size) for size in hidden_sizes],
        "dropout": dropout,
    }
    architecture = model.get("architecture", "mlp")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"model.architecture must be one of {', '.join(ARCHITECTURES)}.")
    norm = model.get("norm", "none")
    if norm not in NORMS:
        raise ValueError(f"model.norm must be one of {', '.join(NORMS)}.")
    input_norm = model.get("input_norm", "none")
    if input_norm not in NORMS:
        raise ValueError(f"model.input_norm must be one of {', '.join(NORMS)}.")
    activation = model.get("activation", "relu")
    if activation not in ACTIVATIONS:
        raise ValueError(f"model.activation must be one of {', '.join(ACTIVATIONS)}.")

    if architecture != "mlp":
        model_options["architecture"] = architecture
        model_options["residual_blocks"] = positive_int(model.get("residual_blocks", 3))
        model_options["residual_dim"] = positive_int(model.get("residual_dim", 256))
        factor = model.get("residual_hidden_factor", 1.0)
        if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
            raise ValueError("model.residual_hidden_factor must be a positive finite number.")
        model_options["residual_hidden_factor"] = float(factor)
    if norm != "none":
        model_options["norm"] = norm
    if input_norm != "none":
        model_options["input_norm"] = input_norm
    if activation != "relu":
        model_options["activation"] = activation
    if "use_embeddings" in model and not bool(model["use_embeddings"]):
        model_options["use_embeddings"] = False
    if bool(model.get("wide_skip", False)):
        model_options["wide_skip"] = True

    args.model_options = model_options
    args.preprocessing_options = normalize_preprocessing_options(preprocessing)
