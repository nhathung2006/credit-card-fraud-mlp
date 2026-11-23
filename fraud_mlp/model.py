"""MLP and Residual MLP architectures for one binary classification logit per transaction."""

import torch
from torch import nn

from .data import DENSE_FEATURE_COUNT, dense_feature_count, embedding_sizes


def _make_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    raise ValueError(f"Unsupported activation: {name}")


def _make_norm(name, dim):
    if name == "none":
        return None
    if name == "batch":
        return nn.BatchNorm1d(dim)
    if name == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported normalization: {name}")


class ResidualBlock(nn.Module):
    """Pre-norm tabular residual block with skip connection."""

    def __init__(self, dim, hidden_dim, *, dropout=0.2, norm="layer", activation="silu"):
        super().__init__()
        layers = []
        norm_layer = _make_norm(norm, dim)
        if norm_layer is not None:
            layers.append(norm_layer)
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(_make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class FraudMLP(nn.Module):
    # Defaults also restore checkpoints created before config.toml was added.
    def __init__(
        self,
        category_sizes,
        *,
        embedding_dim=4,
        hidden_sizes=(256, 128, 64),
        dropout=0.2,
        dense_feature_count=DENSE_FEATURE_COUNT,
        architecture="mlp",
        norm="none",
        input_norm="none",
        activation="relu",
        residual_blocks=3,
        residual_dim=256,
        residual_hidden_factor=1.0,
        use_embeddings=True,
        wide_skip=False,
    ):
        super().__init__()
        self.architecture = architecture
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.embeddings = nn.ModuleList(
                nn.Embedding(size, embedding_dim, padding_idx=0) for size in category_sizes
            )
            input_size = dense_feature_count + len(category_sizes) * embedding_dim
        else:
            self.embeddings = nn.ModuleList()
            input_size = dense_feature_count

        self.input_norm = _make_norm(input_norm, input_size)
        self.wide = nn.Linear(input_size, 1) if wide_skip else None

        if architecture == "mlp":
            layers = []
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(input_size, hidden_size))
                norm_layer = _make_norm(norm, hidden_size)
                if norm_layer is not None:
                    layers.append(norm_layer)
                layers.append(_make_activation(activation))
                layers.append(nn.Dropout(dropout))
                input_size = hidden_size
            layers.append(nn.Linear(input_size, 1))
            self.classifier = nn.Sequential(*layers)
        elif architecture == "resmlp":
            block_norm = norm if norm != "none" else "layer"
            hidden_dim = max(1, int(round(residual_dim * residual_hidden_factor)))
            self.input_proj = nn.Linear(input_size, residual_dim)
            self.blocks = nn.ModuleList(
                ResidualBlock(
                    residual_dim,
                    hidden_dim,
                    dropout=dropout,
                    norm=block_norm,
                    activation=activation,
                )
                for _ in range(residual_blocks)
            )
            head_layers = []
            head_norm = _make_norm(block_norm, residual_dim)
            if head_norm is not None:
                head_layers.append(head_norm)
            head_layers.extend([_make_activation(activation), nn.Linear(residual_dim, 1)])
            self.head = nn.Sequential(*head_layers)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, dense, categorical):
        if self.use_embeddings:
            embeddings = [
                embedding(categorical[:, index])
                for index, embedding in enumerate(self.embeddings)
            ]
            features = torch.cat([dense, *embeddings], dim=1)
        else:
            features = dense
        if self.input_norm is not None:
            features = self.input_norm(features)
        if self.architecture == "mlp":
            out = self.classifier(features).squeeze(1)
        else:
            hidden = self.input_proj(features)
            for block in self.blocks:
                hidden = block(hidden)
            out = self.head(hidden).squeeze(1)
        if self.wide is not None:
            out = out + self.wide(features).squeeze(1)
        return out


def build_model(preprocessor, model_options=None):
    options = dict(model_options or {})
    options.setdefault("dense_feature_count", dense_feature_count(preprocessor))
    return FraudMLP(embedding_sizes(preprocessor), **options)
