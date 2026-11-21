"""MLP architecture for one binary classification logit per transaction."""

import torch
from torch import nn

from .data import DENSE_FEATURE_COUNT


class FraudMLP(nn.Module):
    # Defaults also restore checkpoints created before config.toml was added.
    def __init__(self, category_sizes, *, embedding_dim=4, hidden_sizes=(256, 128, 64), dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim, padding_idx=0) for size in category_sizes
        )

        input_size = DENSE_FEATURE_COUNT + len(category_sizes) * embedding_dim
        layers = []
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, dense, categorical):
        embeddings = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        features = torch.cat([dense, *embeddings], dim=1)
        return self.classifier(features).squeeze(1)
