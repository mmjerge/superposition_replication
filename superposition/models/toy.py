"""Toy autoencoder model for studying superposition.

Implements the basic W^T W autoencoder from the Anthropic superposition paper,
where features are compressed through a bottleneck and reconstructed.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

from superposition.models.base import SuperpositionModel


class ToyModel(SuperpositionModel):
    """Toy autoencoder model for superposition experiments.

    Architecture: features -> W -> hidden -> W^T -> features + bias -> ReLU

    This is the simplest model that demonstrates superposition: when num_features > num_hidden,
    the model must learn to compress features into a lower-dimensional space, potentially
    encoding multiple features in overlapping directions.
    """

    def __init__(
        self,
        num_features: int = 5,
        num_hidden: int = 2,
        num_instances: int = 10,
        feature_probability: Optional[torch.Tensor] = None,
        importance: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            num_features=num_features,
            num_hidden=num_hidden,
            num_instances=num_instances,
            feature_probability=feature_probability,
            importance=importance,
            device=device,
        )

        self.W = nn.Parameter(
            torch.empty(self.num_instances, self.num_features, self.num_hidden)
        )
        nn.init.xavier_normal_(self.W)

        self.b_final = nn.Parameter(
            torch.zeros(self.num_instances, self.num_features)
        )

        self.to_device()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.

        Args:
            features: Input of shape (batch, num_instances, num_features).

        Returns:
            Reconstruction of shape (batch, num_instances, num_features).
        """
        # Encode: (batch, instances, features) @ (instances, features, hidden)
        hidden = torch.einsum("bif,ifh->bih", features, self.W)
        # Decode: (batch, instances, hidden) @ (instances, features, hidden) -> contract over h
        out = torch.einsum("bih,ifh->bif", hidden, self.W)
        out = out + self.b_final.unsqueeze(0)
        return F.relu(out)

    def get_weight_matrix(self) -> torch.Tensor:
        """Get the weight matrix W for visualization.

        Returns:
            Weight tensor of shape (num_instances, num_features, num_hidden).
        """
        return self.W.detach()
