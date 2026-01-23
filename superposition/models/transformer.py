"""Transformer-based model for studying superposition at scale.

Uses a GPT2-based architecture to study how superposition manifests
in larger, more realistic transformer models with attention mechanisms.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Model
from typing import Optional

from superposition.models.base import SuperpositionModel


class TransformerModel(SuperpositionModel):
    """GPT2-based transformer for studying superposition.

    Architecture: features -> Linear -> GPT2 -> Linear -> features -> ReLU

    Studies how transformer attention mechanisms interact with superposition,
    providing insight into how larger language models might encode features
    in overlapping representations.
    """

    def __init__(
        self,
        num_features: int = 128,
        num_hidden: int = 64,
        num_instances: int = 10,
        n_layers: int = 4,
        n_heads: int = 4,
        n_positions: int = 32,
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

        self.gpt2_config = GPT2Config(
            n_positions=n_positions,
            n_embd=num_hidden,
            n_layer=n_layers,
            n_head=n_heads,
            n_inner=num_hidden * 4,
        )

        self.input_projection = nn.Linear(num_features, num_hidden)
        self.transformer = GPT2Model(self.gpt2_config)
        self.output_projection = nn.Linear(num_hidden, num_features)

        self.to_device()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer autoencoder.

        Args:
            features: Input of shape (batch, num_instances, num_features).

        Returns:
            Reconstruction of shape (batch, num_instances, num_features).
        """
        batch_size = features.shape[0]

        # Flatten instances into batch dimension
        features_flat = features.view(-1, self.num_features)

        # Project to hidden size and add sequence dimension
        hidden = self.input_projection(features_flat).unsqueeze(1)

        # Pass through transformer
        transformer_out = self.transformer(inputs_embeds=hidden).last_hidden_state
        transformer_out = transformer_out.squeeze(1)

        # Project back to feature space
        output = self.output_projection(transformer_out)
        output = output.view(batch_size, self.num_instances, self.num_features)

        return F.relu(output)

    def get_input_weights(self) -> torch.Tensor:
        """Get input projection weights for visualization.

        Returns:
            Weight tensor of shape (num_hidden, num_features).
        """
        return self.input_projection.weight.detach()

    def get_output_weights(self) -> torch.Tensor:
        """Get output projection weights for visualization.

        Returns:
            Weight tensor of shape (num_features, num_hidden).
        """
        return self.output_projection.weight.detach()
