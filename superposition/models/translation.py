"""Translation model with bottleneck for studying superposition in seq2seq.

Uses a pre-trained MarianMT model with a learned bottleneck layer to study
how superposition emerges when a translation model is forced to compress
its encoder representations through a narrow hidden layer.
"""

import torch
from torch import nn
from transformers import MarianMTModel, MarianTokenizer
from typing import Optional

from superposition.utils.logging import get_logger

logger = get_logger(__name__)


class TranslationModel(nn.Module):
    """Translation model with learned bottleneck for superposition experiments.

    Freezes the pre-trained MarianMT model and trains only the bottleneck layers,
    studying how the model learns to compress and expand encoder representations.
    """

    def __init__(
        self,
        base_model_name: str = "Helsinki-NLP/opus-mt-en-fr",
        hidden_size: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading pre-trained model: {base_model_name}")
        self.base_model = MarianMTModel.from_pretrained(base_model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(base_model_name)

        encoder_dim = self.base_model.config.d_model
        self.encoder_bottleneck = nn.Linear(encoder_dim, hidden_size)
        self.decoder_expansion = nn.Linear(hidden_size, encoder_dim)

        # Freeze the base model - only train bottleneck layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"TranslationModel: encoder_dim={encoder_dim}, hidden_size={hidden_size}, "
            f"trainable_params={trainable:,} / total={total:,}"
        )

        self.to(self._device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward pass through bottleneck translation model.

        Args:
            input_ids: Source token IDs.
            attention_mask: Attention mask for source.
            labels: Target token IDs (for computing loss).

        Returns:
            Model outputs with loss if labels provided.
        """
        # Run frozen encoder without gradient tracking to save memory.
        # All encoder params are frozen, so no gradients flow through it.
        with torch.no_grad():
            encoder_outputs = self.base_model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Compress through bottleneck (trainable — gradients flow via weights)
        hidden_states = self.encoder_bottleneck(encoder_outputs[0])
        expanded_states = self.decoder_expansion(hidden_states)

        outputs = self.base_model(
            encoder_outputs=(expanded_states,),
            labels=labels,
            attention_mask=attention_mask,
        )

        return outputs

    def get_bottleneck_weights(self) -> torch.Tensor:
        """Get encoder bottleneck weights for visualization.

        Returns:
            Weight tensor of shape (hidden_size, encoder_dim).
        """
        return self.encoder_bottleneck.weight.detach()

    def get_trainable_parameters(self):
        """Get only the trainable bottleneck parameters for the optimizer."""
        return [
            {"params": self.encoder_bottleneck.parameters()},
            {"params": self.decoder_expansion.parameters()},
        ]
