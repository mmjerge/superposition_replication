"""Base model class with shared superposition functionality."""

import torch
from torch import nn
from abc import abstractmethod
from typing import Optional

from superposition.utils.reproducibility import get_device
from superposition.utils.logging import get_logger

logger = get_logger(__name__)


class SuperpositionModel(nn.Module):
    """Base class for all superposition models.

    Provides shared functionality for feature probability, importance weighting,
    data generation, and device management.
    """

    def __init__(
        self,
        num_features: int,
        num_hidden: int,
        num_instances: int,
        feature_probability: Optional[torch.Tensor] = None,
        importance: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_instances = num_instances
        self._device = device or get_device()

        # Feature probability: how likely each feature is active (per instance)
        if feature_probability is not None:
            self.register_buffer("feature_probability", feature_probability)
        else:
            fp = (20 ** -torch.linspace(0, 1, num_instances))[:, None]
            self.register_buffer("feature_probability", fp)

        # Importance: relative weight of each feature in the loss
        if importance is not None:
            self.register_buffer("importance", importance)
        else:
            imp = (0.9 ** torch.arange(num_features))[None, :]
            self.register_buffer("importance", imp)

        logger.info(
            f"Initialized {self.__class__.__name__}: "
            f"features={num_features}, hidden={num_hidden}, "
            f"instances={num_instances}, device={self._device}"
        )

    def generate_data(self, num_batch: int) -> torch.Tensor:
        """Generate sparse feature data for training.

        Args:
            num_batch: Number of samples in the batch.

        Returns:
            Tensor of shape (num_batch, num_instances, num_features) with sparse features.
        """
        feature = torch.rand(
            (num_batch, self.num_instances, self.num_features),
            device=self._device,
        )
        mask = torch.rand(
            (num_batch, self.num_instances, self.num_features),
            device=self._device,
        ) <= self.feature_probability
        return torch.where(mask, feature, torch.zeros((), device=self._device))

    def compute_loss(self, batch: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss.

        Args:
            batch: Input features.
            output: Model reconstruction.

        Returns:
            Scalar loss tensor.
        """
        error = self.importance * (batch.abs() - output) ** 2
        return error.mean()

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        ...

    def to_device(self) -> "SuperpositionModel":
        """Move model to its configured device."""
        return self.to(self._device)
