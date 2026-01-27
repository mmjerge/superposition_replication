"""Computation-in-superposition model.

Extends the toy model to study whether networks can perform computation
(not just storage) while features remain in superposition. The Anthropic
paper demonstrates this with absolute value computation.

This model takes sparse features, compresses them through a bottleneck,
and must output a nonlinear function of the inputs (e.g., absolute value,
squared features, or thresholded features) rather than simply reconstructing
them. If the model succeeds, it proves computation can occur in superposition.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

from superposition.models.base import SuperpositionModel
from superposition.utils.logging import get_logger

logger = get_logger(__name__)


class ComputationModel(SuperpositionModel):
    """Model for studying computation in superposition.

    Architecture: features -> W_enc -> hidden -> MLP -> W_dec -> output
    Target: a nonlinear function of the input features.

    Unlike the toy model which reconstructs inputs, this model must compute
    a target function while features are compressed through a bottleneck.
    """

    def __init__(
        self,
        num_features: int = 5,
        num_hidden: int = 2,
        num_instances: int = 10,
        target_fn: str = "abs",
        mlp_hidden: int = 16,
        feature_probability: Optional[torch.Tensor] = None,
        importance: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the computation model.

        Args:
            num_features: Number of input/output features.
            num_hidden: Bottleneck dimension (compression).
            num_instances: Number of sparsity instances.
            target_fn: Target computation ('abs', 'square', 'threshold', 'relu').
            mlp_hidden: Hidden dimension of the computation MLP.
            feature_probability: Custom feature probabilities.
            importance: Custom feature importance weights.
            device: Torch device.
        """
        super().__init__(
            num_features=num_features,
            num_hidden=num_hidden,
            num_instances=num_instances,
            feature_probability=feature_probability,
            importance=importance,
            device=device,
        )

        self.target_fn_name = target_fn
        self.target_fn = self._get_target_fn(target_fn)

        # Encoder: features -> bottleneck
        self.W_enc = nn.Parameter(
            torch.empty(num_instances, num_features, num_hidden)
        )
        nn.init.xavier_normal_(self.W_enc)

        # Computation MLP in the bottleneck space
        self.mlp = nn.Sequential(
            nn.Linear(num_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_hidden),
        )

        # Decoder: bottleneck -> output
        self.W_dec = nn.Parameter(
            torch.empty(num_instances, num_hidden, num_features)
        )
        nn.init.xavier_normal_(self.W_dec)

        self.b_final = nn.Parameter(
            torch.zeros(num_instances, num_features)
        )

        self.to_device()

        logger.info(
            f"ComputationModel: target_fn={target_fn}, "
            f"mlp_hidden={mlp_hidden}"
        )

    @staticmethod
    def _get_target_fn(name: str):
        """Get the target computation function.

        Args:
            name: Function name.

        Returns:
            Callable that maps input features to target outputs.
        """
        fns = {
            "abs": lambda x: x.abs(),
            "square": lambda x: x ** 2,
            "threshold": lambda x: (x > 0.5).float(),
            "relu": lambda x: F.relu(x - 0.3),
        }
        if name not in fns:
            raise ValueError(
                f"Unknown target_fn: {name}. Choose from {list(fns.keys())}"
            )
        return fns[name]

    def generate_data(self, num_batch: int) -> torch.Tensor:
        """Generate sparse input data with both positive and negative values.

        Unlike the base class which generates [0, 1) values, this generates
        values in [-1, 1) to make computations like abs() non-trivial.

        Args:
            num_batch: Number of samples.

        Returns:
            Tensor of shape (num_batch, num_instances, num_features).
        """
        feature = 2 * torch.rand(
            (num_batch, self.num_instances, self.num_features),
            device=self._device,
        ) - 1  # Uniform in [-1, 1)

        mask = torch.rand(
            (num_batch, self.num_instances, self.num_features),
            device=self._device,
        ) <= self.feature_probability

        return torch.where(mask, feature, torch.zeros((), device=self._device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode, compute, decode.

        Args:
            features: Input of shape (batch, num_instances, num_features).

        Returns:
            Output of shape (batch, num_instances, num_features).
        """
        # Encode into bottleneck
        hidden = torch.einsum("bif,ifh->bih", features, self.W_enc)

        # Apply computation MLP (shared across instances)
        batch_size, num_inst, hidden_dim = hidden.shape
        hidden_flat = hidden.reshape(-1, hidden_dim)
        computed = self.mlp(hidden_flat)
        computed = computed.reshape(batch_size, num_inst, hidden_dim)

        # Decode from bottleneck
        output = torch.einsum("bih,ihf->bif", computed, self.W_dec)
        output = output + self.b_final.unsqueeze(0)

        return output

    def compute_loss(self, batch: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Compute loss against the target function.

        Args:
            batch: Input features.
            output: Model output.

        Returns:
            Scalar loss tensor.
        """
        target = self.target_fn(batch)
        error = self.importance * (target - output) ** 2
        return error.mean()

    def get_encoder_weights(self) -> torch.Tensor:
        """Get encoder weight matrix for analysis."""
        return self.W_enc.detach()

    def get_decoder_weights(self) -> torch.Tensor:
        """Get decoder weight matrix for analysis."""
        return self.W_dec.detach()

    def evaluate_accuracy(self, num_samples: int = 10000) -> dict:
        """Evaluate how well the model computes the target function.

        Args:
            num_samples: Number of test samples.

        Returns:
            Dictionary with per-instance MSE and correlation metrics.
        """
        self.eval()
        with torch.no_grad():
            batch = self.generate_data(num_samples)
            output = self(batch)
            target = self.target_fn(batch)

            # Per-instance metrics
            per_instance_mse = ((target - output) ** 2).mean(dim=(0, 2))
            per_instance_corr = []

            for i in range(self.num_instances):
                t = target[:, i, :].flatten()
                o = output[:, i, :].flatten()
                if t.std() > 1e-8 and o.std() > 1e-8:
                    corr = torch.corrcoef(torch.stack([t, o]))[0, 1].item()
                else:
                    corr = 0.0
                per_instance_corr.append(corr)

        return {
            "mse_per_instance": per_instance_mse.cpu().numpy(),
            "correlation_per_instance": np.array(per_instance_corr),
            "mean_mse": per_instance_mse.mean().item(),
            "mean_correlation": np.mean(per_instance_corr),
        }


# Needed for evaluate_accuracy
import numpy as np
