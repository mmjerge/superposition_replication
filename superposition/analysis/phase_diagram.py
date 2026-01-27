"""Phase diagram analysis for superposition transitions.

Replicates the key result from the Anthropic paper: a phase diagram showing
when features transition between 'not represented', 'dedicated neuron', and
'superposition' as feature probability and importance vary.

The diagram sweeps over a grid of (feature_probability, relative_importance)
values, trains a toy model at each point, and measures the dimensionality
each feature occupies in the hidden space.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional

from superposition.utils.logging import get_logger

logger = get_logger(__name__)


def compute_feature_dimensionality(W: torch.Tensor) -> torch.Tensor:
    """Compute the dimensionality each feature occupies in the hidden space.

    For each feature i, its dimensionality is ||w_i||^2 where w_i is the
    feature's direction vector in hidden space. When the hidden dimension is
    smaller than the feature count, a feature using a full orthogonal direction
    will have ||w_i||^2 ≈ 1, while unused features have ||w_i||^2 ≈ 0.

    Args:
        W: Weight matrix of shape (num_features, num_hidden).

    Returns:
        Tensor of shape (num_features,) with dimensionality values in [0, 1+].
    """
    return (W ** 2).sum(dim=-1)


def run_phase_diagram_sweep(
    num_features: int = 5,
    num_hidden: int = 2,
    sparsity_steps: int = 20,
    importance_steps: int = 20,
    training_steps: int = 10000,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
) -> dict:
    """Run the full phase diagram sweep.

    Trains a toy model for each (feature_probability, importance_ratio) pair
    and records the learned feature dimensionality.

    Args:
        num_features: Number of features in each toy model.
        num_hidden: Hidden dimension (bottleneck size).
        sparsity_steps: Number of grid points along the sparsity axis.
        importance_steps: Number of grid points along the importance axis.
        training_steps: Training steps per model.
        batch_size: Batch size for training.
        learning_rate: Learning rate.
        device: Torch device.

    Returns:
        Dictionary with keys:
            - feature_probs: 1D array of feature probability values
            - importance_ratios: 1D array of importance ratio values
            - dimensionality: 2D array of shape (importance_steps, sparsity_steps)
              with mean feature dimensionality at each grid point
            - dimensionality_per_feature: 3D array of shape
              (importance_steps, sparsity_steps, num_features)
    """
    from superposition.models.toy import ToyModel

    if device is None:
        from superposition.utils.reproducibility import get_device
        device = get_device()

    # Sweep ranges (log scale for sparsity, linear for importance)
    feature_probs = np.logspace(-2, 0, sparsity_steps)  # 0.01 to 1.0
    importance_ratios = np.logspace(-1, 0, importance_steps)  # 0.1 to 1.0

    dim_grid = np.zeros((importance_steps, sparsity_steps))
    dim_per_feature = np.zeros((importance_steps, sparsity_steps, num_features))

    total = importance_steps * sparsity_steps
    count = 0

    logger.info(
        f"Starting phase diagram sweep: {total} models, "
        f"{training_steps} steps each, "
        f"features={num_features}, hidden={num_hidden}"
    )

    for i, imp_ratio in enumerate(importance_ratios):
        for j, feat_prob in enumerate(feature_probs):
            count += 1
            if count % 10 == 0 or count == 1:
                logger.info(f"  Training model {count}/{total} "
                            f"(prob={feat_prob:.3f}, imp={imp_ratio:.3f})")

            # Create importance: first feature has ratio imp_ratio, rest decay
            importance = torch.zeros(1, num_features)
            importance[0, 0] = imp_ratio
            for k in range(1, num_features):
                importance[0, k] = 0.9 ** k

            # Create uniform feature probability (single instance)
            fp = torch.full((1, 1), feat_prob)

            model = ToyModel(
                num_features=num_features,
                num_hidden=num_hidden,
                num_instances=1,
                feature_probability=fp,
                importance=importance,
                device=device,
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            model.train()

            for step in range(training_steps):
                optimizer.zero_grad(set_to_none=True)
                batch = model.generate_data(batch_size)
                output = model(batch)
                loss = model.compute_loss(batch, output)
                loss.backward()
                optimizer.step()

            # Extract dimensionality from trained weights
            with torch.no_grad():
                W = model.W.detach().cpu()[0]  # (num_features, num_hidden)
                feat_dims = compute_feature_dimensionality(W).numpy()

            dim_grid[i, j] = feat_dims.mean()
            dim_per_feature[i, j] = feat_dims

    logger.info("Phase diagram sweep complete.")

    return {
        "feature_probs": feature_probs,
        "importance_ratios": importance_ratios,
        "dimensionality": dim_grid,
        "dimensionality_per_feature": dim_per_feature,
    }


def plot_phase_diagram(
    results: dict,
    save_path: str = "images/phase_diagram.png",
    feature_index: Optional[int] = None,
    title: Optional[str] = None,
) -> None:
    """Plot the phase diagram from sweep results.

    Args:
        results: Output of run_phase_diagram_sweep().
        save_path: Path to save the figure.
        feature_index: If specified, plot dimensionality for this feature only.
            If None, plot mean dimensionality across all features.
        title: Plot title. Auto-generated if None.
    """
    feature_probs = results["feature_probs"]
    importance_ratios = results["importance_ratios"]

    if feature_index is not None:
        data = results["dimensionality_per_feature"][:, :, feature_index]
        default_title = f"Phase Diagram: Feature {feature_index} Dimensionality"
    else:
        data = results["dimensionality"]
        default_title = "Phase Diagram: Mean Feature Dimensionality"

    if title is None:
        title = default_title

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a diverging colormap centered at the boundary between
    # "not represented" (0) and "full dimension" (1)
    norm = mcolors.Normalize(vmin=0, vmax=max(1.0, data.max()))
    im = ax.pcolormesh(
        feature_probs,
        importance_ratios,
        data,
        norm=norm,
        cmap="viridis",
        shading="nearest",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Feature Probability (1 - Sparsity)", fontsize=12)
    ax.set_ylabel("Relative Importance", fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Dimensionality (||w||²)", fontsize=11)

    # Annotate phase regions
    ax.text(
        0.7, 0.95, "Dense → dedicated neurons",
        transform=ax.transAxes, fontsize=9, style="italic",
        verticalalignment="top", color="white",
    )
    ax.text(
        0.02, 0.05, "Sparse + low importance → not represented",
        transform=ax.transAxes, fontsize=9, style="italic",
        verticalalignment="bottom", color="white",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Phase diagram saved to {save_path}")


def plot_feature_dimensionality_curves(
    results: dict,
    save_path: str = "images/feature_dim_curves.png",
    title: Optional[str] = None,
) -> None:
    """Plot dimensionality vs sparsity curves for each feature at fixed importance.

    Shows how each feature's representation changes as sparsity varies,
    using the middle importance level.

    Args:
        results: Output of run_phase_diagram_sweep().
        save_path: Path to save the figure.
        title: Plot title.
    """
    feature_probs = results["feature_probs"]
    dim_per_feature = results["dimensionality_per_feature"]
    num_features = dim_per_feature.shape[2]

    # Use middle importance level
    mid_imp = dim_per_feature.shape[0] // 2

    if title is None:
        imp_val = results["importance_ratios"][mid_imp]
        title = f"Feature Dimensionality vs Sparsity (importance≈{imp_val:.2f})"

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_features))
    for k in range(num_features):
        ax.plot(
            feature_probs,
            dim_per_feature[mid_imp, :, k],
            color=colors[k],
            linewidth=2,
            label=f"Feature {k}",
            alpha=0.8,
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Full dimension")
    ax.set_xscale("log")
    ax.set_xlabel("Feature Probability", fontsize=12)
    ax.set_ylabel("Dimensionality (||w||²)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Feature dimensionality curves saved to {save_path}")
