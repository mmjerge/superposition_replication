"""Feature interference analysis via cosine similarity heatmaps.

Replaces the radial/vector plots with a clearer representation of which
features are orthogonal (independent) and which are superposed (aligned)
by computing pairwise cosine similarity of learned weight directions.
"""

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from superposition.utils.logging import get_logger

logger = get_logger(__name__)


def compute_cosine_similarity_matrix(weight_matrix: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity between feature vectors.

    Args:
        weight_matrix: Weight matrix of shape (num_features, hidden_dim).
            Each row is a feature's learned direction in the hidden space.

    Returns:
        Cosine similarity matrix of shape (num_features, num_features).
        Values range from -1 (anti-aligned) to 1 (perfectly aligned).
        Diagonal is always 1.
    """
    # Normalize rows to unit vectors
    normed_weights = F.normalize(weight_matrix.float(), p=2, dim=1)
    # Cosine similarity = dot product of normalized vectors
    cosine_sim = torch.matmul(normed_weights, normed_weights.T)
    return cosine_sim


def compute_interference_heatmap(
    model,
    save_path: str = "images/interference_heatmap.png",
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
    model_type: str = "toy",
) -> torch.Tensor:
    """Generate and save a cosine similarity heatmap for a model's weights.

    This visualization explicitly shows:
    - Diagonal = 1 (feature with itself)
    - Off-diagonal near 0 = features are orthogonal (no interference)
    - Off-diagonal near +/-1 = features are superposed (sharing direction)

    Args:
        model: A superposition model with extractable weight matrices.
        save_path: Path to save the heatmap figure.
        title: Plot title. Auto-generated if None.
        figsize: Figure size in inches.
        model_type: Type of model ('toy', 'transformer', 'translation',
            'computation', 'continuous_thought').

    Returns:
        The cosine similarity matrix as a tensor.
    """
    # Extract the appropriate weight matrix based on model type
    with torch.no_grad():
        if model_type == "toy":
            # ToyModel: W shape is (num_instances, num_features, num_hidden)
            # Use instance 0 (or average across instances)
            weights = model.W.detach()
            if weights.dim() == 3:
                # Average across instances for a summary view
                weight_matrix = weights.mean(dim=0)  # (num_features, num_hidden)
            else:
                weight_matrix = weights
        elif model_type == "transformer":
            # TransformerModel: input_projection weight is (hidden, features)
            weight_matrix = model.input_projection.weight.detach().T  # -> (features, hidden)
        elif model_type == "translation":
            # TranslationModel: encoder_bottleneck weight is (hidden, encoder_dim)
            weight_matrix = model.encoder_bottleneck.weight.detach().T  # -> (encoder_dim, hidden)
        elif model_type == "computation":
            # ComputationModel: W_enc shape is (num_instances, num_features, num_hidden)
            weights = model.W_enc.detach()
            if weights.dim() == 3:
                # Average across instances for a summary view
                weight_matrix = weights.mean(dim=0)  # (num_features, num_hidden)
            else:
                weight_matrix = weights
        elif model_type == "continuous_thought":
            # ContinuousThoughtModel: encoder_bottleneck weight is (hidden, encoder_dim)
            weight_matrix = model.encoder_bottleneck.weight.detach().T  # -> (encoder_dim, hidden)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    weight_matrix = weight_matrix.cpu()
    cosine_sim = compute_cosine_similarity_matrix(weight_matrix)

    # Generate the heatmap
    if title is None:
        title = f"Feature Interference Pattern ({model_type} model, h={weight_matrix.shape[1]})"

    plot_interference_heatmap(
        cosine_sim.numpy(),
        save_path=save_path,
        title=title,
        figsize=figsize,
    )

    return cosine_sim


def plot_interference_heatmap(
    similarity_matrix: np.ndarray,
    save_path: str = "images/interference_heatmap.png",
    title: str = "Feature Interference Pattern (Cosine Similarity)",
    figsize: tuple = (10, 8),
    feature_labels: Optional[list] = None,
    max_features_labeled: int = 30,
) -> None:
    """Plot a cosine similarity heatmap showing feature interference.

    Args:
        similarity_matrix: Square matrix of pairwise cosine similarities.
        save_path: Path to save the figure.
        title: Plot title.
        figsize: Figure size in inches.
        feature_labels: Optional labels for features.
        max_features_labeled: Max number of features to show labels for.
    """
    n_features = similarity_matrix.shape[0]
    show_labels = n_features <= max_features_labeled

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        similarity_matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=feature_labels if (show_labels and feature_labels) else False,
        yticklabels=feature_labels if (show_labels and feature_labels) else False,
        square=True,
        ax=ax,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
    )

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Feature Index", fontsize=10)
    ax.set_ylabel("Feature Index", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Interference heatmap saved to {save_path}")

    # Report summary statistics
    mask = ~np.eye(n_features, dtype=bool)
    off_diag = similarity_matrix[mask]
    logger.info(
        f"Off-diagonal stats: mean={off_diag.mean():.4f}, "
        f"std={off_diag.std():.4f}, "
        f"max_abs={np.abs(off_diag).max():.4f}"
    )


def compute_interference_per_instance(
    model,
    save_dir: str = "images",
) -> dict:
    """Compute interference heatmaps for each instance in a toy model.

    For models with multiple instances (varying sparsity), this shows how
    interference patterns change with feature probability.

    Args:
        model: ToyModel with W of shape (num_instances, num_features, num_hidden).
        save_dir: Directory to save per-instance heatmaps.

    Returns:
        Dictionary mapping instance index to its cosine similarity matrix.
    """
    with torch.no_grad():
        weights = model.W.detach().cpu()

    results = {}
    num_instances = weights.shape[0]
    feature_probs = model.feature_probability.cpu().squeeze()

    for i in range(num_instances):
        weight_matrix = weights[i]  # (num_features, num_hidden)
        cosine_sim = compute_cosine_similarity_matrix(weight_matrix)
        results[i] = cosine_sim

        prob_str = f"{feature_probs[i].item():.3f}"
        plot_interference_heatmap(
            cosine_sim.numpy(),
            save_path=f"{save_dir}/interference_instance_{i}_p{prob_str}.png",
            title=f"Instance {i} (feature_prob={prob_str})",
            figsize=(6, 5),
        )

    return results
