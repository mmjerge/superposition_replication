"""Visualization utilities for superposition experiments."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SuperpositionVisualizer:
    """Handles all visualization for superposition experiments.

    Supports TensorBoard and wandb logging, weight vector plots, and embedding visualizations.
    """

    def __init__(
        self,
        log_dir: str = "runs/experiment",
        save_dir: str = "images",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[dict] = None,
    ):
        self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb requested but not installed. Install with: uv sync --extra wandb")
                self.use_wandb = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard and wandb."""
        self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log a histogram to TensorBoard and wandb."""
        self.writer.add_histogram(tag, values, step)
        if self.use_wandb:
            wandb.log({tag: wandb.Histogram(values)}, step=step)

    def plot_weight_vectors_2d(
        self,
        weights: np.ndarray,
        step: int,
        num_instances: int = 1,
        save_path: Optional[str] = None,
        title_prefix: str = "Instance",
    ) -> None:
        """Plot weight vectors in 2D space showing superposition patterns.

        Args:
            weights: Weight matrix of shape (num_features, hidden_dim) or
                     (num_instances, num_features, hidden_dim).
            step: Current training step.
            num_instances: Number of subplot instances.
            save_path: Path to save the figure. If None, auto-generated.
            title_prefix: Prefix for subplot titles.
        """
        if weights.ndim == 2:
            weights = weights[np.newaxis, ...]

        # Use first 2 hidden dims for 2D projection
        weights_2d = weights[:, :, :2]

        fig, axs = plt.subplots(1, num_instances, figsize=(2 * num_instances, 2), dpi=200)
        if num_instances == 1:
            axs = [axs]

        for i, ax in enumerate(axs):
            if i >= len(weights_2d):
                break
            instance_weights = weights_2d[i]

            # Normalize for visibility
            max_mag = np.abs(instance_weights).max()
            if max_mag > 0:
                instance_weights = instance_weights / max_mag

            num_features = len(instance_weights)
            colors = plt.cm.YlGn(np.linspace(0.3, 0.9, num_features))

            for j, (wx, wy) in enumerate(instance_weights):
                ax.plot([0, wx], [0, wy], "-", color=colors[j], linewidth=2, alpha=0.7)
                ax.scatter(wx, wy, color=colors[j], s=80, alpha=0.8, zorder=3)

            self._style_vector_axes(ax)
            ax.set_title(f"{title_prefix} {i + 1}", fontsize=8, pad=5)

        plt.tight_layout()

        if save_path is None:
            save_path = f"{self.save_dir}/vector_plot_step_{step}.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        self.writer.add_figure("weight_vectors", fig, step)

        if self.use_wandb:
            wandb.log({"weight_vectors": wandb.Image(fig)}, step=step)

        plt.close(fig)

    def plot_intro_diagram(
        self,
        weights: np.ndarray,
        save_path: str = "images/intro_diagram.pdf",
    ) -> None:
        """Plot the intro diagram showing feature vectors across instances.

        Args:
            weights: Weight matrix of shape (num_instances, num_features, hidden_dim).
            save_path: Path to save the figure.
        """
        num_instances, num_features, _ = weights.shape
        colors = sns.color_palette("viridis", n_colors=num_features)

        fig, axs = plt.subplots(1, num_instances, figsize=(2 * num_instances, 2), dpi=200)
        if num_instances == 1:
            axs = [axs]

        for i, ax in enumerate(axs):
            for j in range(num_features):
                ax.scatter(weights[i, j, 0], weights[i, j, 1], color=colors[j], s=50)
                ax.plot(
                    [0, weights[i, j, 0]], [0, weights[i, j, 1]],
                    color=colors[j], alpha=0.5,
                )

            ax.set_aspect("equal")
            ax.set_facecolor("#FCFBF8")
            z = max(1.5, np.abs(weights[i]).max() * 1.1)
            ax.set_xlim((-z, z))
            ax.set_ylim((-z, z))

            ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_position("center")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def plot_translation_vectors(
        self,
        weights: np.ndarray,
        step: int,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot weight vectors for translation bottleneck using quiver plot.

        Args:
            weights: Weight matrix of shape (n, 2) - first 2 dims of bottleneck weights.
            step: Current training step.
            save_path: Path to save the figure.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        colors = ["red", "blue", "green", "purple"]
        for i, (vec, color) in enumerate(zip(weights[:4], colors)):
            ax.quiver(
                0, 0, vec[0], vec[1],
                angles="xy", scale_units="xy", scale=1,
                color=color, label=f"Vector {i + 1}",
            )

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"Weight Vectors at Step {step}")

        if save_path is None:
            save_path = f"{self.save_dir}/weight_plot_{step}.png"

        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        self.writer.add_figure("translation_weights", fig, step)

        if self.use_wandb:
            wandb.log({"translation_weights": wandb.Image(fig)}, step=step)

        plt.close(fig)

    def plot_embeddings_3d(
        self,
        embeddings: np.ndarray,
        labels: list,
        step: int,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot 3D PCA visualization of embeddings.

        Args:
            embeddings: Reduced embeddings of shape (n, 3).
            labels: List of labels for each point.
            step: Current training step.
            save_path: Path to save the figure.
        """
        from sklearn.decomposition import PCA

        if embeddings.shape[1] > 3:
            pca = PCA(n_components=3)
            embeddings = pca.fit_transform(embeddings)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], alpha=0.7)

        for i, label in enumerate(labels):
            if i % 10 == 0:
                ax.text(embeddings[i, 0], embeddings[i, 1], embeddings[i, 2], label, fontsize=6)

        ax.set_title("Embeddings (PCA 3D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        if save_path is None:
            save_path = f"{self.save_dir}/embeddings_step_{step}_3d.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def close(self) -> None:
        """Close the TensorBoard writer and finish wandb run."""
        self.writer.close()
        if self.use_wandb:
            wandb.finish()

    @staticmethod
    def _style_vector_axes(ax) -> None:
        """Apply consistent styling to vector plot axes."""
        ax.set_facecolor("white")
        ax.spines["left"].set_position("center")
        ax.spines["bottom"].set_position("center")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect("equal")
        limit = 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xticks([-1, -0.5, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0.5, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
