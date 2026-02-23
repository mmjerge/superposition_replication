"""Generate a single-panel transformer feature embedding figure for the paper."""

import sys
sys.path.insert(0, "/sfs/weka/scratch/mj6ux/Projects/superposition_replication")

import torch
import numpy as np
import matplotlib.pyplot as plt

from superposition.models.transformer import TransformerModel


def train_transformer(num_steps=1000):
    """Train a transformer model and return it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(
        num_features=128,
        num_hidden=64,
        num_instances=10,
        n_layers=4,
        n_heads=4,
        n_positions=32,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for step in range(num_steps):
        batch = model.generate_data(32)
        output = model(batch)
        loss = model.compute_loss(batch, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item():.6f}")

    print(f"Final loss: {loss.item():.6f}")
    return model


def plot_single_panel(model, save_path):
    """Plot a single panel showing transformer feature embeddings in 2D."""
    # Get input projection weights: shape (num_hidden, num_features)
    weights = model.get_input_weights().cpu().numpy()

    # Transpose to (num_features, num_hidden) so each row is a feature's direction
    weights = weights.T  # (128, 64)

    # Project to 2D using first two hidden dimensions
    weights_2d = weights[:, :2]

    # Normalize for visibility
    max_mag = np.abs(weights_2d).max()
    if max_mag > 0:
        weights_2d = weights_2d / max_mag

    num_features = len(weights_2d)
    colors = plt.cm.YlGn(np.linspace(0.3, 0.9, num_features))

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=200)

    for j, (wx, wy) in enumerate(weights_2d):
        ax.plot([0, wx], [0, wy], "-", color=colors[j], linewidth=2, alpha=0.7)
        ax.scatter(wx, wy, color=colors[j], s=80, alpha=0.8, zorder=3)

    # Style the axes
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
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    print("Training transformer model...")
    model = train_transformer(num_steps=1000)

    save_path = "/sfs/weka/scratch/mj6ux/Projects/superposition_replication/tmlr_paper/figures/transformer_features.png"
    print(f"Generating figure at {save_path}...")
    plot_single_panel(model, save_path)
    print("Done!")
