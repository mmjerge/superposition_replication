"""Geometric structure analysis for superposition.

Detects the polytope structures that emerge when features are stored in
superposition: antipodal pairs (digons), triangles, pentagons, tetrahedra,
and other uniform polytope arrangements.

The Anthropic paper shows that features don't just overlap randomly --
they organize into specific geometric patterns that minimize interference
while maximizing the number of stored features.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from collections import Counter

from superposition.utils.logging import get_logger

logger = get_logger(__name__)


# Known polytope structures and their characteristic angles
POLYTOPE_SIGNATURES = {
    "antipodal_pair": {
        "description": "Two features at 180° (digon)",
        "expected_angle": 180.0,
        "tolerance": 15.0,
        "min_features": 2,
    },
    "triangle": {
        "description": "Three features at 120° (equilateral triangle)",
        "expected_angle": 120.0,
        "tolerance": 15.0,
        "min_features": 3,
    },
    "square": {
        "description": "Four features at 90° (square)",
        "expected_angle": 90.0,
        "tolerance": 15.0,
        "min_features": 4,
    },
    "pentagon": {
        "description": "Five features at 72° (regular pentagon)",
        "expected_angle": 72.0,
        "tolerance": 12.0,
        "min_features": 5,
    },
    "tetrahedron": {
        "description": "Four features at 109.5° (regular tetrahedron)",
        "expected_angle": 109.47,
        "tolerance": 15.0,
        "min_features": 4,
    },
}


def compute_pairwise_angles(W: torch.Tensor) -> np.ndarray:
    """Compute pairwise angles between feature direction vectors.

    Args:
        W: Weight matrix of shape (num_features, num_hidden).

    Returns:
        Symmetric matrix of shape (num_features, num_features) with angles
        in degrees. Diagonal is 0.
    """
    W_normed = F.normalize(W.float(), p=2, dim=1)
    cosines = torch.clamp(W_normed @ W_normed.T, -1.0, 1.0)
    angles_rad = torch.acos(cosines)
    angles_deg = torch.rad2deg(angles_rad).numpy()
    np.fill_diagonal(angles_deg, 0.0)
    return angles_deg


def classify_feature_geometry(
    angle_matrix: np.ndarray,
    feature_norms: np.ndarray,
    norm_threshold: float = 0.1,
) -> dict:
    """Classify the geometric arrangement of features.

    Examines the pairwise angle distribution among active features (those
    with sufficient weight norm) and matches against known polytope signatures.

    Args:
        angle_matrix: Pairwise angles in degrees (num_features, num_features).
        feature_norms: L2 norms of feature vectors (num_features,).
        norm_threshold: Minimum norm to consider a feature as "active".

    Returns:
        Dictionary with:
            - active_features: indices of features with sufficient norm
            - angle_distribution: histogram of pairwise angles
            - detected_structures: list of (structure_name, confidence, feature_indices)
            - dominant_structure: name of the best-matching polytope
    """
    active = np.where(feature_norms > norm_threshold)[0]
    n_active = len(active)

    if n_active < 2:
        return {
            "active_features": active,
            "angle_distribution": np.array([]),
            "detected_structures": [],
            "dominant_structure": "none",
        }

    # Extract upper triangle of pairwise angles among active features
    active_angles = angle_matrix[np.ix_(active, active)]
    upper_tri = active_angles[np.triu_indices(n_active, k=1)]

    # Match against known polytope signatures
    detected = []
    for name, sig in POLYTOPE_SIGNATURES.items():
        if n_active < sig["min_features"]:
            continue

        expected = sig["expected_angle"]
        tol = sig["tolerance"]
        matching = np.abs(upper_tri - expected) < tol
        match_frac = matching.mean() if len(upper_tri) > 0 else 0.0

        if match_frac > 0.3:  # At least 30% of angles match
            detected.append((name, float(match_frac), active.tolist()))

    # Sort by confidence
    detected.sort(key=lambda x: x[1], reverse=True)
    dominant = detected[0][0] if detected else "irregular"

    return {
        "active_features": active,
        "angle_distribution": upper_tri,
        "detected_structures": detected,
        "dominant_structure": dominant,
    }


def analyze_geometry_across_instances(
    W: torch.Tensor,
    feature_probability: Optional[torch.Tensor] = None,
) -> List[dict]:
    """Analyze geometric structure for each instance in a toy model.

    Args:
        W: Weight matrix of shape (num_instances, num_features, num_hidden).
        feature_probability: Optional tensor of shape (num_instances, 1)
            with feature probabilities per instance.

    Returns:
        List of classification results, one per instance.
    """
    num_instances = W.shape[0]
    results = []

    for i in range(num_instances):
        Wi = W[i].cpu()
        norms = Wi.norm(dim=1).numpy()
        angles = compute_pairwise_angles(Wi)
        classification = classify_feature_geometry(angles, norms)

        if feature_probability is not None:
            classification["feature_probability"] = float(
                feature_probability[i].squeeze()
            )

        classification["instance"] = i
        results.append(classification)

    return results


def plot_geometric_analysis(
    W: torch.Tensor,
    save_path: str = "images/geometric_analysis.png",
    feature_probability: Optional[torch.Tensor] = None,
    title: Optional[str] = None,
) -> List[dict]:
    """Generate a full geometric analysis plot.

    Creates a multi-panel figure showing:
    - Top row: feature vectors in 2D for selected instances
    - Bottom left: angle distribution histogram
    - Bottom right: detected structures per instance

    Args:
        W: Weight matrix of shape (num_instances, num_features, num_hidden).
        save_path: Path to save the figure.
        feature_probability: Optional feature probabilities per instance.
        title: Plot title.

    Returns:
        List of per-instance classification results.
    """
    results = analyze_geometry_across_instances(W, feature_probability)
    num_instances = W.shape[0]

    # Select up to 6 representative instances
    show_instances = min(6, num_instances)
    step = max(1, num_instances // show_instances)
    selected = list(range(0, num_instances, step))[:show_instances]

    fig = plt.figure(figsize=(14, 10))

    # Top row: feature vectors in 2D
    for idx, inst in enumerate(selected):
        ax = fig.add_subplot(2, show_instances, idx + 1)
        Wi = W[inst].cpu().numpy()

        # Project to 2D if needed
        if Wi.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            Wi_2d = pca.fit_transform(Wi)
        else:
            Wi_2d = Wi[:, :2]

        norms = np.linalg.norm(Wi, axis=1)
        active_mask = norms > 0.1

        colors = plt.cm.tab10(np.linspace(0, 1, len(Wi_2d)))
        for j, (x, y) in enumerate(Wi_2d):
            alpha = 0.9 if active_mask[j] else 0.2
            ax.plot([0, x], [0, y], color=colors[j], linewidth=2, alpha=alpha)
            ax.scatter(x, y, color=colors[j], s=60, alpha=alpha, zorder=3)

        ax.set_aspect("equal")
        lim = max(1.2, np.abs(Wi_2d).max() * 1.2) if len(Wi_2d) > 0 else 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        r = results[inst]
        prob_str = f" (p={r['feature_probability']:.3f})" if "feature_probability" in r else ""
        ax.set_title(f"Inst {inst}{prob_str}\n{r['dominant_structure']}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom left: aggregate angle distribution
    ax_hist = fig.add_subplot(2, 2, 3)
    all_angles = np.concatenate(
        [r["angle_distribution"] for r in results if len(r["angle_distribution"]) > 0]
    )
    if len(all_angles) > 0:
        ax_hist.hist(all_angles, bins=36, range=(0, 180), color="steelblue",
                     edgecolor="white", alpha=0.8)
        # Mark expected angles for known polytopes
        for name, sig in POLYTOPE_SIGNATURES.items():
            ax_hist.axvline(sig["expected_angle"], color="red", linestyle="--",
                            alpha=0.5, linewidth=1)
            ax_hist.text(sig["expected_angle"] + 1, ax_hist.get_ylim()[1] * 0.9,
                         name.replace("_", "\n"), fontsize=7, color="red",
                         rotation=90, va="top")

    ax_hist.set_xlabel("Pairwise Angle (degrees)", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.set_title("Angle Distribution (all instances)", fontsize=11)

    # Bottom right: structure summary
    ax_summary = fig.add_subplot(2, 2, 4)
    ax_summary.axis("off")

    structure_counts = Counter(r["dominant_structure"] for r in results)
    summary_lines = ["Detected Structures:\n"]
    for struct, count in structure_counts.most_common():
        pct = 100 * count / len(results)
        desc = POLYTOPE_SIGNATURES.get(struct, {}).get("description", struct)
        summary_lines.append(f"  {struct}: {count}/{len(results)} ({pct:.0f}%) — {desc}")

    summary_lines.append(f"\nTotal instances: {num_instances}")
    summary_lines.append(f"Features: {W.shape[1]}, Hidden: {W.shape[2]}")

    ax_summary.text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=ax_summary.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
    )

    if title is None:
        title = "Geometric Structure of Superposition"
    fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Geometric analysis saved to {save_path}")
    return results
