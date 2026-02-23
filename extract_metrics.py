"""Extract quantitative metrics from all model checkpoints for the empirical section.

Computes:
- Gram matrix statistics (mean off-diagonal, max coherence, near-orthogonal fraction)
- Effective capacity (n_eff) from weight norms
- Welch bound comparison
- Cross-model comparisons
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from superposition.config import PRESETS


def compute_gram_metrics(weight_matrix: torch.Tensor) -> dict:
    """Compute interference metrics from a weight matrix.

    Args:
        weight_matrix: (n_features, d) where n_features are packed into d dims.

    Returns:
        Dictionary of metrics.
    """
    n, d = weight_matrix.shape

    # Compute norms
    norms = weight_matrix.norm(dim=1)

    # Normalize for cosine similarity
    normed = F.normalize(weight_matrix.float(), p=2, dim=1)
    G = (normed @ normed.T).cpu().numpy()

    # Off-diagonal mask
    mask = ~np.eye(n, dtype=bool)
    off_diag = np.abs(G[mask])

    # Metrics
    mu_mean = float(off_diag.mean())
    mu_max = float(off_diag.max())
    mu_std = float(off_diag.std())

    # Fraction of near-orthogonal pairs (|G_ij| < 0.1)
    near_orth = float((off_diag < 0.1).mean())

    # Fraction with moderate interference (0.1 <= |G_ij| < 0.3)
    moderate = float(((off_diag >= 0.1) & (off_diag < 0.3)).mean())

    # Fraction with strong interference (|G_ij| >= 0.3)
    strong = float((off_diag >= 0.3).mean())

    # Effective capacity: count features with ||w_i|| >= 0.5
    n_eff_05 = int((norms >= 0.5).sum().item())

    # Also count with threshold 0.1
    n_eff_01 = int((norms >= 0.1).sum().item())

    # Mean norm
    mean_norm = float(norms.mean().item())

    # Welch bound: μ >= sqrt((n-d) / (d(n-1)))
    if n > d and n > 1:
        welch_bound = float(np.sqrt((n - d) / (d * (n - 1))))
    else:
        welch_bound = 0.0

    # Ratio to Welch bound
    welch_ratio = mu_max / welch_bound if welch_bound > 0 else float('inf')

    return {
        "n_features": n,
        "d_bottleneck": d,
        "compression_ratio": n / d if d > 0 else float('inf'),
        "mu_mean": mu_mean,
        "mu_max": mu_max,
        "mu_std": mu_std,
        "near_orthogonal_frac": near_orth,
        "moderate_interference_frac": moderate,
        "strong_interference_frac": strong,
        "n_eff_05": n_eff_05,
        "n_eff_01": n_eff_01,
        "mean_norm": mean_norm,
        "welch_bound": welch_bound,
        "welch_ratio": welch_ratio,
    }


def extract_weight_matrix(checkpoint_path: str, model_type: str, preset_name: str) -> torch.Tensor:
    """Load a checkpoint and extract the bottleneck weight matrix.

    Returns weight_matrix of shape (n_features, d_bottleneck).
    """
    device = torch.device("cpu")
    state_dict = torch.load(checkpoint_path, map_location=device)

    if model_type == "toy":
        # W shape: (num_instances, num_features, num_hidden)
        W = state_dict["W"]
        # Average across instances
        return W.mean(dim=0)  # (num_features, num_hidden)

    elif model_type == "translation":
        # encoder_bottleneck.weight: (hidden, encoder_dim)
        return state_dict["encoder_bottleneck.weight"].T  # (encoder_dim, hidden)

    elif model_type == "computation":
        # W_enc: (num_instances, num_features, num_hidden)
        W = state_dict["W_enc"]
        return W.mean(dim=0)

    elif model_type == "continuous_thought":
        # bottleneck_down.weight: (bottleneck_dim, hidden_size=768)
        return state_dict["bottleneck_down.weight"].T  # (768, bottleneck_dim)

    elif model_type == "coconut":
        config = PRESETS.get(preset_name)
        bottleneck_dim = config.model.bottleneck_dim if config else 256

        if "bottleneck_down.weight" in state_dict:
            w = state_dict["bottleneck_down.weight"]
            # Check if it's a real bottleneck (dims differ) or passthrough
            if w.shape[0] != w.shape[1]:
                return w.T  # (768, bottleneck_dim)
            else:
                # No compression — use embedding weights instead
                return state_dict["embedding.weight"]
        else:
            return state_dict["embedding.weight"]

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main():
    checkpoint_dir = "checkpoints"
    output_path = "images/metrics_summary.json"

    # Define all models to analyze
    models = [
        ("toy_small", "toy", "toy_small"),
        ("translation", "translation", "translation"),
        ("computation_abs", "computation", "computation_abs"),
        ("continuous_thought", "continuous_thought", "continuous_thought"),
        ("continuous_thought_d32", "continuous_thought", "continuous_thought_d32"),
        ("continuous_thought_d64", "continuous_thought", "continuous_thought_d64"),
        ("continuous_thought_d128", "continuous_thought", "continuous_thought_d128"),
        ("continuous_thought_d256", "continuous_thought", "continuous_thought_d256"),
        ("coconut", "coconut", "coconut"),
        ("coconut_no_bottleneck", "coconut", "coconut_no_bottleneck"),
        ("coconut_d32", "coconut", "coconut_d32"),
        ("coconut_d64", "coconut", "coconut_d64"),
        ("coconut_d128", "coconut", "coconut_d128"),
        ("coconut_d256", "coconut", "coconut_d256"),
    ]

    all_metrics = {}

    for name, model_type, preset in models:
        ckpt_path = os.path.join(checkpoint_dir, f"{name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {name}: checkpoint not found at {ckpt_path}")
            continue

        try:
            W = extract_weight_matrix(ckpt_path, model_type, preset)
            metrics = compute_gram_metrics(W)
            all_metrics[name] = metrics

            print(f"  {name:30s} | n={metrics['n_features']:4d}, d={metrics['d_bottleneck']:4d} | "
                  f"μ_max={metrics['mu_max']:.4f} | μ_mean={metrics['mu_mean']:.4f} | "
                  f"orth={metrics['near_orthogonal_frac']:.2%} | "
                  f"Welch={metrics['welch_bound']:.4f} | ratio={metrics['welch_ratio']:.2f}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save metrics
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")

    # Print comparison tables
    print("\n" + "=" * 100)
    print("COMPARISON: Continuous Thought vs Coconut at same bottleneck dimension")
    print("=" * 100)
    print(f"{'Model':30s} | {'d':>4s} | {'μ_max':>8s} | {'μ_mean':>8s} | {'orth%':>8s} | {'n_eff':>5s} | {'Welch':>8s} | {'ratio':>6s}")
    print("-" * 100)

    for d in [32, 64, 128, 256]:
        ct_key = f"continuous_thought_d{d}"
        co_key = f"coconut_d{d}"

        for key in [ct_key, co_key]:
            if key in all_metrics:
                m = all_metrics[key]
                print(f"  {key:28s} | {m['d_bottleneck']:4d} | {m['mu_max']:8.4f} | "
                      f"{m['mu_mean']:8.4f} | {m['near_orthogonal_frac']:7.2%} | "
                      f"{m['n_eff_05']:5d} | {m['welch_bound']:8.4f} | {m['welch_ratio']:6.2f}")
        print()

    # Special: coconut_no_bottleneck
    if "coconut_no_bottleneck" in all_metrics:
        m = all_metrics["coconut_no_bottleneck"]
        print(f"  {'coconut_no_bottleneck':28s} | {m['d_bottleneck']:4d} | {m['mu_max']:8.4f} | "
              f"{m['mu_mean']:8.4f} | {m['near_orthogonal_frac']:7.2%} | "
              f"{m['n_eff_05']:5d} | {m['welch_bound']:8.4f} | {m['welch_ratio']:6.2f}")

    # Print capacity scaling
    print("\n" + "=" * 80)
    print("CAPACITY SCALING: n_eff vs d")
    print("=" * 80)
    print(f"{'d':>4s} | {'CT n_eff':>8s} | {'CO n_eff':>8s} | {'Welch bound':>11s} | {'Predicted':>9s}")
    print("-" * 50)

    for d in [32, 64, 128, 256]:
        ct_key = f"continuous_thought_d{d}"
        co_key = f"coconut_d{d}"
        ct_neff = all_metrics.get(ct_key, {}).get("n_eff_05", "N/A")
        co_neff = all_metrics.get(co_key, {}).get("n_eff_05", "N/A")

        # Welch bound for n=768 features in d dims
        n = 768
        wb = np.sqrt((n - d) / (d * (n - 1))) if n > d else 0

        print(f"  {d:3d} | {str(ct_neff):>8s} | {str(co_neff):>8s} | {wb:11.4f} | {d:9d}")

    # Phase diagram data analysis
    phase_data_path = "images/phase_diagram_data.npz"
    if os.path.exists(phase_data_path):
        print("\n" + "=" * 80)
        print("PHASE DIAGRAM ANALYSIS")
        print("=" * 80)

        data = np.load(phase_data_path)
        feature_probs = data["feature_probs"]
        importance_ratios = data["importance_ratios"]
        dimensionality = data["dimensionality"]  # (importance_steps, sparsity_steps)

        print(f"Feature probability range: [{feature_probs.min():.4f}, {feature_probs.max():.4f}]")
        print(f"Importance ratio range: [{importance_ratios.min():.4f}, {importance_ratios.max():.4f}]")
        print(f"Dimensionality grid shape: {dimensionality.shape}")

        # Find transition boundaries
        # Superposition threshold: dimensionality > 1.0 indicates superposition
        superposition_mask = dimensionality > 1.0

        # For each importance level, find the sparsity at which superposition begins
        print("\nSuperposition boundary (sparsity S at which dim > 1.0):")
        for i_idx in range(0, len(importance_ratios), max(1, len(importance_ratios) // 5)):
            imp = importance_ratios[i_idx]
            row = dimensionality[i_idx]
            sp_indices = np.where(row > 1.0)[0]
            if len(sp_indices) > 0:
                # Superposition at low feature prob (high sparsity)
                threshold_prob = feature_probs[sp_indices[0]]
                threshold_sparsity = 1 - threshold_prob
                print(f"  importance={imp:.3f}: S_crit ≈ {threshold_sparsity:.3f} (prob={threshold_prob:.4f})")
            else:
                print(f"  importance={imp:.3f}: no superposition detected")

        # Mean dimensionality in different regimes
        sparse_mask = feature_probs < 0.1
        dense_mask = feature_probs > 0.5

        sparse_dim = dimensionality[:, sparse_mask].mean() if sparse_mask.any() else float('nan')
        dense_dim = dimensionality[:, dense_mask].mean() if dense_mask.any() else float('nan')

        print(f"\nMean dimensionality in sparse regime (prob < 0.1): {sparse_dim:.3f}")
        print(f"Mean dimensionality in dense regime (prob > 0.5): {dense_dim:.3f}")

    # Toy model per-instance analysis
    toy_path = os.path.join(checkpoint_dir, "toy_small.pt")
    if os.path.exists(toy_path):
        print("\n" + "=" * 80)
        print("TOY MODEL: Per-Instance Analysis")
        print("=" * 80)

        state_dict = torch.load(toy_path, map_location="cpu")
        W = state_dict["W"]  # (num_instances, num_features, num_hidden)
        feature_probs = state_dict.get("feature_probability", None)

        n_instances, n_features, n_hidden = W.shape
        print(f"Shape: {n_instances} instances × {n_features} features × {n_hidden} hidden")

        for i in range(n_instances):
            Wi = W[i]  # (n_features, n_hidden)
            norms = Wi.norm(dim=1)
            normed = F.normalize(Wi.float(), p=2, dim=1)
            G = (normed @ normed.T).numpy()
            mask = ~np.eye(n_features, dtype=bool)
            off_diag = np.abs(G[mask])

            prob_str = f"{feature_probs[i].item():.4f}" if feature_probs is not None else "?"
            n_active = int((norms > 0.1).sum().item())

            print(f"  Instance {i:2d} (prob={prob_str}): "
                  f"active={n_active}/{n_features}, "
                  f"μ_max={off_diag.max():.4f}, "
                  f"μ_mean={off_diag.mean():.4f}, "
                  f"norms=[{', '.join(f'{n:.3f}' for n in norms.tolist())}]")


if __name__ == "__main__":
    main()
