"""Unified configuration system using dataclasses.

Supports loading from YAML files, command-line overrides, and programmatic construction.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: str = "toy"  # "toy", "transformer", "translation", "computation", "continuous_thought", "coconut"
    num_features: int = 5
    num_hidden: int = 2
    num_instances: int = 10

    # Transformer-specific
    n_layers: int = 4
    n_heads: int = 4
    n_positions: int = 32

    # Translation-specific
    base_model_name: str = "Helsinki-NLP/opus-mt-en-fr"

    # Computation-in-superposition specific
    target_fn: str = "abs"  # "abs", "square", "threshold", "relu"
    mlp_hidden: int = 16

    # Continuous thought specific
    num_thought_steps: int = 4
    thought_mlp_expansion: int = 2
    use_confidence_head: bool = True

    # Coconut-specific (faithful implementation)
    coconut_base_model: str = "gpt2"  # "gpt2", "gpt2-medium", etc.
    bottleneck_dim: int = 256  # Bottleneck dimension for superposition study
    num_latent_tokens: int = 4  # Number of latent reasoning tokens


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 1024
    num_steps: int = 10000
    learning_rate: float = 1e-3
    scheduler_type: str = "constant"  # "constant", "linear", "cosine"
    seed: int = 42

    # Translation-specific
    num_epochs: int = 10
    max_samples: Optional[int] = None
    gradient_accumulation_steps: int = 1


@dataclass
class VisualizationConfig:
    """Configuration for visualization and logging."""
    viz_interval: int = 100
    log_dir: str = "runs"
    save_dir: str = "images"
    checkpoint_dir: str = "checkpoints"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration combining all sub-configs."""
    name: str = "superposition_experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def to_dict(self) -> dict:
        """Convert config to a flat dictionary."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            ExperimentConfig instance.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # Support both nested and flat YAML formats
        if "toy_model_config" in raw:
            # Legacy format compatibility
            cfg = raw["toy_model_config"]
            return cls(
                name=cfg.get("model_name", "superposition_experiment"),
                model=ModelConfig(
                    model_type="toy",
                    num_features=cfg.get("num_features", 5),
                    num_hidden=cfg.get("num_hidden", 2),
                    num_instances=cfg.get("num_instances", 10),
                ),
                training=TrainingConfig(
                    batch_size=cfg.get("batch_size", 1024),
                    num_steps=cfg.get("num_of_steps", 10000),
                    learning_rate=float(cfg.get("learning_rate", 1e-3)),
                    scheduler_type=cfg.get("scheduler_type", "constant"),
                ),
            )

        model_cfg = raw.get("model", {})
        training_cfg = raw.get("training", {})
        viz_cfg = raw.get("visualization", {})

        return cls(
            name=raw.get("name", "superposition_experiment"),
            model=ModelConfig(**model_cfg) if model_cfg else ModelConfig(),
            training=TrainingConfig(**training_cfg) if training_cfg else TrainingConfig(),
            visualization=VisualizationConfig(**viz_cfg) if viz_cfg else VisualizationConfig(),
        )

    def save_yaml(self, path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to write the YAML file.
        """
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# Preset configurations for quick experimentation
PRESETS = {
    "toy_small": ExperimentConfig(
        name="toy_small",
        model=ModelConfig(model_type="toy", num_features=5, num_hidden=2, num_instances=10),
        training=TrainingConfig(batch_size=1024, num_steps=10000, learning_rate=1e-3),
    ),
    "toy_large": ExperimentConfig(
        name="toy_large",
        model=ModelConfig(model_type="toy", num_features=20, num_hidden=5, num_instances=10),
        training=TrainingConfig(batch_size=2048, num_steps=25000, learning_rate=1e-3),
    ),
    "transformer_small": ExperimentConfig(
        name="transformer_small",
        model=ModelConfig(
            model_type="transformer", num_features=64, num_hidden=32, num_instances=8
        ),
        training=TrainingConfig(batch_size=32, num_steps=1000, learning_rate=1e-4),
    ),
    "transformer_large": ExperimentConfig(
        name="transformer_large",
        model=ModelConfig(
            model_type="transformer", num_features=128, num_hidden=64, num_instances=10
        ),
        training=TrainingConfig(batch_size=32, num_steps=5000, learning_rate=1e-4),
    ),
    "translation": ExperimentConfig(
        name="translation",
        model=ModelConfig(
            model_type="translation",
            base_model_name="Helsinki-NLP/opus-mt-en-fr",
            num_hidden=256,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=10, learning_rate=1e-4, max_samples=10000
        ),
    ),
    "computation_abs": ExperimentConfig(
        name="computation_abs",
        model=ModelConfig(
            model_type="computation", num_features=10, num_hidden=3,
            num_instances=10, target_fn="abs", mlp_hidden=32,
        ),
        training=TrainingConfig(batch_size=1024, num_steps=15000, learning_rate=1e-3),
    ),
    "computation_square": ExperimentConfig(
        name="computation_square",
        model=ModelConfig(
            model_type="computation", num_features=10, num_hidden=3,
            num_instances=10, target_fn="square", mlp_hidden=32,
        ),
        training=TrainingConfig(batch_size=1024, num_steps=15000, learning_rate=1e-3),
    ),
    "continuous_thought": ExperimentConfig(
        name="continuous_thought",
        model=ModelConfig(
            model_type="continuous_thought",
            coconut_base_model="gpt2",  # GPT2 base for decoder-only architecture
            bottleneck_dim=256,  # Bottleneck for representational superposition
            num_thought_steps=4,
            thought_mlp_expansion=2,
            use_confidence_head=True,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    # Coconut: Faithful implementation with bottleneck for superposition study
    "coconut": ExperimentConfig(
        name="coconut",
        model=ModelConfig(
            model_type="coconut",
            coconut_base_model="gpt2",
            bottleneck_dim=256,
            num_latent_tokens=4,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    "coconut_no_bottleneck": ExperimentConfig(
        name="coconut_no_bottleneck",
        model=ModelConfig(
            model_type="coconut",
            coconut_base_model="gpt2",
            bottleneck_dim=768,  # Same as hidden_size, no compression
            num_latent_tokens=4,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    # =========================================================================
    # Bottleneck Size Experiments - ContinuousThought
    # These presets test how superposition changes with bottleneck compression
    # =========================================================================
    "continuous_thought_d32": ExperimentConfig(
        name="continuous_thought_d32",
        model=ModelConfig(
            model_type="continuous_thought",
            coconut_base_model="gpt2",
            bottleneck_dim=32,  # Severe compression: 768 -> 32 (24x)
            num_thought_steps=4,
            thought_mlp_expansion=2,
            use_confidence_head=True,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    "continuous_thought_d64": ExperimentConfig(
        name="continuous_thought_d64",
        model=ModelConfig(
            model_type="continuous_thought",
            coconut_base_model="gpt2",
            bottleneck_dim=64,  # Heavy compression: 768 -> 64 (12x)
            num_thought_steps=4,
            thought_mlp_expansion=2,
            use_confidence_head=True,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    "continuous_thought_d128": ExperimentConfig(
        name="continuous_thought_d128",
        model=ModelConfig(
            model_type="continuous_thought",
            coconut_base_model="gpt2",
            bottleneck_dim=128,  # Moderate compression: 768 -> 128 (6x)
            num_thought_steps=4,
            thought_mlp_expansion=2,
            use_confidence_head=True,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    # Alias for d256 (default)
    "continuous_thought_d256": ExperimentConfig(
        name="continuous_thought_d256",
        model=ModelConfig(
            model_type="continuous_thought",
            coconut_base_model="gpt2",
            bottleneck_dim=256,  # Light compression: 768 -> 256 (3x)
            num_thought_steps=4,
            thought_mlp_expansion=2,
            use_confidence_head=True,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    # =========================================================================
    # Thought Steps Experiments - Test if more steps compensate for compression
    # =========================================================================
    "continuous_thought_d32_steps8": ExperimentConfig(
        name="continuous_thought_d32_steps8",
        model=ModelConfig(
            model_type="continuous_thought",
            coconut_base_model="gpt2",
            bottleneck_dim=32,
            num_thought_steps=8,  # More steps to compensate for compression
            thought_mlp_expansion=2,
            use_confidence_head=True,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    "continuous_thought_d64_steps8": ExperimentConfig(
        name="continuous_thought_d64_steps8",
        model=ModelConfig(
            model_type="continuous_thought",
            coconut_base_model="gpt2",
            bottleneck_dim=64,
            num_thought_steps=8,
            thought_mlp_expansion=2,
            use_confidence_head=True,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    # =========================================================================
    # Bottleneck Size Experiments - Coconut
    # =========================================================================
    "coconut_d32": ExperimentConfig(
        name="coconut_d32",
        model=ModelConfig(
            model_type="coconut",
            coconut_base_model="gpt2",
            bottleneck_dim=32,
            num_latent_tokens=4,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    "coconut_d64": ExperimentConfig(
        name="coconut_d64",
        model=ModelConfig(
            model_type="coconut",
            coconut_base_model="gpt2",
            bottleneck_dim=64,
            num_latent_tokens=4,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    "coconut_d128": ExperimentConfig(
        name="coconut_d128",
        model=ModelConfig(
            model_type="coconut",
            coconut_base_model="gpt2",
            bottleneck_dim=128,
            num_latent_tokens=4,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
    "coconut_d256": ExperimentConfig(
        name="coconut_d256",
        model=ModelConfig(
            model_type="coconut",
            coconut_base_model="gpt2",
            bottleneck_dim=256,
            num_latent_tokens=4,
        ),
        training=TrainingConfig(
            batch_size=4, num_epochs=5, learning_rate=1e-4, max_samples=5000
        ),
    ),
}
