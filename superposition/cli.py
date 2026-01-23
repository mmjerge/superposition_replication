"""Command-line interface for running superposition experiments."""

import argparse
import os
import sys

from superposition.config import ExperimentConfig, PRESETS
from superposition.utils.reproducibility import set_seed, get_device
from superposition.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Superposition Replication Study - Run experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run toy model with default settings
  python -m superposition train --model toy

  # Run transformer with custom parameters
  python -m superposition train --model transformer --num-features 128 --num-hidden 64

  # Run from a config file
  python -m superposition train --config experiment.yaml

  # Use a preset configuration
  python -m superposition train --preset toy_large

  # List available presets
  python -m superposition presets
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a superposition model")
    train_parser.add_argument(
        "--model", type=str, choices=["toy", "transformer", "translation"],
        default="toy", help="Model type to train",
    )
    train_parser.add_argument("--config", type=str, help="Path to YAML config file")
    train_parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), help="Use a preset configuration")

    # Model parameters
    train_parser.add_argument("--num-features", type=int, help="Number of features")
    train_parser.add_argument("--num-hidden", type=int, help="Hidden dimension size")
    train_parser.add_argument("--num-instances", type=int, help="Number of instances")

    # Training parameters
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--num-steps", type=int, help="Number of training steps")
    train_parser.add_argument("--num-epochs", type=int, help="Number of epochs (translation)")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--scheduler", type=str, choices=["constant", "linear", "cosine"], help="LR scheduler")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--max-samples", type=int, help="Max training samples (translation)")

    # Visualization parameters
    train_parser.add_argument("--viz-interval", type=int, help="Visualization interval (steps)")
    train_parser.add_argument("--log-dir", type=str, help="TensorBoard log directory")
    train_parser.add_argument("--save-dir", type=str, help="Image save directory")
    train_parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")

    # Presets command
    subparsers.add_parser("presets", help="List available preset configurations")

    return parser


def resolve_config(args) -> ExperimentConfig:
    """Build an ExperimentConfig from CLI arguments, applying overrides."""
    if args.preset:
        config = PRESETS[args.preset]
    elif args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()
        config.model.model_type = args.model

    # Apply CLI overrides
    if args.num_features is not None:
        config.model.num_features = args.num_features
    if args.num_hidden is not None:
        config.model.num_hidden = args.num_hidden
    if args.num_instances is not None:
        config.model.num_instances = args.num_instances
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_steps is not None:
        config.training.num_steps = args.num_steps
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.scheduler is not None:
        config.training.scheduler_type = args.scheduler
    if args.seed is not None:
        config.training.seed = args.seed
    if args.max_samples is not None:
        config.training.max_samples = args.max_samples
    if args.viz_interval is not None:
        config.visualization.viz_interval = args.viz_interval
    if args.log_dir is not None:
        config.visualization.log_dir = args.log_dir
    if args.save_dir is not None:
        config.visualization.save_dir = args.save_dir
    if args.no_tensorboard:
        config.visualization.use_tensorboard = False

    config.model.model_type = args.model or config.model.model_type

    return config


def run_train(config: ExperimentConfig) -> None:
    """Execute a training run with the given config."""
    from superposition.training.trainer import Trainer

    set_seed(config.training.seed)
    device = get_device()

    # Ensure output directories exist
    os.makedirs(config.visualization.save_dir, exist_ok=True)
    os.makedirs(config.visualization.log_dir, exist_ok=True)

    logger.info(f"Experiment: {config.name}")
    logger.info(f"Model type: {config.model.model_type}")
    logger.info(f"Device: {device}")

    model_type = config.model.model_type
    trainer = Trainer(config)

    if model_type == "toy":
        from superposition.models.toy import ToyModel

        model = ToyModel(
            num_features=config.model.num_features,
            num_hidden=config.model.num_hidden,
            num_instances=config.model.num_instances,
            device=device,
        )
        trainer.train_superposition_model(model)

    elif model_type == "transformer":
        from superposition.models.transformer import TransformerModel

        model = TransformerModel(
            num_features=config.model.num_features,
            num_hidden=config.model.num_hidden,
            num_instances=config.model.num_instances,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            device=device,
        )
        trainer.train_superposition_model(model)

    elif model_type == "translation":
        from superposition.models.translation import TranslationModel
        from superposition.utils.data import TranslationDataset, create_dataloaders
        from transformers import MarianTokenizer

        tokenizer = MarianTokenizer.from_pretrained(config.model.base_model_name)
        dataset = TranslationDataset(
            tokenizer=tokenizer,
            max_samples=config.training.max_samples,
        )
        train_loader, val_loader = create_dataloaders(
            dataset,
            batch_size=config.training.batch_size,
        )

        model = TranslationModel(
            base_model_name=config.model.base_model_name,
            hidden_size=config.model.num_hidden,
            device=device,
        )
        trainer.train_translation_model(model, train_loader, val_loader)

    else:
        logger.error(f"Unknown model type: {model_type}")
        sys.exit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "presets":
        print("\nAvailable presets:\n")
        for name, preset in PRESETS.items():
            print(f"  {name:20s} - {preset.model.model_type} model, "
                  f"features={preset.model.num_features}, "
                  f"hidden={preset.model.num_hidden}")
        print()
        sys.exit(0)

    if args.command == "train":
        config = resolve_config(args)
        run_train(config)


if __name__ == "__main__":
    main()
