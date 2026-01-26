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

  # Run with Weights & Biases logging
  python -m superposition train --model toy --wandb --wandb-project my-superposition-study

  # Run from a config file
  python -m superposition train --config config/config.yaml

  # Use a preset configuration
  python -m superposition train --preset toy_large

  # Analyze a trained model
  python -m superposition analyze --analysis interference --model toy --checkpoint model.pt
  python -m superposition analyze --analysis activations --model translation --checkpoint model.pt
  python -m superposition analyze --analysis embeddings --model translation --checkpoint model.pt

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
    train_parser.add_argument("--gradient-accumulation-steps", type=int, help="Gradient accumulation steps")

    # Visualization parameters
    train_parser.add_argument("--viz-interval", type=int, help="Visualization interval (steps)")
    train_parser.add_argument("--log-dir", type=str, help="TensorBoard log directory")
    train_parser.add_argument("--save-dir", type=str, help="Image save directory")
    train_parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")
    train_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    train_parser.add_argument("--wandb-project", type=str, help="W&B project name")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run analysis on a trained model"
    )
    analyze_parser.add_argument(
        "--analysis", type=str, required=True,
        choices=["activations", "interference", "embeddings"],
        help="Type of analysis to run",
    )
    analyze_parser.add_argument(
        "--model", type=str, required=True,
        choices=["toy", "transformer", "translation"],
        help="Model type",
    )
    analyze_parser.add_argument(
        "--checkpoint", type=str,
        help="Path to model checkpoint (.pt file)",
    )
    analyze_parser.add_argument(
        "--config", type=str,
        help="Path to YAML config used during training",
    )
    analyze_parser.add_argument(
        "--save-dir", type=str, default="images",
        help="Directory to save analysis outputs",
    )
    analyze_parser.add_argument(
        "--top-k", type=int, default=10,
        help="Top-k activations per neuron (for activations analysis)",
    )
    analyze_parser.add_argument(
        "--max-samples", type=int, default=5000,
        help="Max tokens/samples for analysis",
    )
    analyze_parser.add_argument(
        "--method", type=str, default="tsne", choices=["tsne", "pca"],
        help="Dimensionality reduction method (for embeddings analysis)",
    )
    analyze_parser.add_argument("--seed", type=int, default=42, help="Random seed")

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
    if hasattr(args, 'gradient_accumulation_steps') and args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.viz_interval is not None:
        config.visualization.viz_interval = args.viz_interval
    if args.log_dir is not None:
        config.visualization.log_dir = args.log_dir
    if args.save_dir is not None:
        config.visualization.save_dir = args.save_dir
    if args.no_tensorboard:
        config.visualization.use_tensorboard = False
    if hasattr(args, 'wandb') and args.wandb:
        config.visualization.use_wandb = True
    if hasattr(args, 'wandb_project') and args.wandb_project is not None:
        config.visualization.wandb_project = args.wandb_project

    config.model.model_type = args.model or config.model.model_type

    return config


def run_train(config: ExperimentConfig) -> None:
    """Execute a training run with the given config."""
    from superposition.training.trainer import Trainer
    from superposition.utils.reproducibility import setup_distributed, cleanup_distributed, is_main_process

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()

    set_seed(config.training.seed + rank)  # Different seed per process
    device = get_device(local_rank=local_rank)

    # Ensure output directories exist (only on main process)
    if is_main_process():
        os.makedirs(config.visualization.save_dir, exist_ok=True)
        os.makedirs(config.visualization.log_dir, exist_ok=True)

        logger.info(f"Experiment: {config.name}")
        logger.info(f"Model type: {config.model.model_type}")
        logger.info(f"Device: {device}")

    model_type = config.model.model_type

    # Auto-scale batch size for translation models to avoid OOM.
    # Translation models (seq2seq with long sequences) need much smaller
    # per-GPU batch sizes than toy/transformer models.
    if model_type == "translation":
        max_translation_batch = 16
        if config.training.batch_size > max_translation_batch:
            original_bs = config.training.batch_size
            accum_steps = max(1, original_bs // max_translation_batch)
            config.training.batch_size = max_translation_batch
            config.training.gradient_accumulation_steps = accum_steps
            if is_main_process():
                logger.warning(
                    f"Batch size {original_bs} is too large for translation model. "
                    f"Auto-scaled to batch_size={max_translation_batch} with "
                    f"gradient_accumulation_steps={accum_steps} "
                    f"(effective batch_size={max_translation_batch * accum_steps})"
                )

    trainer = Trainer(config, rank=rank, world_size=world_size)

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
            distributed=(world_size > 1),
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

    # Cleanup distributed training
    cleanup_distributed()


def run_analyze(args) -> None:
    """Run analysis on a trained model."""
    import torch

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    model_type = args.model
    analysis_type = args.analysis

    logger.info(f"Analysis: {analysis_type} on {model_type} model")

    # Load model
    if model_type == "toy":
        from superposition.models.toy import ToyModel

        config = ExperimentConfig.from_yaml(args.config) if args.config else PRESETS["toy_small"]
        model = ToyModel(
            num_features=config.model.num_features,
            num_hidden=config.model.num_hidden,
            num_instances=config.model.num_instances,
            device=device,
        )
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            logger.info(f"Loaded checkpoint: {args.checkpoint}")

    elif model_type == "transformer":
        from superposition.models.transformer import TransformerModel

        config = ExperimentConfig.from_yaml(args.config) if args.config else PRESETS["transformer_small"]
        model = TransformerModel(
            num_features=config.model.num_features,
            num_hidden=config.model.num_hidden,
            num_instances=config.model.num_instances,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            device=device,
        )
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            logger.info(f"Loaded checkpoint: {args.checkpoint}")

    elif model_type == "translation":
        from superposition.models.translation import TranslationModel

        config = ExperimentConfig.from_yaml(args.config) if args.config else PRESETS["translation"]
        model = TranslationModel(
            base_model_name=config.model.base_model_name,
            hidden_size=config.model.num_hidden,
            device=device,
        )
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.error(f"Unknown model type: {model_type}")
        sys.exit(1)

    # Run the requested analysis
    if analysis_type == "interference":
        from superposition.analysis.interference import compute_interference_heatmap, compute_interference_per_instance

        if model_type == "toy":
            compute_interference_per_instance(model, save_dir=args.save_dir)
        else:
            compute_interference_heatmap(
                model,
                save_path=f"{args.save_dir}/interference_heatmap.png",
                model_type=model_type,
            )

    elif analysis_type == "activations":
        if model_type != "translation":
            logger.error("Max-activating examples analysis requires a translation model")
            sys.exit(1)

        from superposition.analysis.max_activations import (
            get_max_activating_examples,
            find_polysemantic_neurons,
            format_activation_table,
        )
        from superposition.utils.data import TranslationDataset, create_dataloaders

        tokenizer = model.tokenizer
        dataset = TranslationDataset(
            tokenizer=tokenizer,
            max_samples=args.max_samples,
        )
        _, val_loader = create_dataloaders(dataset, batch_size=16)

        results = get_max_activating_examples(
            model, val_loader, tokenizer, k=args.top_k
        )
        polysemantic = find_polysemantic_neurons(results)

        # Print results
        print("\n" + "=" * 80)
        print("MAX-ACTIVATING EXAMPLES (All Neurons)")
        print("=" * 80)
        print(format_activation_table(results))

        if polysemantic:
            print("\n" + "=" * 80)
            print(f"POLYSEMANTIC NEURONS ({len(polysemantic)} found)")
            print("=" * 80)
            print(format_activation_table(results, neuron_indices=polysemantic))

        # Save results to file
        output_path = f"{args.save_dir}/max_activations.txt"
        with open(output_path, "w") as f:
            f.write("Max-Activating Examples Analysis\n")
            f.write("=" * 80 + "\n\n")
            f.write(format_activation_table(results))
            f.write(f"\n\nPolysemantic neurons: {polysemantic}\n")
        logger.info(f"Results saved to {output_path}")

    elif analysis_type == "embeddings":
        if model_type != "translation":
            logger.error("POS-tagged embedding analysis requires a translation model")
            sys.exit(1)

        from superposition.analysis.embeddings import (
            extract_bottleneck_embeddings,
            plot_embeddings_by_pos,
        )
        from superposition.utils.data import TranslationDataset, create_dataloaders

        tokenizer = model.tokenizer
        dataset = TranslationDataset(
            tokenizer=tokenizer,
            max_samples=args.max_samples,
        )
        _, val_loader = create_dataloaders(dataset, batch_size=16)

        embeddings, tokens, _ = extract_bottleneck_embeddings(
            model, val_loader, tokenizer, max_tokens=args.max_samples
        )

        plot_embeddings_by_pos(
            embeddings,
            tokens,
            save_path=f"{args.save_dir}/embeddings_pos_colored.png",
            sample_size=min(2000, len(tokens)),
            method=args.method,
        )


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

    if args.command == "analyze":
        run_analyze(args)


if __name__ == "__main__":
    main()
