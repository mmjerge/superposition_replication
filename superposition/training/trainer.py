"""Unified training loop for all superposition models."""

import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from typing import Optional

from superposition.config import ExperimentConfig
from superposition.utils.logging import get_logger
from superposition.utils.visualization import SuperpositionVisualizer
from superposition.utils.reproducibility import is_main_process

logger = get_logger(__name__)


class Trainer:
    """Unified trainer for superposition experiments.

    Handles training loops for toy, transformer, and translation models with
    consistent logging, visualization, and checkpointing. Supports both single-GPU
    and distributed multi-GPU training.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

        # Only initialize wandb and visualizer on main process
        if is_main_process():
            # Initialize wandb if configured
            wandb_config = None
            if config.visualization.use_wandb:
                try:
                    import wandb
                    wandb.init(
                        project=config.visualization.wandb_project or "superposition-replication",
                        name=config.name,
                        config=config.to_dict(),
                    )
                    wandb_config = config.to_dict()
                except ImportError:
                    logger.warning("wandb not installed. Install with: uv sync --extra wandb")

            self.visualizer = SuperpositionVisualizer(
                log_dir=f"{config.visualization.log_dir}/{config.name}",
                save_dir=config.visualization.save_dir,
                use_wandb=config.visualization.use_wandb,
                wandb_project=config.visualization.wandb_project,
                wandb_config=wandb_config,
            )
        else:
            self.visualizer = None

    def train_superposition_model(self, model) -> dict:
        """Train a toy or transformer superposition model.

        Args:
            model: A SuperpositionModel instance (ToyModel or TransformerModel).

        Returns:
            Dictionary with training metrics.
        """
        # Wrap model with DDP if distributed
        if self.is_distributed:
            model = DDP(model, device_ids=[self.rank])

        cfg = self.config.training
        lr = cfg.learning_rate
        num_steps = cfg.num_steps
        batch_size = cfg.batch_size

        scheduler_fns = {
            "linear": lambda step: 1 - (step / num_steps),
            "cosine": lambda step: np.cos(0.5 * np.pi * step / max(num_steps - 1, 1)),
            "constant": lambda step: 1.0,
        }
        lr_fn = scheduler_fns.get(cfg.scheduler_type, scheduler_fns["constant"])

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        metrics = {"losses": [], "final_loss": 0.0}
        running_loss = 0.0

        if is_main_process():
            logger.info(f"Starting training: {num_steps} steps, batch_size={batch_size}, lr={lr}")
            if self.is_distributed:
                logger.info(f"Distributed training on {self.world_size} GPUs")

        # Only show progress bar on main process
        pbar = tqdm(range(num_steps), desc="Training", ncols=100, disable=not is_main_process())

        for step in pbar:
            # Update learning rate
            step_lr = lr * lr_fn(step)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            optimizer.zero_grad(set_to_none=True)

            # Get the actual model (unwrap DDP if needed)
            actual_model = model.module if isinstance(model, DDP) else model
            batch = actual_model.generate_data(batch_size)
            output = model(batch)
            loss = actual_model.compute_loss(batch, output)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            running_loss = 0.9 * running_loss + 0.1 * loss_val if running_loss else loss_val
            metrics["losses"].append(loss_val)

            if is_main_process():
                pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{running_loss:.4f}", lr=f"{step_lr:.2e}")

                # Logging and visualization (only on main process)
                if self.config.visualization.use_tensorboard:
                    self.visualizer.log_scalar("Loss/train", loss_val, step)
                    self.visualizer.log_scalar("LR", step_lr, step)

                viz_interval = self.config.visualization.viz_interval
                if step % viz_interval == 0:
                    self._visualize_model(actual_model, step)

        metrics["final_loss"] = running_loss

        if is_main_process():
            logger.info(f"Training complete. Final avg loss: {running_loss:.6f}")
            checkpoint_dir = self.config.visualization.checkpoint_dir
            checkpoint_path = os.path.join(checkpoint_dir, f"{self.config.name}.pt")
            self._save_checkpoint(model, checkpoint_path)
            self.visualizer.close()

        return metrics

    def train_translation_model(self, model, train_loader, val_loader=None) -> dict:
        """Train a translation bottleneck model.

        Args:
            model: A TranslationModel instance.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.

        Returns:
            Dictionary with training metrics.
        """
        # Wrap model with DDP if distributed
        if self.is_distributed:
            model = DDP(model, device_ids=[self.rank])

        cfg = self.config.training
        actual_model = model.module if isinstance(model, DDP) else model
        optimizer = torch.optim.AdamW(actual_model.get_trainable_parameters(), lr=cfg.learning_rate)
        device = actual_model._device
        grad_accum_steps = cfg.gradient_accumulation_steps

        metrics = {"epoch_losses": [], "bleu_scores": []}

        if is_main_process():
            logger.info(f"Starting translation training: {cfg.num_epochs} epochs, lr={cfg.learning_rate}")
            if grad_accum_steps > 1:
                logger.info(
                    f"Gradient accumulation: {grad_accum_steps} steps "
                    f"(effective batch_size={cfg.batch_size * grad_accum_steps})"
                )
            if self.is_distributed:
                logger.info(f"Distributed training on {self.world_size} GPUs")

        for epoch in range(cfg.num_epochs):
            # Set epoch for DistributedSampler if using distributed training
            if self.is_distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            running_loss = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
                disable=not is_main_process()
            )

            optimizer.zero_grad()
            for i, batch in enumerate(pbar):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps
                loss.backward()

                if (i + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                loss_val = loss.item() * grad_accum_steps  # unscaled loss for logging
                running_loss = 0.9 * running_loss + 0.1 * loss_val if running_loss else loss_val

                if is_main_process():
                    pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{running_loss:.4f}")

                    global_step = epoch * len(train_loader) + i
                    if self.config.visualization.use_tensorboard and i % self.config.visualization.viz_interval == 0:
                        self.visualizer.log_scalar("Loss/train", loss_val, global_step)
                        weights = actual_model.get_bottleneck_weights().cpu().numpy()
                        self.visualizer.log_histogram("bottleneck/weights", weights, global_step)

            # Flush remaining accumulated gradients at end of epoch
            if (i + 1) % grad_accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            metrics["epoch_losses"].append(running_loss)

            # All ranks must synchronize before validation to prevent
            # non-main ranks from starting the next epoch's DDP training
            # while rank 0 is still evaluating (which causes NCCL timeout).
            if self.is_distributed:
                dist.barrier()

            if is_main_process():
                logger.info(f"Epoch {epoch + 1} complete. Avg loss: {running_loss:.4f}")

                # Validation with BLEU (only on main process)
                if val_loader is not None:
                    bleu_score = self._evaluate_translation(actual_model, val_loader)
                    metrics["bleu_scores"].append(bleu_score)
                    if self.config.visualization.use_tensorboard:
                        self.visualizer.log_scalar("BLEU/validation", bleu_score, epoch)

            # Wait for rank 0 to finish validation before all ranks
            # proceed to the next epoch together.
            if self.is_distributed:
                dist.barrier()

        if is_main_process():
            checkpoint_dir = self.config.visualization.checkpoint_dir
            checkpoint_path = os.path.join(checkpoint_dir, f"{self.config.name}.pt")
            self._save_checkpoint(model, checkpoint_path)
            self.visualizer.close()

        return metrics

    def _save_checkpoint(self, model, checkpoint_path: str) -> None:
        """Save model checkpoint to disk.

        Args:
            model: The model to save (will unwrap DDP if needed).
            checkpoint_path: Path to save the checkpoint file.
        """
        actual_model = model.module if isinstance(model, DDP) else model
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(actual_model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _visualize_model(self, model, step: int) -> None:
        """Generate visualizations based on model type."""
        from superposition.models.toy import ToyModel
        from superposition.models.transformer import TransformerModel

        if isinstance(model, ToyModel):
            weights = model.get_weight_matrix().cpu().numpy()
            self.visualizer.plot_intro_diagram(weights, save_path=f"{self.config.visualization.save_dir}/weights_step_{step}.png")
        elif isinstance(model, TransformerModel):
            weights = model.get_input_weights().cpu().numpy()
            self.visualizer.plot_weight_vectors_2d(
                weights[np.newaxis, ...],
                step=step,
                num_instances=min(model.num_instances, 8),
            )

    def _evaluate_translation(self, model, val_loader) -> float:
        """Evaluate translation model with BLEU score."""
        try:
            from evaluate import load as load_metric

            model.eval()
            bleu_metric = load_metric("sacrebleu")
            translations = []
            references = []
            device = model._device

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model.base_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=50,
                    )

                    decoded_preds = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    decoded_labels = model.tokenizer.batch_decode(labels, skip_special_tokens=True)

                    translations.extend(decoded_preds)
                    references.extend([[label] for label in decoded_labels])

            result = bleu_metric.compute(predictions=translations, references=references)
            bleu_score = result["score"]
            logger.info(f"BLEU score: {bleu_score:.2f}")
            return bleu_score

        except ImportError:
            logger.warning("'evaluate' package not installed, skipping BLEU evaluation")
            return 0.0
