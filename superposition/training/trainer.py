"""Unified training loop for all superposition models."""

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

from superposition.config import ExperimentConfig
from superposition.utils.logging import get_logger
from superposition.utils.visualization import SuperpositionVisualizer

logger = get_logger(__name__)


class Trainer:
    """Unified trainer for superposition experiments.

    Handles training loops for toy, transformer, and translation models with
    consistent logging, visualization, and checkpointing.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.visualizer = SuperpositionVisualizer(
            log_dir=f"{config.visualization.log_dir}/{config.name}",
            save_dir=config.visualization.save_dir,
        )

    def train_superposition_model(self, model) -> dict:
        """Train a toy or transformer superposition model.

        Args:
            model: A SuperpositionModel instance (ToyModel or TransformerModel).

        Returns:
            Dictionary with training metrics.
        """
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

        logger.info(f"Starting training: {num_steps} steps, batch_size={batch_size}, lr={lr}")

        with tqdm(range(num_steps), desc="Training", ncols=100) as pbar:
            for step in pbar:
                # Update learning rate
                step_lr = lr * lr_fn(step)
                for group in optimizer.param_groups:
                    group["lr"] = step_lr

                optimizer.zero_grad(set_to_none=True)
                batch = model.generate_data(batch_size)
                output = model(batch)
                loss = model.compute_loss(batch, output)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                running_loss = 0.9 * running_loss + 0.1 * loss_val if running_loss else loss_val
                metrics["losses"].append(loss_val)

                pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{running_loss:.4f}", lr=f"{step_lr:.2e}")

                # Logging and visualization
                if self.config.visualization.use_tensorboard:
                    self.visualizer.log_scalar("Loss/train", loss_val, step)
                    self.visualizer.log_scalar("LR", step_lr, step)

                viz_interval = self.config.visualization.viz_interval
                if step % viz_interval == 0:
                    self._visualize_model(model, step)

        metrics["final_loss"] = running_loss
        logger.info(f"Training complete. Final avg loss: {running_loss:.6f}")
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
        cfg = self.config.training
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=cfg.learning_rate)
        device = model._device

        metrics = {"epoch_losses": [], "bleu_scores": []}

        logger.info(f"Starting translation training: {cfg.num_epochs} epochs, lr={cfg.learning_rate}")

        for epoch in range(cfg.num_epochs):
            model.train()
            running_loss = 0.0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}") as pbar:
                for i, batch in enumerate(pbar):
                    optimizer.zero_grad()
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    loss_val = loss.item()
                    running_loss = 0.9 * running_loss + 0.1 * loss_val if running_loss else loss_val
                    pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{running_loss:.4f}")

                    global_step = epoch * len(train_loader) + i
                    if self.config.visualization.use_tensorboard and i % self.config.visualization.viz_interval == 0:
                        self.visualizer.log_scalar("Loss/train", loss_val, global_step)
                        weights = model.get_bottleneck_weights().cpu().numpy()
                        self.visualizer.log_histogram("bottleneck/weights", weights, global_step)

            metrics["epoch_losses"].append(running_loss)
            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {running_loss:.4f}")

            # Validation with BLEU
            if val_loader is not None:
                bleu_score = self._evaluate_translation(model, val_loader)
                metrics["bleu_scores"].append(bleu_score)
                if self.config.visualization.use_tensorboard:
                    self.visualizer.log_scalar("BLEU/validation", bleu_score, epoch)

        self.visualizer.close()
        return metrics

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
