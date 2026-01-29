"""Continuous thought bottleneck model for bridging superposition paradigms.

This model bridges two definitions of superposition:
1. Anthropic's "Toy Models of Superposition" (2022): features stored in
   overlapping neural representations due to dimensional compression.
2. Zhu et al.'s "Reasoning by Superposition" (2025): multiple reasoning
   traces encoded simultaneously in continuous thought vectors.

================================================================================
COCONUT CITATION AND ATTRIBUTION
================================================================================

This implementation is INSPIRED BY but NOT IDENTICAL TO the Coconut architecture:

    Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Tian, Y. (2024).
    Training Large Language Models to Reason in a Continuous Latent Space.
    arXiv preprint arXiv:2412.06769.
    https://github.com/facebookresearch/coconut
    License: MIT

WHAT COCONUT DOES (original):
- Works on causal LMs (GPT2, Llama)
- Uses special <bot>/<eot> latent tokens in the input sequence
- Replaces latent token embeddings with hidden states from previous position
- Multi-pass forward through the FULL model for each latent token
- Progressive multi-stage training (increase latent tokens over stages)

WHAT THIS IMPLEMENTATION DOES (adaptation for superposition study):
- Works on seq2seq translation (MarianMT) with a learned bottleneck
- Uses a FIXED number of thought steps (no latent tokens in input)
- Gated recurrent update in BOTTLENECK SPACE (not full model passes)
- Hidden feedback projection with residual connection
- Confidence estimation head (not in original Coconut)

KEY DIFFERENCES:
- Coconut: hidden_states[i-1] replaces embedding[i] for latent tokens
- This: gated_update(hidden) + residual feedback in bottleneck space
- Coconut: reasoning through full transformer layers per thought
- This: reasoning through small MLP in compressed bottleneck

The goal is to study whether REPRESENTATIONAL SUPERPOSITION (Anthropic) interacts
with ITERATIVE LATENT REASONING (Coconut-style) in a bottleneck setting.
================================================================================

Architecture:
    encoder → bottleneck → [thought_step × T] → expansion → decoder
    where each thought step refines the bottleneck state via a learned
    gated update rule, optionally producing a confidence estimate.
"""

import torch
from torch import nn
from transformers import MarianMTModel, MarianTokenizer
from typing import Optional

from superposition.utils.logging import get_logger

logger = get_logger(__name__)


class ContinuousThoughtModel(nn.Module):
    """Translation model with iterative continuous thought refinement.

    NOTE: This is an ADAPTATION inspired by Coconut, not a direct port.
    See module docstring for detailed attribution and differences.

    Key components:
    - encoder_bottleneck: Compresses encoder output (FROM: Anthropic superposition)
    - thought_step: MLP that proposes updates (INSPIRED BY: Coconut latent reasoning)
    - thought_gate: Controls update blending (OUR ADDITION: not in Coconut)
    - hidden_feedback: Projects state for next step (INSPIRED BY: Coconut feedback)
    - confidence_head: Estimates confidence (OUR ADDITION: not in Coconut)
    - decoder_expansion: Expands back to decoder (FROM: Anthropic superposition)

    References:
        - Coconut: https://github.com/facebookresearch/coconut (MIT License)
        - Paper: arXiv:2412.06769
    """

    def __init__(
        self,
        base_model_name: str = "Helsinki-NLP/opus-mt-en-fr",
        hidden_size: int = 256,
        num_thought_steps: int = 4,
        thought_mlp_expansion: int = 2,
        use_confidence_head: bool = True,
        device: Optional[torch.device] = None,
    ):
        """Initialize the continuous thought model.

        Args:
            base_model_name: Pre-trained MarianMT model name.
            hidden_size: Bottleneck dimension.
            num_thought_steps: Number of iterative refinement steps (T).
            thought_mlp_expansion: Expansion factor for the thought MLP.
            use_confidence_head: Whether to produce per-step confidence scores.
            device: Torch device.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_thought_steps = num_thought_steps
        self.active_thought_steps = num_thought_steps  # For curriculum learning
        self.use_confidence_head = use_confidence_head
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Loading pre-trained model: {base_model_name}")
        self.base_model = MarianMTModel.from_pretrained(base_model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(base_model_name)

        encoder_dim = self.base_model.config.d_model

        # Bottleneck compression
        self.encoder_bottleneck = nn.Linear(encoder_dim, hidden_size)

        # Thought refinement: recurrent update to bottleneck representation
        thought_dim = hidden_size * thought_mlp_expansion
        self.thought_step = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, thought_dim),
            nn.GELU(),
            nn.Linear(thought_dim, hidden_size),
        )

        # Gating mechanism: how much to update vs retain at each step
        # Inspired by Coconut's hidden state feedback where previous hidden
        # states are used to update the current representation
        self.thought_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        # Hidden state feedback projection (Coconut-inspired)
        # Projects the output hidden state back to input space for next step
        self.hidden_feedback = nn.Linear(hidden_size, hidden_size)

        # Confidence estimation head (per thought step)
        if use_confidence_head:
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid(),
            )

        # Expansion back to encoder dim
        self.decoder_expansion = nn.Linear(hidden_size, encoder_dim)

        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ContinuousThoughtModel: encoder_dim={encoder_dim}, "
            f"hidden_size={hidden_size}, thought_steps={num_thought_steps}, "
            f"trainable_params={trainable:,} / total={total:,}"
        )

        self.to(self._device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_thought_states: bool = False,
    ):
        """Forward pass with iterative thought refinement.

        Args:
            input_ids: Source token IDs.
            attention_mask: Attention mask for source.
            labels: Target token IDs (for computing loss).
            return_thought_states: If True, return intermediate thought states
                and confidence scores for analysis.

        Returns:
            Model outputs. If return_thought_states=True, also returns a dict
            with 'thought_states' and 'confidences' tensors.
        """
        # Frozen encoder
        with torch.no_grad():
            encoder_outputs = self.base_model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Compress to bottleneck
        hidden = self.encoder_bottleneck(encoder_outputs[0])

        # Iterative thought refinement (Coconut-style continuous reasoning)
        # Use active_thought_steps for curriculum learning support
        thought_states = [hidden] if return_thought_states else None
        confidences = [] if return_thought_states and self.use_confidence_head else None

        for t in range(self.active_thought_steps):
            # [INSPIRED BY COCONUT] Compute update - analogous to Coconut's
            # multi-pass forward, but here we use a small MLP in bottleneck space
            # instead of full transformer layers
            update = self.thought_step(hidden)

            # [OUR ADDITION - NOT IN COCONUT] Gated update mechanism
            # Coconut directly replaces embeddings; we use learned gating
            gate = self.thought_gate(torch.cat([hidden, update], dim=-1))
            hidden = gate * update + (1 - gate) * hidden

            # [INSPIRED BY COCONUT] Hidden state feedback
            # Coconut: hidden_states[i-1] -> embedding[i]
            # Here: project and add as residual for next iteration
            hidden = self.hidden_feedback(hidden) + hidden

            if return_thought_states:
                thought_states.append(hidden.detach())
                if self.use_confidence_head:
                    # Mean-pool over sequence for per-step confidence
                    pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
                    pooled = pooled / attention_mask.sum(dim=1, keepdim=True)
                    conf = self.confidence_head(pooled)
                    confidences.append(conf.detach())

        # Expand back to encoder dim
        expanded = self.decoder_expansion(hidden)

        # Decode through frozen decoder
        outputs = self.base_model(
            encoder_outputs=(expanded,),
            labels=labels,
            attention_mask=attention_mask,
        )

        if return_thought_states:
            extra = {
                "thought_states": torch.stack(thought_states, dim=0),
            }
            if confidences:
                extra["confidences"] = torch.stack(confidences, dim=0)
            return outputs, extra

        return outputs

    def get_bottleneck_weights(self) -> torch.Tensor:
        """Get encoder bottleneck weights for visualization."""
        return self.encoder_bottleneck.weight.detach()

    def get_thought_step_weights(self) -> dict:
        """Get thought step MLP weights for analysis."""
        return {
            name: param.detach()
            for name, param in self.thought_step.named_parameters()
        }

    def get_trainable_parameters(self):
        """Get only the trainable parameters for the optimizer."""
        return [p for p in self.parameters() if p.requires_grad]

    def set_active_thought_steps(self, num_steps: int) -> None:
        """Set the number of active thought steps for curriculum learning.

        This enables multi-stage training inspired by Coconut, where the model
        starts with fewer thought steps and gradually increases complexity.

        Args:
            num_steps: Number of thought steps to use (1 to num_thought_steps).
        """
        if num_steps < 1 or num_steps > self.num_thought_steps:
            raise ValueError(
                f"num_steps must be between 1 and {self.num_thought_steps}, "
                f"got {num_steps}"
            )
        self.active_thought_steps = num_steps
        logger.info(f"Set active thought steps to {num_steps}/{self.num_thought_steps}")

    def get_curriculum_schedule(self, total_epochs: int) -> list[int]:
        """Get a curriculum schedule for multi-stage training.

        Returns a list of epoch indices where the number of thought steps
        should increase. Inspired by Coconut's multi-stage training approach.

        Args:
            total_epochs: Total number of training epochs.

        Returns:
            List of (epoch, num_steps) tuples for curriculum progression.
        """
        if self.num_thought_steps == 1:
            return [(0, 1)]

        epochs_per_stage = max(1, total_epochs // self.num_thought_steps)
        schedule = []
        for stage in range(self.num_thought_steps):
            epoch = stage * epochs_per_stage
            num_steps = stage + 1
            schedule.append((epoch, num_steps))
        return schedule

    def analyze_thought_evolution(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """Analyze how the bottleneck representation evolves across thought steps.

        Computes per-step metrics: representation change magnitude, cosine
        similarity between consecutive steps, and confidence evolution.

        Args:
            input_ids: Source token IDs.
            attention_mask: Attention mask.

        Returns:
            Dictionary with analysis results.
        """
        self.eval()
        with torch.no_grad():
            _, extra = self.forward(
                input_ids, attention_mask, return_thought_states=True
            )

        states = extra["thought_states"]  # (T+1, batch, seq, hidden)
        num_steps = states.shape[0]

        # Compute per-step change magnitude
        step_changes = []
        step_cosines = []
        for t in range(1, num_steps):
            diff = (states[t] - states[t - 1]).norm(dim=-1).mean().item()
            step_changes.append(diff)

            # Cosine similarity between consecutive steps
            cos = nn.functional.cosine_similarity(
                states[t].flatten(1), states[t - 1].flatten(1), dim=1
            ).mean().item()
            step_cosines.append(cos)

        result = {
            "step_change_magnitude": step_changes,
            "step_cosine_similarity": step_cosines,
            "total_change": (states[-1] - states[0]).norm(dim=-1).mean().item(),
        }

        if "confidences" in extra:
            result["confidence_evolution"] = [
                c.mean().item() for c in extra["confidences"]
            ]

        return result
