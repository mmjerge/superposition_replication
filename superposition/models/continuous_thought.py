"""Continuous thought model bridging representational and reasoning superposition.

This model bridges two definitions of superposition:
1. Anthropic's "Toy Models of Superposition" (2022): features stored in
   overlapping neural representations due to dimensional compression.
2. Coconut's reasoning superposition (Hao et al., 2024): iterative reasoning
   in continuous latent space via hidden state feedback.

================================================================================
DESIGN RATIONALE
================================================================================

To fairly compare with Coconut and study the interaction between representational
and reasoning superposition, this model:

1. Uses GPT2 (decoder-only) - SAME architecture as Coconut
2. Adds a BOTTLENECK layer - forces representational superposition
3. Has iterative THOUGHT STEPS - enables reasoning superposition study

Architecture:
    input → embedding → GPT2_layers → BOTTLENECK → thought_steps → output
                                         ↑              ↑
                              Representational    Reasoning
                              Superposition       Superposition

This lets us ask: "When features are compressed (superposed) in a bottleneck,
how does that affect the model's ability to do iterative reasoning?"

================================================================================
COCONUT CITATION
================================================================================

    Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Tian, Y. (2024).
    Training Large Language Models to Reason in a Continuous Latent Space.
    arXiv preprint arXiv:2412.06769.
    https://github.com/facebookresearch/coconut
    License: MIT

================================================================================
"""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional

from superposition.utils.logging import get_logger

logger = get_logger(__name__)


class ContinuousThoughtModel(nn.Module):
    """GPT2-based model with bottleneck and iterative thought steps.

    This model bridges representational superposition (via bottleneck compression)
    with reasoning superposition (via iterative thought refinement).

    Architecture:
        input_ids → GPT2 → hidden_states → BOTTLENECK → thought_steps → lm_head → logits
                                              ↓
                                    Forces features to superpose
                                    (representational superposition)

    The thought steps iteratively refine the bottleneck representation before
    projecting back to vocabulary space, enabling study of how compressed
    representations affect reasoning.

    Key components:
    - bottleneck_down: Compresses hidden states (forces representational superposition)
    - thought_step: MLP that proposes updates (reasoning in compressed space)
    - thought_gate: Controls update blending
    - hidden_feedback: Projects state for next iteration
    - confidence_head: Estimates confidence at each step
    - bottleneck_up: Expands back to hidden dimension

    References:
        - Coconut: https://github.com/facebookresearch/coconut (MIT License)
        - Anthropic: https://transformer-circuits.pub/2022/toy_model/index.html
    """

    def __init__(
        self,
        base_model_name: str = "gpt2",
        bottleneck_dim: int = 256,
        num_thought_steps: int = 4,
        thought_mlp_expansion: int = 2,
        use_confidence_head: bool = True,
        device: Optional[torch.device] = None,
    ):
        """Initialize the continuous thought model.

        Args:
            base_model_name: Pre-trained GPT2 model name (gpt2, gpt2-medium, etc.)
            bottleneck_dim: Dimension of bottleneck (smaller = more superposition)
            num_thought_steps: Number of iterative refinement steps
            thought_mlp_expansion: Expansion factor for thought MLP
            use_confidence_head: Whether to produce per-step confidence scores
            device: Torch device
        """
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.num_thought_steps = num_thought_steps
        self.active_thought_steps = num_thought_steps  # For curriculum learning
        self.use_confidence_head = use_confidence_head
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load GPT2 (decoder-only, same as Coconut)
        logger.info(f"Loading base model: {base_model_name}")
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base_model.config.pad_token_id = self.tokenizer.eos_token_id

        self.hidden_size = self.base_model.config.n_embd  # 768 for gpt2

        # ==================== BOTTLENECK (Representational Superposition) ====================
        # Compress hidden states to force features to share dimensions
        self.bottleneck_down = nn.Linear(self.hidden_size, bottleneck_dim)
        self.bottleneck_up = nn.Linear(bottleneck_dim, self.hidden_size)

        # ==================== THOUGHT STEPS (Reasoning Superposition) ====================
        # Iterative refinement in the compressed bottleneck space
        thought_dim = bottleneck_dim * thought_mlp_expansion
        self.thought_step = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, thought_dim),
            nn.GELU(),
            nn.Linear(thought_dim, bottleneck_dim),
        )

        # Gating mechanism: how much to update vs retain at each step
        self.thought_gate = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.Sigmoid(),
        )

        # Hidden state feedback projection
        self.hidden_feedback = nn.Linear(bottleneck_dim, bottleneck_dim)

        # Confidence estimation head (per thought step)
        if use_confidence_head:
            self.confidence_head = nn.Sequential(
                nn.Linear(bottleneck_dim, bottleneck_dim // 4),
                nn.ReLU(),
                nn.Linear(bottleneck_dim // 4, 1),
                nn.Sigmoid(),
            )

        # Freeze base model - only train bottleneck and thought components
        for param in self.base_model.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ContinuousThoughtModel: hidden_size={self.hidden_size}, "
            f"bottleneck_dim={bottleneck_dim}, thought_steps={num_thought_steps}, "
            f"trainable_params={trainable:,} / total={total:,}"
        )

        self.to(self._device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_thought_states: bool = False,
    ):
        """Forward pass with bottleneck compression and iterative thought refinement.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target token IDs for loss computation
            return_thought_states: If True, return intermediate states for analysis

        Returns:
            Model outputs with loss. If return_thought_states=True, also returns
            dict with 'thought_states' and 'confidences'.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Get hidden states from frozen GPT2
        with torch.no_grad():
            transformer_outputs = self.base_model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = transformer_outputs.last_hidden_state  # [batch, seq, hidden]

        # ==================== BOTTLENECK COMPRESSION ====================
        # Compress to bottleneck dimension (forces representational superposition)
        compressed = self.bottleneck_down(hidden_states)  # [batch, seq, bottleneck]

        # ==================== ITERATIVE THOUGHT REFINEMENT ====================
        thought_states = [compressed] if return_thought_states else None
        confidences = [] if return_thought_states and self.use_confidence_head else None

        hidden = compressed
        for t in range(self.active_thought_steps):
            # Compute update proposal
            update = self.thought_step(hidden)

            # Gated update: blend old state with proposed update
            gate = self.thought_gate(torch.cat([hidden, update], dim=-1))
            hidden = gate * update + (1 - gate) * hidden

            # Hidden state feedback (residual)
            hidden = self.hidden_feedback(hidden) + hidden

            if return_thought_states:
                thought_states.append(hidden.detach())
                if self.use_confidence_head:
                    # Mean-pool over sequence for per-step confidence
                    pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
                    pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    conf = self.confidence_head(pooled)
                    confidences.append(conf.detach())

        # ==================== EXPAND AND COMPUTE LOGITS ====================
        expanded = self.bottleneck_up(hidden)  # [batch, seq, hidden]

        # Use GPT2's language model head
        logits = self.base_model.lm_head(expanded)  # [batch, seq, vocab]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Create output object
        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
        outputs = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
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
        """Get bottleneck projection weights for interference analysis."""
        return self.bottleneck_down.weight.detach()

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
        """Set the number of active thought steps for curriculum learning."""
        if num_steps < 1 or num_steps > self.num_thought_steps:
            raise ValueError(
                f"num_steps must be between 1 and {self.num_thought_steps}, "
                f"got {num_steps}"
            )
        self.active_thought_steps = num_steps
        logger.info(f"Set active thought steps to {num_steps}/{self.num_thought_steps}")

    def get_curriculum_schedule(self, total_epochs: int) -> list:
        """Get a curriculum schedule for multi-stage training."""
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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Analyze how the bottleneck representation evolves across thought steps."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        self.eval()
        with torch.no_grad():
            _, extra = self.forward(
                input_ids, attention_mask, return_thought_states=True
            )

        states = extra["thought_states"]  # (T+1, batch, seq, bottleneck)
        num_steps = states.shape[0]

        step_changes = []
        step_cosines = []
        for t in range(1, num_steps):
            diff = (states[t] - states[t - 1]).norm(dim=-1).mean().item()
            step_changes.append(diff)

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

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate text using the model."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(generated, attention_mask=None)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated = torch.cat([generated, next_token], dim=1)

        return generated
