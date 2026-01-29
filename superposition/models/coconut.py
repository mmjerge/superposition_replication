"""Coconut (Chain of Continuous Thought) model with bottleneck for superposition study.

This is a FAITHFUL implementation of the Coconut architecture from:

    Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Tian, Y. (2024).
    Training Large Language Models to Reason in a Continuous Latent Space.
    arXiv preprint arXiv:2412.06769.
    https://github.com/facebookresearch/coconut
    License: MIT

The core Coconut mechanism:
1. Special latent tokens (<bot>, <eot>, <latent>) are added to vocabulary
2. During forward pass, latent token positions are identified
3. Hidden states from position [i-1] REPLACE embeddings at position [i] for latent tokens
4. Multiple forward passes through the FULL model for each latent token
5. KV cache is reused for efficiency

This implementation adds a BOTTLENECK LAYER to study superposition:
- After the base model's embedding, we compress through a bottleneck
- This forces representational superposition (Anthropic-style)
- Combined with Coconut's iterative reasoning, we can study the interaction

Architecture:
    input_ids → embedding → [BOTTLENECK] → transformer → hidden_states
                                ↓
    latent_token[i] ← hidden_states[i-1] (feedback loop)
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from typing import Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from superposition.utils.logging import get_logger

logger = get_logger(__name__)

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "hidden_states"])
MAX_N_LATENT = 8


class CoconutBottleneckModel(nn.Module):
    """Coconut model with bottleneck layer for superposition analysis.

    This combines:
    1. COCONUT (faithful): Latent tokens with hidden state feedback
    2. ANTHROPIC: Bottleneck compression for superposition study

    The bottleneck is applied AFTER embedding, before the transformer layers.
    This creates a compressed representation space where we can study:
    - How features superpose in the bottleneck
    - How Coconut's iterative reasoning affects superposition patterns
    - Whether confidence/accuracy correlates with geometric structure

    Based on: https://github.com/facebookresearch/coconut (MIT License)
    """

    # Special token definitions
    LATENT_TOKEN = "<|latent|>"
    BOT_TOKEN = "<|bot|>"  # Beginning of thought
    EOT_TOKEN = "<|eot|>"  # End of thought

    def __init__(
        self,
        model_name: str = "gpt2",
        bottleneck_dim: Optional[int] = 256,
        num_latent_tokens: int = 4,
        device: Optional[torch.device] = None,
    ):
        """Initialize Coconut model with bottleneck.

        Args:
            model_name: HuggingFace model name (gpt2, gpt2-medium, etc.)
            bottleneck_dim: Dimension of bottleneck. If None, no bottleneck is used.
            num_latent_tokens: Default number of latent tokens for generation.
            device: Torch device.
        """
        super().__init__()

        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_latent_tokens = num_latent_tokens
        self.bottleneck_dim = bottleneck_dim
        self.gen_forward_cnt = 0

        # Load base model and tokenizer
        logger.info(f"Loading base model: {model_name}")
        self.base_causallm = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                self.LATENT_TOKEN,
                self.BOT_TOKEN,
                self.EOT_TOKEN,
            ],
            "pad_token": "<|pad|>",
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        self.base_causallm.resize_token_embeddings(len(self.tokenizer))
        logger.info(f"Added {num_added} special tokens")

        # Store special token IDs
        self.latent_token_id = self.tokenizer.convert_tokens_to_ids(self.LATENT_TOKEN)
        self.bot_token_id = self.tokenizer.convert_tokens_to_ids(self.BOT_TOKEN)
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids(self.EOT_TOKEN)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        # Get embedding layer
        self.embedding = self.base_causallm.transformer.get_input_embeddings()
        self.hidden_size = self.base_causallm.config.n_embd

        # [BOTTLENECK FOR SUPERPOSITION STUDY]
        # This is our addition - compress embeddings to study superposition
        if bottleneck_dim is not None and bottleneck_dim < self.hidden_size:
            self.use_bottleneck = True
            self.bottleneck_down = nn.Linear(self.hidden_size, bottleneck_dim)
            self.bottleneck_up = nn.Linear(bottleneck_dim, self.hidden_size)
            logger.info(
                f"Bottleneck enabled: {self.hidden_size} -> {bottleneck_dim} -> {self.hidden_size}"
            )
        else:
            self.use_bottleneck = False
            self.bottleneck_down = None
            self.bottleneck_up = None
            logger.info("No bottleneck (full dimensionality)")

        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")

        self.to(self._device)

    def apply_bottleneck(self, embeds: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck compression and expansion.

        Args:
            embeds: Embeddings of shape (batch, seq_len, hidden_size)

        Returns:
            Processed embeddings, same shape as input.
        """
        if not self.use_bottleneck:
            return embeds

        # Compress -> Expand (forces superposition in bottleneck_dim space)
        compressed = self.bottleneck_down(embeds)
        expanded = self.bottleneck_up(compressed)
        return expanded

    def get_bottleneck_representation(self, embeds: torch.Tensor) -> torch.Tensor:
        """Get the compressed bottleneck representation for analysis.

        Args:
            embeds: Embeddings of shape (batch, seq_len, hidden_size)

        Returns:
            Compressed representation of shape (batch, seq_len, bottleneck_dim)
        """
        if not self.use_bottleneck:
            return embeds
        return self.bottleneck_down(embeds)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        return_bottleneck_states: bool = False,
        **kwargs,
    ):
        """Forward pass with Coconut's latent token mechanism.

        This is the FAITHFUL Coconut forward pass:
        1. Find all latent token positions
        2. For each latent token, do a forward pass up to that position
        3. Replace the latent token embedding with hidden_states[i-1]
        4. Continue to next latent token or end

        Args:
            input_ids: Token IDs with latent tokens inserted
            attention_mask: Attention mask
            labels: Target labels for loss computation
            position_ids: Position IDs (auto-generated if None)
            return_bottleneck_states: If True, return bottleneck representations

        Returns:
            Outputs namedtuple with loss, inputs_embeds, logits, hidden_states
        """
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        logits_list = []
        bottleneck_states = [] if return_bottleneck_states else None

        # [COCONUT] Find latent token positions
        latent_indices = (input_ids == self.latent_token_id).nonzero()

        # Organize by batch instance
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
        if max_n_latents == 0:
            max_n_latents = max(1, max([len(l) for l in latent_lists]))

        # Get initial embeddings and apply bottleneck
        inputs_embeds = self.embedding(input_ids)

        # [BOTTLENECK] Apply compression
        inputs_embeds = self.apply_bottleneck(inputs_embeds)

        if return_bottleneck_states and self.use_bottleneck:
            bottleneck_states.append(self.get_bottleneck_representation(
                self.embedding(input_ids)
            ).detach())

        # Determine initial compute range
        next_compute_range = (0, input_ids.shape[1])
        if max_n_latents > 0 and len(latent_indices) > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        # [COCONUT] Multi-pass forward for each latent token
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # First forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0]:next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0]:next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0]:next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                # Reuse KV cache
                past_key_values = [
                    (
                        k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0]:next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0]:next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits_list.append(outputs.logits)

            # Update compute range for next pass
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[-1]  # Last layer
            kv_cache = outputs.past_key_values

            # [COCONUT CORE] Feedback hidden states to latent token embeddings
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # Reconstruct inputs_embeds with hidden state feedback
            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                # [COCONUT] Replace latent token with preceding hidden state
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            inputs_embeds = torch.stack([
                torch.stack(tensor_list[batch_idx])
                for batch_idx in range(inputs_embeds.shape[0])
            ])

            if return_bottleneck_states and self.use_bottleneck:
                # Track how bottleneck representation evolves
                bottleneck_states.append(
                    self.bottleneck_down(inputs_embeds).detach()
                )

        # Final forward pass
        final_past_kv = None
        if kv_cache is not None:
            final_past_kv = [
                (
                    k[:, :, :next_compute_range[0], :],
                    v[:, :, :next_compute_range[0], :],
                )
                for k, v in kv_cache
            ]

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0]:next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=final_past_kv,
            output_hidden_states=True,
        )

        logits_list.append(outputs.logits)
        self.gen_forward_cnt += max_n_latents + 1

        # Concatenate all logits
        logits = torch.cat(logits_list, dim=-2)

        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        result = Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            hidden_states=bottleneck_states if return_bottleneck_states else None,
        )

        return result

    def prepare_input_with_latent_tokens(
        self,
        text: str,
        num_latent: Optional[int] = None,
    ) -> dict:
        """Prepare input by inserting latent tokens.

        Args:
            text: Input text
            num_latent: Number of latent tokens to insert (default: self.num_latent_tokens)

        Returns:
            Dict with input_ids, attention_mask, labels
        """
        if num_latent is None:
            num_latent = self.num_latent_tokens

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Insert latent tokens after the input
        latent_tokens = [self.latent_token_id] * num_latent
        tokens_with_latent = tokens + latent_tokens

        input_ids = torch.tensor([tokens_with_latent], device=self._device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        # Mask loss for latent tokens
        labels[:, -num_latent:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_bottleneck_weights(self) -> Optional[torch.Tensor]:
        """Get bottleneck projection weights for interference analysis."""
        if not self.use_bottleneck:
            return None
        return self.bottleneck_down.weight.detach()

    def analyze_thought_evolution(
        self,
        text: str,
        num_latent: Optional[int] = None,
    ) -> dict:
        """Analyze how representations evolve across latent token processing.

        Args:
            text: Input text
            num_latent: Number of latent tokens

        Returns:
            Dict with analysis metrics
        """
        self.eval()
        inputs = self.prepare_input_with_latent_tokens(text, num_latent)

        with torch.no_grad():
            outputs = self(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["labels"],
                return_bottleneck_states=True,
            )

        if outputs.hidden_states is None or len(outputs.hidden_states) < 2:
            return {"error": "No bottleneck states to analyze"}

        states = outputs.hidden_states
        num_steps = len(states)

        step_changes = []
        step_cosines = []

        for t in range(1, num_steps):
            diff = (states[t] - states[t - 1]).norm(dim=-1).mean().item()
            step_changes.append(diff)

            cos = nn.functional.cosine_similarity(
                states[t].flatten(1), states[t - 1].flatten(1), dim=1
            ).mean().item()
            step_cosines.append(cos)

        return {
            "num_latent_passes": num_steps,
            "step_change_magnitude": step_changes,
            "step_cosine_similarity": step_cosines,
            "total_change": (states[-1] - states[0]).norm(dim=-1).mean().item(),
        }

    def generate(
        self,
        text: str,
        max_new_tokens: int = 50,
        num_latent: Optional[int] = None,
    ) -> str:
        """Generate text using Coconut reasoning.

        Args:
            text: Input prompt
            max_new_tokens: Maximum tokens to generate
            num_latent: Number of latent reasoning tokens

        Returns:
            Generated text
        """
        self.eval()
        inputs = self.prepare_input_with_latent_tokens(text, num_latent)

        with torch.no_grad():
            outputs = self(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["labels"],
            )

            # Get embeddings after latent processing
            inputs_embeds = outputs.inputs_embeds

            # Generate tokens autoregressively
            generated_ids = inputs["input_ids"][0].tolist()

            for _ in range(max_new_tokens):
                next_token = torch.argmax(outputs.logits[0, -1]).item()
                if next_token == self.eos_token_id:
                    break

                generated_ids.append(next_token)

                # Get embedding for new token
                new_embed = self.embedding(
                    torch.tensor([[next_token]], device=self._device)
                )
                if self.use_bottleneck:
                    new_embed = self.apply_bottleneck(new_embed)

                inputs_embeds = torch.cat([inputs_embeds, new_embed], dim=1)

                outputs = self.base_causallm(inputs_embeds=inputs_embeds)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
