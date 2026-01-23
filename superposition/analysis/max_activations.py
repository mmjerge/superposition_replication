"""Max-activating examples analysis for identifying polysemantic neurons.

Passes validation data through the translation model and identifies which
tokens trigger the highest activation for each bottleneck neuron, revealing
polysemanticity (e.g., a single neuron responding to both "bank" (finance)
and "bank" (river)).
"""

import collections
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from superposition.utils.logging import get_logger

logger = get_logger(__name__)


def get_max_activating_examples(
    model,
    dataloader,
    tokenizer,
    k: int = 10,
    bottleneck_layer: Optional[str] = None,
) -> Dict[int, List[Tuple[str, float]]]:
    """Identify the top-k tokens that maximally activate each bottleneck neuron.

    For each dimension in the bottleneck layer, finds which input tokens produce
    the strongest activation. Polysemantic neurons will show semantically diverse
    tokens in their top-k list.

    Args:
        model: Translation model with encoder bottleneck.
        dataloader: DataLoader yielding batches with 'input_ids' and 'attention_mask'.
        tokenizer: Tokenizer for converting IDs back to readable tokens.
        k: Number of top activations to return per neuron.
        bottleneck_layer: Name of the bottleneck layer attribute on the model.
            Defaults to 'encoder_bottleneck'.

    Returns:
        Dictionary mapping neuron index to list of (token, activation_value) tuples,
        sorted by activation strength.

    Example:
        >>> results = get_max_activating_examples(model, val_loader, tokenizer)
        >>> print("Neuron 4 top tokens:", results[4])
        [('bank', 3.21), ('river', 2.89), ('financial', 2.45), ...]
    """
    model.eval()
    activations_store: List[torch.Tensor] = []
    all_tokens: List[str] = []
    all_attention_masks: List[torch.Tensor] = []

    # Resolve the bottleneck layer
    layer_name = bottleneck_layer or "encoder_bottleneck"
    if not hasattr(model, layer_name):
        raise AttributeError(
            f"Model has no attribute '{layer_name}'. "
            f"Available attributes: {[n for n, _ in model.named_modules()]}"
        )
    target_layer = getattr(model, layer_name)

    # Register forward hook to capture bottleneck activations
    def hook_fn(module, input, output):
        activations_store.append(output.detach().cpu())

    handle = target_layer.register_forward_hook(hook_fn)

    # Pass data through model and collect activations + tokens
    logger.info("Collecting bottleneck activations...")
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Activation collection"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Run encoder through bottleneck
            encoder_outputs = model.base_model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Trigger the bottleneck forward
            _ = target_layer(encoder_outputs[0])

            # Store tokens and masks for alignment
            for seq_ids, mask in zip(input_ids.cpu(), attention_mask.cpu()):
                tokens = tokenizer.convert_ids_to_tokens(seq_ids.tolist())
                all_tokens.extend(tokens)
                all_attention_masks.append(mask)

    handle.remove()

    # Concatenate all activations: (total_tokens, bottleneck_dim)
    concatenated_acts = torch.cat(activations_store, dim=0)
    # Flatten batch and sequence dims
    concatenated_acts = concatenated_acts.view(-1, concatenated_acts.shape[-1])

    # Build attention mask to filter padding tokens
    attention_flat = torch.cat(all_attention_masks, dim=0).view(-1)
    valid_mask = attention_flat.bool()

    # Filter out padding positions
    valid_acts = concatenated_acts[valid_mask]
    valid_tokens = [tok for tok, is_valid in zip(all_tokens, valid_mask.tolist()) if is_valid]

    num_neurons = valid_acts.shape[1]
    logger.info(
        f"Analyzing {num_neurons} neurons across {len(valid_tokens)} valid tokens"
    )

    # Find top-k activating tokens for each neuron
    results: Dict[int, List[Tuple[str, float]]] = {}

    for neuron_idx in range(num_neurons):
        neuron_acts = valid_acts[:, neuron_idx]
        top_k = min(k, len(neuron_acts))
        top_values, top_indices = torch.topk(neuron_acts, top_k)

        top_entries = [
            (valid_tokens[idx], val.item())
            for idx, val in zip(top_indices.tolist(), top_values)
        ]
        results[neuron_idx] = top_entries

    return results


def find_polysemantic_neurons(
    results: Dict[int, List[Tuple[str, float]]],
    diversity_threshold: float = 0.5,
) -> List[int]:
    """Identify neurons whose top activations span diverse token types.

    A neuron is considered polysemantic if its top-k tokens have high
    lexical diversity (many unique tokens relative to total).

    Args:
        results: Output from get_max_activating_examples.
        diversity_threshold: Minimum ratio of unique tokens to consider
            a neuron polysemantic.

    Returns:
        List of neuron indices identified as polysemantic.
    """
    polysemantic = []

    for neuron_idx, entries in results.items():
        tokens = [tok.lower().strip("_▁") for tok, _ in entries]
        # Filter empty/padding tokens
        tokens = [t for t in tokens if t]
        if not tokens:
            continue
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio >= diversity_threshold:
            polysemantic.append(neuron_idx)

    logger.info(
        f"Found {len(polysemantic)}/{len(results)} polysemantic neurons "
        f"(diversity >= {diversity_threshold})"
    )
    return polysemantic


def format_activation_table(
    results: Dict[int, List[Tuple[str, float]]],
    neuron_indices: Optional[List[int]] = None,
    max_neurons: int = 20,
) -> str:
    """Format max-activation results as a readable table.

    Args:
        results: Output from get_max_activating_examples.
        neuron_indices: Specific neurons to include. If None, uses first max_neurons.
        max_neurons: Maximum number of neurons to display.

    Returns:
        Formatted string table.
    """
    if neuron_indices is None:
        neuron_indices = list(results.keys())[:max_neurons]

    lines = []
    lines.append(f"{'Neuron':>8} | {'Top Activating Tokens (token: activation)'}")
    lines.append("-" * 80)

    for idx in neuron_indices:
        entries = results.get(idx, [])
        token_strs = [f"{tok}:{val:.2f}" for tok, val in entries[:8]]
        lines.append(f"{idx:>8} | {', '.join(token_strs)}")

    return "\n".join(lines)
