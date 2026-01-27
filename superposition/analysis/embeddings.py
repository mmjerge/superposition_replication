"""Embedding visualization with linguistic structure (POS tagging).

Replaces unreadable scatter plots with semantically-colored visualizations
that prove linguistic structure is preserved through the bottleneck by
coloring points by Part-of-Speech tags.
"""

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from superposition.utils.logging import get_logger

logger = get_logger(__name__)


def extract_bottleneck_embeddings(
    model,
    dataloader,
    tokenizer,
    max_tokens: int = 5000,
) -> tuple:
    """Extract bottleneck representations and their corresponding tokens.

    Passes data through the encoder and bottleneck layer, collecting
    the compressed representations alongside readable token strings.

    Args:
        model: Translation model with encoder bottleneck.
        dataloader: DataLoader yielding batches with 'input_ids' and 'attention_mask'.
        tokenizer: Tokenizer for ID-to-token conversion.
        max_tokens: Maximum number of tokens to collect.

    Returns:
        Tuple of (embeddings, tokens, attention_masks) where:
            - embeddings: Tensor of shape (num_valid_tokens, bottleneck_dim)
            - tokens: List of token strings
            - attention_masks: Boolean tensor indicating valid (non-padding) positions
    """
    model.eval()
    all_embeddings = []
    all_tokens = []
    all_masks = []
    total_tokens = 0
    device = next(model.parameters()).device

    logger.info("Extracting bottleneck embeddings...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if total_tokens >= max_tokens:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get encoder outputs
            encoder_outputs = model.base_model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Pass through bottleneck
            bottleneck_out = model.encoder_bottleneck(encoder_outputs[0])

            all_embeddings.append(bottleneck_out.cpu())
            all_masks.append(attention_mask.cpu())

            # Convert token IDs to strings
            for seq_ids in input_ids.cpu():
                tokens = tokenizer.convert_ids_to_tokens(seq_ids.tolist())
                all_tokens.extend(tokens)

            total_tokens += input_ids.numel()

    # Concatenate and flatten
    embeddings = torch.cat(all_embeddings, dim=0).view(-1, all_embeddings[0].shape[-1])
    masks = torch.cat(all_masks, dim=0).view(-1).bool()

    # Filter padding
    valid_embeddings = embeddings[masks]
    valid_tokens = [tok for tok, m in zip(all_tokens, masks.tolist()) if m]

    # Truncate to max_tokens
    valid_embeddings = valid_embeddings[:max_tokens]
    valid_tokens = valid_tokens[:max_tokens]

    logger.info(f"Extracted {len(valid_tokens)} valid token embeddings")
    return valid_embeddings, valid_tokens, masks


def get_pos_tags(tokens: List[str]) -> List[str]:
    """Get Universal POS tags for a list of tokens.

    Handles subword tokens (e.g., MarianMT's sentencepiece tokens) by
    stripping leading markers before tagging.

    Args:
        tokens: List of token strings (may include subword markers like '▁').

    Returns:
        List of POS tag strings (e.g., 'NOUN', 'VERB', 'ADJ').
    """
    import nltk

    # Ensure required NLTK data is available
    _nltk_resources = [
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/universal_tagset", "universal_tagset"),
    ]
    for path, name in _nltk_resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

    # Clean subword markers for better POS tagging
    cleaned_tokens = []
    for tok in tokens:
        clean = tok.replace("▁", "").replace("Ġ", "").strip()
        if not clean or clean in ("<pad>", "<s>", "</s>", "<unk>"):
            cleaned_tokens.append("<pad>")
        else:
            cleaned_tokens.append(clean)

    # Tag tokens
    tagged = nltk.pos_tag(cleaned_tokens, tagset="universal")
    pos_tags = [tag for _, tag in tagged]

    # Replace padding with a special tag
    pos_tags = [tag if tok != "<pad>" else "PAD" for tag, tok in zip(pos_tags, cleaned_tokens)]

    return pos_tags


def plot_embeddings_by_pos(
    embeddings: torch.Tensor,
    tokens: List[str],
    save_path: str = "images/embeddings_pos_colored.png",
    sample_size: int = 2000,
    method: str = "tsne",
    perplexity: float = 30.0,
    title: Optional[str] = None,
) -> None:
    """Plot bottleneck embeddings colored by Part-of-Speech.

    Visually demonstrates that linguistic structure (noun clusters, verb clusters)
    is preserved even after compression through the superposition bottleneck.

    Args:
        embeddings: Tensor of shape (num_tokens, bottleneck_dim).
        tokens: List of token strings corresponding to embeddings.
        save_path: Path to save the figure.
        sample_size: Number of tokens to include (for readability).
        method: Dimensionality reduction method ('tsne' or 'pca').
        perplexity: t-SNE perplexity parameter.
        title: Plot title. Auto-generated if None.
    """
    # Sample if needed
    n = min(sample_size, len(tokens))
    if n < len(tokens):
        indices = np.random.choice(len(tokens), n, replace=False)
        embeddings_sample = embeddings[indices].numpy()
        tokens_sample = [tokens[i] for i in indices]
    else:
        embeddings_sample = embeddings[:n].numpy()
        tokens_sample = tokens[:n]

    # Get POS tags
    logger.info("Computing POS tags...")
    pos_tags = get_pos_tags(tokens_sample)

    # Filter out PAD tokens
    valid_mask = [tag != "PAD" for tag in pos_tags]
    embeddings_filtered = embeddings_sample[valid_mask]
    tokens_filtered = [t for t, v in zip(tokens_sample, valid_mask) if v]
    pos_filtered = [p for p, v in zip(pos_tags, valid_mask) if v]

    if len(embeddings_filtered) < 10:
        logger.warning("Too few valid tokens for visualization after filtering")
        return

    # Dimensionality reduction
    logger.info(f"Running {method.upper()} reduction on {len(embeddings_filtered)} tokens...")
    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(embeddings_filtered) - 1))
        reduced = reducer.fit_transform(embeddings_filtered)
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings_filtered)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'pca'.")

    # Build DataFrame for plotting
    import pandas as pd

    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "POS": pos_filtered,
        "Token": tokens_filtered,
    })

    # Define POS color order for consistency
    pos_order = ["NOUN", "VERB", "ADJ", "ADV", "ADP", "DET", "PRON", "CONJ", "NUM", "PRT", "."]
    present_pos = [p for p in pos_order if p in df["POS"].unique()]
    remaining = [p for p in df["POS"].unique() if p not in pos_order]
    hue_order = present_pos + remaining

    # Plot
    if title is None:
        title = f"Bottleneck Embeddings (d={embeddings.shape[1]}) Colored by Part-of-Speech"

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="POS",
        hue_order=hue_order,
        style="POS",
        alpha=0.7,
        s=40,
        ax=ax,
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="POS Tag")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"POS-colored embedding plot saved to {save_path}")

    # Report POS distribution
    pos_counts = df["POS"].value_counts()
    logger.info(f"POS distribution:\n{pos_counts.to_string()}")
