"""Dataset utilities for superposition experiments."""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple


class SyntheticSuperpositionDataset(Dataset):
    """Generates synthetic sparse feature data for superposition experiments.

    Creates batches of sparse feature vectors where each feature is active
    with a given probability, simulating the sparse feature distributions
    that lead to superposition in neural networks.
    """

    def __init__(
        self,
        num_instances: int,
        num_features: int,
        feature_probability: torch.Tensor,
        num_samples: int = 10000,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_instances = num_instances
        self.num_features = num_features
        self.feature_probability = feature_probability
        self.num_samples = num_samples
        self.device = device

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        feature = torch.rand(self.num_instances, self.num_features)
        mask = torch.rand(self.num_instances, self.num_features) <= self.feature_probability.cpu()
        return torch.where(mask, feature, torch.zeros(()))


class TranslationDataset(Dataset):
    """Dataset for translation superposition experiments using HuggingFace datasets.

    Loads parallel text data and tokenizes it for use with sequence-to-sequence models.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 128,
        max_samples: Optional[int] = None,
        dataset_name: str = "iwslt2017",
        dataset_config: str = "iwslt2017-en-fr",
        src_lang: str = "en",
        tgt_lang: str = "fr",
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)[split]
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]

        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": src_encoding["input_ids"].squeeze(),
            "attention_mask": src_encoding["attention_mask"].squeeze(),
            "labels": tgt_encoding["input_ids"].squeeze(),
        }


def create_dataloaders(
    dataset: Dataset,
    batch_size: int,
    train_split: float = 0.8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Split a dataset and create train/val dataloaders.

    Args:
        dataset: The full dataset.
        batch_size: Batch size for both loaders.
        train_split: Fraction of data for training.
        num_workers: Number of dataloader workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
