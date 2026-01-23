"""Reproducibility utilities for consistent experiment results."""

import os
import random
import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True, local_rank: int = -1) -> torch.device:
    """Get the best available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available.
        local_rank: Local rank for distributed training. If -1, uses single GPU mode.

    Returns:
        torch.device for computation.
    """
    if prefer_cuda and torch.cuda.is_available():
        if local_rank >= 0:
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cuda")
    return torch.device("cpu")


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training process group.

    Returns:
        Tuple of (rank, local_rank, world_size).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Cleanup distributed training process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0).

    Returns:
        True if main process or not distributed.
    """
    return not dist.is_initialized() or dist.get_rank() == 0
