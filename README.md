# Superposition Replication Study

This repository replicates and extends the findings from Anthropic's ["Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html) paper. It provides modular implementations for studying how neural networks encode multiple features in overlapping representations.

## Project Structure

```
superposition_replication/
├── superposition/                    # Main package
│   ├── __init__.py
│   ├── __main__.py                   # Entry point for `python -m superposition`
│   ├── cli.py                        # Command-line interface
│   ├── config.py                     # Dataclass-based configuration system
│   ├── models/
│   │   ├── base.py                   # Base class with shared functionality
│   │   ├── toy.py                    # W^T W autoencoder model
│   │   ├── transformer.py           # GPT2-based transformer model
│   │   └── translation.py           # MarianMT bottleneck model
│   ├── training/
│   │   └── trainer.py               # Unified training loop
│   ├── analysis/
│   │   ├── max_activations.py       # Polysemantic neuron detection
│   │   ├── interference.py          # Cosine similarity heatmaps
│   │   └── embeddings.py            # POS-tagged embedding visualization
│   └── utils/
│       ├── data.py                   # Dataset classes and loaders
│       ├── logging.py               # Structured logging
│       ├── reproducibility.py       # Seeding and device management
│       └── visualization.py         # Plotting and TensorBoard utilities
├── config/                           # Configuration files
│   ├── config.yaml                   # Default experiment configuration
│   ├── toy_models_config.yaml       # Legacy toy model config
│   └── test.sbatch                   # SLURM batch script (HPC)
├── tests/
│   └── tests.py                     # pytest test suite
├── notebooks/
│   └── anthropic_toy_models.ipynb   # Interactive toy model experiments
├── plots/
│   ├── intro_diagram_transformer_superposition.py
│   └── intro_diagram_translation_superposition.py
├── legacy/                           # Original standalone scripts
│   ├── toy_models_reproduction.py
│   ├── transformer_superposition.py
│   └── translation_superposition.py
├── checkpoints/                      # Model checkpoints (saved after training)
├── images/                           # Generated visualizations
├── runs/                             # TensorBoard logs
├── pyproject.toml                    # Package metadata and dependencies
├── uv.lock                           # Dependency lockfile (uv)
├── environment.yaml                  # Conda environment specification
└── LICENSE
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project and all dependencies
uv sync

# Install with optional wandb support
uv sync --extra wandb
```

### Alternative: pip install

```bash
pip install -e .

# With wandb support
pip install -e ".[wandb]"
```

## Usage

### CLI Interface

```bash
# Run toy model with defaults
uv run python -m superposition train --model toy

# Run transformer model with custom parameters
uv run python -m superposition train --model transformer --num-features 128 --num-hidden 64 --num-steps 5000

# Run from a config file
uv run python -m superposition train --config config/config.yaml

# Use a preset configuration
uv run python -m superposition train --preset toy_large

# Translation model with bottleneck
uv run python -m superposition train --model translation --num-hidden 256 --max-samples 10000

# List available presets
uv run python -m superposition presets
```

### Analysis Tools

Model checkpoints are saved automatically to `checkpoints/` after training (e.g. `checkpoints/translation.pt`). Three analysis tools address key questions about superposition:

```bash
# 1. Cosine similarity heatmap: which features interfere?
uv run python -m superposition analyze --analysis interference --model toy

# 2. Max-activating examples: which tokens share a neuron? (polysemanticity)
uv run python -m superposition analyze --analysis activations --model translation --checkpoint checkpoints/translation.pt

# 3. POS-tagged embeddings: is linguistic structure preserved through the bottleneck?
uv run python -m superposition analyze --analysis embeddings --model translation --checkpoint checkpoints/translation.pt --method tsne
```

### Programmatic Usage

```python
from superposition.models import ToyModel, TransformerModel
from superposition.config import ExperimentConfig, PRESETS
from superposition.training import Trainer
from superposition.utils import set_seed

# Set up reproducibility
set_seed(42)

# Use a preset or build custom config
config = PRESETS["toy_small"]

# Create model and train
model = ToyModel(num_features=10, num_hidden=3, num_instances=10)
trainer = Trainer(config)
metrics = trainer.train_superposition_model(model)

# Run interference analysis
from superposition.analysis import compute_interference_heatmap
compute_interference_heatmap(model, model_type="toy", save_path="images/interference.png")
```

### Configuration

Experiments are configured via YAML files or CLI arguments. See `config/config.yaml` for the full schema:

```yaml
name: my_experiment
model:
  model_type: toy
  num_features: 5
  num_hidden: 2
  num_instances: 10
training:
  batch_size: 1024
  num_steps: 10000
  learning_rate: 0.001
  scheduler_type: cosine
  seed: 42
visualization:
  viz_interval: 100
  log_dir: runs
  save_dir: images
  checkpoint_dir: checkpoints
  use_tensorboard: true
  use_wandb: false
  wandb_project: null
```

### Experiment Tracking

The repository supports both **TensorBoard** and **Weights & Biases** for experiment tracking:

#### TensorBoard (default)

```bash
# TensorBoard is enabled by default
uv run python -m superposition train --model toy

# View logs
tensorboard --logdir runs
```

#### Weights & Biases

```bash
# Enable wandb via CLI
uv run python -m superposition train --model toy --wandb --wandb-project my-project

# Or via config file
# Set use_wandb: true in your YAML config
```

Both logging systems can be used simultaneously. wandb requires the optional dependency:
```bash
uv sync --extra wandb
```

### Multi-GPU Training (DistributedDataParallel)

The repository supports distributed training across multiple GPUs using PyTorch's DistributedDataParallel (DDP):

```bash
# Train on 4 GPUs
torchrun --nproc_per_node=4 -m superposition train --model translation --wandb --wandb-project my-project

# Train on 8 GPUs with custom parameters
torchrun --nproc_per_node=8 -m superposition train \
  --model translation \
  --num-hidden 512 \
  --batch-size 32 \
  --num-epochs 20 \
  --wandb --wandb-project superposition-study

# Train across multiple nodes (e.g., 2 nodes with 4 GPUs each)
# On node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=NODE0_IP --master_port=29500 \
  -m superposition train --model translation --wandb

# On node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=NODE0_IP --master_port=29500 \
  -m superposition train --model translation --wandb
```

**Key features:**
- Automatic model parallelization with DDP
- Efficient gradient synchronization across GPUs
- Linear scaling with number of GPUs
- Logging and visualization only on rank 0 (avoids duplicates)

**Note:** When using distributed training:
- Batch size is per GPU (effective batch size = batch_size × num_gpus)
- Only the main process (rank 0) logs to TensorBoard/wandb
- Each process gets a slightly different random seed for better exploration

## Models

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| **Toy** | W^T W autoencoder demonstrating basic superposition | `num_features`, `num_hidden`, `num_instances` |
| **Transformer** | GPT2-based model studying superposition with attention | `num_features`, `num_hidden`, `n_layers`, `n_heads` |
| **Translation** | MarianMT with learned bottleneck | `base_model_name`, `hidden_size` |

## Analysis

| Analysis | Purpose | Output |
|----------|---------|--------|
| **Interference Heatmap** | Shows cosine similarity between learned feature directions | Heatmap PNG showing orthogonal vs. superposed features |
| **Max-Activating Examples** | Identifies tokens that maximally activate each bottleneck neuron | Table of polysemantic neurons and their top tokens |
| **POS-Tagged Embeddings** | Colors embeddings by Part-of-Speech to verify linguistic structure | Scatter plot with NOUN/VERB/ADJ clusters |

## Testing

```bash
# Run tests
uv run pytest tests/tests.py -v

# Run with coverage
uv run pytest tests/tests.py --cov=superposition --cov-report=term-missing
```

## Key Concepts

- **Superposition**: Networks encoding more features than dimensions by using overlapping representations
- **Feature Probability**: Sparsity level of each feature (sparser features are more likely to superpose)
- **Importance**: Relative weight of features in the loss function
- **Polysemanticity**: Individual neurons responding to multiple unrelated features
- **Interference**: The degree to which feature directions overlap (measured by cosine similarity)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{superposition_replication,
  title = {Model Superposition Replication Study},
  year = {2024},
  url = {https://github.com/mmjerge/superposition_replication}
}
```
