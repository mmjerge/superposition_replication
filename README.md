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
│   └── utils/
│       ├── data.py                   # Dataset classes and loaders
│       ├── logging.py               # Structured logging
│       ├── reproducibility.py       # Seeding and device management
│       └── visualization.py         # Plotting and TensorBoard utilities
├── config.yaml                       # Default experiment configuration
├── pyproject.toml                    # Package metadata and dependencies
├── environment.yaml                  # Conda environment specification
├── images/                           # Generated visualizations
├── runs/                             # TensorBoard logs
└── (legacy scripts)                  # Original standalone scripts
```

## Installation

```bash
# From source (recommended for development)
pip install -e .

# Or with conda environment
conda env create -f environment.yaml
conda activate superposition
pip install -e .
```

## Usage

### CLI Interface

```bash
# Run toy model with defaults
python -m superposition train --model toy

# Run transformer model with custom parameters
python -m superposition train --model transformer --num-features 128 --num-hidden 64 --num-steps 5000

# Run from a config file
python -m superposition train --config config.yaml

# Use a preset configuration
python -m superposition train --preset toy_large

# Translation model with bottleneck
python -m superposition train --model translation --num-hidden 256 --max-samples 10000

# List available presets
python -m superposition presets
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

# Or build from scratch
config = ExperimentConfig()
config.model.num_features = 10
config.model.num_hidden = 3
config.training.num_steps = 5000

# Create model and train
model = ToyModel(num_features=10, num_hidden=3, num_instances=10)
trainer = Trainer(config)
metrics = trainer.train_superposition_model(model)
```

### Configuration

Experiments are configured via YAML files or CLI arguments. See `config.yaml` for the full schema:

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
```

## Models

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| **Toy** | W^T W autoencoder demonstrating basic superposition | `num_features`, `num_hidden`, `num_instances` |
| **Transformer** | GPT2-based model studying superposition with attention | `num_features`, `num_hidden`, `n_layers`, `n_heads` |
| **Translation** | MarianMT with learned bottleneck | `base_model_name`, `hidden_size` |

## Key Concepts

- **Superposition**: Networks encoding more features than dimensions by using overlapping representations
- **Feature Probability**: Sparsity level of each feature (sparser features are more likely to superpose)
- **Importance**: Relative weight of features in the loss function
- **Polysemanticity**: Individual neurons responding to multiple unrelated features

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
