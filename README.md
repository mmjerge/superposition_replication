# Superposition Replication Study

This repository contains implementations and experiments exploring superposition phenomena in different neural network architectures. Superposition is a phenomenon where neural networks learn to encode multiple features in the same set of weights or neurons, effectively compressing information through overlapping representations.

## Project Structure

```
├── images/                               # Visualization outputs and diagrams
├── runs/                                 # Training logs and experiment results
├── scripts/                              # Utility scripts and helpers
├── anthropic_toy_models.ipynb            # Jupyter notebook with toy model implementations
├── environment.yaml                      # Conda environment specification
├── intro_transformer_superposition.py     # Basic transformer superposition examples
├── intro_translation_superposition.py     # Translation model superposition examples
├── toy_models_config.yaml                # Configuration for toy model experiments
├── toy_models_reproduction.py            # Scripts to reproduce toy model results
├── transformer_superposition.py           # Advanced transformer implementations
├── translation_superposition.py          # Advanced translation model experiments
└── LICENSE                               # Project license
```

## Getting Started

### Prerequisites

To run the experiments, you'll need Python 3.7+ and conda installed. Set up the environment using:

```bash
conda env create -f environment.yaml
conda activate superposition
```

### Running Experiments

The repository includes several implementations:

1. **Toy Models**
   ```bash
   jupyter notebook anthropic_toy_models.ipynb
   ```
   Explore basic superposition concepts with simple model implementations.

2. **Basic Examples**
   ```bash
   python intro_transformer_superposition.py
   python intro_translation_superposition.py
   ```
   Demonstrates superposition patterns in transformer and translation architectures.

3. **Toy Model Reproduction**
   ```bash
   python toy_models_reproduction.py
   ```
   Reproduces results from toy model experiments using configurations in `toy_models_config.yaml`.

4. **Advanced Experiments**
   ```bash
   python transformer_superposition.py
   python translation_superposition.py
   ```
   Contains more sophisticated experiments and analysis tools.

## Key Features

- Implementation of superposition detection methods
- Interactive toy models for understanding basic concepts
- Visualization tools for analyzing learned representations
- Comparative analysis across different model architectures
- Experiments with various training configurations
- Tools for measuring and quantifying superposition effects

## Visualization and Analysis

Results and visualizations are stored in the `images/` directory. Training logs and metrics can be found in the `runs/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the terms included in the LICENSE file.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{superposition_replication,
  title = {Model Superposition Replication Study},
  year = {2024},
  url = {https://github.com/mmjerge/superposition_replication}
}
```
