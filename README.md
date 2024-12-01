# Model Superposition Replication Study

This repository contains implementations and experiments exploring superposition phenomena in different neural network architectures. Superposition is a phenomenon where neural networks learn to encode multiple features in the same set of weights or neurons, effectively compressing information through overlapping representations.

## Project Structure

```
├── images/           # Visualization outputs and diagrams
├── runs/             # Training logs and experiment results
├── scripts/         # Utility scripts and helpers
├── environment.yaml  # Conda environment specification
├── intro_transformer_superposition.py   # Basic transformer superposition examples
├── intro_translation_superposition.py   # Translation model superposition examples
├── transformer_superposition.py         # Advanced transformer implementations
├── translation_superposition.py         # Advanced translation model experiments
└── LICENSE          # Project license
```

## Getting Started

### Prerequisites

To run the experiments, you'll need Python 3.11 and conda installed. Set up the environment using:

```bash
conda env create -f environment.yaml
conda activate superposition
```

### Running Experiments

The repository includes several example implementations:

1. **Basic Transformer Superposition**
   ```bash
   python intro_transformer_superposition.py
   ```
   Demonstrates basic superposition patterns in transformer architectures.

2. **Translation Model Superposition**
   ```bash
   python intro_translation_superposition.py
   ```
   Shows how superposition manifests in translation tasks.

3. **Advanced Experiments**
   ```bash
   python transformer_superposition.py
   python translation_superposition.py
   ```
   Contains more sophisticated experiments and analysis tools.

## Key Features

- Implementation of superposition detection methods
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
