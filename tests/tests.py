"""Test suite for the superposition replication package.

Tests cover configuration, model instantiation, forward passes,
data generation, analysis utilities, and CLI argument parsing.

Run with: uv run pytest tests/tests.py -v
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for the configuration system."""

    def test_experiment_config_defaults(self):
        from superposition.config import ExperimentConfig

        config = ExperimentConfig()
        assert config.name == "superposition_experiment"
        assert config.model.model_type == "toy"
        assert config.model.num_features == 5
        assert config.model.num_hidden == 2
        assert config.model.num_instances == 10
        assert config.training.batch_size == 1024
        assert config.training.num_steps == 10000
        assert config.training.learning_rate == 1e-3
        assert config.training.scheduler_type == "constant"
        assert config.visualization.viz_interval == 100

    def test_presets_exist(self):
        from superposition.config import PRESETS

        assert "toy_small" in PRESETS
        assert "toy_large" in PRESETS
        assert "transformer_small" in PRESETS
        assert "transformer_large" in PRESETS
        assert "translation" in PRESETS

    def test_preset_model_types(self):
        from superposition.config import PRESETS

        assert PRESETS["toy_small"].model.model_type == "toy"
        assert PRESETS["transformer_small"].model.model_type == "transformer"
        assert PRESETS["translation"].model.model_type == "translation"

    def test_config_to_dict(self):
        from superposition.config import ExperimentConfig

        config = ExperimentConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "model" in d
        assert "training" in d
        assert "visualization" in d
        assert d["model"]["model_type"] == "toy"

    def test_config_from_yaml(self, tmp_path):
        from superposition.config import ExperimentConfig

        yaml_content = """
name: test_experiment
model:
  model_type: transformer
  num_features: 32
  num_hidden: 16
  num_instances: 5
training:
  batch_size: 64
  num_steps: 500
  learning_rate: 0.0001
  scheduler_type: cosine
  seed: 123
visualization:
  viz_interval: 50
  log_dir: test_runs
  save_dir: test_images
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml_content)

        config = ExperimentConfig.from_yaml(str(config_path))
        assert config.name == "test_experiment"
        assert config.model.model_type == "transformer"
        assert config.model.num_features == 32
        assert config.training.batch_size == 64
        assert config.training.scheduler_type == "cosine"
        assert config.visualization.viz_interval == 50

    def test_config_from_legacy_yaml(self, tmp_path):
        from superposition.config import ExperimentConfig

        yaml_content = """
toy_model_config:
  model_name: test_autoencoder
  num_features: 8
  num_hidden: 3
  num_instances: 5
  batch_size: 512
  num_of_steps: 2000
  learning_rate: 0.01
  scheduler_type: linear
"""
        config_path = tmp_path / "legacy_config.yaml"
        config_path.write_text(yaml_content)

        config = ExperimentConfig.from_yaml(str(config_path))
        assert config.model.model_type == "toy"
        assert config.model.num_features == 8
        assert config.model.num_hidden == 3
        assert config.training.batch_size == 512
        assert config.training.num_steps == 2000
        assert config.training.scheduler_type == "linear"

    def test_config_save_yaml(self, tmp_path):
        from superposition.config import ExperimentConfig

        config = ExperimentConfig(name="save_test")
        config.model.num_features = 42
        save_path = str(tmp_path / "saved.yaml")
        config.save_yaml(save_path)

        loaded = ExperimentConfig.from_yaml(save_path)
        assert loaded.name == "save_test"
        assert loaded.model.num_features == 42


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestToyModel:
    """Tests for the ToyModel autoencoder."""

    def test_instantiation(self):
        from superposition.models.toy import ToyModel

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3)
        assert model.num_features == 5
        assert model.num_hidden == 2
        assert model.num_instances == 3

    def test_weight_shape(self):
        from superposition.models.toy import ToyModel

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3)
        assert model.W.shape == (3, 5, 2)
        assert model.b_final.shape == (3, 5)

    def test_forward_shape(self):
        from superposition.models.toy import ToyModel

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3, device=torch.device("cpu"))
        x = torch.randn(8, 3, 5)
        out = model(x)
        assert out.shape == (8, 3, 5)

    def test_forward_relu(self):
        from superposition.models.toy import ToyModel

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3, device=torch.device("cpu"))
        x = torch.randn(8, 3, 5)
        out = model(x)
        assert (out >= 0).all(), "Output should be non-negative (ReLU)"

    def test_generate_data_shape(self):
        from superposition.models.toy import ToyModel

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3)
        data = model.generate_data(16)
        assert data.shape == (16, 3, 5)

    def test_generate_data_sparsity(self):
        from superposition.models.toy import ToyModel

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3)
        data = model.generate_data(1000)
        # Data should have many zeros (sparse)
        zero_fraction = (data == 0).float().mean().item()
        assert zero_fraction > 0.1, "Data should be sparse"

    def test_compute_loss(self):
        from superposition.models.toy import ToyModel

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3, device=torch.device("cpu"))
        batch = model.generate_data(8)
        output = model(batch)
        loss = model.compute_loss(batch, output)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_get_weight_matrix(self):
        from superposition.models.toy import ToyModel

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3)
        W = model.get_weight_matrix()
        assert W.shape == (3, 5, 2)
        assert not W.requires_grad

    def test_feature_probability_buffer(self):
        from superposition.models.toy import ToyModel

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3)
        assert hasattr(model, "feature_probability")
        assert model.feature_probability.shape == (3, 1)

    def test_importance_buffer(self):
        from superposition.models.toy import ToyModel

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3)
        assert hasattr(model, "importance")
        assert model.importance.shape == (1, 5)

    def test_custom_feature_probability(self):
        from superposition.models.toy import ToyModel

        fp = torch.ones(4, 1) * 0.5
        model = ToyModel(num_features=3, num_hidden=2, num_instances=4, feature_probability=fp, device=torch.device("cpu"))
        assert torch.allclose(model.feature_probability, fp)


class TestTransformerModel:
    """Tests for the GPT2-based TransformerModel."""

    def test_instantiation(self):
        from superposition.models.transformer import TransformerModel

        model = TransformerModel(num_features=16, num_hidden=8, num_instances=2, n_layers=2, n_heads=2)
        assert model.num_features == 16
        assert model.num_hidden == 8

    def test_forward_shape(self):
        from superposition.models.transformer import TransformerModel

        model = TransformerModel(num_features=16, num_hidden=8, num_instances=2, n_layers=2, n_heads=2, device=torch.device("cpu"))
        x = torch.randn(4, 2, 16)
        out = model(x)
        assert out.shape == (4, 2, 16)

    def test_forward_relu(self):
        from superposition.models.transformer import TransformerModel

        model = TransformerModel(num_features=16, num_hidden=8, num_instances=2, n_layers=2, n_heads=2, device=torch.device("cpu"))
        x = torch.randn(4, 2, 16)
        out = model(x)
        assert (out >= 0).all()

    def test_generate_data_shape(self):
        from superposition.models.transformer import TransformerModel

        model = TransformerModel(num_features=16, num_hidden=8, num_instances=2, n_layers=2, n_heads=2)
        data = model.generate_data(8)
        assert data.shape == (8, 2, 16)

    def test_get_input_weights(self):
        from superposition.models.transformer import TransformerModel

        model = TransformerModel(num_features=16, num_hidden=8, num_instances=2, n_layers=2, n_heads=2)
        w = model.get_input_weights()
        assert w.shape == (8, 16)
        assert not w.requires_grad

    def test_get_output_weights(self):
        from superposition.models.transformer import TransformerModel

        model = TransformerModel(num_features=16, num_hidden=8, num_instances=2, n_layers=2, n_heads=2)
        w = model.get_output_weights()
        assert w.shape == (16, 8)
        assert not w.requires_grad

    def test_compute_loss(self):
        from superposition.models.transformer import TransformerModel

        model = TransformerModel(num_features=16, num_hidden=8, num_instances=2, n_layers=2, n_heads=2)
        batch = model.generate_data(4)
        output = model(batch)
        loss = model.compute_loss(batch, output)
        assert loss.shape == ()
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Analysis Tests
# ---------------------------------------------------------------------------


class TestInterferenceAnalysis:
    """Tests for the cosine similarity / interference heatmap analysis."""

    def test_cosine_similarity_identity(self):
        from superposition.analysis.interference import compute_cosine_similarity_matrix

        # Orthogonal vectors should have 0 off-diagonal similarity
        W = torch.eye(3, 3)
        sim = compute_cosine_similarity_matrix(W)
        assert sim.shape == (3, 3)
        # Diagonal should be 1
        assert torch.allclose(sim.diag(), torch.ones(3), atol=1e-5)
        # Off-diagonal should be 0 (orthogonal vectors)
        off_diag_mask = ~torch.eye(3, dtype=bool)
        assert torch.allclose(sim[off_diag_mask], torch.zeros(6), atol=1e-5)

    def test_cosine_similarity_parallel(self):
        from superposition.analysis.interference import compute_cosine_similarity_matrix

        # Parallel vectors should have similarity 1
        W = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        sim = compute_cosine_similarity_matrix(W)
        assert torch.allclose(sim, torch.ones(3, 3), atol=1e-5)

    def test_cosine_similarity_antiparallel(self):
        from superposition.analysis.interference import compute_cosine_similarity_matrix

        W = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        sim = compute_cosine_similarity_matrix(W)
        assert torch.isclose(sim[0, 1], torch.tensor(-1.0), atol=1e-5)

    def test_cosine_similarity_orthogonal(self):
        from superposition.analysis.interference import compute_cosine_similarity_matrix

        W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        sim = compute_cosine_similarity_matrix(W)
        assert torch.isclose(sim[0, 1], torch.tensor(0.0), atol=1e-5)

    def test_heatmap_saves_file(self, tmp_path):
        from superposition.analysis.interference import plot_interference_heatmap

        sim_matrix = np.random.randn(5, 5)
        save_path = str(tmp_path / "test_heatmap.png")
        plot_interference_heatmap(sim_matrix, save_path=save_path)

        import os
        assert os.path.exists(save_path)

    def test_compute_interference_heatmap_toy(self, tmp_path):
        from superposition.models.toy import ToyModel
        from superposition.analysis.interference import compute_interference_heatmap

        model = ToyModel(num_features=5, num_hidden=2, num_instances=3)
        save_path = str(tmp_path / "interference.png")
        sim = compute_interference_heatmap(model, save_path=save_path, model_type="toy")
        assert sim.shape == (5, 5)

    def test_compute_interference_heatmap_transformer(self, tmp_path):
        from superposition.models.transformer import TransformerModel
        from superposition.analysis.interference import compute_interference_heatmap

        model = TransformerModel(num_features=16, num_hidden=8, num_instances=2, n_layers=2, n_heads=2)
        save_path = str(tmp_path / "interference_t.png")
        sim = compute_interference_heatmap(model, save_path=save_path, model_type="transformer")
        assert sim.shape == (16, 16)

    def test_per_instance_heatmaps(self, tmp_path):
        from superposition.models.toy import ToyModel
        from superposition.analysis.interference import compute_interference_per_instance

        model = ToyModel(num_features=4, num_hidden=2, num_instances=3)
        results = compute_interference_per_instance(model, save_dir=str(tmp_path))
        assert len(results) == 3
        for i, sim in results.items():
            assert sim.shape == (4, 4)


class TestMaxActivations:
    """Tests for the max-activating examples analysis."""

    def test_find_polysemantic_neurons_diverse(self):
        from superposition.analysis.max_activations import find_polysemantic_neurons

        # All unique tokens -> polysemantic
        results = {
            0: [("cat", 3.0), ("dog", 2.5), ("fish", 2.0), ("bird", 1.5)],
            1: [("the", 3.0), ("the", 2.5), ("the", 2.0), ("the", 1.5)],
        }
        poly = find_polysemantic_neurons(results, diversity_threshold=0.5)
        assert 0 in poly
        assert 1 not in poly

    def test_find_polysemantic_neurons_threshold(self):
        from superposition.analysis.max_activations import find_polysemantic_neurons

        results = {
            0: [("a", 3.0), ("b", 2.5), ("a", 2.0), ("b", 1.5)],
        }
        # 2 unique out of 4 = 0.5
        poly_low = find_polysemantic_neurons(results, diversity_threshold=0.4)
        poly_high = find_polysemantic_neurons(results, diversity_threshold=0.6)
        assert 0 in poly_low
        assert 0 not in poly_high

    def test_format_activation_table(self):
        from superposition.analysis.max_activations import format_activation_table

        results = {
            0: [("hello", 2.5), ("world", 1.8)],
            1: [("foo", 3.1)],
        }
        table = format_activation_table(results)
        assert "hello" in table
        assert "world" in table
        assert "foo" in table
        assert "Neuron" in table

    def test_format_table_specific_neurons(self):
        from superposition.analysis.max_activations import format_activation_table

        results = {
            0: [("a", 1.0)],
            1: [("b", 2.0)],
            2: [("c", 3.0)],
        }
        table = format_activation_table(results, neuron_indices=[1, 2])
        assert "b" in table
        assert "c" in table


# ---------------------------------------------------------------------------
# Utility Tests
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Tests for seeding and device management."""

    def test_set_seed_deterministic(self):
        from superposition.utils.reproducibility import set_seed

        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_get_device_cpu(self):
        from superposition.utils.reproducibility import get_device

        device = get_device(prefer_cuda=False)
        assert device == torch.device("cpu")

    def test_get_device_default(self):
        from superposition.utils.reproducibility import get_device

        device = get_device()
        assert isinstance(device, torch.device)


class TestLogging:
    """Tests for the logging utility."""

    def test_get_logger(self):
        from superposition.utils.logging import get_logger
        import logging

        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_logger_level(self):
        from superposition.utils.logging import get_logger
        import logging

        logger = get_logger("test_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG


class TestDataUtils:
    """Tests for dataset utilities."""

    def test_synthetic_dataset_length(self):
        from superposition.utils.data import SyntheticSuperpositionDataset

        fp = torch.ones(3, 1) * 0.5
        ds = SyntheticSuperpositionDataset(
            num_instances=3, num_features=5, feature_probability=fp, num_samples=100
        )
        assert len(ds) == 100

    def test_synthetic_dataset_item_shape(self):
        from superposition.utils.data import SyntheticSuperpositionDataset

        fp = torch.ones(3, 1) * 0.5
        ds = SyntheticSuperpositionDataset(
            num_instances=3, num_features=5, feature_probability=fp, num_samples=100
        )
        item = ds[0]
        assert item.shape == (3, 5)

    def test_create_dataloaders(self):
        from superposition.utils.data import SyntheticSuperpositionDataset, create_dataloaders

        fp = torch.ones(2, 1) * 0.5
        ds = SyntheticSuperpositionDataset(
            num_instances=2, num_features=4, feature_probability=fp, num_samples=100
        )
        train_loader, val_loader = create_dataloaders(ds, batch_size=10, train_split=0.8)
        assert len(train_loader.dataset) == 80
        assert len(val_loader.dataset) == 20


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for CLI argument parsing."""

    def test_parser_train_defaults(self):
        from superposition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["train", "--model", "toy"])
        assert args.command == "train"
        assert args.model == "toy"
        assert args.seed == 42

    def test_parser_train_with_overrides(self):
        from superposition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train", "--model", "transformer",
            "--num-features", "64",
            "--num-hidden", "32",
            "--batch-size", "128",
            "--lr", "0.001",
            "--scheduler", "cosine",
        ])
        assert args.model == "transformer"
        assert args.num_features == 64
        assert args.num_hidden == 32
        assert args.batch_size == 128
        assert args.lr == 0.001
        assert args.scheduler == "cosine"

    def test_parser_analyze(self):
        from superposition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "analyze", "--analysis", "interference",
            "--model", "toy",
        ])
        assert args.command == "analyze"
        assert args.analysis == "interference"
        assert args.model == "toy"

    def test_parser_analyze_embeddings(self):
        from superposition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "analyze", "--analysis", "embeddings",
            "--model", "translation",
            "--method", "pca",
            "--max-samples", "1000",
        ])
        assert args.analysis == "embeddings"
        assert args.method == "pca"
        assert args.max_samples == 1000

    def test_parser_presets(self):
        from superposition.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["presets"])
        assert args.command == "presets"

    def test_resolve_config_from_preset(self):
        from superposition.cli import build_parser, resolve_config

        parser = build_parser()
        args = parser.parse_args(["train", "--model", "toy", "--preset", "toy_large"])
        config = resolve_config(args)
        assert config.model.num_features == 20
        assert config.model.num_hidden == 5

    def test_resolve_config_cli_overrides(self):
        from superposition.cli import build_parser, resolve_config

        parser = build_parser()
        args = parser.parse_args([
            "train", "--model", "toy",
            "--num-features", "99",
            "--lr", "0.05",
        ])
        config = resolve_config(args)
        assert config.model.num_features == 99
        assert config.training.learning_rate == 0.05


# ---------------------------------------------------------------------------
# Training Tests
# ---------------------------------------------------------------------------


class TestTrainer:
    """Tests for the unified training loop (short runs)."""

    def test_train_toy_model_short(self):
        from superposition.models.toy import ToyModel
        from superposition.training.trainer import Trainer
        from superposition.config import ExperimentConfig, TrainingConfig, VisualizationConfig

        config = ExperimentConfig(
            training=TrainingConfig(batch_size=8, num_steps=5, learning_rate=1e-3),
            visualization=VisualizationConfig(viz_interval=10, use_tensorboard=False),
        )
        model = ToyModel(num_features=4, num_hidden=2, num_instances=2, device=torch.device("cpu"))
        trainer = Trainer(config)
        metrics = trainer.train_superposition_model(model)

        assert "losses" in metrics
        assert len(metrics["losses"]) == 5
        assert metrics["final_loss"] > 0

    def test_train_transformer_model_short(self):
        from superposition.models.transformer import TransformerModel
        from superposition.training.trainer import Trainer
        from superposition.config import ExperimentConfig, TrainingConfig, VisualizationConfig

        config = ExperimentConfig(
            training=TrainingConfig(batch_size=4, num_steps=3, learning_rate=1e-4),
            visualization=VisualizationConfig(viz_interval=10, use_tensorboard=False),
        )
        model = TransformerModel(
            num_features=8, num_hidden=4, num_instances=2, n_layers=1, n_heads=2
        )
        trainer = Trainer(config)
        metrics = trainer.train_superposition_model(model)

        assert len(metrics["losses"]) == 3

    def test_loss_decreases_toy(self):
        from superposition.models.toy import ToyModel
        from superposition.training.trainer import Trainer
        from superposition.config import ExperimentConfig, TrainingConfig, VisualizationConfig

        config = ExperimentConfig(
            training=TrainingConfig(batch_size=64, num_steps=100, learning_rate=1e-2),
            visualization=VisualizationConfig(viz_interval=200, use_tensorboard=False),
        )
        model = ToyModel(num_features=3, num_hidden=2, num_instances=2, device=torch.device("cpu"))
        trainer = Trainer(config)
        metrics = trainer.train_superposition_model(model)

        # Loss should generally decrease over 100 steps
        early_avg = np.mean(metrics["losses"][:10])
        late_avg = np.mean(metrics["losses"][-10:])
        assert late_avg < early_avg, "Loss should decrease during training"
