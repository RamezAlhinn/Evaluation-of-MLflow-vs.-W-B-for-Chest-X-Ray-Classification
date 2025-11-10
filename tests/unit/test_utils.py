"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from src.utils.metrics import (
    calculate_accuracy,
    calculate_metrics,
    MetricsTracker,
    AverageMeter
)
from src.utils.config import Config, load_config, validate_config
from src.utils.logging_utils import setup_logger, TrainingLogger


@pytest.mark.unit
class TestMetrics:
    """Test suite for metrics utilities."""

    def test_calculate_accuracy(self, sample_predictions):
        """Test accuracy calculation."""
        y_true, y_pred = sample_predictions
        accuracy = calculate_accuracy(y_true, y_pred)

        assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
        assert isinstance(accuracy, float), "Accuracy should be a float"

    def test_calculate_metrics(self, sample_predictions, class_names):
        """Test comprehensive metrics calculation."""
        y_true, y_pred = sample_predictions
        metrics = calculate_metrics(y_true, y_pred, class_names)

        # Check required keys
        required_keys = ['accuracy', 'precision', 'recall', 'f1']
        for key in required_keys:
            assert key in metrics, f"Metrics should contain '{key}'"

        # Check per-class metrics
        for class_name in class_names:
            assert f'precision_{class_name}' in metrics
            assert f'recall_{class_name}' in metrics
            assert f'f1_{class_name}' in metrics

    def test_metrics_tracker(self):
        """Test MetricsTracker functionality."""
        tracker = MetricsTracker()

        # Add metrics for multiple steps
        for i in range(5):
            tracker.add_metrics({
                'loss': 0.5 - i * 0.1,
                'accuracy': 0.7 + i * 0.05
            }, step=i)

        # Test get_metric
        losses = tracker.get_metric('loss')
        assert len(losses) == 5, "Should track 5 loss values"

        # Test get_best
        best_step, best_acc = tracker.get_best('accuracy', mode='max')
        assert best_step == 4, "Best accuracy should be at step 4"
        assert best_acc == 0.9, "Best accuracy should be 0.9"

        # Test get_latest
        latest_loss = tracker.get_latest('loss')
        assert latest_loss == 0.1, "Latest loss should be 0.1"

        # Test get_average
        avg_acc = tracker.get_average('accuracy')
        assert 0.7 <= avg_acc <= 0.9, "Average accuracy should be in range"

    def test_average_meter(self):
        """Test AverageMeter functionality."""
        meter = AverageMeter('loss')

        # Update with values
        values = [0.5, 0.4, 0.3, 0.35]
        for val in values:
            meter.update(val)

        expected_avg = sum(values) / len(values)
        assert abs(meter.avg - expected_avg) < 1e-6, \
            "Average should match expected value"
        assert meter.val == 0.35, "Val should be last updated value"


@pytest.mark.unit
class TestConfig:
    """Test suite for configuration utilities."""

    def test_config_initialization(self, sample_config):
        """Test Config initialization."""
        config = Config(sample_config)

        assert config.model.architecture == 'CustomCNN'
        assert config.training.batch_size == 4

    def test_config_get_with_default(self, sample_config):
        """Test Config.get() with default values."""
        config = Config(sample_config)

        # Existing key
        lr = config.get('training.learning_rate', 0.01)
        assert lr == 0.001

        # Non-existing key
        dropout = config.get('model.dropout', 0.5)
        assert dropout == 0.5

    def test_config_set(self, sample_config):
        """Test Config.set() functionality."""
        config = Config(sample_config)

        config.set('model.dropout', 0.5)
        assert config.get('model.dropout') == 0.5

    def test_config_update(self, sample_config):
        """Test Config.update() functionality."""
        config = Config(sample_config)

        updates = {'training': {'epochs': 10}}
        config.update(updates)

        assert config.training.epochs == 10

    def test_config_to_dict(self, sample_config):
        """Test Config.to_dict() conversion."""
        config = Config(sample_config)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['model']['architecture'] == 'CustomCNN'

    def test_load_config(self, temp_config_file):
        """Test loading configuration from file."""
        config = load_config(temp_config_file)

        assert isinstance(config, Config)
        assert config.model.architecture == 'CustomCNN'

    def test_validate_config(self, sample_config):
        """Test configuration validation."""
        config = Config(sample_config)

        required_keys = ['model.architecture', 'training.epochs']
        schema = {'training.epochs': int, 'training.batch_size': int}

        # Should pass validation
        assert validate_config(config, required_keys, schema)

    def test_validate_config_missing_key(self, sample_config):
        """Test validation fails with missing key."""
        config = Config(sample_config)

        required_keys = ['model.nonexistent']

        with pytest.raises(ValueError, match="Missing required config key"):
            validate_config(config, required_keys)

    def test_validate_config_wrong_type(self, sample_config):
        """Test validation fails with wrong type."""
        config = Config(sample_config)

        schema = {'training.epochs': str}  # epochs is int, not str

        with pytest.raises(ValueError, match="wrong type"):
            validate_config(config, [], schema)


@pytest.mark.unit
class TestLogging:
    """Test suite for logging utilities."""

    def test_setup_logger(self, tmp_path):
        """Test logger setup."""
        log_file = tmp_path / "test.log"
        logger = setup_logger('test', log_file=str(log_file))

        assert logger is not None
        logger.info("Test message")

        # Check log file was created
        assert log_file.exists()

    def test_training_logger(self, tmp_path):
        """Test TrainingLogger functionality."""
        log_file = tmp_path / "training.log"
        base_logger = setup_logger('training', log_file=str(log_file))
        train_logger = TrainingLogger(base_logger)

        # Test logging methods don't raise errors
        train_logger.log_epoch_start(0, 10)
        train_logger.log_epoch_end(0, 0.5, 85.0, 0.4, 87.0, 0.001)
        train_logger.log_best_model(0, "val_accuracy", 87.0)
        train_logger.log_training_complete(10, 90.0, 3600.0)

        # Check log file has content
        assert log_file.stat().st_size > 0
