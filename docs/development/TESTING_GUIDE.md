# Testing Guide

## Table of Contents
- [Introduction](#introduction)
- [Testing Philosophy](#testing-philosophy)
- [Test Structure](#test-structure)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Testing Best Practices](#testing-best-practices)
- [Test Coverage](#test-coverage)
- [Continuous Integration](#continuous-integration)

## Introduction

This guide teaches you how to write effective tests for machine learning projects. Testing is crucial for:
- **Catching bugs early**: Find issues before they reach production
- **Confidence in changes**: Refactor safely knowing tests will catch regressions
- **Documentation**: Tests show how code should be used
- **Design feedback**: Hard-to-test code often indicates design problems

## Testing Philosophy

### Test Pyramid

```
        ┌─────────────┐
        │     E2E     │  ← Few, slow, expensive
        │   Tests     │
        ├─────────────┤
        │ Integration │  ← Some, medium speed
        │   Tests     │
        ├─────────────┤
        │    Unit     │  ← Many, fast, cheap
        │   Tests     │
        └─────────────┘
```

**Unit Tests (70%)**:
- Test individual functions/classes in isolation
- Fast execution (<1ms per test)
- No external dependencies (databases, APIs, file I/O)

**Integration Tests (20%)**:
- Test how components work together
- May use real files, databases
- Slower execution (100ms - 1s per test)

**End-to-End Tests (10%)**:
- Test entire workflows
- Use real data and models
- Slowest execution (minutes)

### What to Test in ML Projects

✅ **Always Test**:
- Data loading and preprocessing
- Model architecture (forward pass, output shapes)
- Metric calculations
- Configuration parsing
- Utility functions

⚠️ **Test Carefully**:
- Training convergence (use small toy datasets)
- Experiment tracking integration (use mocks)

❌ **Don't Test**:
- Third-party library internals (PyTorch, MLflow)
- Model accuracy on real datasets (too slow, non-deterministic)

## Test Structure

### Directory Layout

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_models.py       # Model tests
│   ├── test_data_loader.py  # Data loading tests
│   ├── test_tracking.py     # Tracker tests
│   └── test_utils.py        # Utility tests
├── integration/
│   ├── __init__.py
│   ├── test_training_pipeline.py
│   └── test_data_pipeline.py
├── fixtures/
│   ├── sample_images/       # Test images
│   ├── configs/             # Test configs
│   └── models/              # Saved test models
└── README.md
```

### Naming Conventions

**Test Files**: `test_<module_name>.py`
- `test_models.py` → tests for `src/models/`
- `test_data_loader.py` → tests for `src/data/data_loader.py`

**Test Functions**: `test_<what>_<condition>_<expected>`
- `test_model_forward_pass_correct_shape()`
- `test_dataset_invalid_path_raises_error()`
- `test_tracker_logs_metrics_successfully()`

**Fixture Functions**: `<what>_fixture` or just `<what>`
- `sample_dataset()`, `trained_model()`, `config_fixture()`

## Unit Testing

### Example 1: Testing the Model

```python
# tests/unit/test_models.py
import pytest
import torch
import torch.nn as nn
from src.models.cnn_model import CustomCXRClassifier


class TestCustomCXRClassifier:
    """Unit tests for CustomCXRClassifier."""

    @pytest.fixture
    def model(self):
        """Fixture: Create a model instance."""
        return CustomCXRClassifier(num_classes=3)

    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert isinstance(model, nn.Module)
        assert model.num_classes == 3

    def test_forward_pass_correct_output_shape(self, model):
        """Test forward pass produces correct output shape."""
        # Arrange
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 128, 128)

        # Act
        output = model(input_tensor)

        # Assert
        assert output.shape == (batch_size, 3), \
            f"Expected shape ({batch_size}, 3), got {output.shape}"

    def test_forward_pass_output_is_probability(self, model):
        """Test that output is a valid probability distribution."""
        # Arrange
        input_tensor = torch.randn(1, 3, 128, 128)

        # Act
        output = model(input_tensor)

        # Assert
        assert torch.allclose(output.sum(dim=1), torch.tensor([1.0]), atol=1e-5), \
            "Output probabilities should sum to 1"
        assert (output >= 0).all() and (output <= 1).all(), \
            "Output probabilities should be between 0 and 1"

    def test_model_can_handle_different_batch_sizes(self, model):
        """Test model handles various batch sizes."""
        batch_sizes = [1, 2, 8, 16, 32]

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 128, 128)
            output = model(input_tensor)
            assert output.shape[0] == batch_size, \
                f"Failed for batch size {batch_size}"

    def test_model_in_training_mode(self, model):
        """Test model behavior in training mode."""
        model.train()
        assert model.training, "Model should be in training mode"

        # Dropout should be active
        input_tensor = torch.randn(1, 3, 128, 128)
        output1 = model(input_tensor)
        output2 = model(input_tensor)

        # With dropout, outputs should differ
        assert not torch.allclose(output1, output2, atol=1e-5), \
            "Outputs should differ in training mode due to dropout"

    def test_model_in_eval_mode(self, model):
        """Test model behavior in evaluation mode."""
        model.eval()
        assert not model.training, "Model should be in evaluation mode"

        # Dropout should be inactive
        input_tensor = torch.randn(1, 3, 128, 128)
        output1 = model(input_tensor)
        output2 = model(input_tensor)

        # Without dropout, outputs should be identical
        assert torch.allclose(output1, output2), \
            "Outputs should be identical in eval mode"

    def test_model_parameters_have_gradients(self, model):
        """Test that model parameters require gradients."""
        for name, param in model.named_parameters():
            assert param.requires_grad, \
                f"Parameter {name} should require gradients"

    def test_backward_pass_updates_gradients(self, model):
        """Test that backward pass computes gradients."""
        # Arrange
        model.train()
        input_tensor = torch.randn(2, 3, 128, 128)
        target = torch.tensor([0, 1])
        criterion = nn.CrossEntropyLoss()

        # Act
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()

        # Assert
        for name, param in model.named_parameters():
            assert param.grad is not None, \
                f"Parameter {name} should have gradients after backward()"
            assert not torch.isnan(param.grad).any(), \
                f"Parameter {name} has NaN gradients"

    @pytest.mark.parametrize("num_classes", [2, 3, 5, 10])
    def test_model_with_different_num_classes(self, num_classes):
        """Test model works with various number of output classes."""
        model = CustomCXRClassifier(num_classes=num_classes)
        input_tensor = torch.randn(1, 3, 128, 128)
        output = model(input_tensor)

        assert output.shape[1] == num_classes, \
            f"Output should have {num_classes} classes"

    def test_model_can_be_saved_and_loaded(self, model, tmp_path):
        """Test model serialization and deserialization."""
        # Arrange
        save_path = tmp_path / "model.pth"

        # Act - Save
        torch.save(model.state_dict(), save_path)

        # Act - Load
        loaded_model = CustomCXRClassifier(num_classes=3)
        loaded_model.load_state_dict(torch.load(save_path))

        # Assert
        input_tensor = torch.randn(1, 3, 128, 128)
        model.eval()
        loaded_model.eval()

        output1 = model(input_tensor)
        output2 = loaded_model(input_tensor)

        assert torch.allclose(output1, output2), \
            "Loaded model should produce identical outputs"
```

### Example 2: Testing the Dataset

```python
# tests/unit/test_data_loader.py
import pytest
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.data_loader import COVID19ChestXRayDataset


class TestCOVID19ChestXRayDataset:
    """Unit tests for COVID19ChestXRayDataset."""

    @pytest.fixture
    def sample_dataset_dir(self, tmp_path):
        """Fixture: Create a sample dataset directory."""
        # Create directory structure
        classes = ['COVID', 'Viral Pneumonia', 'Normal']
        for class_name in classes:
            class_dir = tmp_path / class_name
            class_dir.mkdir()

            # Create 3 sample images per class
            for i in range(3):
                img = Image.new('RGB', (128, 128), color=(i*80, i*80, i*80))
                img.save(class_dir / f'image_{i}.png')

        return str(tmp_path)

    def test_dataset_initialization(self, sample_dataset_dir):
        """Test dataset initializes correctly."""
        dataset = COVID19ChestXRayDataset(root_dir=sample_dataset_dir)
        assert len(dataset) == 9, "Should have 9 images (3 per class)"

    def test_dataset_class_mapping(self, sample_dataset_dir):
        """Test correct class label assignment."""
        dataset = COVID19ChestXRayDataset(root_dir=sample_dataset_dir)

        # Check class distribution
        labels = [label for _, label in dataset.images]
        assert labels.count(0) == 3, "Should have 3 COVID images"
        assert labels.count(1) == 3, "Should have 3 Pneumonia images"
        assert labels.count(2) == 3, "Should have 3 Normal images"

    def test_dataset_getitem_returns_correct_types(self, sample_dataset_dir):
        """Test __getitem__ returns correct types."""
        dataset = COVID19ChestXRayDataset(
            root_dir=sample_dataset_dir,
            transform=transforms.ToTensor()
        )

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor), "Image should be a tensor"
        assert isinstance(label, int), "Label should be an integer"

    def test_dataset_transform_applied(self, sample_dataset_dir):
        """Test that transforms are applied correctly."""
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        dataset = COVID19ChestXRayDataset(
            root_dir=sample_dataset_dir,
            transform=transform
        )

        image, _ = dataset[0]
        assert image.shape == (3, 64, 64), "Transform should resize to 64x64"

    def test_dataset_with_invalid_path_raises_error(self):
        """Test that invalid path raises appropriate error."""
        with pytest.raises(ValueError, match="Dataset directory .* does not exist"):
            COVID19ChestXRayDataset(root_dir="/nonexistent/path")

    def test_dataset_handles_corrupted_image(self, tmp_path):
        """Test dataset handles corrupted images gracefully."""
        # Create valid directory structure
        class_dir = tmp_path / "COVID"
        class_dir.mkdir()

        # Create corrupted image file
        corrupted_file = class_dir / "corrupted.png"
        corrupted_file.write_text("This is not an image")

        # Create valid image
        valid_img = Image.new('RGB', (128, 128))
        valid_img.save(class_dir / "valid.png")

        dataset = COVID19ChestXRayDataset(root_dir=str(tmp_path))

        # Should load without crashing
        assert len(dataset) == 2

    def test_dataset_works_with_dataloader(self, sample_dataset_dir):
        """Test dataset integrates with PyTorch DataLoader."""
        dataset = COVID19ChestXRayDataset(
            root_dir=sample_dataset_dir,
            transform=transforms.ToTensor()
        )

        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Test one batch
        images, labels = next(iter(dataloader))

        assert images.shape[0] == 4, "Batch size should be 4"
        assert labels.shape[0] == 4, "Should have 4 labels"
        assert images.shape[1:] == (3, 128, 128), "Image shape should be (3, 128, 128)"

    @pytest.mark.parametrize("split", [0.7, 0.8, 0.9])
    def test_dataset_split_ratios(self, sample_dataset_dir, split):
        """Test dataset splitting with various ratios."""
        from torch.utils.data import random_split

        dataset = COVID19ChestXRayDataset(root_dir=sample_dataset_dir)
        total_size = len(dataset)

        train_size = int(split * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )

        assert len(train_dataset) == train_size
        assert len(val_dataset) == val_size
```

### Example 3: Testing Experiment Trackers

```python
# tests/unit/test_tracking.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.tracking.mlflow_tracker import MLflowTracker
from src.tracking.wandb_tracker import WandBTracker


class TestMLflowTracker:
    """Unit tests for MLflowTracker."""

    @pytest.fixture
    def tracker(self):
        """Fixture: Create MLflowTracker instance."""
        return MLflowTracker()

    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_start_run_calls_mlflow_api(self, mock_mlflow, tracker):
        """Test start_run calls MLflow API correctly."""
        # Act
        tracker.start_run(experiment_name="test_exp", run_name="test_run")

        # Assert
        mock_mlflow.set_experiment.assert_called_once_with("test_exp")
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run")

    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_log_metrics_calls_mlflow_api(self, mock_mlflow, tracker):
        """Test log_metrics calls MLflow API correctly."""
        # Arrange
        metrics = {'loss': 0.5, 'accuracy': 0.85}

        # Act
        tracker.log_metrics(metrics, step=10)

        # Assert
        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=10)

    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_log_params_calls_mlflow_api(self, mock_mlflow, tracker):
        """Test log_params calls MLflow API correctly."""
        # Arrange
        params = {'learning_rate': 0.001, 'batch_size': 32}

        # Act
        tracker.log_params(params)

        # Assert
        mock_mlflow.log_params.assert_called_once_with(params)

    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_end_run_calls_mlflow_api(self, mock_mlflow, tracker):
        """Test end_run calls MLflow API correctly."""
        # Act
        tracker.end_run()

        # Assert
        mock_mlflow.end_run.assert_called_once()


class TestWandBTracker:
    """Unit tests for WandBTracker."""

    @pytest.fixture
    def tracker(self):
        """Fixture: Create WandBTracker instance."""
        return WandBTracker(api_key="test_key")

    @patch('src.tracking.wandb_tracker.wandb')
    def test_start_run_initializes_wandb(self, mock_wandb, tracker):
        """Test start_run initializes W&B correctly."""
        # Act
        tracker.start_run(experiment_name="test_project", run_name="test_run")

        # Assert
        mock_wandb.init.assert_called_once_with(
            project="test_project",
            name="test_run"
        )

    @patch('src.tracking.wandb_tracker.wandb')
    def test_log_metrics_logs_to_wandb(self, mock_wandb, tracker):
        """Test log_metrics logs to W&B correctly."""
        # Arrange
        metrics = {'loss': 0.5, 'accuracy': 0.85}

        # Act
        tracker.log_metrics(metrics, step=10)

        # Assert
        mock_wandb.log.assert_called_once_with(metrics, step=10)
```

## Integration Testing

### Example: Testing Training Pipeline

```python
# tests/integration/test_training_pipeline.py
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.cnn_model import CustomCXRClassifier


class TestTrainingPipeline:
    """Integration tests for training pipeline."""

    @pytest.fixture
    def toy_dataset(self):
        """Fixture: Create small toy dataset for fast testing."""
        # Create random data
        images = torch.randn(20, 3, 128, 128)
        labels = torch.randint(0, 3, (20,))

        dataset = TensorDataset(images, labels)
        train_size = 16
        val_size = 4

        train_dataset = TensorDataset(images[:train_size], labels[:train_size])
        val_dataset = TensorDataset(images[train_size:], labels[train_size:])

        return train_dataset, val_dataset

    @pytest.fixture
    def model(self):
        """Fixture: Create model instance."""
        return CustomCXRClassifier(num_classes=3)

    def test_one_training_epoch_completes(self, model, toy_dataset):
        """Test that one training epoch completes successfully."""
        # Arrange
        train_dataset, _ = toy_dataset
        train_loader = DataLoader(train_dataset, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Act
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Assert
        assert total_loss > 0, "Training should produce non-zero loss"
        assert not torch.isnan(torch.tensor(total_loss)), "Loss should not be NaN"

    def test_model_improves_on_toy_dataset(self, model, toy_dataset):
        """Test that model loss decreases over epochs (sanity check)."""
        # Arrange
        train_dataset, _ = toy_dataset
        train_loader = DataLoader(train_dataset, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Act - Train for 10 epochs
        losses = []
        for epoch in range(10):
            epoch_loss = 0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)

        # Assert - Loss should generally decrease
        # (May not be monotonic, but first should be > last)
        assert losses[0] > losses[-1], \
            "Model should improve (loss should decrease)"

    def test_validation_loop_completes(self, model, toy_dataset):
        """Test that validation loop completes without errors."""
        # Arrange
        _, val_dataset = toy_dataset
        val_loader = DataLoader(val_dataset, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        # Act
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total

        # Assert
        assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
        assert total_loss >= 0, "Loss should be non-negative"
```

## Testing Best Practices

### 1. Arrange-Act-Assert Pattern

Always structure tests with three clear sections:
```python
def test_example():
    # Arrange: Set up test data and dependencies
    model = CustomCXRClassifier(num_classes=3)
    input_tensor = torch.randn(1, 3, 128, 128)

    # Act: Execute the code under test
    output = model(input_tensor)

    # Assert: Verify the results
    assert output.shape == (1, 3)
```

### 2. Test One Thing at a Time

```python
# BAD: Testing multiple things
def test_model():
    model = CustomCXRClassifier()
    assert model.num_classes == 3
    assert model(input).shape == (1, 3)
    assert model.training == False

# GOOD: Separate tests
def test_model_num_classes():
    model = CustomCXRClassifier(num_classes=3)
    assert model.num_classes == 3

def test_model_output_shape():
    model = CustomCXRClassifier()
    output = model(torch.randn(1, 3, 128, 128))
    assert output.shape == (1, 3)

def test_model_default_mode():
    model = CustomCXRClassifier()
    assert model.training == True  # PyTorch default
```

### 3. Use Descriptive Names

```python
# BAD
def test_1(): ...
def test_model(): ...

# GOOD
def test_model_forward_pass_correct_output_shape(): ...
def test_dataset_handles_missing_images_gracefully(): ...
```

### 4. Use Fixtures for Reusable Setup

```python
# tests/conftest.py - Shared across all tests
import pytest
import torch

@pytest.fixture
def sample_input():
    """Reusable sample input tensor."""
    return torch.randn(1, 3, 128, 128)

@pytest.fixture
def sample_model():
    """Reusable model instance."""
    return CustomCXRClassifier(num_classes=3)

# Usage in tests:
def test_something(sample_model, sample_input):
    output = sample_model(sample_input)
    assert output.shape == (1, 3)
```

### 5. Use Parametrize for Multiple Test Cases

```python
@pytest.mark.parametrize("batch_size,expected_shape", [
    (1, (1, 3)),
    (4, (4, 3)),
    (16, (16, 3)),
    (32, (32, 3)),
])
def test_model_handles_various_batch_sizes(batch_size, expected_shape):
    model = CustomCXRClassifier()
    input_tensor = torch.randn(batch_size, 3, 128, 128)
    output = model(input_tensor)
    assert output.shape == expected_shape
```

### 6. Mock External Dependencies

```python
# Don't test external APIs directly
@patch('requests.post')
def test_send_metrics_to_api(mock_post):
    mock_post.return_value.status_code = 200

    # Your code that calls requests.post
    result = send_metrics({'accuracy': 0.9})

    mock_post.assert_called_once()
    assert result.status_code == 200
```

## Test Coverage

### Measuring Coverage

```bash
# Install pytest-cov
pip install pytest-cov

# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

### Coverage Goals

- **Overall**: Aim for 80%+ coverage
- **Critical paths**: 100% coverage for core logic
- **Edge cases**: Cover error handling
- **UI/Scripts**: Lower priority (60-70% OK)

### Example Coverage Report

```
Name                             Stmts   Miss  Cover
----------------------------------------------------
src/__init__.py                      2      0   100%
src/models/__init__.py               1      0   100%
src/models/cnn_model.py             45      3    93%
src/data/__init__.py                 1      0   100%
src/data/data_loader.py             67     12    82%
src/tracking/base_tracker.py        15      0   100%
src/tracking/mlflow_tracker.py      42      8    81%
----------------------------------------------------
TOTAL                              173     23    87%
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Summary

### Testing Checklist for Students

Before submitting your code, ensure:

- [ ] All tests pass locally (`pytest`)
- [ ] Coverage is above 80% (`pytest --cov`)
- [ ] No warnings in test output
- [ ] Tests are well-named and descriptive
- [ ] Fixtures are used for common setup
- [ ] External dependencies are mocked
- [ ] Tests run quickly (<10 seconds for unit tests)
- [ ] CI pipeline passes

### Common Testing Mistakes

1. ❌ Testing implementation details instead of behavior
2. ❌ Tests that depend on execution order
3. ❌ Tests that depend on external resources (network, files)
4. ❌ Slow tests (training on large datasets)
5. ❌ Not testing edge cases and error conditions

### Further Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Machine Learning Code](https://madewithml.com/courses/mlops/testing/)
- [Google Testing Blog](https://testing.googleblog.com/)
- [Test Driven Development by Kent Beck](https://www.oreilly.com/library/view/test-driven-development/0321146530/)
