"""
Pytest configuration and shared fixtures for all tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil


@pytest.fixture
def device():
    """Fixture: Get computing device (CPU for tests)."""
    return torch.device('cpu')


@pytest.fixture
def sample_config():
    """Fixture: Sample configuration dictionary."""
    return {
        'model': {
            'architecture': 'CustomCNN',
            'num_classes': 3,
            'in_channels': 3
        },
        'training': {
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        },
        'data': {
            'image_size': 128,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        }
    }


@pytest.fixture
def sample_image():
    """Fixture: Generate a sample image tensor."""
    return torch.randn(3, 128, 128)


@pytest.fixture
def sample_batch():
    """Fixture: Generate a sample batch of images and labels."""
    images = torch.randn(4, 3, 128, 128)
    labels = torch.randint(0, 3, (4,))
    return images, labels


@pytest.fixture
def temp_dataset_dir():
    """
    Fixture: Create a temporary dataset directory structure.

    Creates a directory with sample images for testing:
    temp_dir/
        COVID/
            image_0.png
            image_1.png
        Viral Pneumonia/
            image_0.png
            image_1.png
        Normal/
            image_0.png
            image_1.png
    """
    temp_dir = tempfile.mkdtemp()

    # Create class directories
    classes = ['COVID', 'Viral Pneumonia', 'Normal']
    for class_name in classes:
        class_dir = Path(temp_dir) / class_name
        class_dir.mkdir()

        # Create 2 sample images per class
        for i in range(2):
            img = Image.new('RGB', (128, 128), color=(i*100, i*100, i*100))
            img.save(class_dir / f'image_{i}.png')

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_config_file():
    """Fixture: Create a temporary configuration file."""
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        delete=False
    )

    config_content = """
model:
  architecture: CustomCNN
  num_classes: 3

training:
  epochs: 2
  batch_size: 4
  learning_rate: 0.001

data:
  image_size: 128
"""

    temp_file.write(config_content)
    temp_file.close()

    yield temp_file.name

    # Cleanup
    Path(temp_file.name).unlink()


@pytest.fixture
def sample_model():
    """Fixture: Create a sample model instance."""
    from src.models.cnn_model import CustomCXRClassifier
    return CustomCXRClassifier(num_classes=3)


@pytest.fixture
def sample_predictions():
    """Fixture: Generate sample predictions and ground truth."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 2, 0, 2, 1])
    return y_true, y_pred


@pytest.fixture
def class_names():
    """Fixture: List of class names."""
    return ['COVID', 'Viral Pneumonia', 'Normal']


@pytest.fixture(autouse=True)
def set_random_seed():
    """Fixture: Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def mock_tracker():
    """Fixture: Create a mock experiment tracker."""
    class MockTracker:
        def __init__(self):
            self.metrics = {}
            self.params = {}
            self.artifacts = []
            self._is_running = False

        def start_run(self, run_name=None, tags=None, config=None):
            self._is_running = True

        def log_metrics(self, metrics, step=None):
            for key, value in metrics.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)

        def log_params(self, params):
            self.params.update(params)

        def log_artifact(self, artifact_path, artifact_type=None):
            self.artifacts.append(artifact_path)

        def log_model(self, model, artifact_path="model"):
            self.artifacts.append(f"model:{artifact_path}")

        def end_run(self):
            self._is_running = False

        def set_tag(self, key, value):
            pass

    return MockTracker()


# Markers for categorizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
