# API Reference

## Table of Contents
- [Models](#models)
- [Data](#data)
- [Tracking](#tracking)
- [Utilities](#utilities)
- [Training Scripts](#training-scripts)

## Models

### CustomCXRClassifier

**Location**: `src/models/cnn_model.py`

Custom CNN for chest X-ray classification.

#### Class Definition

```python
class CustomCXRClassifier(nn.Module):
    """
    Custom CNN for COVID-19 chest X-ray classification.

    Architecture:
        - 4 convolutional blocks with progressive channel increase
        - MaxPooling after each conv block
        - Dropout for regularization
        - 2 fully connected layers
        - Softmax output

    Attributes:
        num_classes (int): Number of output classes
        fc_input_features (int): Input features for first FC layer
    """
```

#### Constructor

```python
def __init__(self, num_classes: int = 3)
```

**Parameters**:
- `num_classes` (int, optional): Number of output classes. Default: 3

**Example**:
```python
from src.models.cnn_model import CustomCXRClassifier

# Create model for 3-class classification
model = CustomCXRClassifier(num_classes=3)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

#### forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through the network.

**Parameters**:
- `x` (torch.Tensor): Input tensor of shape `(batch_size, 3, 128, 128)`

**Returns**:
- `torch.Tensor`: Output predictions of shape `(batch_size, num_classes)`

**Example**:
```python
model = CustomCXRClassifier()
input_tensor = torch.randn(4, 3, 128, 128)  # Batch of 4 images
output = model(input_tensor)
print(output.shape)  # torch.Size([4, 3])
```

---

## Data

### COVID19ChestXRayDataset

**Location**: `src/data/data_loader.py`

PyTorch Dataset for COVID-19 chest X-ray images.

#### Class Definition

```python
class COVID19ChestXRayDataset(Dataset):
    """
    Custom Dataset for COVID-19 Chest X-Ray images.

    Supports multiple folder structure formats and applies
    configurable transforms.

    Attributes:
        root_dir (str): Root directory of dataset
        transform (callable, optional): Transform to apply to images
        images (List[Tuple[str, int]]): List of (image_path, label) tuples
        class_names (List[str]): Names of classes
    """
```

#### Constructor

```python
def __init__(
    self,
    root_dir: str,
    transform: Optional[Callable] = None
)
```

**Parameters**:
- `root_dir` (str): Path to dataset root directory
- `transform` (callable, optional): Optional transform to be applied on images

**Raises**:
- `ValueError`: If dataset directory doesn't exist or has invalid structure

**Example**:
```python
from src.data.data_loader import COVID19ChestXRayDataset
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

# Create dataset
dataset = COVID19ChestXRayDataset(
    root_dir='data/covid19_xray',
    transform=transform
)

print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.class_names}")
```

#### \_\_getitem\_\_

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]
```

Get a single sample from the dataset.

**Parameters**:
- `idx` (int): Index of sample to retrieve

**Returns**:
- `Tuple[torch.Tensor, int]`: Tuple of (image, label)

**Example**:
```python
dataset = COVID19ChestXRayDataset(root_dir='data/covid19_xray')
image, label = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label: {label} ({dataset.class_names[label]})")
```

#### \_\_len\_\_

```python
def __len__(self) -> int
```

Get dataset size.

**Returns**:
- `int`: Number of samples in dataset

---

### get_data_loaders

**Location**: `src/data/data_loader.py`

Create train, validation, and test data loaders.

```python
def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]
```

**Parameters**:
- `data_dir` (str): Path to dataset directory
- `batch_size` (int): Batch size for data loaders
- `num_workers` (int): Number of worker processes for data loading
- `train_ratio` (float): Proportion of data for training
- `val_ratio` (float): Proportion of data for validation
- `test_ratio` (float): Proportion of data for testing
- `seed` (int): Random seed for reproducibility

**Returns**:
- `Tuple[DataLoader, DataLoader, DataLoader]`: Train, validation, and test loaders

**Example**:
```python
from src.data.data_loader import get_data_loaders

train_loader, val_loader, test_loader = get_data_loaders(
    data_dir='data/covid19_xray',
    batch_size=32,
    num_workers=4,
    seed=42
)

# Iterate through training data
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    break
```

---

## Tracking

### BaseTracker (Abstract)

**Location**: `src/tracking/base_tracker.py`

Abstract base class for experiment trackers.

#### Class Definition

```python
class BaseTracker(ABC):
    """
    Abstract base class for experiment tracking.

    Defines the interface that all tracking implementations must follow.
    Use this for creating custom tracking backends.
    """
```

#### start_run

```python
@abstractmethod
def start_run(
    self,
    experiment_name: str,
    run_name: str,
    config: Optional[Dict[str, Any]] = None
) -> None
```

Initialize a new tracking run.

**Parameters**:
- `experiment_name` (str): Name of the experiment/project
- `run_name` (str): Name of this specific run
- `config` (dict, optional): Configuration parameters to log

#### log_metrics

```python
@abstractmethod
def log_metrics(
    self,
    metrics: Dict[str, float],
    step: int
) -> None
```

Log metrics at a specific step.

**Parameters**:
- `metrics` (dict): Dictionary of metric names and values
- `step` (int): Step/epoch number

#### log_params

```python
@abstractmethod
def log_params(self, params: Dict[str, Any]) -> None
```

Log hyperparameters.

**Parameters**:
- `params` (dict): Dictionary of parameter names and values

#### log_artifact

```python
@abstractmethod
def log_artifact(self, artifact_path: str) -> None
```

Log a file artifact.

**Parameters**:
- `artifact_path` (str): Path to file to log

#### end_run

```python
@abstractmethod
def end_run(self) -> None
```

Finalize and end the tracking run.

---

### MLflowTracker

**Location**: `src/tracking/mlflow_tracker.py`

MLflow implementation of BaseTracker.

#### Class Definition

```python
class MLflowTracker(BaseTracker):
    """
    MLflow experiment tracker.

    Implements experiment tracking using MLflow.
    """
```

#### Constructor

```python
def __init__(self, tracking_uri: Optional[str] = None)
```

**Parameters**:
- `tracking_uri` (str, optional): MLflow tracking server URI

**Example**:
```python
from src.tracking.mlflow_tracker import MLflowTracker

# Use default local tracking
tracker = MLflowTracker()

# Or specify remote tracking server
tracker = MLflowTracker(tracking_uri='http://localhost:5000')

# Start tracking
tracker.start_run(
    experiment_name='covid_xray_classification',
    run_name='experiment_001',
    config={'learning_rate': 0.001, 'batch_size': 32}
)

# Log metrics
for epoch in range(10):
    metrics = {'loss': 0.5, 'accuracy': 0.85}
    tracker.log_metrics(metrics, step=epoch)

# End tracking
tracker.end_run()
```

---

### WandBTracker

**Location**: `src/tracking/wandb_tracker.py`

Weights & Biases implementation of BaseTracker.

#### Class Definition

```python
class WandBTracker(BaseTracker):
    """
    Weights & Biases experiment tracker.

    Implements experiment tracking using W&B.
    """
```

#### Constructor

```python
def __init__(self, api_key: Optional[str] = None)
```

**Parameters**:
- `api_key` (str, optional): W&B API key (reads from environment if not provided)

**Example**:
```python
from src.tracking.wandb_tracker import WandBTracker
import os

# Read API key from environment
tracker = WandBTracker(api_key=os.environ.get('WANDB_API_KEY'))

# Start tracking
tracker.start_run(
    experiment_name='covid-xray',
    run_name='experiment_001',
    config={'learning_rate': 0.001, 'batch_size': 32}
)

# Log metrics
for epoch in range(10):
    metrics = {'loss': 0.5, 'accuracy': 0.85}
    tracker.log_metrics(metrics, step=epoch)

# End tracking
tracker.end_run()
```

---

## Utilities

### Metrics

**Location**: `src/utils/metrics.py`

#### calculate_metrics

```python
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, float]
```

Calculate classification metrics.

**Parameters**:
- `y_true` (np.ndarray): Ground truth labels
- `y_pred` (np.ndarray): Predicted labels
- `class_names` (List[str]): List of class names

**Returns**:
- `Dict[str, float]`: Dictionary containing:
  - `accuracy`: Overall accuracy
  - `precision_macro`: Macro-averaged precision
  - `recall_macro`: Macro-averaged recall
  - `f1_macro`: Macro-averaged F1 score
  - `precision_<class>`: Per-class precision
  - `recall_<class>`: Per-class recall
  - `f1_<class>`: Per-class F1 score

**Example**:
```python
from src.utils.metrics import calculate_metrics
import numpy as np

y_true = np.array([0, 1, 2, 0, 1])
y_pred = np.array([0, 1, 2, 1, 1])
class_names = ['COVID', 'Pneumonia', 'Normal']

metrics = calculate_metrics(y_true, y_pred, class_names)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_macro']:.4f}")
print(f"COVID Precision: {metrics['precision_COVID']:.4f}")
```

---

### Configuration

**Location**: `src/utils/config.py`

#### load_config

```python
def load_config(config_path: str) -> Dict[str, Any]
```

Load configuration from YAML file.

**Parameters**:
- `config_path` (str): Path to YAML configuration file

**Returns**:
- `Dict[str, Any]`: Configuration dictionary

**Raises**:
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example**:
```python
from src.utils.config import load_config

config = load_config('configs/mlflow/default_config.yaml')

print(f"Learning rate: {config['training']['learning_rate']}")
print(f"Batch size: {config['training']['batch_size']}")
print(f"Epochs: {config['training']['epochs']}")
```

---

### Logging

**Location**: `src/utils/logging.py`

#### setup_logger

```python
def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger
```

Set up a logger with console and optional file handlers.

**Parameters**:
- `name` (str): Logger name
- `log_file` (str, optional): Path to log file
- `level` (int): Logging level (e.g., logging.INFO, logging.DEBUG)

**Returns**:
- `logging.Logger`: Configured logger instance

**Example**:
```python
from src.utils.logging import setup_logger
import logging

# Console logging only
logger = setup_logger('training', level=logging.INFO)
logger.info('Training started')

# Console and file logging
logger = setup_logger(
    'training',
    log_file='logs/training.log',
    level=logging.DEBUG
)
logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
```

---

## Training Scripts

### train_mlflow.py

**Location**: `scripts/train_mlflow.py`

Main training script with MLflow tracking.

#### Usage

```bash
# Basic usage
python scripts/train_mlflow.py

# With custom config
python scripts/train_mlflow.py --config configs/mlflow/custom_config.yaml

# With command-line overrides
python scripts/train_mlflow.py --epochs 50 --batch-size 64 --lr 0.0001
```

#### Command-Line Arguments

- `--config`: Path to configuration file (default: `configs/mlflow/default_config.yaml`)
- `--data-dir`: Path to dataset directory (default: `data/covid19_xray`)
- `--epochs`: Number of training epochs (default: from config)
- `--batch-size`: Batch size (default: from config)
- `--lr`: Learning rate (default: from config)
- `--device`: Device to use ('cuda' or 'cpu', default: auto-detect)
- `--seed`: Random seed (default: 42)

---

### train_wandb.py

**Location**: `scripts/train_wandb.py`

Main training script with W&B tracking.

#### Usage

```bash
# Basic usage
python scripts/train_wandb.py

# With custom config
python scripts/train_wandb.py --config configs/wandb/default_config.yaml

# With custom project and run names
python scripts/train_wandb.py --project my-project --run-name experiment-001
```

#### Command-Line Arguments

Similar to `train_mlflow.py`, with additional:
- `--project`: W&B project name
- `--run-name`: W&B run name
- `--api-key`: W&B API key (or set `WANDB_API_KEY` environment variable)

---

### compare_mlflow_wandb.py

**Location**: `scripts/compare_mlflow_wandb.py`

Script to compare MLflow and W&B side-by-side.

#### Usage

```bash
# Run comparison with both trackers
python scripts/compare_mlflow_wandb.py --config configs/comparison_config.yaml

# Compare with specific settings
python scripts/compare_mlflow_wandb.py \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001
```

---

## Type Definitions

### Common Types

```python
from typing import Dict, List, Tuple, Optional, Any, Callable
import torch
from torch.utils.data import DataLoader
import numpy as np

# Configuration dictionary
Config = Dict[str, Any]

# Metrics dictionary
Metrics = Dict[str, float]

# Image and label batch
Batch = Tuple[torch.Tensor, torch.Tensor]

# Dataset split
DatasetSplit = Tuple[DataLoader, DataLoader, DataLoader]
```

---

## Error Handling

All API functions follow consistent error handling:

```python
try:
    result = api_function(params)
except FileNotFoundError as e:
    # Handle missing files
    logger.error(f"File not found: {e}")
except ValueError as e:
    # Handle invalid parameters
    logger.error(f"Invalid value: {e}")
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
    raise
```

---

## Examples

### Complete Training Example

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.models.cnn_model import CustomCXRClassifier
from src.data.data_loader import get_data_loaders
from src.tracking.mlflow_tracker import MLflowTracker
from src.utils.config import load_config
from src.utils.logging import setup_logger

# Setup
logger = setup_logger('training')
config = load_config('configs/mlflow/default_config.yaml')

# Data
train_loader, val_loader, test_loader = get_data_loaders(
    data_dir='data/covid19_xray',
    batch_size=config['training']['batch_size']
)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomCXRClassifier(num_classes=3).to(device)

# Training components
optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Tracking
tracker = MLflowTracker()
tracker.start_run(
    experiment_name='covid_xray_classification',
    run_name='experiment_001',
    config=config
)

# Training loop
for epoch in range(config['training']['epochs']):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Log metrics
    metrics = {
        'train_loss': train_loss / len(train_loader),
        'val_loss': val_loss / len(val_loader),
        'val_accuracy': correct / total
    }
    tracker.log_metrics(metrics, step=epoch)

    logger.info(f"Epoch {epoch}: {metrics}")

    scheduler.step()

# Save model
torch.save(model.state_dict(), 'best_model.pth')
tracker.log_artifact('best_model.pth')
tracker.end_run()
```

---

## Further Reading

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [W&B Documentation](https://docs.wandb.ai/)
- [Project README](../README.md)
- [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md)
