# Development Best Practices

## Table of Contents
- [Code Style](#code-style)
- [Python Best Practices](#python-best-practices)
- [Machine Learning Best Practices](#machine-learning-best-practices)
- [Git Workflow](#git-workflow)
- [Documentation Standards](#documentation-standards)
- [Performance Optimization](#performance-optimization)
- [Security Considerations](#security-considerations)

## Code Style

### PEP 8 Compliance

Follow [PEP 8](https://pep8.org/) - Python's official style guide:

```python
# BAD
def myFunction(x,y):
    return x+y

# GOOD
def my_function(x, y):
    """Add two numbers together."""
    return x + y
```

### Line Length

- **Maximum 88 characters** (Black formatter default)
- Break long lines logically:

```python
# BAD
result = some_function(argument1, argument2, argument3, argument4, argument5, argument6)

# GOOD
result = some_function(
    argument1,
    argument2,
    argument3,
    argument4,
    argument5,
    argument6
)
```

### Imports

Order imports following the standard:

```python
# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# 2. Third-party imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 3. Local application imports
from src.models.cnn_model import CustomCXRClassifier
from src.data.data_loader import COVID19ChestXRayDataset
from src.utils.metrics import calculate_metrics
```

Use `isort` to automate:
```bash
pip install isort
isort src/
```

### Naming Conventions

```python
# Constants: UPPER_CASE
MAX_EPOCHS = 100
LEARNING_RATE = 0.001

# Classes: PascalCase
class CustomCXRClassifier:
    pass

# Functions and variables: snake_case
def train_model(config):
    learning_rate = config['lr']
    batch_size = config['batch_size']

# Private methods: leading underscore
def _internal_helper():
    pass

# Protected variables: leading underscore
self._hidden_state = None
```

## Python Best Practices

### Type Hints

Always use type hints for function signatures:

```python
# BAD
def train_model(model, data_loader, epochs):
    ...

# GOOD
from typing import Dict, Any
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Train a model for specified number of epochs.

    Args:
        model: PyTorch model to train
        data_loader: DataLoader with training data
        epochs: Number of training epochs
        config: Configuration dictionary

    Returns:
        Dictionary containing training metrics
    """
    ...
```

### Error Handling

Always handle errors gracefully:

```python
# BAD
def load_image(path):
    return Image.open(path)

# GOOD
def load_image(path: str) -> Optional[Image.Image]:
    """
    Load an image from file path.

    Args:
        path: Path to image file

    Returns:
        PIL Image if successful, None otherwise

    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        return Image.open(path)
    except Exception as e:
        logger.error(f"Failed to load image {path}: {e}")
        return None
```

### Context Managers

Use context managers for resource management:

```python
# BAD
f = open('config.yaml')
config = yaml.safe_load(f)
f.close()

# GOOD
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# BAD
torch.set_grad_enabled(False)
model.eval()
output = model(input)
torch.set_grad_enabled(True)

# GOOD
with torch.no_grad():
    model.eval()
    output = model(input)
```

### List Comprehensions

Use list comprehensions for simple transformations:

```python
# BAD
squares = []
for x in range(10):
    squares.append(x**2)

# GOOD
squares = [x**2 for x in range(10)]

# BAD (too complex for comprehension)
results = [process(x) if validate(x) else default(x) if x > 0 else fallback(x) for x in data]

# GOOD (break down complex logic)
results = []
for x in data:
    if validate(x):
        results.append(process(x))
    elif x > 0:
        results.append(default(x))
    else:
        results.append(fallback(x))
```

### F-strings for Formatting

Use f-strings (Python 3.6+):

```python
# BAD
print("Epoch: " + str(epoch) + ", Loss: " + str(loss))
print("Epoch: {}, Loss: {}".format(epoch, loss))

# GOOD
print(f"Epoch: {epoch}, Loss: {loss:.4f}")

# Complex formatting
metrics = {'accuracy': 0.8543, 'loss': 0.3241}
print(f"Accuracy: {metrics['accuracy']:.2%}, Loss: {metrics['loss']:.4f}")
# Output: Accuracy: 85.43%, Loss: 0.3241
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Computes accuracy, precision, recall, and F1 score for multi-class
    classification tasks.

    Args:
        y_true: Ground truth labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        class_names: List of class names for per-class metrics

    Returns:
        Dictionary with metric names as keys and values as floats:
            - 'accuracy': Overall accuracy
            - 'precision': Macro-averaged precision
            - 'recall': Macro-averaged recall
            - 'f1': Macro-averaged F1 score

    Raises:
        ValueError: If y_true and y_pred have different lengths

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1])
        >>> y_pred = np.array([0, 1, 2, 1, 1])
        >>> metrics = calculate_metrics(y_true, y_pred, ['A', 'B', 'C'])
        >>> print(metrics['accuracy'])
        0.8
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    accuracy = (y_true == y_pred).mean()
    # ... more calculations

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1_score
    }
```

## Machine Learning Best Practices

### Reproducibility

Always set random seeds:

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call at start of training
set_seed(42)
```

### Data Validation

Always validate input data:

```python
def validate_dataset(dataset: Dataset) -> bool:
    """
    Validate dataset integrity.

    Args:
        dataset: PyTorch Dataset to validate

    Returns:
        True if valid, False otherwise
    """
    if len(dataset) == 0:
        logger.error("Dataset is empty")
        return False

    # Sample first item
    try:
        image, label = dataset[0]
    except Exception as e:
        logger.error(f"Failed to load first item: {e}")
        return False

    # Check shapes
    if image.ndim != 3:
        logger.error(f"Expected 3D image, got {image.ndim}D")
        return False

    # Check label validity
    if not isinstance(label, int):
        logger.error(f"Expected integer label, got {type(label)}")
        return False

    logger.info(f"Dataset validation passed: {len(dataset)} samples")
    return True
```

### Gradient Monitoring

Monitor gradients during training:

```python
def check_gradients(model: nn.Module) -> Dict[str, float]:
    """
    Check model gradients for issues.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with gradient statistics
    """
    total_norm = 0.0
    max_grad = 0.0
    num_zero_grads = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_grad = max(max_grad, param_norm)

            if param_norm == 0:
                num_zero_grads += 1
                logger.warning(f"Zero gradient in {name}")

            if torch.isnan(param.grad).any():
                logger.error(f"NaN gradient in {name}")

    total_norm = total_norm ** 0.5

    return {
        'total_norm': total_norm,
        'max_grad': max_grad,
        'num_zero_grads': num_zero_grads
    }

# Use during training
loss.backward()
grad_stats = check_gradients(model)
if grad_stats['total_norm'] > 10.0:
    logger.warning(f"Large gradient norm: {grad_stats['total_norm']}")
optimizer.step()
```

### Model Checkpointing

Save comprehensive checkpoints:

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str
):
    """
    Save model checkpoint with metadata.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Training metrics
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str
) -> int:
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        path: Checkpoint path

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    return checkpoint['epoch']
```

### Train/Val/Test Separation

Properly separate data:

```python
from torch.utils.data import random_split

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: Full dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1"

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    logger.info(f"Dataset split: {train_size} train, "
                f"{val_size} val, {test_size} test")

    return train_dataset, val_dataset, test_dataset
```

## Git Workflow

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(model): add ResNet-18 architecture option

Implements ResNet-18 as an alternative to CustomCNN.
Includes configuration parameters in config.yaml.

Closes #42

---

fix(data): handle corrupted images gracefully

Previous implementation crashed on corrupted files.
Now returns black image and logs warning.

---

docs(readme): update installation instructions

Add CUDA installation steps for GPU support.
```

### Branch Naming

```
<type>/<short-description>

Examples:
- feature/add-resnet-model
- fix/data-loader-memory-leak
- docs/update-api-reference
- refactor/tracker-base-class
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Coverage increased/maintained

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

## Documentation Standards

### README Structure

Every module should have a README:

```markdown
# Module Name

Brief description of module purpose.

## Features
- Feature 1
- Feature 2

## Usage
```python
from module import Class

obj = Class()
obj.method()
```

## API Reference
See [API.md](API.md)

## Examples
See [examples/](examples/)
```

### Code Comments

```python
# GOOD: Comments explain WHY, not WHAT
# Use ImageNet normalization for transfer learning compatibility
transform = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])

# BAD: Comments just repeat code
# Normalize the image
transform = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])

# GOOD: Comments for complex logic
# Learning rate warm-up: linearly increase lr for first 5 epochs
# to prevent early divergence with large batch sizes
if epoch < 5:
    lr_scale = (epoch + 1) / 5.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * lr_scale

# GOOD: TODO comments
# TODO(username): Implement early stopping callback
# TODO(username): Add support for multi-GPU training
```

## Performance Optimization

### PyTorch-Specific

```python
# 1. Pin memory for faster GPU transfer
train_loader = DataLoader(dataset, pin_memory=True)

# 2. Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 3. Avoid CPU-GPU synchronization in loops
# BAD: .item() synchronizes CPU and GPU
for i, (images, labels) in enumerate(train_loader):
    loss = criterion(model(images), labels)
    losses.append(loss.item())  # Synchronization!

# GOOD: Accumulate losses, compute after loop
losses = []
for i, (images, labels) in enumerate(train_loader):
    loss = criterion(model(images), labels)
    losses.append(loss.detach())

losses_array = torch.stack(losses).cpu().numpy()

# 4. Use DataLoader num_workers
# Adjust based on your system
train_loader = DataLoader(dataset, num_workers=4)
```

### Memory Management

```python
# 1. Delete large tensors when done
def process_batch(batch):
    result = expensive_operation(batch)
    del batch  # Free memory immediately
    return result

# 2. Use in-place operations
# BAD
x = x + 1

# GOOD
x += 1  # In-place addition

# 3. Clear cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 4. Gradient accumulation for large models
accumulation_steps = 4

for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Security Considerations

### Avoid Hardcoded Secrets

```python
# BAD
api_key = "sk-1234567890abcdef"
wandb.login(api_key=api_key)

# GOOD
import os
api_key = os.environ.get('WANDB_API_KEY')
if not api_key:
    raise ValueError("WANDB_API_KEY environment variable not set")
wandb.login(api_key=api_key)
```

### Input Validation

```python
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    # Validate path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Ensure it's a file, not a directory
    if not os.path.isfile(config_path):
        raise ValueError(f"Config path is not a file: {config_path}")

    # Check file extension
    if not config_path.endswith('.yaml') and not config_path.endswith('.yml'):
        raise ValueError(f"Config must be YAML file: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  # Use safe_load, not load

    # Validate required keys
    required_keys = ['model', 'training', 'data']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    return config
```

### Safe File Operations

```python
import os
from pathlib import Path

def safe_save_model(model: nn.Module, save_path: str):
    """Safely save model to disk."""
    # Convert to Path object for better handling
    save_path = Path(save_path)

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to temporary file first
    temp_path = save_path.with_suffix('.tmp')

    try:
        torch.save(model.state_dict(), temp_path)
        # Atomic rename (on most filesystems)
        temp_path.replace(save_path)
        logger.info(f"Model saved: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        if temp_path.exists():
            temp_path.unlink()  # Clean up temp file
        raise
```

## Summary Checklist

Before committing code, ensure:

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] Error handling is implemented
- [ ] Tests are written and passing
- [ ] Code is documented
- [ ] No hardcoded secrets
- [ ] Git commit message is descriptive
- [ ] Code has been self-reviewed
- [ ] No debugging print statements

## Tools to Use

Install and configure these tools:

```bash
# Code formatting
pip install black isort

# Linting
pip install flake8 pylint

# Type checking
pip install mypy

# Testing
pip install pytest pytest-cov

# Pre-commit hooks
pip install pre-commit

# Run all checks
black src/
isort src/
flake8 src/
mypy src/
pytest --cov=src
```

## Further Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Real Python Tutorials](https://realpython.com/)
- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/best_practices.html)
- [Clean Code by Robert C. Martin](https://www.oreilly.com/library/view/clean-code-a/9780136083238/)
