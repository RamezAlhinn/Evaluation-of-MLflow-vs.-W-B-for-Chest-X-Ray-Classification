# Software Engineering Refactoring Guide

## Introduction

This document explains the comprehensive refactoring of the Chest X-Ray Classification project, applying software engineering best practices and abstraction principles. This guide is written for engineering students learning these concepts.

## Table of Contents

1. [Problems Identified](#problems-identified)
2. [Software Engineering Principles Applied](#software-engineering-principles-applied)
3. [Refactoring Strategy](#refactoring-strategy)
4. [Detailed Changes](#detailed-changes)
5. [How to Use the New Architecture](#how-to-use-the-new-architecture)
6. [Learning Resources](#learning-resources)

---

## Problems Identified

### 1. Tight Coupling (High Severity)

**What it means:** Components are directly dependent on each other's implementation details, making it hard to change or test individual parts.

**Examples found:**
- Training logic embedded directly in `mlflow_tracker.py` and `wandb_tracker.py`
- Data loading logic tightly coupled to file system structure
- Model training can't work without specific tracker implementations

**Why it's a problem:**
- Hard to unit test individual components
- Can't swap implementations (e.g., use different data source)
- Changes in one component break others
- Code reuse is difficult

**Analogy:** Imagine a car where the engine is welded directly to the chassis. You can't replace just the engine or test it separately. That's tight coupling.

### 2. Code Duplication (High Severity)

**What it means:** Same or very similar code appears in multiple places (DRY principle violation - Don't Repeat Yourself).

**Examples found:**
```python
# Training loop appears in BOTH mlflow_tracker.py AND wandb_tracker.py
# 90% identical code, only logging differs
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        # ... same training logic ...
```

**Why it's a problem:**
- Bug fixes must be applied multiple times
- Inconsistencies creep in over time
- More code to maintain
- Higher chance of errors

**Fix:** Extract common logic into a shared base class or utility function.

### 3. Hardcoded Configuration (Medium Severity)

**What it means:** Values that should be configurable are written directly in code or config files.

**Examples found:**
```python
# In cnn_model.py
INPUT_CHANNELS = 3  # What if we want 1-channel grayscale?
NUM_CLASSES = 3     # What if dataset has 2 or 4 classes?

# In configs/*.yaml
dataset_path: "Covid19-dataset"  # Won't work on other machines
```

**Why it's a problem:**
- Not portable across different environments
- Can't easily experiment with different datasets
- Requires code changes for configuration changes
- Hard to deploy to different environments (dev/staging/prod)

### 4. Missing Abstraction (High Severity)

**What it means:** No abstract interfaces defining contracts between components.

**Examples found:**
- No base class for trackers (MLflow and W&B should implement same interface)
- No abstract data loader class
- No trainer interface

**Why it's a problem:**
- Can't write polymorphic code
- Hard to add new implementations
- No clear contract/API between components
- Difficult to mock for testing

**Analogy:** Like having different remote controls for TV, stereo, and lights with completely different button layouts. An abstract remote interface would define: "all remotes have power, volume, and channel buttons in the same place."

### 5. Model Architecture Issues (High Severity)

**Critical bug found:**
```python
# In cnn_model.py forward() method
x = F.softmax(x, dim=1)  # WRONG when using CrossEntropyLoss!
```

**Why it's wrong:**
- `CrossEntropyLoss` in PyTorch already includes softmax internally
- Applying softmax twice causes numerical instability
- Gradients computed incorrectly during backpropagation
- Model learns slower or fails to converge

**Correct approach:** Return raw logits from forward(), let loss function handle softmax.

### 6. No Separation of Concerns (Medium Severity)

**What it means:** Single components doing multiple unrelated things.

**Examples found:**
- Tracker classes handle both experiment tracking AND model training
- Data loader mixes data loading, validation, and transformation
- Scripts mix argument parsing, configuration, and execution

**Why it's a problem:**
- Components become bloated and hard to understand
- Changes affect multiple responsibilities
- Testing becomes complex
- Violates Single Responsibility Principle

### 7. Poor Error Handling (Medium Severity)

**Examples found:**
```python
# In data_loader.py
except Exception as e:
    return torch.zeros(3, 128, 128)  # Returns fake data silently!
```

**Why it's a problem:**
- Errors hidden instead of being fixed
- Hard to debug when things go wrong
- Silent failures corrupt results
- No logging of what went wrong

### 8. No Testing Infrastructure (High Severity)

**What's missing:**
- Zero unit tests
- No integration tests
- No validation that refactoring doesn't break functionality

**Why it's a problem:**
- Can't verify correctness
- Refactoring is risky
- Regressions go undetected
- Code quality degrades over time

### 9. Missing Type Hints (Low Severity, High Impact)

**What it means:** Function signatures don't specify parameter and return types.

**Example:**
```python
# Before
def load_data(dataset_path, batch_size, train_split):
    ...

# After (with type hints)
def load_data(
    dataset_path: str,
    batch_size: int,
    train_split: float
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ...
```

**Why it helps:**
- IDEs provide better autocomplete
- Catch type errors before running code
- Self-documenting code
- Static analysis tools can verify correctness

### 10. Inconsistent Logging (Low Severity)

**Examples found:**
```python
print(f"Epoch {epoch}")  # Using print statements
# vs proper logging:
logger.info(f"Epoch {epoch}")
```

**Why proper logging is better:**
- Can control log levels (DEBUG, INFO, WARNING, ERROR)
- Can redirect to files
- Can filter logs by module
- Production-ready

---

## Software Engineering Principles Applied

### 1. SOLID Principles

#### S - Single Responsibility Principle (SRP)

**Definition:** A class should have only one reason to change.

**Application in refactoring:**

**Before:**
```python
class MLflowTracker:
    def train_model(self, model, train_loader, val_loader, num_epochs):
        # This class does BOTH tracking AND training
        mlflow.log_param(...)
        for epoch in range(num_epochs):
            # training logic...
```

**After:**
```python
class Trainer:
    """Single responsibility: train models"""
    def train(self, model, train_loader, val_loader, num_epochs):
        # Pure training logic

class MLflowTracker:
    """Single responsibility: track experiments"""
    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)
```

Each class now has one clear purpose.

#### O - Open/Closed Principle (OCP)

**Definition:** Software entities should be open for extension but closed for modification.

**Application:**

**Before:** To add a new tracker (like TensorBoard), you'd have to create a completely new training function.

**After:**
```python
class BaseTracker(ABC):
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]): pass

class MLflowTracker(BaseTracker):
    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)

class WandBTracker(BaseTracker):
    def log_metrics(self, metrics):
        wandb.log(metrics)

# NEW: Add TensorBoard without modifying existing code
class TensorBoardTracker(BaseTracker):
    def log_metrics(self, metrics):
        self.writer.add_scalars(metrics)
```

You can add new trackers without changing existing classes.

#### L - Liskov Substitution Principle (LSP)

**Definition:** Objects of a superclass should be replaceable with objects of subclasses without breaking functionality.

**Application:**
```python
# Any BaseTracker can be used interchangeably
def train_with_tracking(trainer: Trainer, tracker: BaseTracker):
    # Works with MLflow, W&B, or any future tracker
    metrics = trainer.train_epoch()
    tracker.log_metrics(metrics)

# You can swap trackers without changing code
tracker = MLflowTracker()  # or WandBTracker() or TensorBoardTracker()
train_with_tracking(trainer, tracker)
```

#### I - Interface Segregation Principle (ISP)

**Definition:** Clients shouldn't be forced to depend on interfaces they don't use.

**Application:**
```python
# Instead of one giant interface:
class ITracker(ABC):
    @abstractmethod
    def log_metrics(self): pass
    @abstractmethod
    def log_model(self): pass
    @abstractmethod
    def log_artifacts(self): pass
    # ... 20 more methods

# Split into focused interfaces:
class IMetricsLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: Dict): pass

class IModelLogger(ABC):
    @abstractmethod
    def log_model(self, model): pass

# Implement only what you need
class SimpleTracker(IMetricsLogger):
    def log_metrics(self, metrics):
        print(metrics)
```

#### D - Dependency Inversion Principle (DIP)

**Definition:** High-level modules shouldn't depend on low-level modules. Both should depend on abstractions.

**Application:**

**Before (Direct Dependency):**
```python
class Trainer:
    def __init__(self):
        self.tracker = MLflowTracker()  # Hard dependency on concrete class
```

**After (Dependency Injection):**
```python
class Trainer:
    def __init__(self, tracker: BaseTracker):  # Depends on abstraction
        self.tracker = tracker

# Inject the dependency
mlflow_tracker = MLflowTracker()
trainer = Trainer(tracker=mlflow_tracker)
```

Now `Trainer` doesn't know or care about MLflow specifically. You can inject any tracker.

### 2. Design Patterns

#### Strategy Pattern

**What it is:** Defines a family of algorithms, encapsulates each one, and makes them interchangeable.

**Application:**
```python
# Different tracking strategies
tracker = MLflowTracker() if use_mlflow else WandBTracker()
trainer = Trainer(tracker=tracker)  # Same trainer, different strategy
```

#### Factory Pattern

**What it is:** Creates objects without specifying the exact class.

**Application:**
```python
class TrackerFactory:
    @staticmethod
    def create_tracker(tracker_type: str, config: Config) -> BaseTracker:
        if tracker_type == "mlflow":
            return MLflowTracker(config.mlflow)
        elif tracker_type == "wandb":
            return WandBTracker(config.wandb)
        else:
            raise ValueError(f"Unknown tracker: {tracker_type}")

# Usage
tracker = TrackerFactory.create_tracker("mlflow", config)
```

#### Template Method Pattern

**What it is:** Defines the skeleton of an algorithm, letting subclasses override specific steps.

**Application:**
```python
class BaseTracker(ABC):
    def track_training(self, trainer: Trainer):
        self.start_run()  # Template method
        for epoch in range(trainer.num_epochs):
            metrics = trainer.train_epoch()
            self.log_metrics(metrics)  # Subclass implements this
        self.end_run()

    @abstractmethod
    def log_metrics(self, metrics): pass
```

### 3. Composition Over Inheritance

**Principle:** Favor object composition over class inheritance.

**Why:** Inheritance creates tight coupling and "is-a" relationships. Composition creates flexible "has-a" relationships.

**Application:**
```python
# Instead of:
class MLflowTrainer(Trainer):  # Inheritance
    pass

# Use:
class Trainer:
    def __init__(self, tracker: BaseTracker):  # Composition
        self.tracker = tracker
```

### 4. Dependency Injection

**What it is:** Provide dependencies from outside rather than creating them internally.

**Benefits:**
- Easier testing (inject mocks)
- Flexible configuration
- Loose coupling

**Application throughout refactored code:**
```python
# Constructor injection
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        tracker: BaseTracker,
        config: TrainingConfig
    ):
        self.model = model
        self.tracker = tracker
        self.config = config
```

### 5. Configuration as Code

**Principle:** Configuration should be:
1. Externalized (not in code)
2. Validated (catch errors early)
3. Typed (know what to expect)
4. Hierarchical (organized in groups)

**Application:**
```python
@dataclass
class DataConfig:
    dataset_path: str = os.getenv("DATASET_PATH", "Covid19-dataset")
    batch_size: int = 32
    image_size: int = 128

    def __post_init__(self):
        """Validate configuration"""
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.image_size not in [64, 128, 224]:
            raise ValueError("image_size must be 64, 128, or 224")
```

### 6. Separation of Concerns

**Principle:** Different concerns should be handled by different components.

**Application:**

| Concern | Component | Responsibility |
|---------|-----------|----------------|
| Data loading | DataLoader | Load and preprocess data |
| Model definition | CNNModel | Define architecture |
| Training | Trainer | Training loop logic |
| Tracking | Tracker | Log metrics and artifacts |
| Configuration | Config classes | Manage settings |
| Orchestration | Scripts | Coordinate components |

---

## Refactoring Strategy

### Phase 1: Create Abstractions (Foundation)

1. **Define base classes (interfaces)**
   - `BaseTracker` - abstract tracker interface
   - `BaseDataLoader` - abstract data loader interface
   - `BaseModel` - model interface (optional)

2. **Create configuration system**
   - Data classes for type-safe configuration
   - Validation logic
   - Environment variable support

### Phase 2: Extract Core Logic (Decoupling)

3. **Create Trainer class**
   - Extract training loop from trackers
   - Make it tracker-agnostic
   - Add early stopping, checkpointing

4. **Refactor trackers**
   - Implement BaseTracker interface
   - Remove training logic
   - Keep only tracking functionality

### Phase 3: Improve Quality (Best Practices)

5. **Add type hints** throughout codebase
6. **Implement proper logging** with logging module
7. **Fix model architecture bugs** (softmax issue)
8. **Add docstrings** where missing

### Phase 4: Make Configurable (Flexibility)

9. **Remove hardcoded values**
   - Use configuration classes
   - Support environment variables
   - Add sensible defaults

10. **Add dependency injection**
    - Constructor injection for all dependencies
    - Factory pattern for object creation

### Phase 5: Testing & Documentation (Quality Assurance)

11. **Add unit tests** for core components
12. **Update documentation** with examples
13. **Create usage examples** for new architecture

---

## Detailed Changes

### Change 1: Create Base Tracker Interface

**Location:** Create `src/tracking/base_tracker.py`

**Why:** Provides common interface for all tracking implementations, enabling polymorphism.

**Code:**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch.nn as nn

class BaseTracker(ABC):
    """
    Abstract base class for experiment tracking systems.

    This defines the contract that all tracker implementations must follow.
    By programming to this interface, we can swap tracking systems without
    changing other code (Dependency Inversion Principle).
    """

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None, **kwargs) -> None:
        """
        Start a new tracking run.

        Args:
            run_name: Optional name for the run
            **kwargs: Additional tracker-specific parameters
        """
        pass

    @abstractmethod
    def end_run(self) -> None:
        """End the current tracking run and cleanup resources."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params: Dictionary of parameter names and values
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (epoch, batch, etc.)
        """
        pass

    @abstractmethod
    def log_model(self, model: nn.Module, artifact_path: str) -> None:
        """
        Log a trained model.

        Args:
            model: PyTorch model to log
            artifact_path: Path where model should be stored
        """
        pass
```

**Learning points:**
- `ABC` (Abstract Base Class) ensures this can't be instantiated directly
- `@abstractmethod` decorator forces subclasses to implement these methods
- Type hints make the contract explicit
- Docstrings explain what each method should do

### Change 2: Refactor MLflow Tracker

**Location:** Modify `src/tracking/mlflow_tracker.py`

**Changes:**
1. Inherit from `BaseTracker`
2. Remove training logic (move to `Trainer`)
3. Keep only tracking functionality
4. Add type hints

**Before (excerpt):**
```python
class MLflowTracker:
    def train_with_mlflow(self, model, train_loader, val_loader, ...):
        mlflow.start_run()
        optimizer = optim.Adam(...)  # Training logic in tracker
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                # ... training code ...
```

**After:**
```python
from typing import Dict, Any, Optional
import mlflow
from .base_tracker import BaseTracker

class MLflowTracker(BaseTracker):
    """
    MLflow implementation of BaseTracker.

    Handles experiment tracking using MLflow's API.
    """

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: Optional MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = self.experiment.experiment_id

    def start_run(self, run_name: Optional[str] = None, **kwargs) -> None:
        """Start MLflow run."""
        mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )

    def end_run(self) -> None:
        """End MLflow run."""
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: nn.Module, artifact_path: str) -> None:
        """Log PyTorch model to MLflow."""
        mlflow.pytorch.log_model(model, artifact_path)
```

**Learning points:**
- Class now focused solely on MLflow integration (Single Responsibility)
- Implements all abstract methods from BaseTracker
- No training logic - that's moved to Trainer
- Configuration passed via constructor (Dependency Injection)

### Change 3: Create Trainer Class

**Location:** Create `src/training/trainer.py`

**Why:** Separate training logic from tracking. Trainer doesn't know or care about tracking implementation.

**Code:**
```python
from typing import Optional, Dict, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """
    Configuration for training process.

    Using dataclass provides:
    - Type checking
    - Default values
    - Automatic __init__, __repr__
    """
    num_epochs: int = 10
    learning_rate: float = 0.001
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    early_stopping_patience: int = 5
    checkpoint_dir: Optional[str] = None

class Trainer:
    """
    Handles model training independent of tracking system.

    This class follows the Single Responsibility Principle - it only
    handles training logic. Tracking is injected as a dependency.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        tracker: Optional[BaseTracker] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration
            tracker: Optional experiment tracker (Dependency Injection!)
            optimizer: Optional optimizer (defaults to Adam)
            criterion: Optional loss function (defaults to CrossEntropyLoss)
        """
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tracker = tracker
        self.device = config.device

        # Default optimizer if none provided
        self.optimizer = optimizer or optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )

        # Default loss function
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        logger.info(f"Trainer initialized with device: {self.device}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }

    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    def train(self) -> Dict[str, Any]:
        """
        Full training loop with early stopping.

        Returns:
            Training history and final metrics
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        # Log hyperparameters if tracker available
        if self.tracker:
            self.tracker.log_params({
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs,
                'device': self.device,
                'optimizer': self.optimizer.__class__.__name__
            })

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}

            # Log to tracker if available
            if self.tracker:
                self.tracker.log_metrics(metrics, step=epoch)

            # Log to console
            logger.info(
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
            )

            # Early stopping check
            if self._should_stop_early(val_metrics['val_loss']):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        logger.info("Training completed")
        return metrics

    def _should_stop_early(self, val_loss: float) -> bool:
        """
        Check if early stopping criteria met.

        Args:
            val_loss: Current validation loss

        Returns:
            True if should stop training
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        if self.patience_counter >= self.config.early_stopping_patience:
            return True

        return False
```

**Learning points:**
- `TrainingConfig` dataclass makes configuration explicit and typed
- Trainer is completely independent of tracking system (Dependency Inversion)
- Tracker is optional - can train without tracking
- Optimizer and criterion can be injected (flexibility)
- Early stopping logic included
- Proper logging instead of print statements
- Private method `_should_stop_early` (encapsulation)

### Change 4: Create Configuration System

**Location:** Create `src/config/config.py`

**Why:** Centralize configuration, add validation, support environment variables.

**Code:**
```python
from dataclasses import dataclass, field
from typing import Optional, List
import os
from pathlib import Path

@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset_path: str = field(
        default_factory=lambda: os.getenv("DATASET_PATH", "Covid19-dataset")
    )
    batch_size: int = 32
    image_size: int = 128
    num_workers: int = 4
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Check splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(
                f"Splits must sum to 1.0, got {total_split}"
            )

        # Check batch size
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

        # Check image size
        if self.image_size not in [64, 128, 224]:
            raise ValueError("image_size must be 64, 128, or 224")

        # Check dataset path exists
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.dataset_path}"
            )

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    input_channels: int = 3
    num_classes: int = 3
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.3, 0.4, 0.25])

    def __post_init__(self):
        """Validate configuration."""
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        for rate in self.dropout_rates:
            if not 0 <= rate < 1:
                raise ValueError(f"Dropout rate {rate} must be in [0, 1)")

@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_epochs: int = 10
    learning_rate: float = 0.001
    device: str = field(
        default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    early_stopping_patience: int = 5
    checkpoint_dir: Optional[str] = None
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

@dataclass
class MLflowConfig:
    """Configuration for MLflow tracking."""
    experiment_name: str = "chest-xray-classification"
    tracking_uri: Optional[str] = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI")
    )
    artifact_location: Optional[str] = None

@dataclass
class WandBConfig:
    """Configuration for Weights & Biases tracking."""
    project: str = "chest-xray-classification"
    entity: Optional[str] = field(
        default_factory=lambda: os.getenv("WANDB_ENTITY")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("WANDB_API_KEY")
    )

@dataclass
class ExperimentConfig:
    """
    Master configuration combining all sub-configs.

    This is the main configuration object used throughout the application.
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ExperimentConfig instance
        """
        import yaml

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create config with nested dataclasses
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            mlflow=MLflowConfig(**config_dict.get('mlflow', {})),
            wandb=WandBConfig(**config_dict.get('wandb', {}))
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
```

**Learning points:**
- Dataclasses provide clean configuration objects
- `field(default_factory=...)` allows dynamic defaults (environment variables)
- `__post_init__` validates configuration immediately
- Hierarchical structure (`ExperimentConfig` contains sub-configs)
- `from_yaml` class method provides alternate constructor
- Environment variables with defaults (e.g., `os.getenv("DATASET_PATH", "Covid19-dataset")`)
- Type hints make expectations clear

### Change 5: Fix Model Architecture

**Location:** Modify `src/models/cnn_model.py`

**Critical fixes:**
1. Remove softmax from forward() method
2. Make parameters configurable
3. Calculate FC input size dynamically

**Before:**
```python
INPUT_CHANNELS = 3  # Global constant
NUM_CLASSES = 3     # Global constant

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Hardcoded architecture
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # ...
        self.fc_input_features = 6272  # Only works for 128x128

    def forward(self, x):
        # ... conv layers ...
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # WRONG!
        return x
```

**After:**
```python
from typing import List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for CNN model architecture."""
    input_channels: int = 3
    num_classes: int = 3
    dropout_rates: List[float] = (0.25, 0.3, 0.4, 0.25)
    image_size: int = 128

class CNNModel(nn.Module):
    """
    Convolutional Neural Network for image classification.

    Architecture:
        Conv(16) -> Pool -> Conv(64) -> Pool -> Conv(128) -> Conv(128) -> Pool -> FC

    Attribution:
        Based on architecture from:
        https://github.com/Vinay10100/Chest-X-Ray-Classification
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize model.

        Args:
            config: Model configuration containing architecture parameters
        """
        super().__init__()
        self.config = config

        # First convolutional block
        self.conv1 = nn.Conv2d(
            config.input_channels,
            16,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(config.dropout_rates[0])

        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(config.dropout_rates[1])

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(config.dropout_rates[2])

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(config.dropout_rates[3])

        # Calculate FC input size dynamically
        self.fc_input_features = self._calculate_fc_input_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_features, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, config.num_classes)

    def _calculate_fc_input_size(self) -> int:
        """
        Dynamically calculate FC layer input size.

        This handles different image sizes automatically by running
        a dummy forward pass through conv layers.

        Returns:
            Number of features after conv layers
        """
        # Create dummy input with configured size
        dummy_input = torch.zeros(
            1,
            self.config.input_channels,
            self.config.image_size,
            self.config.image_size
        )

        # Forward through conv layers only
        x = self._forward_conv_layers(dummy_input)

        # Return flattened size
        return x.view(1, -1).size(1)

    def _forward_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional layers only.

        Extracted as separate method to enable FC size calculation
        and improve code organization.
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)

        Note:
            Returns RAW LOGITS, not probabilities.
            If using CrossEntropyLoss, do NOT apply softmax here!
            CrossEntropyLoss expects logits and applies log_softmax internally.
        """
        # Convolutional layers
        x = self._forward_conv_layers(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        # Return logits (NO SOFTMAX!)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.

        Use this method when you want probabilities instead of logits.

        Args:
            x: Input tensor

        Returns:
            Probability distribution over classes
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
```

**Learning points:**
- **Critical fix:** Removed softmax from forward() - returns raw logits
- Added `predict_proba()` for when probabilities are actually needed
- Configuration injected via dataclass (flexible architecture)
- Dynamic FC size calculation works for any image size
- Extracted `_forward_conv_layers()` (code reuse, organization)
- Comprehensive docstrings explain the "why"
- Type hints on all methods
- Single underscore prefix for private methods (Python convention)

### Change 6: Refactor Data Loader

**Location:** Modify `src/data/data_loader.py`

**Changes:**
1. Make transforms configurable
2. Better error handling
3. Add logging
4. Remove silent error masking

**Key improvements:**
```python
import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataLoadingConfig:
    """Configuration for data loading."""
    dataset_path: str
    batch_size: int = 32
    image_size: int = 128
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True
    # Augmentation parameters
    random_horizontal_flip_prob: float = 0.5
    random_rotation_degrees: int = 10
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2

class ChestXRayDataset(Dataset):
    """Custom dataset for chest X-ray images."""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory containing class subdirectories
            transform: Optional transforms to apply
            class_names: Optional list of expected class names
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Discover classes and load image paths
        self.classes, self.class_to_idx, self.samples = self._load_samples(class_names)

        logger.info(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        logger.info(f"Classes: {self.classes}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.

        Args:
            idx: Index of item

        Returns:
            Tuple of (image_tensor, label)

        Raises:
            RuntimeError: If image fails to load
        """
        img_path, label = self.samples[idx]

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # DON'T return fake data - raise error so we know about it
            raise RuntimeError(f"Failed to load image {img_path}") from e

    def _load_samples(
        self,
        expected_classes: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict[str, int], List[Tuple[Path, int]]]:
        """
        Load all samples from directory structure.

        Args:
            expected_classes: Optional list of expected class names

        Returns:
            Tuple of (class_names, class_to_idx, samples)

        Raises:
            ValueError: If no valid classes found
        """
        # Discover class directories
        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.root_dir}")

        # Map class names
        classes = sorted([d.name for d in class_dirs])
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # Validate expected classes if provided
        if expected_classes:
            for expected in expected_classes:
                if expected not in classes:
                    logger.warning(
                        f"Expected class '{expected}' not found. "
                        f"Available classes: {classes}"
                    )

        # Load all image paths
        samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png'}

        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = class_to_idx[class_name]

            # Find all images in class directory
            images = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in valid_extensions
            ]

            logger.info(f"Class '{class_name}': {len(images)} images")

            for img_path in images:
                samples.append((img_path, class_idx))

        if not samples:
            raise ValueError(f"No valid images found in {self.root_dir}")

        return classes, class_to_idx, samples

def create_data_loaders(
    config: DataLoadingConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        config: Data loading configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating data loaders from {config.dataset_path}")

    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=config.random_horizontal_flip_prob),
        transforms.RandomRotation(config.random_rotation_degrees),
        transforms.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load full dataset
    full_dataset = ChestXRayDataset(
        root_dir=config.dataset_path,
        transform=None  # Will set per split
    )

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        config.train_split,
        config.val_split,
        config.test_split
    )

    # Set transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    logger.info(
        f"Created data loaders: "
        f"train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    return train_loader, val_loader, test_loader
```

**Learning points:**
- Configuration object makes transforms customizable
- Proper error handling - raises exceptions instead of masking errors
- Logging provides visibility into data loading process
- Configurable augmentation parameters
- Clean separation: Dataset class vs. data loader creation
- Type hints make usage clear
- Comprehensive docstrings

---

## How to Use the New Architecture

### Example 1: Training with MLflow

```python
import logging
from src.config.config import ExperimentConfig
from src.data.data_loader import create_data_loaders
from src.models.cnn_model import CNNModel
from src.training.trainer import Trainer, TrainingConfig
from src.tracking.mlflow_tracker import MLflowTracker

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load configuration
config = ExperimentConfig.from_yaml('configs/experiment.yaml')

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(config.data)

# Create model
model = CNNModel(config.model)

# Create tracker (Dependency Injection)
tracker = MLflowTracker(
    experiment_name=config.mlflow.experiment_name,
    tracking_uri=config.mlflow.tracking_uri
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config.training,
    tracker=tracker  # Inject tracker
)

# Start tracking and train
tracker.start_run(run_name="experiment_1")
try:
    metrics = trainer.train()
    tracker.log_model(model, "model")
finally:
    tracker.end_run()
```

### Example 2: Training with W&B

```python
# Same setup as above, just swap the tracker
from src.tracking.wandb_tracker import WandBTracker

tracker = WandBTracker(
    project=config.wandb.project,
    entity=config.wandb.entity
)

# Rest is identical - polymorphism in action!
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config.training,
    tracker=tracker  # Different tracker, same interface
)
```

### Example 3: Training without tracking

```python
# Don't pass a tracker
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config.training,
    tracker=None  # No tracking
)

# Still works!
metrics = trainer.train()
```

### Example 4: Custom optimizer

```python
import torch.optim as optim

# Create custom optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

# Inject it
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config.training,
    tracker=tracker,
    optimizer=optimizer  # Custom optimizer
)
```

### Example 5: Using environment variables

```bash
# Set environment variables
export DATASET_PATH="/data/chest-xrays"
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
export WANDB_API_KEY="your-api-key"

# Run script - automatically uses environment variables
python scripts/train.py --config configs/experiment.yaml
```

### Example 6: Configuration in code

```python
from src.config.config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig
)

# Create configuration programmatically
config = ExperimentConfig(
    data=DataConfig(
        dataset_path="/custom/path",
        batch_size=64,
        image_size=224
    ),
    model=ModelConfig(
        num_classes=4,  # Different number of classes
        dropout_rates=[0.3, 0.3, 0.4, 0.3]
    ),
    training=TrainingConfig(
        num_epochs=50,
        learning_rate=0.0001,
        early_stopping_patience=10
    )
)

# Use configuration
train_loader, val_loader, test_loader = create_data_loaders(config.data)
model = CNNModel(config.model)
trainer = Trainer(model, train_loader, val_loader, config.training)
```

---

## Benefits of Refactoring

### 1. Testability

**Before:** Hard to test because everything is coupled.

**After:**
```python
import unittest
from unittest.mock import Mock

class TestTrainer(unittest.TestCase):
    def test_training_logs_metrics(self):
        # Create mock tracker
        mock_tracker = Mock(spec=BaseTracker)

        # Create trainer with mock
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            tracker=mock_tracker  # Inject mock
        )

        # Train
        trainer.train()

        # Verify tracker was called
        mock_tracker.log_metrics.assert_called()
```

### 2. Flexibility

- Easy to swap implementations
- Add new trackers without changing existing code
- Experiment with different configurations
- Support multiple datasets

### 3. Maintainability

- Clear responsibilities
- Easy to locate bugs
- Changes isolated to single components
- Self-documenting code with type hints

### 4. Reusability

- Components can be used independently
- Trainer works with any tracker
- Data loader works with any model
- Configuration system reusable

### 5. Scalability

- Easy to add new features
- Can extend without modifying existing code
- Clear extension points

---

## Learning Resources

### Books

1. **"Clean Code" by Robert C. Martin**
   - Covers naming, functions, classes, and code organization
   - Essential for writing maintainable code

2. **"Design Patterns: Elements of Reusable Object-Oriented Software"**
   - The original design patterns book
   - Covers patterns used in this refactoring

3. **"Refactoring: Improving the Design of Existing Code" by Martin Fowler**
   - Techniques for improving code structure
   - Catalog of refactoring patterns

### Online Resources

1. **SOLID Principles**
   - https://www.digitalocean.com/community/conceptual_articles/s-o-l-i-d-the-first-five-principles-of-object-oriented-design

2. **Python Type Hints**
   - https://docs.python.org/3/library/typing.html
   - https://mypy.readthedocs.io/

3. **Design Patterns in Python**
   - https://refactoring.guru/design-patterns/python

4. **Dataclasses**
   - https://docs.python.org/3/library/dataclasses.html

### Key Concepts Summary

1. **Abstraction:** Hide complexity behind simple interfaces
2. **Encapsulation:** Bundle data and methods that operate on that data
3. **Inheritance:** Code reuse through "is-a" relationships
4. **Polymorphism:** Multiple implementations of same interface
5. **Composition:** Building complex objects from simpler ones
6. **Dependency Injection:** Provide dependencies from outside
7. **Separation of Concerns:** Different parts handle different responsibilities
8. **DRY (Don't Repeat Yourself):** Avoid code duplication
9. **KISS (Keep It Simple, Stupid):** Simplest solution that works
10. **YAGNI (You Aren't Gonna Need It):** Don't add features until needed

---

## Glossary

**Abstraction:** Hiding implementation details behind an interface.

**Abstract Base Class (ABC):** A class that cannot be instantiated and defines methods that subclasses must implement.

**Coupling:** The degree to which one component depends on another. Tight coupling is bad, loose coupling is good.

**Dataclass:** Python class that automatically generates common methods like `__init__`, `__repr__`, etc.

**Dependency Injection:** Providing dependencies from outside rather than creating them internally.

**Design Pattern:** A reusable solution to a commonly occurring problem in software design.

**Factory:** An object or method that creates other objects.

**Interface:** A contract defining what methods a class must implement (in Python, usually an ABC).

**Polymorphism:** Ability to use objects of different types through a common interface.

**Refactoring:** Improving code structure without changing external behavior.

**Type Hint:** Annotation specifying the expected type of a variable or parameter.

---

## Conclusion

This refactoring transforms the codebase from a functional but tightly-coupled implementation into a well-structured, maintainable system following software engineering best practices.

Key achievements:
- ✓ Eliminated tight coupling through abstraction
- ✓ Removed code duplication
- ✓ Made configuration flexible and validated
- ✓ Fixed critical model architecture bug
- ✓ Improved testability with dependency injection
- ✓ Added proper logging
- ✓ Made code self-documenting with type hints
- ✓ Organized code with clear separation of concerns

The architecture is now:
- **Testable:** Components can be tested in isolation
- **Flexible:** Easy to swap implementations
- **Maintainable:** Clear responsibilities and organization
- **Extensible:** Can add features without modifying existing code
- **Professional:** Follows industry best practices

Remember: Good software engineering is not about being clever or complex. It's about being clear, simple, and maintainable. Write code that your future self (or teammates) will thank you for!
