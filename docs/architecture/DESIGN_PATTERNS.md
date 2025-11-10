# Design Patterns in the Project

## Table of Contents
- [Introduction](#introduction)
- [Creational Patterns](#creational-patterns)
- [Structural Patterns](#structural-patterns)
- [Behavioral Patterns](#behavioral-patterns)
- [Pattern Implementation Examples](#pattern-implementation-examples)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

## Introduction

This document explains the design patterns used throughout the project. Understanding these patterns will help you:
- Recognize common software design solutions
- Understand why certain code is structured the way it is
- Apply these patterns in your own projects
- Write more maintainable and extensible code

## Creational Patterns

### 1. Factory Pattern

**Purpose**: Encapsulate object creation logic

**Where Used**: Model and DataLoader creation

**Implementation**:
```python
# src/utils/factory.py (future implementation)
def create_model(config: dict) -> nn.Module:
    """Factory method for creating models based on configuration."""
    model_type = config['model']['architecture']

    if model_type == 'CustomCNN':
        return CustomCXRClassifier(
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['dropout_rate']
        )
    elif model_type == 'ResNet':
        return create_resnet_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Usage
model = create_model(config)  # Don't need to know which class to instantiate
```

**Benefits**:
- Decouples object creation from usage
- Makes it easy to add new model types
- Configuration-driven model selection

**Real Example in Code**:
```python
# scripts/train_mlflow.py
model = CustomCXRClassifier(num_classes=3)  # Direct instantiation

# Better approach (future):
model = ModelFactory.create(config)  # Factory handles details
```

### 2. Builder Pattern

**Purpose**: Construct complex objects step by step

**Where Used**: DataLoader configuration

**Implementation**:
```python
# Current approach in scripts/train_mlflow.py
train_loader = DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Builder pattern approach (future enhancement):
class DataLoaderBuilder:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.batch_size = 32
        self.shuffle = False
        self.num_workers = 0

    def with_batch_size(self, size: int):
        self.batch_size = size
        return self

    def with_shuffle(self, shuffle: bool = True):
        self.shuffle = shuffle
        return self

    def with_workers(self, num_workers: int):
        self.num_workers = num_workers
        return self

    def build(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

# Usage:
loader = (DataLoaderBuilder(dataset)
    .with_batch_size(32)
    .with_shuffle()
    .with_workers(4)
    .build())
```

**Benefits**:
- Fluent, readable API
- Optional parameters are clear
- Easy to add new configuration options

### 3. Singleton Pattern

**Purpose**: Ensure only one instance of a class exists

**Where Used**: Configuration management, logging

**Implementation**:
```python
# src/utils/config.py (future implementation)
class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(self, config_path: str):
        """Load configuration file once."""
        if self._config is None:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        return self._config

    @property
    def config(self):
        if self._config is None:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self._config

# Usage:
config_manager = ConfigManager()
config_manager.load_config('configs/mlflow/default_config.yaml')

# Anywhere else in the code:
config = ConfigManager().config  # Returns the same instance
```

**Benefits**:
- Configuration loaded once, shared everywhere
- Prevents multiple file reads
- Consistent configuration across modules

## Structural Patterns

### 1. Adapter Pattern

**Purpose**: Make incompatible interfaces work together

**Where Used**: Dataset handling for different folder structures

**Implementation**:
```python
# src/data/data_loader.py
class COVID19ChestXRayDataset(Dataset):
    def _find_images(self, root_dir: str) -> List[Tuple[str, int]]:
        """
        Adapter: handles different dataset folder structures.
        Adapts various naming conventions to a common interface.
        """
        images = []

        # Try different folder naming conventions
        class_folders = [
            ['COVID', 'Viral Pneumonia', 'Normal'],  # Format 1
            ['Covid', 'Pneumonia', 'Normal'],        # Format 2
            ['covid', 'pneumonia', 'normal']         # Format 3
        ]

        for folders in class_folders:
            if all(os.path.exists(os.path.join(root_dir, folder))
                   for folder in folders):
                # Adapt to common format
                for label, folder in enumerate(folders):
                    folder_path = os.path.join(root_dir, folder)
                    for img_file in os.listdir(folder_path):
                        images.append((
                            os.path.join(folder_path, img_file),
                            label
                        ))
                break

        return images
```

**Benefits**:
- Works with datasets from different sources
- Hides complexity from users
- Consistent interface regardless of underlying structure

### 2. Facade Pattern

**Purpose**: Provide a simplified interface to a complex subsystem

**Where Used**: Training scripts as facades to complex training pipelines

**Implementation**:
```python
# scripts/train_mlflow.py acts as a Facade
def main():
    """
    Facade: Simplifies the complex process of:
    - Loading configuration
    - Setting up datasets
    - Initializing models
    - Training
    - Tracking experiments
    """
    # Complex subsystem 1: Configuration
    config = load_config()

    # Complex subsystem 2: Data preparation
    train_loader, val_loader, test_loader = prepare_data(config)

    # Complex subsystem 3: Model setup
    model = setup_model(config)

    # Complex subsystem 4: Training
    train_with_mlflow(model, train_loader, val_loader, config)

    # Simple interface for users: just run python train_mlflow.py
```

**Benefits**:
- Simple entry point for complex operations
- Hides implementation details
- Easy to use for beginners

### 3. Decorator Pattern

**Purpose**: Add behavior to objects dynamically

**Where Used**: Data augmentation, metric computation

**Implementation**:
```python
# torchvision transforms use decorator pattern
transform = transforms.Compose([
    transforms.Resize((128, 128)),          # Decorator 1
    transforms.RandomHorizontalFlip(),      # Decorator 2
    transforms.RandomRotation(10),          # Decorator 3
    transforms.ColorJitter(0.1, 0.1, 0.1),  # Decorator 4
    transforms.ToTensor(),                  # Decorator 5
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])  # Decorator 6
])

# Each transform wraps the previous one, adding functionality
```

**Custom Decorator Example**:
```python
# src/utils/decorators.py (future implementation)
import time
import functools

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Usage:
@timing_decorator
def train_epoch(model, train_loader, optimizer, criterion):
    # Training code
    pass
```

**Benefits**:
- Add functionality without modifying original code
- Compose behaviors flexibly
- Follow Open/Closed Principle

## Behavioral Patterns

### 1. Strategy Pattern ⭐ (Most Important)

**Purpose**: Define a family of algorithms, encapsulate each one, and make them interchangeable

**Where Used**: Experiment tracking (MLflow vs W&B)

**Implementation**:
```python
# src/tracking/base_tracker.py
from abc import ABC, abstractmethod

class BaseTracker(ABC):
    """
    Strategy interface: defines the contract all trackers must follow.
    """

    @abstractmethod
    def start_run(self, experiment_name: str, run_name: str):
        """Initialize a tracking run."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics at a specific step."""
        pass

    @abstractmethod
    def log_params(self, params: dict):
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_artifact(self, artifact_path: str):
        """Log artifacts (files, models)."""
        pass

    @abstractmethod
    def end_run(self):
        """Finalize the tracking run."""
        pass


# src/tracking/mlflow_tracker.py
class MLflowTracker(BaseTracker):
    """Concrete Strategy: MLflow implementation."""

    def start_run(self, experiment_name: str, run_name: str):
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)

    def log_metrics(self, metrics: dict, step: int):
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_artifact(self, artifact_path: str):
        mlflow.log_artifact(artifact_path)

    def end_run(self):
        mlflow.end_run()


# src/tracking/wandb_tracker.py
class WandBTracker(BaseTracker):
    """Concrete Strategy: W&B implementation."""

    def start_run(self, experiment_name: str, run_name: str):
        wandb.init(project=experiment_name, name=run_name)

    def log_metrics(self, metrics: dict, step: int):
        wandb.log(metrics, step=step)

    def log_params(self, params: dict):
        wandb.config.update(params)

    def log_artifact(self, artifact_path: str):
        wandb.save(artifact_path)

    def end_run(self):
        wandb.finish()


# Context: Uses a strategy
def train_model(model, data_loader, tracker: BaseTracker):
    """
    This function doesn't care which tracker is used.
    It just calls the interface methods.
    """
    tracker.start_run("covid-xray", "experiment_1")

    for epoch in range(10):
        # Training code...
        metrics = {'loss': 0.5, 'accuracy': 0.85}
        tracker.log_metrics(metrics, step=epoch)

    tracker.end_run()

# Usage:
train_model(model, data_loader, MLflowTracker())  # Use MLflow
train_model(model, data_loader, WandBTracker())   # Use W&B
```

**Benefits**:
- Easy to switch between tracking platforms
- Can add new trackers without modifying training code
- Testing is easier (can use mock tracker)
- Follows Dependency Inversion Principle

### 2. Template Method Pattern

**Purpose**: Define skeleton of an algorithm, let subclasses override specific steps

**Where Used**: Training loop structure

**Implementation**:
```python
# src/training/base_trainer.py (future implementation)
class BaseTrainer(ABC):
    """Template: defines the training algorithm structure."""

    def train(self, num_epochs: int):
        """
        Template method: defines the overall training structure.
        Subclasses can override specific steps.
        """
        self.on_train_begin()

        for epoch in range(num_epochs):
            self.on_epoch_begin(epoch)

            # Training phase
            train_metrics = self.train_epoch(epoch)
            self.on_train_epoch_end(epoch, train_metrics)

            # Validation phase
            val_metrics = self.validate_epoch(epoch)
            self.on_val_epoch_end(epoch, val_metrics)

            # Learning rate scheduling
            self.step_scheduler()

            self.on_epoch_end(epoch)

        self.on_train_end()

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict:
        """Subclasses must implement training logic."""
        pass

    @abstractmethod
    def validate_epoch(self, epoch: int) -> dict:
        """Subclasses must implement validation logic."""
        pass

    def on_train_begin(self):
        """Hook: called before training starts."""
        pass

    def on_epoch_begin(self, epoch: int):
        """Hook: called at the start of each epoch."""
        pass

    def on_train_epoch_end(self, epoch: int, metrics: dict):
        """Hook: called after training epoch."""
        print(f"Epoch {epoch} train metrics: {metrics}")

    def on_val_epoch_end(self, epoch: int, metrics: dict):
        """Hook: called after validation epoch."""
        print(f"Epoch {epoch} val metrics: {metrics}")

    def step_scheduler(self):
        """Hook: called to update learning rate."""
        pass

    def on_epoch_end(self, epoch: int):
        """Hook: called at the end of each epoch."""
        pass

    def on_train_end(self):
        """Hook: called after training completes."""
        pass


# Concrete implementation
class MLflowTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, tracker):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.tracker = tracker

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        # Training loop implementation
        return {'loss': 0.5, 'accuracy': 0.85}

    def validate_epoch(self, epoch: int) -> dict:
        self.model.eval()
        # Validation loop implementation
        return {'loss': 0.3, 'accuracy': 0.90}

    def on_train_epoch_end(self, epoch: int, metrics: dict):
        super().on_train_epoch_end(epoch, metrics)
        self.tracker.log_metrics(metrics, step=epoch)
```

**Benefits**:
- Consistent training structure
- Easy to customize specific steps
- Reduces code duplication
- Enforces a standard workflow

### 3. Observer Pattern

**Purpose**: Define one-to-many dependency so when one object changes state, all dependents are notified

**Where Used**: Metric logging to multiple destinations

**Implementation**:
```python
# src/utils/observers.py (future implementation)
class MetricSubject:
    """Subject: maintains list of observers and notifies them of changes."""

    def __init__(self):
        self._observers = []

    def attach(self, observer):
        """Attach an observer."""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer):
        """Detach an observer."""
        self._observers.remove(observer)

    def notify(self, metrics: dict, step: int):
        """Notify all observers about metric updates."""
        for observer in self._observers:
            observer.update(metrics, step)


class MetricObserver(ABC):
    """Observer interface."""

    @abstractmethod
    def update(self, metrics: dict, step: int):
        """Called when subject notifies observers."""
        pass


class ConsoleLogger(MetricObserver):
    """Concrete Observer: logs to console."""

    def update(self, metrics: dict, step: int):
        print(f"Step {step}: {metrics}")


class FileLogger(MetricObserver):
    """Concrete Observer: logs to file."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def update(self, metrics: dict, step: int):
        with open(self.filepath, 'a') as f:
            f.write(f"Step {step}: {metrics}\n")


class TrackerLogger(MetricObserver):
    """Concrete Observer: logs to experiment tracker."""

    def __init__(self, tracker: BaseTracker):
        self.tracker = tracker

    def update(self, metrics: dict, step: int):
        self.tracker.log_metrics(metrics, step)


# Usage:
subject = MetricSubject()
subject.attach(ConsoleLogger())
subject.attach(FileLogger('metrics.log'))
subject.attach(TrackerLogger(MLflowTracker()))

# When metrics are computed:
metrics = {'loss': 0.5, 'accuracy': 0.85}
subject.notify(metrics, step=10)  # All observers are updated
```

**Benefits**:
- Loose coupling between metric source and destinations
- Easy to add new logging destinations
- Single point of metric emission

### 4. Iterator Pattern

**Purpose**: Provide a way to access elements sequentially without exposing underlying representation

**Where Used**: DataLoader (PyTorch built-in)

**Implementation**:
```python
# PyTorch's DataLoader uses Iterator pattern internally
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Usage:
for batch_idx, (images, labels) in enumerate(train_loader):
    # Process batch
    # We don't need to know how data is stored or fetched
    pass
```

## Pattern Implementation Examples

### Example 1: Adding a New Tracking Platform (Strategy Pattern)

Suppose you want to add **Neptune.ai** as a tracking platform:

**Step 1**: Implement the strategy interface
```python
# src/tracking/neptune_tracker.py
import neptune.new as neptune
from src.tracking.base_tracker import BaseTracker

class NeptuneTracker(BaseTracker):
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.run = None

    def start_run(self, experiment_name: str, run_name: str):
        self.run = neptune.init_run(
            project=experiment_name,
            name=run_name,
            api_token=self.api_token
        )

    def log_metrics(self, metrics: dict, step: int):
        for key, value in metrics.items():
            self.run[key].log(value, step=step)

    def log_params(self, params: dict):
        self.run['parameters'] = params

    def log_artifact(self, artifact_path: str):
        self.run['artifacts'].upload(artifact_path)

    def end_run(self):
        self.run.stop()
```

**Step 2**: Use it without changing training code
```python
# scripts/train_neptune.py
from src.tracking.neptune_tracker import NeptuneTracker

tracker = NeptuneTracker(api_token="YOUR_API_TOKEN")
train_model(model, data_loader, tracker)  # Same interface!
```

### Example 2: Adding Callbacks (Observer Pattern)

```python
# src/utils/callbacks.py
class CallbackList:
    """Manages multiple callbacks (observers)."""

    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = callbacks or []

    def on_epoch_end(self, epoch: int, logs: dict):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)


class Callback(ABC):
    """Callback interface."""

    def on_epoch_end(self, epoch: int, logs: dict):
        pass


class EarlyStopping(Callback):
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def on_epoch_end(self, epoch: int, logs: dict):
        val_loss = logs.get('val_loss')

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                logs['stop_training'] = True


class ModelCheckpoint(Callback):
    """Save model when validation loss improves."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch: int, logs: dict):
        val_loss = logs.get('val_loss')

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            model = logs.get('model')
            torch.save(model.state_dict(), self.filepath)
            print(f"Model saved at epoch {epoch}")


# Usage:
callbacks = CallbackList([
    EarlyStopping(patience=5),
    ModelCheckpoint('best_model.pth')
])

for epoch in range(num_epochs):
    # Training and validation...
    logs = {'val_loss': val_loss, 'model': model}
    callbacks.on_epoch_end(epoch, logs)

    if logs.get('stop_training'):
        break
```

## Anti-Patterns to Avoid

### 1. ❌ God Object

**What it is**: A class that knows too much or does too much

**Example**:
```python
# BAD: Everything in one class
class TrainingManager:
    def load_data(self): pass
    def create_model(self): pass
    def train_model(self): pass
    def evaluate_model(self): pass
    def log_to_mlflow(self): pass
    def log_to_wandb(self): pass
    def save_plots(self): pass
    def send_email_notification(self): pass
    # ... 50 more methods
```

**Solution**: Follow Single Responsibility Principle
```python
# GOOD: Separate concerns
class DataLoader: ...
class ModelFactory: ...
class Trainer: ...
class Evaluator: ...
class ExperimentTracker: ...
class Visualizer: ...
class Notifier: ...
```

### 2. ❌ Spaghetti Code

**What it is**: Code with poor structure and flow control

**Example**:
```python
# BAD: Complex nested logic
def train():
    if config:
        if model:
            if data:
                for epoch in range(epochs):
                    if epoch % 2 == 0:
                        if use_mlflow:
                            # log to mlflow
                        else:
                            if use_wandb:
                                # log to wandb
                            else:
                                # print
```

**Solution**: Use patterns and early returns
```python
# GOOD: Clean, flat structure
def train(model, data, tracker: BaseTracker):
    for epoch in range(epochs):
        metrics = train_epoch(model, data)
        tracker.log_metrics(metrics, epoch)
```

### 3. ❌ Hardcoding

**What it is**: Embedding configuration values directly in code

**Example**:
```python
# BAD
model = CustomCNN(num_classes=3, dropout=0.5)
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
```

**Solution**: Use configuration files
```python
# GOOD
config = load_config('config.yaml')
model = CustomCNN(**config['model'])
optimizer = Adam(model.parameters(), **config['optimizer'])
scheduler = StepLR(optimizer, **config['scheduler'])
```

### 4. ❌ Copy-Paste Programming

**What it is**: Duplicating code instead of refactoring

**Example**: `mlflow_tracker.py` and `wandb_tracker.py` share 70% identical code

**Solution**: Extract common logic to base class (Strategy Pattern)

## Summary

### Key Patterns in This Project:
1. **Strategy Pattern**: Interchangeable experiment trackers
2. **Factory Pattern**: Creating models and data loaders
3. **Adapter Pattern**: Handling different dataset structures
4. **Facade Pattern**: Simplifying complex training pipelines
5. **Template Method**: Defining training loop structure
6. **Observer Pattern**: Multi-destination metric logging

### Learning Path for Students:
1. Start with **Strategy Pattern** (most visible in the project)
2. Understand **Facade Pattern** (entry point scripts)
3. Learn **Factory Pattern** (object creation)
4. Study **Template Method** (training structure)
5. Explore **Observer Pattern** (event handling)

### Further Reading:
- "Design Patterns: Elements of Reusable Object-Oriented Software" by Gang of Four
- "Head First Design Patterns" by Freeman & Freeman
- [Refactoring Guru - Design Patterns](https://refactoring.guru/design-patterns)
- [Python Design Patterns](https://python-patterns.guide/)
