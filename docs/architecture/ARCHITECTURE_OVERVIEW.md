# Architecture Overview

## Table of Contents
- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Design Principles](#design-principles)
- [Layer Architecture](#layer-architecture)
- [Component Relationships](#component-relationships)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)

## Introduction

This project demonstrates a **production-grade machine learning architecture** for comparing experiment tracking platforms (MLflow vs. W&B) in the context of COVID-19 chest X-ray classification. The architecture follows industry best practices for modularity, testability, and maintainability.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT TRACKING LAYER                 │
│  ┌──────────────────┐           ┌──────────────────┐       │
│  │  MLflow Tracker  │           │   W&B Tracker    │       │
│  └────────┬─────────┘           └────────┬─────────┘       │
│           │                              │                  │
│           └───────────┬──────────────────┘                  │
│                       │                                     │
│                       ▼                                     │
│           ┌───────────────────────┐                        │
│           │    Base Tracker       │                        │
│           │  (Abstract Interface) │                        │
│           └───────────┬───────────┘                        │
└───────────────────────┼─────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────┐
│                       ▼       TRAINING LAYER                │
│           ┌───────────────────────┐                        │
│           │   Training Pipeline   │                        │
│           │  - Training loop      │                        │
│           │  - Validation         │                        │
│           │  - Model checkpointing│                        │
│           └───────────┬───────────┘                        │
│                       │                                     │
│           ┌───────────┴───────────┐                        │
│           │                       │                        │
│           ▼                       ▼                        │
│  ┌─────────────────┐     ┌──────────────────┐            │
│  │   Model Layer   │     │  Evaluation      │            │
│  │  CustomCXRNet   │     │  Metrics & Loss  │            │
│  └────────┬────────┘     └──────────────────┘            │
└───────────┼──────────────────────────────────────────────┘
            │
┌───────────┼──────────────────────────────────────────────┐
│           ▼           DATA LAYER                          │
│  ┌──────────────────────────────┐                        │
│  │  COVID19ChestXRayDataset     │                        │
│  │  - Data loading              │                        │
│  │  - Preprocessing             │                        │
│  │  - Augmentation              │                        │
│  └──────────┬───────────────────┘                        │
│             │                                             │
│             ▼                                             │
│  ┌──────────────────────────────┐                        │
│  │      Raw Image Data          │                        │
│  │  (COVID/Pneumonia/Normal)    │                        │
│  └──────────────────────────────┘                        │
└───────────────────────────────────────────────────────────┘
```

### Architecture Style

The project follows a **Layered Architecture Pattern** with:
- **Separation of Concerns**: Each layer has a distinct responsibility
- **Dependency Inversion**: Higher layers depend on abstractions, not implementations
- **Single Responsibility**: Each module does one thing well
- **Open/Closed Principle**: Open for extension, closed for modification

## Design Principles

### 1. **Modularity**
Every component is self-contained and can be tested independently:
```python
src/
├── models/      # Model definitions (can be imported and used elsewhere)
├── data/        # Data handling (independent of models)
├── tracking/    # Experiment tracking (pluggable backends)
└── utils/       # Shared utilities (no dependencies on other layers)
```

### 2. **Configuration-Driven Design**
All hyperparameters and settings are externalized to YAML files:
```yaml
# configs/mlflow/default_config.yaml
model:
  architecture: CustomCNN
  input_size: [128, 128]

training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001
```

### 3. **Dependency Injection**
Components receive their dependencies rather than creating them:
```python
def train_with_mlflow(
    model: nn.Module,           # Injected
    train_loader: DataLoader,   # Injected
    val_loader: DataLoader,     # Injected
    config: dict,               # Injected
    tracker: BaseTracker        # Injected
):
    # Training logic uses injected dependencies
```

### 4. **Strategy Pattern for Tracking**
Different tracking backends implement the same interface:
```python
class BaseTracker(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        pass

class MLflowTracker(BaseTracker):
    def log_metrics(self, metrics: dict, step: int):
        mlflow.log_metrics(metrics, step=step)

class WandBTracker(BaseTracker):
    def log_metrics(self, metrics: dict, step: int):
        wandb.log(metrics, step=step)
```

## Layer Architecture

### Layer 1: Data Layer (`src/data/`)

**Responsibility**: Loading, preprocessing, and augmenting image data

**Key Components**:
- `COVID19ChestXRayDataset`: Custom PyTorch Dataset
- Data augmentation pipeline
- Train/validation/test splitting

**Design Decisions**:
- Uses PyTorch's Dataset/DataLoader abstraction
- Implements error handling for corrupted images
- Supports multiple dataset folder structures
- ImageNet normalization for transfer learning compatibility

```python
# Data Layer Interface
class COVID19ChestXRayDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: Path to dataset root
            transform: Optional torchvision transforms
        """

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Returns (image, label) tuple"""

    def __len__(self) -> int:
        """Returns dataset size"""
```

### Layer 2: Model Layer (`src/models/`)

**Responsibility**: Neural network architecture definitions

**Key Components**:
- `CustomCXRClassifier`: 4-block CNN architecture
- Model configuration management
- Forward pass implementation

**Design Decisions**:
- Modular convolutional blocks
- Progressive dropout for regularization
- Softmax in forward pass (note: redundant with CrossEntropyLoss)
- Configurable architecture parameters

**Model Architecture**:
```
Input (3, 128, 128)
    ↓
[Conv Block 1] → 16 filters, 3x3, MaxPool
    ↓
[Conv Block 2] → 64 filters, 3x3, MaxPool
    ↓
[Conv Block 3] → 128 filters, 3x3, MaxPool
    ↓
[Conv Block 4] → 128 filters, 3x3, MaxPool
    ↓
Flatten → FC(128) → Dropout → FC(64) → Output(3)
    ↓
Softmax → [COVID, Pneumonia, Normal]
```

### Layer 3: Training Layer (`scripts/`)

**Responsibility**: Orchestrating the training process

**Key Components**:
- Training loop implementation
- Validation loop
- Model checkpointing
- Learning rate scheduling
- Early stopping

**Design Decisions**:
- Decoupled from tracking implementation
- Epoch-based training
- Best model selection based on validation accuracy
- Gradient clipping for stability

**Training Pipeline**:
```python
1. Initialize model, optimizer, scheduler
2. For each epoch:
    a. Training phase:
        - Forward pass
        - Loss calculation
        - Backward pass
        - Parameter update
        - Log training metrics
    b. Validation phase:
        - Forward pass (no gradients)
        - Compute metrics
        - Log validation metrics
        - Save best model
    c. Learning rate scheduling
3. Return best model and metrics
```

### Layer 4: Tracking Layer (`src/tracking/`)

**Responsibility**: Experiment tracking and metric logging

**Key Components**:
- `BaseTracker`: Abstract interface
- `MLflowTracker`: MLflow implementation
- `WandBTracker`: Weights & Biases implementation

**Design Decisions**:
- Strategy pattern for pluggable backends
- Consistent interface across platforms
- Comprehensive metric logging
- Artifact management (models, plots)

**Tracked Metrics**:
- Loss (training and validation)
- Accuracy (overall and per-class)
- Precision, Recall, F1 (macro and per-class)
- Confusion matrices
- Learning rate
- Model checkpoints

### Layer 5: Utility Layer (`src/utils/`)

**Responsibility**: Shared utilities and helper functions

**Key Components**:
- Logging configuration
- Metric calculation helpers
- Visualization utilities
- Configuration loaders

## Component Relationships

### Dependency Graph

```
┌──────────────┐
│   Scripts    │  (Executable entry points)
└──────┬───────┘
       │ uses
       ▼
┌──────────────┐
│   Tracking   │  (Experiment tracking)
└──────┬───────┘
       │ uses
       ▼
┌──────────────┐     ┌──────────────┐
│    Models    │◄────┤    Utils     │
└──────┬───────┘     └──────────────┘
       │ uses
       ▼
┌──────────────┐
│     Data     │  (Data loading)
└──────────────┘
```

**Key Relationships**:
1. Scripts depend on all layers (orchestration)
2. Tracking layer is agnostic to model/data details
3. Models depend on data for input shape
4. Utils are used by all layers (no dependencies)

### Interface Contracts

**Model Interface**:
```python
class Model(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (batch_size, 3, H, W)
        Returns:
            Output predictions (batch_size, num_classes)
        """
```

**Tracker Interface**:
```python
class BaseTracker(ABC):
    @abstractmethod
    def start_run(self, experiment_name: str, run_name: str):
        """Initialize tracking run"""

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        """Log scalar metrics"""

    @abstractmethod
    def log_artifact(self, path: str):
        """Log file artifacts"""

    @abstractmethod
    def end_run(self):
        """Finalize tracking run"""
```

## Data Flow

### Training Data Flow

```
1. Dataset Loading
   └─→ COVID19ChestXRayDataset.__init__()
       └─→ Scan directories for images
       └─→ Create image path → label mapping

2. Data Augmentation
   └─→ torchvision.transforms.Compose([
       Resize(128x128),
       RandomHorizontalFlip(),
       RandomRotation(10°),
       ColorJitter(),
       ToTensor(),
       Normalize(ImageNet stats)
   ])

3. Batching
   └─→ DataLoader(
       dataset,
       batch_size=32,
       shuffle=True,
       num_workers=2
   )

4. Training Loop
   └─→ for batch in train_loader:
       ├─→ images, labels = batch
       ├─→ predictions = model(images)
       ├─→ loss = criterion(predictions, labels)
       ├─→ loss.backward()
       ├─→ optimizer.step()
       └─→ tracker.log_metrics({
               'train_loss': loss.item(),
               'train_accuracy': acc
           })

5. Validation Loop
   └─→ with torch.no_grad():
       └─→ for batch in val_loader:
           ├─→ predictions = model(images)
           ├─→ compute metrics
           └─→ tracker.log_metrics({
                   'val_loss': loss,
                   'val_accuracy': acc
               })

6. Model Checkpointing
   └─→ if val_acc > best_val_acc:
       └─→ torch.save(model.state_dict(), 'best_model.pth')
       └─→ tracker.log_artifact('best_model.pth')
```

### Experiment Tracking Flow

```
MLflow Flow:
1. mlflow.start_run(run_name="experiment_1")
2. mlflow.log_params(config)
3. For each epoch:
   └─→ mlflow.log_metrics(metrics, step=epoch)
4. mlflow.log_artifact("model.pth")
5. mlflow.end_run()

W&B Flow:
1. wandb.init(project="covid-xray", name="experiment_1")
2. wandb.config.update(config)
3. For each epoch:
   └─→ wandb.log(metrics, step=epoch)
4. wandb.save("model.pth")
5. wandb.finish()
```

## Technology Stack

### Core Frameworks
- **PyTorch 2.0+**: Deep learning framework
- **torchvision**: Image preprocessing and augmentation
- **NumPy**: Numerical computing
- **scikit-learn**: Metrics and data splitting

### Experiment Tracking
- **MLflow 2.8+**: Experiment tracking, model registry
- **Weights & Biases 0.15+**: Real-time experiment tracking

### Data Management
- **KaggleHub**: Dataset downloading
- **PIL (Pillow)**: Image loading and manipulation

### Visualization
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualizations

### Configuration
- **PyYAML**: YAML configuration parsing

### Development Tools
- **pytest**: Testing framework
- **mypy**: Static type checking
- **black**: Code formatting
- **flake8**: Linting

## Key Architectural Decisions

### Decision 1: PyTorch over TensorFlow
**Rationale**:
- More Pythonic API
- Better debugging experience
- Growing adoption in research community
- Dynamic computation graphs

### Decision 2: YAML Configuration Files
**Rationale**:
- Human-readable
- Easy to version control
- Supports hierarchical configuration
- No code changes for hyperparameter tuning

### Decision 3: Dual Tracking Implementation
**Rationale**:
- Direct comparison of platforms
- Demonstrates platform-agnostic design
- Educational value for students
- Real-world scenario simulation

### Decision 4: Separate Scripts from Library Code
**Rationale**:
- Reusable components in `src/`
- Clear entry points in `scripts/`
- Easier testing
- Better maintainability

### Decision 5: Custom Dataset Class
**Rationale**:
- Flexibility for various folder structures
- Custom augmentation pipeline
- Error handling for corrupted images
- Educational value

## Design Patterns Used

### 1. **Strategy Pattern**
Used in tracking layer to allow interchangeable experiment tracking backends.

### 2. **Factory Pattern**
Used for creating model instances and data loaders based on configuration.

### 3. **Template Method Pattern**
Training loop provides template, subclasses implement specific tracking logic.

### 4. **Singleton Pattern**
Configuration objects are loaded once and reused.

### 5. **Observer Pattern**
Metrics are logged to multiple observers (console, file, tracking platform).

## Scalability Considerations

### Current Architecture Supports:
- ✅ Adding new model architectures
- ✅ Adding new tracking platforms
- ✅ Adding new datasets
- ✅ Hyperparameter tuning
- ✅ Multi-GPU training (with minor modifications)

### Future Enhancements:
- 🔄 Distributed training across multiple nodes
- 🔄 Real-time inference pipeline
- 🔄 Model serving with MLflow
- 🔄 Automated hyperparameter optimization (Optuna/Ray Tune)
- 🔄 Data versioning with DVC

## References

- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/best_practices.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [W&B Documentation](https://docs.wandb.ai/)
- [Clean Code by Robert C. Martin](https://www.oreilly.com/library/view/clean-code-a/9780136083238/)
- [Design Patterns: Elements of Reusable Object-Oriented Software](https://en.wikipedia.org/wiki/Design_Patterns)
