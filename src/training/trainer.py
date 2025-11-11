"""
Trainer class implementing training logic independent of tracking systems.

This module demonstrates the Single Responsibility Principle and Dependency Injection.
The Trainer focuses solely on training models, while tracking is delegated to
injected tracker implementations.
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Import base tracker (abstract interface)
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tracking.base_tracker import BaseTracker


# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for the training process.

    This dataclass encapsulates all training-related hyperparameters,
    making them explicit, typed, and validated.

    Why use a dataclass?
        - Type hints for IDE support
        - Automatic __init__, __repr__, __eq__
        - Validation via __post_init__
        - Immutability option
        - Self-documenting code

    Example:
        >>> config = TrainingConfig(
        ...     num_epochs=50,
        ...     learning_rate=0.0001,
        ...     early_stopping_patience=10
        ... )
    """

    # Basic training parameters
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0

    # Device configuration
    device: str = field(
        default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_step_size: int = 7
    scheduler_gamma: float = 0.1

    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Optional[str] = "checkpoints"

    # Logging
    log_interval: int = 10  # Log every N batches

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be positive, got {self.early_stopping_patience}"
            )


class Trainer:
    """
    Handles model training independent of experiment tracking system.

    This class demonstrates several software engineering principles:

    1. **Single Responsibility Principle (SRP)**:
       - Only responsible for training models
       - Tracking delegated to BaseTracker implementations

    2. **Dependency Inversion Principle (DIP)**:
       - Depends on BaseTracker abstraction, not concrete implementations
       - Can work with any tracker (MLflow, W&B, custom, or none)

    3. **Open/Closed Principle (OCP)**:
       - Open for extension (new trackers) without modification
       - Core training logic doesn't change when adding trackers

    4. **Dependency Injection**:
       - All dependencies provided via constructor
       - Easy to test with mocks
       - Flexible configuration

    Example:
        >>> # With MLflow
        >>> mlflow_tracker = MLflowTracker("my-experiment")
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=config,
        ...     tracker=mlflow_tracker
        ... )
        >>>
        >>> # With W&B
        >>> wandb_tracker = WandBTracker("my-project")
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=config,
        ...     tracker=wandb_tracker  # Just swap the tracker!
        ... )
        >>>
        >>> # Without tracking
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=config,
        ...     tracker=None  # No tracking
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        tracker: Optional[BaseTracker] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        class_names: Optional[list] = None
    ):
        """
        Initialize the trainer.

        All dependencies are injected via constructor (Dependency Injection pattern).

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration
            tracker: Optional experiment tracker (BaseTracker interface)
            optimizer: Optional optimizer (defaults to Adam)
            criterion: Optional loss function (defaults to CrossEntropyLoss)
            class_names: Optional list of class names for reporting

        Note:
            The tracker parameter accepts any BaseTracker implementation.
            This is polymorphism in action - we don't care about the specific
            tracker type, only that it implements the BaseTracker interface.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.tracker = tracker
        self.class_names = class_names or []

        # Move model to device
        self.model = model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer (default to Adam if not provided)
        self.optimizer = optimizer or optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")

        # Loss function (default to CrossEntropyLoss)
        self.criterion = criterion or nn.CrossEntropyLoss()
        logger.info(f"Using loss function: {self.criterion.__class__.__name__}")

        # Learning rate scheduler
        self.scheduler = None
        if config.use_scheduler:
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.scheduler_step_size,
                gamma=config.scheduler_gamma
            )
            logger.info(
                f"Using StepLR scheduler: step_size={config.scheduler_step_size}, "
                f"gamma={config.scheduler_gamma}"
            )

        # Early stopping state
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.best_model_state = None

        # Checkpoint directory
        if config.save_checkpoints and config.checkpoint_dir:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        # Set random seed for reproducibility
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary containing training metrics (loss, accuracy)

        Process:
            1. Set model to training mode
            2. Iterate through batches
            3. Forward pass
            4. Compute loss
            5. Backward pass
            6. Update weights
            7. Track metrics
        """
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Note: model returns logits, not probabilities (after refactoring)
            logits = self.model(images)

            # Compute loss
            # CrossEntropyLoss expects logits, not probabilities
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log progress periodically
            if (batch_idx + 1) % self.config.log_interval == 0:
                batch_loss = loss.item()
                batch_acc = 100.0 * correct / total
                logger.debug(
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {batch_loss:.4f} Acc: {batch_acc:.2f}%"
                )

        # Calculate epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

        The @torch.no_grad() decorator disables gradient computation,
        saving memory and speeding up inference.

        Returns:
            Dictionary containing validation metrics (loss, accuracy)
        """
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(images)

            # Compute loss
            loss = self.criterion(logits, labels)

            # Track metrics
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on test set with detailed metrics.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dictionary containing:
                - test_loss: Average loss
                - test_accuracy: Overall accuracy
                - per_class_metrics: Per-class precision, recall, f1
                - confusion_matrix: Confusion matrix
                - classification_report: Detailed sklearn report
        """
        self.model.eval()

        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []

        for images, labels in test_loader:
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(images)

            # Compute loss
            loss = self.criterion(logits, labels)
            running_loss += loss.item()

            # Get predictions
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            # Store for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

        # Calculate metrics
        avg_loss = running_loss / len(test_loader)
        accuracy = 100.0 * np.mean(np.array(all_predictions) == np.array(all_labels))

        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Classification report
        target_names = self.class_names if self.class_names else None
        class_report = classification_report(
            all_labels,
            all_predictions,
            target_names=target_names,
            output_dict=True
        )

        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }

    def _should_stop_early(self, val_loss: float, val_accuracy: float) -> bool:
        """
        Check if early stopping criteria are met.

        Early stopping prevents overfitting by monitoring validation performance
        and stopping when it stops improving.

        Args:
            val_loss: Current validation loss
            val_accuracy: Current validation accuracy

        Returns:
            True if training should stop

        Logic:
            - Track best validation loss
            - If current loss is better (by min_delta), reset patience counter
            - If not, increment patience counter
            - Stop if patience counter exceeds patience threshold
        """
        # Check if this is the best model so far
        improvement = self.best_val_loss - val_loss

        if improvement > self.config.early_stopping_min_delta:
            # Model improved
            self.best_val_loss = val_loss
            self.best_val_accuracy = val_accuracy
            self.epochs_without_improvement = 0

            # Save best model state
            self.best_model_state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }

            logger.info(
                f"✓ New best model: val_loss={val_loss:.4f}, "
                f"val_acc={val_accuracy:.2f}%"
            )

            return False

        else:
            # No improvement
            self.epochs_without_improvement += 1
            logger.info(
                f"No improvement for {self.epochs_without_improvement} epoch(s). "
                f"Best val_loss: {self.best_val_loss:.4f}"
            )

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {self.epochs_without_improvement} "
                    f"epochs without improvement"
                )
                return True

            return False

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Current metrics
        """
        if not self.config.save_checkpoints:
            return

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def train(self) -> Dict[str, Any]:
        """
        Full training loop with early stopping and checkpointing.

        This is the main training method that orchestrates the entire process.

        Returns:
            Dictionary containing training history and final metrics

        Process:
            1. Log hyperparameters to tracker (if available)
            2. For each epoch:
               a. Train for one epoch
               b. Validate
               c. Log metrics to tracker
               d. Check early stopping
               e. Update learning rate (if using scheduler)
               f. Save checkpoints
            3. Restore best model
            4. Return results

        Note:
            The tracker is optional. If None, training proceeds without tracking.
            This demonstrates the Dependency Inversion Principle - we don't
            depend on a specific tracker, and can even work without one.
        """
        logger.info("=" * 70)
        logger.info("Starting training")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Batch size: {self.train_loader.batch_size}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info("=" * 70)

        # Log hyperparameters to tracker (if available)
        if self.tracker:
            params = {
                'learning_rate': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'num_epochs': self.config.num_epochs,
                'batch_size': self.train_loader.batch_size,
                'optimizer': self.optimizer.__class__.__name__,
                'criterion': self.criterion.__class__.__name__,
                'device': str(self.device),
                'use_scheduler': self.config.use_scheduler,
                'seed': self.config.seed
            }

            if self.config.use_scheduler:
                params['scheduler_step_size'] = self.config.scheduler_step_size
                params['scheduler_gamma'] = self.config.scheduler_gamma

            self.tracker.log_params(params)

        # Training loop
        training_history = []

        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch [{epoch + 1}/{self.config.num_epochs}]")
            logger.info("-" * 70)

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Combine metrics
            epoch_metrics = {
                **train_metrics,
                **val_metrics,
                'epoch': epoch + 1,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }

            training_history.append(epoch_metrics)

            # Log metrics
            logger.info(
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
            )

            # Log to tracker (if available)
            if self.tracker:
                self.tracker.log_metrics(epoch_metrics, step=epoch)

            # Early stopping check
            if self.config.use_early_stopping:
                should_stop = self._should_stop_early(
                    val_metrics['val_loss'],
                    val_metrics['val_accuracy']
                )

                if should_stop:
                    logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Save checkpoint
            self._save_checkpoint(epoch + 1, epoch_metrics)

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            logger.info("\n✓ Restored best model from training")
            logger.info(
                f"Best val_loss: {self.best_val_loss:.4f}, "
                f"Best val_acc: {self.best_val_accuracy:.2f}%"
            )

        logger.info("=" * 70)
        logger.info("Training completed")
        logger.info("=" * 70)

        return {
            'history': training_history,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy
        }

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Example:
            >>> trainer.load_checkpoint("checkpoints/checkpoint_epoch_10.pt")
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint.get('epoch', 'unknown')}")

        if 'metrics' in checkpoint:
            logger.info(f"Metrics: {checkpoint['metrics']}")
