"""
Base tracker interface for experiment tracking systems.

This module defines the abstract interface that all tracker implementations
must follow, enabling polymorphism and dependency inversion.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch.nn as nn


class BaseTracker(ABC):
    """
    Abstract base class for experiment tracking systems.

    This defines the contract that all tracker implementations (MLflow, W&B, etc.)
    must follow. By programming to this interface instead of concrete implementations,
    we achieve:

    1. Dependency Inversion: High-level code depends on abstraction, not concrete trackers
    2. Open/Closed Principle: Can add new trackers without modifying existing code
    3. Liskov Substitution: Any BaseTracker can be swapped for another
    4. Testability: Easy to create mock trackers for testing

    Example:
        >>> tracker = MLflowTracker(experiment_name="my-experiment")
        >>> tracker.start_run(run_name="run-1")
        >>> tracker.log_params({"learning_rate": 0.001})
        >>> tracker.log_metrics({"accuracy": 0.95}, step=10)
        >>> tracker.end_run()
    """

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None, **kwargs) -> None:
        """
        Start a new tracking run.

        A 'run' represents a single execution of a training experiment.
        All subsequent logging calls will be associated with this run.

        Args:
            run_name: Optional name for the run. If None, tracker should generate one.
            **kwargs: Additional tracker-specific parameters.

        Raises:
            RuntimeError: If a run is already active.

        Example:
            >>> tracker.start_run(run_name="baseline-model")
        """
        pass

    @abstractmethod
    def end_run(self) -> None:
        """
        End the current tracking run and cleanup resources.

        This should be called when training/evaluation is complete.
        Use in a try-finally block to ensure cleanup even if errors occur.

        Raises:
            RuntimeError: If no run is currently active.

        Example:
            >>> tracker.start_run()
            >>> try:
            ...     # training code
            ...     tracker.log_metrics(metrics)
            ... finally:
            ...     tracker.end_run()
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters and configuration.

        Parameters are typically logged once at the start of a run and
        represent the configuration that doesn't change during training.

        Args:
            params: Dictionary mapping parameter names to values.
                   Values can be any JSON-serializable type.

        Raises:
            RuntimeError: If no run is currently active.

        Example:
            >>> tracker.log_params({
            ...     "learning_rate": 0.001,
            ...     "batch_size": 32,
            ...     "optimizer": "Adam",
            ...     "model_type": "CNN"
            ... })

        Note:
            Some trackers may have limitations on parameter types or lengths.
            Keep parameter names descriptive but concise.
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics during or after training.

        Metrics are values that change during training (loss, accuracy, etc.).
        They can be logged multiple times at different steps/epochs.

        Args:
            metrics: Dictionary mapping metric names to float values.
            step: Optional step number (epoch, batch number, etc.).
                 Used to plot metrics over time.

        Raises:
            RuntimeError: If no run is currently active.

        Example:
            >>> # Log training metrics
            >>> tracker.log_metrics({
            ...     "train_loss": 0.5,
            ...     "train_accuracy": 0.85
            ... }, step=1)
            >>>
            >>> # Log validation metrics
            >>> tracker.log_metrics({
            ...     "val_loss": 0.6,
            ...     "val_accuracy": 0.82
            ... }, step=1)

        Note:
            Use consistent metric names across runs for easy comparison.
            Prefix with "train_", "val_", "test_" for clarity.
        """
        pass

    @abstractmethod
    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        **kwargs
    ) -> None:
        """
        Log a trained PyTorch model.

        Saves the model so it can be loaded later for inference or
        continued training.

        Args:
            model: PyTorch model to log.
            artifact_path: Path/name for the saved model artifact.
            **kwargs: Additional tracker-specific parameters (e.g., model signature).

        Raises:
            RuntimeError: If no run is currently active.

        Example:
            >>> tracker.log_model(model, artifact_path="final_model")

        Note:
            Different trackers may save models in different formats:
            - MLflow: Uses MLflow's PyTorch format
            - W&B: Saves state dict and creates artifact
            Ensure you use the tracker's loading mechanism to restore models.
        """
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log a file or directory as an artifact.

        This is an optional method with a default no-op implementation.
        Trackers can override to support artifact logging.

        Args:
            local_path: Local file or directory path to log.
            artifact_path: Optional path within artifact store.

        Example:
            >>> tracker.log_artifact("confusion_matrix.png", "images/")
            >>> tracker.log_artifact("config.yaml")
        """
        pass

    def log_figure(self, figure: Any, artifact_path: str) -> None:
        """
        Log a matplotlib figure or similar visualization.

        This is an optional method with a default no-op implementation.
        Trackers can override to support figure logging.

        Args:
            figure: Matplotlib figure or similar visualization object.
            artifact_path: Path/name for the saved figure.

        Example:
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots()
            >>> ax.plot([1, 2, 3], [4, 5, 6])
            >>> tracker.log_figure(fig, "training_curve.png")
        """
        pass

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for the current run.

        Tags are key-value pairs used for organizing and filtering runs.
        Unlike params, tags are typically used for categorization.

        This is an optional method with a default no-op implementation.

        Args:
            tags: Dictionary of tag names and values.

        Example:
            >>> tracker.set_tags({
            ...     "model_type": "cnn",
            ...     "dataset": "chest-xray",
            ...     "experiment_phase": "baseline"
            ... })
        """
        pass


class DummyTracker(BaseTracker):
    """
    Dummy tracker that logs to stdout.

    Useful for testing or when you want to train without a tracking system.
    Implements BaseTracker interface but just prints instead of tracking.

    Example:
        >>> tracker = DummyTracker()
        >>> tracker.start_run("test")
        >>> tracker.log_params({"lr": 0.001})
        >>> tracker.log_metrics({"loss": 0.5}, step=1)
        >>> tracker.end_run()
    """

    def __init__(self):
        """Initialize dummy tracker."""
        self.active_run = False

    def start_run(self, run_name: Optional[str] = None, **kwargs) -> None:
        """Start dummy run."""
        print(f"[DummyTracker] Starting run: {run_name}")
        self.active_run = True

    def end_run(self) -> None:
        """End dummy run."""
        print("[DummyTracker] Ending run")
        self.active_run = False

    def log_params(self, params: Dict[str, Any]) -> None:
        """Print parameters."""
        print(f"[DummyTracker] Parameters: {params}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Print metrics."""
        step_str = f" (step {step})" if step is not None else ""
        print(f"[DummyTracker] Metrics{step_str}: {metrics}")

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        **kwargs
    ) -> None:
        """Print model logging."""
        print(f"[DummyTracker] Logging model to: {artifact_path}")
