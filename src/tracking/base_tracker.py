"""
Base Tracker Abstract Class
Defines the interface for experiment tracking platforms
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path


class BaseTracker(ABC):
    """
    Abstract base class for experiment tracking.

    This class defines the interface that all tracking implementations
    (MLflow, W&B, Neptune, etc.) must follow. Using this pattern allows
    for:
    - Easy switching between tracking platforms
    - Consistent API across different backends
    - Better testability with mock trackers
    - Following Dependency Inversion Principle

    Example:
        >>> class MyTracker(BaseTracker):
        ...     def start_run(self, experiment_name, run_name, config):
        ...         # Implementation
        ...         pass
        ...
        >>> tracker = MyTracker()
        >>> tracker.start_run("my_experiment", "run_001")
    """

    def __init__(self, experiment_name: str = "default_experiment"):
        """
        Initialize base tracker.

        Args:
            experiment_name: Name of the experiment/project
        """
        self.experiment_name = experiment_name
        self._is_running = False

    @abstractmethod
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a new tracking run.

        Args:
            run_name: Optional name for this specific run
            tags: Optional dictionary of tags for categorization
            config: Optional configuration/hyperparameters to log

        Raises:
            RuntimeError: If a run is already active
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters or configuration parameters.

        Args:
            params: Dictionary of parameter names and values

        Raises:
            RuntimeError: If no run is active

        Example:
            >>> tracker.log_params({
            ...     'learning_rate': 0.001,
            ...     'batch_size': 32,
            ...     'optimizer': 'Adam'
            ... })
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics at a specific step (epoch/iteration).

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (epoch/iteration)

        Raises:
            RuntimeError: If no run is active

        Example:
            >>> for epoch in range(10):
            ...     metrics = {'loss': 0.5, 'accuracy': 0.85}
            ...     tracker.log_metrics(metrics, step=epoch)
        """
        pass

    @abstractmethod
    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: Optional[str] = None
    ) -> None:
        """
        Log a file artifact (model, plot, data file, etc.).

        Args:
            artifact_path: Path to the file to log
            artifact_type: Optional type/category of artifact

        Raises:
            FileNotFoundError: If artifact file doesn't exist
            RuntimeError: If no run is active

        Example:
            >>> tracker.log_artifact('model.pth', artifact_type='model')
            >>> tracker.log_artifact('confusion_matrix.png', artifact_type='plot')
        """
        pass

    @abstractmethod
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model"
    ) -> None:
        """
        Log a trained model.

        Args:
            model: Model object to log (e.g., PyTorch nn.Module)
            artifact_path: Path/name for the model artifact

        Raises:
            RuntimeError: If no run is active

        Example:
            >>> tracker.log_model(model, artifact_path="best_model")
        """
        pass

    @abstractmethod
    def end_run(self) -> None:
        """
        Finalize and end the current tracking run.

        Raises:
            RuntimeError: If no run is active

        Example:
            >>> tracker.start_run("experiment_001")
            >>> # ... training code ...
            >>> tracker.end_run()
        """
        pass

    @abstractmethod
    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag for the current run.

        Args:
            key: Tag key
            value: Tag value

        Raises:
            RuntimeError: If no run is active

        Example:
            >>> tracker.set_tag("model_version", "v1.0")
            >>> tracker.set_tag("experiment_type", "baseline")
        """
        pass

    def log_batch_metrics(
        self,
        metrics_list: List[Dict[str, float]],
        start_step: int = 0
    ) -> None:
        """
        Log multiple metrics at once (batch logging).

        This is a convenience method that calls log_metrics multiple times.
        Subclasses can override for more efficient batch operations.

        Args:
            metrics_list: List of metric dictionaries
            start_step: Starting step number

        Example:
            >>> metrics_list = [
            ...     {'loss': 0.5, 'acc': 0.8},
            ...     {'loss': 0.4, 'acc': 0.85},
            ...     {'loss': 0.3, 'acc': 0.9}
            ... ]
            >>> tracker.log_batch_metrics(metrics_list, start_step=0)
        """
        for i, metrics in enumerate(metrics_list):
            self.log_metrics(metrics, step=start_step + i)

    def log_dict(
        self,
        data: Dict[str, Any],
        prefix: str = ""
    ) -> None:
        """
        Log a nested dictionary as parameters.

        Flattens nested dictionaries using dot notation.

        Args:
            data: Dictionary to log
            prefix: Optional prefix for keys

        Example:
            >>> config = {
            ...     'model': {'layers': 4, 'hidden_size': 128},
            ...     'training': {'lr': 0.001, 'epochs': 10}
            ... }
            >>> tracker.log_dict(config)
            # Logs: model.layers=4, model.hidden_size=128, etc.
        """
        flattened = self._flatten_dict(data, prefix)
        self.log_params(flattened)

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "."
    ) -> Dict[str, Any]:
        """
        Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    self._flatten_dict(v, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)

    @property
    def is_running(self) -> bool:
        """Check if a run is currently active."""
        return self._is_running

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._is_running:
            self.end_run()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"experiment='{self.experiment_name}', "
            f"running={self._is_running})"
        )
