"""
Metrics Utilities
Provides functions for calculating and tracking model performance metrics
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy as a float between 0 and 1

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1])
        >>> y_pred = np.array([0, 1, 2, 1, 1])
        >>> acc = calculate_accuracy(y_true, y_pred)
        >>> print(f"Accuracy: {acc:.2%}")
    """
    return accuracy_score(y_true, y_pred)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names for per-class metrics
        average: Averaging method ('macro', 'micro', 'weighted')

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
            - Per-class metrics if class_names is provided

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1])
        >>> y_pred = np.array([0, 1, 2, 1, 1])
        >>> class_names = ['COVID', 'Pneumonia', 'Normal']
        >>> metrics = calculate_metrics(y_true, y_pred, class_names)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        >>> print(f"F1 Score: {metrics['f1']:.4f}")
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Per-class metrics
    if class_names is not None:
        per_class_precision, per_class_recall, per_class_f1, _ = \
            precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name}'] = per_class_precision[i]
            metrics[f'recall_{class_name}'] = per_class_recall[i]
            metrics[f'f1_{class_name}'] = per_class_f1[i]

    return metrics


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Calculate confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Confusion matrix as numpy array

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1])
        >>> y_pred = np.array([0, 1, 2, 1, 1])
        >>> cm = calculate_confusion_matrix(y_true, y_pred)
        >>> print(cm)
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Get detailed classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Classification report as string

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1])
        >>> y_pred = np.array([0, 1, 2, 1, 1])
        >>> class_names = ['COVID', 'Pneumonia', 'Normal']
        >>> report = get_classification_report(y_true, y_pred, class_names)
        >>> print(report)
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )


class MetricsTracker:
    """
    Track metrics over multiple epochs/steps.

    Useful for monitoring training progress and creating plots.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.steps = []

    def add_metrics(self, metrics: Dict[str, float], step: int):
        """
        Add metrics for a specific step.

        Args:
            metrics: Dictionary of metric names and values
            step: Step number (epoch/iteration)
        """
        self.steps.append(step)

        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_metric(self, name: str) -> List[float]:
        """
        Get all values for a specific metric.

        Args:
            name: Metric name

        Returns:
            List of metric values

        Raises:
            KeyError: If metric name doesn't exist
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found")
        return self.metrics[name]

    def get_best(
        self,
        metric_name: str,
        mode: str = 'max'
    ) -> Tuple[int, float]:
        """
        Get the best value and step for a metric.

        Args:
            metric_name: Name of the metric
            mode: 'max' for highest value, 'min' for lowest value

        Returns:
            Tuple of (step, value)

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.add_metrics({'loss': 0.5, 'acc': 0.8}, 0)
            >>> tracker.add_metrics({'loss': 0.4, 'acc': 0.85}, 1)
            >>> best_step, best_acc = tracker.get_best('acc', mode='max')
            >>> print(f"Best accuracy {best_acc:.2%} at step {best_step}")
        """
        values = self.get_metric(metric_name)

        if mode == 'max':
            best_idx = np.argmax(values)
        elif mode == 'min':
            best_idx = np.argmin(values)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'max' or 'min'.")

        return self.steps[best_idx], values[best_idx]

    def get_latest(self, metric_name: str) -> float:
        """
        Get the latest value for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Latest metric value
        """
        values = self.get_metric(metric_name)
        return values[-1]

    def get_average(
        self,
        metric_name: str,
        last_n: Optional[int] = None
    ) -> float:
        """
        Get average value for a metric.

        Args:
            metric_name: Name of the metric
            last_n: If provided, average over last N values

        Returns:
            Average metric value

        Example:
            >>> tracker.add_metrics({'loss': 0.5}, 0)
            >>> tracker.add_metrics({'loss': 0.4}, 1)
            >>> tracker.add_metrics({'loss': 0.3}, 2)
            >>> avg_loss = tracker.get_average('loss', last_n=2)
            >>> print(f"Average of last 2: {avg_loss:.4f}")  # 0.35
        """
        values = self.get_metric(metric_name)

        if last_n is not None:
            values = values[-last_n:]

        return np.mean(values)

    def to_dict(self) -> Dict[str, List[float]]:
        """
        Convert tracker data to dictionary.

        Returns:
            Dictionary with all metrics and steps
        """
        return {
            'steps': self.steps,
            **self.metrics
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MetricsTracker("
            f"metrics={list(self.metrics.keys())}, "
            f"steps={len(self.steps)})"
        )


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics like loss during training.
    """

    def __init__(self, name: str = "metric"):
        """
        Initialize average meter.

        Args:
            name: Name of the metric being tracked
        """
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics.

        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.

    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader with evaluation data
        criterion: Loss function
        device: Device to evaluate on
        class_names: Optional list of class names

    Returns:
        Dictionary containing evaluation metrics

    Example:
        >>> model = CustomCXRClassifier()
        >>> test_loader = DataLoader(test_dataset, batch_size=32)
        >>> criterion = nn.CrossEntropyLoss()
        >>> device = torch.device('cuda')
        >>> metrics = evaluate_model(model, test_loader, criterion, device)
        >>> print(f"Test Accuracy: {metrics['accuracy']:.2%}")
    """
    model.eval()
    all_preds = []
    all_labels = []
    loss_meter = AverageMeter('loss')

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update loss
            loss_meter.update(loss.item(), images.size(0))

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = calculate_metrics(y_true, y_pred, class_names)
    metrics['loss'] = loss_meter.avg

    return metrics


if __name__ == '__main__':
    # Test metrics functions
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 2, 0, 2, 1])
    class_names = ['COVID', 'Pneumonia', 'Normal']

    # Test calculate_metrics
    metrics = calculate_metrics(y_true, y_pred, class_names)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    # Test classification report
    report = get_classification_report(y_true, y_pred, class_names)
    print(f"\nClassification Report:\n{report}")

    # Test MetricsTracker
    print("\nMetricsTracker:")
    tracker = MetricsTracker()
    for i in range(5):
        tracker.add_metrics({
            'loss': 0.5 - i * 0.1,
            'accuracy': 0.7 + i * 0.05
        }, step=i)

    best_step, best_acc = tracker.get_best('accuracy', mode='max')
    print(f"Best accuracy: {best_acc:.4f} at step {best_step}")

    # Test AverageMeter
    print("\nAverageMeter:")
    meter = AverageMeter('loss')
    for loss in [0.5, 0.4, 0.3, 0.35]:
        meter.update(loss)
        print(meter)
