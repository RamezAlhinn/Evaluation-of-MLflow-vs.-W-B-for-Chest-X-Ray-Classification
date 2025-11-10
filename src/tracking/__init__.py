"""
Experiment tracking integrations.

This module provides a unified interface for experiment tracking across
different platforms (MLflow, W&B, Neptune, etc.) using the Strategy Pattern.

The BaseTracker abstract class defines the interface that all tracking
implementations must follow, ensuring consistency and interchangeability.
"""

from src.tracking.base_tracker import BaseTracker
from src.tracking.mlflow_tracker import (
    MLflowTracker,
    train_with_mlflow,
    evaluate_with_mlflow
)
from src.tracking.wandb_tracker import (
    WandBTracker,
    train_with_wandb,
    evaluate_with_wandb
)

__all__ = [
    'BaseTracker',
    'MLflowTracker',
    'train_with_mlflow',
    'evaluate_with_mlflow',
    'WandBTracker',
    'train_with_wandb',
    'evaluate_with_wandb'
]
