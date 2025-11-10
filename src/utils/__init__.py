"""
Utility functions for the project.

This module provides comprehensive utilities for:
- Logging: Professional logging with colored output
- Metrics: Calculation and tracking of model performance metrics
- Configuration: YAML/JSON config management with validation
"""

from src.utils.logging_utils import (
    setup_logger,
    get_logger,
    TrainingLogger,
    create_training_logger,
    logger
)

from src.utils.metrics import (
    calculate_accuracy,
    calculate_metrics,
    calculate_confusion_matrix,
    get_classification_report,
    MetricsTracker,
    AverageMeter,
    evaluate_model
)

from src.utils.config import (
    Config,
    load_config,
    save_config,
    merge_configs,
    validate_config,
    ConfigManager
)

__all__ = [
    # Logging
    'setup_logger',
    'get_logger',
    'TrainingLogger',
    'create_training_logger',
    'logger',

    # Metrics
    'calculate_accuracy',
    'calculate_metrics',
    'calculate_confusion_matrix',
    'get_classification_report',
    'MetricsTracker',
    'AverageMeter',
    'evaluate_model',

    # Configuration
    'Config',
    'load_config',
    'save_config',
    'merge_configs',
    'validate_config',
    'ConfigManager',
]
