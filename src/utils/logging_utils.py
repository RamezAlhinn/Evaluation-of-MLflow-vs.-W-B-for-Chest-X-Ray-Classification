"""
Logging Utilities
Provides centralized logging configuration for the project
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logger(
    name: str = __name__,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_colors: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name (typically __name__ of the module)
        log_file: Optional path to log file
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        use_colors: Whether to use colored output for console

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger('training', log_file='logs/train.log')
        >>> logger.info('Training started')
        >>> logger.debug('Debug information')
        >>> logger.error('An error occurred')
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatters
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if use_colors:
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger('training')
        >>> logger.info('Message')
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Specialized logger for training progress.

    Provides formatted output for training metrics and progress tracking.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize training logger.

        Args:
            logger: Base logger instance
        """
        self.logger = logger

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log the start of an epoch."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch [{epoch+1}/{total_epochs}]")
        self.logger.info(f"{'='*60}")

    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float
    ):
        """
        Log epoch results.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            lr: Current learning rate
        """
        self.logger.info(f"Epoch {epoch+1} Results:")
        self.logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        self.logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        self.logger.info(f"  Learning Rate: {lr:.6f}")

    def log_batch_progress(
        self,
        batch_idx: int,
        total_batches: int,
        loss: float,
        interval: int = 10
    ):
        """
        Log batch training progress.

        Args:
            batch_idx: Current batch index
            total_batches: Total number of batches
            loss: Current batch loss
            interval: Log every N batches
        """
        if (batch_idx + 1) % interval == 0:
            progress = (batch_idx + 1) / total_batches * 100
            self.logger.info(
                f"  Batch [{batch_idx+1}/{total_batches}] "
                f"({progress:.1f}%) | Loss: {loss:.4f}"
            )

    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """
        Log best model checkpoint.

        Args:
            epoch: Epoch where best model was found
            metric_name: Name of the metric
            metric_value: Value of the metric
        """
        self.logger.info(f"\n🎯 New best model!")
        self.logger.info(f"  Epoch: {epoch+1}")
        self.logger.info(f"  {metric_name}: {metric_value:.4f}")

    def log_training_complete(
        self,
        total_epochs: int,
        best_metric: float,
        training_time: float
    ):
        """
        Log training completion summary.

        Args:
            total_epochs: Total number of epochs trained
            best_metric: Best metric achieved
            training_time: Total training time in seconds
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info("✅ Training Complete!")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  Total Epochs: {total_epochs}")
        self.logger.info(f"  Best Validation Accuracy: {best_metric:.2f}%")
        self.logger.info(f"  Total Training Time: {training_time/60:.2f} minutes")

    def log_error(self, error: Exception):
        """
        Log an error with traceback.

        Args:
            error: Exception that occurred
        """
        self.logger.error(f"\n❌ Error occurred: {str(error)}", exc_info=True)


def create_training_logger(
    name: str = "training",
    log_dir: str = "logs"
) -> TrainingLogger:
    """
    Create a training logger with automatic log file naming.

    Args:
        name: Logger name
        log_dir: Directory for log files

    Returns:
        TrainingLogger instance

    Example:
        >>> logger = create_training_logger('training', 'logs')
        >>> logger.log_epoch_start(0, 10)
        >>> logger.log_epoch_end(0, 0.5, 85.3, 0.4, 87.1, 0.001)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{name}_{timestamp}.log"

    base_logger = setup_logger(name, log_file=log_file)
    return TrainingLogger(base_logger)


# Global logger instance for convenience
logger = setup_logger('covid_xray_classification')


if __name__ == '__main__':
    # Test the logging functionality
    logger = setup_logger('test', log_file='logs/test.log')

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test training logger
    train_logger = create_training_logger('training_test')
    train_logger.log_epoch_start(0, 10)
    train_logger.log_batch_progress(9, 100, 0.5, interval=10)
    train_logger.log_epoch_end(0, 0.5, 85.3, 0.4, 87.1, 0.001)
    train_logger.log_best_model(0, "val_accuracy", 87.1)
    train_logger.log_training_complete(10, 90.5, 3600)
