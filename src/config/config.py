"""
Configuration management system with validation.

This module provides type-safe, validated configuration using Python dataclasses.
Benefits:
- Type hints for IDE support and static analysis
- Automatic validation via __post_init__
- Support for environment variables
- Hierarchical configuration structure
- Default values with sensible fallbacks
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import os
import yaml
import torch


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.

    Environment variables:
        DATASET_PATH: Override dataset_path

    Example:
        >>> config = DataConfig(
        ...     dataset_path="/data/chest-xray",
        ...     batch_size=64,
        ...     image_size=224
        ... )
    """

    dataset_path: str = field(
        default_factory=lambda: os.getenv("DATASET_PATH", "Covid19-dataset")
    )
    batch_size: int = 32
    image_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True

    # Dataset split ratios
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Data augmentation parameters
    random_horizontal_flip_prob: float = 0.5
    random_rotation_degrees: int = 10
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2

    # Normalization (ImageNet stats by default)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    def __post_init__(self):
        """
        Validate configuration after initialization.

        This method is automatically called by dataclass after __init__.
        It catches configuration errors early, before they cause runtime failures.
        """
        # Validate splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(
                f"Data splits must sum to 1.0, got {total_split} "
                f"(train={self.train_split}, val={self.val_split}, test={self.test_split})"
            )

        # Validate individual splits are in valid range
        for name, value in [
            ("train_split", self.train_split),
            ("val_split", self.val_split),
            ("test_split", self.test_split)
        ]:
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")

        # Validate batch size
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # Validate image size (common sizes for CNNs)
        valid_sizes = [64, 128, 224, 256, 512]
        if self.image_size not in valid_sizes:
            raise ValueError(
                f"image_size should be one of {valid_sizes}, got {self.image_size}"
            )

        # Validate num_workers
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

        # Check dataset path exists (warning only, might not exist yet)
        dataset_path = Path(self.dataset_path)
        if not dataset_path.exists():
            import warnings
            warnings.warn(
                f"Dataset path does not exist: {self.dataset_path}. "
                f"Ensure it exists before training."
            )

        # Validate augmentation parameters
        if not 0 <= self.random_horizontal_flip_prob <= 1:
            raise ValueError(
                f"random_horizontal_flip_prob must be in [0, 1], "
                f"got {self.random_horizontal_flip_prob}"
            )

        if self.random_rotation_degrees < 0:
            raise ValueError(
                f"random_rotation_degrees must be non-negative, "
                f"got {self.random_rotation_degrees}"
            )

        # Validate normalization parameters
        if len(self.normalize_mean) != 3 or len(self.normalize_std) != 3:
            raise ValueError(
                "normalize_mean and normalize_std must have 3 values (RGB channels)"
            )


@dataclass
class ModelConfig:
    """
    Configuration for model architecture.

    Example:
        >>> config = ModelConfig(
        ...     num_classes=4,
        ...     dropout_rates=[0.3, 0.4, 0.4, 0.3]
        ... )
    """

    input_channels: int = 3
    num_classes: int = 3
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.3, 0.4, 0.25])

    def __post_init__(self):
        """Validate model configuration."""
        # Validate num_classes
        if self.num_classes < 2:
            raise ValueError(
                f"num_classes must be at least 2 for classification, got {self.num_classes}"
            )

        # Validate input_channels
        if self.input_channels not in [1, 3, 4]:
            raise ValueError(
                f"input_channels should typically be 1 (grayscale), 3 (RGB), or 4 (RGBA), "
                f"got {self.input_channels}"
            )

        # Validate dropout rates
        if len(self.dropout_rates) != 4:
            raise ValueError(
                f"dropout_rates must have 4 values (one per conv block), "
                f"got {len(self.dropout_rates)}"
            )

        for i, rate in enumerate(self.dropout_rates):
            if not 0 <= rate < 1:
                raise ValueError(
                    f"dropout_rates[{i}] must be in [0, 1), got {rate}"
                )


@dataclass
class TrainingConfig:
    """
    Configuration for training process.

    Environment variables:
        DEVICE: Override device selection

    Example:
        >>> config = TrainingConfig(
        ...     num_epochs=50,
        ...     learning_rate=0.0001,
        ...     early_stopping_patience=10
        ... )
    """

    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    device: str = field(
        default_factory=lambda: os.getenv(
            "DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_step_size: int = 7
    scheduler_gamma: float = 0.1

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Optional[str] = "checkpoints"

    # Reproducibility
    seed: int = 42

    # Logging frequency
    log_interval: int = 10  # Log every N batches

    def __post_init__(self):
        """Validate training configuration."""
        # Validate num_epochs
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

        # Validate learning_rate
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        # Validate weight_decay
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")

        # Validate device
        valid_devices = ["cpu", "cuda", "mps"]  # mps for Apple Silicon
        device_type = self.device.split(":")[0]  # Handle "cuda:0", "cuda:1", etc.
        if device_type not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got {self.device}"
            )

        # Validate scheduler parameters
        if self.scheduler_step_size < 1:
            raise ValueError(
                f"scheduler_step_size must be positive, got {self.scheduler_step_size}"
            )

        if not 0 < self.scheduler_gamma <= 1:
            raise ValueError(
                f"scheduler_gamma must be in (0, 1], got {self.scheduler_gamma}"
            )

        # Validate early stopping
        if self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be positive, got {self.early_stopping_patience}"
            )

        if self.early_stopping_min_delta < 0:
            raise ValueError(
                f"early_stopping_min_delta must be non-negative, "
                f"got {self.early_stopping_min_delta}"
            )

        # Validate checkpoint_dir if checkpointing enabled
        if self.save_checkpoints and self.checkpoint_dir:
            checkpoint_path = Path(self.checkpoint_dir)
            # Create directory if it doesn't exist
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Validate log_interval
        if self.log_interval < 1:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")


@dataclass
class MLflowConfig:
    """
    Configuration for MLflow experiment tracking.

    Environment variables:
        MLFLOW_TRACKING_URI: MLflow server URI
        MLFLOW_EXPERIMENT_NAME: Override experiment_name

    Example:
        >>> config = MLflowConfig(
        ...     experiment_name="my-experiment",
        ...     tracking_uri="http://localhost:5000"
        ... )
    """

    experiment_name: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            "chest-xray-classification"
        )
    )
    tracking_uri: Optional[str] = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI")
    )
    artifact_location: Optional[str] = None
    run_name: Optional[str] = None

    # Model registry settings
    register_model: bool = False
    model_name: Optional[str] = None

    def __post_init__(self):
        """Validate MLflow configuration."""
        # Validate experiment_name is not empty
        if not self.experiment_name or not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")

        # If registering model, model_name is required
        if self.register_model and not self.model_name:
            raise ValueError("model_name is required when register_model is True")


@dataclass
class WandBConfig:
    """
    Configuration for Weights & Biases experiment tracking.

    Environment variables:
        WANDB_PROJECT: Override project name
        WANDB_ENTITY: W&B team/user name
        WANDB_API_KEY: W&B API key for authentication

    Example:
        >>> config = WandBConfig(
        ...     project="my-project",
        ...     entity="my-team"
        ... )
    """

    project: str = field(
        default_factory=lambda: os.getenv(
            "WANDB_PROJECT",
            "chest-xray-classification"
        )
    )
    entity: Optional[str] = field(
        default_factory=lambda: os.getenv("WANDB_ENTITY")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("WANDB_API_KEY")
    )
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    # W&B specific settings
    mode: str = "online"  # "online", "offline", or "disabled"
    save_code: bool = True
    log_model: bool = True

    def __post_init__(self):
        """Validate W&B configuration."""
        # Validate project is not empty
        if not self.project or not self.project.strip():
            raise ValueError("project cannot be empty")

        # Validate mode
        valid_modes = ["online", "offline", "disabled"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode}")


@dataclass
class ExperimentConfig:
    """
    Master configuration combining all sub-configurations.

    This is the main configuration object used throughout the application.
    It aggregates all specific configurations into a single, hierarchical structure.

    Example:
        >>> # Create from defaults
        >>> config = ExperimentConfig()
        >>>
        >>> # Create with custom values
        >>> config = ExperimentConfig(
        ...     data=DataConfig(batch_size=64),
        ...     training=TrainingConfig(num_epochs=50)
        ... )
        >>>
        >>> # Load from YAML
        >>> config = ExperimentConfig.from_yaml("config.yaml")
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ExperimentConfig instance populated from YAML

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid or configuration is invalid

        Example:
            >>> config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        YAML format:
            ```yaml
            data:
              dataset_path: "Covid19-dataset"
              batch_size: 32
              image_size: 128

            model:
              num_classes: 3
              dropout_rates: [0.25, 0.3, 0.4, 0.25]

            training:
              num_epochs: 10
              learning_rate: 0.001
              device: "cuda"

            mlflow:
              experiment_name: "my-experiment"
              tracking_uri: "http://localhost:5000"

            wandb:
              project: "my-project"
              entity: "my-team"
            ```
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if not config_dict:
            config_dict = {}

        # Create config with nested dataclasses
        # Use empty dict if section not present in YAML
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            mlflow=MLflowConfig(**config_dict.get("mlflow", {})),
            wandb=WandBConfig(**config_dict.get("wandb", {})),
        )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration

        Example:
            >>> config = ExperimentConfig()
            >>> config_dict = config.to_dict()
            >>> print(config_dict["data"]["batch_size"])
            32
        """
        from dataclasses import asdict

        return asdict(self)

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path where YAML should be saved

        Example:
            >>> config = ExperimentConfig()
            >>> config.to_yaml("my_config.yaml")
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        """Pretty string representation of configuration."""
        lines = ["ExperimentConfig("]
        for field_name in ["data", "model", "training", "mlflow", "wandb"]:
            field_value = getattr(self, field_name)
            lines.append(f"  {field_name}={field_value},")
        lines.append(")")
        return "\n".join(lines)
