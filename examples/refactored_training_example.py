"""
Example demonstrating the refactored architecture.

This script shows how to use the new architecture with proper separation of concerns,
dependency injection, and abstraction.

Run this example:
    python examples/refactored_training_example.py

Key improvements demonstrated:
1. Configuration via dataclasses
2. Dependency injection
3. Abstraction (BaseTracker interface)
4. Separation of concerns (Trainer, Model, Data, Tracking)
5. Type hints
6. Proper logging
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

# Import refactored components
from src.config.config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig
)
from src.models.cnn_model_refactored import CustomCXRClassifier, ModelConfig as CNNConfig
from src.training.trainer import Trainer
from src.tracking.base_tracker import DummyTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dummy_data(batch_size: int = 32, num_samples: int = 100):
    """
    Create dummy data for demonstration purposes.

    In a real scenario, you would use the actual data loader from src/data/data_loader.py

    Args:
        batch_size: Batch size
        num_samples: Total number of samples

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import TensorDataset

    # Create random data
    # 3 channels, 128x128 images, 3 classes
    X = torch.randn(num_samples, 3, 128, 128)
    y = torch.randint(0, 3, (num_samples,))

    # Create dataset
    dataset = TensorDataset(X, y)

    # Split into train/val/test (70/15/15)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def example_1_basic_training():
    """
    Example 1: Basic training without experiment tracking.

    Demonstrates:
    - Creating configuration
    - Initializing model
    - Creating trainer without tracker
    - Training
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Training (No Tracking)")
    print("=" * 80)

    # 1. Create configuration
    model_config = CNNConfig(
        num_classes=3,
        image_size=128,
        input_channels=3
    )

    training_config = TrainingConfig(
        num_epochs=3,  # Short for demo
        learning_rate=0.001,
        device='cpu',  # Use CPU for demo
        use_early_stopping=False,
        save_checkpoints=False
    )

    # 2. Create data loaders (dummy data for demo)
    train_loader, val_loader, test_loader = create_dummy_data(batch_size=16)

    # 3. Create model
    model = CustomCXRClassifier(model_config)
    logger.info(f"Model created with {model.get_num_parameters():,} parameters")

    # 4. Create trainer WITHOUT tracker (tracker=None)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        tracker=None,  # No tracking
        class_names=["COVID-19", "Viral Pneumonia", "Normal"]
    )

    # 5. Train
    results = trainer.train()

    # 6. Evaluate
    test_metrics = trainer.evaluate(test_loader)

    print(f"\nTraining completed!")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Best val accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"Test accuracy: {test_metrics['test_accuracy']:.2f}%")
    print("\n" + "=" * 80)


def example_2_training_with_dummy_tracker():
    """
    Example 2: Training with DummyTracker.

    Demonstrates:
    - Using a tracker (DummyTracker prints to stdout)
    - Dependency injection pattern
    - How training code doesn't change when tracker changes
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Training with DummyTracker")
    print("=" * 80)

    # Configuration
    model_config = CNNConfig(num_classes=3, image_size=128)
    training_config = TrainingConfig(
        num_epochs=2,
        learning_rate=0.001,
        device='cpu',
        use_early_stopping=False,
        save_checkpoints=False
    )

    # Data
    train_loader, val_loader, test_loader = create_dummy_data(batch_size=16)

    # Model
    model = CustomCXRClassifier(model_config)

    # Tracker (DummyTracker just prints to console)
    tracker = DummyTracker()

    # Trainer with tracker injected
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        tracker=tracker,  # Inject tracker
        class_names=["COVID-19", "Viral Pneumonia", "Normal"]
    )

    # Start tracking run
    tracker.start_run(run_name="dummy_example")

    try:
        # Train
        results = trainer.train()

        # Log final model
        tracker.log_model(model, artifact_path="final_model")

        # Evaluate
        test_metrics = trainer.evaluate(test_loader)

        # Log test metrics
        tracker.log_metrics({
            'test_loss': test_metrics['test_loss'],
            'test_accuracy': test_metrics['test_accuracy']
        })

    finally:
        # Always end run
        tracker.end_run()

    print("\n" + "=" * 80)


def example_3_custom_optimizer_and_loss():
    """
    Example 3: Custom optimizer and loss function.

    Demonstrates:
    - Injecting custom optimizer
    - Injecting custom loss function
    - Flexibility of dependency injection
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Optimizer and Loss")
    print("=" * 80)

    # Configuration
    model_config = CNNConfig(num_classes=3, image_size=128)
    training_config = TrainingConfig(
        num_epochs=2,
        learning_rate=0.01,  # Higher LR for SGD
        device='cpu',
        use_early_stopping=False,
        save_checkpoints=False
    )

    # Data
    train_loader, val_loader, test_loader = create_dummy_data(batch_size=16)

    # Model
    model = CustomCXRClassifier(model_config)

    # Custom optimizer (SGD instead of Adam)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=training_config.learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )

    # Custom loss function (with class weights for imbalanced data)
    class_weights = torch.tensor([1.0, 1.5, 1.2])  # Give more weight to minority class
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Trainer with custom optimizer and loss injected
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        tracker=None,
        optimizer=optimizer,  # Custom optimizer
        criterion=criterion,   # Custom loss function
        class_names=["COVID-19", "Viral Pneumonia", "Normal"]
    )

    # Train
    results = trainer.train()

    print(f"\nUsed optimizer: {optimizer.__class__.__name__}")
    print(f"Used loss: {criterion.__class__.__name__}")
    print(f"Best val accuracy: {results['best_val_accuracy']:.2f}%")
    print("\n" + "=" * 80)


def example_4_configuration_from_code():
    """
    Example 4: Creating configuration programmatically.

    Demonstrates:
    - Building configuration in code
    - Dataclass features (validation, defaults)
    - Type safety
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Configuration from Code")
    print("=" * 80)

    # Create custom configuration for 4-class problem with larger images
    model_config = CNNConfig(
        num_classes=4,  # Changed to 4 classes
        image_size=224,  # Larger images
        conv_filters=(32, 64, 128, 256),  # More filters
        dropout_rates=(0.3, 0.3, 0.4, 0.5),
        fc_hidden_size=256,
        fc_intermediate_size=128
    )

    training_config = TrainingConfig(
        num_epochs=2,
        learning_rate=0.0001,  # Lower LR for larger model
        device='cpu',
        use_scheduler=True,
        scheduler_step_size=5,
        scheduler_gamma=0.5,
        use_early_stopping=True,
        early_stopping_patience=3,
        save_checkpoints=False
    )

    print("\nModel Configuration:")
    print(f"  Classes: {model_config.num_classes}")
    print(f"  Image size: {model_config.image_size}x{model_config.image_size}")
    print(f"  Conv filters: {model_config.conv_filters}")
    print(f"  Dropout rates: {model_config.dropout_rates}")

    print("\nTraining Configuration:")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Scheduler: {training_config.use_scheduler}")
    print(f"  Early stopping: {training_config.use_early_stopping}")

    # Create model with custom config
    model = CustomCXRClassifier(model_config)
    print(f"\nModel has {model.get_num_parameters():,} parameters")

    # Note: Would need 4-class data for actual training
    print("\n✓ Configuration created successfully")
    print("  (Training skipped - would need 4-class dataset)")
    print("\n" + "=" * 80)


def example_5_swapping_trackers():
    """
    Example 5: Swapping trackers (Polymorphism).

    Demonstrates:
    - Polymorphism via BaseTracker interface
    - Same training code works with different trackers
    - Open/Closed Principle
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Swapping Trackers (Polymorphism)")
    print("=" * 80)

    # Configuration
    model_config = CNNConfig(num_classes=3, image_size=128)
    training_config = TrainingConfig(
        num_epochs=2,
        learning_rate=0.001,
        device='cpu',
        use_early_stopping=False,
        save_checkpoints=False
    )

    # Data
    train_loader, val_loader, test_loader = create_dummy_data(batch_size=16)

    # Function to train with any tracker
    def train_with_tracker(tracker, tracker_name):
        """Train model with given tracker."""
        print(f"\n--- Training with {tracker_name} ---")

        # Create fresh model
        model = CustomCXRClassifier(model_config)

        # Create trainer (same code, different tracker!)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            tracker=tracker,  # Inject tracker
            class_names=["COVID-19", "Viral Pneumonia", "Normal"]
        )

        # Train
        if tracker:
            tracker.start_run(run_name=f"example_{tracker_name}")

        try:
            results = trainer.train()
            print(f"✓ {tracker_name}: Best val acc = {results['best_val_accuracy']:.2f}%")

            if tracker:
                tracker.log_model(model, "final_model")

        finally:
            if tracker:
                tracker.end_run()

    # Train with different trackers
    train_with_tracker(None, "No Tracker")
    train_with_tracker(DummyTracker(), "DummyTracker")

    # In real usage, you would also do:
    # train_with_tracker(MLflowTracker("experiment"), "MLflow")
    # train_with_tracker(WandBTracker("project"), "W&B")

    print("\n✓ Same training code works with any tracker!")
    print("  This is polymorphism in action!")
    print("\n" + "=" * 80)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("REFACTORED ARCHITECTURE EXAMPLES")
    print("=" * 80)
    print("\nThis script demonstrates the refactored architecture with:")
    print("  1. Separation of concerns")
    print("  2. Dependency injection")
    print("  3. Abstraction via interfaces")
    print("  4. Configuration management")
    print("  5. Type safety")
    print("  6. Flexibility and extensibility")
    print("\n" + "=" * 80)

    # Run examples
    try:
        example_1_basic_training()
        example_2_training_with_dummy_tracker()
        example_3_custom_optimizer_and_loss()
        example_4_configuration_from_code()
        example_5_swapping_trackers()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  ✓ Trainer is independent of tracking system")
        print("  ✓ Can swap trackers without changing training code")
        print("  ✓ Configuration is explicit and validated")
        print("  ✓ Dependencies are injected, not hardcoded")
        print("  ✓ Easy to test with mocks")
        print("  ✓ Follows SOLID principles")
        print("\n" + "=" * 80)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
