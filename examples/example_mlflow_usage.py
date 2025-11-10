"""
Example: How to use MLflow for experiment tracking
This script demonstrates various MLflow features and usage patterns
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow
import mlflow.pytorch
import torch
from src.models.cnn_model import CustomCXRClassifier
from src.data.data_loader import get_data_loaders
from src.tracking.mlflow_tracker import train_with_mlflow, evaluate_with_mlflow, MLflowTracker


def example_basic_training():
    """Example: Basic training with MLflow"""
    print("=" * 60)
    print("Example 1: Basic Training with MLflow")
    print("=" * 60)
    
    # Configuration
    dataset_path = "Covid19-dataset"
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'image_size': 128,
        'lr_step_size': 7,
        'lr_gamma': 0.1,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'random_seed': 42
    }
    
    # Load data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(dataset_path, config)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCXRClassifier(in_channels=3, num_classes=len(class_names))
    
    # Train with MLflow
    trained_model, history = train_with_mlflow(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        num_epochs=10,
        device=device,
        class_names=class_names,
        experiment_name="Example-Experiment",
        run_name="basic_training_example"
    )
    
    print("Training completed! Check MLflow UI: python -m mlflow ui")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "1":
            example_basic_training()
        else:
            print("Invalid example number. Use 1")
    else:
        print("MLflow Usage Examples")
        print("=" * 60)
        print("Run examples with: python examples/example_mlflow_usage.py <number>")
        print("\nAvailable examples:")
        print("  1. Basic Training with MLflow")
        print("\nExample:")
        print("  python examples/example_mlflow_usage.py 1")
