"""
Training Script with MLflow Tracking
Trains the CNN model for Chest X-Ray Classification using MLflow for experiment tracking
"""

import argparse
import torch
import torch.nn as nn
from CNN_Model import CustomCXRClassifier
from data_loader import get_data_loaders
from MlFlow import train_with_mlflow, evaluate_with_mlflow
import os


def main():
    parser = argparse.ArgumentParser(description='Train CNN with MLflow tracking')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the COVID-19 dataset directory')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size for resizing (default: 128)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (default: cuda if available)')
    parser.add_argument('--experiment_name', type=str, default='Chest-XRay-Classification-MLflow',
                        help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='MLflow run name (optional)')
    parser.add_argument('--test', action='store_true',
                        help='Evaluate on test set after training')
    
    args = parser.parse_args()
    
    # Check if dataset path exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return
    
    # Device configuration
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Configuration dictionary
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size,
        'lr_step_size': 7,
        'lr_gamma': 0.1,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'random_seed': 42
    }
    
    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        args.dataset_path, config
    )
    
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = CustomCXRClassifier(in_channels=3, num_classes=len(class_names))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model with MLflow tracking
    print("\n" + "="*60)
    print("Starting training with MLflow tracking...")
    print("="*60)
    
    trained_model, history = train_with_mlflow(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        num_epochs=args.epochs,
        device=device,
        class_names=class_names,
        experiment_name=args.experiment_name,
        run_name=args.run_name
    )
    
    # Evaluate on test set if requested
    if args.test:
        print("\n" + "="*60)
        print("Evaluating on test set...")
        print("="*60)
        
        test_metrics = evaluate_with_mlflow(
            model=trained_model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            log_to_mlflow=True
        )
        
        print("\nTest Set Results:")
        print(f"  Accuracy: {test_metrics['test_accuracy']:.2f}%")
        print(f"  Precision (Macro): {test_metrics['test_precision_macro']:.4f}")
        print(f"  Recall (Macro): {test_metrics['test_recall_macro']:.4f}")
        print(f"  F1 Score (Macro): {test_metrics['test_f1_macro']:.4f}")
    
    print("\nTraining completed!")
    print("View MLflow UI: mlflow ui")


if __name__ == '__main__':
    main()

