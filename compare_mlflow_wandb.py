"""
Comparison Script for MLflow vs W&B
Runs the same experiment with both tracking tools and compares results
"""

import argparse
import torch
import torch.nn as nn
from CNN_Model import CustomCXRClassifier
from data_loader import get_data_loaders
from MlFlow import train_with_mlflow, evaluate_with_mlflow
from WD import train_with_wandb, evaluate_with_wandb
import os
import time
import wandb


def run_mlflow_experiment(dataset_path, config, epochs, device, class_names, 
                          experiment_name, run_name):
    """Run training with MLflow tracking"""
    print("\n" + "="*60)
    print("RUNNING MLFLOW EXPERIMENT")
    print("="*60)
    
    # Load dataset
    train_loader, val_loader, test_loader, _ = get_data_loaders(dataset_path, config)
    
    # Initialize model
    model = CustomCXRClassifier(in_channels=3, num_classes=len(class_names))
    
    # Train
    start_time = time.time()
    trained_model, history = train_with_mlflow(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        num_epochs=epochs,
        device=device,
        class_names=class_names,
        experiment_name=experiment_name,
        run_name=run_name
    )
    mlflow_time = time.time() - start_time
    
    # Evaluate
    test_metrics = evaluate_with_mlflow(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        log_to_mlflow=True
    )
    
    return {
        'history': history,
        'test_metrics': test_metrics,
        'training_time': mlflow_time
    }


def run_wandb_experiment(dataset_path, config, epochs, device, class_names, 
                         project_name, run_name, entity):
    """Run training with W&B tracking"""
    print("\n" + "="*60)
    print("RUNNING W&B EXPERIMENT")
    print("="*60)
    
    # Load dataset
    train_loader, val_loader, test_loader, _ = get_data_loaders(dataset_path, config)
    
    # Initialize model
    model = CustomCXRClassifier(in_channels=3, num_classes=len(class_names))
    
    # Train
    start_time = time.time()
    trained_model, history, run = train_with_wandb(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        num_epochs=epochs,
        device=device,
        class_names=class_names,
        project_name=project_name,
        run_name=run_name,
        entity=entity
    )
    wandb_time = time.time() - start_time
    
    # Evaluate on test set (before finishing the run)
    test_metrics = evaluate_with_wandb(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        log_to_wandb=True,
        project_name=project_name,
        run_name=None,  # Use current run
        entity=entity
    )
    
    # Finish the run after evaluation
    if wandb.run is not None:
        wandb.finish()
    
    return {
        'history': history,
        'test_metrics': test_metrics,
        'training_time': wandb_time
    }


def print_comparison(mlflow_results, wandb_results):
    """Print comparison of MLflow and W&B results"""
    print("\n" + "="*60)
    print("COMPARISON: MLFLOW vs W&B")
    print("="*60)
    
    print("\n--- Training Time ---")
    print(f"MLflow: {mlflow_results['training_time']:.2f} seconds")
    print(f"W&B:    {wandb_results['training_time']:.2f} seconds")
    print(f"Difference: {abs(mlflow_results['training_time'] - wandb_results['training_time']):.2f} seconds")
    
    print("\n--- Best Validation Accuracy ---")
    mlflow_best_val = max(mlflow_results['history']['val_acc'])
    wandb_best_val = max(wandb_results['history']['val_acc'])
    print(f"MLflow: {mlflow_best_val:.2f}%")
    print(f"W&B:    {wandb_best_val:.2f}%")
    print(f"Difference: {abs(mlflow_best_val - wandb_best_val):.2f}%")
    
    print("\n--- Test Set Performance ---")
    print(f"\nMLflow:")
    print(f"  Accuracy:  {mlflow_results['test_metrics']['test_accuracy']:.2f}%")
    print(f"  Precision: {mlflow_results['test_metrics']['test_precision_macro']:.4f}")
    print(f"  Recall:    {mlflow_results['test_metrics']['test_recall_macro']:.4f}")
    print(f"  F1 Score:  {mlflow_results['test_metrics']['test_f1_macro']:.4f}")
    
    print(f"\nW&B:")
    print(f"  Accuracy:  {wandb_results['test_metrics']['test/accuracy']:.2f}%")
    print(f"  Precision: {wandb_results['test_metrics']['test/precision_macro']:.4f}")
    print(f"  Recall:    {wandb_results['test_metrics']['test/recall_macro']:.4f}")
    print(f"  F1 Score:  {wandb_results['test_metrics']['test/f1_macro']:.4f}")
    
    print("\n--- Key Differences ---")
    print("MLflow:")
    print("  - Local tracking by default (./mlruns)")
    print("  - Simple UI: mlflow ui")
    print("  - Good for local experiments and model registry")
    print("  - Integrated with MLflow model serving")
    
    print("\nW&B:")
    print("  - Cloud-based by default (requires account)")
    print("  - Rich visualization and collaboration features")
    print("  - Real-time monitoring and alerts")
    print("  - Advanced experiment comparison tools")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Compare MLflow vs W&B')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the COVID-19 dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size for resizing (default: 128)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (default: cuda if available)')
    parser.add_argument('--mlflow_experiment', type=str, default='Chest-XRay-Classification-MLflow',
                        help='MLflow experiment name')
    parser.add_argument('--wandb_project', type=str, default='Chest-XRay-Classification-WB',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/team name (optional)')
    parser.add_argument('--skip_mlflow', action='store_true',
                        help='Skip MLflow experiment')
    parser.add_argument('--skip_wandb', action='store_true',
                        help='Skip W&B experiment')
    
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
    
    # Load dataset to get class names
    _, _, _, class_names = get_data_loaders(args.dataset_path, config)
    
    mlflow_results = None
    wandb_results = None
    
    # Run MLflow experiment
    if not args.skip_mlflow:
        mlflow_results = run_mlflow_experiment(
            dataset_path=args.dataset_path,
            config=config,
            epochs=args.epochs,
            device=device,
            class_names=class_names,
            experiment_name=args.mlflow_experiment,
            run_name='comparison_run'
        )
    else:
        print("Skipping MLflow experiment")
    
    # Run W&B experiment
    if not args.skip_wandb:
        wandb_results = run_wandb_experiment(
            dataset_path=args.dataset_path,
            config=config,
            epochs=args.epochs,
            device=device,
            class_names=class_names,
            project_name=args.wandb_project,
            run_name='comparison_run',
            entity=args.wandb_entity
        )
    else:
        print("Skipping W&B experiment")
    
    # Print comparison
    if mlflow_results and wandb_results:
        print_comparison(mlflow_results, wandb_results)
    elif mlflow_results:
        print("\nMLflow experiment completed. Run W&B experiment for comparison.")
    elif wandb_results:
        print("\nW&B experiment completed. Run MLflow experiment for comparison.")
    
    print("\nComparison completed!")


if __name__ == '__main__':
    main()

