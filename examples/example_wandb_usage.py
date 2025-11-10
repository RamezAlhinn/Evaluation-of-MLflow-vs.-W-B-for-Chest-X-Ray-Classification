"""
Example: How to use Weights & Biases for experiment tracking
This script demonstrates various W&B features and usage patterns
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import wandb
import torch
from src.models.cnn_model import CustomCXRClassifier
from src.data.data_loader import get_data_loaders
from src.tracking.wandb_tracker import train_with_wandb, evaluate_with_wandb, WandBTracker


def example_basic_training():
    """Example: Basic training with W&B"""
    print("=" * 60)
    print("Example 1: Basic Training with W&B")
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
        'random_seed': 42,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'model_type': 'CustomCXRClassifier'
    }
    
    # Load data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(dataset_path, config)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCXRClassifier(in_channels=3, num_classes=len(class_names))
    
    # Train with W&B
    trained_model, history, run = train_with_wandb(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        num_epochs=10,
        device=device,
        class_names=class_names,
        project_name="Example-Project-WB",
        run_name="basic_training_example"
    )
    
    # Finish the run
    if wandb.run is not None:
        wandb.finish()
    
    print("Training completed! Check W&B dashboard: https://wandb.ai")


def example_custom_tracking():
    """Example: Custom W&B tracking"""
    print("=" * 60)
    print("Example 2: Custom W&B Tracking")
    print("=" * 60)
    
    # Initialize W&B
    wandb.init(
        project="Custom-Project-WB",
        name="custom_run",
        tags=["example", "custom"],
        config={
            'custom_param_1': 'value1',
            'custom_param_2': 42,
            'custom_param_3': 0.5
        }
    )
    
    # Log custom metrics
    for step in range(10):
        wandb.log({
            'custom_metric': step * 0.1,
            'another_metric': step * 0.2
        }, step=step)
    
    # Finish the run
    wandb.finish()
    print("Custom tracking completed!")


def example_load_model():
    """Example: Load a saved model from W&B"""
    print("=" * 60)
    print("Example 3: Load Model from W&B")
    print("=" * 60)
    
    try:
        # Initialize W&B API
        api = wandb.Api()
        
        # Get all runs in a project
        runs = api.runs("your-entity/your-project")
        
        if len(runs) == 0:
            print("No runs found. Please train a model first.")
            return
        
        # Get the best run
        best_run = max(runs, key=lambda r: r.summary.get('val/accuracy', 0))
        
        print(f"Best run ID: {best_run.id}")
        print(f"Best validation accuracy: {best_run.summary.get('val/accuracy', 0):.2f}%")
        
        # Download model artifact
        artifacts = best_run.logged_artifacts()
        model_artifact = None
        for artifact in artifacts:
            if artifact.type == 'model':
                model_artifact = artifact
                break
        
        if model_artifact:
            artifact_dir = model_artifact.download()
            print(f"Model artifact downloaded to: {artifact_dir}")
            
            # Load model
            model = CustomCXRClassifier(in_channels=3, num_classes=3)
            import os
            model_path = os.path.join(artifact_dir, "model.pth")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                print("Model loaded successfully!")
            else:
                print(f"Model file not found at: {model_path}")
        else:
            print("No model artifact found in this run")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you're logged in to W&B and the project/run exists")


def example_compare_runs():
    """Example: Compare multiple runs programmatically"""
    print("=" * 60)
    print("Example 4: Compare Runs Programmatically")
    print("=" * 60)
    
    try:
        # Initialize W&B API
        api = wandb.Api()
        
        # Get runs
        runs = api.runs("your-entity/your-project")
        
        if len(runs) == 0:
            print("No runs found.")
            return
        
        print(f"Found {len(runs)} runs")
        print("\nRun Comparison:")
        print("-" * 60)
        
        # Compare key metrics
        for run in runs:
            print(f"\nRun: {run.name}")
            print(f"  Run ID: {run.id}")
            print(f"  Validation Accuracy: {run.summary.get('val/accuracy', 'N/A')}")
            print(f"  Learning Rate: {run.config.get('learning_rate', 'N/A')}")
            print(f"  Batch Size: {run.config.get('batch_size', 'N/A')}")
            print(f"  Epochs: {run.config.get('num_epochs', 'N/A')}")
        
        # Find best run
        if runs:
            best_run = max(runs, key=lambda r: r.summary.get('val/accuracy', 0))
            print(f"\n{'='*60}")
            print("Best Run:")
            print(f"  Run ID: {best_run.id}")
            print(f"  Validation Accuracy: {best_run.summary.get('val/accuracy', 0):.2f}%")
            print(f"  Parameters: LR={best_run.config.get('learning_rate')}, "
                  f"BS={best_run.config.get('batch_size')}")
    except Exception as e:
        print(f"Error comparing runs: {e}")
        print("Make sure you're logged in to W&B and the project exists")


def example_export_results():
    """Example: Export W&B results to CSV"""
    print("=" * 60)
    print("Example 5: Export Results to CSV")
    print("=" * 60)
    
    try:
        import pandas as pd
        
        # Initialize W&B API
        api = wandb.Api()
        
        # Get runs
        runs = api.runs("your-entity/your-project")
        
        if len(runs) == 0:
            print("No runs found.")
            return
        
        # Convert to DataFrame
        runs_data = []
        for run in runs:
            run_data = {
                'run_id': run.id,
                'name': run.name,
                'val_accuracy': run.summary.get('val/accuracy', None),
                'val_loss': run.summary.get('val/loss', None),
                'learning_rate': run.config.get('learning_rate', None),
                'batch_size': run.config.get('batch_size', None),
                'num_epochs': run.config.get('num_epochs', None),
            }
            runs_data.append(run_data)
        
        df = pd.DataFrame(runs_data)
        
        # Export to CSV
        output_file = "wandb_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Results exported to: {output_file}")
        print(f"Total runs: {len(runs)}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error exporting results: {e}")
        print("Make sure you're logged in to W&B and the project exists")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "1":
            example_basic_training()
        elif example == "2":
            example_custom_tracking()
        elif example == "3":
            example_load_model()
        elif example == "4":
            example_compare_runs()
        elif example == "5":
            example_export_results()
        else:
            print("Invalid example number. Use 1-5")
    else:
        print("W&B Usage Examples")
        print("=" * 60)
        print("Run examples with: python example_wandb_usage.py <number>")
        print("\nAvailable examples:")
        print("  1. Basic Training with W&B")
        print("  2. Custom W&B Tracking")
        print("  3. Load Model from W&B")
        print("  4. Compare Runs Programmatically")
        print("  5. Export Results to CSV")
        print("\nExample:")
        print("  python example_wandb_usage.py 1")
        print("\nNote: Make sure you're logged in to W&B (wandb login)")

