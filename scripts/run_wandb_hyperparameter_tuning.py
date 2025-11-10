"""
Hyperparameter Tuning Script with Parameter Matrix for W&B
Runs multiple experiments with different configurations from YAML config files
"""

import argparse
import yaml
import os
import sys
import itertools
import random
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import wandb
from src.models.cnn_model import CustomCXRClassifier
from src.data.data_loader import get_data_loaders
from src.tracking.wandb_tracker import train_with_wandb, evaluate_with_wandb


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_parameter_combinations(parameter_grid):
    """Generate all combinations from parameter grid"""
    keys = parameter_grid.keys()
    values = parameter_grid.values()
    
    combinations = []
    for combination in itertools.product(*values):
        combo_dict = dict(zip(keys, combination))
        combinations.append(combo_dict)
    
    return combinations


def create_run_name(template, params, experiment_name=None):
    """Create run name from template and parameters"""
    if experiment_name:
        return experiment_name
    
    run_name = template
    for key, value in params.items():
        placeholder = f"{{{key}}}"
        if placeholder in run_name:
            # Format value appropriately
            if isinstance(value, float):
                formatted_value = f"{value:.4f}".rstrip('0').rstrip('.')
            else:
                formatted_value = str(value)
            run_name = run_name.replace(placeholder, formatted_value)
    
    return run_name


def run_experiment(config, params, experiment_name=None, run_name=None):
    """Run a single experiment with given parameters"""
    print("\n" + "="*80)
    print(f"Running Experiment: {run_name or 'Unnamed'}")
    print("="*80)
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Merge base config with experiment parameters
    full_config = config['base_config'].copy()
    full_config.update(params)
    
    # Handle device
    if full_config.get('device') == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(full_config['device'])
    
    print(f"Using device: {device}")
    
    # Prepare config for data loader
    data_config = {
        'batch_size': full_config['batch_size'],
        'image_size': full_config['image_size'],
        'train_split': full_config['train_split'],
        'val_split': full_config['val_split'],
        'test_split': full_config['test_split'],
        'random_seed': full_config['random_seed'],
        'learning_rate': full_config['learning_rate'],
        'lr_step_size': full_config.get('lr_step_size', 7),
        'lr_gamma': full_config.get('lr_gamma', 0.1),
        'optimizer': full_config.get('optimizer', 'Adam'),
        'loss_function': full_config.get('loss_function', 'CrossEntropyLoss'),
        'model_type': 'CustomCXRClassifier'
    }
    
    try:
        # Load dataset
        print("Loading dataset...")
        train_loader, val_loader, test_loader, class_names = get_data_loaders(
            full_config['dataset_path'], data_config
        )
        
        print(f"Classes: {class_names}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Initialize model
        print("Initializing model...")
        model = CustomCXRClassifier(in_channels=3, num_classes=len(class_names))
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Get W&B config
        wandb_config = config.get('wandb_config', {})
        project_name = wandb_config.get('project_name', 'Hyperparameter-Tuning-WB')
        entity = wandb_config.get('entity', None)
        
        # Train model
        trained_model, history, run = train_with_wandb(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=data_config,
            num_epochs=full_config['num_epochs'],
            device=device,
            class_names=class_names,
            project_name=project_name,
            run_name=run_name,
            entity=entity
        )
        
        # Evaluate on test set if requested
        if full_config.get('test_after_training', False):
            print("\nEvaluating on test set...")
            test_metrics = evaluate_with_wandb(
                model=trained_model,
                test_loader=test_loader,
                device=device,
                class_names=class_names,
                log_to_wandb=True,
                project_name=project_name,
                run_name=None,  # Use current active run
                entity=entity
            )
            
            print("\nTest Set Results:")
            print(f"  Accuracy: {test_metrics.get('test/accuracy', test_metrics.get('test_accuracy', 0)):.2f}%")
            print(f"  Precision (Macro): {test_metrics.get('test/precision_macro', test_metrics.get('test_precision_macro', 0)):.4f}")
            print(f"  Recall (Macro): {test_metrics.get('test/recall_macro', test_metrics.get('test_recall_macro', 0)):.4f}")
            print(f"  F1 Score (Macro): {test_metrics.get('test/f1_macro', test_metrics.get('test_f1_macro', 0)):.4f}")
        
        # Finish the run
        if wandb.run is not None:
            wandb.finish()
        
        print(f"\n✓ Experiment '{run_name}' completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Experiment '{run_name}' failed with error: {e}")
        import traceback
        traceback.print_exc()
        # Make sure to finish the run even on error
        if wandb.run is not None:
            wandb.finish()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run hyperparameter tuning with parameter matrix from YAML config for W&B'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/wandb/hyperparameters.yaml',
        help='Path to YAML configuration file (default: configs/wandb/hyperparameters.yaml)'
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Use quick test configuration (fewer experiments)'
    )
    parser.add_argument(
        '--max-experiments',
        type=int,
        default=None,
        help='Maximum number of experiments to run (overrides config)'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle experiments before running (overrides config)'
    )
    parser.add_argument(
        '--entity',
        type=str,
        default=None,
        help='W&B entity/team name (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Check if W&B is logged in
    try:
        wandb.login()
    except Exception as e:
        print("Warning: Could not verify W&B login. Make sure you've run 'wandb login'")
        print("Continuing anyway...")
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Please create a config file or use the default: configs/wandb/hyperparameters.yaml")
        sys.exit(1)
    
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override entity if provided
    if args.entity:
        if 'wandb_config' not in config:
            config['wandb_config'] = {}
        config['wandb_config']['entity'] = args.entity
    
    # Determine which experiments to run
    experiments = []
    
    # Check if using specific experiments or parameter grid
    if 'experiments' in config and config['experiments']:
        # Use specific experiments
        print("Using specific experiments from config...")
        for exp in config['experiments']:
            params = config['base_config'].copy()
            params.update(exp)
            exp_name = exp.get('name', 'Unnamed')
            experiments.append((params, exp_name))
    elif 'parameter_grid' in config and config['parameter_grid']:
        # Use parameter grid
        print("Using parameter grid from config...")
        if args.quick and 'quick_test' in config:
            print("Using quick test configuration...")
            grid = config['quick_test']
        else:
            grid = config['parameter_grid']
        
        combinations = generate_parameter_combinations(grid)
        print(f"Generated {len(combinations)} parameter combinations")
        
        # Create run names
        wandb_config = config.get('wandb_config', {})
        template = wandb_config.get('run_name_template', 'experiment_{idx}')
        use_run_names = wandb_config.get('use_run_names', True)
        
        for idx, params in enumerate(combinations):
            if use_run_names:
                run_name = create_run_name(template, params)
            else:
                run_name = f"experiment_{idx+1}"
            experiments.append((params, run_name))
    else:
        print("Error: No experiments or parameter_grid defined in config file")
        sys.exit(1)
    
    # Limit number of experiments
    max_experiments = args.max_experiments or config.get('execution', {}).get('max_experiments', None)
    if max_experiments and len(experiments) > max_experiments:
        print(f"Limiting to {max_experiments} experiments (out of {len(experiments)} total)")
        experiments = experiments[:max_experiments]
    
    # Shuffle if requested
    shuffle = args.shuffle or config.get('execution', {}).get('shuffle', False)
    if shuffle:
        print("Shuffling experiments...")
        random.shuffle(experiments)
    
    # Run experiments
    print(f"\n{'='*80}")
    print(f"Starting {len(experiments)} experiments")
    print(f"W&B Project: {config.get('wandb_config', {}).get('project_name', 'Hyperparameter-Tuning-WB')}")
    print(f"{'='*80}")
    
    results = []
    continue_on_error = config.get('execution', {}).get('continue_on_error', True)
    
    for idx, (params, run_name) in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}]")
        success = run_experiment(config, params, run_name=run_name)
        results.append((run_name, success))
        
        if not success and not continue_on_error:
            print("Stopping due to error (continue_on_error is False)")
            break
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for run_name, success in results:
            if not success:
                print(f"  - {run_name}")
    
    print("\n" + "="*80)
    print("View results in W&B dashboard:")
    project_name = config.get('wandb_config', {}).get('project_name', 'Hyperparameter-Tuning-WB')
    entity = config.get('wandb_config', {}).get('entity', '')
    if entity:
        print(f"  https://wandb.ai/{entity}/{project_name}")
    else:
        print(f"  https://wandb.ai/{project_name}")
    print("="*80)


if __name__ == '__main__':
    main()

