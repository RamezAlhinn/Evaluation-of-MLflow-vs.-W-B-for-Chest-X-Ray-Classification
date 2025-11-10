# Hyperparameter Tuning Guide

This guide explains how to use the parameter matrix system for hyperparameter tuning.

## Quick Start

### 1. Run with Default Configuration
```bash
python run_hyperparameter_tuning.py
```

### 2. Run with Specific Config File
```bash
python run_hyperparameter_tuning.py --config configs/experiments.yaml
```

### 3. Run Quick Test (Fewer Experiments)
```bash
python run_hyperparameter_tuning.py --quick
```

## Configuration Files

### 1. `configs/hyperparameters.yaml` - Parameter Grid
Defines a grid of parameters to try. All combinations will be generated.

```yaml
parameter_grid:
  learning_rate: [0.001, 0.0001, 0.01]
  batch_size: [32, 64, 16]
  num_epochs: [20, 30, 50]
```

This will generate 3 × 3 × 3 = 27 experiments.

### 2. `configs/experiments.yaml` - Specific Experiments
Define exact experiments to run with specific parameters.

```yaml
experiments:
  - name: "baseline"
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 20
```

## Modifying Parameters

### Option 1: Edit Parameter Grid (Hyperparameter Search)

Edit `configs/hyperparameters.yaml`:

```yaml
parameter_grid:
  learning_rate: [0.001, 0.005, 0.01]  # Add/remove values
  batch_size: [16, 32, 64, 128]        # Add more batch sizes
  num_epochs: [10, 20, 30]             # Modify epochs
  lr_gamma: [0.1, 0.5]                 # Add learning rate decay
```

**Note**: This generates ALL combinations. For 4×4×3×2 = 96 experiments!

### Option 2: Define Specific Experiments (Recommended)

Edit `configs/experiments.yaml`:

```yaml
experiments:
  # Add your experiment here
  - name: "my_custom_experiment"
    learning_rate: 0.002
    batch_size: 48
    num_epochs: 25
    lr_gamma: 0.2
    lr_step_size: 8
```

Add as many experiments as you want!

### Option 3: Create New Config File

1. Copy `configs/experiments.yaml` to `configs/my_config.yaml`
2. Modify the parameters
3. Run: `python run_hyperparameter_tuning.py --config configs/my_config.yaml`

## Configuration Structure

### Base Configuration
Parameters that apply to all experiments:

```yaml
base_config:
  dataset_path: "Covid19-dataset"
  image_size: 128
  device: "auto"  # "auto", "cuda", or "cpu"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
  optimizer: "Adam"
  loss_function: "CrossEntropyLoss"
  test_after_training: true
```

### Experiment Parameters
Parameters that vary between experiments:

- `learning_rate`: Learning rate (float)
- `batch_size`: Batch size (int)
- `num_epochs`: Number of epochs (int)
- `lr_gamma`: Learning rate decay factor (float)
- `lr_step_size`: Learning rate decay step size (int)

### MLflow Configuration
```yaml
mlflow_config:
  experiment_name: "Hyperparameter-Tuning"
  use_run_names: true
  run_name_template: "lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}"
```

### Execution Settings
```yaml
execution:
  run_all_combinations: false  # If true, runs all combinations
  max_experiments: 10          # Limit number of experiments
  shuffle: true                # Shuffle before running
  continue_on_error: true      # Continue if one fails
```

## Command Line Options

```bash
python run_hyperparameter_tuning.py [OPTIONS]

Options:
  --config PATH              Path to YAML config file
  --quick                    Use quick test configuration
  --max-experiments N        Maximum number of experiments to run
  --shuffle                  Shuffle experiments before running
```

## Examples

### Example 1: Run Specific Experiments
```bash
python run_hyperparameter_tuning.py --config configs/experiments.yaml
```

### Example 2: Quick Test (Few Experiments)
```bash
python run_hyperparameter_tuning.py --quick
```

### Example 3: Limit Number of Experiments
```bash
python run_hyperparameter_tuning.py --max-experiments 5
```

### Example 4: Custom Config File
```bash
python run_hyperparameter_tuning.py --config configs/my_custom_config.yaml
```

## Workflow

### 1. Plan Your Experiments
- Decide which parameters to tune
- Choose parameter values to test
- Estimate number of experiments

### 2. Create/Edit Config File
- Use `configs/experiments.yaml` for specific experiments (recommended)
- Use `configs/hyperparameters.yaml` for grid search
- Or create your own config file

### 3. Run Experiments
```bash
python run_hyperparameter_tuning.py --config configs/experiments.yaml
```

### 4. Monitor Progress
- Watch terminal output for each experiment
- Check MLflow UI: `python -m mlflow ui`
- View results in real-time

### 5. Analyze Results
- Compare experiments in MLflow UI
- Identify best hyperparameters
- Run more experiments if needed

## Tips

### 1. Start Small
- Begin with a few experiments
- Use `--quick` flag for testing
- Gradually expand parameter space

### 2. Use Descriptive Names
```yaml
- name: "high_lr_large_batch_50epochs"
  learning_rate: 0.01
  batch_size: 64
  num_epochs: 50
```

### 3. Organize Experiments
Create different config files for different purposes:
- `configs/baseline.yaml` - Baseline experiments
- `configs/learning_rate_tuning.yaml` - Learning rate experiments
- `configs/batch_size_tuning.yaml` - Batch size experiments

### 4. Limit Experiments
Use `max_experiments` to avoid running too many:
```yaml
execution:
  max_experiments: 10
```

Or use command line:
```bash
python run_hyperparameter_tuning.py --max-experiments 10
```

### 5. Shuffle Experiments
Shuffle to avoid bias from running order:
```bash
python run_hyperparameter_tuning.py --shuffle
```

## Viewing Results

### MLflow UI
```bash
python -m mlflow ui
```
Open http://localhost:5000

### Compare Experiments
1. Select multiple runs in MLflow UI
2. Click "Compare"
3. View metrics, parameters, and training curves

### Export Results
Use the MLflow API to export results to CSV (see `example_mlflow_usage.py`).

## Common Parameter Ranges

### Learning Rate
- Typical range: `[0.0001, 0.001, 0.01]`
- Lower: More stable, slower convergence
- Higher: Faster convergence, may overshoot

### Batch Size
- Typical range: `[16, 32, 64, 128]`
- Smaller: More updates, more noise
- Larger: Stable gradients, less memory efficient

### Number of Epochs
- Typical range: `[10, 20, 30, 50]`
- More epochs: Better fit, risk of overfitting
- Fewer epochs: Faster, may underfit

### Learning Rate Decay
- `lr_gamma`: `[0.1, 0.5, 0.9]` (lower = faster decay)
- `lr_step_size`: `[5, 7, 10]` (epochs between decays)

## Troubleshooting

### Too Many Experiments
- Use `--max-experiments` to limit
- Edit config to reduce parameter grid
- Use specific experiments instead of grid

### Experiments Failing
- Check dataset path is correct
- Verify parameters are valid
- Set `continue_on_error: true` to continue

### Config File Errors
- Check YAML syntax
- Verify all required parameters are present
- Use a YAML validator

## Next Steps

1. Run baseline experiments
2. Identify promising parameter ranges
3. Refine and run more experiments
4. Compare results in MLflow UI
5. Select best model for deployment

## Additional Resources

- MLflow Guide: [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)
- MLflow Quick Start: [MLFLOW_QUICK_START.md](MLFLOW_QUICK_START.md)
- Example Usage: `python example_mlflow_usage.py`

