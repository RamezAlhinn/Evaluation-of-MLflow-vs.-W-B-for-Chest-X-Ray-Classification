# Parameter Matrix Example

This document shows examples of how to use the parameter matrix system.

## Example 1: Simple Experiment Configuration

### File: `configs/experiments.yaml`

```yaml
experiments:
  - name: "baseline"
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 20
  
  - name: "high_lr"
    learning_rate: 0.01
    batch_size: 32
    num_epochs: 20
  
  - name: "large_batch"
    learning_rate: 0.001
    batch_size: 64
    num_epochs: 20
```

### Run:
```bash
python run_hyperparameter_tuning.py --config configs/experiments.yaml
```

### Result:
3 experiments will run automatically, all tracked in MLflow!

## Example 2: Parameter Grid (All Combinations)

### File: `configs/hyperparameters.yaml`

```yaml
parameter_grid:
  learning_rate: [0.001, 0.01]
  batch_size: [32, 64]
  num_epochs: [20, 30]
```

### Run:
```bash
python run_hyperparameter_tuning.py
```

### Result:
2 × 2 × 2 = 8 experiments will run (all combinations)

## Example 3: Adding a New Experiment

### Step 1: Edit `configs/experiments.yaml`

Add this to the `experiments` list:

```yaml
experiments:
  # ... existing experiments ...
  
  - name: "my_custom_experiment"
    learning_rate: 0.002
    batch_size: 48
    num_epochs: 25
    lr_gamma: 0.2
    lr_step_size: 8
```

### Step 2: Run
```bash
python run_hyperparameter_tuning.py --config configs/experiments.yaml
```

### Step 3: View Results
```bash
python -m mlflow ui
```

## Example 4: Learning Rate Sweep

### File: `configs/learning_rate_sweep.yaml`

```yaml
base_config:
  dataset_path: "Covid19-dataset"
  image_size: 128
  device: "auto"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
  test_after_training: true

mlflow_config:
  experiment_name: "Learning-Rate-Sweep"
  use_run_names: true
  run_name_template: "lr_{learning_rate}"

experiments:
  - name: "lr_0.0001"
    learning_rate: 0.0001
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
  
  - name: "lr_0.0005"
    learning_rate: 0.0005
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
  
  - name: "lr_0.001"
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
  
  - name: "lr_0.005"
    learning_rate: 0.005
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
  
  - name: "lr_0.01"
    learning_rate: 0.01
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
```

### Run:
```bash
python run_hyperparameter_tuning.py --config configs/learning_rate_sweep.yaml
```

## Example 5: Batch Size Comparison

### File: `configs/batch_size_comparison.yaml`

```yaml
experiments:
  - name: "batch_16"
    learning_rate: 0.001
    batch_size: 16
    num_epochs: 20
  
  - name: "batch_32"
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 20
  
  - name: "batch_64"
    learning_rate: 0.001
    batch_size: 64
    num_epochs: 20
  
  - name: "batch_128"
    learning_rate: 0.001
    batch_size: 128
    num_epochs: 20
```

## Example 6: Comprehensive Grid Search

### File: `configs/comprehensive_search.yaml`

```yaml
parameter_grid:
  learning_rate: [0.0001, 0.001, 0.01]
  batch_size: [16, 32, 64]
  num_epochs: [20, 30]
  lr_gamma: [0.1, 0.5]

execution:
  max_experiments: 20  # Limit to 20 experiments
  shuffle: true
```

This would generate 3 × 3 × 2 × 2 = 36 combinations, but limited to 20.

## Tips

1. **Start Small**: Begin with 2-3 experiments to test
2. **Use Descriptive Names**: Makes it easy to identify experiments
3. **Organize by Purpose**: Create separate config files for different tuning purposes
4. **Limit Experiments**: Use `max_experiments` to avoid running too many
5. **Compare in MLflow**: Use MLflow UI to compare all experiments easily

## Viewing Results

After running experiments:

```bash
python -m mlflow ui
```

Then:
1. Open http://localhost:5000
2. Select your experiment
3. Compare multiple runs
4. Identify best hyperparameters

## Next Steps

1. Run baseline experiments
2. Identify promising ranges
3. Refine parameters
4. Run more targeted experiments
5. Select best model

