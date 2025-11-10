# Quick Start: Hyperparameter Tuning

## What is This?

A parameter matrix system that allows you to easily run multiple experiments with different hyperparameter configurations. Just edit a YAML file and run!

## Quick Start

### 1. Edit Configuration File

Edit `configs/experiments.yaml` and add your experiments:

```yaml
experiments:
  - name: "my_experiment"
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
```

### 2. Run Experiments

```bash
python run_hyperparameter_tuning.py --config configs/experiments.yaml
```

### 3. View Results

```bash
python -m mlflow ui
```

Open http://localhost:5000 to compare all experiments!

## Example: Adding a New Experiment

1. Open `configs/experiments.yaml`
2. Add a new experiment:

```yaml
experiments:
  # ... existing experiments ...
  
  - name: "my_new_experiment"
    learning_rate: 0.002
    batch_size: 64
    num_epochs: 30
    lr_gamma: 0.2
    lr_step_size: 10
```

3. Run: `python run_hyperparameter_tuning.py --config configs/experiments.yaml`

That's it! The experiment will run automatically.

## Configuration Files

- `configs/experiments.yaml` - Define specific experiments (recommended)
- `configs/hyperparameters.yaml` - Parameter grid (all combinations)
- `configs/quick_test.yaml` - Quick test with few experiments

## Common Modifications

### Change Learning Rate
```yaml
- name: "experiment_name"
  learning_rate: 0.01  # Change this value
```

### Change Batch Size
```yaml
- name: "experiment_name"
  batch_size: 64  # Change this value
```

### Change Number of Epochs
```yaml
- name: "experiment_name"
  num_epochs: 50  # Change this value
```

## Full Documentation

See [HYPERPARAMETER_TUNING_GUIDE.md](HYPERPARAMETER_TUNING_GUIDE.md) for complete documentation.

