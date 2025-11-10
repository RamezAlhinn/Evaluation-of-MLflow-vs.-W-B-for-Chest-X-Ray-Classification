# Project Structure - Quick Reference

## 📁 Directory Organization

```
.
├── src/                    # Source code (models, data, tracking)
├── scripts/                # Executable scripts (training, tuning, comparison)
├── configs/                # Configuration files (YAML)
│   ├── mlflow/            # MLflow experiment configs
│   └── wandb/             # W&B experiment configs
├── docs/                   # Documentation
│   ├── mlflow/            # MLflow guides
│   ├── wandb/             # W&B guides
│   └── examples/          # Example docs
├── examples/               # Example scripts
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
└── main.py                # Main entry point
```

## 🚀 Quick Commands

### Training
```bash
# MLflow
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20

# W&B
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20
```

### Hyperparameter Tuning
```bash
# MLflow
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml

# W&B
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml
```

### View Results
```bash
# MLflow UI
python -m mlflow ui

# W&B (automatic upload to https://wandb.ai)
```

## 📝 Modify Experiments

Edit configuration files to add/remove experiments:

- **MLflow**: `configs/mlflow/experiments.yaml`
- **W&B**: `configs/wandb/experiments.yaml`

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed information.

