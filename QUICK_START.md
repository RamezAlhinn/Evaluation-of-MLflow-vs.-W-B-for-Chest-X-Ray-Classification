# Quick Start Guide

Get started with the project in minutes!

## 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification-main

# Install dependencies
pip install -r requirements.txt
```

## 2. Download Dataset

```bash
python main.py --download
```

## 3. Train with MLflow

```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
```

View results:
```bash
python -m mlflow ui
```
Open http://localhost:5000

## 4. Train with W&B

```bash
# Login first
wandb login

# Then train
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20
```

View results at https://wandb.ai

## 5. Hyperparameter Tuning

### MLflow
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml
```

### W&B
```bash
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml
```

## 6. Compare Both Tools

```bash
python scripts/compare_mlflow_wandb.py --dataset_path "Covid19-dataset" --epochs 10
```

## Project Structure

```
.
├── src/              # Source code
├── scripts/          # Training scripts
├── configs/          # Configuration files
├── docs/             # Documentation
├── examples/         # Example scripts
└── tests/            # Unit tests
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details.

## Next Steps

- Read [README.md](README.md) for detailed instructions
- Check [docs/](docs/) for tool-specific guides
- See [examples/](examples/) for usage examples

