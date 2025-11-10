# MLflow Usage Guide

This guide explains how to use MLflow for experiment tracking in this Chest X-Ray Classification project.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Training with MLflow](#training-with-mlflow)
3. [Viewing Results](#viewing-results)
4. [Comparing Experiments](#comparing-experiments)
5. [Loading Saved Models](#loading-saved-models)
6. [MLflow UI Features](#mlflow-ui-features)
7. [Best Practices](#best-practices)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python main.py --download
```

### 3. Train with MLflow
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
```

### 4. View Results
```bash
python -m mlflow ui
```
Then open http://localhost:5000 in your browser.

## Training with MLflow

### Basic Training
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
```

### Advanced Options
```bash
python scripts/train_mlflow.py \
    --dataset_path "Covid19-dataset" \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --image_size 128 \
    --experiment_name "MyExperiment" \
    --run_name "baseline_model" \
    --test
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset_path` | Path to dataset directory | Required |
| `--epochs` | Number of training epochs | 20 |
| `--batch_size` | Batch size for training | 32 |
| `--learning_rate` | Learning rate | 0.001 |
| `--image_size` | Image size for resizing | 128 |
| `--device` | Device (cuda/cpu) | cuda if available |
| `--experiment_name` | MLflow experiment name | Chest-XRay-Classification-MLflow |
| `--run_name` | Name for this run | Auto-generated |
| `--test` | Evaluate on test set | False |

## Viewing Results

### Start MLflow UI

**On Windows (recommended):**
```bash
python -m mlflow ui
```

**On Linux/Mac:**
```bash
mlflow ui
```

The UI will be available at: **http://localhost:5000**

**Note**: If the `mlflow` command is not found, always use `python -m mlflow ui` instead.

### MLflow UI Features

1. **Experiments List**: View all your experiments
2. **Runs Comparison**: Compare multiple runs side-by-side
3. **Metrics Visualization**: See training curves, validation metrics
4. **Parameters**: View hyperparameters for each run
5. **Artifacts**: Download models, confusion matrices, etc.
6. **Model Registry**: Register and manage model versions

### Navigation in MLflow UI

- **Experiments**: Left sidebar shows all experiments
- **Runs**: Each training session is a "run"
- **Metrics**: Click on a run to see detailed metrics
- **Compare**: Select multiple runs and click "Compare" to compare them
- **Download**: Click on artifacts to download models or files

## Comparing Experiments

### Compare Multiple Runs

1. Start MLflow UI: `python -m mlflow ui`
2. Open http://localhost:5000
3. Select multiple runs (checkboxes)
4. Click "Compare" button
5. View side-by-side comparison of:
   - Parameters (hyperparameters)
   - Metrics (accuracy, loss, etc.)
   - Training curves

### Example: Training Multiple Models

```bash
# Run 1: Baseline
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --run_name "baseline"

# Run 2: Higher learning rate
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --learning_rate 0.01 --run_name "lr_0.01"

# Run 3: Larger batch size
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --batch_size 64 --run_name "batch_64"
```

Then compare all three runs in MLflow UI.

## Loading Saved Models

### Using MLflow to Load Models

```python
import mlflow
import mlflow.pytorch
import torch

# Load a specific run's model
run_id = "your-run-id-here"
model_uri = f"runs:/{run_id}/model"
model = mlflow.pytorch.load_model(model_uri)

# Use the model for inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)
```

### Find Run ID

1. Open MLflow UI: `python -m mlflow ui`
2. Click on a run
3. Copy the Run ID from the run details page
4. Or use the MLflow API:

```python
import mlflow

# Get all runs in an experiment
experiment = mlflow.get_experiment_by_name("Chest-XRay-Classification-MLflow")
runs = mlflow.search_runs(experiment.experiment_id)

# Get the best run
best_run = runs.loc[runs['metrics.val_accuracy'].idxmax()]
best_run_id = best_run['run_id']
```

## MLflow UI Features

### 1. Metrics Tracking
- **Training Loss**: Tracked every epoch
- **Training Accuracy**: Tracked every epoch
- **Validation Loss**: Tracked every epoch
- **Validation Accuracy**: Tracked every epoch
- **Per-class Metrics**: Precision, Recall, F1-score for each class
- **Best Validation Accuracy**: Logged at the end

### 2. Parameters Logged
- Learning rate
- Batch size
- Number of epochs
- Image size
- Optimizer type
- Loss function
- Model architecture

### 3. Artifacts
- **Model**: Full PyTorch model saved
- **Confusion Matrix**: Saved as numpy array
- **Test Metrics**: If `--test` flag is used

### 4. Run Information
- Run ID
- Run name (if provided)
- Start/End time
- Status
- Tags (customizable)

## Best Practices

### 1. Use Descriptive Run Names
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --run_name "baseline_lr0.001_bs32"
```

### 2. Organize Experiments
```bash
# Different experiments for different purposes
python scripts/train_mlflow.py --experiment_name "Hyperparameter-Tuning" --run_name "trial_1"
python scripts/train_mlflow.py --experiment_name "Model-Architecture" --run_name "resnet50"
```

### 3. Track All Important Parameters
All hyperparameters are automatically logged. You can add custom parameters by modifying the training script.

### 4. Use Tags for Organization
You can add tags to runs programmatically:
```python
with mlflow.start_run(tags={"model_type": "CNN", "dataset": "COVID-19"}):
    # Training code
```

### 5. Regular Evaluation
Always use `--test` flag to evaluate on test set:
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --test
```

## Advanced Usage

### Custom Tracking URI

By default, MLflow stores data in `./mlruns`. You can change this:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Use SQLite database
# or
mlflow.set_tracking_uri("http://localhost:5000")  # Use remote server
```

### Query Runs Programmatically

```python
import mlflow
import pandas as pd

# Search runs
runs = mlflow.search_runs(
    experiment_names=["Chest-XRay-Classification-MLflow"],
    filter_string="metrics.val_accuracy > 0.8"
)

# Get best run
best_run = runs.loc[runs['metrics.val_accuracy'].idxmax()]
print(f"Best accuracy: {best_run['metrics.val_accuracy']}")
print(f"Run ID: {best_run['run_id']}")
```

### Export Results

```python
import mlflow

# Export runs to CSV
runs = mlflow.search_runs(experiment_names=["Chest-XRay-Classification-MLflow"])
runs.to_csv("experiment_results.csv", index=False)
```

## Troubleshooting

### MLflow UI Not Starting

**On Windows:**
```bash
# Use Python module syntax (recommended)
python -m mlflow ui

# Or use different port
python -m mlflow ui --port 5001
```

**On Linux/Mac:**
```bash
# Check if port 5000 is in use
mlflow ui --port 5001  # Use different port
```

**If `mlflow` command not found:**
Always use `python -m mlflow ui` instead of `mlflow ui`. This works on all platforms.

### Cannot Find Runs
- Make sure you're in the project directory where `mlruns/` folder exists
- Check experiment name matches
- Verify runs were completed successfully

### Model Loading Issues
- Ensure you're using the same PyTorch version
- Check that the model architecture matches
- Verify the run_id is correct

## Example Workflow

### Complete Training and Evaluation Workflow

```bash
# 1. Train model
python scripts/train_mlflow.py \
    --dataset_path "Covid19-dataset" \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --run_name "final_model" \
    --test

# 2. Start MLflow UI
python -m mlflow ui

# 3. Open browser to http://localhost:5000
# 4. View results, compare runs, download models
```

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow PyTorch Integration](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

## Quick Reference

```bash
# Train with MLflow
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20

# View UI
python -m mlflow ui

# Access UI
# Open http://localhost:5000 in browser

# Compare experiments
# Select multiple runs in UI and click "Compare"

# Load model
# Use run_id from UI: mlflow.pytorch.load_model(f"runs:/{run_id}/model")
```

