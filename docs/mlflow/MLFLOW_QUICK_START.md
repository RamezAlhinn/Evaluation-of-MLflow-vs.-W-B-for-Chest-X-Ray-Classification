# MLflow Quick Start Guide

## 🚀 Quick Commands

### Train a Model
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
```

### View Results

**Option 1: Using Python module (recommended on Windows)**
```bash
python -m mlflow ui
```

**Option 2: Using helper script**
```bash
python scripts/start_mlflow_ui.py
```

**Option 3: Using MLflow CLI (Linux/Mac)**
```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Train with Test Evaluation
```bash
python train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --test
```

## 📊 What Gets Tracked

### Metrics (Automatically Logged)
- Training Loss (per epoch)
- Training Accuracy (per epoch)
- Validation Loss (per epoch)
- Validation Accuracy (per epoch)
- Per-class Precision, Recall, F1-score
- Test metrics (if `--test` flag used)

### Parameters (Automatically Logged)
- Learning rate
- Batch size
- Number of epochs
- Image size
- Optimizer settings
- Model architecture

### Artifacts (Automatically Saved)
- Full PyTorch model
- Confusion matrix
- Test metrics (if available)

## 🎯 Common Use Cases

### 1. Basic Training
```bash
python train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
```

### 2. Hyperparameter Tuning
```bash
# Run 1
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --learning_rate 0.001 --run_name "lr_0.001"

# Run 2
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --learning_rate 0.01 --run_name "lr_0.01"

# Run 3
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --batch_size 64 --run_name "batch_64"
```

Then compare in MLflow UI!

### 3. Custom Experiment Name
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --experiment_name "MyExperiment"
```

### 4. Full Training with Evaluation
```bash
python scripts/train_mlflow.py \
    --dataset_path "Covid19-dataset" \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --run_name "final_model" \
    --test
```

## 🔍 Viewing Results

### Start MLflow UI

**On Windows (if `mlflow` command not found):**
```bash
python -m mlflow ui
```

**On Linux/Mac:**
```bash
mlflow ui
```

**Alternative (if above doesn't work):**
```bash
python -m mlflow ui --port 5000
```

### Access UI
- URL: http://localhost:5000
- View experiments, runs, metrics
- Compare multiple runs
- Download models
- **Note**: Keep the terminal window open while using the UI

### Compare Runs
1. Open MLflow UI
2. Select multiple runs (checkboxes)
3. Click "Compare" button
4. View side-by-side metrics and parameters

## 💾 Load Saved Models

### Python Code
```python
import mlflow
import mlflow.pytorch

# Load model by run ID
run_id = "your-run-id-here"
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

# Use model
model.eval()
output = model(input_tensor)
```

### Find Run ID
1. Open MLflow UI
2. Click on a run
3. Copy Run ID from run details

## 📈 MLflow UI Features

- **Experiments**: Organize runs into experiments
- **Runs**: Each training session is a run
- **Metrics**: View training curves and metrics
- **Parameters**: See hyperparameters for each run
- **Artifacts**: Download models and files
- **Compare**: Side-by-side comparison of runs

## 🎓 Best Practices

1. **Use descriptive run names**
   ```bash
   --run_name "baseline_lr0.001_bs32"
   ```

2. **Organize experiments**
   ```bash
   --experiment_name "Hyperparameter-Tuning"
   ```

3. **Always evaluate on test set**
   ```bash
   --test
   ```

4. **Compare multiple runs** in MLflow UI to find best hyperparameters

## 📚 More Information

- Full guide: [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)
- Examples: `python examples/example_mlflow_usage.py`
- MLflow docs: https://mlflow.org/docs/latest/index.html

## 🆘 Troubleshooting

### MLflow UI not starting?

**On Windows:**
```bash
# Use Python module syntax
python -m mlflow ui

# Or specify port
python -m mlflow ui --port 5001
```

**On Linux/Mac:**
```bash
mlflow ui --port 5001  # Use different port
```

### MLflow command not found?
The `mlflow` CLI might not be in your PATH. Use:
```bash
python -m mlflow ui
```
instead of `mlflow ui`

### Can't find runs?
- Check you're in the project directory
- Verify `mlruns/` folder exists
- Check experiment name matches

### Model loading issues?
- Ensure same PyTorch version
- Check model architecture matches
- Verify run_id is correct

