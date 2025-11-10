# W&B Quick Start Guide

## 🚀 Quick Commands

### Login to W&B
```bash
wandb login
```
Follow the instructions to create a free account and get your API key.

### Train a Model
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20
```

### View Results
Go to https://wandb.ai and select your project to view results in the dashboard.

### Train with Test Evaluation
```bash
python train_wandb.py --dataset_path "Covid19-dataset" --epochs 20 --test
```

## 📊 What Gets Tracked

### Metrics (Automatically Logged)
- Training Loss (per epoch)
- Training Accuracy (per epoch)
- Validation Loss (per epoch)
- Validation Accuracy (per epoch)
- Per-class Precision, Recall, F1-score
- Test metrics (if `--test` flag used)
- System metrics (CPU, GPU, memory)

### Parameters (Automatically Logged)
- Learning rate
- Batch size
- Number of epochs
- Image size
- Optimizer settings
- Model architecture

### Artifacts (Automatically Saved)
- Full PyTorch model (as artifact)
- Confusion matrix (as image)
- Test metrics (if available)

## 🎯 Common Use Cases

### 1. Basic Training
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20
```

### 2. Hyperparameter Tuning
```bash
# Run 1
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --learning_rate 0.001 --run_name "lr_0.001"

# Run 2
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --learning_rate 0.01 --run_name "lr_0.01"

# Run 3
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --batch_size 64 --run_name "batch_64"
```

Then compare in W&B dashboard!

### 3. Custom Project Name
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --project_name "MyProject"
```

### 4. Team/Entity Sharing
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --entity "my-team" --project_name "Shared-Project"
```

### 5. Full Training with Evaluation
```bash
python scripts/train_wandb.py \
    --dataset_path "Covid19-dataset" \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --run_name "final_model" \
    --test
```

## 🔍 Viewing Results

### W&B Dashboard
1. Go to https://wandb.ai
2. Login to your account
3. Select your project
4. View runs, metrics, and visualizations

### Compare Runs
1. Go to W&B dashboard
2. Select multiple runs (checkboxes)
3. Click "Compare" button
4. View side-by-side metrics and parameters

## 💾 Load Saved Models

### Python Code
```python
import wandb
import torch
from CNN_Model import CustomCXRClassifier

# Initialize W&B API
api = wandb.Api()

# Get a specific run
run = api.run("your-entity/your-project/run_id")

# Download model artifact
artifact = run.use_artifact("model:latest")
artifact_dir = artifact.download()

# Load model
model = CustomCXRClassifier(in_channels=3, num_classes=3)
model.load_state_dict(torch.load(os.path.join(artifact_dir, "model.pth")))
```

### Find Run ID
1. Go to W&B dashboard
2. Click on a run
3. Copy Run ID from URL or run details

## 📈 W&B Dashboard Features

- **Runs Table**: View all your runs in a table
- **Metrics**: View training curves and metrics
- **Parameters**: See hyperparameters for each run
- **System Metrics**: CPU, GPU, memory usage
- **Media**: View confusion matrices, sample predictions
- **Artifacts**: Download models and files
- **Compare**: Side-by-side comparison of runs
- **Sweeps**: Organize hyperparameter sweeps

## 🎓 Best Practices

1. **Use descriptive run names**
   ```bash
   --run_name "baseline_lr0.001_bs32"
   ```

2. **Organize projects**
   ```bash
   --project_name "Hyperparameter-Tuning"
   ```

3. **Use teams/entities for collaboration**
   ```bash
   --entity "my-team"
   ```

4. **Always evaluate on test set**
   ```bash
   --test
   ```

5. **Compare multiple runs** in W&B dashboard to find best hyperparameters

## 📚 More Information

- Full guide: [docs/wandb/WANDB_GUIDE.md](docs/wandb/WANDB_GUIDE.md)
- Examples: `python examples/example_wandb_usage.py`
- W&B docs: https://docs.wandb.ai/

## 🆘 Troubleshooting

### W&B not logging?
- Make sure you're logged in: `wandb login`
- Check internet connection
- Verify API key is correct

### Can't find runs?
- Check project name matches
- Verify you're logged into correct W&B account
- Check entity/team name if using teams

### Offline mode?
If you're offline, W&B can run in offline mode:
```bash
wandb offline
```
Sync later with:
```bash
wandb sync
```

## Quick Reference

```bash
# Login to W&B
wandb login

# Train with W&B
python train_wandb.py --dataset_path "Covid19-dataset" --epochs 20

# View results
# Go to https://wandb.ai

# Compare experiments
# Select multiple runs in W&B dashboard and click "Compare"

# Load model
# Use W&B API: wandb.Api().run("entity/project/run_id")
```

