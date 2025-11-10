# Migration Guide: New Project Structure

This guide helps you migrate to the new improved project structure.

## What Changed

### 1. Source Code Organization
- **Before**: Files in root (`CNN_Model.py`, `data_loader.py`, etc.)
- **After**: Organized in `src/` with subdirectories:
  - `src/models/` - Model definitions
  - `src/data/` - Data handling
  - `src/tracking/` - Experiment tracking
  - `src/utils/` - Utilities

### 2. Scripts Organization
- **Before**: Training scripts in root
- **After**: All scripts in `scripts/` directory

### 3. Configuration Files
- **Before**: All configs in `configs/`
- **After**: Organized by tool:
  - `configs/mlflow/` - MLflow configurations
  - `configs/wandb/` - W&B configurations

### 4. Documentation
- **Before**: Documentation files in root
- **After**: Organized in `docs/`:
  - `docs/mlflow/` - MLflow documentation
  - `docs/wandb/` - W&B documentation
  - `docs/examples/` - Example documentation

## Import Changes

### Old Imports
```python
from CNN_Model import CustomCXRClassifier
from data_loader import get_data_loaders
from MlFlow import train_with_mlflow
from WD import train_with_wandb
```

### New Imports
```python
from src.models.cnn_model import CustomCXRClassifier
from src.data.data_loader import get_data_loaders
from src.tracking.mlflow_tracker import train_with_mlflow
from src.tracking.wandb_tracker import train_with_wandb
```

## Config Path Changes

### Old Paths
```python
config_path = "configs/hyperparameters.yaml"
config_path = "configs/wandb_experiments.yaml"
```

### New Paths
```python
config_path = "configs/mlflow/hyperparameters.yaml"
config_path = "configs/wandb/experiments.yaml"
```

## Running Scripts

### Before
```bash
python train_wandb.py --dataset_path "Covid19-dataset"
python run_hyperparameter_tuning.py --config configs/experiments.yaml
```

### After
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset"
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml
```

Or if you install the package:
```bash
pip install -e .
train-wandb --dataset_path "Covid19-dataset"
```

## Documentation Paths

### Before
```bash
# View MLflow guide
cat MLFLOW_QUICK_START.md
```

### After
```bash
# View MLflow guide
cat docs/mlflow/MLFLOW_QUICK_START.md
```

## Benefits of New Structure

1. **Better Organization**: Clear separation of concerns
2. **Easier Maintenance**: Easy to find and modify files
3. **Professional**: Follows Python project best practices
4. **Scalable**: Easy to add new features
5. **Installable**: Can be installed as a package

## Migration Steps

1. **Update Your Scripts**: If you have custom scripts, update imports
2. **Update Config Paths**: Update any hardcoded config paths
3. **Update Documentation References**: Update any documentation links
4. **Test**: Run tests to ensure everything works
5. **Clean Up**: Remove old files if migration is successful

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root:
```bash
# From project root
python scripts/train_wandb.py
```

Or add the project root to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/train_wandb.py
```

### Config Not Found
Update config paths in your scripts:
```python
# Old
config_path = "configs/experiments.yaml"

# New
config_path = "configs/mlflow/experiments.yaml"
```

### Module Not Found
Install the package in development mode:
```bash
pip install -e .
```

This makes the `src` package available system-wide.

## Need Help?

If you encounter issues during migration:
1. Check that all files were moved correctly
2. Verify imports are updated
3. Check config paths are correct
4. Ensure you're running from the project root

## Rollback

If you need to rollback:
1. Use git to restore previous structure
2. Or manually move files back to root
3. Update imports back to old format

