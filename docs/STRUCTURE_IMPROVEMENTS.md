# Project Structure Improvements

This document outlines the improvements made to the project structure for better organization and maintainability.

## Improvements Made

### 1. Organized Source Code (`src/`)
- **models/**: Model architecture definitions
- **data/**: Data loading and preprocessing
- **tracking/**: Experiment tracking integrations (MLflow, W&B)
- **utils/**: Utility functions and helpers

### 2. Centralized Scripts (`scripts/`)
- All training and execution scripts in one place
- Easy to find and run scripts
- Consistent import structure

### 3. Organized Configuration (`configs/`)
- Separated by tool: `mlflow/` and `wandb/`
- Easy to modify and extend
- Clear organization

### 4. Comprehensive Documentation (`docs/`)
- Organized by topic: `mlflow/`, `wandb/`, `examples/`
- Easy to find relevant documentation
- Clear structure

### 5. Example Scripts (`examples/`)
- Separate directory for example scripts
- Demonstrates usage patterns
- Easy to reference

### 6. Test Directory (`tests/`)
- Dedicated tests directory
- Ready for unit tests
- Follows Python best practices

## Key Changes

### File Organization
- ✅ Moved all scripts to `scripts/` directory
- ✅ Organized documentation in `docs/` directory
- ✅ Separated configs by tool (`configs/mlflow/`, `configs/wandb/`)
- ✅ Created proper package structure with `src/`
- ✅ Added example scripts to `examples/` directory

### Import Structure
- ✅ All scripts use `from src.*` imports
- ✅ Added project root to Python path in scripts
- ✅ Consistent import pattern across all files

### Documentation
- ✅ Organized documentation by tool and topic
- ✅ Updated README with new structure
- ✅ Created PROJECT_STRUCTURE.md for detailed documentation

## Benefits

1. **Better Organization**: Clear separation of concerns
2. **Easier Navigation**: Easy to find files
3. **Scalability**: Easy to add new features
4. **Maintainability**: Clean structure makes maintenance easier
5. **Professional**: Follows Python project best practices
6. **Documentation**: Well-organized documentation

## Running Scripts

All scripts can be run from the project root:

```bash
# Training scripts
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20

# Hyperparameter tuning
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml

# Comparison
python scripts/compare_mlflow_wandb.py --dataset_path "Covid19-dataset" --epochs 10

# MLflow UI
python scripts/start_mlflow_ui.py
```

## Next Steps

1. ✅ Organize files into proper directories
2. ✅ Update imports in all Python files
3. ✅ Update documentation references
4. ✅ Test that all scripts work
5. ⏭️ Add unit tests
6. ⏭️ Add integration tests
7. ⏭️ Create setup.py for package installation

## Migration Notes

- All scripts now use `from src.*` imports
- Scripts automatically add project root to Python path
- Configuration files are in `configs/mlflow/` and `configs/wandb/`
- Documentation is in `docs/` directory
- Examples are in `examples/` directory

## File Locations

### Source Code
- Models: `src/models/cnn_model.py`
- Data Loading: `src/data/data_loader.py`
- MLflow Tracker: `src/tracking/mlflow_tracker.py`
- W&B Tracker: `src/tracking/wandb_tracker.py`

### Scripts
- Train MLflow: `scripts/train_mlflow.py`
- Train W&B: `scripts/train_wandb.py`
- Compare: `scripts/compare_mlflow_wandb.py`
- Hyperparameter Tuning: `scripts/run_hyperparameter_tuning.py`
- W&B Hyperparameter Tuning: `scripts/run_wandb_hyperparameter_tuning.py`

### Configuration
- MLflow: `configs/mlflow/*.yaml`
- W&B: `configs/wandb/*.yaml`

### Documentation
- MLflow: `docs/mlflow/*.md`
- W&B: `docs/wandb/*.md`
- Examples: `docs/examples/*.md`

