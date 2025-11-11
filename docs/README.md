# Documentation Index

Welcome to the Chest X-Ray Classification project documentation!

## 📖 Documentation Structure

### 🚀 Getting Started (Start Here!)

These documents will get you up and running quickly:

1. **[Project Overview](PROJECT_OVERVIEW.md)** - Visual guide with diagrams ⭐ NEW!
   - Architecture diagrams
   - Workflow visualizations
   - Component interactions
   - User journey maps

2. **[Main README](../README.md)** - Complete project overview
   - Dataset information
   - Installation instructions
   - Feature comparison (MLflow vs W&B)
   - Architecture details

2. **[Getting Started Guide](../GETTING_STARTED.md)** - Quick command reference
   - Makefile commands (recommended)
   - Manual command alternatives
   - Common troubleshooting

3. **[Before & After Guide](../BEFORE_AFTER.md)** - See the improvements
   - Project transformation overview
   - Command comparison
   - Best practices learned

### 📚 Detailed Guides

Located in [`docs/guides/`](guides/):

- **[Quick Start](guides/QUICK_START.md)** - Step-by-step tutorial for beginners
- **[W&B Quick Start](guides/WANDB_QUICK_START.md)** - W&B-specific getting started guide
- **[Project Structure](guides/PROJECT_STRUCTURE.md)** - Detailed project organization
- **[Project Structure Summary](guides/PROJECT_STRUCTURE_SUMMARY.md)** - Condensed structure overview
- **[Migration Guide](guides/MIGRATION_GUIDE.md)** - Guide for migrating between versions
- **[Structure Improvements](guides/STRUCTURE_IMPROVEMENTS.md)** - Recent architectural improvements
- **[README Structure](guides/README_STRUCTURE.md)** - Documentation organization

### 🔧 Configuration

Configuration files are organized by tracking tool:

- **[MLflow Configs](../configs/mlflow/)** - MLflow experiment configurations
  - `experiments.yaml` - Predefined experiments
  - `hyperparameters.yaml` - Hyperparameter search grid
  - `quick_test.yaml` - Quick test configuration

- **[W&B Configs](../configs/wandb/)** - Weights & Biases configurations
  - `hyperparameters.yaml` - Hyperparameter search grid

### 📝 Examples

Located in [`examples/`](../examples/):

- `example_mlflow_usage.py` - MLflow integration examples
- `example_wandb_usage.py` - W&B integration examples

### 🔬 Technical Documentation

- **[MLflow Documentation](mlflow/)** - MLflow-specific details
- **[W&B Documentation](wandb/)** - W&B-specific details
- **[Example Documentation](examples/)** - Code examples and tutorials

---

## 🗺️ Documentation Navigation Guide

### "I want to..."

**...get started quickly**
→ Read [Getting Started Guide](../GETTING_STARTED.md) and run `make help`

**...understand the project structure**
→ Read [Project Structure](guides/PROJECT_STRUCTURE.md) or [Summary](guides/PROJECT_STRUCTURE_SUMMARY.md)

**...run experiments with W&B**
→ Read [W&B Quick Start](guides/WANDB_QUICK_START.md) and run `make wandb-quick`

**...run experiments with MLflow**
→ Read [Main README](../README.md) and run `make mlflow-quick`

**...compare MLflow and W&B**
→ Read [Main README](../README.md) comparison section and run `make compare`

**...understand the improvements made**
→ Read [Before & After Guide](../BEFORE_AFTER.md)

**...configure hyperparameter searches**
→ Edit [`configs/wandb/hyperparameters.yaml`](../configs/wandb/hyperparameters.yaml) or [`configs/mlflow/hyperparameters.yaml`](../configs/mlflow/hyperparameters.yaml)

**...see code examples**
→ Check [`examples/`](../examples/) directory

**...migrate from an older version**
→ Read [Migration Guide](guides/MIGRATION_GUIDE.md)

---

## 🎯 Quick Reference

### Essential Commands

```bash
# See all available commands
make help

# Quick tests
make wandb-quick    # W&B quick test
make mlflow-quick   # MLflow quick test

# Full experiments
make wandb-tune     # W&B hyperparameter tuning
make mlflow-tune    # MLflow hyperparameter tuning

# Comparison
make compare        # Compare both tools

# UI
make mlflow-ui      # Start MLflow interface
```

### File Organization

```
.
├── README.md                    # Start here: main documentation
├── GETTING_STARTED.md           # Quick command reference
├── BEFORE_AFTER.md              # Transformation guide
├── Makefile                     # Command shortcuts
│
├── docs/                        # All documentation
│   ├── README.md                # This file (documentation index)
│   └── guides/                  # Detailed guides
│
├── configs/                     # Configuration files
│   ├── mlflow/                  # MLflow configs
│   └── wandb/                   # W&B configs
│
├── src/                         # Source code
│   ├── models/                  # Model definitions
│   ├── data/                    # Data loaders
│   ├── tracking/                # Experiment tracking
│   └── utils/                   # Utilities
│
├── scripts/                     # Execution scripts
└── examples/                    # Code examples
```

---

## 📊 Documentation Quality

This documentation follows best practices:

- ✅ **Progressive Disclosure**: Simple → Detailed information flow
- ✅ **Clear Entry Points**: README, Getting Started, Before/After
- ✅ **Organized Hierarchy**: Logical folder structure
- ✅ **Cross-Referenced**: Easy navigation between documents
- ✅ **Examples Included**: Code samples and usage examples
- ✅ **Quick Reference**: Makefile commands and this index
- ✅ **Troubleshooting**: Common issues addressed
- ✅ **Visual Structure**: File trees and comparison tables

---

## 🤝 Contributing to Documentation

When adding new documentation:

1. **Entry-level docs**: Update [Getting Started](../GETTING_STARTED.md)
2. **Detailed guides**: Add to [`docs/guides/`](guides/)
3. **Technical specs**: Add to relevant subdirectory ([`mlflow/`](mlflow/), [`wandb/`](wandb/))
4. **Code examples**: Add to [`examples/`](../examples/)
5. **Update this index**: Add links to new documents here

---

**Need help? Start with [Getting Started Guide](../GETTING_STARTED.md) or run `make help`**
