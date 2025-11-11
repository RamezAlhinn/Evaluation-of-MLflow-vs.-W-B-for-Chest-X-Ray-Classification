# Evaluation of MLflow vs. W&B for Chest X-Ray Classification

This project evaluates and compares **MLflow** and **Weights & Biases (W&B)** for experiment tracking and model management in a deep learning classification task.

---

## 🚀 Quick Start

```bash
# Install dependencies
make install

# Download dataset
make download

# Run quick W&B experiment (recommended for first-time users)
make wandb-quick

# Or try MLflow
make mlflow-quick

# Compare both tracking tools
make compare
```

📚 **New here?** Check out:
- [Getting Started Guide](GETTING_STARTED.md) - Command reference
- [Before & After Guide](BEFORE_AFTER.md) - See the improvements
- [Documentation Index](docs/README.md) - Complete documentation guide

---

## 📑 Table of Contents

- [Quick Start](#-quick-start)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Download Dataset](#1-download-dataset)
  - [Train with MLflow](#2-train-with-mlflow)
  - [Train with W&B](#3-train-with-wb)
  - [Hyperparameter Tuning](#4-hyperparameter-tuning)
  - [Compare Tools](#5-compare-mlflow-and-wb)
- [Comparison: MLflow vs W&B](#comparison-mlflow-vs-wb)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [License & Citation](#license)

---

## Dataset

**COVID-19 Image Dataset**
- **Source**: Kaggle (pranavraikokte/covid19-image-dataset)
- **Task**: 3-Way Classification
- **Classes**: 
  - COVID-19
  - Viral Pneumonia
  - Normal

## Project Structure

```
.
├── src/                          # Core source code
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   └── cnn_model.py          # Custom CNN architecture
│   ├── data/                     # Data handling
│   │   ├── __init__.py
│   │   └── data_loader.py        # Data loading utilities
│   ├── tracking/                 # Experiment tracking
│   │   ├── __init__.py
│   │   ├── mlflow_tracker.py     # MLflow integration
│   │   └── wandb_tracker.py      # W&B integration
│   └── utils/                    # Utility functions
│       └── __init__.py
│
├── scripts/                      # Training and execution scripts
│   ├── train_mlflow.py           # Train with MLflow
│   ├── train_wandb.py            # Train with W&B
│   ├── compare_mlflow_wandb.py   # Compare both tools
│   ├── run_hyperparameter_tuning.py      # MLflow hyperparameter tuning
│   ├── run_wandb_hyperparameter_tuning.py # W&B hyperparameter tuning
│   └── start_mlflow_ui.py        # Start MLflow UI
│
├── examples/                     # Example scripts
│   ├── example_mlflow_usage.py
│   └── example_wandb_usage.py
│
├── configs/                      # Configuration files
│   ├── mlflow/                   # MLflow configurations
│   │   ├── experiments.yaml
│   │   ├── hyperparameters.yaml
│   │   └── quick_test.yaml
│   └── wandb/                    # W&B configurations
│       ├── experiments.yaml
│       ├── hyperparameters.yaml
│       └── quick_test.yaml
│
├── docs/                         # Documentation
│   ├── mlflow/                   # MLflow documentation
│   ├── wandb/                    # W&B documentation
│   └── examples/                 # Example documentation
│
├── tests/                        # Unit tests
│   └── __init__.py
│
├── notebooks/                    # Jupyter notebooks (optional)
│
├── Covid19-dataset/              # Dataset directory
│   ├── train/
│   └── test/
│
├── mlruns/                       # MLflow runs (gitignored)
├── wandb/                        # W&B runs (gitignored)
│
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Documentation

- **[Quick Start Guide](docs/guides/QUICK_START.md)** - Get started quickly with the project
- **[W&B Quick Start](docs/guides/WANDB_QUICK_START.md)** - Quick start guide for W&B experiments
- **[Project Structure](docs/guides/PROJECT_STRUCTURE.md)** - Detailed project structure documentation
- **[Migration Guide](docs/guides/MIGRATION_GUIDE.md)** - Guide for migrating between versions
- **[Structure Improvements](docs/guides/STRUCTURE_IMPROVEMENTS.md)** - Recent structure improvements

## Model Architecture

The project uses a custom CNN architecture (`CustomCXRClassifier`) designed for Chest X-Ray classification. The model architecture is based on the implementation from [Vinay10100/Chest-X-Ray-Classification](https://github.com/Vinay10100/Chest-X-Ray-Classification).

### Architecture Details:

- **Input**: RGB images (128x128 pixels)
- **Architecture**: 
  - Convolutional layer 1: 16 filters, 3x3 kernel, ReLU activation, followed by MaxPooling2D (2x2 pool size)
  - Convolutional layer 2: 64 filters, 3x3 kernel, ReLU activation, padding='same', followed by MaxPooling2D (2x2 pool size), Dropout (0.25)
  - Convolutional layer 3: 128 filters, 3x3 kernel, ReLU activation, padding='same', followed by MaxPooling2D (2x2 pool size), Dropout (0.3)
  - Convolutional layer 4: 128 filters, 3x3 kernel, ReLU activation, padding='same', followed by MaxPooling2D (2x2 pool size), Dropout (0.4)
  - Flatten layer
  - Dense layer 1: 128 neurons, ReLU activation, Dropout (0.25)
  - Dense layer 2: 64 neurons, ReLU activation
  - Output layer: 3 neurons (one for each class), softmax activation
- **Output**: 3 classes (COVID-19, Viral Pneumonia, Normal)
- **Features**: Dropout regularization, MaxPooling, Fully Connected layers

**Note**: This architecture has been adapted from the original implementation to work with PyTorch and integrated with MLflow and W&B for experiment tracking.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification-main
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle API (for dataset download)

1. Create a Kaggle account at https://www.kaggle.com/
2. Go to Account → API → Create New Token
3. Download `kaggle.json` and place it in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

### 4. Set up W&B (optional, for W&B tracking)

```bash
wandb login
```

Follow the instructions to create a free account and get your API key.

## Usage

**TIP**: We provide a [Makefile](Makefile) to simplify commands. See [GETTING_STARTED.md](GETTING_STARTED.md) for quick reference.

```bash
# See all available commands
make help

# Quick examples
make wandb-quick      # Quick W&B test
make mlflow-quick     # Quick MLflow test
make compare          # Compare both tools
```

### 1. Download Dataset

```bash
# Using Makefile
make download

# Or manually
python main.py --download
```

This will download the COVID-19 Image Dataset from Kaggle to your local directory.

### 2. Train with MLflow

```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --batch_size 32
```

**Note**: If you install the package (`pip install -e .`), you can also use:
```bash
train-mlflow --dataset_path "Covid19-dataset" --epochs 20
```

**Options:**
- `--dataset_path`: Path to the dataset directory (required)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--image_size`: Image size for resizing (default: 128)
- `--device`: Device to use (default: cuda if available, else cpu)
- `--experiment_name`: MLflow experiment name (default: Chest-XRay-Classification-MLflow)
- `--run_name`: MLflow run name (optional)
- `--test`: Evaluate on test set after training

**View MLflow UI:**
```bash
# On Windows (recommended)
python -m mlflow ui

# On Linux/Mac
mlflow ui
```
Then open http://localhost:5000 in your browser.

**Note**: If `mlflow` command is not found, use `python -m mlflow ui` instead.

**📖 For detailed MLflow usage instructions, see [docs/mlflow/MLFLOW_GUIDE.md](docs/mlflow/MLFLOW_GUIDE.md)**

### 2.1. Hyperparameter Tuning with Parameter Matrix

Run multiple experiments with different configurations easily:

```bash
# Run with default configuration (parameter grid)
python scripts/run_hyperparameter_tuning.py

# Run specific experiments from config file
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml

# Quick test with fewer experiments
python scripts/run_hyperparameter_tuning.py --quick
```

**Modify parameters easily:**
1. Edit `configs/mlflow/experiments.yaml` to add/remove experiments
2. Edit `configs/mlflow/hyperparameters.yaml` for grid search
3. Run the script to execute all experiments

**📖 See [docs/mlflow/HYPERPARAMETER_TUNING_GUIDE.md](docs/mlflow/HYPERPARAMETER_TUNING_GUIDE.md) for detailed instructions**

### 3. Train with W&B

**First, login to W&B:**
```bash
wandb login
```

**Then train:**
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20 --batch_size 32
```

**Options:**
- `--dataset_path`: Path to the dataset directory (required)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--image_size`: Image size for resizing (default: 128)
- `--device`: Device to use (default: cuda if available, else cpu)
- `--project_name`: W&B project name (default: Chest-XRay-Classification-WB)
- `--run_name`: W&B run name (optional)
- `--entity`: W&B entity/team name (optional)
- `--test`: Evaluate on test set after training

**View Results:**
Results are automatically uploaded to your W&B dashboard at https://wandb.ai

**📖 For detailed W&B usage instructions, see [docs/wandb/WANDB_GUIDE.md](docs/wandb/WANDB_GUIDE.md)**

### 3.1. Hyperparameter Tuning with W&B Parameter Matrix

Run multiple experiments with different configurations easily:

```bash
# Run with default configuration (parameter grid)
python scripts/run_wandb_hyperparameter_tuning.py

# Run specific experiments from config file
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml

# Quick test with fewer experiments
python scripts/run_wandb_hyperparameter_tuning.py --quick
```

**Modify parameters easily:**
1. Edit `configs/wandb/experiments.yaml` to add/remove experiments
2. Edit `configs/wandb/hyperparameters.yaml` for grid search
3. Run the script to execute all experiments

**📖 See [docs/wandb/WANDB_HYPERPARAMETER_TUNING_GUIDE.md](docs/wandb/WANDB_HYPERPARAMETER_TUNING_GUIDE.md) for detailed instructions**

### 4. Compare MLflow vs W&B

```bash
python scripts/compare_mlflow_wandb.py --dataset_path "Covid19-dataset" --epochs 10
```

This script runs the same experiment with both tracking tools and provides a comparison of:
- Training time
- Model performance metrics
- Best validation accuracy
- Test set performance

**Options:**
- `--dataset_path`: Path to the dataset directory (required)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--image_size`: Image size for resizing (default: 128)
- `--device`: Device to use (default: cuda if available, else cpu)
- `--mlflow_experiment`: MLflow experiment name
- `--wandb_project`: W&B project name
- `--wandb_entity`: W&B entity/team name (optional)
- `--skip_mlflow`: Skip MLflow experiment
- `--skip_wandb`: Skip W&B experiment

## Dataset Structure

The dataset should be organized as follows:

```
dataset_path/
├── COVID-19/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Viral Pneumonia/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Normal/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The data loader automatically handles variations in folder names (case-insensitive matching).

## Features Tracked

### MLflow
- Hyperparameters (learning rate, batch size, epochs, etc.)
- Training and validation metrics (loss, accuracy)
- Per-class metrics (precision, recall, F1-score)
- Model artifacts
- Confusion matrix
- Best model checkpoint

### W&B
- Hyperparameters (learning rate, batch size, epochs, etc.)
- Training and validation metrics (loss, accuracy)
- Per-class metrics (precision, recall, F1-score)
- Real-time metrics visualization
- Confusion matrix plots
- Model artifacts
- Gradient and parameter tracking
- Learning rate scheduling

## Comparison: MLflow vs W&B

### MLflow
**Pros:**
- ✅ Local tracking by default (no account required)
- ✅ Simple UI: `mlflow ui`
- ✅ Good for local experiments and model registry
- ✅ Integrated with MLflow model serving
- ✅ Open-source and self-hostable

**Cons:**
- ❌ Basic visualization compared to W&B
- ❌ Limited collaboration features
- ❌ No real-time monitoring

### W&B
**Pros:**
- ✅ Rich visualization and collaboration features
- ✅ Real-time monitoring and alerts
- ✅ Advanced experiment comparison tools
- ✅ Cloud-based (accessible from anywhere)
- ✅ Great for team collaboration

**Cons:**
- ❌ Requires account (free tier available)
- ❌ Cloud-based (may require internet)
- ❌ More complex setup for self-hosting

## Results

After running the comparison script, you'll see:
- Training time comparison
- Best validation accuracy for each tool
- Test set performance metrics
- Detailed comparison of features

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended
- Sufficient disk space for dataset and model artifacts

## Troubleshooting

### Dataset Download Issues
- Ensure Kaggle API credentials are set up correctly
- Check that `kagglehub` is installed: `pip install kagglehub`
- Verify internet connection

### W&B Login Issues
- Run `wandb login` and follow the instructions
- Ensure you have a W&B account (free tier is available)

### CUDA/GPU Issues
- Install PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Issues
- Reduce batch size: `--batch_size 16`
- Reduce image size: `--image_size 64`
- Use CPU if GPU memory is limited: `--device cpu`

## License

This project is for educational and research purposes.

## Citation

If you use this project, please cite:
- COVID-19 Image Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
- Model Architecture: [Vinay10100/Chest-X-Ray-Classification](https://github.com/Vinay10100/Chest-X-Ray-Classification)
- MLflow: [MLflow Documentation](https://mlflow.org/)
- Weights & Biases: [W&B Documentation](https://wandb.ai/)

## Author

Evaluation of MLflow vs. W&B for Chest X-Ray Classification

## Acknowledgments

- [Vinay10100](https://github.com/Vinay10100) for the original CNN architecture implementation in [Chest-X-Ray-Classification](https://github.com/Vinay10100/Chest-X-Ray-Classification)
- Kaggle for hosting the COVID-19 Image Dataset
- MLflow team for the excellent experiment tracking tool
- Weights & Biases team for the comprehensive MLOps platform
