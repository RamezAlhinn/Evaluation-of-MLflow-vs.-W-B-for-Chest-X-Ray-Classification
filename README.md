# Evaluation of MLflow vs. W&B for Chest X-Ray Classification

This project evaluates and compares **MLflow** and **Weights & Biases (W&B)** for experiment tracking and model management in a deep learning classification task.

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
├── CNN_Model.py              # Custom CNN architecture for Chest X-Ray classification
├── data_loader.py            # Data loading and preprocessing utilities
├── MlFlow.py                 # MLflow integration for PyTorch
├── WD.py                     # W&B integration for PyTorch
├── train_mlflow.py           # Training script with MLflow tracking
├── train_wandb.py            # Training script with W&B tracking
├── compare_mlflow_wandb.py   # Comparison script for both tools
├── main.py                   # Main entry point and dataset download
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Model Architecture

The project uses a custom CNN architecture (`CustomCXRClassifier`) designed for Chest X-Ray classification:

- **Input**: RGB images (128x128 pixels)
- **Architecture**: 4 convolutional blocks with increasing filters (16, 64, 128, 128)
- **Output**: 3 classes (COVID-19, Viral Pneumonia, Normal)
- **Features**: Dropout regularization, MaxPooling, Fully Connected layers

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

### 1. Download Dataset

```bash
python main.py --download
```

This will download the COVID-19 Image Dataset from Kaggle to your local directory.

### 2. Train with MLflow

```bash
python train_mlflow.py --dataset_path <path_to_dataset> --epochs 20 --batch_size 32
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
mlflow ui
```
Then open http://localhost:5000 in your browser.

### 3. Train with W&B

```bash
python train_wandb.py --dataset_path <path_to_dataset> --epochs 20 --batch_size 32
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

**View W&B Dashboard:**
Results are automatically uploaded to the W&B cloud dashboard. A link will be printed in the terminal.

### 4. Compare MLflow vs W&B

```bash
python compare_mlflow_wandb.py --dataset_path <path_to_dataset> --epochs 10
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
- MLflow: [MLflow Documentation](https://mlflow.org/)
- Weights & Biases: [W&B Documentation](https://wandb.ai/)

## Author

Evaluation of MLflow vs. W&B for Chest X-Ray Classification

## Acknowledgments

- Kaggle for hosting the COVID-19 Image Dataset
- MLflow team for the excellent experiment tracking tool
- Weights & Biases team for the comprehensive MLOps platform
