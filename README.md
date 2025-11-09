# Evaluation of MLflow vs. W&B for Chest X-Ray Classification

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Model Architecture](#model-architecture)
4. [Project Structure](#project-structure)
5. [Installation and Setup](#installation-and-setup)
6. [Project Phases](#project-phases)
7. [Experiment Tracking Comparison](#experiment-tracking-comparison)
8. [Usage Instructions](#usage-instructions)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Results and Findings](#results-and-findings)
11. [Key Comparisons](#key-comparisons)
12. [References](#references)

---

## Project Overview

This project provides an experimental evaluation comparing **MLflow** and **Weights & Biases (W&B)** for tracking machine learning experiments in the context of medical image classification. The goal is to assess user-friendliness, feature sets, and integration capabilities of both tools using a real-world deep learning task: COVID-19 chest X-ray classification.

### Objectives
- Compare MLflow and W&B in terms of:
  - **User-friendliness**: Ease of setup, API simplicity, and learning curve
  - **Feature set**: Logging capabilities, visualization tools, model management, and collaboration features
  - **Integration**: How well each tool integrates into existing ML workflows
- Implement a CNN-based image classifier for COVID-19 detection
- Track experiments, parameters, metrics, and artifacts using both platforms
- Provide recommendations based on practical usage

---

## Dataset Information

### COVID-19 Image Dataset
- **Type**: 3-way classification task
- **Classes**:
  1. COVID-19
  2. Viral Pneumonia
  3. Normal (Healthy)
- **Source**: [COVID-19 Image Dataset on Kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
- **Image Type**: Chest X-ray images
- **Format**: JPEG/PNG
- **Preprocessing Requirements**:
  - Image resizing (224x224 or 256x256)
  - Normalization (pixel values scaled to [0, 1])
  - Data augmentation (rotation, flipping, zoom)

### Dataset Structure
```
dataset/
├── COVID-19/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Viral_Pneumonia/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Normal/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

## Model Architecture

The CNN model is designed for chest X-ray classification with the following architecture:

### Architecture Details

| Layer | Type | Filters/Units | Kernel Size | Activation | Additional Info |
|-------|------|---------------|-------------|------------|-----------------|
| Conv1 | Convolutional | 16 | 3x3 | ReLU | Input layer |
| Pool1 | MaxPooling2D | - | 2x2 | - | - |
| Conv2 | Convolutional | 64 | 3x3 | ReLU | Padding: same |
| Pool2 | MaxPooling2D | - | 2x2 | - | - |
| Drop1 | Dropout | - | - | - | Rate: 0.25 |
| Conv3 | Convolutional | 128 | 3x3 | ReLU | Padding: same |
| Pool3 | MaxPooling2D | - | 2x2 | - | - |
| Drop2 | Dropout | - | - | - | Rate: 0.30 |
| Conv4 | Convolutional | 128 | 3x3 | ReLU | Padding: same |
| Pool4 | MaxPooling2D | - | 2x2 | - | - |
| Drop3 | Dropout | - | - | - | Rate: 0.40 |
| Flatten | Flatten | - | - | - | - |
| Dense1 | Dense | 128 | - | ReLU | - |
| Drop4 | Dropout | - | - | - | Rate: 0.25 |
| Dense2 | Dense | 64 | - | ReLU | - |
| Output | Dense | 3 | - | Softmax | 3 classes |

### Model Summary
- **Total Convolutional Layers**: 4
- **Total Dense Layers**: 3 (including output)
- **Dropout Layers**: 4 (for regularization)
- **Activation Functions**: ReLU (hidden layers), Softmax (output)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam (configurable)
- **Input Shape**: (256, 256, 3) or (224, 224, 3)

---

## Project Structure

```
Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification/
├── data/
│   ├── raw/                     # Raw dataset
│   ├── processed/               # Preprocessed images
│   └── splits/                  # Train/val/test splits
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_mlflow_experiments.ipynb
│   └── 03_wandb_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── model.py                # CNN model definition
│   ├── train_mlflow.py         # Training with MLflow tracking
│   ├── train_wandb.py          # Training with W&B tracking
│   ├── evaluate.py             # Model evaluation
│   └── utils.py                # Helper functions
├── models/
│   ├── mlflow_models/          # Models saved via MLflow
│   └── wandb_models/           # Models saved via W&B
├── mlruns/                     # MLflow tracking directory
├── wandb/                      # W&B tracking directory
├── config/
│   ├── mlflow_config.yaml
│   └── wandb_config.yaml
├── results/
│   ├── comparison_report.md
│   ├── mlflow_metrics.json
│   ├── wandb_metrics.json
│   └── visualizations/
├── requirements.txt
├── setup.py
├── main.py
└── README.md
```

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip or conda
- CUDA-compatible GPU (optional, for faster training)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification.git
cd Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n xray-tracking python=3.9
conda activate xray-tracking
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
1. Download the COVID-19 Image Dataset from Kaggle
2. Extract to `data/raw/` directory
3. Run preprocessing script:
```bash
python src/data_loader.py --preprocess
```

### Step 5: Setup Tracking Tools

#### MLflow Setup
```bash
# MLflow is ready to use out of the box
# Start MLflow UI (optional)
mlflow ui --port 5000
```
Access at: http://localhost:5000

#### W&B Setup
```bash
# Login to W&B
wandb login

# Or set API key
export WANDB_API_KEY=your_api_key_here
```

---

## Project Phases

### Phase 1: Project Setup and Data Preparation
**Duration**: 1-2 days

**Tasks**:
1. Set up project repository and environment
2. Download and organize the COVID-19 dataset
3. Implement data loading pipeline
4. Create train/validation/test splits (70/15/15)
5. Implement data augmentation
6. Verify data preprocessing

**Deliverables**:
- Organized dataset structure
- Data loader implementation
- Basic exploratory data analysis notebook

---

### Phase 2: Model Development
**Duration**: 2-3 days

**Tasks**:
1. Implement CNN architecture as specified
2. Create model building functions
3. Set up training loop
4. Implement evaluation metrics (accuracy, precision, recall, F1-score)
5. Test model training without tracking (baseline)

**Deliverables**:
- Model implementation in `src/model.py`
- Training script baseline
- Initial model performance metrics

---

### Phase 3: MLflow Integration
**Duration**: 2-3 days

**Tasks**:
1. Install and configure MLflow
2. Integrate MLflow tracking into training pipeline
3. Log parameters (learning rate, batch size, epochs, etc.)
4. Log metrics (loss, accuracy, per-class metrics)
5. Log artifacts (model weights, confusion matrices, plots)
6. Implement model registry functionality
7. Run multiple experiments with different hyperparameters
8. Document MLflow UI navigation and features

**Tracked Items**:
- **Parameters**: Learning rate, batch size, optimizer, dropout rates, image size
- **Metrics**: Training/validation loss, accuracy, precision, recall, F1-score
- **Artifacts**: Model checkpoints, confusion matrix plots, training curves, ROC curves
- **Models**: Saved model with MLflow format

**Deliverables**:
- `train_mlflow.py` script
- MLflow experiment runs (minimum 5 experiments)
- MLflow UI screenshots and documentation

---

### Phase 4: Weights & Biases Integration
**Duration**: 2-3 days

**Tasks**:
1. Install and configure W&B
2. Create W&B project
3. Integrate W&B tracking into training pipeline
4. Log parameters, metrics, and artifacts
5. Implement W&B-specific features:
   - Real-time metric visualization
   - Image logging (sample predictions)
   - Model versioning
   - Hyperparameter sweeps
6. Run multiple experiments matching MLflow experiments
7. Document W&B dashboard features

**Tracked Items**:
- **Parameters**: Same as MLflow for fair comparison
- **Metrics**: Same as MLflow
- **Artifacts**: Model checkpoints, visualizations, sample predictions
- **Special Features**: System metrics, gradient distributions, learning rate schedules

**Deliverables**:
- `train_wandb.py` script
- W&B experiment runs (minimum 5 experiments)
- W&B dashboard screenshots and documentation

---

### Phase 5: Comparative Analysis
**Duration**: 3-4 days

**Tasks**:
1. Run identical experiments on both platforms
2. Compare ease of setup and initialization
3. Evaluate logging capabilities
4. Assess visualization quality and interactivity
5. Test collaboration features
6. Measure performance overhead
7. Compare model management features
8. Evaluate documentation and community support
9. Analyze cost implications

**Comparison Criteria**:
- **User-friendliness** (1-10 scale)
- **Setup complexity**
- **API simplicity**
- **Learning curve**
- **Visualization quality**
- **Real-time monitoring**
- **Collaboration features**
- **Model versioning**
- **Integration ease**
- **Performance overhead**
- **Cost** (free tier vs. paid features)

**Deliverables**:
- Detailed comparison report
- Side-by-side feature comparison table
- Performance benchmarks
- Use case recommendations

---

### Phase 6: Documentation and Reporting
**Duration**: 2 days

**Tasks**:
1. Write comprehensive README
2. Create comparison report
3. Document best practices for each tool
4. Create tutorial notebooks
5. Prepare presentation slides
6. Record demo videos (optional)

**Deliverables**:
- Complete README.md
- Comparison report (`results/comparison_report.md`)
- Tutorial notebooks
- Presentation materials

---

## Experiment Tracking Comparison

### MLflow Features

#### Pros
- **Open-source and self-hosted**: Complete control over data
- **No account required**: Can run entirely offline
- **Language agnostic**: Works with any ML framework
- **Model registry**: Built-in model versioning and stage transitions
- **Reproducibility**: Easy to reproduce experiments
- **Deployment tools**: Built-in model serving capabilities
- **SQL backend support**: Can use databases for scalability

#### Cons
- **UI limitations**: Less interactive and modern compared to W&B
- **No real-time collaboration**: Primarily single-user focused
- **Limited visualization**: Basic plots, requires custom code for advanced viz
- **Self-hosting overhead**: Need to manage infrastructure
- **No built-in hyperparameter tuning**: Requires external tools

---

### Weights & Biases Features

#### Pros
- **Beautiful UI**: Modern, interactive dashboards
- **Real-time collaboration**: Team features, commenting, report sharing
- **Advanced visualizations**: Interactive plots, media logging (images, audio)
- **Hyperparameter sweeps**: Built-in automated tuning
- **System monitoring**: Automatic GPU/CPU/memory tracking
- **Easy sharing**: Public project links, embedded reports
- **Artifacts versioning**: Advanced artifact management
- **Integration library**: Extensive integrations with popular frameworks

#### Cons
- **Cloud-dependent**: Requires internet connection (self-hosted option exists but complex)
- **Account required**: Need to create account and login
- **Data privacy**: Data stored on W&B servers (concern for sensitive data)
- **Cost**: Free tier limited, paid plans can be expensive for teams
- **Vendor lock-in**: Harder to migrate away from W&B

---

## Usage Instructions

### Training with MLflow
```bash
# Basic training
python src/train_mlflow.py \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001

# With custom experiment name
python src/train_mlflow.py \
    --experiment_name "covid-cnn-exp-1" \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32

# View results
mlflow ui --port 5000
```

### Training with W&B
```bash
# Basic training
python src/train_wandb.py \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --project_name "covid-xray-classification"

# With hyperparameter sweep
wandb sweep config/sweep_config.yaml
wandb agent <sweep_id>
```

### Running Both in Parallel
```bash
# Run main script that trains with both tools
python main.py --mode both --epochs 50 --batch_size 32
```

### Model Evaluation
```bash
# Evaluate MLflow model
python src/evaluate.py \
    --model_path models/mlflow_models/run_id/model \
    --test_dir data/processed/test

# Evaluate W&B model
python src/evaluate.py \
    --model_path models/wandb_models/model_v1 \
    --test_dir data/processed/test
```

---

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Class-wise prediction breakdown
- **ROC-AUC**: Area under ROC curve for each class

### Tracking Tool Metrics
- **Setup Time**: Time to configure and start first experiment
- **Logging Overhead**: Performance impact on training time
- **API Calls**: Number of function calls required for basic tracking
- **Storage Usage**: Disk/cloud space consumed
- **UI Response Time**: Dashboard loading and interaction speed

---

## Results and Findings

### Model Performance
*(To be updated after experiments)*

| Metric | Value |
|--------|-------|
| Training Accuracy | TBD |
| Validation Accuracy | TBD |
| Test Accuracy | TBD |
| COVID-19 Precision | TBD |
| Viral Pneumonia Precision | TBD |
| Normal Precision | TBD |
| Macro F1-Score | TBD |

### Confusion Matrix
*(To be added after training)*

---

## Key Comparisons

### Feature Comparison Matrix

| Feature | MLflow | W&B | Winner |
|---------|--------|-----|--------|
| **Setup Ease** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | MLflow |
| **UI/UX** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | W&B |
| **Visualization** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | W&B |
| **Collaboration** | ⭐⭐ | ⭐⭐⭐⭐⭐ | W&B |
| **Model Registry** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | MLflow |
| **Hyperparameter Tuning** | ⭐⭐ | ⭐⭐⭐⭐⭐ | W&B |
| **Data Privacy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | MLflow |
| **Cost** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | MLflow |
| **Integration** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | W&B |
| **Real-time Monitoring** | ⭐⭐ | ⭐⭐⭐⭐⭐ | W&B |
| **Offline Support** | ⭐⭐⭐⭐⭐ | ⭐⭐ | MLflow |
| **Documentation** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | W&B |

### Use Case Recommendations

#### Choose MLflow if:
- You need complete data privacy and control
- Working in regulated industries (healthcare, finance)
- Require offline capability
- Want to avoid vendor lock-in
- Need robust model deployment tools
- Working solo or in small teams
- Cost is a primary concern

#### Choose W&B if:
- You prioritize visualization and UI/UX
- Working in collaborative team environment
- Need real-time experiment monitoring
- Want automated hyperparameter tuning
- Require advanced media logging (images, videos)
- Can work with cloud-based solution
- Budget allows for paid features

#### Hybrid Approach:
- Use MLflow for production model registry and deployment
- Use W&B during experimentation and development
- Migrate final models to MLflow for governance

---

## Lessons Learned

### MLflow
- Best for production-grade ML pipelines
- Excellent for model governance and compliance
- Requires more manual work for visualizations
- Perfect for self-hosted, secure environments

### W&B
- Exceptional for research and experimentation
- Collaboration features accelerate team productivity
- Automatic system monitoring saves debugging time
- Dashboard experience significantly improves workflow

### General Insights
- Both tools serve different but complementary purposes
- Choice depends on organizational requirements
- Consider starting with W&B for research, MLflow for production
- Integration overhead is minimal for both tools

---

## Future Work
- Implement ensemble methods combining multiple models
- Extend comparison to other tracking tools (Neptune.ai, Comet.ml)
- Add distributed training comparison
- Evaluate tools for model explainability tracking
- Test on larger datasets and more complex architectures

---

## References

### Dataset
- [COVID-19 Image Dataset - Kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

### Tracking Tools
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

### Papers and Articles
- MLflow: A Platform for the Machine Learning Lifecycle
- Weights & Biases: Best Practices for ML Experiment Tracking

### Related Work
- TensorBoard for TensorFlow/PyTorch visualization
- DVC (Data Version Control) for data pipeline management
- Kubeflow for Kubernetes-based ML workflows

---

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License

## Authors
- Your Name - Initial work

## Acknowledgments
- COVID-19 Image Dataset contributors
- MLflow and W&B development teams
- Open-source ML community

---

## Contact
For questions or feedback, please open an issue or contact [your.email@example.com]

---

**Project Status**: 🚧 In Development

**Last Updated**: November 2025
