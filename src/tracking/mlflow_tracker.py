"""
MLflow Integration for PyTorch Model Training
Tracks experiments, metrics, parameters, and models for Chest X-Ray Classification
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import os


class MLflowTracker:
    """
    Wrapper class for MLflow tracking in PyTorch training
    """
    
    def __init__(self, experiment_name="Chest-XRay-Classification-MLflow", 
                 tracking_uri=None):
        """
        Initialize MLflow tracking
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: local ./mlruns)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception:
            # If experiment creation fails, use default
            experiment_id = "0"
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
    
    def start_run(self, run_name=None, tags=None):
        """Start a new MLflow run"""
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params):
        """Log hyperparameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics, step=None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, artifact_path="model"):
        """Log PyTorch model"""
        mlflow.pytorch.log_model(model, artifact_path)
    
    def log_artifacts(self, local_dir, artifact_path=None):
        """Log artifacts (files/directories)"""
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_image(self, image, artifact_path):
        """Log an image artifact"""
        mlflow.log_image(image, artifact_path)
    
    def end_run(self):
        """End the current run"""
        mlflow.end_run()


def train_with_mlflow(model, train_loader, val_loader, config, num_epochs, 
                      device, class_names, experiment_name=None, run_name=None):
    """
    Train a PyTorch model with MLflow tracking
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Dictionary of hyperparameters to log
        num_epochs: Number of training epochs
        device: Device to train on (cuda/cpu)
        class_names: List of class names
        experiment_name: Name of MLflow experiment
        run_name: Name of the run
    
    Returns:
        Trained model and training history
    """
    
    # Initialize MLflow tracker
    tracker = MLflowTracker(experiment_name=experiment_name or "Chest-XRay-Classification-MLflow")
    
    # Move model to device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.get('lr_step_size', 7), 
                                          gamma=config.get('lr_gamma', 0.1))
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Start MLflow run
    with tracker.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow_params = {
            'learning_rate': config.get('learning_rate', 0.001),
            'batch_size': config.get('batch_size', 32),
            'num_epochs': num_epochs,
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss',
            'image_size': config.get('image_size', 128),
            'model_type': 'CustomCXRClassifier',
        }
        # Add any additional config parameters
        for key, value in config.items():
            if key not in mlflow_params:
                mlflow_params[key] = value
        
        tracker.log_params(mlflow_params)
        
        best_val_acc = 0.0
        best_model_state = None
        
        print("Starting training with MLflow tracking...")
        print(f"Experiment: {tracker.experiment_name}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # Log metrics to MLflow
            metrics = {
                'train_loss': avg_train_loss,
                'train_accuracy': train_acc,
                'val_loss': avg_val_loss,
                'val_accuracy': val_acc,
                'learning_rate': current_lr
            }
            tracker.log_metrics(metrics, step=epoch)
            
            # Calculate per-class metrics for validation set
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average=None, zero_division=0
            )
            
            for i, class_name in enumerate(class_names):
                tracker.log_metrics({
                    f'val_precision_{class_name}': precision[i],
                    f'val_recall_{class_name}': recall[i],
                    f'val_f1_{class_name}': f1[i]
                }, step=epoch)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # Print progress
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {current_lr:.6f}')
            print('-' * 60)
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Log best validation accuracy
        tracker.log_metrics({'best_val_accuracy': best_val_acc})
        
        # Log model
        tracker.log_model(model, artifact_path="model")
        
        # Log confusion matrix as artifact
        import tempfile
        import time
        cm = confusion_matrix(all_labels, all_preds)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.npy', delete=False)
        temp_file.close()
        try:
            np.save(temp_file.name, cm)
            mlflow.log_artifact(temp_file.name, 'confusion_matrix')
        finally:
            # Wait a bit and try to delete, ignore errors on Windows
            time.sleep(0.1)
            try:
                os.remove(temp_file.name)
            except (PermissionError, OSError):
                pass  # File might be locked on Windows, that's okay
        
        print(f"\nTraining completed!")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"View results: mlflow ui")
    
    return model, history


def evaluate_with_mlflow(model, test_loader, device, class_names, log_to_mlflow=True):
    """
    Evaluate model on test set and log results to MLflow
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names
        log_to_mlflow: Whether to log metrics to MLflow (default: True)
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    precision_macro = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )[0]
    recall_macro = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )[1]
    f1_macro = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )[2]
    
    avg_test_loss = test_loss / len(test_loader)
    
    metrics = {
        'test_loss': avg_test_loss,
        'test_accuracy': accuracy * 100,
        'test_precision_macro': precision_macro,
        'test_recall_macro': recall_macro,
        'test_f1_macro': f1_macro
    }
    
    # Log to MLflow if requested and a run is active
    if log_to_mlflow:
        try:
            # Check if there's an active run
            active_run = mlflow.active_run()
            if active_run:
                tracker = MLflowTracker()
                tracker.log_metrics(metrics)
                
                for i, class_name in enumerate(class_names):
                    tracker.log_metrics({
                        f'test_precision_{class_name}': precision[i],
                        f'test_recall_{class_name}': recall[i],
                        f'test_f1_{class_name}': f1[i]
                    })
                
                # Log confusion matrix
                import tempfile
                import time
                cm = confusion_matrix(all_labels, all_preds)
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.npy', delete=False)
                temp_file.close()
                try:
                    np.save(temp_file.name, cm)
                    mlflow.log_artifact(temp_file.name, 'test_confusion_matrix')
                finally:
                    # Wait a bit and try to delete, ignore errors on Windows
                    time.sleep(0.1)
                    try:
                        os.remove(temp_file.name)
                    except (PermissionError, OSError):
                        pass  # File might be locked on Windows, that's okay
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")
    
    return metrics
