"""
Weights & Biases (W&B) Integration for PyTorch Model Training
Tracks experiments, metrics, parameters, and models for Chest X-Ray Classification
"""

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io


class WandBTracker:
    """
    Wrapper class for W&B tracking in PyTorch training
    """
    
    def __init__(self, project_name="Chest-XRay-Classification-WB", 
                 entity=None, config=None):
        """
        Initialize W&B tracking
        
        Args:
            project_name: Name of the W&B project
            entity: W&B entity/team name (optional)
            config: Dictionary of hyperparameters/config
        """
        self.project_name = project_name
        self.entity = entity
        self.config = config or {}
    
    def init(self, run_name=None, tags=None, resume=None):
        """Initialize a new W&B run"""
        return wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=self.config,
            name=run_name,
            tags=tags,
            resume=resume
        )
    
    def log(self, metrics, step=None):
        """Log metrics"""
        wandb.log(metrics, step=step)
    
    def log_model(self, model, artifact_path="model"):
        """Log PyTorch model as artifact"""
        torch.save(model.state_dict(), artifact_path)
        wandb.save(artifact_path)
    
    def watch(self, model, log_freq=100):
        """Watch model gradients and parameters"""
        wandb.watch(model, log_freq=log_freq)
    
    def finish(self):
        """Finish the current run"""
        wandb.finish()


def train_with_wandb(model, train_loader, val_loader, config, num_epochs, 
                     device, class_names, project_name=None, run_name=None, 
                     entity=None):
    """
    Train a PyTorch model with W&B tracking
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Dictionary of hyperparameters to log
        num_epochs: Number of training epochs
        device: Device to train on (cuda/cpu)
        class_names: List of class names
        project_name: Name of W&B project
        run_name: Name of the run
        entity: W&B entity/team name (optional)
    
    Returns:
        Trained model and training history
    """
    
    # Initialize W&B tracker
    tracker = WandBTracker(
        project_name=project_name or "Chest-XRay-Classification-WB",
        entity=entity,
        config=config
    )
    
    # Start W&B run
    run = tracker.init(run_name=run_name)
    
    # Move model to device
    model = model.to(device)
    
    # Watch model (log gradients and parameters)
    tracker.watch(model, log_freq=100)
    
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
    
    print("Starting training with W&B tracking...")
    print(f"Project: {tracker.project_name}")
    print(f"Run: {run.name}")
    print(f"View results: {run.url}")
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
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
            
            # Log batch-level metrics occasionally
            if batch_idx % 10 == 0:
                batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
                tracker.log({
                    'batch_loss': loss.item(),
                    'batch_accuracy': batch_acc
                }, step=epoch * len(train_loader) + batch_idx)
        
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
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
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
        
        # Log metrics to W&B
        metrics = {
            'epoch': epoch + 1,
            'train/loss': avg_train_loss,
            'train/accuracy': train_acc,
            'val/loss': avg_val_loss,
            'val/accuracy': val_acc,
            'val/precision_macro': precision_macro,
            'val/recall_macro': recall_macro,
            'val/f1_macro': f1_macro,
            'learning_rate': current_lr
        }
        
        # Log per-class metrics
        for i, class_name in enumerate(class_names):
            metrics[f'val/precision_{class_name}'] = precision[i]
            metrics[f'val/recall_{class_name}'] = recall[i]
            metrics[f'val/f1_{class_name}'] = f1[i]
        
        tracker.log(metrics, step=epoch)
        
        # Create and log confusion matrix every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            cm = confusion_matrix(all_labels, all_preds)
            fig = plot_confusion_matrix(cm, class_names)
            tracker.log({'confusion_matrix': wandb.Image(fig)})
            plt.close(fig)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            # Log model artifact (save periodically, not every epoch)
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
                    torch.save(model.state_dict(), f.name)
                    artifact = wandb.Artifact('best_model', type='model')
                    artifact.add_file(f.name)
                    run.log_artifact(artifact)
                    import os
                    os.remove(f.name)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {current_lr:.6f}')
        print('-' * 60)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Log best validation accuracy
    tracker.log({'best_val_accuracy': best_val_acc})
    
    # Final confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    fig = plot_confusion_matrix(cm, class_names, title='Final Validation Confusion Matrix')
    tracker.log({'final_confusion_matrix': wandb.Image(fig)})
    plt.close(fig)
    
    print(f"\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"View results: {run.url}")
    
    # Don't finish the run here - let the caller decide
    # This allows evaluation to be logged to the same run
    return model, history, run


def evaluate_with_wandb(model, test_loader, device, class_names, log_to_wandb=True, 
                       project_name=None, run_name=None, entity=None):
    """
    Evaluate model on test set and log results to W&B
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        class_names: List of class names
        log_to_wandb: Whether to log metrics to W&B (default: True)
        project_name: Name of W&B project (optional, used if log_to_wandb=True and new run needed)
        run_name: Name of the run (optional)
        entity: W&B entity/team name (optional)
    
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
        'test/loss': avg_test_loss,
        'test/accuracy': accuracy * 100,
        'test/precision_macro': precision_macro,
        'test/recall_macro': recall_macro,
        'test/f1_macro': f1_macro
    }
    
    # Log per-class metrics
    for i, class_name in enumerate(class_names):
        metrics[f'test/precision_{class_name}'] = precision[i]
        metrics[f'test/recall_{class_name}'] = recall[i]
        metrics[f'test/f1_{class_name}'] = f1[i]
    
    # Log to W&B if requested
    if log_to_wandb:
        try:
            # Check if there's an active W&B run
            if wandb.run is not None:
                # Log to current run
                wandb.log(metrics)
                
                # Log confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                fig = plot_confusion_matrix(cm, class_names, title='Test Set Confusion Matrix')
                wandb.log({'test_confusion_matrix': wandb.Image(fig)})
                plt.close(fig)
            elif run_name:
                # Create new run for evaluation
                tracker = WandBTracker(
                    project_name=project_name or "Chest-XRay-Classification-WB",
                    entity=entity
                )
                run = tracker.init(run_name=run_name)
                tracker.log(metrics)
                
                # Log confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                fig = plot_confusion_matrix(cm, class_names, title='Test Set Confusion Matrix')
                tracker.log({'test_confusion_matrix': wandb.Image(fig)})
                plt.close(fig)
                
                tracker.finish()
        except Exception as e:
            print(f"Warning: Could not log to W&B: {e}")
    
    return metrics


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Plot confusion matrix as a figure
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        title: Title of the plot
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()
