"""
Data Loading Module for COVID-19 Chest X-Ray Classification Dataset
Supports 3-way classification: COVID-19, Viral Pneumonia, Normal
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


class COVID19ChestXRayDataset(Dataset):
    """
    Custom Dataset class for COVID-19 Chest X-Ray images.
    Supports 3 classes: COVID-19, Viral Pneumonia, Normal
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of labels (0: COVID-19, 1: Viral Pneumonia, 2: Normal)
            transform: Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (128, 128), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def load_dataset_from_directory(dataset_path, image_size=128, batch_size=32, 
                                train_split=0.8, val_split=0.1, test_split=0.1, 
                                random_seed=42):
    """
    Load COVID-19 dataset from directory structure.
    
    Expected directory structure:
    dataset_path/
        COVID-19/
            image1.jpg
            image2.jpg
            ...
        Viral Pneumonia/
            image1.jpg
            ...
        Normal/
            image1.jpg
            ...
    
    Args:
        dataset_path: Path to the dataset directory
        image_size: Target image size for resizing (default: 128)
        batch_size: Batch size for DataLoader (default: 32)
        train_split: Proportion of data for training (default: 0.8)
        val_split: Proportion of data for validation (default: 0.1)
        test_split: Proportion of data for testing (default: 0.1)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    
    # Class mapping
    class_names = ['COVID-19', 'Viral Pneumonia', 'Normal']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    # Scan for common variations of class folder names
    possible_class_names = {
        'COVID-19': ['COVID-19', 'Covid', 'covid', 'covid-19', 'COVID'],
        'Viral Pneumonia': ['Viral Pneumonia', 'Viral', 'viral', 'viral_pneumonia'],
        'Normal': ['Normal', 'normal', 'NORMAL']
    }
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Find class directories
    found_classes = {}
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            for class_name, variants in possible_class_names.items():
                if item in variants and class_name not in found_classes:
                    found_classes[class_name] = item_path
                    break
    
    # If exact match not found, try case-insensitive match
    if len(found_classes) < 3:
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                item_lower = item.lower()
                for class_name, variants in possible_class_names.items():
                    if class_name not in found_classes:
                        for variant in variants:
                            if item_lower == variant.lower():
                                found_classes[class_name] = item_path
                                break
    
    # Load images from found class directories
    for class_name, class_idx in class_to_idx.items():
        if class_name in found_classes:
            class_dir = found_classes[class_name]
            # Supported image extensions
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            
            for filename in os.listdir(class_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(class_dir, filename)
                    image_paths.append(image_path)
                    labels.append(class_idx)
        else:
            print(f"Warning: Class directory '{class_name}' not found in dataset path")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in dataset path: {dataset_path}")
    
    print(f"Total images found: {len(image_paths)}")
    print(f"Class distribution:")
    for class_name, class_idx in class_to_idx.items():
        count = labels.count(class_idx)
        print(f"  {class_name}: {count}")
    
    # Split dataset
    # First split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, 
        test_size=test_split, 
        random_state=random_seed,
        stratify=labels
    )
    
    # Second split: train vs val
    val_size = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_seed,
        stratify=y_temp
    )
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(X_train)} images")
    print(f"  Validation: {len(X_val)} images")
    print(f"  Testing: {len(X_test)} images")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = COVID19ChestXRayDataset(X_train, y_train, transform=train_transform)
    val_dataset = COVID19ChestXRayDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = COVID19ChestXRayDataset(X_test, y_test, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names


def get_data_loaders(dataset_path, config):
    """
    Convenience function to get data loaders using a config dictionary.
    
    Args:
        dataset_path: Path to the dataset directory
        config: Dictionary with configuration parameters:
            - image_size: Target image size (default: 128)
            - batch_size: Batch size (default: 32)
            - train_split: Training split ratio (default: 0.8)
            - val_split: Validation split ratio (default: 0.1)
            - test_split: Test split ratio (default: 0.1)
            - random_seed: Random seed (default: 42)
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    return load_dataset_from_directory(
        dataset_path=dataset_path,
        image_size=config.get('image_size', 128),
        batch_size=config.get('batch_size', 32),
        train_split=config.get('train_split', 0.8),
        val_split=config.get('val_split', 0.1),
        test_split=config.get('test_split', 0.1),
        random_seed=config.get('random_seed', 42)
    )

