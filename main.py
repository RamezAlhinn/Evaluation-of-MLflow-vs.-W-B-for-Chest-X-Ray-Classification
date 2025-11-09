"""
Main Entry Point for COVID-19 Chest X-Ray Classification Project
Evaluation of MLflow vs. W&B for Chest X-Ray Classification

Dataset: COVID-19 Image Dataset
3-Way Classification: COVID-19, Viral Pneumonia, Normal
"""

import argparse
import os
import kagglehub


def download_dataset():
    """Download the COVID-19 dataset from Kaggle"""
    print("Downloading COVID-19 Image Dataset from Kaggle...")
    try:
        # Download latest version
        path = kagglehub.dataset_download("pranavraikokte/covid19-image-dataset")
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have:")
        print("1. Kaggle API credentials set up (~/.kaggle/kaggle.json)")
        print("2. kagglehub package installed (pip install kagglehub)")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='COVID-19 Chest X-Ray Classification - MLflow vs W&B Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download dataset
  python main.py --download
  
  # Train with MLflow
  python train_mlflow.py --dataset_path <path_to_dataset> --epochs 20
  
  # Train with W&B
  python train_wandb.py --dataset_path <path_to_dataset> --epochs 20
  
  # Compare MLflow vs W&B
  python compare_mlflow_wandb.py --dataset_path <path_to_dataset> --epochs 10
        """
    )
    parser.add_argument('--download', action='store_true',
                        help='Download the COVID-19 dataset from Kaggle')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to the dataset directory (if already downloaded)')
    
    args = parser.parse_args()
    
    if args.download:
        dataset_path = download_dataset()
        if dataset_path:
            print(f"\nDataset ready at: {dataset_path}")
            print("\nNext steps:")
            print("1. Train with MLflow: python train_mlflow.py --dataset_path <path> --epochs 20")
            print("2. Train with W&B: python train_wandb.py --dataset_path <path> --epochs 20")
            print("3. Compare both: python compare_mlflow_wandb.py --dataset_path <path> --epochs 10")
    else:
        print("COVID-19 Chest X-Ray Classification Project")
        print("=" * 60)
        print("\nThis project evaluates MLflow vs W&B for Chest X-Ray Classification")
        print("\nDataset: COVID-19 Image Dataset")
        print("Classes: COVID-19, Viral Pneumonia, Normal")
        print("\nAvailable scripts:")
        print("1. train_mlflow.py - Train model with MLflow tracking")
        print("2. train_wandb.py - Train model with W&B tracking")
        print("3. compare_mlflow_wandb.py - Compare both tracking tools")
        print("\nTo download the dataset, run: python main.py --download")
        print("\nFor help with any script, use: python <script_name> --help")


if __name__ == '__main__':
    main()
