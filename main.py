"""
Main Entry Point for COVID-19 Chest X-Ray Classification Project
Evaluation of MLflow vs. W&B for Chest X-Ray Classification

Dataset: COVID-19 Image Dataset
3-Way Classification: COVID-19, Viral Pneumonia, Normal
"""

import argparse
import os


def download_dataset():
    """Download the COVID-19 dataset from Kaggle and save it in the project directory"""
    try:
        import kagglehub
    except ImportError:
        print("Error: kagglehub package is not installed.")
        print("Please install it using: pip install kagglehub")
        print("Or install all requirements: pip install -r requirements.txt")
        return None
    
    # Get the project directory (where this script is located)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(project_dir, "Covid19-dataset")
    
    # Check if dataset already exists in project directory
    if os.path.exists(target_dir):
        print(f"Dataset already exists in project directory: {target_dir}")
        # Verify it has the train folder with classes
        train_dir = os.path.join(target_dir, "train")
        if os.path.exists(train_dir):
            classes = [item for item in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, item))]
            if any(c.lower() in ['covid', 'normal', 'viral'] for c in classes):
                print("Dataset appears to be complete. Using existing dataset.")
                return target_dir
        else:
            print("Dataset folder exists but appears incomplete. Re-downloading...")
    
    print("Downloading COVID-19 Image Dataset from Kaggle...")
    try:
        # Download to kaggle cache first
        cache_path = kagglehub.dataset_download("pranavraikokte/covid19-image-dataset")
        print(f"Dataset downloaded to cache: {cache_path}")
        
        # Copy or move dataset to project directory
        import shutil
        import zipfile
        import glob
        
        # Check if the dataset is in a zip file
        zip_files = glob.glob(os.path.join(cache_path, "*.zip"))
        
        if zip_files:
            # Extract zip file to project directory
            print("Extracting dataset from zip file...")
            zip_path = zip_files[0]
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to a temporary directory first
                temp_extract = os.path.join(project_dir, "temp_extract")
                zip_ref.extractall(temp_extract)
                
                # Find the Covid19-dataset folder in extracted files
                for root, dirs, files in os.walk(temp_extract):
                    if 'Covid19-dataset' in dirs:
                        source_dataset = os.path.join(root, 'Covid19-dataset')
                        # Move to project directory
                        if os.path.exists(target_dir):
                            shutil.rmtree(target_dir)
                        shutil.move(source_dataset, target_dir)
                        # Clean up temp directory
                        shutil.rmtree(temp_extract)
                        print(f"Dataset extracted to: {target_dir}")
                        break
                else:
                    # If Covid19-dataset not found, check for train/test folders directly
                    for root, dirs, files in os.walk(temp_extract):
                        if 'train' in dirs and 'test' in dirs:
                            # Create Covid19-dataset structure
                            os.makedirs(target_dir, exist_ok=True)
                            shutil.move(os.path.join(root, 'train'), 
                                      os.path.join(target_dir, 'train'))
                            shutil.move(os.path.join(root, 'test'), 
                                      os.path.join(target_dir, 'test'))
                            # Clean up temp directory
                            shutil.rmtree(temp_extract)
                            print(f"Dataset extracted to: {target_dir}")
                            break
        else:
            # Dataset is already extracted, copy it to project directory
            print("Copying dataset to project directory...")
            
            # Look for Covid19-dataset folder in cache
            covid_dataset_cache = os.path.join(cache_path, "Covid19-dataset")
            if os.path.exists(covid_dataset_cache):
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.copytree(covid_dataset_cache, target_dir)
                print(f"Dataset copied to: {target_dir}")
            else:
                # Check if cache_path itself contains train/test folders
                if os.path.exists(os.path.join(cache_path, "train")) and \
                   os.path.exists(os.path.join(cache_path, "test")):
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copytree(os.path.join(cache_path, "train"), 
                                  os.path.join(target_dir, "train"))
                    shutil.copytree(os.path.join(cache_path, "test"), 
                                  os.path.join(target_dir, "test"))
                    print(f"Dataset copied to: {target_dir}")
                else:
                    # Try to find the dataset structure
                    for item in os.listdir(cache_path):
                        item_path = os.path.join(cache_path, item)
                        if os.path.isdir(item_path):
                            # Check if this directory contains train/test
                            if 'train' in os.listdir(item_path):
                                if os.path.exists(target_dir):
                                    shutil.rmtree(target_dir)
                                shutil.copytree(item_path, target_dir)
                                print(f"Dataset copied to: {target_dir}")
                                break
        
        # Verify the dataset was copied correctly
        if os.path.exists(target_dir):
            train_dir = os.path.join(target_dir, "train")
            if os.path.exists(train_dir):
                classes = [item for item in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, item))]
                if classes:
                    print(f"Dataset successfully saved to: {target_dir}")
                    print(f"Found classes: {classes}")
                    return target_dir
        
        print(f"Warning: Dataset structure may be unexpected. Check: {target_dir}")
        return target_dir
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
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
            print(f"\n{'='*60}")
            print(f"Dataset ready at: {dataset_path}")
            print(f"{'='*60}")
            print("\nNext steps:")
            # Use relative path for easier usage
            if os.path.isabs(dataset_path):
                rel_path = os.path.relpath(dataset_path, os.path.dirname(__file__))
                if not rel_path.startswith('..'):
                    dataset_path_display = rel_path
                else:
                    dataset_path_display = dataset_path
            else:
                dataset_path_display = dataset_path
            
            print(f"\n1. Train with MLflow:")
            print(f"   python train_mlflow.py --dataset_path \"{dataset_path_display}\" --epochs 20")
            print(f"\n2. Train with W&B:")
            print(f"   python train_wandb.py --dataset_path \"{dataset_path_display}\" --epochs 20")
            print(f"\n3. Compare both:")
            print(f"   python compare_mlflow_wandb.py --dataset_path \"{dataset_path_display}\" --epochs 10")
            print(f"\nNote: You can also use just 'Covid19-dataset' as the path if running from the project directory.")
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
