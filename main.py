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
    # Get the project directory (where this script is located)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(project_dir, "Covid19-dataset")
    
    # Check if dataset already exists in project directory
    if os.path.exists(target_dir):
        print(f"Dataset already exists in project directory: {target_dir}")
        # Verify it has the train folder with classes
        train_dir = os.path.join(target_dir, "train")
        if os.path.exists(train_dir):
            try:
                classes = [item for item in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, item))]
                if any(c.lower() in ['covid', 'normal', 'viral'] for c in classes):
                    print("Dataset appears to be complete. Using existing dataset.")
                    return target_dir
            except (OSError, PermissionError):
                pass
        print("Dataset folder exists but appears incomplete. Re-downloading...")
        import shutil
        try:
            shutil.rmtree(target_dir)
        except Exception as e:
            print(f"Warning: Could not remove existing directory: {e}")
    
    print("Downloading COVID-19 Image Dataset from Kaggle...")
    
    # Try using kaggle API first (more reliable)
    kaggle_api_available = False
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        kaggle_api_available = True
        
        print("Using Kaggle API to download dataset...")
        # Create a temporary download directory
        temp_download = os.path.join(project_dir, "temp_kaggle_download")
        os.makedirs(temp_download, exist_ok=True)
        
        try:
            # Download the dataset
            api.dataset_download_files(
                "pranavraikokte/covid19-image-dataset",
                path=temp_download,
                unzip=True
            )
            
            # Find and move the dataset to target directory
            import shutil
            import zipfile
            import glob
            
            # Look for zip files first (in case unzip=False or didn't work)
            zip_files = glob.glob(os.path.join(temp_download, "*.zip"))
            if zip_files:
                print("Extracting zip file...")
                with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                    zip_ref.extractall(temp_download)
            
            # Find the dataset structure
            dataset_found = False
            for root, dirs, files in os.walk(temp_download):
                if 'Covid19-dataset' in dirs:
                    source = os.path.join(root, 'Covid19-dataset')
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                    shutil.move(source, target_dir)
                    dataset_found = True
                    break
                elif 'train' in dirs and 'test' in dirs:
                    # Create Covid19-dataset structure
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(os.path.join(root, 'train'), 
                              os.path.join(target_dir, 'train'))
                    shutil.move(os.path.join(root, 'test'), 
                              os.path.join(target_dir, 'test'))
                    dataset_found = True
                    break
            
            if dataset_found:
                print(f"Dataset downloaded and saved to: {target_dir}")
            else:
                raise FileNotFoundError("Could not find dataset structure in downloaded files")
            
            # Clean up temp directory
            if os.path.exists(temp_download):
                shutil.rmtree(temp_download)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_download):
                import shutil
                shutil.rmtree(temp_download, ignore_errors=True)
            raise e
            
    except ImportError:
        kaggle_api_available = False
        print("Kaggle API package not found.")
    except OSError as e:
        # Kaggle API authentication error
        kaggle_api_available = False
        if "kaggle.json" in str(e):
            print("Kaggle API credentials not found.")
            print("To use Kaggle API, please:")
            print("1. Go to https://www.kaggle.com/ and sign in")
            print("2. Go to Account → API → Create New Token")
            print("3. Download kaggle.json")
            print("4. Place it in: C:\\Users\\pc\\.kaggle\\kaggle.json")
            print("5. Set permissions: chmod 600 kaggle.json (on Linux/Mac)")
            print("\nFalling back to kagglehub...")
        else:
            raise e
    except Exception as e:
        if kaggle_api_available:
            print(f"Error with Kaggle API: {e}")
            print("Falling back to kagglehub...")
        else:
            raise e
            
    
    # Fall back to kagglehub if kaggle API is not available or failed
    if not kaggle_api_available:
        try:
            import kagglehub
            import shutil
            import zipfile
            import glob
            
            print("Trying kagglehub to download dataset...")
            # Download using kagglehub
            cache_path = kagglehub.dataset_download("pranavraikokte/covid19-image-dataset")
            print(f"Dataset downloaded to cache: {cache_path}")
            
            # Check if cache_path exists and has content
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Cache path does not exist: {cache_path}")
            
            try:
                cache_items = os.listdir(cache_path)
            except (OSError, PermissionError):
                cache_items = []
            
            if len(cache_items) == 0:
                # Check parent and sibling directories - kagglehub might download to a different location
                print("Cache directory is empty, searching for dataset in parent directories...")
                parent = os.path.dirname(cache_path)
                dataset_found_in_cache = False
                
                if os.path.exists(parent):
                    # Search recursively in parent directory
                    for root, dirs, files in os.walk(parent):
                        # Skip if we've gone too deep (more than 3 levels from parent)
                        depth = root[len(parent):].count(os.sep)
                        if depth > 3:
                            continue
                            
                        if 'Covid19-dataset' in dirs:
                            source = os.path.join(root, 'Covid19-dataset')
                            if os.path.exists(os.path.join(source, 'train')):
                                if os.path.exists(target_dir):
                                    shutil.rmtree(target_dir)
                                shutil.copytree(source, target_dir)
                                print(f"Dataset found and copied to: {target_dir}")
                                dataset_found_in_cache = True
                                break
                        elif 'train' in dirs:
                            train_path = os.path.join(root, 'train')
                            # Verify it has class folders
                            try:
                                train_classes = os.listdir(train_path)
                                if any(c.lower() in ['covid', 'normal', 'viral'] for c in train_classes):
                                    # This looks like our dataset
                                    os.makedirs(target_dir, exist_ok=True)
                                    shutil.copytree(train_path, os.path.join(target_dir, 'train'))
                                    # Check for test folder in same directory
                                    if 'test' in dirs:
                                        shutil.copytree(os.path.join(root, 'test'), 
                                                      os.path.join(target_dir, 'test'))
                                    print(f"Dataset found and copied to: {target_dir}")
                                    dataset_found_in_cache = True
                                    break
                            except (OSError, PermissionError):
                                continue
                
                if not dataset_found_in_cache:
                    raise FileNotFoundError(
                        f"Dataset not found. kagglehub cache directory is empty: {cache_path}\n"
                        "Please try:\n"
                        "1. Manually download the dataset from: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset\n"
                        "2. Extract it to the project directory as 'Covid19-dataset'\n"
                        "3. Or set up Kaggle API credentials (see instructions above)"
                    )
            else:
                # Cache has content, process it
                dataset_copied = False
                
                # Check for zip files
                zip_files = glob.glob(os.path.join(cache_path, "*.zip"))
                if zip_files:
                    print("Found zip file, extracting...")
                    temp_extract = os.path.join(project_dir, "temp_extract")
                    os.makedirs(temp_extract, exist_ok=True)
                    try:
                        with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                            zip_ref.extractall(temp_extract)
                        
                        # Find dataset structure in extracted files
                        for root, dirs, files in os.walk(temp_extract):
                            if 'Covid19-dataset' in dirs:
                                source = os.path.join(root, 'Covid19-dataset')
                                if os.path.exists(target_dir):
                                    shutil.rmtree(target_dir)
                                shutil.move(source, target_dir)
                                dataset_copied = True
                                break
                            elif 'train' in dirs and 'test' in dirs:
                                os.makedirs(target_dir, exist_ok=True)
                                shutil.move(os.path.join(root, 'train'), 
                                          os.path.join(target_dir, 'train'))
                                shutil.move(os.path.join(root, 'test'), 
                                          os.path.join(target_dir, 'test'))
                                dataset_copied = True
                                break
                    finally:
                        # Clean up temp directory
                        if os.path.exists(temp_extract):
                            shutil.rmtree(temp_extract, ignore_errors=True)
                
                if not dataset_copied:
                    # Try copying directly from cache
                    if os.path.exists(os.path.join(cache_path, "Covid19-dataset")):
                        source = os.path.join(cache_path, "Covid19-dataset")
                        if os.path.exists(target_dir):
                            shutil.rmtree(target_dir)
                        shutil.copytree(source, target_dir)
                        dataset_copied = True
                    elif os.path.exists(os.path.join(cache_path, "train")):
                        os.makedirs(target_dir, exist_ok=True)
                        shutil.copytree(os.path.join(cache_path, "train"), 
                                      os.path.join(target_dir, "train"))
                        if os.path.exists(os.path.join(cache_path, "test")):
                            shutil.copytree(os.path.join(cache_path, "test"), 
                                          os.path.join(target_dir, "test"))
                        dataset_copied = True
                    else:
                        # Try to find dataset in subdirectories
                        for item in cache_items:
                            item_path = os.path.join(cache_path, item)
                            if os.path.isdir(item_path):
                                try:
                                    sub_items = os.listdir(item_path)
                                    if 'train' in sub_items or 'Covid19-dataset' in sub_items:
                                        if os.path.exists(target_dir):
                                            shutil.rmtree(target_dir)
                                        shutil.copytree(item_path, target_dir)
                                        dataset_copied = True
                                        break
                                except (OSError, PermissionError):
                                    continue
                    
                    if not dataset_copied:
                        raise FileNotFoundError(
                            f"Could not find dataset structure in cache: {cache_path}\n"
                            f"Found items: {cache_items[:10]}"
                        )
                                    
        except ImportError:
            print("\n" + "="*60)
            print("ERROR: Neither 'kaggle' nor 'kagglehub' package is installed.")
            print("="*60)
            print("Please install one of them:")
            print("  pip install kaggle")
            print("  OR")
            print("  pip install kagglehub")
            print("Or install all requirements: pip install -r requirements.txt")
            print("\nAlternatively, manually download the dataset from:")
            print("  https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset")
            print("And extract it to: Covid19-dataset/")
            return None
        except Exception as e:
            print(f"\nError with kagglehub: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "="*60)
            print("MANUAL DOWNLOAD INSTRUCTIONS:")
            print("="*60)
            print("1. Go to: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset")
            print("2. Click 'Download' button")
            print("3. Extract the zip file")
            print(f"4. Copy the 'Covid19-dataset' folder to: {project_dir}")
            print("5. The folder should contain 'train' and 'test' subdirectories")
            return None
    
    # Verify the dataset was downloaded correctly
    if os.path.exists(target_dir):
        train_dir = os.path.join(target_dir, "train")
        if os.path.exists(train_dir):
            try:
                classes = [item for item in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, item))]
                if classes:
                    print(f"\n✓ Dataset successfully downloaded to: {target_dir}")
                    print(f"✓ Found classes: {classes}")
                    # Count images
                    total_images = 0
                    for class_name in classes:
                        class_dir = os.path.join(train_dir, class_name)
                        images = [f for f in os.listdir(class_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                        total_images += len(images)
                        print(f"  - {class_name}: {len(images)} images")
                    print(f"✓ Total training images: {total_images}")
                    return target_dir
            except Exception as e:
                print(f"Warning: Could not verify dataset structure: {e}")
    
    print(f"\n⚠ Warning: Dataset may not have been downloaded correctly.")
    print(f"Please check: {target_dir}")
    if os.path.exists(target_dir):
        return target_dir
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
  python scripts/train_mlflow.py --dataset_path <path_to_dataset> --epochs 20
  
  # Train with W&B
  python scripts/train_wandb.py --dataset_path <path_to_dataset> --epochs 20
  
  # Compare MLflow vs W&B
  python scripts/compare_mlflow_wandb.py --dataset_path <path_to_dataset> --epochs 10
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
            print(f"   python scripts/train_mlflow.py --dataset_path \"{dataset_path_display}\" --epochs 20")
            print(f"\n2. Train with W&B:")
            print(f"   python scripts/train_wandb.py --dataset_path \"{dataset_path_display}\" --epochs 20")
            print(f"\n3. Compare both:")
            print(f"   python scripts/compare_mlflow_wandb.py --dataset_path \"{dataset_path_display}\" --epochs 10")
            print(f"\n4. Hyperparameter Tuning (MLflow):")
            print(f"   python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml")
            print(f"\n5. Hyperparameter Tuning (W&B):")
            print(f"   python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml")
            print(f"\nNote: You can also use just 'Covid19-dataset' as the path if running from the project directory.")
    else:
        print("COVID-19 Chest X-Ray Classification Project")
        print("=" * 60)
        print("\nThis project evaluates MLflow vs W&B for Chest X-Ray Classification")
        print("\nDataset: COVID-19 Image Dataset")
        print("Classes: COVID-19, Viral Pneumonia, Normal")
        print("\nAvailable scripts:")
        print("1. scripts/train_mlflow.py - Train model with MLflow tracking")
        print("2. scripts/train_wandb.py - Train model with W&B tracking")
        print("3. scripts/compare_mlflow_wandb.py - Compare both tracking tools")
        print("4. scripts/run_hyperparameter_tuning.py - MLflow hyperparameter tuning")
        print("5. scripts/run_wandb_hyperparameter_tuning.py - W&B hyperparameter tuning")
        print("\nTo download the dataset, run: python main.py --download")
        print("\nFor help with any script, use: python <script_path> --help")
        print("\nSee README.md for detailed usage instructions.")


if __name__ == '__main__':
    main()
