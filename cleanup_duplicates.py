"""
Clean up duplicate files in the project root
Moves remaining files to their proper locations
"""

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Files to move from root to proper locations
MOVES = {
    # Scripts in root → scripts/
    'compare_mlflow_wandb.py': 'scripts/compare_mlflow_wandb.py',
    'run_hyperparameter_tuning.py': 'scripts/run_hyperparameter_tuning.py',
    'run_wandb_hyperparameter_tuning.py': 'scripts/run_wandb_hyperparameter_tuning.py',
    'start_mlflow_ui.py': 'scripts/start_mlflow_ui.py',
    'train_wandb.py': 'scripts/train_wandb.py',
    
    # Examples in root → examples/
    'example_wandb_usage.py': 'examples/example_wandb_usage.py',
    
    # Documentation in root → docs/
    'MLFLOW_QUICK_START.md': 'docs/mlflow/MLFLOW_QUICK_START.md',
    'HYPERPARAMETER_TUNING_GUIDE.md': 'docs/mlflow/HYPERPARAMETER_TUNING_GUIDE.md',
    'PARAMETER_MATRIX_EXAMPLE.md': 'docs/examples/PARAMETER_MATRIX_EXAMPLE.md',
    'WANDB_GUIDE.md': 'docs/wandb/WANDB_GUIDE.md',
    'WANDB_QUICK_START.md': 'docs/wandb/WANDB_QUICK_START.md',
    'WANDB_HYPERPARAMETER_TUNING_GUIDE.md': 'docs/wandb/WANDB_HYPERPARAMETER_TUNING_GUIDE.md',
    'WANDB_PARAMETER_MATRIX_EXAMPLE.md': 'docs/examples/WANDB_PARAMETER_MATRIX_EXAMPLE.md',
    'README_HYPERPARAMETER_TUNING.md': 'docs/README_HYPERPARAMETER_TUNING.md',
}

# Files to remove (temporary/organization scripts)
REMOVE = [
    'improve_structure.py',
    'organize_project.py',
    'cleanup_duplicates.py',  # Remove itself after running
]

def cleanup():
    """Clean up duplicate files"""
    print("Cleaning up duplicate files...")
    print("=" * 60)
    
    moved = 0
    for source, dest in MOVES.items():
        source_path = PROJECT_ROOT / source
        dest_path = PROJECT_ROOT / dest
        
        if source_path.exists():
            # Skip if destination already exists (and is different file)
            if dest_path.exists():
                if not source_path.samefile(dest_path):
                    print(f"⊘ Skipped (destination exists): {source}")
                    # Remove source if destination exists and is different
                    source_path.unlink()
                    print(f"  ✓ Removed duplicate: {source}")
                continue
            
            # Create destination directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(source_path), str(dest_path))
            print(f"✓ Moved: {source} → {dest}")
            moved += 1
    
    # Remove temporary files
    removed = 0
    for file in REMOVE:
        file_path = PROJECT_ROOT / file
        if file_path.exists() and file != 'cleanup_duplicates.py':
            file_path.unlink()
            print(f"✓ Removed: {file}")
            removed += 1
    
    print("\\n" + "=" * 60)
    print(f"Cleanup complete!")
    print(f"  - Moved {moved} files")
    print(f"  - Removed {removed} files")
    
    # Remove this script itself at the end
    if (PROJECT_ROOT / 'cleanup_duplicates.py').exists():
        print("\\nNote: Run this script again to remove it after verification.")

if __name__ == '__main__':
    cleanup()

