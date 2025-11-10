"""
Setup script for Chest X-Ray Classification Project

This package provides tools for training and evaluating deep learning models
for COVID-19 chest X-ray classification with experiment tracking using
MLflow and Weights & Biases.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
with open(readme_file, "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
with open(requirements_file, "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

# Read development requirements
dev_requirements_file = Path(__file__).parent / "requirements-dev.txt"
try:
    with open(dev_requirements_file, "r", encoding="utf-8") as fh:
        dev_requirements = [
            line.strip()
            for line in fh
            if line.strip() and not line.startswith("#")
        ]
except FileNotFoundError:
    dev_requirements = []

setup(
    name="covid-xray-classification",
    version="1.0.0",
    author="Masters Lecture Project Contributors",
    author_email="your.email@example.com",
    description=(
        "Production-grade evaluation of MLflow vs. W&B for "
        "COVID-19 Chest X-Ray Classification"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification/issues",
        "Documentation": "https://github.com/yourusername/Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification/docs",
        "Source Code": "https://github.com/yourusername/Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    keywords=[
        "machine-learning",
        "deep-learning",
        "pytorch",
        "mlflow",
        "wandb",
        "experiment-tracking",
        "covid19",
        "chest-xray",
        "medical-imaging",
        "classification",
        "computer-vision",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-mlflow=scripts.train_mlflow:main",
            "train-wandb=scripts.train_wandb:main",
            "compare-tracking=scripts.compare_mlflow_wandb:main",
            "run-mlflow-ui=scripts.start_mlflow_ui:main",
            "tune-hyperparams-mlflow=scripts.run_hyperparameter_tuning:main",
            "tune-hyperparams-wandb=scripts.run_wandb_hyperparameter_tuning:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
    license="MIT",
)

