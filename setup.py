#!/usr/bin/env python3
"""
Setup script for Coffee Recommendation LLM Fine-tuning Project

This script helps set up the environment and download the required dataset.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False
    return True


def create_directories():
    """Create necessary directories."""
    directories = ["data", "models", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✓ Directories created")


def download_dataset():
    """Download the coffee dataset from Kaggle."""
    print("\nDataset Setup:")
    print("=" * 50)
    print("This project uses the Coffee Reviews Dataset from Kaggle.")
    print("Please follow these steps to download the dataset:")
    print()
    print("1. Visit: https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset")
    print("2. Download the dataset (coffee-reviews-dataset.zip)")
    print("3. Extract the zip file")
    print("4. Place the CSV file in the 'data' directory as 'coffee_reviews.csv'")
    print()

    # Check if dataset exists
    dataset_path = Path("data/coffee_reviews.csv")
    if dataset_path.exists():
        print("✓ Dataset found at data/coffee_reviews.csv")
        return True
    else:
        print(
            "✗ Dataset not found. Please download and place it in the data directory."
        )
        return False


def run_data_preprocessing():
    """Run the data preprocessing script."""
    print("\nRunning data preprocessing...")
    try:
        subprocess.check_call([sys.executable, "src/data_preprocessing.py"])
        print("✓ Data preprocessing completed")
        return True
    except subprocess.CalledProcessError:
        print("✗ Data preprocessing failed")
        return False


def check_gpu():
    """Check if GPU is available for training."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ GPU not available. Training will use CPU (slower)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed. GPU check skipped.")
        return False


def main():
    """Main setup function."""
    print("Coffee Recommendation LLM Fine-tuning Setup")
    print("=" * 50)

    # Check Python version
    check_python_version()

    # Create directories
    create_directories()

    # Install dependencies
    if not install_dependencies():
        print("Setup failed. Please check the error messages above.")
        return

    # Check GPU
    check_gpu()

    # Download dataset
    dataset_ready = download_dataset()

    if dataset_ready:
        # Run preprocessing
        if run_data_preprocessing():
            print("\n" + "=" * 50)
            print("✓ Setup completed successfully!")
            print("\nNext steps:")
            print("1. Train the model: python src/training.py")
            print("2. Test the model: python src/inference.py")
            print("3. Launch web interface: python src/web_app.py")
        else:
            print("\n✗ Setup failed during data preprocessing")
    else:
        print("\n⚠ Setup completed but dataset needs to be downloaded manually")
        print("Please download the dataset and run: python src/data_preprocessing.py")


if __name__ == "__main__":
    main()
