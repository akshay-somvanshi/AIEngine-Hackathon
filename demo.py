#!/usr/bin/env python3
"""
Coffee Recommendation System Demo

This script demonstrates the complete workflow of the coffee recommendation system.
"""

import os
import json
from pathlib import Path


def check_dataset():
    """Check if the dataset is available."""
    print("🔍 Checking dataset availability...")

    # Check for Kaggle dataset
    kaggle_dataset = Path("data/coffee_reviews.csv")
    if kaggle_dataset.exists():
        print("✓ Kaggle dataset found")
        return "kaggle"

    # Check for synthetic dataset
    synthetic_dataset = Path("data/coffee_dataset.json")
    if synthetic_dataset.exists():
        print("✓ Synthetic dataset found")
        return "synthetic"

    print("✗ No dataset found")
    return None


def show_dataset_info(dataset_type):
    """Show information about the available dataset."""
    print(f"\n📊 Dataset Information:")
    print("=" * 40)

    if dataset_type == "kaggle":
        print("• Real Coffee Reviews Dataset from Kaggle")
        print("• 2,000+ coffee reviews from various brands")
        print("• Contains review text, ratings, origins, flavors")
        print("• High-quality real-world data")
    else:
        print("• Synthetic Coffee Dataset")
        print("• 2,000 generated coffee recommendations")
        print("• Covers various moods and preferences")
        print("• Structured for fine-tuning")

    print()


def show_example_queries():
    """Show example queries for testing."""
    print("💡 Example Queries for Testing:")
    print("=" * 40)

    examples = [
        "I'm feeling stressed and need something energizing with bold flavors",
        "I'm tired and want a smooth, mild coffee to relax with",
        "I need to focus on work and want moderate caffeine",
        "I'm in a creative mood and want something unique",
        "I'm meeting friends and want a social coffee",
        "I'm adventurous and want to try something traditional",
        "I'm productive and need sustained energy",
        "I'm relaxed and want something creamy and smooth",
    ]

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")

    print()


def show_training_info():
    """Show information about the training process."""
    print("🧠 Training Information:")
    print("=" * 40)
    print("• Base Model: Microsoft DialoGPT-medium (355M parameters)")
    print("• Fine-tuning: Instruction-following format")
    print("• Training Time: ~30-60 minutes (GPU), ~2-4 hours (CPU)")
    print("• Output: Structured coffee recommendations")
    print()


def show_usage_instructions():
    """Show usage instructions."""
    print("🚀 Usage Instructions:")
    print("=" * 40)
    print("1. Setup Environment:")
    print("   python setup.py")
    print()
    print("2. Download Dataset (if using Kaggle):")
    print(
        "   - Visit: https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset"
    )
    print("   - Download and place in data/coffee_reviews.csv")
    print()
    print("3. Preprocess Data:")
    print("   python src/data_preprocessing.py  # For Kaggle dataset")
    print("   python src/data_generation.py     # For synthetic dataset")
    print()
    print("4. Train Model:")
    print("   python src/training.py")
    print()
    print("5. Test Recommendations:")
    print("   python src/inference.py")
    print()
    print("6. Launch Web Interface:")
    print("   python src/web_app.py")
    print()


def show_project_structure():
    """Show the project structure."""
    print("📁 Project Structure:")
    print("=" * 40)

    structure = """
├── data/                          # Dataset files
│   ├── coffee_reviews.csv         # Kaggle dataset
│   ├── coffee_dataset.json        # Synthetic dataset
│   └── training_data.json         # Training format
├── src/                           # Source code
│   ├── data_preprocessing.py      # Process Kaggle data
│   ├── data_generation.py         # Generate synthetic data
│   ├── training.py               # Fine-tune model
│   ├── inference.py              # Generate recommendations
│   └── web_app.py               # Web interface
├── models/                        # Trained models
│   └── coffee_recommender/       # Fine-tuned model
├── setup.py                      # Setup script
├── demo.py                       # This demo script
├── requirements.txt              # Dependencies
└── README.md                    # Documentation
"""

    print(structure)


def main():
    """Main demo function."""
    print("☕ Coffee Recommendation LLM Fine-tuning Demo")
    print("=" * 60)
    print()

    # Check dataset
    dataset_type = check_dataset()

    if dataset_type:
        show_dataset_info(dataset_type)
    else:
        print(
            "⚠️  No dataset found. Please download the Kaggle dataset or generate synthetic data."
        )
        print()

    # Show project information
    show_project_structure()
    show_training_info()
    show_example_queries()
    show_usage_instructions()

    print("🎯 Next Steps:")
    print("=" * 40)
    if dataset_type == "kaggle":
        print("1. Run: python src/data_preprocessing.py")
    else:
        print("1. Dataset is ready for training")
    print("2. Run: python src/training.py")
    print("3. Run: python src/inference.py")
    print("4. Run: python src/web_app.py")
    print()
    print("✨ Enjoy building your coffee recommendation system!")


if __name__ == "__main__":
    main()
