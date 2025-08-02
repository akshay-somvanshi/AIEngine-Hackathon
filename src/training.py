#!/usr/bin/env python3
"""
Coffee Recommendation Model Training Script

This script fine-tunes a language model to recommend coffee based on
user preferences and mood using the processed coffee dataset.
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import os
from typing import Dict, List, Any


def load_training_data(
    file_path: str = "data/training_data.json",
) -> List[Dict[str, str]]:
    """Load the training data."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} training examples")
        return data
    except FileNotFoundError:
        print(f"Training data file {file_path} not found.")
        print("Please run data_preprocessing.py first.")
        return []


def format_prompt(example: Dict[str, str]) -> str:
    """Format the training example as a prompt."""
    instruction = example.get(
        "instruction", "Recommend coffee based on preferences and mood"
    )
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    # Format as instruction-following prompt
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"

    return prompt


def prepare_dataset(data: List[Dict[str, str]], tokenizer) -> Dataset:
    """Prepare the dataset for training."""
    formatted_data = []

    for example in data:
        prompt = format_prompt(example)
        formatted_data.append({"text": prompt})

    # Create dataset
    dataset = Dataset.from_list(formatted_data)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    return tokenized_dataset


def create_model_and_tokenizer(model_name: str = "microsoft/DialoGPT-medium"):
    """Create model and tokenizer."""
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def train_model(
    model,
    tokenizer,
    train_dataset,
    output_dir: str = "models/coffee_recommender",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
):
    """Train the model."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")

    return trainer


def main():
    """Main training function."""
    print("Starting coffee recommendation model training...")

    # Load training data
    training_data = load_training_data()
    if not training_data:
        return

    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer()

    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = prepare_dataset(training_data, tokenizer)

    # Train model
    trainer = train_model(model, tokenizer, train_dataset)

    print("Training completed successfully!")
    print("You can now use the model for coffee recommendations.")


if __name__ == "__main__":
    main()
