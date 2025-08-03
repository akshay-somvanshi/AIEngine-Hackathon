#!/usr/bin/env python3
"""
Hybrid Coffee Recommendation Model Training Script

This script fine-tunes a language model with embedding integration for coffee recommendations.
Combines semantic similarity matching with natural language generation.
"""

import json
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import os
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    logger.info("Using Apple MPS device")
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU device")


class HybridCoffeeRecommender(nn.Module):
    """
    Hybrid model that combines embeddings with language model capabilities.
    Uses embeddings for semantic matching and LLM for natural response generation.
    """

    def __init__(
        self,
        base_model_name: str = "microsoft/DialoGPT-small",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        hidden_dim: int = 512,
    ):
        super().__init__()

        # Load base language model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.language_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Embedding model for semantic matching
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Embedding processing layers
        self.embedding_projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fusion layer to combine embeddings with text features
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + self.language_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.language_model.config.hidden_size),
        )

        # Similarity prediction head
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the language model."""
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the language model."""
        self.language_model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query_embeddings: torch.Tensor,
        coffee_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):

        # Process embeddings
        query_features = self.embedding_projector(query_embeddings)
        coffee_features = self.embedding_projector(coffee_embeddings)

        # Compute similarity score
        combined_features = torch.cat([query_features, coffee_features], dim=-1)
        similarity_score = self.similarity_head(combined_features)

        # Get language model output
        lm_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # Get the last hidden state for fusion
        last_hidden_state = lm_outputs.hidden_states[-1]

        # Combine embedding features with text features
        # Use the first token's hidden state for fusion
        text_features = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        combined_features_for_fusion = torch.cat(
            [query_features, text_features], dim=-1
        )
        fusion_features = self.fusion_layer(combined_features_for_fusion)

        # Return combined outputs
        outputs = {
            "loss": lm_outputs.loss,
            "logits": lm_outputs.logits,
            "similarity_score": similarity_score,
            "hidden_states": lm_outputs.hidden_states,
        }

        # Add similarity loss if in training mode
        if labels is not None and self.training:
            # Similarity loss (higher similarity for positive examples)
            similarity_loss = nn.MSELoss()(
                similarity_score.squeeze(), torch.ones_like(similarity_score.squeeze())
            )
            outputs["loss"] = outputs["loss"] + 0.1 * similarity_loss

        return outputs


def validate_training_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and clean training data."""
    valid_data = []
    required_fields = ["input", "output"]

    for i, example in enumerate(data):
        try:
            # Check required fields
            for field in required_fields:
                if field not in example:
                    logger.warning(f"Example {i} missing field: {field}")
                    continue

            # Validate JSON format
            if isinstance(example["input"], str):
                input_data = json.loads(example["input"])
            else:
                input_data = example["input"]

            if isinstance(example["output"], str):
                output_data = json.loads(example["output"])
            else:
                output_data = example["output"]

            # Ensure we have a query
            query = input_data.get("query", "")
            if not query or len(query.strip()) < 5:
                logger.warning(f"Example {i} has invalid query: {query}")
                continue

            # Ensure we have a recommendation
            recommendation = output_data.get("recommendation", {})
            coffee_name = recommendation.get("name", "")
            if not coffee_name:
                logger.warning(f"Example {i} missing coffee name")
                continue

            valid_data.append(example)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Example {i} has invalid format: {e}")
            continue

    logger.info(f"Validated {len(valid_data)} examples out of {len(data)}")
    return valid_data


def format_hybrid_prompt(example: Dict[str, Any], embedding_model) -> Dict[str, Any]:
    """Format training example with hybrid prompt structure."""

    try:
        # Parse input and output
        if isinstance(example["input"], str):
            input_data = json.loads(example["input"])
        else:
            input_data = example["input"]

        if isinstance(example["output"], str):
            output_data = json.loads(example["output"])
        else:
            output_data = example["output"]

        query = input_data.get("query", "")
        recommendation = output_data.get("recommendation", {})

        coffee_name = recommendation.get("name", "Unknown Coffee")
        coffee_roaster = recommendation.get("roaster", "Unknown Roaster")
        coffee_origin = recommendation.get("origin", "Unknown Origin")
        coffee_description = recommendation.get("description", "")
        why_recommended = recommendation.get(
            "why_recommended", "Good match for your preferences"
        )

        # Generate embeddings for query and coffee description
        query_embedding = embedding_model.encode([query])[0].tolist()
        coffee_text = f"{coffee_name} {coffee_description}"
        coffee_embedding = embedding_model.encode([coffee_text])[0].tolist()

        # Create a simple prompt format for training
        prompt_text = f"""User: {query}

Assistant: Based on your preferences, I recommend {coffee_name} by {coffee_roaster}.

Origin: {coffee_origin}
Description: {coffee_description}

Why this coffee: {why_recommended}<|endoftext|>"""

        return {
            "text": prompt_text,
            "query": query,
            "coffee_name": coffee_name,
            "query_embedding": query_embedding,
            "coffee_embedding": coffee_embedding,
        }

    except Exception as e:
        logger.error(f"Error formatting example: {e}")
        return None


def prepare_hybrid_dataset(
    data: List[Dict[str, Any]], tokenizer, embedding_model
) -> Dataset:
    """Prepare dataset with hybrid processing."""
    formatted_data = []

    logger.info("Formatting hybrid training data...")
    for i, example in enumerate(data):
        try:
            formatted_example = format_hybrid_prompt(example, embedding_model)
            if formatted_example:
                formatted_data.append(formatted_example)
        except Exception as e:
            logger.warning(f"Error processing example {i}: {e}")
            continue

    logger.info(f"Prepared {len(formatted_data)} valid examples")

    if len(formatted_data) == 0:
        raise ValueError("No valid training examples found!")

    # Create dataset
    dataset = Dataset.from_list(formatted_data)

    # Tokenize the text
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,  # Reduced for memory efficiency
            return_tensors="pt",
        )

        # Add embeddings as additional features
        tokenized["query_embeddings"] = examples["query_embedding"]
        tokenized["coffee_embeddings"] = examples["coffee_embedding"]

        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "query", "coffee_name"]
    )

    return tokenized_dataset


class HybridDataCollator:
    """Custom data collator for hybrid training."""

    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Extract embeddings
        query_embeddings = []
        coffee_embeddings = []

        for feature in features:
            query_embeddings.append(feature.pop("query_embeddings"))
            coffee_embeddings.append(feature.pop("coffee_embeddings"))

        # Standard tokenizer collation for text
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=256,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Add embeddings to batch
        batch["query_embeddings"] = torch.tensor(query_embeddings, dtype=torch.float32)
        batch["coffee_embeddings"] = torch.tensor(
            coffee_embeddings, dtype=torch.float32
        )

        # Set labels for language modeling
        batch["labels"] = batch["input_ids"].clone()

        return batch


class HybridTrainer(Trainer):
    """Custom trainer for hybrid coffee recommendation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation."""

        # Extract embeddings from inputs
        query_embeddings = inputs.pop("query_embeddings")
        coffee_embeddings = inputs.pop("coffee_embeddings")

        # Forward pass with embeddings
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            query_embeddings=query_embeddings,
            coffee_embeddings=coffee_embeddings,
            labels=inputs.get("labels"),
        )

        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss


def create_hybrid_model_and_tokenizer(model_name: str = "microsoft/DialoGPT-small"):
    """Create hybrid model and tokenizer."""
    try:
        logger.info(f"Creating hybrid model: {model_name}")

        # Create hybrid model
        model = HybridCoffeeRecommender(base_model_name=model_name)
        tokenizer = model.tokenizer

        # Add special tokens
        special_tokens = ["<|user|>", "<|assistant|>"]
        tokenizer.add_tokens(special_tokens)
        model.language_model.resize_token_embeddings(len(tokenizer))

        logger.info("Hybrid model and tokenizer created successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error creating hybrid model: {e}")
        raise


def train_hybrid_model(
    model,
    tokenizer,
    train_dataset,
    output_dir: str = "models/coffee_hybrid_recommender",
    num_epochs: int = 5,
    batch_size: int = 8,  # Increased for better GPU utilization
    learning_rate: float = 1e-5,
):
    """Train the hybrid model."""
    try:
        # Move model to device
        model.to(DEVICE)
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)"
        )
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Reduced since batch_size increased
            warmup_steps=50,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_steps=200,
            save_strategy="steps",
            load_best_model_at_end=False,
            fp16=torch.cuda.is_available(),  # Enable fp16 for CUDA
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            dataloader_pin_memory=torch.cuda.is_available(),
            remove_unused_columns=False,
            report_to=None,
            gradient_checkpointing=True,
            optim="adamw_torch",
            # For multi-GPU, launch with torchrun or accelerate
        )
        # Custom data collator for embeddings
        data_collator = HybridDataCollator(tokenizer=tokenizer)
        # Initialize custom trainer
        trainer = HybridTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        # Train the model
        logger.info("Starting hybrid training...")
        trainer.train()
        # Save the model
        logger.info(f"Saving hybrid model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        # Save the full model state
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "base_model_name": model.language_model.name_or_path,
                    "embedding_model_name": "all-MiniLM-L6-v2",
                    "embedding_dim": 384,
                    "hidden_dim": 512,
                },
            },
            f"{output_dir}/full_model.pt",
        )
        logger.info(f"Hybrid model saved to {output_dir}")
        return trainer
    except Exception as e:
        logger.error(f"Hybrid training failed: {e}")
        raise


def test_hybrid_model(
    model, tokenizer, embedding_model, test_query: str = "I need an energizing coffee"
):
    """Test the trained hybrid model."""
    logger.info(f"Testing hybrid model with query: '{test_query}'")
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([test_query])[0]
        query_embedding_tensor = torch.tensor(query_embedding).unsqueeze(0).to(DEVICE)
        # Format test input
        test_prompt = f"""User: {test_query}\n\nAssistant: Based on your preferences, I recommend"""
        # Tokenize
        inputs = tokenizer(
            test_prompt, return_tensors="pt", truncation=True, max_length=128
        )
        # Move to same device as model
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        # Create dummy coffee embedding for testing
        dummy_coffee_embedding = torch.randn(1, 384).to(DEVICE)
        # Generate response (with autocast for mixed precision)
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model.language_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Model response:")
        logger.info(response[len(test_prompt) :])
    except Exception as e:
        logger.error(f"Error testing hybrid model: {e}")


def load_training_data(
    file_path: str = "data/llm_training_data.json",
) -> List[Dict[str, Any]]:
    """Load and validate training data."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} training examples")

        # Validate data
        valid_data = validate_training_data(data)

        if len(valid_data) == 0:
            raise ValueError("No valid training data found!")

        return valid_data

    except FileNotFoundError:
        logger.error(f"Training data file {file_path} not found.")
        logger.info("Please run the data preprocessing script first.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in training data: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return []


def main():
    """Main training function for hybrid coffee recommendation."""
    logger.info("Starting hybrid coffee recommendation model training...")
    logger.info("=" * 70)

    try:
        # Load training data
        training_data = load_training_data()
        if not training_data:
            logger.error(
                "No training data found. Please run the data preprocessing script first."
            )
            return

        # Create hybrid model and tokenizer
        model, tokenizer = create_hybrid_model_and_tokenizer()

        # Load embedding model
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Prepare dataset
        logger.info("Preparing hybrid dataset...")
        train_dataset = prepare_hybrid_dataset(
            training_data, tokenizer, embedding_model
        )

        logger.info(f"Training on {len(train_dataset)} examples")

        # Train model
        trainer = train_hybrid_model(model, tokenizer, train_dataset)

        logger.info("Training Summary:")
        logger.info("- Model: Hybrid (DialoGPT-small + Embeddings)")
        logger.info("- Training strategy: Embedding-aware fine-tuning")
        logger.info("- Batch size: 1 (with gradient accumulation)")
        logger.info("- Memory optimizations: gradient checkpointing, fp16")

        # Test the model
        test_hybrid_model(model, tokenizer, embedding_model)

        logger.info("\n" + "=" * 70)
        logger.info("HYBRID TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("Your model can now use embeddings for semantic matching.")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("Please check the error message above and fix the issue.")
        raise


if __name__ == "__main__":
    main()
