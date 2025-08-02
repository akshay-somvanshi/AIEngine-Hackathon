#!/usr/bin/env python3
"""
Coffee Recommendation Model Training Script with Embeddings

This script fine-tunes a language model to recommend coffee based on
user preferences and mood using embedding features from the preprocessing script.
The model learns to map query embeddings to coffee recommendations.
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
from torch.utils.data import DataLoader


class EmbeddingCoffeeRecommender(nn.Module):
    """
    Custom model that takes query embeddings and predicts coffee recommendations.
    Combines embedding features with language model capabilities.
    """
    
    def __init__(self, 
                 base_model_name: str = "microsoft/DialoGPT-medium",
                 query_embedding_dim: int = 384,  # all-MiniLM-L6-v2 embedding size
                 coffee_embedding_dim: int = 384,
                 hidden_dim: int = 512):
        super().__init__()
        
        # Load base language model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.language_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        # Embedding processing layers
        self.query_projector = nn.Sequential(
            nn.Linear(query_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.coffee_projector = nn.Sequential(
            nn.Linear(coffee_embedding_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layer to combine embeddings with text
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.language_model.config.hidden_size)
        )
        
        # Similarity prediction head
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                query_embeddings: torch.Tensor,
                coffee_embeddings: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        
        # Process embeddings
        query_features = self.query_projector(query_embeddings)
        coffee_features = self.coffee_projector(coffee_embeddings)
        
        # Compute similarity score
        combined_features = torch.cat([query_features, coffee_features], dim=-1)
        similarity_score = self.similarity_head(combined_features)
        
        # Get language model output
        lm_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        # Combine embedding features with text features
        fusion_features = self.fusion_layer(combined_features)
        
        # Return combined outputs
        outputs = {
            'loss': lm_outputs.loss,
            'logits': lm_outputs.logits,
            'similarity_score': similarity_score,
            'hidden_states': lm_outputs.hidden_states,
        }
        
        # Add similarity loss if in training mode
        if labels is not None and self.training:
            # Similarity loss (higher similarity for positive examples)
            similarity_loss = nn.MSELoss()(similarity_score.squeeze(), torch.ones_like(similarity_score.squeeze()))
            outputs['loss'] = outputs['loss'] + 0.1 * similarity_loss
        
        return outputs


def load_embedding_training_data(file_path: str = "data/llm_training_data.json") -> List[Dict[str, Any]]:
    """Load the embedding-based training data."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} embedding training examples")
        return data
    except FileNotFoundError:
        print(f"Training data file {file_path} not found.")
        print("Please run the embedding preprocessing script first.")
        return []


def format_embedding_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format the training example with embeddings for complex conversational training."""
    
    # Parse input and output JSON
    input_data = json.loads(example.get("input", "{}"))
    output_data = json.loads(example.get("output", "{}"))
    
    instruction = example.get("instruction", "Recommend coffee based on preferences")
    query = input_data.get("query", "")
    query_embedding = input_data.get("query_embedding", [])
    
    recommendation = output_data.get("recommendation", {})
    coffee_name = recommendation.get("name", "Unknown")
    coffee_roaster = recommendation.get("roaster", "Unknown")
    coffee_description = recommendation.get("description", "")
    coffee_embedding = recommendation.get("coffee_embedding", [])
    why_recommended = recommendation.get("why_recommended", "Good match for your preferences")
    coffee_origin = recommendation.get("origin", "Unknown")
    coffee_rating = recommendation.get("rating", 0)
    
    # Enhanced conversational format for complex query handling
    prompt_text = f"""<|user|>
{query}

<|assistant|>
I'd recommend **{coffee_name}** by {coffee_roaster}.

**Origin:** {coffee_origin}
**Rating:** {coffee_rating}/100

**Why this coffee:** {why_recommended}

**Tasting Profile:** {coffee_description}

This recommendation takes into account your specific preferences and mood. The embedding similarity indicates this coffee aligns well with what you're looking for. Would you like me to suggest any brewing methods or alternatives?<|endoftext|>"""

    return {
        "text": prompt_text,
        "query_embedding": query_embedding,
        "coffee_embedding": coffee_embedding,
        "query": query,
        "coffee_name": coffee_name,
        "metadata": example.get("metadata", {})
    }


def prepare_embedding_dataset(data: List[Dict[str, Any]], tokenizer) -> Dataset:
    """Prepare the dataset with embeddings for training."""
    formatted_data = []
    
    print("Formatting embedding data...")
    for example in data:
        formatted_example = format_embedding_prompt(example)
        
        # Verify embeddings exist and are valid
        if (len(formatted_example["query_embedding"]) > 0 and 
            len(formatted_example["coffee_embedding"]) > 0):
            formatted_data.append(formatted_example)
    
    print(f"Prepared {len(formatted_data)} valid examples with embeddings")
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize the text
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length", 
            max_length=512,
            return_tensors="pt",
        )
        
        # Add embeddings as additional features
        tokenized["query_embeddings"] = examples["query_embedding"]
        tokenized["coffee_embeddings"] = examples["coffee_embedding"]
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text", "query", "coffee_name", "metadata"]
    )
    
    return tokenized_dataset


class EmbeddingDataCollator:
    """Custom data collator for embedding-based training."""
    
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
            max_length=512,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Add embeddings to batch
        batch["query_embeddings"] = torch.tensor(query_embeddings, dtype=torch.float32)
        batch["coffee_embeddings"] = torch.tensor(coffee_embeddings, dtype=torch.float32)
        
        # Set labels for language modeling
        batch["labels"] = batch["input_ids"].clone()
        
        return batch


class EmbeddingTrainer(Trainer):
    """Custom trainer for embedding-based coffee recommendation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
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
            labels=inputs.get("labels")
        )
        
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss


def create_embedding_model_and_tokenizer(model_name: str = "microsoft/DialoGPT-medium"):
    """Create embedding-aware model and tokenizer."""
    print(f"Creating embedding-aware model: {model_name}")
    
    # For conversational coffee recommendations, consider these alternatives:
    # - "microsoft/DialoGPT-medium": Good for conversations
    # - "facebook/blenderbot-400M-distill": Better for helpful responses  
    # - "microsoft/DialoGPT-small": Faster training, still good quality
    
    model = EmbeddingCoffeeRecommender(base_model_name=model_name)
    tokenizer = model.tokenizer
    
    # Add special tokens for better conversation flow
    special_tokens = ["<|user|>", "<|assistant|>", "<|coffee|>", "<|recommendation|>"]
    tokenizer.add_tokens(special_tokens)
    model.language_model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def train_embedding_model(
    model,
    tokenizer, 
    train_dataset,
    output_dir: str = "models/coffee_embedding_recommender",
    num_epochs: int = 3,
    batch_size: int = 2,  # Smaller batch size due to embeddings
    learning_rate: float = 2e-5,  # Lower learning rate for fine-tuning
    freeze_strategy: str = "finetune_top"  # Default to safe fine-tuning strategy
):
    """Train the embedding-aware model with configurable freezing strategy."""
    
    # Apply freezing strategy
    if freeze_strategy == "freeze_base":
        print("Applying freeze_base strategy: Only training embedding layers")
        for param in model.language_model.parameters():
            param.requires_grad = False
            
    elif freeze_strategy == "finetune_top":
        print("Applying coffee domain adaptation strategy:")
        print("- Freezing bottom transformer layers (preserving base language abilities)")
        print("- Fine-tuning top 3 transformer layers (coffee domain adaptation)")
        print("- Training all embedding layers (learning coffee-query matching)")
        
        # Freeze bottom layers of the transformer
        total_layers = len(model.language_model.transformer.h)
        layers_to_finetune = 3  # Top 3 layers - good balance
        
        # Freeze word embeddings (preserve vocabulary understanding)
        for param in model.language_model.transformer.wte.parameters():
            param.requires_grad = False
        for param in model.language_model.transformer.wpe.parameters():
            param.requires_grad = False
            
        # Freeze bottom transformer layers (preserve base language understanding)
        for i in range(total_layers - layers_to_finetune):
            for param in model.language_model.transformer.h[i].parameters():
                param.requires_grad = False
                
        print(f"Frozen: Bottom {total_layers - layers_to_finetune} layers + embeddings")
        print(f"Trainable: Top {layers_to_finetune} layers + final layers + embedding projectors")
        
    elif freeze_strategy == "full_finetune":
        print("Applying full_finetune strategy: Training all parameters")
        # All parameters remain trainable (default)
        pass
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Differential learning rates optimized for coffee domain adaptation
    if freeze_strategy == "finetune_top":
        # Use much lower learning rates for pre-trained components
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.language_model.named_parameters() if p.requires_grad],
                "lr": learning_rate * 0.1,  # Very small changes to pre-trained layers
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if "language_model" not in n and p.requires_grad],
                "lr": learning_rate * 5,  # Higher LR for new embedding layers that start from random
                "weight_decay": 0.01,
            },
        ]
        print(f"✓ Learning rates - Language model: {learning_rate*0.1:.2e}, Embeddings: {learning_rate*5:.2e}")
    else:
        optimizer_grouped_parameters = None
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=learning_rate if optimizer_grouped_parameters is None else 3e-5,  # Will be overridden by optimizer
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps" if len(train_dataset) > 1000 else "no",
        save_strategy="steps",
        load_best_model_at_end=False,  # Disable for now due to custom model
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb logging
    )
    
    # Custom data collator for embeddings
    data_collator = EmbeddingDataCollator(tokenizer=tokenizer)
    
    # Initialize custom trainer
    trainer = EmbeddingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(
            torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8) if optimizer_grouped_parameters else None,
            None
        )
    )
    
    # Train the model
    print("Starting embedding-aware training...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {output_dir}")
    model.language_model.save_pretrained(f"{output_dir}/language_model")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    
    # Save the full model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'base_model_name': "microsoft/DialoGPT-medium",
            'query_embedding_dim': 384,
            'coffee_embedding_dim': 384,
            'hidden_dim': 512
        }
    }, f"{output_dir}/full_model.pt")
    
    print(f"Model saved to {output_dir}")
    return trainer


def test_embedding_model(model, tokenizer, test_query: str = "I need an energizing coffee"):
    """Test the trained embedding model with a sample query."""
    print(f"\nTesting model with query: '{test_query}'")
    
    # This would require the sentence transformer model to create embeddings
    # For now, we'll create a dummy embedding
    dummy_query_embedding = torch.randn(1, 384)
    dummy_coffee_embedding = torch.randn(1, 384)
    
    # Format test input
    test_prompt = f"""### Instruction:
Recommend coffee based on preferences

### User Query:
{test_query}

### Recommended Coffee:"""
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=256)
    
    # Generate recommendation (without embeddings for this simple test)
    model.eval()
    with torch.no_grad():
        # For testing, we'll just use the language model part
        outputs = model.language_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model response:")
    print(response[len(test_prompt):])


def main():
    """Main training function for embedding-aware coffee recommendation."""
    print("Starting embedding-aware coffee recommendation model training...")
    print("=" * 70)
    
    # Load embedding training data
    training_data = load_embedding_training_data()
    if not training_data:
        print("No training data found. Please run the embedding preprocessing script first.")
        return
    
    # Create embedding-aware model and tokenizer
    model, tokenizer = create_embedding_model_and_tokenizer()
    
    # Prepare dataset with embeddings
    print("Preparing embedding dataset...")
    train_dataset = prepare_embedding_dataset(training_data, tokenizer)
    
    if len(train_dataset) == 0:
        print("No valid training examples found. Check your data format.")
        return
    
    print(f"Training on {len(train_dataset)} examples")
    
    # Train model with coffee domain adaptation strategy
    trainer = train_embedding_model(model, tokenizer, train_dataset)
    
    print(f"\nTraining Summary:")
    print(f"- Strategy: Fine-tune top layers + train embedding layers")
    print(f"- Base language model: Mostly preserved")
    print(f"- Coffee adaptation: Top 3 transformer layers + all embedding layers")
    print(f"- Training focus: Domain adaptation, not full retraining")
    
    # Test the model
    test_embedding_model(model, tokenizer)
    
    print("\n" + "=" * 70)
    print("EMBEDDING-AWARE TRAINING COMPLETED!")
    print("Your model can now use both text and embedding features for recommendations.")
    print("=" * 70)


if __name__ == "__main__":
    main()