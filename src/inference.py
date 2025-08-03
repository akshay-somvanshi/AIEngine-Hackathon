#!/usr/bin/env python3
"""
Hybrid Coffee Recommendation Inference Script

This script uses the fine-tuned hybrid model to generate coffee recommendations
based on user input. Combines semantic similarity matching with natural language generation.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class HybridCoffeeRecommendationInference:
    """Hybrid coffee recommendation system with embedding support."""

    def __init__(
        self,
        model_path: str = "models/coffee_hybrid_recommender",
        coffee_data_path: str = "data/unique_coffees.json",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize the hybrid inference system."""
        self.model_path = model_path
        self.coffee_data_path = coffee_data_path
        self.embedding_model_name = embedding_model_name

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        self.coffee_database = []

        # Load all components
        self.load_model()
        self.load_embedding_model()
        self.load_coffee_database()

    def load_model(self):
        """Load the fine-tuned hybrid model."""
        try:
            logger.info(f"Loading hybrid model from {self.model_path}...")

            # Check if we have the full model checkpoint
            full_model_path = os.path.join(self.model_path, "full_model.pt")

            if os.path.exists(full_model_path):
                # Load the full custom model
                checkpoint = torch.load(full_model_path, map_location="cpu")
                model_config = checkpoint["model_config"]

                # Create model with same architecture
                self.model = HybridCoffeeRecommender(
                    base_model_name=model_config["base_model_name"],
                    embedding_model_name=model_config["embedding_model_name"],
                    embedding_dim=model_config["embedding_dim"],
                    hidden_dim=model_config["hidden_dim"],
                )

                # Add special tokens to match training
                special_tokens = ["<|user|>", "<|assistant|>"]
                self.model.tokenizer.add_tokens(special_tokens)
                self.model.language_model.resize_token_embeddings(
                    len(self.model.tokenizer)
                )

                # Load trained weights
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.tokenizer = self.model.tokenizer

                logger.info("Full hybrid model loaded successfully!")

            else:
                # Fallback: load just the language model part
                logger.info("Full model not found, loading language model only...")
                lm_path = os.path.join(self.model_path, "language_model")
                tokenizer_path = os.path.join(self.model_path, "tokenizer")

                if os.path.exists(lm_path) and os.path.exists(tokenizer_path):
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    language_model = AutoModelForCausalLM.from_pretrained(
                        lm_path, torch_dtype=torch.float16, device_map="auto"
                    )

                    # Create wrapper with just language model
                    self.model = HybridCoffeeRecommender()
                    self.model.language_model = language_model
                    self.model.tokenizer = self.tokenizer

                    logger.info("Language model loaded successfully!")
                else:
                    raise FileNotFoundError(
                        "No valid model found in the specified path"
                    )

            # Set to evaluation mode
            self.model.eval()

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error("Please ensure the model has been trained first.")
            self.model = None
            self.tokenizer = None

    def load_embedding_model(self):
        """Load the sentence transformer model for embeddings."""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.embedding_model = None

    def load_coffee_database(self):
        """Load the coffee database with pre-computed embeddings."""
        try:
            if os.path.exists(self.coffee_data_path):
                with open(self.coffee_data_path, "r") as f:
                    self.coffee_database = json.load(f)

                # Ensure all coffees have embeddings
                for coffee in self.coffee_database:
                    if "embedding" not in coffee or not coffee["embedding"]:
                        if self.embedding_model:
                            coffee["embedding"] = self.embedding_model.encode(
                                [coffee["description"]]
                            )[0].tolist()

                logger.info(f"Loaded {len(self.coffee_database)} coffees from database")
            else:
                logger.info(f"Coffee database not found at {self.coffee_data_path}")
                logger.info("Using fallback coffee data...")
                self.coffee_database = self.create_fallback_database()

        except Exception as e:
            logger.error(f"Error loading coffee database: {e}")
            self.coffee_database = []

    def create_fallback_database(self) -> List[Dict[str, Any]]:
        """Create a small fallback coffee database for testing."""
        fallback_coffees = [
            {
                "name": "Ethiopian Yirgacheffe",
                "roaster": "Counter Culture Coffee",
                "origin": "Ethiopia",
                "rating": 91,
                "description": "Bright, floral, citrusy coffee with tea-like qualities and vibrant acidity. Perfect for pour-over brewing.",
                "characteristics": {"moods": ["energizing"], "flavors": ["fruity"]},
            },
            {
                "name": "Guatemala Huehuetenango",
                "roaster": "Sweet Maria's",
                "origin": "Guatemala",
                "rating": 88,
                "description": "Full-bodied coffee with chocolate and nut notes. Smooth finish with balanced acidity.",
                "characteristics": {
                    "moods": ["comforting"],
                    "flavors": ["chocolate", "nutty"],
                },
            },
            {
                "name": "Colombian Supremo",
                "roaster": "Blue Bottle Coffee",
                "origin": "Colombia",
                "rating": 86,
                "description": "Well-balanced coffee with caramel sweetness and mild acidity. Great for espresso or drip brewing.",
                "characteristics": {"moods": ["energizing"], "flavors": ["sweet"]},
            },
        ]

        # Add embeddings to fallback data
        if self.embedding_model:
            for coffee in fallback_coffees:
                coffee["embedding"] = self.embedding_model.encode(
                    [coffee["description"]]
                )[0].tolist()

        return fallback_coffees

    def find_best_coffee_matches(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Find best coffee matches using embedding similarity."""
        if not self.embedding_model or not self.coffee_database:
            return []

        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Calculate similarities
        matches = []
        for coffee in self.coffee_database:
            if "embedding" in coffee and coffee["embedding"]:
                coffee_emb = np.array(coffee["embedding"])
                similarity = np.dot(query_embedding, coffee_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(coffee_emb)
                )
                matches.append((similarity, coffee))

        # Sort by similarity and return top k
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[:top_k]

    def generate_recommendation(
        self,
        user_input: str,
        conversation_history: List[str] = None,
        max_length: int = 400,
        use_embeddings: bool = True,
    ) -> str:
        """Generate coffee recommendation with conversation support."""

        if not self.model or not self.tokenizer:
            return "Model not loaded. Please train the model first."

        try:
            # Build conversation context
            conversation_context = ""
            if conversation_history:
                conversation_context = (
                    "\n".join(conversation_history[-3:]) + "\n"
                )  # Last 3 exchanges

            if use_embeddings and self.embedding_model:
                # Enhanced mode: use embeddings to find best matches first
                coffee_matches = self.find_best_coffee_matches(user_input, top_k=3)

                if coffee_matches:
                    # Use the best match to inform the generation
                    best_match = coffee_matches[0][1]

                    # Create conversational prompt with coffee context
                    if conversation_history:
                        prompt = f"""{conversation_context}User: {user_input}

Assistant: """
                    else:
                        prompt = f"""User: {user_input}

Assistant: Based on your preferences, I'd recommend **{best_match['name']}** by {best_match['roaster']}.

**Origin:** {best_match['origin']}
**Rating:** {best_match['rating']}/100

**Why this coffee:** {best_match['description']}

This recommendation is based on semantic similarity to your query. I'm confident this coffee will match your preferences because it has the characteristics you're looking for. What do you think about this recommendation? Would you like me to suggest alternatives or explain more about why this coffee fits your mood? """

                else:
                    # Fallback to basic prompt
                    if conversation_history:
                        prompt = f"""{conversation_context}User: {user_input}

Assistant: """
                    else:
                        prompt = f"""User: {user_input}

Assistant: Based on your preferences, I'd recommend """
            else:
                # Basic mode: use simple prompt
                if conversation_history:
                    prompt = f"""{conversation_context}User: {user_input}

Assistant: """
                else:
                    prompt = f"""User: {user_input}

Assistant: Based on your preferences, I'd recommend """

            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            )

            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                if hasattr(self.model, "language_model"):
                    # Use the language model part for generation
                    outputs = self.model.language_model.generate(
                        **inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=0.8,  # Slightly higher for more conversational responses
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    # Direct generation
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            elif "### Response:" in response:
                response = response.split("### Response:")[-1].strip()

            # Clean up the response to remove conversation history that got included
            if "User:" in response:
                # Remove everything before the last "Assistant:" or start fresh
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                else:
                    # If no "Assistant:" found, try to extract just the recommendation part
                    lines = response.split("\n")
                    cleaned_lines = []
                    for line in lines:
                        if not line.strip().startswith(
                            "User:"
                        ) and not line.strip().startswith("Assistant:"):
                            cleaned_lines.append(line)
                    response = "\n".join(cleaned_lines).strip()

            return response

        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return f"Sorry, I encountered an error: {e}"

    def get_detailed_recommendation(self, user_input: str) -> Dict[str, Any]:
        """Get detailed recommendation with similarity scores and alternatives."""

        # Get coffee matches with similarity scores
        coffee_matches = self.find_best_coffee_matches(user_input, top_k=5)

        # Generate text recommendation
        text_recommendation = self.generate_recommendation(user_input)

        # Format detailed response
        detailed_response = {
            "query": user_input,
            "primary_recommendation": text_recommendation,
            "coffee_matches": [],
            "similarity_scores": [],
        }

        for similarity, coffee in coffee_matches:
            detailed_response["coffee_matches"].append(
                {
                    "name": coffee["name"],
                    "roaster": coffee["roaster"],
                    "origin": coffee["origin"],
                    "rating": coffee["rating"],
                    "description": coffee["description"][:100] + "...",
                    "similarity_score": round(similarity, 3),
                }
            )
            detailed_response["similarity_scores"].append(round(similarity, 3))

        return detailed_response

    def interactive_mode(self):
        """Run interactive mode for coffee recommendations with conversation support."""
        print("Hybrid Coffee Recommendation Chatbot")
        print("=" * 60)
        print("Features:")
        print("- Embedding-aware recommendations")
        print("- Semantic similarity matching")
        print("- Conversational responses")
        print("- Memory of previous exchanges")
        print()
        print("Commands:")
        print("  'quit' or 'exit' - Exit the system")
        print("  'detailed <query>' - Get detailed recommendation with scores")
        print("  'simple <query>' - Get basic recommendation without embeddings")
        print("  'clear' - Clear conversation history")
        print()

        conversation_history = []

        while True:
            try:
                user_input = input("You: ")

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Thanks for using the coffee recommendation chatbot!")
                    break

                if user_input.lower() == "clear":
                    conversation_history = []
                    print("Conversation history cleared.")
                    continue

                if not user_input.strip():
                    print("Please provide some input.")
                    continue

                # Handle special commands
                if user_input.lower().startswith("detailed "):
                    query = user_input[9:]  # Remove "detailed "
                    print("\nGenerating detailed recommendation...")
                    result = self.get_detailed_recommendation(query)

                    print(f"\nDetailed Recommendation:")
                    print(f"Query: {result['query']}")
                    print(f"\nResponse: {result['primary_recommendation']}")
                    print(f"\nTop Coffee Matches:")
                    for match in result["coffee_matches"][:3]:
                        print(
                            f"- {match['name']} by {match['roaster']} (similarity: {match['similarity_score']})"
                        )

                elif user_input.lower().startswith("simple "):
                    query = user_input[7:]  # Remove "simple "
                    print("\nGenerating simple recommendation...")
                    recommendation = self.generate_recommendation(
                        query,
                        conversation_history=conversation_history,
                        use_embeddings=False,
                    )
                    print(f"\nAssistant: {recommendation}")

                else:
                    # Standard recommendation with conversation history
                    print("\nGenerating recommendation...")
                    recommendation = self.generate_recommendation(
                        user_input, conversation_history=conversation_history
                    )
                    print(f"\nAssistant: {recommendation}")

                # Add to conversation history (clean version)
                conversation_history.append(f"User: {user_input}")
                # Clean the recommendation before adding to history
                clean_recommendation = recommendation
                if "User:" in clean_recommendation:
                    # Remove any conversation history that got included
                    if "Assistant:" in clean_recommendation:
                        clean_recommendation = clean_recommendation.split("Assistant:")[
                            -1
                        ].strip()
                conversation_history.append(f"Assistant: {clean_recommendation}")

                # Keep only last 6 exchanges (3 user-assistant pairs)
                if len(conversation_history) > 6:
                    conversation_history = conversation_history[-6:]

                print("\n" + "=" * 60 + "\n")

            except KeyboardInterrupt:
                print("\nThanks for using the coffee recommendation chatbot!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def load_enhanced_example_queries() -> List[str]:
    """Load enhanced example queries for testing."""
    return [
        "I'm feeling stressed and need something energizing but not too acidic",
        "I want a smooth coffee for reading on a rainy Sunday morning",
        "Something complex for impressing coffee snob friends at dinner",
        "I usually drink dark roast but want to try something fruity",
        "Low caffeine option that still has bold flavors for afternoon",
        "Coffee for making cold brew that won't be too bitter",
        "Something cozy and comforting for winter evenings",
        "Best coffee for espresso machine that brings out chocolate notes",
        "I'm new to specialty coffee, something approachable but interesting",
        "Coffee that pairs well with dark chocolate desserts",
    ]


def test_enhanced_model(model_path: str = "models/coffee_hybrid_recommender"):
    """Test the enhanced model with complex queries."""
    recommender = HybridCoffeeRecommendationInference(model_path)

    if not recommender.model:
        logger.error("Model not available for testing.")
        return

    print("Testing Hybrid Coffee Recommendation Model")
    print("=" * 60)

    example_queries = load_enhanced_example_queries()

    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query}")

        # Test both modes
        print(f"\nEmbedding-Enhanced Output:")
        result = recommender.get_detailed_recommendation(query)
        print(result["primary_recommendation"])

        if result["coffee_matches"]:
            print(
                f"\nTop Match: {result['coffee_matches'][0]['name']} (similarity: {result['coffee_matches'][0]['similarity_score']})"
            )

        print("-" * 60)


def main():
    """Main function with enhanced options."""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Coffee Recommendation System")
    parser.add_argument(
        "--mode",
        choices=["interactive", "test"],
        default="interactive",
        help="Run mode: interactive or test",
    )
    parser.add_argument(
        "--model-path",
        default="models/coffee_hybrid_recommender",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--coffee-data",
        default="data/unique_coffees.json",
        help="Path to coffee database",
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        recommender = HybridCoffeeRecommendationInference(
            model_path=args.model_path, coffee_data_path=args.coffee_data
        )
        recommender.interactive_mode()
    elif args.mode == "test":
        test_enhanced_model(args.model_path)


if __name__ == "__main__":
    main()
