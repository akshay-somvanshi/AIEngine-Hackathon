#!/usr/bin/env python3
"""
Coffee Recommendation Inference Script

This script uses the fine-tuned model to generate coffee recommendations
based on user input describing preferences and mood.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import Dict, List, Any


class CoffeeRecommender:
    """Coffee recommendation model wrapper."""

    def __init__(self, model_path: str = "models/coffee_recommender"):
        """Initialize the model."""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            print(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.float16, device_map="auto"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model has been trained first.")
            self.model = None
            self.tokenizer = None

    def generate_recommendation(self, user_input: str, max_length: int = 300) -> str:
        """Generate coffee recommendation based on user input."""
        if not self.model or not self.tokenizer:
            return "Model not loaded. Please train the model first."

        # Format the prompt
        prompt = f"""### Instruction:
Recommend coffee based on preferences and mood

### Input:
{user_input}

### Response:
"""

        # Tokenize input
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part
        response = response.split("### Response:")[-1].strip()

        return response

    def interactive_mode(self):
        """Run interactive mode for coffee recommendations."""
        print("Coffee Recommendation System")
        print("=" * 40)
        print("Type 'quit' to exit")
        print()

        while True:
            try:
                user_input = input("Describe your coffee preferences and mood: ")

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not user_input.strip():
                    print("Please provide some input.")
                    continue

                print("\nGenerating recommendation...")
                recommendation = self.generate_recommendation(user_input)
                print(f"\nRecommendation:\n{recommendation}")
                print("\n" + "=" * 40 + "\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def load_example_queries() -> List[str]:
    """Load example queries for testing."""
    return [
        "I'm feeling stressed and need something energizing. I prefer bold flavors.",
        "I'm tired and want a smooth, mild coffee to relax with.",
        "I need to focus on work and want something with moderate caffeine.",
        "I'm in a creative mood and want something unique and inspiring.",
        "I'm meeting friends and want a social coffee that's enjoyable.",
        "I'm adventurous and want to try something traditional and complex.",
        "I'm productive and need sustained energy throughout the day.",
        "I'm relaxed and want something creamy and smooth.",
    ]


def test_model(model_path: str = "models/coffee_recommender"):
    """Test the model with example queries."""
    recommender = CoffeeRecommender(model_path)

    if not recommender.model:
        print("Model not available for testing.")
        return

    print("Testing Coffee Recommendation Model")
    print("=" * 50)

    example_queries = load_example_queries()

    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {query}")
        print(f"Output: {recommender.generate_recommendation(query)}")
        print("-" * 50)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Coffee Recommendation System")
    parser.add_argument(
        "--mode",
        choices=["interactive", "test"],
        default="interactive",
        help="Run mode: interactive or test",
    )
    parser.add_argument(
        "--model-path",
        default="models/coffee_recommender",
        help="Path to the trained model",
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        recommender = CoffeeRecommender(args.model_path)
        recommender.interactive_mode()
    elif args.mode == "test":
        test_model(args.model_path)


if __name__ == "__main__":
    main()
