#!/usr/bin/env python3
"""
Coffee Recommendation Comment Generator

This module generates personalized comments explaining why a recommended coffee
is well-suited for the user's preferences.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


class CoffeeCommentGenerator:
    """Generates personalized comments about coffee recommendations."""

    def __init__(self, model_name="microsoft/DialoGPT-small"):
        """Initialize the comment generator."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else None,
            )

            logger.info(f"Comment generator initialized with {model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize comment generator: {e}")
            self.model = None
            self.tokenizer = None

    def generate_comment(self, user_query: str, coffee_data: dict) -> str:
        """
        Generate a personalized comment about why the coffee is well-suited.

        Args:
            user_query: The user's original query/preferences
            coffee_data: Dictionary containing coffee information

        Returns:
            A personalized comment explaining why this coffee is perfect
        """
        if not self.model or not self.tokenizer:
            return self._generate_fallback_comment(user_query, coffee_data)

        try:
            # Create a detailed prompt for the model
            prompt = self._create_comment_prompt(user_query, coffee_data)

            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            )

            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate comment
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part
            if prompt in response:
                comment = response.replace(prompt, "").strip()
            else:
                comment = response.strip()

            # Clean up the comment
            comment = self._clean_comment(comment)

            return (
                comment
                if comment
                else self._generate_fallback_comment(user_query, coffee_data)
            )

        except Exception as e:
            logger.error(f"Error generating comment: {e}")
            return self._generate_fallback_comment(user_query, coffee_data)

    def _create_comment_prompt(self, user_query: str, coffee_data: dict) -> str:
        """Create a detailed prompt for comment generation."""

        coffee_name = coffee_data.get("name", "Unknown Coffee")
        roaster = coffee_data.get("roaster", "Unknown Roaster")
        origin = coffee_data.get("origin", "Unknown Origin")
        description = coffee_data.get("description", "")
        rating = coffee_data.get("rating", "N/A")

        # Extract key words from user query for better matching
        user_keywords = self._extract_keywords(user_query.lower())

        prompt = f"""User is looking for: "{user_query}"
User's key preferences: {', '.join(user_keywords)}

Coffee: {coffee_name} by {roaster}
Origin: {origin}
Rating: {rating}/100
Description: {description}

Based on the user's specific request for "{user_query}", 
explain why this coffee is the perfect match:

This coffee is perfect for you because: """

        return prompt

    def _extract_keywords(self, query: str) -> list:
        """Extract relevant keywords from user query."""
        keywords = []

        # Common coffee preferences
        preference_keywords = [
            "vanilla",
            "chocolate",
            "caramel",
            "fruity",
            "bright",
            "smooth",
            "bold",
            "strong",
            "light",
            "dark",
            "medium",
            "energizing",
            "energetic",
            "calming",
            "relaxing",
            "sweet",
            "bitter",
            "acidic",
            "balanced",
            "floral",
            "nutty",
            "earthy",
            "spicy",
            "citrus",
            "berry",
            "cherry",
            "tired",
            "focused",
            "creative",
            "social",
            "adventurous",
            "gentle",
            "intense",
            "mild",
            "complex",
            "simple",
            "unique",
            "traditional",
            "coffee",
            "espresso",
            "latte",
            "cappuccino",
            "americano",
            "mocha",
            "macchiato",
            "flat white",
            "cold brew",
            "drip",
            "pour over",
            "french press",
            "aeropress",
            "chemex",
        ]

        for keyword in preference_keywords:
            if keyword in query:
                keywords.append(keyword)

        # Add mood-related keywords
        mood_keywords = [
            "happy",
            "sad",
            "stressed",
            "excited",
            "tired",
            "awake",
            "focused",
            "creative",
            "social",
            "alone",
            "energetic",
            "relaxed",
            "adventurous",
            "good",
            "bad",
            "great",
            "amazing",
            "wonderful",
            "terrible",
            "awful",
            "nice",
            "lovely",
            "perfect",
        ]

        for mood in mood_keywords:
            if mood in query:
                keywords.append(mood)

        # Add general descriptive words
        general_words = [
            "like",
            "love",
            "want",
            "need",
            "prefer",
            "enjoy",
            "hate",
            "dislike",
            "favorite",
            "best",
            "worst",
            "good",
            "bad",
            "great",
            "amazing",
            "wonderful",
            "terrible",
            "awful",
            "nice",
            "lovely",
            "perfect",
            "delicious",
            "tasty",
            "yummy",
            "bitter",
            "sour",
            "sweet",
            "strong",
            "weak",
            "hot",
            "cold",
            "warm",
            "fresh",
            "rich",
            "smooth",
            "creamy",
            "thick",
            "thin",
            "light",
            "heavy",
            "something",
            "anything",
            "everything",
            "nothing",
            "whatever",
            "sip",
            "drink",
            "have",
            "get",
            "find",
            "try",
            "taste",
            "sample",
            "morning",
            "afternoon",
            "evening",
            "night",
            "today",
            "now",
            "later",
            "soon",
            "quick",
            "fast",
            "slow",
            "easy",
            "simple",
            "casual",
            "relaxed",
            "comfortable",
            "cozy",
            "pleasant",
            "nice",
            "good",
            "fine",
            "okay",
            "alright",
            "decent",
            "solid",
            "reliable",
            "daily",
            "regular",
            "usual",
            "normal",
            "standard",
            "typical",
            "favorite",
            "go-to",
            "staple",
            "classic",
            "traditional",
            "modern",
            "new",
            "different",
            "unique",
            "special",
            "particular",
            "specific",
        ]

        for word in general_words:
            if word in query:
                keywords.append(word)

        # Add conversational phrases and patterns
        conversational_patterns = [
            "i want",
            "i need",
            "i like",
            "i love",
            "i prefer",
            "i enjoy",
            "give me",
            "show me",
            "find me",
            "recommend",
            "suggest",
            "something to",
            "anything that",
            "whatever is",
            "just want",
            "looking for",
            "in the mood for",
            "feel like",
            "craving",
            "sip on",
            "drink",
            "have",
            "get",
            "try",
            "taste",
            "sample",
            "morning coffee",
            "afternoon pick-me-up",
            "evening drink",
            "start my day",
            "wake me up",
            "keep me going",
            "relax with",
            "enjoy while",
            "perfect for",
            "great with",
            "goes well with",
            "comfort drink",
            "treat myself",
            "indulge in",
            "splurge on",
        ]

        for pattern in conversational_patterns:
            if pattern in query.lower():
                keywords.append(pattern.replace(" ", "_"))

        # If no specific keywords found, extract any meaningful words
        if not keywords:
            # Split query into words and filter out common stop words
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "it",
                "we",
                "they",
                "me",
                "him",
                "her",
                "us",
                "them",
                "my",
                "your",
                "his",
                "her",
                "its",
                "our",
                "their",
                "mine",
                "yours",
                "his",
                "hers",
                "ours",
                "theirs",
                "to",
                "on",
                "in",
                "at",
                "for",
                "with",
                "by",
            }
            words = query.lower().split()
            meaningful_words = [
                word for word in words if word not in stop_words and len(word) > 2
            ]
            keywords.extend(meaningful_words[:3])  # Take up to 3 meaningful words

        return keywords if keywords else ["coffee preferences"]

    def _clean_comment(self, comment: str) -> str:
        """Clean up the generated comment."""
        # Remove any remaining conversation markers
        lines = comment.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("User:")
                and not line.startswith("Assistant:")
                and not line.startswith("Comment:")
                and len(line) > 5
            ):
                cleaned_lines.append(line)

        comment = " ".join(cleaned_lines)

        # Ensure it ends with proper punctuation
        if comment and not comment.endswith((".", "!", "?")):
            comment += "."

        return comment

    def _generate_fallback_comment(self, user_query: str, coffee_data: dict) -> str:
        """Generate a fallback comment when the model is not available."""

        coffee_name = coffee_data.get("name", "this coffee")
        description = coffee_data.get("description", "").lower()
        user_query_lower = user_query.lower()

        # Extract key characteristics from the description
        characteristics = []
        if "vanilla" in description:
            characteristics.append("vanilla notes")
        if "chocolate" in description:
            characteristics.append("rich chocolate flavors")
        if "fruity" in description or "berry" in description:
            characteristics.append("bright fruity notes")
        if "bright" in description or "citrus" in description:
            characteristics.append("bright acidity")
        if "smooth" in description or "balanced" in description:
            characteristics.append("smooth, balanced profile")
        if "floral" in description:
            characteristics.append("delicate floral notes")
        if "energizing" in description or "bold" in description:
            characteristics.append("energizing qualities")
        if "nutty" in description:
            characteristics.append("nutty undertones")
        if "earthy" in description:
            characteristics.append("earthy notes")
        if "spicy" in description:
            characteristics.append("spicy notes")
        if "caramel" in description:
            characteristics.append("caramel sweetness")
        if "creamy" in description:
            characteristics.append("creamy texture")
        if "rich" in description:
            characteristics.append("rich, full-bodied flavor")
        if "light" in description:
            characteristics.append("light, delicate profile")
        if "dark" in description:
            characteristics.append("dark, robust character")

        # Enhanced matching based on user query
        user_keywords = self._extract_keywords(user_query_lower)

        # Direct keyword matches
        for keyword in user_keywords:
            if keyword in description:
                if keyword == "vanilla":
                    return f"This {coffee_name} is perfect for you because it features vanilla notes that directly match your preference for vanilla flavors."
                elif keyword == "chocolate":
                    return f"This {coffee_name} is ideal because it offers rich chocolate flavors that align with your chocolate preference."
                elif keyword in ["energizing", "energetic", "energy"]:
                    return (
                        f"This {coffee_name} is perfect for your need for energy - "
                        f"it has bright, lively characteristics that will give you "
                        f"the boost you're looking for."
                    )
                elif keyword in ["smooth", "gentle", "mild"]:
                    return (
                        f"This {coffee_name} offers a smooth, balanced profile "
                        f"that's perfect for your preference for gentle flavors."
                    )
                elif keyword in ["bold", "strong", "intense"]:
                    return (
                        f"This {coffee_name} has bold, strong characteristics "
                        f"that match your preference for intense flavors."
                    )
                elif keyword in ["bright", "fruity", "citrus"]:
                    return (
                        f"This {coffee_name} features bright, fruity notes "
                        f"that align with your preference for lively flavors."
                    )
                elif keyword in ["calming", "relaxing"]:
                    return (
                        f"This {coffee_name} has a calming, smooth profile "
                        f"that's perfect for your need to relax."
                    )
                elif keyword in ["creative", "adventurous"]:
                    return (
                        f"This {coffee_name} offers unique, complex flavors "
                        f"that will inspire your creativity and adventurous spirit."
                    )
                elif keyword in ["social", "friendly"]:
                    return (
                        f"This {coffee_name} has a pleasant, approachable profile "
                        f"that's perfect for social situations."
                    )
                elif keyword in ["focused", "concentration"]:
                    return (
                        f"This {coffee_name} has a balanced, focused profile "
                        f"that will help you concentrate on your work."
                    )
                elif keyword in ["tired", "awake"]:
                    return (
                        f"This {coffee_name} has energizing qualities "
                        f"that will help wake you up and keep you alert."
                    )
                else:
                    return (
                        f"This {coffee_name} is perfect for you because it "
                        f"features {keyword} characteristics that directly "
                        f"match your preferences."
                    )

        # Mood-based matching
        if any(word in user_query_lower for word in ["happy", "good mood", "cheerful"]):
            return (
                f"This {coffee_name} has bright, uplifting flavors that will "
                f"complement your happy mood and make your day even better."
            )
        elif any(
            word in user_query_lower for word in ["stressed", "anxious", "worried"]
        ):
            return (
                f"This {coffee_name} offers a smooth, calming profile that's "
                f"perfect for helping you relax and reduce stress."
            )
        elif any(word in user_query_lower for word in ["tired", "exhausted", "sleepy"]):
            return (
                f"This {coffee_name} has energizing qualities that will help "
                f"wake you up and give you the energy boost you need."
            )
        elif any(
            word in user_query_lower for word in ["creative", "artistic", "inspired"]
        ):
            return (
                f"This {coffee_name} has complex, inspiring flavors that will "
                f"spark your creativity and artistic spirit."
            )
        elif any(word in user_query_lower for word in ["social", "meeting", "friends"]):
            return (
                f"This {coffee_name} has a pleasant, approachable profile "
                f"that's perfect for enjoying with friends and social gatherings."
            )

        # General sentiment analysis
        positive_words = [
            "like",
            "love",
            "good",
            "great",
            "amazing",
            "wonderful",
            "perfect",
            "delicious",
            "tasty",
            "yummy",
            "enjoy",
            "favorite",
            "best",
        ]
        negative_words = [
            "hate",
            "dislike",
            "bad",
            "terrible",
            "awful",
            "worst",
            "bitter",
            "sour",
        ]

        # Handle casual, conversational queries
        casual_queries = [
            "sip on",
            "drink",
            "have",
            "get",
            "try",
            "taste",
            "sample",
            "something to",
            "anything that",
            "whatever is",
            "just want",
            "looking for",
            "in the mood for",
            "feel like",
            "craving",
        ]

        for phrase in casual_queries:
            if phrase in user_query_lower:
                return (
                    f"This {coffee_name} is perfect for casual sipping and "
                    f"enjoyment. It offers a pleasant, approachable flavor "
                    f"that's great for relaxed moments and everyday enjoyment."
                )

        # Handle time-based or context-specific queries
        time_contexts = {
            "morning": f"This {coffee_name} is ideal for starting your day - it has bright, energizing qualities that will wake you up and get you going.",
            "afternoon": f"This {coffee_name} is perfect for an afternoon pick-me-up - it provides just the right amount of energy without being too intense.",
            "evening": f"This {coffee_name} offers a smooth, gentle profile that's perfect for evening relaxation and winding down.",
            "night": f"This {coffee_name} has a mild, comforting profile that's great for evening enjoyment without keeping you up.",
            "today": f"This {coffee_name} is a great choice for today - it offers a well-balanced flavor that should satisfy your current mood.",
            "now": f"This {coffee_name} is perfect for right now - it provides immediate satisfaction with its pleasant, approachable profile.",
        }

        for time_word, response in time_contexts.items():
            if time_word in user_query_lower:
                return response

        if any(word in user_query_lower for word in positive_words):
            return (
                f"This {coffee_name} is an excellent choice that should "
                f"delight your taste buds and provide a wonderful coffee experience."
            )
        elif any(word in user_query_lower for word in negative_words):
            return (
                f"This {coffee_name} offers a different profile that might "
                f"change your mind about coffee - it's smooth and approachable."
            )

        # If we have characteristics but no direct matches, use them
        if characteristics:
            char_text = ", ".join(characteristics)
            return (
                f"This {coffee_name} is perfect for you because it features "
                f"{char_text} that align with your preferences for '{user_query}'."
            )

        # Generic but helpful responses based on query length and content
        if len(user_query.split()) <= 2:
            # Short queries like "coffee", "latte", "strong"
            return (
                f"This {coffee_name} is a great choice for your coffee preference. "
                f"It offers a well-balanced flavor profile that should satisfy your taste."
            )
        else:
            # Longer, more specific queries
            return (
                f"Based on your request for '{user_query}', this {coffee_name} "
                f"is well-suited to your preferences and should provide an excellent "
                f"coffee experience that matches what you're looking for."
            )


def generate_coffee_comment(
    user_query: str, coffee_data: dict, model_name: str = "microsoft/DialoGPT-small"
) -> str:
    """
    Convenience function to generate a coffee comment.

    Args:
        user_query: The user's original query/preferences
        coffee_data: Dictionary containing coffee information
        model_name: The model to use for generation

    Returns:
        A personalized comment about the coffee recommendation
    """
    generator = CoffeeCommentGenerator(model_name)
    return generator.generate_comment(user_query, coffee_data)
