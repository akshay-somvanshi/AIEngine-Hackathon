#!/usr/bin/env python3
"""
Coffee Dataset Generation Script

This script generates a comprehensive dataset for fine-tuning an LLM to recommend
coffee based on user preferences and mood, using real coffee data from
coffee_analysis.csv.
"""

import json
import pandas as pd
import random
from typing import List, Dict, Any
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize sentence transformer for semantic similarity
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loaded sentence transformer for semantic mood matching")
except Exception as e:
    print(f"Warning: Could not load sentence transformer: {e}")
    embedding_model = None


def load_coffee_analysis_data(
    file_path: str = "data/coffee_analysis.csv",
) -> pd.DataFrame:
    """Load the real coffee analysis data."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Coffee analysis file not found: {file_path}")

        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} coffee samples from {file_path}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading coffee data: {e}")
        return pd.DataFrame()


def find_semantically_similar_mood(
    input_mood: str, reference_moods: List[str], threshold: float = 0.7
) -> str:
    """Find the most semantically similar mood using embeddings."""
    if embedding_model is None:
        # Fallback to exact matching
        input_mood_lower = input_mood.lower()
        for mood in reference_moods:
            if input_mood_lower == mood.lower():
                return mood
        return "balanced"  # Default fallback

    try:
        # Encode the input mood
        input_embedding = embedding_model.encode([input_mood])[0]

        # Encode all reference moods
        reference_embeddings = embedding_model.encode(reference_moods)

        # Calculate similarities
        similarities = []
        for ref_embedding in reference_embeddings:
            similarity = np.dot(input_embedding, ref_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(ref_embedding)
            )
            similarities.append(similarity)

        # Find the most similar mood
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]

        if max_similarity >= threshold:
            return reference_moods[max_similarity_idx]
        else:
            return "balanced"  # Default if no good match

    except Exception as e:
        print(f"Error in semantic mood matching: {e}")
        return "balanced"


def map_mood_to_preferences_embedding(mood: str) -> Dict[str, Any]:
    """Map any mood to appropriate preferences using semantic similarity."""
    mood = mood.lower().strip()

    # Base moods with their preferences
    base_moods = {
        "energetic": {
            "preferences": ["bold", "strong", "intense", "high caffeine"],
            "context": "I want something with character and energy",
        },
        "tired": {
            "preferences": ["energizing", "bold flavors", "high caffeine"],
            "context": "I need something to wake me up",
        },
        "focused": {
            "preferences": ["balanced", "medium caffeine", "complex flavors"],
            "context": "I need to concentrate on work",
        },
        "happy": {
            "preferences": ["bright", "fruity", "sweet", "enjoyable"],
            "context": "I'm in a good mood and want something enjoyable",
        },
        "stressed": {
            "preferences": ["smooth", "mild", "relaxing", "calming"],
            "context": "I want something calming and gentle",
        },
        "relaxed": {
            "preferences": ["smooth", "creamy", "mild", "gentle"],
            "context": "I want something gentle and comforting",
        },
        "creative": {
            "preferences": ["complex", "unique", "inspiring", "artistic"],
            "context": "I want something that sparks creativity",
        },
        "social": {
            "preferences": ["smooth", "enjoyable", "conversational", "pleasant"],
            "context": "I want something to enjoy with others",
        },
        "adventurous": {
            "preferences": ["unique", "bold", "exotic", "different"],
            "context": "I want to try something new and exciting",
        },
        "balanced": {
            "preferences": ["balanced", "medium caffeine", "pleasant"],
            "context": "I want something suitable for my mood",
        },
    }

    # Find semantically similar mood
    reference_moods = list(base_moods.keys())
    similar_mood = find_semantically_similar_mood(mood, reference_moods)

    # Return mapped preferences
    if similar_mood in base_moods:
        return {
            "mood": mood,
            "preferences": base_moods[similar_mood]["preferences"],
            "context": base_moods[similar_mood]["context"],
            "similar_mood": similar_mood,  # For debugging
        }
    else:
        # Default fallback
        return {
            "mood": mood,
            "preferences": ["balanced", "medium caffeine", "pleasant"],
            "context": f"I'm feeling {mood} and want something suitable",
            "similar_mood": "balanced",
        }


def map_mood_to_preferences(mood: str) -> Dict[str, Any]:
    """Map any mood to appropriate preferences using semantic similarity."""
    mood = mood.lower().strip()

    # Mood synonyms and variations
    mood_mappings = {
        # Energy-related moods
        "energetic": [
            "energised",
            "energized",
            "energizing",
            "energizing",
            "awake",
            "alert",
            "lively",
            "vibrant",
        ],
        "tired": ["exhausted", "sleepy", "fatigued", "drowsy", "weary", "drained"],
        "focused": ["concentrated", "attentive", "mindful", "productive", "determined"],
        # Emotional states
        "happy": ["joyful", "cheerful", "content", "pleased", "satisfied", "upbeat"],
        "stressed": ["anxious", "worried", "tense", "overwhelmed", "pressured"],
        "relaxed": ["calm", "peaceful", "serene", "tranquil", "at ease", "chilled"],
        # Activity-based moods
        "creative": ["inspired", "artistic", "imaginative", "innovative"],
        "social": ["friendly", "outgoing", "sociable", "extroverted"],
        "adventurous": ["bold", "daring", "exploratory", "curious"],
    }

    # Find the base mood
    base_mood = None
    for base, variations in mood_mappings.items():
        if mood in variations or mood == base:
            base_mood = base
            break

    # Default mood preferences
    mood_preferences = {
        "energetic": {
            "preferences": ["bold", "strong", "intense", "high caffeine"],
            "context": "I want something with character and energy",
        },
        "tired": {
            "preferences": ["energizing", "bold flavors", "high caffeine"],
            "context": "I need something to wake me up",
        },
        "focused": {
            "preferences": ["balanced", "medium caffeine", "complex flavors"],
            "context": "I need to concentrate on work",
        },
        "happy": {
            "preferences": ["bright", "fruity", "sweet", "enjoyable"],
            "context": "I'm in a good mood and want something enjoyable",
        },
        "stressed": {
            "preferences": ["smooth", "mild", "relaxing", "calming"],
            "context": "I want something calming and gentle",
        },
        "relaxed": {
            "preferences": ["smooth", "creamy", "mild", "gentle"],
            "context": "I want something gentle and comforting",
        },
        "creative": {
            "preferences": ["complex", "unique", "inspiring", "artistic"],
            "context": "I want something that sparks creativity",
        },
        "social": {
            "preferences": ["smooth", "enjoyable", "conversational", "pleasant"],
            "context": "I want something to enjoy with others",
        },
        "adventurous": {
            "preferences": ["unique", "bold", "exotic", "different"],
            "context": "I want to try something new and exciting",
        },
    }

    # Return mapped preferences or default
    if base_mood and base_mood in mood_preferences:
        return {
            "mood": mood,
            "preferences": mood_preferences[base_mood]["preferences"],
            "context": mood_preferences[base_mood]["context"],
        }
    else:
        # Default for unknown moods
        return {
            "mood": mood,
            "preferences": ["balanced", "medium caffeine", "pleasant"],
            "context": f"I'm feeling {mood} and want something suitable",
        }


def extract_coffee_info(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract coffee information from the dataset."""
    coffee_info = []

    for _, row in df.iterrows():
        # Extract coffee name and origin
        coffee_name = row.get("name", "Unknown Coffee")
        origin = row.get("origin_1", "Unknown Origin")

        # Combine descriptions for flavor analysis
        desc_1 = str(row.get("desc_1", ""))
        desc_2 = str(row.get("desc_2", ""))
        desc_3 = str(row.get("desc_3", ""))

        # Create combined description
        combined_desc = f"{desc_1} {desc_2} {desc_3}".strip()

        # Extract flavors from descriptions
        flavors = extract_flavors_from_description(combined_desc)

        # Get rating if available
        rating = row.get("rating", 0)

        coffee_info.append(
            {
                "name": coffee_name,
                "origin": origin,
                "description": combined_desc,
                "flavors": flavors,
                "rating": rating,
                "roaster": row.get("roaster", "Unknown Roaster"),
                "roast_level": row.get("roast_level", "Medium"),
            }
        )

    return coffee_info


def extract_flavors_from_description(description: str) -> List[str]:
    """Extract flavor notes from coffee description."""
    description = description.lower()
    flavors = []

    # Common coffee flavor keywords
    flavor_keywords = {
        "fruity": ["fruit", "berry", "citrus", "apple", "cherry", "orange", "lemon"],
        "chocolate": ["chocolate", "cocoa", "dark chocolate", "milk chocolate"],
        "nutty": ["nut", "almond", "hazelnut", "walnut", "pecan"],
        "caramel": ["caramel", "toffee", "butterscotch"],
        "earthy": ["earth", "soil", "mushroom", "woody"],
        "floral": ["flower", "jasmine", "rose", "lavender"],
        "spicy": ["spice", "cinnamon", "clove", "pepper"],
        "smooth": ["smooth", "creamy", "silky"],
        "bright": ["bright", "vibrant", "lively"],
        "balanced": ["balanced", "harmonious", "well-rounded"],
        "bold": ["bold", "strong", "intense"],
        "mild": ["mild", "gentle", "soft"],
        "sweet": ["sweet", "honey", "sugar"],
        "acidic": ["acid", "bright", "tart"],
        "full-bodied": ["full-bodied", "rich", "heavy"],
    }

    for flavor_category, keywords in flavor_keywords.items():
        for keyword in keywords:
            if keyword in description:
                flavors.append(flavor_category)
                break

    # If no flavors found, add some default ones
    if not flavors:
        if "smooth" in description or "creamy" in description:
            flavors.append("smooth")
        elif "bold" in description or "strong" in description:
            flavors.append("bold")
        else:
            flavors.append("balanced")

    return list(set(flavors))  # Remove duplicates


def generate_input_text(
    mood: str, preferences: List[str], additional_context: str = ""
) -> str:
    """Generate input text based on mood and preferences."""
    input_parts = []

    if mood:
        input_parts.append(f"I'm feeling {mood}")

    if preferences:
        pref_text = ", ".join(preferences)
        input_parts.append(f"I prefer {pref_text}")

    if additional_context:
        input_parts.append(additional_context)

    return ". ".join(input_parts) + "."


def generate_output_text(coffee_info: Dict[str, Any]) -> str:
    """Generate output text based on real coffee information."""
    name = coffee_info["name"]
    origin = coffee_info["origin"]
    flavors = coffee_info["flavors"]
    description = coffee_info["description"]
    rating = coffee_info["rating"]
    roaster = coffee_info["roaster"]

    # Create structured output
    output_parts = [
        f"**Coffee:** {name}",
        f"**Origin:** {origin}",
        f"**Roaster:** {roaster}",
        f"**Flavors:** {', '.join(flavors)}",
    ]

    if rating > 0:
        output_parts.append(f"**Rating:** {rating}/100")

    if description and len(description.strip()) > 10:
        # Truncate description if too long
        short_desc = (
            description[:200] + "..." if len(description) > 200 else description
        )
        output_parts.append(f"**Description:** {short_desc}")

    return "\n".join(output_parts)


def generate_dataset(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate dataset using real coffee data."""
    # Load real coffee data
    df = load_coffee_analysis_data()
    if df.empty:
        print("Failed to load coffee data, returning empty dataset")
        return []

    coffee_info_list = extract_coffee_info(df)
    if not coffee_info_list:
        print("No coffee information extracted, returning empty dataset")
        return []

    print(f"Extracted {len(coffee_info_list)} coffee samples with flavor information")

    # Common moods for training data generation
    common_moods = [
        "tired",
        "energetic",
        "focused",
        "happy",
        "stressed",
        "relaxed",
        "energised",
        "exhausted",
        "concentrated",
        "joyful",
        "anxious",
        "calm",
        "creative",
        "social",
        "adventurous",
        "inspired",
        "worried",
        "peaceful",
    ]

    dataset = []

    for _ in range(num_samples):
        # Randomly select mood and coffee
        mood = random.choice(common_moods)
        coffee_info = random.choice(coffee_info_list)

        # Map mood to preferences using semantic similarity
        mood_pref = map_mood_to_preferences_embedding(mood)

        # Generate input and output
        input_text = generate_input_text(
            mood_pref["mood"], mood_pref["preferences"], mood_pref["context"]
        )

        output_text = generate_output_text(coffee_info)

        # Create training example
        example = {
            "input": input_text,
            "output": output_text,
            "coffee_name": coffee_info["name"],
            "origin": coffee_info["origin"],
            "flavors": coffee_info["flavors"],
            "mood": mood_pref["mood"],
            "preferences": mood_pref["preferences"],
        }

        dataset.append(example)

    print(f"Generated {len(dataset)} training examples")
    return dataset


def save_dataset(dataset: List[Dict[str, Any]], output_dir: str = "data"):
    """Save the generated dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    output_file = os.path.join(output_dir, "llm_training_data_2.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to {output_file}")

    # Print sample examples
    print("\nSample training examples:")
    for i, example in enumerate(dataset[:3]):
        print(f"\nExample {i+1}:")
        print(f"Input: {example['input']}")
        print(f"Output: {example['output']}")


def main():
    """Main function to generate the dataset."""
    print("Generating coffee recommendation dataset using real coffee data...")

    # Generate dataset
    dataset = generate_dataset(num_samples=500)  # Reduced for real data

    if dataset:
        # Save dataset
        save_dataset(dataset)
        print(f"\nDataset generation completed successfully!")
        print(f"Total examples: {len(dataset)}")
    else:
        print("Dataset generation failed!")


if __name__ == "__main__":
    main()
