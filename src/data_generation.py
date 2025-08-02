#!/usr/bin/env python3
"""
Coffee Dataset Generation Script

This script generates a comprehensive dataset for fine-tuning an LLM to recommend
coffee based on user preferences and mood.
"""

import json
import pandas as pd
import random
from typing import List, Dict, Any

# Coffee types and their characteristics
COFFEE_TYPES = {
    "Espresso": {
        "roast_level": "Dark",
        "caffeine_level": "High",
        "brewing_method": "Espresso machine",
        "flavors": ["Bold", "Earthy", "Chocolate", "Caramel"],
        "best_for": ["Energy boost", "Quick caffeine", "Bold flavor lovers"],
    },
    "Americano": {
        "roast_level": "Medium-Dark",
        "caffeine_level": "Medium-High",
        "brewing_method": "Espresso + hot water",
        "flavors": ["Smooth", "Balanced", "Mild bitterness"],
        "best_for": ["Morning routine", "Smooth coffee experience"],
    },
    "Cappuccino": {
        "roast_level": "Medium",
        "caffeine_level": "Medium",
        "brewing_method": "Espresso + steamed milk + foam",
        "flavors": ["Creamy", "Smooth", "Mild", "Sweet"],
        "best_for": ["Relaxation", "Mild coffee lovers", "Afternoon break"],
    },
    "Latte": {
        "roast_level": "Medium",
        "caffeine_level": "Medium",
        "brewing_method": "Espresso + steamed milk",
        "flavors": ["Creamy", "Smooth", "Mild", "Sweet"],
        "best_for": ["Relaxation", "Mild coffee lovers", "Social drinking"],
    },
    "Pour Over": {
        "roast_level": "Light-Medium",
        "caffeine_level": "Medium",
        "brewing_method": "Pour over filter",
        "flavors": ["Bright", "Fruity", "Clean", "Complex"],
        "best_for": ["Focus", "Flavor appreciation", "Mindful drinking"],
    },
    "French Press": {
        "roast_level": "Medium-Dark",
        "caffeine_level": "Medium-High",
        "brewing_method": "French press immersion",
        "flavors": ["Full-bodied", "Rich", "Oily", "Bold"],
        "best_for": ["Rich flavor", "Full-bodied experience", "Weekend mornings"],
    },
    "Cold Brew": {
        "roast_level": "Medium",
        "caffeine_level": "High",
        "brewing_method": "Cold water extraction",
        "flavors": ["Smooth", "Low acidity", "Sweet", "Mild"],
        "best_for": ["Hot weather", "Smooth caffeine", "Iced coffee lovers"],
    },
    "Turkish Coffee": {
        "roast_level": "Dark",
        "caffeine_level": "High",
        "brewing_method": "Boiled with sugar",
        "flavors": ["Strong", "Sweet", "Spicy", "Traditional"],
        "best_for": ["Cultural experience", "Strong coffee", "Traditional taste"],
    },
}

# Bean origins and their characteristics
BEAN_ORIGINS = {
    "Ethiopian Yirgacheffe": ["Bright", "Fruity", "Floral", "Light body"],
    "Colombian": ["Balanced", "Medium body", "Nutty", "Smooth"],
    "Brazilian": ["Low acidity", "Nutty", "Chocolate", "Full body"],
    "Guatemalan": ["Spicy", "Chocolate", "Medium body", "Complex"],
    "Costa Rican": ["Bright", "Clean", "Medium body", "Fruity"],
    "Sumatran": ["Earthy", "Full body", "Low acidity", "Spicy"],
    "Kenyan": ["Bright", "Fruity", "Wine-like", "Medium body"],
    "Peruvian": ["Mild", "Smooth", "Medium body", "Balanced"],
}

# Mood and preference combinations
MOOD_PREFERENCES = [
    {
        "mood": "stressed",
        "preferences": ["energizing", "bold flavors", "high caffeine"],
        "recommendations": ["Espresso", "Turkish Coffee", "Cold Brew"],
    },
    {
        "mood": "tired",
        "preferences": ["energizing", "strong", "quick boost"],
        "recommendations": ["Espresso", "Americano", "Cold Brew"],
    },
    {
        "mood": "relaxed",
        "preferences": ["smooth", "mild", "creamy"],
        "recommendations": ["Latte", "Cappuccino", "Pour Over"],
    },
    {
        "mood": "focused",
        "preferences": ["clear mind", "moderate caffeine", "complex flavors"],
        "recommendations": ["Pour Over", "French Press", "Americano"],
    },
    {
        "mood": "social",
        "preferences": ["enjoyable", "smooth", "mild"],
        "recommendations": ["Latte", "Cappuccino", "Pour Over"],
    },
    {
        "mood": "creative",
        "preferences": ["inspiration", "bright flavors", "moderate caffeine"],
        "recommendations": ["Pour Over", "Cold Brew", "French Press"],
    },
    {
        "mood": "productive",
        "preferences": ["sustained energy", "balanced", "moderate caffeine"],
        "recommendations": ["Americano", "French Press", "Espresso"],
    },
    {
        "mood": "adventurous",
        "preferences": ["unique flavors", "complex", "traditional"],
        "recommendations": ["Turkish Coffee", "Cold Brew", "French Press"],
    },
]


def generate_input_text(
    mood: str, preferences: List[str], additional_context: str = ""
) -> str:
    """Generate natural language input text."""
    templates = [
        f"I'm feeling {mood} and need something {', '.join(preferences)}. {additional_context}",
        f"Looking for coffee recommendations. I'm {mood} and prefer {', '.join(preferences)}. {additional_context}",
        f"Need coffee advice. Currently {mood} and want something {', '.join(preferences)}. {additional_context}",
        f"I'm in a {mood} mood today. Can you suggest coffee that's {', '.join(preferences)}? {additional_context}",
        f"Feeling {mood} and craving coffee that's {', '.join(preferences)}. {additional_context}",
    ]
    return random.choice(templates).strip()


def generate_output_text(
    coffee_type: str, coffee_info: Dict, bean_origin: str, bean_flavors: List[str]
) -> str:
    """Generate structured output text."""
    output = f"""Coffee Type: {coffee_type}
Roast Level: {coffee_info['roast_level']}
Bean Origin: {bean_origin}
Flavors: {', '.join(coffee_info['flavors'] + bean_flavors[:2])}
Brewing Method: {coffee_info['brewing_method']}
Caffeine Level: {coffee_info['caffeine_level']}
Best For: {', '.join(coffee_info['best_for'])}
Additional Notes: Perfect for your current mood and preferences with its {', '.join(coffee_info['flavors'][:2])} profile"""
    return output


def generate_dataset(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate the complete dataset."""
    dataset = []

    for i in range(num_samples):
        # Select random mood and preferences
        mood_config = random.choice(MOOD_PREFERENCES)
        mood = mood_config["mood"]
        preferences = mood_config["preferences"]

        # Add some random preferences
        additional_prefs = random.sample(
            [
                "sweet",
                "bitter",
                "smooth",
                "bold",
                "light",
                "dark",
                "creamy",
                "strong",
                "mild",
                "complex",
                "simple",
                "traditional",
                "modern",
                "organic",
                "fair trade",
            ],
            random.randint(1, 3),
        )
        all_preferences = preferences + additional_prefs

        # Select coffee type based on mood recommendations
        coffee_type = random.choice(mood_config["recommendations"])
        coffee_info = COFFEE_TYPES[coffee_type]

        # Select bean origin
        bean_origin = random.choice(list(BEAN_ORIGINS.keys()))
        bean_flavors = BEAN_ORIGINS[bean_origin]

        # Generate additional context
        contexts = [
            "I have about 10 minutes to enjoy it.",
            "I'm working from home today.",
            "I'm meeting friends later.",
            "I have a long day ahead.",
            "I need to stay focused.",
            "I want something special.",
            "I'm trying to cut back on caffeine.",
            "I love trying new flavors.",
            "I prefer traditional methods.",
            "I'm in a hurry this morning.",
        ]
        additional_context = random.choice(contexts)

        # Generate input and output
        input_text = generate_input_text(mood, all_preferences, additional_context)
        output_text = generate_output_text(
            coffee_type, coffee_info, bean_origin, bean_flavors
        )

        dataset.append(
            {
                "id": i,
                "input": input_text,
                "output": output_text,
                "mood": mood,
                "preferences": all_preferences,
                "coffee_type": coffee_type,
                "bean_origin": bean_origin,
            }
        )

    return dataset


def save_dataset(dataset: List[Dict[str, Any]], output_dir: str = "data"):
    """Save dataset in multiple formats."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    with open(f"{output_dir}/coffee_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    # Save as CSV
    df = pd.DataFrame(dataset)
    df.to_csv(f"{output_dir}/coffee_dataset.csv", index=False)

    # Save training format for fine-tuning
    training_data = []
    for item in dataset:
        training_data.append(
            {
                "instruction": "Recommend coffee based on preferences and mood",
                "input": item["input"],
                "output": item["output"],
            }
        )

    with open(f"{output_dir}/training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"Dataset saved to {output_dir}/")
    print(f"Total samples: {len(dataset)}")
    print(f"Formats: JSON, CSV, Training JSON")


def main():
    """Main function to generate and save the dataset."""
    print("Generating coffee recommendation dataset...")

    # Generate dataset
    dataset = generate_dataset(num_samples=2000)

    # Save dataset
    save_dataset(dataset)

    # Print some examples
    print("\nExample entries:")
    for i in range(3):
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {dataset[i]['input']}")
        print(f"Output: {dataset[i]['output']}")


if __name__ == "__main__":
    main()
