#!/usr/bin/env python3
"""
Coffee Dataset Preprocessing Script

This script processes the real coffee reviews dataset from Kaggle to create
training data for fine-tuning an LLM to recommend coffee based on preferences.
"""

import pandas as pd
import json
from typing import List, Dict, Any
import re


def load_coffee_dataset(file_path: str = "data/coffee_reviews.csv") -> pd.DataFrame:
    """Load the coffee reviews dataset."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(df)} reviews")
        return df
    except FileNotFoundError:
        print(f"Dataset file {file_path} not found.")
        print("Please download the coffee reviews dataset from:")
        print("https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset")
        return None


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove special characters but keep spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_mood_and_preferences(review_text: str) -> Dict[str, Any]:
    """Extract mood and preferences from review text."""
    mood_keywords = {
        "energized": ["energizing", "energized", "boost", "wake", "alert"],
        "relaxed": ["relaxing", "calm", "smooth", "gentle", "peaceful"],
        "focused": ["focus", "concentration", "clear", "productive"],
        "stressed": ["stress", "anxiety", "tense", "worried"],
        "tired": ["tired", "exhausted", "fatigue", "sleepy"],
        "creative": ["creative", "inspired", "artistic", "imaginative"],
        "social": ["social", "friendly", "enjoyable", "pleasant"],
        "adventurous": ["adventurous", "unique", "different", "exciting"],
    }

    preference_keywords = {
        "bold": ["bold", "strong", "intense", "robust"],
        "smooth": ["smooth", "mild", "gentle", "soft"],
        "sweet": ["sweet", "caramel", "chocolate", "vanilla"],
        "bitter": ["bitter", "dark", "roasted", "burnt"],
        "fruity": ["fruity", "bright", "citrus", "berry"],
        "nutty": ["nutty", "almond", "hazelnut", "walnut"],
        "creamy": ["creamy", "rich", "full-bodied", "thick"],
        "light": ["light", "delicate", "subtle", "mild"],
        "dark": ["dark", "deep", "intense", "bold"],
    }

    text = review_text.lower()

    # Extract moods
    detected_moods = []
    for mood, keywords in mood_keywords.items():
        if any(keyword in text for keyword in keywords):
            detected_moods.append(mood)

    # Extract preferences
    detected_preferences = []
    for pref, keywords in preference_keywords.items():
        if any(keyword in text for keyword in keywords):
            detected_preferences.append(pref)

    return {"moods": detected_moods, "preferences": detected_preferences}


def create_input_text(row: pd.Series) -> str:
    """Create input text from coffee review data."""
    # Use review text as base
    review_text = clean_text(row.get("review_text", ""))

    # If review is too short, create synthetic input
    if len(review_text) < 20:
        # Extract information from other columns
        brand = clean_text(row.get("brand", ""))
        variety = clean_text(row.get("variety", ""))
        origin = clean_text(row.get("origin", ""))

        # Create synthetic input
        parts = []
        if brand:
            parts.append(f"coffee from {brand}")
        if variety:
            parts.append(f"{variety} variety")
        if origin:
            parts.append(f"from {origin}")

        if parts:
            review_text = f"I'm looking for {' '.join(parts)}"
        else:
            review_text = "I need coffee recommendations"

    return review_text


def create_output_text(row: pd.Series) -> str:
    """Create structured output text from coffee review data."""
    brand = row.get("brand", "Unknown")
    variety = row.get("variety", "Unknown")
    origin = row.get("origin", "Unknown")
    roast_level = row.get("roast_level", "Unknown")

    # Extract flavors from review
    review_text = clean_text(row.get("review_text", ""))

    # Simple flavor extraction
    flavors = []
    flavor_keywords = {
        "chocolate": ["chocolate", "cocoa", "mocha"],
        "caramel": ["caramel", "toffee", "butterscotch"],
        "fruity": ["fruit", "berry", "citrus", "apple", "cherry"],
        "nutty": ["nut", "almond", "hazelnut", "walnut"],
        "earthy": ["earth", "woody", "mushroom"],
        "spicy": ["spice", "pepper", "cinnamon"],
        "floral": ["flower", "jasmine", "rose"],
        "vanilla": ["vanilla", "sweet", "creamy"],
    }

    for flavor, keywords in flavor_keywords.items():
        if any(keyword in review_text for keyword in keywords):
            flavors.append(flavor)

    # If no flavors detected, use common ones based on roast level
    if not flavors:
        if "dark" in str(roast_level).lower():
            flavors = ["bold", "chocolate", "earthy"]
        elif "light" in str(roast_level).lower():
            flavors = ["bright", "fruity", "clean"]
        else:
            flavors = ["balanced", "smooth", "medium"]

    # Create structured output
    output = f"""Coffee Type: {variety}
Brand: {brand}
Origin: {origin}
Roast Level: {roast_level}
Flavors: {', '.join(flavors[:3])}
Rating: {row.get('rating', 'N/A')}/5
Additional Notes: {review_text[:100]}{'...' if len(review_text) > 100 else ''}"""

    return output


def process_dataset(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Process the coffee dataset into training format."""
    training_data = []

    for idx, row in df.iterrows():
        # Skip rows with missing essential data
        if pd.isna(row.get("review_text")) or pd.isna(row.get("brand")):
            continue

        # Create input and output
        input_text = create_input_text(row)
        output_text = create_output_text(row)

        # Extract mood and preferences
        mood_prefs = extract_mood_and_preferences(row.get("review_text", ""))

        # Create training example
        training_example = {
            "id": idx,
            "input": input_text,
            "output": output_text,
            "brand": row.get("brand", ""),
            "variety": row.get("variety", ""),
            "origin": row.get("origin", ""),
            "rating": row.get("rating", 0),
            "moods": mood_prefs["moods"],
            "preferences": mood_prefs["preferences"],
        }

        training_data.append(training_example)

    return training_data


def save_processed_data(data: List[Dict[str, Any]], output_dir: str = "data"):
    """Save processed data in multiple formats."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    with open(f"{output_dir}/processed_coffee_data.json", "w") as f:
        json.dump(data, f, indent=2)

    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/processed_coffee_data.csv", index=False)

    # Save training format for fine-tuning
    training_data = []
    for item in data:
        training_data.append(
            {
                "instruction": "Recommend coffee based on preferences and mood",
                "input": item["input"],
                "output": item["output"],
            }
        )

    with open(f"{output_dir}/training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"Processed data saved to {output_dir}/")
    print(f"Total samples: {len(data)}")
    print(f"Formats: JSON, CSV, Training JSON")


def main():
    """Main function to process the coffee dataset."""
    print("Processing coffee reviews dataset...")

    # Load dataset
    df = load_coffee_dataset()
    if df is None:
        return

    # Process dataset
    processed_data = process_dataset(df)

    # Save processed data
    save_processed_data(processed_data)

    # Print some examples
    print("\nExample processed entries:")
    for i in range(min(3, len(processed_data))):
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {processed_data[i]['input']}")
        print(f"Output: {processed_data[i]['output']}")
        print(f"Moods: {processed_data[i]['moods']}")
        print(f"Preferences: {processed_data[i]['preferences']}")


if __name__ == "__main__":
    main()
