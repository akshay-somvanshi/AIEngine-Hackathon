import pandas as pd
import json
import re
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import random


class CoffeeLLMDataPreparator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the coffee data preparator with embedding model."""
        self.model = SentenceTransformer(model_name)

        # Store unique coffees to prevent duplicates
        self.unique_coffees = {}  # key: (name, roaster), value: coffee data

        # User query templates for different moods and preferences
        self.query_templates = {
            "tired": [
                "I'm feeling tired and need energy",
                "I need something to wake me up",
                "I'm exhausted, what coffee can boost my energy?",
                "I need an energizing coffee",
                "Something strong to fight fatigue",
            ],
            "stressed": [
                "I'm feeling stressed and need something calming",
                "I need a relaxing coffee",
                "Something smooth to help me unwind",
                "I want a comforting coffee",
                "I need something soothing",
            ],
            "focused": [
                "I need to focus on work",
                "I want something for concentration",
                "I need mental clarity",
                "Something to help me study",
                "I want a coffee for productivity",
            ],
            "adventurous": [
                "I want to try something unique",
                "I'm looking for an unusual coffee",
                "Something exotic and different",
                "I want an adventurous flavor",
                "Surprise me with something new",
            ],
            "social": [
                "I'm having friends over",
                "I need something for a social gathering",
                "Something enjoyable to share",
                "I want a crowd-pleasing coffee",
                "Something warm and inviting",
            ],
            "taste_bold": [
                "I want a bold coffee",
                "I like strong, intense flavors",
                "Something robust and powerful",
                "I prefer bold and dark",
                "I want something with a strong taste",
            ],
            "taste_smooth": [
                "I want something smooth",
                "I prefer mild and gentle flavors",
                "Something creamy and soft",
                "I like smooth, easy-drinking coffee",
                "I want something mellow",
            ],
            "taste_fruity": [
                "I want fruity flavors",
                "I like bright, citrusy coffee",
                "Something with berry notes",
                "I prefer fruit-forward coffee",
                "I want something with bright acidity",
            ],
            "taste_chocolate": [
                "I want chocolate flavors",
                "I love chocolatey coffee",
                "Something with cocoa notes",
                "I want rich, chocolate tones",
                "I prefer coffee with chocolate undertones",
            ],
            "taste_nutty": [
                "I want nutty flavors",
                "I like coffee with nut notes",
                "Something with almond or hazelnut",
                "I prefer nutty undertones",
                "I want something with nut characteristics",
            ],
        }

    def load_coffee_dataset(
        self, file_path: str = "data/coffee_analysis.csv"
    ) -> pd.DataFrame:
        """Load the coffee reviews dataset."""
        try:
            # Try different separators and handle the complex CSV format
            df = None

            # First try with semicolon separator
            try:
                df = pd.read_csv(file_path, sep=";")
                print(f"Loaded dataset with semicolon separator: {len(df)} reviews")
            except:
                # Fallback to comma separator
                df = pd.read_csv(file_path)
                print(f"Loaded dataset with comma separator: {len(df)} reviews")

            # Clean up column names (remove extra semicolons)
            df.columns = [col.strip(";") for col in df.columns]
            print(f"Dataset columns: {list(df.columns)}")

            return df
        except FileNotFoundError:
            print(f"Dataset file {file_path} not found.")
            return None

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_coffee_characteristics(self, description: str) -> Dict[str, List[str]]:
        """Extract characteristics from coffee description for matching with queries."""

        mood_indicators = {
            "energizing": [
                "bold",
                "strong",
                "intense",
                "robust",
                "powerful",
                "vigorous",
                "lively",
                "bright",
                "sharp",
                "punchy",
                "invigorating",
            ],
            "relaxing": [
                "smooth",
                "mellow",
                "gentle",
                "soft",
                "calm",
                "peaceful",
                "soothing",
                "mild",
                "delicate",
                "subtle",
                "silky",
            ],
            "comforting": [
                "warm",
                "rich",
                "creamy",
                "cozy",
                "full-bodied",
                "enveloping",
                "comforting",
                "embracing",
                "satisfying",
                "nurturing",
            ],
            "sophisticated": [
                "complex",
                "nuanced",
                "elegant",
                "refined",
                "balanced",
                "sophisticated",
                "layered",
                "intricate",
                "distinguished",
            ],
            "adventurous": [
                "unique",
                "exotic",
                "unusual",
                "distinctive",
                "wild",
                "adventurous",
                "experimental",
                "rare",
                "intriguing",
            ],
        }

        flavor_indicators = {
            "chocolate": [
                "chocolate",
                "cocoa",
                "mocha",
                "dark chocolate",
                "milk chocolate",
            ],
            "fruity": [
                "fruit",
                "berry",
                "cherry",
                "apple",
                "citrus",
                "orange",
                "lemon",
                "grape",
            ],
            "nutty": ["nut", "almond", "hazelnut", "walnut", "pecan", "cashew"],
            "floral": ["floral", "flower", "jasmine", "rose", "lavender", "perfumed"],
            "spicy": ["spice", "pepper", "cinnamon", "clove", "cardamom", "ginger"],
            "sweet": ["sweet", "caramel", "vanilla", "honey", "sugar", "syrup"],
            "bold": ["bold", "strong", "intense", "robust", "powerful", "dark"],
            "smooth": ["smooth", "mild", "gentle", "soft", "creamy", "silky"],
        }

        description_lower = description.lower()

        detected_moods = []
        detected_flavors = []

        for mood, keywords in mood_indicators.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_moods.append(mood)

        for flavor, keywords in flavor_indicators.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_flavors.append(flavor)

        return {"moods": detected_moods, "flavors": detected_flavors}

    def process_unique_coffees(
        self, df: pd.DataFrame
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Process and deduplicate coffees to ensure unique entries."""
        unique_coffees = {}

        print("Processing unique coffees...")

        for idx, row in df.iterrows():
            # Get coffee description
            description_columns = ["desc_1", "desc", "description", "review", "notes"]
            original_desc = ""

            for col in description_columns:
                if col in row.index and pd.notna(row[col]):
                    original_desc = str(row[col]).strip('"')
                    break

            if len(original_desc) < 10:
                continue

            # Create unique key (name + roaster)
            name = str(row.get("name", "Unknown")).strip()
            roaster = str(row.get("roaster", "Unknown")).strip()
            coffee_key = (name, roaster)

            # Skip if we already have this coffee
            if coffee_key in unique_coffees:
                continue

            # Extract coffee characteristics
            characteristics = self.extract_coffee_characteristics(original_desc)

            # Store unique coffee
            unique_coffees[coffee_key] = {
                "name": name,
                "roaster": roaster,
                "origin": row.get("origin_1", row.get("loc_country", "Unknown")),
                "roast_level": row.get("roast", "Unknown"),
                "rating": (
                    float(row.get("rating", 0)) if pd.notna(row.get("rating")) else 0.0
                ),
                "description": original_desc,
                "characteristics": characteristics,
                "embedding": None,  # Will be computed later
            }

        print(f"Found {len(unique_coffees)} unique coffees")
        return unique_coffees

    def create_query_coffee_pairs(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create query-coffee pairs for LLM training with embeddings."""

        # First, get unique coffees
        unique_coffees = self.process_unique_coffees(df)

        # Compute embeddings for unique coffees
        print("Computing coffee embeddings...")
        for coffee_key, coffee_data in unique_coffees.items():
            coffee_embedding = self.model.encode([coffee_data["description"]])[
                0
            ].tolist()
            coffee_data["embedding"] = coffee_embedding

        # Store for later use
        self.unique_coffees = unique_coffees

        training_pairs = []
        print("Creating query-coffee pairs with embeddings...")

        # For each query type, create pairs with suitable coffees
        for query_type, queries in self.query_templates.items():
            print(f"Processing query type: {query_type}")

            for query in queries:
                # Create query embedding
                query_embedding = self.model.encode([query])[0].tolist()

                # Find coffees that match this query type
                matching_coffees = []

                for coffee_key, coffee_data in unique_coffees.items():
                    characteristics = coffee_data["characteristics"]

                    # Check if coffee matches this query type
                    is_match = False

                    if query_type in characteristics["moods"]:
                        is_match = True
                    elif query_type.startswith("taste_"):
                        flavor = query_type.replace("taste_", "")
                        if flavor in characteristics["flavors"]:
                            is_match = True

                    # If no specific match, include some coffees for diversity
                    if (
                        not is_match and random.random() < 0.1
                    ):  # 10% chance for diversity
                        is_match = True

                    if is_match:
                        matching_coffees.append(coffee_data)

                # Limit to top coffees by rating to avoid too many pairs
                matching_coffees.sort(key=lambda x: x["rating"], reverse=True)
                matching_coffees = matching_coffees[:5]  # Top 5 matches per query

                # Create training pairs
                for coffee_data in matching_coffees:
                    training_pair = {
                        "query": query,
                        "query_embedding": query_embedding,
                        "coffee_name": coffee_data["name"],
                        "coffee_roaster": coffee_data["roaster"],
                        "coffee_origin": coffee_data["origin"],
                        "coffee_roast": coffee_data["roast_level"],
                        "coffee_rating": coffee_data["rating"],
                        "coffee_description": coffee_data["description"],
                        "coffee_embedding": coffee_data["embedding"],
                        "query_type": query_type,
                        "coffee_characteristics": coffee_data["characteristics"],
                    }

                    training_pairs.append(training_pair)

        print(f"Created {len(training_pairs)} query-coffee training pairs")
        return training_pairs

    def find_best_matches(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Find best coffee matches for a query, ensuring diversity."""
        query_emb = self.model.encode([query])[0]

        # Calculate similarities for all unique coffees
        matches = []
        for coffee_key, coffee_data in self.unique_coffees.items():
            coffee_emb = np.array(coffee_data["embedding"])
            similarity = np.dot(query_emb, coffee_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(coffee_emb)
            )
            matches.append((similarity, coffee_data))

        # Sort by similarity and return top k unique matches
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[:top_k]

    def create_llm_training_format(
        self, training_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert training pairs to LLM fine-tuning format compatible with our training script."""
        llm_training_data = []

        for pair in training_pairs:
            # Create simple input format that our training script expects
            input_data = {"query": pair["query"]}

            # Create structured output format
            output_data = {
                "recommendation": {
                    "name": pair["coffee_name"],
                    "roaster": pair["coffee_roaster"],
                    "origin": pair["coffee_origin"],
                    "roast_level": pair["coffee_roast"],
                    "rating": pair["coffee_rating"],
                    "description": pair["coffee_description"],
                    "why_recommended": f"This coffee matches your preference for {pair['query_type'].replace('taste_', '').replace('_', ' ')} characteristics.",
                }
            }

            # Format for LLM training (simple format that our training script can handle)
            llm_example = {
                "input": json.dumps(input_data),
                "output": json.dumps(output_data),
            }

            llm_training_data.append(llm_example)

        return llm_training_data

    def create_embedding_features_dataset(
        self, training_pairs: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create a dataset with embedding features for analysis."""

        # Flatten the data for easier analysis
        flattened_data = []

        for pair in training_pairs:
            row = {
                "query": pair["query"],
                "query_type": pair["query_type"],
                "coffee_name": pair["coffee_name"],
                "coffee_roaster": pair["coffee_roaster"],
                "coffee_rating": pair["coffee_rating"],
                "coffee_description": pair["coffee_description"][
                    :200
                ],  # Truncate for readability
            }

            # Add query embedding dimensions as features
            for i, val in enumerate(pair["query_embedding"]):
                row[f"query_emb_{i}"] = val

            # Add coffee embedding dimensions as features
            for i, val in enumerate(pair["coffee_embedding"]):
                row[f"coffee_emb_{i}"] = val

            # Add similarity score between query and coffee embeddings
            query_emb = np.array(pair["query_embedding"])
            coffee_emb = np.array(pair["coffee_embedding"])
            similarity = np.dot(query_emb, coffee_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(coffee_emb)
            )
            row["embedding_similarity"] = similarity

            flattened_data.append(row)

        return pd.DataFrame(flattened_data)

    def save_training_data(
        self, training_pairs: List[Dict[str, Any]], output_dir: str = "data"
    ):
        """Save all training data formats."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save raw training pairs
        with open(f"{output_dir}/coffee_embedding_pairs.json", "w") as f:
            json.dump(training_pairs, f, indent=2)

        # 2. Save LLM training format
        llm_data = self.create_llm_training_format(training_pairs)
        with open(f"{output_dir}/llm_training_data_2.json", "w") as f:
            json.dump(llm_data, f, indent=2)

        # 3. Save embedding features dataset
        embedding_df = self.create_embedding_features_dataset(training_pairs)
        embedding_df.to_csv(f"{output_dir}/embedding_features_dataset.csv", index=False)

        # 4. Save sample for inspection
        sample_data = random.sample(llm_data, min(10, len(llm_data)))
        with open(f"{output_dir}/sample_training_data.json", "w") as f:
            json.dump(sample_data, f, indent=2)

        # 5. Save unique coffees for reference
        unique_coffees_list = [
            coffee_data for coffee_data in self.unique_coffees.values()
        ]
        with open(f"{output_dir}/unique_coffees.json", "w") as f:
            json.dump(unique_coffees_list, f, indent=2)

        print(f"Training data saved to {output_dir}/")
        print(f"- Raw pairs: {len(training_pairs)} entries")
        print(f"- LLM training examples: {len(llm_data)} entries")
        print(
            f"- Embedding features CSV: {embedding_df.shape[0]} rows x {embedding_df.shape[1]} columns"
        )
        print(f"- Unique coffees: {len(self.unique_coffees)} coffees")

    def analyze_embeddings(self, training_pairs: List[Dict[str, Any]]):
        """Analyze the embedding relationships."""
        print("\n" + "=" * 50)
        print("EMBEDDING ANALYSIS")
        print("=" * 50)

        # Calculate similarity statistics
        similarities = []
        query_types = []

        for pair in training_pairs:
            query_emb = np.array(pair["query_embedding"])
            coffee_emb = np.array(pair["coffee_embedding"])
            similarity = np.dot(query_emb, coffee_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(coffee_emb)
            )
            similarities.append(similarity)
            query_types.append(pair["query_type"])

        similarities = np.array(similarities)

        print(f"Embedding Similarity Statistics:")
        print(f"- Mean similarity: {similarities.mean():.3f}")
        print(f"- Std similarity: {similarities.std():.3f}")
        print(f"- Min similarity: {similarities.min():.3f}")
        print(f"- Max similarity: {similarities.max():.3f}")

        # Analyze by query type
        unique_types = list(set(query_types))
        print(f"\nSimilarity by Query Type:")
        for qtype in unique_types:
            type_similarities = [
                s for s, t in zip(similarities, query_types) if t == qtype
            ]
            if type_similarities:  # Check if list is not empty
                print(
                    f"- {qtype}: {np.mean(type_similarities):.3f} ± {np.std(type_similarities):.3f}"
                )

        # Analyze coffee diversity
        self.analyze_coffee_diversity()

    def analyze_coffee_diversity(self):
        """Analyze diversity of coffee recommendations."""
        print(f"\nCoffee Diversity Analysis:")
        print(f"- Total unique coffees: {len(self.unique_coffees)}")

        # Analyze by roaster
        roasters = [coffee["roaster"] for coffee in self.unique_coffees.values()]
        unique_roasters = len(set(roasters))
        print(f"- Unique roasters: {unique_roasters}")

        # Analyze by origin
        origins = [coffee["origin"] for coffee in self.unique_coffees.values()]
        unique_origins = len(set(origins))
        print(f"- Unique origins: {unique_origins}")

        # Show top roasters
        from collections import Counter

        roaster_counts = Counter(roasters)
        print(f"\nTop 5 roasters by coffee count:")
        for roaster, count in roaster_counts.most_common(5):
            print(f"- {roaster}: {count} coffees")

    def test_embedding_quality(self, training_pairs: List[Dict[str, Any]]):
        """Test embedding quality with specific examples."""
        print("\n" + "=" * 60)
        print("EMBEDDING QUALITY TEST")
        print("=" * 60)

        # Test 1: Similar queries should have similar embeddings
        print("Test 1: Query Similarity Test")
        print("-" * 30)

        test_queries = [
            "I'm feeling tired and need energy",
            "I need something to wake me up",
            "I want something smooth and gentle",
            "I need a relaxing coffee",
        ]

        query_embeddings = self.model.encode(test_queries)

        print("Query similarity matrix:")
        for i, q1 in enumerate(test_queries):
            print(f"\n'{q1[:30]}...'")
            for j, q2 in enumerate(test_queries):
                if i != j:
                    similarity = np.dot(query_embeddings[i], query_embeddings[j]) / (
                        np.linalg.norm(query_embeddings[i])
                        * np.linalg.norm(query_embeddings[j])
                    )
                    print(f"  vs '{q2[:30]}...': {similarity:.3f}")

        # Test 2: Find best matches for sample queries (FIXED VERSION)
        print(f"\n\nTest 2: Best Coffee Matches (Diverse Results)")
        print("-" * 30)

        sample_queries = [
            "I'm feeling tired",
            "I want something smooth",
            "I need bold flavors",
        ]

        for query in sample_queries:
            print(f"\nQuery: '{query}'")
            matches = self.find_best_matches(query, top_k=3)

            print("Top 3 matches:")
            for i, (sim, coffee_data) in enumerate(matches):
                print(
                    f"  {i+1}. {coffee_data['name']} by {coffee_data['roaster']} ({sim:.3f})"
                )
                print(f"     '{coffee_data['description'][:60]}...'")

    def interactive_embedding_test(self, training_pairs: List[Dict[str, Any]]):
        """Interactive function to test embeddings with custom queries."""
        print("\n" + "=" * 50)
        print("INTERACTIVE EMBEDDING TEST")
        print("=" * 50)
        print("Enter your coffee preferences to see how embeddings work!")
        print("Type 'quit' to exit")

        while True:
            query = input("\nWhat kind of coffee do you want? ")
            if query.lower() == "quit":
                break

            if not query.strip():
                continue

            print(f"\nSearching for: '{query}'")
            matches = self.find_best_matches(query, top_k=3)

            print("\nTop 3 recommendations:")
            for i, (sim, coffee_data) in enumerate(matches):
                print(f"\n{i+1}. {coffee_data['name']} by {coffee_data['roaster']}")
                print(f"   Similarity: {sim:.3f}")
                print(f"   Rating: {coffee_data['rating']}")
                print(f"   Description: {coffee_data['description'][:100]}...")
                print(f"   Characteristics: {coffee_data['characteristics']}")

    def quick_embedding_check(self, training_pairs: List[Dict[str, Any]]):
        """Quick check to verify embeddings are working."""
        print("\n" + "=" * 40)
        print("QUICK EMBEDDING CHECK")
        print("=" * 40)

        if not training_pairs:
            print("❌ No training pairs found!")
            return False

        # Check 1: Embeddings exist and have correct dimensions
        sample_pair = training_pairs[0]
        query_emb = sample_pair["query_embedding"]
        coffee_emb = sample_pair["coffee_embedding"]

        print(f"Query embedding shape: {len(query_emb)} dimensions")
        print(f"Coffee embedding shape: {len(coffee_emb)} dimensions")

        # Check 2: Embeddings are not all zeros
        query_sum = sum(abs(x) for x in query_emb)
        coffee_sum = sum(abs(x) for x in coffee_emb)

        if query_sum > 0 and coffee_sum > 0:
            print("Embeddings contain non-zero values")
        else:
            print("Embeddings are all zeros!")
            return False

        # Check 3: Test a simple similarity
        similarity = np.dot(query_emb, coffee_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(coffee_emb)
        )
        print(f"Sample similarity score: {similarity:.3f}")

        # Check 4: Verify we have unique coffees
        print(f"Unique coffees available: {len(self.unique_coffees)}")

        # Check 5: Show a sample pairing
        print(f"\nSample pairing:")
        print(f"Query: '{sample_pair['query']}'")
        print(
            f"Coffee: {sample_pair['coffee_name']} by {sample_pair['coffee_roaster']}"
        )
        print(f"Description: {sample_pair['coffee_description'][:60]}...")
        print(f"Match quality: {similarity:.3f}")

        return True


def main():
    """Main function to create LLM training data with embeddings."""
    print("Coffee LLM Training Data Preparation with Embeddings - FIXED VERSION")
    print("=" * 70)

    # Initialize preparator
    preparator = CoffeeLLMDataPreparator()

    # Load dataset
    df = preparator.load_coffee_dataset()
    if df is None:
        return

    # Create query-coffee pairs with embeddings
    training_pairs = preparator.create_query_coffee_pairs(df)

    if not training_pairs:
        print("No training pairs created. Please check your data.")
        return

    # Quick embedding check first
    if not preparator.quick_embedding_check(training_pairs):
        print("Embedding check failed!")
        return

    # Analyze embeddings
    preparator.analyze_embeddings(training_pairs)

    # Test embedding quality with diverse results
    preparator.test_embedding_quality(training_pairs)

    # Save training data
    preparator.save_training_data(training_pairs)

    print("\n" + "=" * 70)
    print("TRAINING DATA READY FOR LLM FINE-TUNING!")
    print("Files created:")
    print("- llm_training_data.json: Ready for fine-tuning")
    print("- embedding_features_dataset.csv: For analysis")
    print("- sample_training_data.json: For inspection")
    print("- unique_coffees.json: List of all unique coffees")
    print("=" * 70)

    # Ask if user wants interactive testing
    print("\nWould you like to test the embeddings interactively? (y/n)")
    response = input().lower().strip()
    if response in ["y", "yes"]:
        preparator.interactive_embedding_test(training_pairs)


if __name__ == "__main__":
    # Install required packages
    print("Installing required packages...")
    os.system("pip install sentence-transformers pandas numpy")

    main()
