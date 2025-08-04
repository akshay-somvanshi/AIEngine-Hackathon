#!/usr/bin/env python3
"""
Coffee Recommendation Web Application

This Flask app provides a web interface for the coffee recommendation system.
Users can input queries and get recommendations with coffee shop locations on a map.
"""

from flask import Flask, render_template, request, jsonify
import json
import os
from src.inference import HybridCoffeeRecommendationInference
from src.comment_generator import CoffeeCommentGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the recommendation system
recommender = None
comment_generator = None


def initialize_recommender():
    """Initialize the coffee recommendation system."""
    global recommender, comment_generator
    try:
        recommender = HybridCoffeeRecommendationInference(
            model_path="models/coffee_hybrid_recommender",
            coffee_data_path="data/unique_coffees.json",
        )

        # Initialize comment generator
        comment_generator = CoffeeCommentGenerator()

        logger.info("Coffee recommendation system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
        return False


def load_coffee_shops():
    """Load coffee shop data."""
    try:
        with open("data/coffee_shops_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("data/coffee_shops_data.json not found, using empty data")
        return {"coffeeShops": []}
    except Exception as e:
        logger.error(f"Error loading coffee shops: {e}")
        return {"coffeeShops": []}


def find_coffee_shops_with_coffees(recommended_coffees):
    """Find coffee shops that have the recommended coffees."""
    coffee_shops_data = load_coffee_shops()
    matching_shops = []

    # Create a set of recommended coffee names for faster lookup
    recommended_names = {coffee["name"].lower() for coffee in recommended_coffees}

    for shop in coffee_shops_data.get("coffeeShops", []):
        shop_coffees = shop.get("coffees", [])
        matching_coffees = []

        for coffee in shop_coffees:
            coffee_name = coffee.get("name", "").lower()
            if any(
                rec_name in coffee_name or coffee_name in rec_name
                for rec_name in recommended_names
            ):
                matching_coffees.append(coffee)

        if matching_coffees:
            shop_info = {
                "name": shop["name"],
                "address": shop["address"],
                "lat": shop["lat"],
                "lng": shop["lng"],
                "matching_coffees": matching_coffees,
            }
            matching_shops.append(shop_info)

    return matching_shops


@app.route("/")
def index():
    """Main page with the coffee recommendation interface."""
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """API endpoint for getting coffee recommendations."""
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Please provide a query"}), 400

        if not recommender:
            return jsonify({"error": "Recommendation system not initialized"}), 500

        # Get coffee recommendations using embeddings
        coffee_matches = recommender.find_best_coffee_matches(query, top_k=3)

        if not coffee_matches:
            return jsonify({"error": "No coffee matches found for your query"}), 404

        # Format recommendations with personalized comments
        recommendations = []
        for similarity_score, coffee in coffee_matches:
            # Generate personalized comment
            comment = ""
            if comment_generator:
                try:
                    comment = comment_generator.generate_comment(query, coffee)
                    # Validate the comment - if it's too short or contains unexpected content, use fallback
                    if len(comment) < 20 or "iced tea" in comment.lower():
                        comment = comment_generator._generate_fallback_comment(
                            query, coffee
                        )
                except Exception as e:
                    logger.warning(f"Failed to generate comment: {e}")
                    comment = comment_generator._generate_fallback_comment(
                        query, coffee
                    )
            else:
                comment = f"This {coffee['name']} is well-suited to your preferences."

            recommendations.append(
                {
                    "name": coffee["name"],
                    "roaster": coffee.get("roaster", "Unknown"),
                    "origin": coffee.get("origin", "Unknown"),
                    "rating": coffee.get("rating", "N/A"),
                    "description": coffee.get("description", ""),
                    "similarity_score": round(similarity_score, 3),
                    "personalized_comment": comment,
                }
            )

        # Find coffee shops that have these coffees
        matching_shops = find_coffee_shops_with_coffees(recommendations)

        return jsonify(
            {
                "query": query,
                "recommendations": recommendations,
                "coffee_shops": matching_shops,
                "total_shops_found": len(matching_shops),
            }
        )

    except Exception as e:
        logger.error(f"Error in recommendation API: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/api/coffee-shops")
def get_coffee_shops():
    """API endpoint to get all coffee shops data."""
    try:
        coffee_shops_data = load_coffee_shops()
        return jsonify(coffee_shops_data)
    except Exception as e:
        logger.error(f"Error loading coffee shops: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Initialize the recommendation system
    if initialize_recommender():
        logger.info("Starting Flask web application...")
        app.run(debug=True, host="0.0.0.0", port=5001)
    else:
        logger.error("Failed to initialize recommendation system. Exiting.")
