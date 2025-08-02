#!/usr/bin/env python3
"""
Coffee Recommendation Web Application

A web interface for the coffee recommendation system using Flask and Gradio.
"""

import gradio as gr
from inference import CoffeeRecommender
import os


def create_recommendation_interface():
    """Create the Gradio interface for coffee recommendations."""

    # Initialize the recommender
    model_path = "models/coffee_recommender"
    recommender = CoffeeRecommender(model_path)

    def generate_recommendation(user_input):
        """Generate coffee recommendation based on user input."""
        if not user_input.strip():
            return "Please provide your coffee preferences and mood."

        try:
            recommendation = recommender.generate_recommendation(user_input)
            return recommendation
        except Exception as e:
            return f"Error generating recommendation: {str(e)}"

    # Create Gradio interface
    interface = gr.Interface(
        fn=generate_recommendation,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Describe your coffee preferences and mood...\nExample: I'm feeling stressed and need something energizing with bold flavors.",
            label="Coffee Preferences & Mood",
        ),
        outputs=gr.Textbox(
            lines=10,
            label="Coffee Recommendation",
            placeholder="Your personalized coffee recommendation will appear here...",
        ),
        title="☕ Coffee Recommendation System",
        description="""
        Get personalized coffee recommendations based on your preferences and mood!
        
        **How to use:**
        1. Describe your current mood (e.g., stressed, relaxed, focused)
        2. Mention your coffee preferences (e.g., bold, smooth, sweet, bitter)
        3. Add any additional context (e.g., time of day, occasion)
        
        **Example inputs:**
        - "I'm feeling stressed and need something energizing with bold flavors"
        - "I'm relaxed and want a smooth, mild coffee to enjoy"
        - "I need to focus on work and want moderate caffeine"
        """,
        examples=[
            [
                "I'm feeling stressed and need something energizing. I prefer bold flavors."
            ],
            ["I'm tired and want a smooth, mild coffee to relax with."],
            ["I need to focus on work and want something with moderate caffeine."],
            ["I'm in a creative mood and want something unique and inspiring."],
            ["I'm meeting friends and want a social coffee that's enjoyable."],
            ["I'm adventurous and want to try something traditional and complex."],
            ["I'm productive and need sustained energy throughout the day."],
            ["I'm relaxed and want something creamy and smooth."],
        ],
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 800px;
            margin: 0 auto;
        }
        """,
    )

    return interface


def create_flask_app():
    """Create a Flask app as an alternative to Gradio."""
    from flask import Flask, render_template, request, jsonify

    app = Flask(__name__)

    # Initialize recommender
    model_path = "models/coffee_recommender"
    recommender = CoffeeRecommender(model_path)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/recommend", methods=["POST"])
    def recommend():
        data = request.get_json()
        user_input = data.get("input", "")

        if not user_input.strip():
            return jsonify({"error": "Please provide input"})

        try:
            recommendation = recommender.generate_recommendation(user_input)
            return jsonify({"recommendation": recommendation})
        except Exception as e:
            return jsonify({"error": str(e)})

    return app


def main():
    """Main function to run the web application."""
    import argparse

    parser = argparse.ArgumentParser(description="Coffee Recommendation Web App")
    parser.add_argument(
        "--interface",
        choices=["gradio", "flask"],
        default="gradio",
        help="Web interface to use",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the server on"
    )

    args = parser.parse_args()

    if args.interface == "gradio":
        print("Starting Gradio interface...")
        interface = create_recommendation_interface()
        interface.launch(server_port=args.port, share=False, show_error=True)
    elif args.interface == "flask":
        print("Starting Flask interface...")
        app = create_flask_app()
        app.run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
