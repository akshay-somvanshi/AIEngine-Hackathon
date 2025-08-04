# Coffee Recommendation System

This project was built on the AI-Engine Hackathon that recommends coffee based on user preferences and mood using the real [Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset) from Kaggle. The model takes natural language input describing coffee preferences and mood, then outputs detailed coffee recommendations including type, roast level, flavors, and most importantly, where the user can source the coffee from. 

## Features

- **Real Dataset**: Uses the actual Coffee Reviews Dataset from Kaggle with 2,000+ coffee reviews
- **Smart Preprocessing**: Extracts mood and preferences from review text
- **Interactive Interface**: Web app testing recommendations
- **Comprehensive Output**: Detailed coffee recommendations with store locations, coffee flavors, origins and rating

## Project Structure

```
├── data/
│   ├── coffee_analysis.csv          # Original Kaggle dataset
│   ├── coffee_shops_data.json       # Synthetically generated data on coffee store inventories
│   └── llm_training_data.json       # The embedded data which relates mood to coffee descriptions
|   └── unique_coffees.json          # List of all the unique coffees with their descriptions and embedding vectors
|
├── src/
│   ├── data_preprocessing.py       # Process Kaggle dataset
│   ├── training.py                 # Fine-tunes a language model with embedding integration for coffee recommendations
│   ├── inference.py                # Generate recommendations
│   └── web_app.py                  # Web interface
|   └── comment_generator           # Generates personalized comments explaining why a recommended coffee
|
├── models/
│   └── coffee_recommender/         # Trained model
├── setup.py                        # Setup script
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
python setup.py

# Or manually install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

1. Visit [Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset)
2. Download the dataset
3. Extract and place `coffee-analysis.csv` in the `data/`.

### 3. Preprocess Data

```bash
python src/data_preprocessing.py
```

### 4. Train Model

```bash
python src/training.py
```

### 5. Test Recommendations

```bash
# Interactive mode
python src/inference.py
```

### 6. Launch Web Interface

```bash
python src/web_app.py
```

## Dataset Information

The project uses the Coffee Reviews Dataset which contains:
- **2,000+ coffee reviews** from various brands and origins
- **Review text** with detailed descriptions
- **Brand, variety, origin** information
- **Roast levels** and ratings
- **Flavor profiles** extracted from reviews

## Model Architecture

- **Base Model**: Microsoft DialoGPT-medium (355M parameters)
- **Fine-tuning**: Instruction-following format
- **Input Format**: Natural language preferences and mood
- **Output Format**: Structured coffee recommendations

## Example Usage

### Input
```
"I'm feeling stressed and need something energizing. I prefer bold flavors and don't mind bitterness."
```

### Output
```
Coffee Type: Espresso
Brand: Stumptown Coffee Roasters
Origin: Ethiopia
Roast Level: Dark
Flavors: Bold, Chocolate, Earthy
Rating: 4.5/5
Additional Notes: Perfect for stress relief with its bold, complex profile and high caffeine content
```

## Configuration

### Training Parameters
- **Model**: `microsoft/DialoGPT-medium`
- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 5e-5
- **Max Length**: 512 tokens

### Hardware Requirements
- **Recommended**: 16GB RAM, GPU training
- **GPU**: CUDA-compatible GPU for faster training

## Web Interface

The project includes a web interface built with Gradio:

- **Interactive chat**: Describe your preferences and get instant recommendations
- **Example queries**: Pre-built examples to test the system
- **Responsive design**: Works on desktop and mobile
- **Real-time generation**: Instant coffee recommendations

## Performance

The fine-tuned model achieves:
- **Contextual understanding** of mood and preferences
- **Accurate flavor matching** based on user descriptions
- **Diverse recommendations** from the real coffee dataset
- **Structured outputs** with detailed coffee information

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset) by Kaggle user schmoyote
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the fine-tuning framework
- [Gradio](https://gradio.app/) for the web interface