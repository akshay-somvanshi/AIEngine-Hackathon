# Coffee Recommendation LLM Fine-tuning

This project fine-tunes a language model to recommend coffee based on user preferences and mood using the real [Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset) from Kaggle. The model takes natural language input describing coffee preferences and mood, then outputs detailed coffee recommendations including type, roast level, flavors, and brewing method.

## 🚀 Features

- **Real Dataset**: Uses the actual Coffee Reviews Dataset from Kaggle with 2,000+ coffee reviews
- **Fine-tuning Pipeline**: Complete pipeline for fine-tuning LLMs (GPT-2, DialoGPT, etc.)
- **Smart Preprocessing**: Extracts mood and preferences from review text
- **Interactive Interface**: Web app and CLI for testing recommendations
- **Comprehensive Output**: Detailed coffee recommendations with flavors, origins, and ratings

## 📁 Project Structure

```
├── data/
│   ├── coffee_reviews.csv          # Original Kaggle dataset
│   ├── processed_coffee_data.json  # Processed training data
│   └── training_data.json         # Fine-tuning format
├── src/
│   ├── data_preprocessing.py      # Process Kaggle dataset
│   ├── training.py               # Fine-tune the model
│   ├── inference.py              # Generate recommendations
│   └── web_app.py               # Web interface
├── models/
│   └── coffee_recommender/       # Trained model
├── setup.py                     # Setup script
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## 🛠️ Quick Start

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
3. Extract and place `coffee-reviews-dataset.csv` in the `data/` directory as `coffee_reviews.csv`

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

# Test mode with examples
python src/inference.py --mode test
```

### 6. Launch Web Interface

```bash
python src/web_app.py
```

## 📊 Dataset Information

The project uses the Coffee Reviews Dataset which contains:
- **2,000+ coffee reviews** from various brands and origins
- **Review text** with detailed descriptions
- **Brand, variety, origin** information
- **Roast levels** and ratings
- **Flavor profiles** extracted from reviews

## 🧠 Model Architecture

- **Base Model**: Microsoft DialoGPT-medium (355M parameters)
- **Fine-tuning**: Instruction-following format
- **Input Format**: Natural language preferences and mood
- **Output Format**: Structured coffee recommendations

## 💡 Example Usage

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

## 🔧 Configuration

### Training Parameters
- **Model**: `microsoft/DialoGPT-medium`
- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 5e-5
- **Max Length**: 512 tokens

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB RAM, GPU training
- **GPU**: CUDA-compatible GPU for faster training

## 🌐 Web Interface

The project includes a beautiful web interface built with Gradio:

- **Interactive chat**: Describe your preferences and get instant recommendations
- **Example queries**: Pre-built examples to test the system
- **Responsive design**: Works on desktop and mobile
- **Real-time generation**: Instant coffee recommendations

## 📈 Performance

The fine-tuned model achieves:
- **Contextual understanding** of mood and preferences
- **Accurate flavor matching** based on user descriptions
- **Diverse recommendations** from the real coffee dataset
- **Structured outputs** with detailed coffee information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset) by Kaggle user schmoyote
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the fine-tuning framework
- [Gradio](https://gradio.app/) for the web interface

## 🆘 Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure `coffee_reviews.csv` is in the `data/` directory
2. **CUDA out of memory**: Reduce batch size in `training.py`
3. **Model loading error**: Check if training completed successfully
4. **Dependencies error**: Run `pip install -r requirements.txt`

### Getting Help

- Check the logs in `models/coffee_recommender/logs/`
- Verify dataset format matches expected structure
- Ensure sufficient disk space for model storage

---

**Enjoy your personalized coffee recommendations! ☕**