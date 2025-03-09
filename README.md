# Sentiment Analysis using NN, CNN and LSTM

## Overview
This repository contains a comprehensive implementation of sentiment analysis on movie reviews using various deep learning approaches. The project uses the IMDb dataset and demonstrates the effectiveness of different neural network architectures for text classification. The models achieve up to 86.5% accuracy using LSTM networks with GloVe word embeddings.

## Features
- Text preprocessing pipeline for cleaning and preparing review text
- Implementation of three neural network architectures:
  - Simple Neural Network (SNN): 74.8% accuracy
  - Convolutional Neural Network (CNN): 85.5% accuracy
  - Long Short-Term Memory Network (LSTM): 86.5% accuracy
- GloVe word embeddings integration (100-dimensional)
- Web interface for real-time sentiment analysis
- Rating prediction on a scale of 1-10

## Installation
```bash
# Clone the repository
git clone https://github.com/sarvkk/sentiment-analysis.git
cd sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Download GloVe embeddings
# Make sure the file a2_glove.6B.100d.txt is in the project directory
```

## Usage
### Running the Web Application
```bash
python app.py
```
Then navigate to http://localhost:5000 in your browser.

### Using the Jupyter Notebook
Open and run the SentimentAnalysis.ipynb notebook to:
- Train models from scratch
- Analyze model performance
- Make predictions on new reviews

## Project Structure
- `ipynb/` - Jupyter notebooks with model training and analysis
- `templates/` - HTML templates for the web interface
- `app.py` - Flask web application
- `b2_preprocessing_function.py` - Text preprocessing utilities
- `b3_tokenizer.json` - Saved tokenizer for input processing
- `lstm_model.h5` - Pre-trained LSTM model
- `c2_IMDb_Unseen_Predictions.csv` - Sample predictions on unseen data

## Requirements
- Python 3.10+
- TensorFlow 2.x
- Keras
- NLTK
- pandas
- NumPy
- Flask
- scikit-learn
- seaborn
- matplotlib

## Model Performance
| Model                 | Accuracy |
|-----------------------|----------|
| Simple Neural Network | 74.8%    |
| CNN                   | 85.5%    |
| LSTM                  | 86.5%    |

## Acknowledgments
This project utilizes the IMDb movie review dataset for training and the GloVe word embeddings for text representation.