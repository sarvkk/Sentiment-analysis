import os
import numpy as np
import json
import logging
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from b2_preprocessing_function import CustomPreprocess
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Create the app object
app = Flask(__name__)

# Constants
MAXLEN = 100
MODEL_PATH = os.environ.get('MODEL_PATH', 'assets/lstm_model.h5')
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', 'b3_tokenizer.json')

# Global variables
model = None
tokenizer = None
preprocessor = None

def load_resources():
    """Load model and tokenizer resources"""
    global model, tokenizer, preprocessor
    try:
        # Load model
        logger.info(f"Loading model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {TOKENIZER_PATH}")
        with open(TOKENIZER_PATH) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        
        # Initialize preprocessor
        preprocessor = CustomPreprocess()
        
        logger.info("All resources loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading resources: {str(e)}")
        return False

# Health check endpoint
@app.route('/health')
def health():
    status = {
        'status': 'healthy' if all([model, tokenizer, preprocessor]) else 'unhealthy',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'preprocessor_initialized': preprocessor is not None
    }
    status_code = 200 if status['status'] == 'healthy' else 503
    return jsonify(status), status_code

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure resources are loaded
    if model is None or tokenizer is None or preprocessor is None:
        if not load_resources():
            logger.error("Failed to load resources on-demand")
            return render_template('index.html', error="Model initialization failed. Please try again later.")
    
    try:
        # Get input text
        query_text = request.form.get('review', '')
        if not query_text:
            return render_template('index.html', error="Please enter a review.")
        
        # Log the incoming request (truncated for privacy)
        truncated_text = query_text[:50] + '...' if len(query_text) > 50 else query_text
        logger.info(f"Processing review: {truncated_text}")
        
        # Preprocess text
        query_processed = preprocessor.preprocess_text(query_text)
        
        # Tokenize and pad text
        query_tokenized = tokenizer.texts_to_sequences([query_processed])
        query_padded = pad_sequences(query_tokenized, padding='post', maxlen=MAXLEN)
        
        # Make prediction
        prediction = model.predict(query_padded, verbose=0)[0][0]
        rating = np.round(prediction * 10, 1)
        
        # Determine sentiment label
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        
        logger.info(f"Prediction result: {sentiment}, Rating: {rating}")
        
        # Return result
        return render_template('index.html', 
                              prediction_text=f"{sentiment} Review with probable IMDb rating as: {rating}")
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', error=f"Error processing your request: {str(e)}")

if __name__ == "__main__":
    # Load resources on startup
    if not load_resources():
        logger.error("Failed to load resources on startup - continuing but app may fail")
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 3000))
    
    # Run app
    app.run(
        host='0.0.0.0',  # Make the server publicly available
        port=port,
        debug=False      # Disable debug mode in production
    )