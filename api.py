from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import nltk
import pandas as pd
import logging
import warnings
from helpers import vn_processing as xt  # Assuming vn_processing contains the stepByStep function

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Download NLTK punkt tokenizer
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load trained models and vectorizer
with open('models/voting_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Function to preprocess text and make predictions
def predict_sentiment(text):
    df = pd.DataFrame({'Comment': text})
    df['Comment Tokenize'] = df['Comment'].apply(xt.stepByStep)  # Preprocess comments

    # Transform the comments to TF-IDF
    X_test = tfidf.transform(df['Comment Tokenize'])
    y_pred = model.predict(X_test)

    df['Label'] = y_pred
    try:
        df['Label'] = df['Label'].astype(int)  # Ensure label is int for mapping
    except Exception as e:
        logging.warning(f"Label conversion to int failed: {e}")

    inv_label_map = {0: 'Tiêu cực', 1: 'Bình thường', 2: 'Tích cực'}

    # Map labels safely; if key not found, label as 'Không xác định'
    df['Label'] = df['Label'].map(lambda x: inv_label_map.get(x, 'Không xác định'))

    return df[['Comment', 'Label']].to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json  # Get JSON data from request
        text = content.get('text')  # Access 'text' key from JSON data

        if not text:  # Check if text is empty
            return jsonify({'error': 'Empty text provided'}), 400

        result = predict_sentiment([text])
        return jsonify(result)
    except KeyError:
        return jsonify({'error': 'Missing required parameter'}), 400
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500  # Use 500 for unexpected errors

# Define API endpoint for prediction
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """ Sentiment analysis API
    ---
    parameters:
      - name: data
        in: body
        required: true
        schema:
          type: object
          properties:
            comment:
              type: string
              description: The comment text for sentiment analysis
    responses:
      200:
        description: Sentiment analysis result
      400:
        description: Bad request
    """
    try:
        data = request.json
        comment = data.get('comment')

        if not comment:  # Check if comment is missing
            return jsonify({'error': 'Empty comment provided'}), 400

        predictions = predict_sentiment([comment])
        return jsonify(predictions)
    except Exception as e:
        logging.error(f"Error during API prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Bind to all interfaces
