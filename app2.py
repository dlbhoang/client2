from flask import Flask, jsonify
from flask_cors import CORS
import pickle
import nltk
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from helpers import vn_processing as xt  # Preprocessing function

nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Load models
with open('models/voting_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# MongoDB Atlas connection
client = MongoClient("mongodb+srv://admin:admin@app.sj5nx.mongodb.net/?retryWrites=true&w=majority&appName=app")
db = client["test"]
comment_collection = db["comment"]

# Label mapping
label_map = {0: 'Tiêu cực', 1: 'Bình thường', 2: 'Tích cực'}
score_map = {'Tiêu cực': 0, 'Bình thường': 1, 'Tích cực': 2}

@app.route('/products/sorted-by-sentiment', methods=['GET'])
def sorted_by_sentiment():
    try:
        comments = list(comment_collection.find({}, {"id_product": 1, "content": 1, "star": 1}))
        if not comments:
            return jsonify([])

        df = pd.DataFrame(comments)
        df['Comment Tokenize'] = df['content'].apply(xt.stepByStep)

        X_test = tfidf.transform(df['Comment Tokenize'])
        df['Label'] = model.predict(X_test)
        df['Sentiment'] = df['Label'].map(label_map)
        df['SentimentScore'] = df['Sentiment'].map(score_map)

        # Group and sort
        grouped = df.groupby('id_product').agg({
            'SentimentScore': 'mean',
            'star': 'mean',
            'content': 'count'
        }).reset_index()

        grouped.rename(columns={
            'SentimentScore': 'avg_sentiment_score',
            'star': 'avg_star',
            'content': 'total_comments'
        }, inplace=True)

        grouped['id_product'] = grouped['id_product'].astype(str)

        # 👉 Sort theo avg_star trước, sau đó avg_sentiment_score
        sorted_result = grouped.sort_values(
            by=['avg_star', 'avg_sentiment_score'], 
            ascending=[False, False]
        )

        return jsonify(sorted_result.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/comments', methods=['GET'])
def get_all_comments():
    try:
        comments = list(comment_collection.find({}))

        for comment in comments:
            comment['_id'] = str(comment['_id'])
            comment['id_product'] = str(comment['id_product'])
            comment['id_user'] = str(comment['id_user'])
            comment['created_at'] = comment['created_at'].isoformat() if 'created_at' in comment else None

        return jsonify(comments)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
