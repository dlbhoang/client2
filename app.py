from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

# Load các model và TF-IDF vectorizer đã lưu
with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('models/voting_model.pkl', 'rb') as f:
    voting_model = pickle.load(f)

from helpers import vn_processing as xt  # Giả sử có stepByStep

label_map = {'Tiêu cực': 0, 'Bình thường': 1, 'Tích cực': 2}
inv_label_map = {v: k for k, v in label_map.items()}

app = Flask(__name__)
CORS(app)  # Cho phép CORS cho tất cả routes

@app.route("/")
def home():
    return jsonify({"message": "API dự đoán nhãn bình luận hoạt động"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    comments = data.get("comments", [])
    
    df = pd.DataFrame({'Comment': comments})
    df['Comment Tokenize'] = df['Comment'].apply(xt.stepByStep)
    X = tfidf.transform(df['Comment Tokenize'])
    y_pred = voting_model.predict(X)
    df['Label'] = [inv_label_map[label] for label in y_pred]
    results = df[['Comment', 'Label']].to_dict(orient='records')
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
