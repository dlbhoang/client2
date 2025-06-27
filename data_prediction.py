import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
from helpers import vn_processing as xt  # Giả sử chứa stepByStep

warnings.filterwarnings('ignore')

# Đọc dữ liệu
data = pd.read_csv('data_cleaned/data_model.csv')

# Ánh xạ nhãn sang số nguyên
label_map = {'Tiêu cực': 0, 'Bình thường': 1, 'Tích cực': 2}
y = data['Label'].map(label_map)

# Loại bỏ dòng có nhãn NaN và 'Comment Tokenize' NaN
valid_idx = y.notna() & data['Comment Tokenize'].notna()
data = data.loc[valid_idx]
y = y.loc[valid_idx]
X = data['Comment Tokenize']

print("Số mẫu sau làm sạch:", len(X), len(y))
print("Các nhãn trong y:", y.unique())

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=0.02, max_df=0.9)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Lưu TF-IDF
with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Khởi tạo mô hình
xgb_model = XGBClassifier(
    objective='multi:softmax', num_class=len(label_map),
    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
logistic_model = LogisticRegression(
    multi_class='multinomial', solver='lbfgs',
    max_iter=500, random_state=42
)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

models = {
    "XGBoost": xgb_model,
    "SVC": svm_model,
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy của {name}: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=label_map.keys()))

# VotingClassifier
voting_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('svm', svm_model),
        ('logistic', logistic_model),
        ('rf', rf_model)
    ],
    voting='soft'
)
voting_model.fit(X_train_tfidf, y_train)

with open('models/voting_model.pkl', 'wb') as f:
    pickle.dump(voting_model, f)

# Hàm dự đoán
def predict_comment(text_list):
    df = pd.DataFrame({'Comment': text_list})
    df['Comment Tokenize'] = df['Comment'].apply(xt.stepByStep)
    X = tfidf.transform(df['Comment Tokenize'])
    y_pred = voting_model.predict(X)
    inv_label_map = {v: k for k, v in label_map.items()}
    df['Label'] = y_pred
    df['Label'] = df['Label'].map(inv_label_map)
    return df[['Comment', 'Label']]

# Test thử
test_comments = ["Không biết có nên quay lại không."]
print(predict_comment(test_comments))
