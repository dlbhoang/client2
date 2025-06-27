# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt 
import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime 
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
import helpers.vn_processing as xt
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Tải dữ liệu từ tệp CSV vào DataFrame
data = pd.read_csv('data_cleaned/data_model.csv')

# Ánh xạ các giá trị trong cột 'Label' sang mã số tương ứng và lưu vào cột 'Label EnCode'
data['Label EnCode'] = data['Label'].map({'Negative': 0, 'Positive': 1, 'Neutral': 2})

# Khởi tạo đối tượng TfidfVectorizer với các tham số cụ thể
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),  # Phạm vi số từ ghép (n-grams) từ 1 đến 3
    min_df=0.02,         # Bỏ qua các từ có tần số xuất hiện thấp hơn 2%
    max_df=0.9           # Bỏ qua các từ có tần số xuất hiện cao hơn 90%
)

# Fit vectorizer để học từ vựng từ cột 'Comment Tokenize' trong DataFrame data
tfidf.fit(data['Comment Tokenize'])

# Biến đổi dữ liệu văn bản từ cột 'Comment Tokenize' thành ma trận TF-IDF
X = tfidf.transform(data['Comment Tokenize'])

# Tạo DataFrame mới với các từ đã được biến đổi thành vectơ
df_new = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

# Lưu trữ mô hình TfidfVectorizer đã được huấn luyện vào tệp tin 'model/tfidf.pkl'
with open('model/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(df_new, data['Label EnCode'], test_size=0.2, random_state=42)

# Danh sách các mô hình cùng với các tham số của chúng
models = [
    ('DecisionTree', DecisionTreeClassifier()),
    ('RandomForest_100', RandomForestClassifier(n_estimators=100)),
    ('RandomForest_200', RandomForestClassifier(n_estimators=200)),
    ('RandomForest_300', RandomForestClassifier(n_estimators=300)),
    ('XGBoost', XGBClassifier(n_estimators=100)),
    ('NaiveBayes', MultinomialNB())
]

# Tạo DataFrame để lưu trữ kết quả đánh giá các mô hình
results = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'accuracy_standard_deviation', 'training_time'])

# Vòng lặp qua tất cả các mô hình và tìm kết quả
for name, model in models:
    # Bắt đầu đồng hồ
    start = datetime.now()
    
    # Huấn luyện mô hình với cross-validation
    cross_val_results = cross_validate(model, X_train, y_train, cv=5, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
    
    # Kết thúc đồng hồ
    end = datetime.now()
    
    # Tính thời gian huấn luyện
    training_time = end - start
    
    # Tính trung bình và độ lệch chuẩn của các điểm số
    mean_accuracy = cross_val_results['test_accuracy'].mean()
    mean_precision = cross_val_results['test_precision_macro'].mean()
    mean_recall = cross_val_results['test_recall_macro'].mean()
    mean_f1 = cross_val_results['test_f1_macro'].mean()
    std_accuracy = cross_val_results['test_accuracy'].std()
    
    # Ghép các kết quả vào DataFrame results
    results = pd.concat([results,
                         pd.DataFrame([[name, 
                                        mean_accuracy, 
                                        mean_precision,
                                        mean_recall, 
                                        mean_f1, 
                                        std_accuracy, 
                                        training_time]],
                                      columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'accuracy_standard_deviation', 'training_time'])])
    
    # In thông tin log
    print('Model: {} runs in {}'.format(name, training_time))

# Sắp xếp kết quả theo độ chính xác giảm dần và đặt lại chỉ số
results.sort_values(by='accuracy', ascending=False, inplace=True)
results.reset_index(drop=True, inplace=True)

# In ra bảng kết quả
print(results)

# Lưu kết quả đánh giá các mô hình vào tệp CSV
results.to_csv('models/model_evaluation_results.csv', index=False)
