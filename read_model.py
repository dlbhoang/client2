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
from helpers.find_adj_word import find_negative_words, find_positive_words
from wordcloud import WordCloud, STOPWORDS
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# Đọc dữ liệu từ tệp CSV vào DataFrame
data = pd.read_csv('data_cleaned/data_model.csv')

# Hiển thị vài dòng đầu tiên của DataFrame để xem cấu trúc dữ liệu
data.head()
# Hiển thị vài dòng cuối cùng của DataFrame để xem cấu trúc dữ liệu
data.tail()
# Hiển thị thông tin tổng quát về DataFrame, bao gồm các cột, số lượng dữ liệu không thiếu và kiểu dữ liệu của từng cột
data.info()
# Ánh xạ các giá trị trong cột 'Label' sang mã số tương ứng và lưu vào cột 'Label EnCode'
data['Label EnCode'] = data['Label'].map({'Negative': 0, 'Positive': 1, 'Neutral': 2})

# Hiển thị thông tin tổng quát về DataFrame sau khi thêm cột 'Label EnCode'
data.info()
# Lấy ngẫu nhiên 10 dòng từ DataFrame để xem mẫu dữ liệu
data.sample(10)
