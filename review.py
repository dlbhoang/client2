import pandas as pd
from sqlalchemy import create_engine
import helpers.vn_processing as xt

# Đọc dữ liệu từ tệp CSV 'review.csv' vào DataFrame reviews
reviews = pd.read_csv('data/review.csv')

# Loại bỏ các hàng có giá trị thiếu trong cột 'Comment' và 'User'
reviews = reviews.dropna(subset=['Comment', 'User'])

# Xóa cột 'Unnamed: 0' khỏi DataFrame reviews
reviews = reviews.drop(columns=['Unnamed: 0'])

# Chuyển đổi kiểu dữ liệu của cột 'UserID' và 'RestaurantID' sang kiểu object (chuỗi)
reviews['UserID'] = reviews['UserID'].astype('str')
reviews['RestaurantID'] = reviews['RestaurantID'].astype('str')

# Nhóm và đếm số lượng đánh giá theo từng mức 'Rating' trong DataFrame reviews
rating = reviews.groupby('Rating')['User'].count().reset_index()
rating = rating.sort_values(by='Rating', ascending=False)

# Áp dụng hàm xt.stepByStep để tách từ trong cột 'Comment' của DataFrame reviews
reviews['Comment Tokenize'] = reviews['Comment'].apply(xt.stepByStep)

# Lưu DataFrame reviews đã được làm sạch vào tệp CSV 'review_cleaned.csv'
reviews.to_csv('data_cleaned/review_cleaned.csv', index=False)

