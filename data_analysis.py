import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Đọc dữ liệu từ file CSV vào DataFrame
restaurants = pd.read_csv('data_cleaned/restaurant_cleaned.csv')
reviews = pd.read_csv('data_cleaned/review_cleaned.csv')

# Merge hai DataFrame restaurants và reviews dựa trên cột 'RestaurantID'
data = pd.merge(restaurants, reviews, how='inner', on='RestaurantID')

# Xóa cột 'Comment' khỏi DataFrame data sau khi merge
data = data.drop(columns=['Comment'])

# Loại bỏ các dòng có giá trị NaN trong cột 'Comment Tokenize'
data.dropna(subset=['Comment Tokenize'], inplace=True)

# Lấy cột 'Comment Tokenize' text
text = data['Comment Tokenize']

# Thiết lập và fit vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=0.02, max_df=0.9)
text_transformed = tfidf.fit_transform(text)

# Tạo DataFrame từ text_transformed
df_text = pd.DataFrame(text_transformed.toarray(), columns=tfidf.get_feature_names_out())

# Tạo từ điển từ khóa tích cực với trọng số
positive_words = {
    "thích": 2, "tốt": 2, "xuất sắc": 3, "tuyệt vời": 3, "tuyệt hảo": 3,
    "đẹp": 2, "ổn": 2, "ngon": 2, "hài lòng": 2, "ưng ý": 2, "hoàn hảo": 3,
    "chất lượng": 2, "thú vị": 2, "nhanh": 2, "tiện lợi": 2, "dễ sử dụng": 2,
    "hiệu quả": 2, "ấn tượng": 2, "nổi bật": 2, "tận hưởng": 2, "tốn ít thời gian": 2,
    "thân thiện": 2, "hấp dẫn": 2, "gợi cảm": 2, "tươi mới": 2, "lạ mắt": 2,
    "cao cấp": 2, "độc đáo": 2, "hợp khẩu vị": 2, "rất tốt": 3, "rất thích": 3,
    "tận tâm": 2, "đáng tin cậy": 2, "đẳng cấp": 3, "an tâm": 2,
    "không thể cưỡng lại": 3, "thỏa mãn": 2, "thúc đẩy": 2, "cảm động": 2,
    "phục vụ tốt": 2, "làm hài lòng": 2, "gây ấn tượng": 2, "nổi trội": 2,
    "sáng tạo": 2, "quý báu": 2, "phù hợp": 2, "hiếm có": 2, "cải thiện": 2,
    "hoà nhã": 2, "chăm chỉ": 2, "cẩn thận": 2, "vui vẻ": 2, "sáng sủa": 2,
    "hào hứng": 2, "đam mê": 2, "vừa vặn": 2, "đáng tiền": 2, "nhiệt tình": 2,
    "best": 2, "good": 2, "nghiện": 2, "ngon nhất": 3, "quá ngon": 3,
    "quá tuyệt": 3, "đúng vị": 2, "điểm cộng": 2, "thức ăn ngon": 2,
    "khá ngon": 2, "niềm nở": 2, "đặc biệt": 2, "tươi ngon": 2, "thơm": 2,
    "sạch sẽ": 2, "món ngon": 2, "ăn rất ngon": 3, "giá rẻ": 2, "thích nhất": 2,
    "đồ ăn ngon": 2, "phục vụ nhanh": 2, "giá hợp": 2, "đa dạng": 2,
    "ngon giá": 2, "nhanh nhẹn": 2, "thoải mái": 2, "quán ngon": 2,
    "khen": 2, "tin tưởng": 2, "tôn trọng": 2, "tràn đầy": 2, "lôi cuốn": 2,
    "trẻ trung": 2, "tích cực": 2, "tinh tế": 3, "năng động": 2,
    "hài hòa": 2, "gần gũi": 2, "ngọt ngào": 2, "tuyệt mỹ": 3, "vượt trội": 3,
    "nhiệt huyết": 2, "phấn khởi": 2, "mang lại": 2, "vững bền": 2,
    "được yêu mến": 3, "quyến rũ": 2, "tỏa sáng": 3, "đang thịnh hành": 2,
    "đầy sức sống": 2, "truyền cảm hứng": 2, "phong phú": 2, "đầy hứa hẹn": 2,
    "trân quý": 2, "ngon lành": 3, "dễ chịu": 2, "nồng nàn": 2,
    "năng lực": 2, "tinh thần": 2, "mở mang": 2, "hòa hợp": 2, "tiến bộ": 3,
    "tự hào": 2, "thành công": 3, "tươi sáng": 2, "tích cực": 3, 
    "rực rỡ": 3, "hài lòng": 3, "đáng yêu": 2, "bền vững": 3,
    "mang lại niềm vui": 3, "sáng tạo": 2, "công bằng": 2,
    "mạnh mẽ": 2, "thân thiện với người dùng": 3, "hiện đại": 2,
    "sang trọng": 3, "có tâm": 2, "dễ dàng": 2, "dễ chịu": 2,
    "hòa bình": 2, "lạc quan": 2, "vui vẻ": 2, "danh tiếng": 3,
    "người hùng": 2, "thương hiệu tốt": 2
}


# Tạo từ điển từ khóa tiêu cực với trọng số
negative_words = {
    "kém": 2, "tệ": 2, "đau": 2, "xấu": 2, "không": 2, "dở": 3, "ức": 2,
    "buồn": 2, "rối": 2, "thô": 2, "lâu": 2, "chán": 2, "tối": 2, "ít": 2,
    "mờ": 2, "mỏng": 2, "lỏng lẻo": 2, "khó": 2, "cùi": 3, "yếu": 2,
    "kém chất lượng": 3, "không thích": 2, "không thú vị": 2, "không ổn": 2,
    "không hợp": 2, "không đáng tin cậy": 2, "không chuyên nghiệp": 2,
    "không phản hồi": 2, "không an toàn": 2, "không phù hợp": 2, "không thân thiện": 2,
    "không linh hoạt": 2, "không đáng giá": 2, "không ấn tượng": 2, "không tốt": 2,
    "chậm": 2, "khó khăn": 2, "phức tạp": 2, "khó hiểu": 2, "khó chịu": 2,
    "gây khó dễ": 2, "rườm rà": 2, "khó truy cập": 2, "thất bại": 3, "tồi tệ": 3,
    "khó xử": 2, "không thể chấp nhận": 3, "không rõ ràng": 2, "không chắc chắn": 2,
    "rối rắm": 2, "không tiện lợi": 2, "không đáng tiền": 2, "chưa đẹp": 2,
    "không đẹp": 2, "bad": 3, "thất vọng": 3, "không ngon": 2, "hôi": 2,
    "không ngon": 2, "không đáng": 2, "không xứng đáng": 2, "điểm trừ": 2,
    "thức ăn tệ": 3, "đồ ăn tệ": 3, "đợi lâu": 2, "nhạt nhẽo": 2,
    "không thoải mái": 2, "không đặc sắc": 2, "tanh": 2, "giá hơi mắc": 2,
    "giá hơi đắt": 2, "không chất lượng": 2, "chê": 2, "khó chịu": 2,
    "thất vọng": 3, "không hài lòng": 3, "vô lý": 2, "không đáp ứng": 2,
    "không đáng giá": 2, "khó xử": 2, "không tin tưởng": 2, "không đáng tin": 2,
    "khó chịu": 2, "không vừa ý": 2, "khó chịu": 2, "bực bội": 3,
    "lừa đảo": 3, "nhàm chán": 2, "không thực tế": 2, "khó khăn": 2,
    "không xứng đáng": 2, "đắt đỏ": 3, "không thỏa mãn": 3, "tồi tệ": 3
}


# Tạo cột 'Positive Score' và 'Negative Score' với trọng số
data['Positive Score'] = data['Comment Tokenize'].apply(
    lambda x: sum(positive_words.get(word, 0) for word in x.lower().split())
)

data['Negative Score'] = data['Comment Tokenize'].apply(
    lambda x: sum(negative_words.get(word, 0) for word in x.lower().split())
)

# Xác định đánh giá là Rất Tích cực, Tích cực, Bình thường, Tiêu cực, hoặc Rất Tiêu cực
def classify_sentiment(row):
    positive_score = row['Positive Score']
    negative_score = row['Negative Score']
    rating = row['Rating']
    # Kiểm tra điều kiện để phân loại cảm xúc
    if positive_score > negative_score:
        # Nếu điểm tích cực cao hơn điểm tiêu cực
        return 'Rất Tích cực' if rating >= 4 else 'Tích cực'
    elif negative_score > positive_score:
        # Nếu điểm tiêu cực cao hơn điểm tích cực
        return 'Rất Tiêu cực' if rating <= 2 else 'Tiêu cực'
    else:
        # Nếu điểm tích cực và tiêu cực bằng nhau
        return 'Bình thường'

# Tạo cột 'Label' dựa trên hàm classify_sentiment
data['Label'] = data.apply(classify_sentiment, axis=1)

# Lưu dữ liệu đã được xử lý vào file CSV
data.to_csv('data_cleaned/data_analysis.csv', index=False)

print("Phân tích cảm xúc hoàn tất và lưu vào file 'data_analysis.csv'")
