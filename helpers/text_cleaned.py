import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

def simple_text_clean(dataframe):
    stop_words = set(stopwords.words("english"))

    # Loại bỏ các liên kết HTTP
    dataframe["Content"] = dataframe["Content"].replace(
        r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
        "",
        regex=True,
    )

    # Loại bỏ ký tự xuống dòng
    dataframe["Content"] = dataframe["Content"].replace(r"[\r\n]+", " ", regex=True)

    # Loại bỏ số, chỉ giữ lại chữ cái
    dataframe["Content"] = dataframe["Content"].replace(r"[\w]*\d+[\w]*", "", regex=True)

    # Loại bỏ dấu câu
    dataframe["Content"] = dataframe["Content"].replace(r"[^\w\s]", " ", regex=True)
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for char in punctuation:
        dataframe["Content"] = dataframe["Content"].replace(char, " ")

    # Loại bỏ nhiều khoảng trắng thành một khoảng trắng
    dataframe["Content"] = dataframe["Content"].replace(r"[\s]{2,}", " ", regex=True)

    # Loại bỏ các dòng bắt đầu bằng khoảng trắng
    dataframe["Content"] = dataframe["Content"].replace(r"^[\s]{1,}", "", regex=True)

    # Loại bỏ các dòng kết thúc bằng khoảng trắng
    dataframe["Content"] = dataframe["Content"].replace(r"[\s]{1,}$", "", regex=True)

    # Chuyển đổi thành chữ thường
    dataframe["Content"] = dataframe["Content"].str.lower()

    # Loại bỏ các dòng rỗng
    dataframe = dataframe[dataframe["Content"].str.len() > 0]

    # Loại bỏ stop words
    def remove_stopwords(text):
        text_split = text.split()
        text = [word for word in text_split if word not in stop_words]
        return " ".join(text)

    dataframe["Content"] = dataframe["Content"].apply(remove_stopwords)

    # Sử dụng WordNet Lemmatizer thay vì Stemming để có kết quả tốt hơn
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_text(text):
        lemmatized = []
        post_tag_list = pos_tag(word_tokenize(text))
        for word, post_tag_val in post_tag_list:
            lemmatized.append(lemmatizer.lemmatize(word, get_wordnet_pos(post_tag_val)))
        text = " ".join(x for x in lemmatized)
        return text

    dataframe["Content"] = dataframe["Content"].apply(lemmatize_text)

    return dataframe