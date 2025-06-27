from pyvi import ViTokenizer, ViPosTagger
from nltk.tokenize import sent_tokenize
import regex
import re

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')
## LOAD EMOJICON
file = open('stopwords/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()

## LOAD TEENCODE
file = open('stopwords/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()

## LOAD TRANSLATE ENGLISH -> VIETNAMESE
file = open('stopwords/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()

## LOAD WRONG WORDS
file = open('stopwords/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()

## LOAD STOPWORDS
file = open('stopwords/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

def process_text(text, emoji_dict, teen_dict, wrong_lst):
    """
    Xử lý văn bản bao gồm chuyển đổi biểu tượng cảm xúc, chuyển đổi teen code, loại bỏ dấu câu và số,
    loại bỏ các từ sai, và chuẩn hóa các câu thành dạng chuẩn.

    Parameters:
    - text (str): Văn bản cần xử lý.
    - emoji_dict (dict): Từ điển biểu tượng cảm xúc.
    - teen_dict (dict): Từ điển teen code.
    - wrong_lst (list): Danh sách từ sai cần loại bỏ.

    Returns:
    - str: Văn bản đã được xử lý.
    """
    text = text.lower()
    text = text.replace("’", '')
    text = regex.sub(r'\.+', ".", text)
    new_sentence = ''
    for sentence in sent_tokenize(text):
        # CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word] + ' ' if word in emoji_dict else word for word in list(sentence))
        # CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        # DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern, sentence))
        # DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence + sentence + '. '
    text = new_sentence
    # DEL excess blank space
    text = regex.sub(r'\s+', ' ', text).strip()
    return text

def removeStopWords(text, stop_words):
    """
    Loại bỏ các stopword từ văn bản.

    Parameters:
    - text (str): Văn bản cần loại bỏ stopword.
    - stop_words (list): Danh sách các stopword.

    Returns:
    - str: Văn bản đã loại bỏ stopword.
    """
    text = text.split()
    result = []
    for word in text:
        if word not in stop_words:
            result.append(word)
    return " ".join(result)

def wordtokenize(text):
    """
    Tokenize văn bản sử dụng ViTokenizer của pyvi.

    Parameters:
    - text (str): Văn bản cần tokenize.

    Returns:
    - str: Chuỗi đã được tokenize.
    """
    return ViTokenizer.tokenize(text)

def process_postag_pyvi(text):
    """
    Xử lý phân loại từng từ trong văn bản thành các nhóm từ loại (POS tagging).

    Parameters:
    - text (str): Văn bản cần xử lý.

    Returns:
    - str: Văn bản đã được xử lý theo POS tagging.
    """
    new_document = ""
    for sentence in sent_tokenize(text):
        sentence = sentence.replace(".", "")
        ## POS tag
        lst_word_type = ["N", "Np", "A", "AB", "V", "VB", "R", "M"]
        tagged_sentence = ViPosTagger.postagging(ViTokenizer.tokenize(sentence))
        sentence = " ".join(
            word if tag in lst_word_type else ""
            for word, tag in zip(tagged_sentence[0], tagged_sentence[1])
        )
        new_document = new_document + sentence + " "
    new_document = regex.sub(r"\s+", " ", new_document).strip()
    return new_document

def removeSpecialChar(text):
    """
    Loại bỏ các ký tự đặc biệt khỏi văn bản.

    Parameters:
    - text (str): Văn bản cần loại bỏ ký tự đặc biệt.

    Returns:
    - str: Văn bản đã loại bỏ ký tự đặc biệt.
    """
    return regex.sub(r"[^\w\s]", "", text)

def normalize_repeated_characters(text):
    """
    Chuẩn hóa các từ có ký tự lặp lại.

    Parameters:
    - text (str): Văn bản cần chuẩn hóa.

    Returns:
    - str: Văn bản đã được chuẩn hóa.
    """
    return re.sub(r'(.)\1+', r'\1', text)

def stepByStep(text):
    """
    Thực hiện từng bước xử lý văn bản bao gồm POS tagging, loại bỏ ký tự đặc biệt,
    loại bỏ stopword và chuẩn hóa từ có ký tự lặp.

    Parameters:
    - text (str): Văn bản cần xử lý.

    Returns:
    - str: Văn bản đã qua các bước xử lý.
    """
    with open("stopwords/vietnamese-stopwords.txt", "r", encoding="utf-8") as file:
        stop_words = file.read()
    stop_words = stop_words.split("\n")

    text = str(text)
    text = process_postag_pyvi(text.lower())
    text = removeSpecialChar(text)
    text = removeStopWords(text, stop_words)
    text = normalize_repeated_characters(text)
    return text

# Ví dụ sử dụng hàm stepByStep
text = "Áo Ba Lỗ"
text_processed = stepByStep(text)
print(text_processed)
