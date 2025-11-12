import pandas as pd
import re
import os
import sys
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import numpy as np


# CẤU HÌNH

EXCEL_FILE_PATH = './du_lieu_tin_tuc.xlsx'   # Dữ liệu 1000 dòng
TEXT_COLUMN_NAME = 'content'
LABEL_COLUMN_NAME = 'label'
VIETNAMESE_STOPWORDS_PATH = './vietnamese-stopwords.txt'
MODEL_SAVE_PATH = './fake_news_rf_model.pkl'



#  HÀM XỬ LÝ DỮ LIỆU

def load_stopwords(file_path):
    """Đọc danh sách stopword tiếng Việt."""
    if not os.path.exists(file_path):
        print(f" Không tìm thấy file stopword tại '{file_path}'. Bỏ qua.")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f if line.strip()]
    print(f" Tải {len(stopwords)} stopword thành công.")
    return stopwords

def preprocess_text(text):
    """Tiền xử lý văn bản cơ bản."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)     # Xóa URL
    text = re.sub(r"\d+", "", text)         # Xóa số
    text = re.sub(r"[^\w\s]", " ", text)    # Xóa ký tự đặc biệt
    text = re.sub(r"\s+", " ", text).strip()
    return text

def vi_tokenizer(text):
    """Tách từ tiếng Việt."""
    tokenized = ViTokenizer.tokenize(text)
    return tokenized.split()

def load_data(file_path, text_col, label_col):
    """Đọc file dữ liệu và xử lý."""
    try:
        df = pd.read_excel(file_path)
        if text_col not in df.columns or label_col not in df.columns:
            print(f" Thiếu cột '{text_col}' hoặc '{label_col}'.")
            sys.exit(1)

        df[text_col] = df[text_col].apply(preprocess_text)
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
        df = df.dropna(subset=[text_col, label_col])
        df = df[df[text_col].str.len() > 0]
        df[label_col] = df[label_col].astype(int)

        print(f" Dữ liệu hợp lệ: {len(df)} mẫu")
        print(" Thống kê nhãn:")
        print(df[label_col].value_counts())
        return df[text_col], df[label_col]
    except Exception as e:
        print(f" Lỗi đọc dữ liệu: {e}")
        sys.exit(1)


#  HÀM CHÍNH

def main():
    print(" Bắt đầu huấn luyện mô hình Random Forest phân loại tin thật/giả...")
    stopwords = load_stopwords(VIETNAMESE_STOPWORDS_PATH)

    if not os.path.exists(EXCEL_FILE_PATH):
        print(f" Không tìm thấy file '{EXCEL_FILE_PATH}'")
        return

    X, y = load_data(EXCEL_FILE_PATH, TEXT_COLUMN_NAME, LABEL_COLUMN_NAME)

    # TF-IDF
    print(" Trích xuất đặc trưng TF-IDF...")
    tfidf = TfidfVectorizer(
        tokenizer=vi_tokenizer,
        stop_words=stopwords if stopwords else None,
        max_df=0.85,
        min_df=1,
        max_features=15000
    )

    X_tfidf = tfidf.fit_transform(X)
    print(f" Số lượng đặc trưng: {len(tfidf.get_feature_names_out())}")

    # Mô hình Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Cross Validation (5-fold)
    print(" Đang đánh giá mô hình (5-fold cross validation)...")
    scores = cross_val_score(rf_model, X_tfidf, y, cv=5, scoring='accuracy')
    print(f" Độ chính xác trung bình: {scores.mean()*100:.2f}%")

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # Huấn luyện mô hình
    print(" Huấn luyện Random Forest...")
    rf_model.fit(X_train, y_train)
    print("Huấn luyện hoàn tất!")

    # Đánh giá
    y_pred = rf_model.predict(X_test)
    print("\n Ma trận nhầm lẫn:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Báo cáo chi tiết:")
    print(classification_report(y_test, y_pred, target_names=['Thật (0)', 'Giả (1)']))

    print(f"Độ chính xác kiểm thử: {accuracy_score(y_test, y_pred)*100:.2f}%")

    # Lưu mô hình
    joblib.dump((rf_model, tfidf), MODEL_SAVE_PATH)
    print(f" Đã lưu mô hình vào: {MODEL_SAVE_PATH}")

    # Dự đoán thử
    print("\n--- DỰ ĐOÁN THỬ ---")
    test_articles = [
        "Nhân chứng kể lại phút đối mặt với tên cướp ngân hàng.",
        "Bức vẽ giúp bạn đánh giá mức độ stress của bản thân.",
        "Cô giáo ở Quảng Bình bắt trò tát bạn học 230 cái.",
        "Giá xăng dầu trong nước tăng nhẹ theo thị trường thế giới.",
        "Cảnh báo mã độc mới lây qua Facebook chỉ bằng cách nhấn Like."
    ]
    for text in test_articles:
        clean_text = preprocess_text(text)
        vec = tfidf.transform([clean_text])
        pred = rf_model.predict(vec)[0]
        prob = rf_model.predict_proba(vec)[0][pred]
        print(f"\n{text}")
        print(f"→ Dự đoán: {'Giả (1)' if pred == 1 else 'Thật (0)'} (Độ tin cậy: {prob*100:.2f}%)")

# ---
if __name__ == "__main__":
    main()
