import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
from utils.text_cleaning import clean_text

# Load mô hình và công cụ
model = joblib.load("models/recommend_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_review.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🤖 Demo 2: Recommend Classification")

# --- Nhập điểm đánh giá ---
st.subheader("📝 Input 1: Đánh giá các tiêu chí (0 - 5)")

salary = st.slider("💵 Salary & benefits", 0, 5, 3)
training = st.slider("📚 Training & learning", 0, 5, 3)
management = st.slider("👨‍💼 Management cares", 0, 5, 3)
culture = st.slider("🎉 Culture & fun", 0, 5, 3)
office = st.slider("🏢 Office & workspace", 0, 5, 3)

# --- Nhập nội dung review ---
st.subheader("🗣 Input 2: Nội dung review của bạn")

review_text = st.text_area("Nhập nội dung review (tiếng Việt hoặc tiếng Anh)", height=150)

if st.button("🎯 Dự đoán Recommend"):
    if not review_text.strip():
        st.warning("Vui lòng nhập nội dung review.")
    else:
        # Xử lý
        cleaned = clean_text(review_text)
        X_text = tfidf_vectorizer.transform([cleaned])
        X_num = scaler.transform([[salary, training, management, culture, office]])
        X_all = hstack([X_text, X_num])
        
        # Dự đoán
        result = model.predict(X_all)[0]
        prob = model.predict_proba(X_all)[0][1] * 100
        
        st.markdown("## ✅ Kết quả:")
        st.success("👍 Recommend!" if result == 1 else "👎 Not Recommend")
        st.info(f"🔢 Xác suất thành công: **{prob:.2f}%**")
