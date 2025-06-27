import streamlit as st
import pandas as pd
import joblib
from utils.text_cleaning import clean_text
import numpy as np
from scipy.sparse import hstack

# Load dữ liệu và model
df = pd.read_csv("data/processed_data.csv")
company_vectorizer = joblib.load("models/company_vectorizer.pkl")
recommend_model = joblib.load("models/recommend_model.pkl")
tfidf_review = joblib.load("models/tfidf_review.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🔍 ITviec Data Science App")

# Tab 1: Gợi ý công ty tương tự
st.header("📌 Gợi ý công ty tương tự")
company_name = st.text_input("Nhập tên công ty:")
if company_name:
    if company_name in df['company_name'].values:
        idx = df[df['company_name'] == company_name].index[0]
        tfidf_matrix = company_vectorizer.transform(df['description'])
        cosine_scores = (tfidf_matrix[idx] @ tfidf_matrix.T).toarray().flatten()
        top_indices = cosine_scores.argsort()[-6:-1][::-1]
        st.write("✅ Các công ty tương tự:")
        for i in top_indices:
            st.write(f"- {df.iloc[i]['company_name']}")
    else:
        st.error("Không tìm thấy công ty!")

# Tab 2: Dự đoán Recommend
st.header("🧠 Dự đoán Recommend / Not Recommend")
review = st.text_area("Nhập review:")
worklife = st.slider("Work-life balance", 1, 5, 3)
salary = st.slider("Salary", 1, 5, 3)
culture = st.slider("Culture", 1, 5, 3)

if st.button("Dự đoán"):
    cleaned = clean_text(review)
    X_text = tfidf_review.transform([cleaned])
    X_num = scaler.transform([[worklife, salary, culture]])
    X_final = hstack([X_text, X_num])
    pred = recommend_model.predict(X_final)
    st.success("👍 Recommend" if pred[0] == 1 else "👎 Not Recommend")