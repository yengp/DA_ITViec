import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import pickle

# Đường dẫn dữ liệu đã xử lý
REVIEW_PATH = 'output/Reviews_ongoing.xlsx'
MODEL_PATH = 'output/recommend_rf.pkl'

# Đọc dữ liệu
print('Đọc dữ liệu...')
df = pd.read_excel(REVIEW_PATH)
# Kiểm tra cột Review Content
df.columns = [col.strip() for col in df.columns]  # Chuẩn hóa tên cột

# Các cột số và cột văn bản
features = ['Salary & benefits', 'Training & learning', 'Management cares about me',
            'Culture & fun', 'Office & workspace']
text_col = "Review Content"
target_col = 'Recommend?'

# Tiền xử lý target
print('Tiền xử lý target...')
df['recommend'] = df[target_col].map({'Yes': 1, 'No': 0})
df = df.dropna(subset=['recommend', text_col])

# Vector hóa văn bản
print('Vector hóa văn bản...')
tfidf = TfidfVectorizer(max_features=2000)
X_text = tfidf.fit_transform(df[text_col])

# Chuẩn hóa dữ liệu số
print('Chuẩn hóa dữ liệu số...')
scaler = StandardScaler()
X_num = scaler.fit_transform(df[features])

# Ghép đặc trưng
X = hstack([X_text, X_num])
y = df['recommend']

# Huấn luyện mô hình
print('Huấn luyện mô hình Random Forest...')
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Lưu pipeline
print('Lưu mô hình, vectorizer, scaler...')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': tfidf, 'scaler': scaler, 'features': features}, f)
print('Đã lưu xong tại', MODEL_PATH)
