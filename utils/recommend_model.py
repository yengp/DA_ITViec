import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import pickle

class RecommendModel:
    def __init__(self):
        self.vectorizer = None
        self.scaler = None
        self.model = None
        self.features = ['Salary & benefits', 'Training & learning', 'Management cares about me',
                        'Culture & fun', 'Office & workspace']

    def fit(self, df):
        X_text = self._fit_vectorizer(df['Review Content'])
        X_num = self._fit_scaler(df[self.features])
        X = hstack([X_text, X_num])
        y = df['recommend']
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X, y)

    def predict(self, review_content, numeric_features):
        X_text = self.vectorizer.transform([review_content])
        X_num = self.scaler.transform([numeric_features])
        X = hstack([X_text, X_num])
        return self.model.predict(X)[0], self.model.predict_proba(X)[0,1]

    def _fit_vectorizer(self, texts):
        self.vectorizer = TfidfVectorizer(max_features=2000)
        return self.vectorizer.fit_transform(texts)

    def _fit_scaler(self, X):
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
