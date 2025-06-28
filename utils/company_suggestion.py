import pandas as pd
import numpy as np
import string
import re
from spellchecker import SpellChecker
import demoji
from deep_translator import GoogleTranslator
import underthesea
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.similarities import Similarity
from gensim.matutils import corpus2csc
from sklearn.cluster import KMeans

# --- TEXT CLEANING ---
teencode_map = {
    "lol": "laughing out loud", "brb": "be right back", "omg": "oh my god", "g2g": "got to go", "u": "you", "r": "are", "ppl": "people", "w8": "wait", "gr8": "great", "l8r": "later", "b4": "before", "thx": "thanks", "np": "no problem", "pls": "please", "btw": "by the way", "imo": "in my opinion", "imho": "in my humble opinion", "afk": "away from keyboard", "asap": "as soon as possible", "fyi": "for your information", "idk": "i don't know", "irl": "in real life", "jk": "just kidding", "lmao": "laughing my ass off", "rofl": "rolling on floor laughing", "bff": "best friends forever", "cuz": "because", "msg": "message", "tmi": "too much information", "yolo": "you only live once", "aka": "also known as", "tl;dr": "too long didn't read", "rn": "right now", "nvm": "nevermind", "wbu": "what about you", "hbu": "how about you", "ttyl": "talk to you later", "fomo": "fear of missing out", "smh": "shaking my head", "ily": "i love you", "ily sm": "i love you so much", "bby": "baby", "xoxo": "hugs and kisses", "cu": "see you", "gg": "good game", "ez": "easy", "fr": "for real", "atm": "at the moment", "tbh": "to be honest", "ikr": "i know right", "gtg": "got to go", "wth": "what the hell", "wtf": "what the fuck", "omw": "on my way", "sry": "sorry", "ty": "thank you", "ur": "your"
}
spell = SpellChecker(language='en')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def normalize_text(text):
    if pd.isna(text) or text is None or str(text).strip() == '':
        return ''
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", ' ', text)
    text = re.sub(r'\S*@\S*\s?', ' ', text)
    text = demoji.replace(text, ' ')
    words = text.split()
    cleaned_words = [teencode_map.get(word, word) for word in words]
    text = ' '.join(cleaned_words)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in text.split()]
    text = ' '.join(corrected_words)
    return text

def translate_text(text, dest_lang='en', src_lang='vi'):
    if pd.isna(text) or text is None or str(text).strip() == '':
        return ''
    text_to_translate = str(text).strip()
    if not text_to_translate:
        return ''
    try:
        translated = GoogleTranslator(source=src_lang, target=dest_lang).translate(text=text_to_translate)
        return translated
    except Exception:
        return text_to_translate

def tokenize_and_lemmatize(text):
    tokens = underthesea.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def preprocess_company_text(df, col):
    df[col] = df[col].fillna('').apply(normalize_text).apply(translate_text).apply(lambda x: ' '.join(tokenize_and_lemmatize(x)))
    return df

def build_gensim_tfidf(df, col):
    texts = df[col].apply(lambda x: x.split()).tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return dictionary, corpus, tfidf, corpus_tfidf

def build_similarity_index(corpus_tfidf, dictionary):
    return Similarity(output_prefix="sim_index", corpus=corpus_tfidf, num_features=len(dictionary))

def recommend_companies_gensim_cosine(user_text, df, col, dictionary, tfidf, corpus_tfidf, top_k=5):
    user_tokens = user_text.split()
    user_bow = dictionary.doc2bow(user_tokens)
    user_tfidf = tfidf[user_bow]
    tfidf_sparse = corpus2csc(corpus_tfidf).transpose()
    user_vec = corpus2csc([user_tfidf], num_terms=len(dictionary)).transpose()
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(user_vec, tfidf_sparse)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    result = df.iloc[top_indices][['id', 'Company Name']].copy()
    result['Similarity'] = np.round(sims[top_indices], 3)
    return result.reset_index(drop=True)

def cluster_companies_kmeans(corpus_tfidf, n_clusters=5):
    tfidf_sparse = corpus2csc(corpus_tfidf).transpose()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(tfidf_sparse)
    return labels, kmeans
