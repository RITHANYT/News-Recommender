import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded correctly
nltk_data_path = os.path.expanduser('~/nltk_data')  # Default path
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)

# Text Preprocessing Function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(str(text).lower())  # Handle NaN with str()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Load and Process Dataset
def load_news_data(filepath):
    data = pd.read_csv(filepath)
    if "text" in data.columns and "title" in data.columns:
        # Combine 'title' and 'text' for better recommendations
        data["processed_content"] = (data["title"] + " " + data["text"]).apply(preprocess_text)
    else:
        raise KeyError("Dataset must contain both 'text' and 'title' columns.")
    return data

# Content-Based Recommendation
def recommend_articles(user_input, news_data, top_n=5):
    # Combine user input with news data
    all_text = list(news_data["processed_content"]) + [preprocess_text(user_input)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)

    # Compute cosine similarity
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    news_data["similarity"] = similarity_scores
    recommendations = news_data.sort_values(by="similarity", ascending=False).head(top_n)
    return recommendations[["title", "link", "similarity"]]
