import streamlit as st
import nltk
import os
from models.recommendation import load_news_data, recommend_articles

# Ensure NLTK resources are downloaded
nltk_data_path = os.path.expanduser('~/nltk_data')  # Default path for NLTK data
nltk.data.path.append(nltk_data_path)

# Check if the punkt tokenizer is available, otherwise download it
try:
    nltk.data.find('tokenizers/punkt')
    st.success("Punkt tokenizer is available!")
except LookupError:
    st.error("Punkt tokenizer not found. Downloading...")
    nltk.download('punkt', download_dir=nltk_data_path)
    st.success("Punkt tokenizer downloaded successfully!")

# Try to load the dataset
try:
    news_data = load_news_data("data/news.csv")
    st.success("News dataset loaded successfully!")
except KeyError as e:
    st.error(f"Error loading dataset: Missing column in dataset. ({str(e)})")
    news_data = None
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    news_data = None

# App Header
st.title("📖 Personalized News Recommendation System")
st.write("Enter a topic or description, and get personalized news articles just for you!")

# User Input
user_input = st.text_area("Describe what you're interested in (e.g., 'AI advancements in 2025')")

# Recommendation Button
if st.button("Recommend"):
    if news_data is not None and user_input.strip():
        recommendations = recommend_articles(user_input, news_data)
        st.write("### Recommended Articles:")
        for _, row in recommendations.iterrows():
            st.write(f"**{row['title']}** ([Read More]({row['link']})) - Similarity: {row['similarity']:.2f}")
    elif news_data is None:
        st.error("Dataset not loaded. Please check the error above.")
    else:
        st.warning("Please enter a topic or description.")
