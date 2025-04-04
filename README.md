📰 Personalized News Recommendation System

This project is a Streamlit-based web app that recommends news articles based on user input using NLP techniques. It matches user queries with news titles using similarity scores and recommends the most relevant articles.

🔍 Project Overview
✅ Users enter a description or topic (e.g., "AI advancements in 2025").

✅ App recommends news articles with highest similarity to the input.

✅ Uses NLTK for preprocessing and cosine similarity for ranking.

🛠️ Tech Stack
Layer	                 Technology
Frontend + UI	         Streamlit
NLP Toolkit	           NLTK
Data Processing	       pandas, NumPy
Recommendation	      Cosine Similarity
Dataset Format	      CSV (title + link columns)


Install Dependencies
pip install -r requirements.txt

Run the App
streamlit run app.py
