import streamlit as st
import pickle
import pandas as pd
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Fetch Movie Poster
# -------------------------------
def fetch_poster(movie_id):
    response = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    )
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


# -------------------------------
# Load Data
# -------------------------------
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags'])   # 🚀 NO .toarray()


# -------------------------------
# Recommend Function (Memory Safe)
# -------------------------------
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]

    similarity_scores = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    movies_list = sorted(
        list(enumerate(similarity_scores)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


# -------------------------------
# UI
# -------------------------------
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    for col, name, poster in zip(
        [col1, col2, col3, col4, col5],
        names,
        posters
    ):
        with col:
            st.text(name)
            st.image(poster)
