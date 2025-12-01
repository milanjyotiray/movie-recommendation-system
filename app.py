import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies_small = movies[["title", "overview", "genres"]].copy()
    movies_small["overview"] = movies_small["overview"].fillna("")
    movies_small["genres"] = movies_small["genres"].fillna("")
    movies_small["text"] = movies_small["overview"] + " " + movies_small["genres"]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_small["text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(
        movies_small.index, index=movies_small["title"].str.lower()
    ).drop_duplicates()

    return movies_small, cosine_sim, indices

def get_recommendations(title, movies_small, cosine_sim, indices, n_recommendations=5):
    title = title.lower()
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : n_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    return movies_small["title"].iloc[movie_indices].tolist()

def main():
    st.title("🎬 Movie Recommendation System")
    st.write("Content-based movie recommender using TMDB 5000 movies dataset.")

    movies_small, cosine_sim, indices = load_data()

    movie_title = st.text_input("Enter a movie title (e.g., Avatar)")
    n_recs = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Recommend"):
        if not movie_title:
            st.warning("Please enter a movie title.")
        else:
            recs = get_recommendations(
                movie_title, movies_small, cosine_sim, indices, n_recs
            )
            if not recs:
                st.error("Movie not found in database. Try another title.")
            else:
                st.success(f"Recommendations for **{movie_title}**:")
                for i, r in enumerate(recs, start=1):
                    st.write(f"{i}. {r}")

if __name__ == "__main__":
    main()
