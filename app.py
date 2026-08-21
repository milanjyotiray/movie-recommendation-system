import streamlit as st
import pandas as pd
import numpy as np
import json
import ast
import random
import urllib.request
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Custom CSS Styling (Full-Width High Contrast Dark Theme)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Background & Typography */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Sidebar Completely */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Force readable text colors across standard HTML elements */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #f8fafc;
    }
    
    /* Header & Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.25) 0%, rgba(139, 92, 246, 0.25) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 2rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.4rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1 !important;
        max-width: 650px;
        margin: 0 auto 1.2rem auto;
    }
    
    .badge-pill {
        display: inline-block;
        background: rgba(229, 9, 20, 0.3);
        color: #ff6b76 !important;
        border: 1px solid rgba(229, 9, 20, 0.5);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    
    /* Movie Spotlight Card */
    .spotlight-card {
        background: #151c2c;
        border: 1px solid #2d3748;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .spotlight-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 0.4rem;
    }
    
    .spotlight-tagline {
        font-style: italic;
        color: #cbd5e1 !important;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }
    
    /* Recommendation Grid Card */
    .rec-card {
        background: #151c2c;
        border: 1px solid #2d3748;
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .rec-card:hover {
        transform: translateY(-4px);
        border-color: #8b5cf6;
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .match-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 4px 10px;
        border-radius: 12px;
    }
    
    .rank-badge {
        background: rgba(255, 255, 255, 0.15);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 4px 8px;
        border-radius: 8px;
    }
    
    .movie-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 0.4rem;
        line-height: 1.3;
    }
    
    .movie-meta {
        font-size: 0.88rem;
        color: #cbd5e1 !important;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .genre-pill {
        display: inline-block;
        background: #232d42;
        color: #e2e8f0 !important;
        font-size: 0.78rem;
        padding: 3px 9px;
        border-radius: 6px;
        margin-right: 4px;
        margin-bottom: 4px;
        border: 1px solid #334155;
    }
    
    .overview-text {
        font-size: 0.9rem;
        color: #e2e8f0 !important;
        line-height: 1.45;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    /* Poster Box */
    .poster-box {
        width: 100%;
        height: 280px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
        font-size: 3.5rem;
        margin-bottom: 0.75rem;
        border: 1px solid #334155;
    }

    /* Streamlit Input & Dropdown Styling */
    div[data-baseweb="select"] > div {
        background-color: #151c2c !important;
        color: #ffffff !important;
        border-color: #334155 !important;
    }
    
    div[data-baseweb="popover"] div, ul[role="listbox"] {
        background-color: #151c2c !important;
        color: #ffffff !important;
    }

    ul[role="listbox"] li span {
        color: #ffffff !important;
    }

    /* Custom Main Footer */
    .custom-footer {
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid #1e293b;
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helper Functions & Poster API
# ---------------------------------------------------------
def parse_genres_str(genres_raw):
    """Parses raw JSON genres string into a list of genre names."""
    if not isinstance(genres_raw, str) or not genres_raw.strip():
        return []
    try:
        genres_list = ast.literal_eval(genres_raw)
        return [g["name"] for g in genres_list if "name" in g]
    except Exception:
        return []

def get_release_year(date_str):
    """Extracts year from YYYY-MM-DD date string."""
    if isinstance(date_str, str) and len(date_str) >= 4:
        return date_str[:4]
    return "N/A"

@st.cache_data(show_spinner=False)
def fetch_poster_url(movie_title):
    """Fetches movie poster URL automatically via OMDb API."""
    try:
        encoded_title = urllib.parse.quote(movie_title)
        url = f"http://www.omdbapi.com/?t={encoded_title}&apikey=trilogy"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode())
            poster = data.get("Poster")
            if poster and poster != "N/A" and poster.startswith("http"):
                return poster
    except Exception:
        pass
    return None

# ---------------------------------------------------------
# Data Loading & Recommendation Engine
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """Loads TMDB dataset and builds TF-IDF Cosine Similarity matrix."""
    movies = pd.read_csv("tmdb_5000_movies.csv")
    
    # Fill missing values
    movies["overview"] = movies["overview"].fillna("")
    movies["genres_raw"] = movies["genres"].fillna("[]")
    movies["tagline"] = movies["tagline"].fillna("")
    movies["vote_average"] = movies["vote_average"].fillna(0.0)
    movies["release_date"] = movies["release_date"].fillna("")
    
    # Process genres text for TF-IDF
    movies["genres_parsed"] = movies["genres_raw"].apply(parse_genres_str)
    movies["genres_text"] = movies["genres_parsed"].apply(lambda x: " ".join(x))
    
    # Combined textual features
    movies["text"] = movies["overview"] + " " + movies["genres_text"]

    # Compute TF-IDF matrix & Cosine Similarity
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Index map for quick title lookup
    indices = pd.Series(
        movies.index, index=movies["title"].str.lower()
    ).drop_duplicates()

    return movies, cosine_sim, indices

def get_recommendations(selected_title, movies, cosine_sim, indices, n_recommendations=5):
    """Returns top N recommended movies with normalized similarity scores and metadata."""
    title_clean = selected_title.strip().lower()
    if title_clean not in indices:
        return []

    idx = indices[title_clean]
    # Handle duplicate titles by picking the first match index
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip index 0 (self-match) and take top N recommendations
    sim_scores = sim_scores[1 : n_recommendations + 1]
    
    if not sim_scores:
        return []

    top_sim_score = sim_scores[0][1]
    
    recommendations = []
    for i, score in sim_scores:
        row = movies.iloc[i]
        
        # Relative match percentage scaling for intuitive user display (top match = ~95%)
        if top_sim_score > 0:
            match_percentage = min(99.0, round((score / top_sim_score) * 95.0, 1))
        else:
            match_percentage = round(score * 100, 1)

        recommendations.append({
            "id": row.get("id", None),
            "title": row["title"],
            "overview": row["overview"],
            "tagline": row["tagline"],
            "genres": row["genres_parsed"],
            "rating": row["vote_average"],
            "year": get_release_year(row["release_date"]),
            "similarity_score": match_percentage,
            "raw_cosine": round(score, 4),
        })

    return recommendations

# ---------------------------------------------------------
# Application Main Function
# ---------------------------------------------------------
def main():
    movies, cosine_sim, indices = load_data()
    all_titles = sorted(movies["title"].tolist())

    # -----------------------------------------------------
    # Main Hero Banner
    # -----------------------------------------------------
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🎬 CineMatch Movie Recommender</div>
            <div class="hero-subtitle">
                Discover personalized movie recommendations powered by Natural Language Processing (NLP) & Content-Based Vector Similarity.
            </div>
            <div>
                <span class="badge-pill">✨ TF-IDF Vectorization</span>
                <span class="badge-pill">🎯 Cosine Similarity</span>
                <span class="badge-pill">🍿 TMDB 5000 Movies</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # Top Control Bar (Search, Count Slider, Surprise Button)
    # -----------------------------------------------------
    col_select, col_num, col_btn = st.columns([5, 3, 2])
    
    # Session state for random choice
    if "selected_movie" not in st.session_state:
        st.session_state.selected_movie = "Avatar"

    with col_select:
        default_idx = all_titles.index(st.session_state.selected_movie) if st.session_state.selected_movie in all_titles else 0
        selected_movie = st.selectbox(
            "🔍 **Search or Choose a Movie Title:**",
            options=all_titles,
            index=default_idx,
            help="Type any movie title from the TMDB 5000 dataset to get instant recommendations."
        )

    with col_num:
        n_recs = st.slider("🍿 **Number of Recommendations:**", min_value=1, max_value=10, value=6)

    with col_btn:
        st.write("") # Alignment spacing
        st.write("") 
        if st.button("🎲 Surprise Me!", use_container_width=True):
            st.session_state.selected_movie = random.choice(all_titles)

    st.markdown("---")

    # -----------------------------------------------------
    # Selected Movie Spotlight Section
    # -----------------------------------------------------
    if selected_movie:
        selected_idx = indices.get(selected_movie.lower())
        if isinstance(selected_idx, pd.Series):
            selected_idx = selected_idx.iloc[0]
            
        if selected_idx is not None:
            sel_row = movies.iloc[selected_idx]
            genres_tags = "".join([f'<span class="genre-pill">{g}</span>' for g in sel_row["genres_parsed"]])
            year = get_release_year(sel_row["release_date"])
            rating = sel_row["vote_average"]
            tagline = sel_row["tagline"]
            overview = sel_row["overview"]

            # Spotlight poster
            spotlight_poster = fetch_poster_url(sel_row['title'])

            col_spot_img, col_spot_info = st.columns([1, 4]) if spotlight_poster else (None, None)

            if spotlight_poster:
                with col_spot_img:
                    st.markdown(
                        f'<img src="{spotlight_poster}" style="width:100%; border-radius:12px; object-fit:cover; max-height:260px; border:1px solid #334155;">',
                        unsafe_allow_html=True
                    )
                with col_spot_info:
                    st.markdown(
                        f"""
                        <div class="spotlight-card" style="margin-bottom:0px;">
                            <div style="color: #a78bfa; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 0.4rem;">
                                Selected Movie
                            </div>
                            <div class="spotlight-title">
                                {sel_row['title']} <span style="font-size: 1rem; color: #cbd5e1; font-weight: 400;">({year})</span>
                            </div>
                            {f'<div class="spotlight-tagline">"{tagline}"</div>' if tagline else ''}
                            <div style="margin-bottom: 0.75rem;">
                                <span style="color: #fbbf24; font-weight: 700; font-size: 0.95rem; margin-right: 12px;">⭐ {rating}/10</span>
                                {genres_tags}
                            </div>
                            <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">
                                {overview if overview else 'No overview available.'}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"""
                    <div class="spotlight-card">
                        <div style="color: #a78bfa; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 0.4rem;">
                            Selected Movie
                        </div>
                        <div class="spotlight-title">
                            {sel_row['title']} <span style="font-size: 1rem; color: #cbd5e1; font-weight: 400;">({year})</span>
                        </div>
                        {f'<div class="spotlight-tagline">"{tagline}"</div>' if tagline else ''}
                        <div style="margin-bottom: 0.75rem;">
                            <span style="color: #fbbf24; font-weight: 700; font-size: 0.95rem; margin-right: 12px;">⭐ {rating}/10</span>
                            {genres_tags}
                        </div>
                        <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">
                            {overview if overview else 'No overview available.'}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------
    # Recommendations Output Grid
    # -----------------------------------------------------
    st.markdown(f"### 🍿 **Top {n_recs} Recommended Movies for You**")

    recommendations = get_recommendations(
        selected_movie, movies, cosine_sim, indices, n_recommendations=n_recs
    )

    if not recommendations:
        st.warning("⚠️ Movie not found in database. Please choose a title from the dropdown.")
    else:
        # Display recommendations in responsive columns (up to 3 per row)
        cols_per_row = 3 if n_recs >= 3 else n_recs
        num_rows = (len(recommendations) + cols_per_row - 1) // cols_per_row

        for r_idx in range(num_rows):
            cols = st.columns(cols_per_row)
            for c_idx in range(cols_per_row):
                item_idx = r_idx * cols_per_row + c_idx
                if item_idx < len(recommendations):
                    rec = recommendations[item_idx]
                    rank = item_idx + 1
                    
                    with cols[c_idx]:
                        # Automatic poster image resolution
                        poster_url = fetch_poster_url(rec["title"])
                        
                        genres_html = "".join([f'<span class="genre-pill">{g}</span>' for g in rec["genres"][:3]])
                        
                        poster_html = (
                            f'<img src="{poster_url}" style="width:100%; height:280px; object-fit:cover; border-radius:10px; margin-bottom:0.75rem; border:1px solid #334155;">'
                            if poster_url
                            else f'<div class="poster-box">🍿</div>'
                        )
                        
                        st.markdown(
                            f"""
                            <div class="rec-card">
                                <div>
                                    {poster_html}
                                    <div class="card-header">
                                        <span class="rank-badge">#{rank}</span>
                                        <span class="match-badge">🎯 {rec['similarity_score']}% Match</span>
                                    </div>
                                    <div class="movie-title">{rec['title']}</div>
                                    <div class="movie-meta">
                                        <span>📅 {rec['year']}</span>
                                        <span>⭐ {rec['rating']}/10</span>
                                    </div>
                                    <div style="margin-bottom: 0.5rem;">
                                        {genres_html}
                                    </div>
                                    <div class="overview-text">{rec['overview']}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # -----------------------------------------------------
    # Clean Footer
    # -----------------------------------------------------
    st.markdown(
        """
        <div class="custom-footer">
            Built with ❤️ using <strong>Streamlit</strong>, <strong>Scikit-Learn</strong> & <strong>TMDB 5000 Dataset</strong>
            <br>
            <a href="https://github.com/milanjyotiray/movie-recommendation-system" target="_blank" style="color: #8b5cf6; text-decoration: none; font-weight: 600;">💻 View on GitHub</a> | 
            <a href="https://movie-recommendation-system-hejeg8p3u4tyfp9rvnst77.streamlit.app" target="_blank" style="color: #8b5cf6; text-decoration: none; font-weight: 600;">🌐 Live Streamlit App</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
