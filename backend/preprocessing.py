"""
preprocessing.py — Data Cleaning & Feature Engineering Pipeline
Smart Content Recommender

This module handles:
1. Loading and cleaning the raw Netflix dataset
2. Feature engineering (metadata soup creation)
3. Sentiment analysis using VADER
4. Building and serializing ML models (TF-IDF, cosine similarity, K-Means)
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'netflix_titles.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DATA_PATH = os.path.join(MODELS_DIR, 'processed_data.pkl')
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, 'tfidf_matrix.pkl')
CAST_MATRIX_PATH = os.path.join(MODELS_DIR, 'cast_matrix.pkl')   # sparse cast matrix
KMEANS_PATH = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
PCA_DATA_PATH = os.path.join(MODELS_DIR, 'pca_data.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')


def load_data():
    """Load the raw Netflix dataset."""
    print("[1/8] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df)} titles ({df['type'].value_counts().to_dict()})")
    return df


def clean_data(df):
    """Clean and preprocess the raw dataframe."""
    print("[2/8] Cleaning data...")

    # Fill missing values
    fill_cols = ['director', 'cast', 'country', 'date_added', 'rating', 'duration', 'listed_in', 'description']
    for col in fill_cols:
        df[col] = df[col].fillna('')

    # Clean whitespace
    str_cols = ['title', 'director', 'cast', 'country', 'listed_in', 'description']
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Parse date_added
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

    # Extract duration number
    df['duration_num'] = df['duration'].apply(_extract_duration_number)

    # Parse genres into a list
    df['genres_list'] = df['listed_in'].apply(
        lambda x: [g.strip() for g in x.split(',') if g.strip()] if x else []
    )

    # Parse cast into a list
    df['cast_list'] = df['cast'].apply(
        lambda x: [c.strip() for c in x.split(',') if c.strip()] if x else []
    )

    # Parse countries into a list
    df['country_list'] = df['country'].apply(
        lambda x: [c.strip() for c in x.split(',') if c.strip()] if x else []
    )

    # Primary country (first listed)
    df['primary_country'] = df['country_list'].apply(lambda x: x[0] if x else 'Unknown')

    print(f"   Cleaned {len(df)} titles")
    return df


def _extract_duration_number(duration_str):
    """Extract numeric duration from strings like '90 min' or '2 Seasons'."""
    if not duration_str:
        return 0
    match = re.search(r'(\d+)', str(duration_str))
    return int(match.group(1)) if match else 0


def add_sentiment(df):
    """Add sentiment analysis scores using VADER."""
    print("[3/8] Running sentiment analysis (VADER)...")
    analyzer = SentimentIntensityAnalyzer()

    sentiments = df['description'].apply(
        lambda x: analyzer.polarity_scores(x) if x else {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
    )

    df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
    df['sentiment_pos'] = sentiments.apply(lambda x: x['pos'])
    df['sentiment_neg'] = sentiments.apply(lambda x: x['neg'])

    # Classify mood based on compound score
    def classify_mood(compound):
        if compound >= 0.3:
            return 'feel-good'
        elif compound <= -0.3:
            return 'dark'
        elif compound <= -0.05:
            return 'intense'
        else:
            return 'thought-provoking'

    df['mood'] = df['sentiment_compound'].apply(classify_mood)

    mood_dist = df['mood'].value_counts().to_dict()
    print(f"   Mood distribution: {mood_dist}")
    return df


def create_metadata_soup(df):
    """
    Create a combined text feature ('soup') for each title by merging
    weighted metadata fields. This soup is used for TF-IDF vectorization.

    Weighting strategy:
    - Genres: 3x (most important for content similarity)
    - Director: 2x (strong indicator of style)
    - Cast: Top 5 actors, 2x (user's suggestion — cast similarity matters)
    - Country: 1x
    - Description: 1x (natural language content)
    - Rating: 1x
    """
    print("[4/8] Creating metadata soup...")

    def _build_soup(row):
        # Clean names: remove spaces so "Johnny Depp" → "johnnydepp"
        # This prevents false matches between actors sharing first/last names
        director = row['director'].lower().replace(' ', '').replace(',', ' ') if row['director'] else ''
        
        # Take top 5 cast members (user requested cast emphasis)
        cast_members = row['cast_list'][:5] if row['cast_list'] else []
        cast = ' '.join([c.lower().replace(' ', '') for c in cast_members])
        
        # Genres
        genres = ' '.join([g.lower().replace(' ', '').replace('&', 'and') for g in row['genres_list']])
        
        # Country
        country = row['primary_country'].lower().replace(' ', '') if row['primary_country'] != 'Unknown' else ''
        
        # Description — keep as natural text
        description = row['description'].lower() if row['description'] else ''
        description = re.sub(r'[^a-z0-9\s]', '', description)
        
        # Rating
        rating = row['rating'].lower().replace('-', '') if row['rating'] else ''

        # Build weighted soup
        # Genres 3x, Director 2x, Cast 2x, rest 1x
        soup_parts = [
            genres, genres, genres,           # 3x weight
            director, director,               # 2x weight
            cast, cast,                       # 2x weight (cast emphasis per user request)
            country,                          # 1x weight
            rating,                           # 1x weight
            description                       # 1x weight
        ]

        return ' '.join(part for part in soup_parts if part)

    df['soup'] = df.apply(_build_soup, axis=1)
    print(f"   Created soup for {len(df)} titles")
    return df


def build_tfidf_model(df):
    """Build TF-IDF vectorizer and sparse TF-IDF matrix.

    Note: We deliberately do NOT compute the full NxN cosine similarity matrix
    here. At ~8000 titles that matrix would be 8000x8000 float64 = ~500 MB,
    causing MemoryError when loaded at server startup. Instead, cosine similarity
    is computed on-the-fly per request in recommender.py.
    """
    print("[5/7] Building TF-IDF model...")

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=20000,
        ngram_range=(1, 2),    # Unigrams + bigrams for better phrase matching
        min_df=2,              # Ignore very rare terms
        max_df=0.95            # Ignore overly common terms
    )

    tfidf_matrix = tfidf.fit_transform(df['soup'])
    print(f"   TF-IDF sparse matrix shape: {tfidf_matrix.shape}")

    return tfidf, tfidf_matrix


def build_cast_similarity(df):
    """
    Build a sparse cast-based feature matrix using CountVectorizer.
    This gives recommendations based primarily on shared cast members.

    Note: We save the sparse matrix (not the full NxN similarity matrix)
    so that cast similarity can be computed on-the-fly per request.
    """
    print("[6/7] Building cast feature matrix...")

    # Create cast text: lowercase, no spaces in names
    cast_text = df['cast_list'].apply(
        lambda actors: ' '.join([a.lower().replace(' ', '') for a in actors]) if actors else ''
    )

    count_vec = CountVectorizer(
        stop_words='english',
        max_features=10000
    )

    cast_matrix = count_vec.fit_transform(cast_text)
    print(f"   Cast sparse matrix shape: {cast_matrix.shape}")

    return cast_matrix


def build_clusters(tfidf_matrix, df, n_clusters=15):
    """
    Cluster content into thematic groups using K-Means.
    Reduce to 2D with TruncatedSVD for visualization.

    Note: We use TruncatedSVD (not PCA) because PCA calls .toarray() internally,
    which densifies the sparse matrix and causes a MemoryError (1.3 GB allocation).
    TruncatedSVD operates on sparse matrices natively.
    """
    print(f"[7/7] Building K-Means clusters (k={n_clusters})...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    clusters = kmeans.fit_predict(tfidf_matrix)
    df['cluster'] = clusters

    # TruncatedSVD for 2D visualization (sparse-compatible, no densification)
    print("   Reducing to 2D with TruncatedSVD...")
    svd = TruncatedSVD(n_components=2, random_state=42)
    svd_result = svd.fit_transform(tfidf_matrix)

    pca_data = {
        'x': svd_result[:, 0].tolist(),
        'y': svd_result[:, 1].tolist(),
        'cluster': clusters.tolist(),
        'title': df['title'].tolist(),
        'type': df['type'].tolist(),
        'show_id': df['show_id'].tolist()
    }

    # Create cluster labels based on most common genres
    cluster_labels = {}
    for c in range(n_clusters):
        cluster_df = df[df['cluster'] == c]
        all_genres = [g for genres in cluster_df['genres_list'] for g in genres]
        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts()
            top_genres = genre_counts.head(2).index.tolist()
            cluster_labels[c] = ' & '.join(top_genres)
        else:
            cluster_labels[c] = f'Cluster {c}'

    pca_data['cluster_labels'] = cluster_labels

    cluster_dist = pd.Series(clusters).value_counts().to_dict()
    print(f"   Cluster distribution: {cluster_dist}")

    return kmeans, pca_data


def save_models(df, tfidf, tfidf_matrix, cast_matrix, kmeans, pca_data):
    """Serialize all models and processed data.

    Dense NxN similarity matrices (cosine_similarity.pkl, cast_similarity.pkl)
    are intentionally NOT saved — they are ~591 MB each and cause MemoryError
    when loaded. Similarity is computed on-the-fly per request instead.
    """
    print("[Saving models...]")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save processed dataframe
    df.to_pickle(PROCESSED_DATA_PATH)
    print(f"   [OK] Processed data -> {PROCESSED_DATA_PATH}")

    # Save TF-IDF vectorizer
    with open(TFIDF_VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"   [OK] TF-IDF vectorizer -> {TFIDF_VECTORIZER_PATH}")

    # Save sparse TF-IDF matrix
    with open(TFIDF_MATRIX_PATH, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    print(f"   [OK] TF-IDF matrix -> {TFIDF_MATRIX_PATH}")

    # Save sparse cast matrix
    with open(CAST_MATRIX_PATH, 'wb') as f:
        pickle.dump(cast_matrix, f)
    print(f"   [OK] Cast matrix -> {CAST_MATRIX_PATH}")

    # Save K-Means model
    with open(KMEANS_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"   [OK] K-Means model -> {KMEANS_PATH}")

    # Save PCA data
    with open(PCA_DATA_PATH, 'wb') as f:
        pickle.dump(pca_data, f)
    print(f"   [OK] PCA data -> {PCA_DATA_PATH}")


def run_pipeline():
    """Execute the full preprocessing pipeline."""
    print("=" * 60)
    print(">> Smart Content Recommender -- Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Load
    df = load_data()

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Sentiment Analysis
    df = add_sentiment(df)

    # Step 4: Feature Engineering
    df = create_metadata_soup(df)

    # Step 5: Build TF-IDF Model (sparse matrix only, no full NxN similarity)
    tfidf, tfidf_matrix = build_tfidf_model(df)

    # Step 6: Build Cast Sparse Matrix
    cast_matrix = build_cast_similarity(df)

    # Step 7: Build Clusters
    kmeans, pca_data = build_clusters(tfidf_matrix, df)

    # Saving
    save_models(df, tfidf, tfidf_matrix, cast_matrix, kmeans, pca_data)

    print("\n" + "=" * 60)
    print("[DONE] Preprocessing complete!")
    print(f"   Total titles processed: {len(df)}")
    print(f"   Movies: {len(df[df['type'] == 'Movie'])}")
    print(f"   TV Shows: {len(df[df['type'] == 'TV Show'])}")
    print(f"   Unique genres: {len(set(g for gl in df['genres_list'] for g in gl))}")
    print(f"   Clusters: {df['cluster'].nunique()}")
    print("=" * 60)


if __name__ == '__main__':
    run_pipeline()
