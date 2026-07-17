# Netflix_recommendation_system
# 🎬 Smart Content Recommender

> Movies & TV shows recommendation engine built on Netflix dataset using TF-IDF, Cosine Similarity, K-Means Clustering, and Sentiment Analysis.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

---

## About

**Smart Content Recommender** is a full-stack ML-powered recommendation system that analyzes 8,800+ Netflix titles to provide intelligent content suggestions. It combines multiple machine learning techniques with a stunning UI to deliver an engaging user experience.

### Key ML Features

| Feature | Technique | Purpose |
|---|---|---|
| Content Similarity | TF-IDF + Cosine Similarity | Find shows/movies with similar content profiles |
| Cast Matching | CountVectorizer Similarity | Recommend titles sharing cast members |
| Mood Classification | VADER Sentiment Analysis | Classify content mood (Feel-Good, Intense, Dark, Thought-Provoking) |
| Thematic Clustering | K-Means + PCA | Group titles into 15 thematic clusters |
| Multi-Profile Matching | Average TF-IDF Vectors | Blend multiple liked titles into a taste profile |
| Explainable AI | Feature Overlap Parsing | Show *why* each recommendation was made |

---

## Features

- **Smart Search** — Real-time autocomplete across 8,800+ titles
- **Content-Based Recommendations** — TF-IDF cosine similarity on combined metadata
- **Cast-Based Recommendations** — Find content with overlapping actors
- **Mood Discovery** — Filter by emotional tone using VADER sentiment analysis
- **Genre Filtering** — Browse across 42 genres
- **Multi-Select Mode** — Pick multiple favorites and get blended recommendations
- **Explainable Results** — Every recommendation includes a human-readable explanation
- **Dataset Statistics** — Interactive visualizations of the Netflix catalog
- **Modern UI** — Dark glassmorphism theme with smooth animations

---

## Tech Stack

### Backend
- **Python 3.11+** — Core language
- **FastAPI** — REST API framework with auto-generated Swagger docs
- **scikit-learn** — TF-IDF Vectorizer, Cosine Similarity, K-Means, PCA
- **VADER (vaderSentiment)** — Sentiment analysis for mood classification
- **pandas / NumPy** — Data processing and manipulation
- **Uvicorn** — ASGI server

### Frontend
- **HTML5** — Semantic markup
- **CSS3** — Custom properties, glassmorphism, keyframe animations
- **Vanilla JavaScript** — No framework dependencies
- **Inter (Google Fonts)** — Modern typography

---

## Project Architecture

```
Netflix_recommendation_system/
├── backend/
│   ├── app.py                    # FastAPI application (10 REST endpoints)
│   ├── recommender.py            # ML recommendation engine
│   ├── preprocessing.py          # Data cleaning & model building pipeline
│   ├── requirements.txt          # Python dependencies
│   ├── data/
│   │   └── netflix_titles.csv    # Raw Netflix dataset (8,807 titles)
│   └── models/                   # Serialized ML models (auto-generated, gitignored)
│       ├── processed_data.pkl
│       ├── tfidf_vectorizer.pkl
│       ├── tfidf_matrix.pkl      # Sparse matrix (used for on-the-fly similarity)
│       ├── cast_matrix.pkl       # Sparse cast matrix
│       ├── kmeans_model.pkl
│       └── pca_data.pkl
├── frontend/
│   ├── index.html                # Main page
│   ├── css/styles.css            # Design system
│   └── js/
│       ├── api.js                # API client with caching
│       ├── components.js         # Reusable UI components
│       └── app.js                # Core application logic
├── render.yaml                   # Render deployment config
└── README.md
```

---

## ML Approach

### 1. Feature Engineering — Metadata Soup

Content features are combined into a weighted text representation:

```python
# Weight distribution:
# Genres:    3x (strongest similarity signal)
# Director:  2x (style indicator)
# Cast:      2x (actor-based similarity)
# Country:   1x
# Rating:    1x
# Description: 1x (natural language content)
```

Names are preprocessed to prevent false token matches (e.g., "Johnny Depp" → "johnnydepp").

### 2. TF-IDF Vectorization

- **20,000 features** with unigram + bigram support
- Stop word removal, min/max document frequency thresholds
- Produces a sparse 8,807 × 20,000 matrix

### 3. Cosine Similarity

Pre-computed pairwise similarity matrix (8,807 × 8,807) enables instant recommendations at query time.

### 4. Cast Similarity

A separate CountVectorizer-based similarity matrix specifically for cast member overlap, enabling "find shows with the same actors" functionality.

### 5. Sentiment Analysis (VADER)

Each description is analyzed for sentiment polarity:
- **Compound score ≥ 0.3** → Feel-Good 
- **Compound score ≤ -0.3** → Dark 
- **Compound score ≤ -0.05** → Intense 
- **Otherwise** → Thought-Provoking 

### 6. K-Means Clustering

15 thematic clusters identified from TF-IDF vectors, with TruncatedSVD reduction to 2D for visualization (TruncatedSVD operates on sparse matrices natively, avoiding the memory cost of PCA). Clusters are auto-labeled by dominant genres.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/titles?q=search` | Search/autocomplete titles |
| `GET` | `/api/title/{show_id}` | Get title details |
| `GET` | `/api/genres` | List all 42 genres |
| `GET` | `/api/stats` | Dataset statistics |
| `GET` | `/api/trending` | Recently added content |
| `POST` | `/api/recommend` | Content-based recommendations |
| `POST` | `/api/recommend/cast` | Cast-based recommendations |
| `POST` | `/api/recommend/multi` | Multi-title profile recommendations |
| `POST` | `/api/recommend/genre` | Genre + mood filtered recommendations |
| `GET` | `/api/clusters` | Cluster visualization data |

Full interactive API docs available at: `http://localhost:8000/docs`

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- pip

### Quick Start

```bash
pip install -r backend/requirements.txt
python backend/preprocessing.py   # generates model files (~1-2 min)
uvicorn app:app --reload --app-dir backend
```

Open **http://localhost:8000** in your browser.

> **Note:** Model files (`*.pkl`) are not included in the repository and are generated locally by `preprocessing.py`. This avoids committing hundreds of MB of binary files to git.

### Step-by-Step

#### 1. Clone the repository

```bash
git clone https://github.com/NeelShah01/Netflix_recommendation_system.git
cd Netflix_recommendation_system
```

#### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

#### 3. Run Preprocessing (Build ML Models)

```bash
python backend/preprocessing.py
```

This will:
- Clean and preprocess the Netflix dataset
- Run VADER sentiment analysis on all descriptions
- Build sparse TF-IDF vectors and cast feature matrix
- Perform K-Means clustering with TruncatedSVD for 2D visualization
- Serialize all models to `backend/models/`

#### 4. Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir backend
```

#### 5. Open in Browser

Navigate to: **http://localhost:8000**

Interactive API docs: **http://localhost:8000/docs**

---

## Dataset

- **Source**: Netflix Movies and TV Shows dataset (Taken from Kaggle)
- **Total Titles**: 8,807 (6,131 Movies + 2,676 TV Shows)
- **Genres**: 42 unique genres
- **Countries**: 748 unique countries
- **Date Range**: 1925 - 2021

---

## Future Improvements

- [ ] Collaborative filtering using user interaction data
- [ ] SBERT (Sentence-BERT) for semantic similarity
- [ ] FAISS for approximate nearest neighbor search at scale
- [ ] User accounts with watchlist and rating history
- [ ] Docker containerization for easy deployment
- [ ] Integration with TMDB API for poster images

---

## License

This project is for educational and portfolio purposes.

