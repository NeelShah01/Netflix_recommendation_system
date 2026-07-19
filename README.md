# Netflix_recommendation_system
# 🎬 Smart Content Recommender

> Movies & TV shows recommendation engine built on Netflix dataset using TF-IDF, Cosine Similarity, K-Means Clustering, Sentiment Analysis, and persistent MongoDB database storage.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

---

## About

**Smart Content Recommender** is a full-stack ML-powered recommendation system that analyzes 8,800+ Netflix titles to provide intelligent content suggestions. It combines multiple machine learning techniques with a MongoDB-backed persistent database and a stunning, responsive front-end user experience.

### Key ML & Database Features

| Feature | Technique / Database | Purpose |
|---|---|---|
| Content Similarity | TF-IDF + Cosine Similarity | Find shows/movies with similar content profiles |
| Cast Matching | CountVectorizer Similarity | Recommend titles sharing cast members |
| Mood Classification | VADER Sentiment Analysis | Classify content mood (Feel-Good, Intense, Dark, Thought-Provoking) |
| Thematic Clustering | K-Means + PCA | Group titles into 15 thematic clusters |
| Multi-Profile Matching | Average TF-IDF Vectors | Blend multiple liked titles into a taste profile |
| Autocomplete Search | Client-side In-memory Index | Local query scanning for instant (<5ms) search autocomplete |
| Database Persistence | MongoDB | Stores users, watchlists, watched history, and movie catalog |
| Fail-Safe Mode | Local File Fallback | Automatic fallback to local files (`*.pkl`) if MongoDB is offline |
| Explainable AI | Feature Overlap Parsing | Show *why* each recommendation was made |

---

## Features

- **Smart & Instant Search** — In-memory pre-loaded catalog index for instant (<5ms) search autocomplete.
- **Dynamic Catalog Editor** — Add, edit, or delete items in the Netflix dataset. Rebuilds the TF-IDF vector matrix on-the-fly.
- **Dynamic Schemas** — Easily add new custom fields dynamically to any item catalog.
- **User Accounts & Sign In** — Persistent Google Identity Sign-In (OAuth) and local Email/Password registration (secured with SHA-256 password hashing).
- **Multiple Custom Watchlists** — Create multiple distinct named lists (e.g., "Watch Later", "Comedy Faves"). Prevents duplicate items.
- **Watched History Tracking** — Mark movies/TV shows as watched and review them in a separate profile page.
- **Content-Based & Cast-Based Recommendations** — Explore similarity scores based on genres, overview description, directors, and actors.
- **Modern UI** — Dark glassmorphic design with a customizable sidebar, particle system background, and responsive layouts.

---

## Tech Stack

### Backend
- **Python 3.11+** — Core language
- **FastAPI** — REST API web framework
- **MongoDB & PyMongo** — Dynamic database and python client
- **python-dotenv** — Environment configuration loader
- **scikit-learn** — TF-IDF Vectorizer, Cosine Similarity, K-Means, PCA
- **VADER (vaderSentiment)** — Sentiment analysis for mood classification
- **pandas / NumPy** — Data preprocessing and manipulation
- **Uvicorn** — ASGI web server

### Frontend
- **HTML5 & Vanilla CSS3** — Semantic elements, custom properties, glassmorphism, responsive keyframe animations
- **Vanilla JavaScript** — Light, responsive client state management (no bloated frameworks)

---

## Project Architecture

```
Netflix_recommendation_system/
├── backend/
│   ├── app.py                    # FastAPI application (20 REST endpoints)
│   ├── database.py               # MongoDB connector and fail-safe driver
│   ├── recommender.py            # ML recommendation engine (load, reload, upsert, delete)
│   ├── preprocessing.py          # Data cleaning & model building pipeline
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # Environment variables template
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
│   ├── index.html                # Main page (sign-in modals, sidebar, layouts)
│   ├── css/styles.css            # Stylesheets
│   └── js/
│       ├── api.js                # API client (fetches catalog/auth endpoints)
│       ├── auth.js               # Client session state manager
│       ├── watchlist.js          # Synchronous cache and mutation handlers
│       ├── components.js         # Cards, carousels, and dropdown rendering
│       └── app.js                # Core controller logic
├── render.yaml                   # Render deployment config
└── README.md
```

---

## API Endpoints

### Catalog APIs
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/titles` | Search/autocomplete titles (fallback) |
| `GET` | `/api/titles/index` | Returns a compact JSON index of all catalog items |
| `GET` | `/api/title/{show_id}` | Get specific title details |
| `GET` | `/api/genres` | List all unique genres |
| `GET` | `/api/stats` | Dataset insights and statistics |
| `GET` | `/api/trending` | Recently added content |

### ML Recommendation APIs
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/recommend` | Content-based recommendations |
| `POST` | `/api/recommend/cast` | Cast-based recommendations |
| `POST` | `/api/recommend/multi` | Blended taste profile recommendations |
| `POST` | `/api/recommend/genre` | Genre + mood filtered recommendations |
| `GET` | `/api/clusters` | Cluster coordinates and labels |
| `GET` | `/api/clusters/{cluster_id}` | Group titles by cluster ID |

### Authentication & Users APIs (MongoDB backed)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Register a new user |
| `POST` | `/api/auth/login` | Log in email credentials |
| `POST` | `/api/auth/google` | Create/sync Google OAuth profiles |

### Watchlists & Watched APIs (MongoDB backed)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/watchlist` | Get all watchlists for a user |
| `POST` | `/api/watchlist/create` | Create a new named watchlist |
| `DELETE` | `/api/watchlist` | Delete a custom watchlist |
| `POST` | `/api/watchlist/item/add` | Push item into a watchlist |
| `POST` | `/api/watchlist/item/remove` | Pull item from a watchlist |
| `GET` | `/api/watched` | Retrieve watched history list |
| `POST` | `/api/watched/add` | Mark item as watched |
| `POST` | `/api/watched/remove` | Unmark item from watched list |

### Catalog Administration APIs
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/catalog/item` | Dynamically insert a new catalog title |
| `PUT` | `/api/catalog/item/{show_id}` | Edit fields or add dynamic new fields |
| `DELETE` | `/api/catalog/item/{show_id}` | Remove a title and rebuild models |

Interactive Swagger docs available at: `http://localhost:8000/docs`

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- MongoDB instance (Local or Cloud)
- Pip / Virtual environment tool

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/NeelShah01/Netflix_recommendation_system.git
cd Netflix_recommendation_system
```

#### 2. Configure Environment Variables
Create a file named `.env` in the `backend/` folder (reference `backend/.env.example`):
```env
MONGO_URI=mongodb://localhost:27017/
```

#### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

#### 4. Build Machine Learning Models
```bash
python backend/preprocessing.py
```
This builds and caches processed data pickles inside `backend/models/`.

#### 5. Run MongoDB Server
Ensure MongoDB is running locally on port 27017:
* **Docker Option**: `docker run -d --name smartrec-mongo -p 27017:27017 mongo:latest`
* **Windows Service**: Install MongoDB Community Edition. It will start automatically.

#### 6. Start the FastAPI Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir backend
```
*At startup, the server automatically seeds the raw Netflix catalog into MongoDB if the collection is empty.*

#### 7. Open in Browser
* Front-End Web Page: **http://localhost:8000**
* Interactive API Documentation: **http://localhost:8000/docs**

---

## License

This project is for educational and portfolio purposes.
