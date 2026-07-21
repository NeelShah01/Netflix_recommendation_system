"""
app.py — FastAPI Application
Smart Content Recommender

REST API serving ML-powered movie/TV show recommendations
from the Netflix dataset. Provides multiple recommendation
strategies with explainable results.

API Documentation: http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from recommender import RecommendationEngine

# ──────────────────────────────────────────────
# Models (Request/Response schemas)
# ──────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """Request body for single-title recommendations."""
    title: str = Field(..., description="Title to get recommendations for")
    n_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations")
    content_type: Optional[str] = Field(None, description="Filter by 'Movie' or 'TV Show'")
    exclude_genres: Optional[List[str]] = None
    user_id: Optional[str] = None


class CastRecommendRequest(BaseModel):
    """Request body for cast-based recommendations."""
    title: str = Field(..., description="Title to find cast-similar content for")
    n_recommendations: int = Field(10, ge=1, le=50)
    content_type: Optional[str] = None
    exclude_genres: Optional[List[str]] = None
    user_id: Optional[str] = None


class MultiRecommendRequest(BaseModel):
    """Request body for multi-select profile recommendations."""
    titles: List[str] = Field(..., min_length=1, max_length=10, description="List of liked titles")
    n_recommendations: int = Field(10, ge=1, le=50)
    content_type: Optional[str] = None
    exclude_genres: Optional[List[str]] = None
    user_id: Optional[str] = None


class GenreRecommendRequest(BaseModel):
    """Request body for genre + mood filtered recommendations."""
    genre: Optional[str] = Field(None, description="Genre to filter by")
    mood: Optional[str] = Field(None, description="Mood: feel-good, intense, dark, thought-provoking")
    content_type: Optional[str] = None
    n_recommendations: int = Field(20, ge=1, le=50)
    exclude_genres: Optional[List[str]] = None
    user_id: Optional[str] = None


# ──────────────────────────────────────────────
# Application Setup
# ──────────────────────────────────────────────

# Initialize recommendation engine
engine = RecommendationEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models at startup, cleanup on shutdown."""
    # Startup
    engine.load_models()
    print(">> Smart Content Recommender API is ready!")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Smart Content Recommender API",
    description="ML-powered movie & TV show recommendation system based on Netflix data",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache-control middleware to prevent browsers caching failed API states
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Serve static frontend files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ──────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend index.html."""
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Smart Content Recommender API", "docs": "/docs"}


@app.get("/api/titles")
async def search_titles(
    q: str = Query("", description="Search query for title autocomplete"),
    limit: int = Query(10, ge=1, le=50)
):
    """Search for titles matching a query string. Used for autocomplete."""
    if not q.strip():
        return []
    results = engine.search_titles(q, limit=limit)
    return results


@app.get("/api/titles/index")
async def get_titles_index():
    """
    Return a compact index of ALL titles for client-side instant search.
    Only includes fields needed for autocomplete: show_id, title, type, release_year, rating, listed_in.
    Called once on page load — eliminates per-keystroke network requests.
    """
    return engine.get_titles_index()


@app.get("/api/title/{show_id}")
async def get_title(show_id: str):
    """Get full details for a specific title by show_id."""
    result = engine.get_title_details(show_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Title with id '{show_id}' not found")
    return result


import urllib.request
import urllib.parse
import json

import ssl

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()


import re

def clean_query(title: str) -> str:
    # Replace ampersands with a space
    cleaned = title.replace("&", " ")
    # Remove non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned)
    # Normalize whitespaces
    cleaned = " ".join(cleaned.split())
    return cleaned


@app.get("/api/media/{show_id}")
def get_media_metadata(show_id: str):
    """
    Fetch poster, backdrop, and trailer key from TMDB API.
    Caches results in MongoDB to optimize rate limits and performance.
    """
    # 1. Fetch title details from the recommendation engine
    title_details = engine.get_title_details(show_id)
    if not title_details:
        raise HTTPException(status_code=404, detail=f"Title with id '{show_id}' not found")
        
    title = title_details.get("title")
    media_type = title_details.get("type")
    release_year = title_details.get("release_year")
    
    # 2. Check cache in MongoDB if database is available
    from database import db_available, titles_col
    if db_available and titles_col is not None:
        try:
            doc = titles_col.find_one({"show_id": show_id})
            if doc and isinstance(doc.get("tmdb_data"), dict):
                return doc["tmdb_data"]
        except Exception as e:
            print(f">> [WARN] Failed to query cache from MongoDB: {e}")
            
    # 3. If cache miss, fetch from TMDB (if key is set)
    if not TMDB_API_KEY:
        return {
            "status": "no_key",
            "poster_path": None,
            "backdrop_path": None,
            "trailer_url": None,
            "vote_average": None,
            "explanation": "Add TMDB_API_KEY to your backend/.env file to enable posters and trailers."
        }
        
    try:
        # Create unverified SSL context to bypass SSL unexpected EOF errors on Windows/TMDB handshake
        ssl_context = ssl._create_unverified_context()
        
        # Search for title on TMDB
        cleaned_title = clean_query(title)
        encoded_query = urllib.parse.quote(cleaned_title)
        tmdb_type = "movie" if media_type == "Movie" else "tv"
        
        search_url = f"http://api.themoviedb.org/3/search/{tmdb_type}?api_key={TMDB_API_KEY}&query={encoded_query}"
        if release_year:
            # For movies search key is 'primary_release_year', for TV shows it is 'first_air_date_year'
            year_key = "primary_release_year" if tmdb_type == "movie" else "first_air_date_year"
            search_url += f"&{year_key}={release_year}"
            
        req = urllib.request.Request(search_url, headers={"User-Agent": "SmartRec/1.0"})
        with urllib.request.urlopen(req, context=ssl_context, timeout=5) as response:
            search_data = json.loads(response.read().decode())
            
        results = search_data.get("results", [])
        if not results and release_year:
            # Try searching again without release year as fallback
            search_url_fallback = f"http://api.themoviedb.org/3/search/{tmdb_type}?api_key={TMDB_API_KEY}&query={encoded_query}"
            req_fb = urllib.request.Request(search_url_fallback, headers={"User-Agent": "SmartRec/1.0"})
            with urllib.request.urlopen(req_fb, context=ssl_context, timeout=5) as response:
                search_data = json.loads(response.read().decode())
                results = search_data.get("results", [])
                
        if not results:
            tmdb_data = {
                "status": "not_found",
                "poster_path": None,
                "backdrop_path": None,
                "trailer_url": None,
                "vote_average": None
            }
            # Cache the negative result to avoid spamming TMDB for missing content
            if db_available and titles_col is not None:
                try:
                    titles_col.update_one({"show_id": show_id}, {"$set": {"tmdb_data": tmdb_data}})
                except Exception:
                    pass
            return tmdb_data
            
        # Get best match and retrieve full details + videos
        best_match = results[0]
        tmdb_id = best_match["id"]
        
        details_url = f"http://api.themoviedb.org/3/{tmdb_type}/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=videos"
        req_details = urllib.request.Request(details_url, headers={"User-Agent": "SmartRec/1.0"})
        with urllib.request.urlopen(req_details, context=ssl_context, timeout=5) as response:
            details = json.loads(response.read().decode())
            
        poster_path = details.get("poster_path")
        backdrop_path = details.get("backdrop_path")
        vote_average = details.get("vote_average")
        
        # Extract YouTube trailer video key
        videos = details.get("videos", {}).get("results", [])
        trailer_key = None
        for vid in videos:
            if vid.get("site") == "YouTube" and vid.get("type") in ["Trailer", "Teaser"]:
                trailer_key = vid.get("key")
                if vid.get("type") == "Trailer":
                    break # Preference for official trailer
                    
        trailer_url = f"https://www.youtube.com/embed/{trailer_key}" if trailer_key else None
        
        # Resolve poster URL
        resolved_poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

        # Upload poster to Cloudinary on-demand if credentials are set
        CLOUDINARY_URL = os.getenv("CLOUDINARY_URL", "").strip()
        if resolved_poster_url and CLOUDINARY_URL:
            try:
                import cloudinary
                import cloudinary.uploader
                # Automatically uploads the TMDB hotlinked image to Cloudinary CDN
                upload_result = cloudinary.uploader.upload(
                    resolved_poster_url,
                    folder="netflix_posters",
                    public_id=show_id,
                    overwrite=True
                )
                if upload_result.get("secure_url"):
                    resolved_poster_url = upload_result["secure_url"]
            except Exception as cl_err:
                print(f">> [WARN] Cloudinary upload failed for '{title}' poster: {cl_err}")

        tmdb_data = {
            "status": "ok",
            "poster_path": resolved_poster_url,
            "backdrop_path": f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else None,
            "trailer_url": trailer_url,
            "vote_average": round(vote_average, 1) if vote_average else None
        }
        
        # Cache results in MongoDB
        if db_available and titles_col is not None:
            try:
                titles_col.update_one({"show_id": show_id}, {"$set": {"tmdb_data": tmdb_data}})
            except Exception as e:
                print(f">> [WARN] Failed to write cache to MongoDB: {e}")
                
        return tmdb_data
        
    except Exception as e:
        print(f">> [ERROR] Failed TMDB lookup for '{title}': {e}")
        return {
            "status": "error",
            "poster_path": None,
            "backdrop_path": None,
            "trailer_url": None,
            "vote_average": None,
            "error_message": str(e)
        }



@app.get("/api/genres")
async def get_genres():
    """Get all available genres sorted alphabetically."""
    return engine.get_all_genres()


@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics for the dashboard section."""
    return engine.get_stats()


@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    """
    Get content-based recommendations for a single title.
    Uses TF-IDF cosine similarity on combined metadata features.
    Each recommendation includes an explanation.
    """
    results = engine.recommend_by_title(
        title=req.title,
        n=req.n_recommendations,
        content_type=req.content_type,
        exclude_genres=req.exclude_genres,
        user_id=req.user_id
    )
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Title '{req.title}' not found. Try searching with /api/titles?q=..."
        )
    return {
        "query_title": req.title,
        "recommendation_type": "content-based",
        "count": len(results),
        "results": results,
        "recommendations": results
    }


@app.post("/api/recommend/cast")
async def recommend_by_cast(req: CastRecommendRequest):
    """
    Get cast-based recommendations for a single title.
    Finds titles with the most overlapping cast members.
    """
    results = engine.recommend_by_cast(
        title=req.title,
        n=req.n_recommendations,
        content_type=req.content_type,
        exclude_genres=req.exclude_genres,
        user_id=req.user_id
    )
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Title '{req.title}' not found or has no cast data."
        )
    return {
        "query_title": req.title,
        "recommendation_type": "cast-based",
        "count": len(results),
        "results": results,
        "recommendations": results
    }


@app.post("/api/recommend/multi")
async def recommend_multi(req: MultiRecommendRequest):
    """
    Get recommendations based on multiple liked titles.
    Computes an average TF-IDF profile and finds nearest neighbors.
    """
    results = engine.recommend_multi_select(
        titles=req.titles,
        n=req.n_recommendations,
        content_type=req.content_type,
        exclude_genres=req.exclude_genres,
        user_id=req.user_id
    )
    if not results:
        raise HTTPException(
            status_code=404,
            detail="None of the provided titles were found in the dataset."
        )
    return {
        "query_titles": req.titles,
        "recommendation_type": "multi-select-profile",
        "count": len(results),
        "results": results,
        "recommendations": results
    }


@app.post("/api/recommend/genre")
async def recommend_by_genre(req: GenreRecommendRequest):
    """
    Get recommendations filtered by genre and/or mood.
    Mood values: feel-good, intense, dark, thought-provoking
    """
    results = engine.recommend_by_genre_mood(
        genre=req.genre,
        mood=req.mood,
        content_type=req.content_type,
        n=req.n_recommendations,
        exclude_genres=req.exclude_genres,
        user_id=req.user_id
    )
    return {
        "filters": {
            "genre": req.genre,
            "mood": req.mood,
            "content_type": req.content_type
        },
        "recommendation_type": "genre-mood-filter",
        "count": len(results),
        "results": results,
        "recommendations": results
    }


@app.get("/api/clusters")
async def get_clusters():
    """Get cluster data for 2D scatter plot visualization."""
    return engine.get_cluster_data()


@app.get("/api/clusters/{cluster_id}")
async def get_cluster_titles(cluster_id: int, n: int = Query(20, ge=1, le=50)):
    """Get titles belonging to a specific cluster."""
    results = engine.get_cluster_titles(cluster_id, n=n)
    label = engine.pca_data['cluster_labels'].get(cluster_id, f'Cluster {cluster_id}')
    return {
        "cluster_id": cluster_id,
        "cluster_label": label,
        "count": len(results),
        "results": results
    }


@app.get("/api/trending")
async def get_trending(
    n: int = Query(20, ge=1, le=50),
    content_type: Optional[str] = Query(None)
):
    """Get trending/recently added content."""
    results = engine.get_trending(n=n, content_type=content_type)
    return {
        "recommendation_type": "trending",
        "count": len(results),
        "results": results
    }


# ──────────────────────────────────────────────
# MongoDB User & Catalog APIs
# ──────────────────────────────────────────────
import hashlib
import time
from database import users_col, watchlists_col, watched_col, titles_col, reviews_col

# Schemas
class RegisterRequest(BaseModel):
    email: str
    password: str
    displayName: str

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleLoginRequest(BaseModel):
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    sub: str

class CreateWatchlistRequest(BaseModel):
    user_id: str
    name: str

class WatchlistItemRequest(BaseModel):
    user_id: str
    list_name: str
    show_id: str
    title: str
    type: str

class WatchedItemRequest(BaseModel):
    user_id: str
    show_id: str
    title: str
    type: str


class ReviewSubmitRequest(BaseModel):
    user_id: str
    displayName: str
    show_id: str
    rating: int = Field(..., ge=1, le=5)
    review_text: str


def sha256_hash(text: str) -> str:
    """Helper to hash string using SHA-256 to match database password storage."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# --- Authentication ---

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    email_clean = req.email.strip().lower()
    if users_col.find_one({"email": email_clean}):
        raise HTTPException(status_code=400, detail="An account with this email already exists.")
    
    uid = "email_" + hashlib.md5(email_clean.encode('utf-8')).hexdigest()
    
    # Generate avatar letter and random gradient hue
    avatar = req.displayName.strip()[0].upper() if req.displayName.strip() else "?"
    hue = abs(hash(email_clean)) % 360
    gradient = f"linear-gradient(135deg, hsl({hue}, 70%, 40%), hsl({(hue + 60) % 360}, 80%, 30%))"

    user_record = {
        "uid": uid,
        "email": email_clean,
        "displayName": req.displayName.strip(),
        "provider": "email",
        "avatar": avatar,
        "gradient": gradient,
        "passwordHash": sha256_hash(req.password),
        "createdAt": int(time.time())
    }
    
    users_col.insert_one(user_record)
    
    # Return user profile without password hash
    del user_record["_id"]
    del user_record["passwordHash"]
    return {"success": True, "user": user_record}


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    email_clean = req.email.strip().lower()
    user = users_col.find_one({"email": email_clean})
    if not user:
        raise HTTPException(status_code=404, detail="No account found for this email. Please register first.")
    
    entered_hash = sha256_hash(req.password)
    if entered_hash != user.get("passwordHash"):
        raise HTTPException(status_code=401, detail="Incorrect password. Please try again.")
    
    user_record = {
        "uid": user.get("uid"),
        "email": user.get("email"),
        "displayName": user.get("displayName", "User"),
        "provider": user.get("provider", "email"),
        "avatar": user.get("avatar") or (user.get("displayName", "U")[0].upper() if user.get("displayName") else "U"),
        "gradient": user.get("gradient") or "linear-gradient(135deg, #e65c00, #F9D423)"
    }
    return {"success": True, "user": user_record}


@app.post("/api/auth/google")
async def google_login(req: GoogleLoginRequest):
    email_clean = req.email.strip().lower()
    user = users_col.find_one({"email": email_clean})
    
    if not user:
        # Register Google user on first login
        uid = f"google_{req.sub}"
        avatar = req.name[0].upper() if req.name else "?"
        hue = abs(hash(email_clean)) % 360
        gradient = f"linear-gradient(135deg, hsl({hue}, 70%, 40%), hsl({(hue + 60) % 360}, 80%, 30%))"
        
        user_record = {
            "uid": uid,
            "email": email_clean,
            "displayName": req.name,
            "provider": "google",
            "avatar": avatar,
            "gradient": gradient,
            "picture": req.picture,
            "createdAt": 1784379430
        }
        users_col.insert_one(user_record)
        user = user_record
    else:
        # Update picture if needed
        if req.picture and user.get("picture") != req.picture:
            users_col.update_one({"email": email_clean}, {"$set": {"picture": req.picture}})
            user["picture"] = req.picture

    user_record = {
        "uid": user.get("uid"),
        "email": user.get("email"),
        "displayName": user.get("displayName", "Google User"),
        "provider": user.get("provider", "google"),
        "avatar": user.get("avatar") or (user.get("displayName", "G")[0].upper() if user.get("displayName") else "G"),
        "gradient": user.get("gradient") or "linear-gradient(135deg, #e65c00, #F9D423)",
        "picture": user.get("picture")
    }
    return {"success": True, "user": user_record}


# --- Watchlists ---

@app.get("/api/watchlist")
async def get_watchlists(user_id: str):
    lists = list(watchlists_col.find({"user_id": user_id}, {"_id": 0}))
    # Format to match client structure: { list_name: items[] }
    result = {}
    for wl in lists:
        result[wl["name"]] = wl.get("items", [])
    return result


@app.post("/api/watchlist/create")
async def create_watchlist(req: CreateWatchlistRequest):
    # Check if list already exists
    existing = watchlists_col.find_one({"user_id": req.user_id, "name": req.name})
    if existing:
        return {"success": True}
    
    watchlists_col.insert_one({
        "user_id": req.user_id,
        "name": req.name,
        "items": []
    })
    return {"success": True}


@app.delete("/api/watchlist")
async def delete_watchlist(user_id: str, name: str):
    watchlists_col.delete_one({"user_id": user_id, "name": name})
    return {"success": True}


@app.post("/api/watchlist/item/add")
async def add_watchlist_item(req: WatchlistItemRequest):
    # Ensure watchlist exists
    wl = watchlists_col.find_one({"user_id": req.user_id, "name": req.list_name})
    if not wl:
        # Auto-create if list doesn't exist
        watchlists_col.insert_one({
            "user_id": req.user_id,
            "name": req.list_name,
            "items": []
        })
        wl = {"items": []}
    
    # Check duplicate
    for item in wl.get("items", []):
        if item.get("show_id") == req.show_id:
            raise HTTPException(status_code=400, detail="Already in the watchlist")
            
    new_item = {
        "show_id": req.show_id,
        "title": req.title,
        "type": req.type
    }
    watchlists_col.update_one(
        {"user_id": req.user_id, "name": req.list_name},
        {"$push": {"items": new_item}}
    )
    return {"success": True}


@app.post("/api/watchlist/item/remove")
async def remove_watchlist_item(req: WatchlistItemRequest):
    watchlists_col.update_one(
        {"user_id": req.user_id, "name": req.list_name},
        {"$pull": {"items": {"show_id": req.show_id}}}
    )
    return {"success": True}


# --- Watched History ---

@app.get("/api/watched")
async def get_watched(user_id: str):
    history = list(watched_col.find({"user_id": user_id}, {"_id": 0}))
    return history


@app.post("/api/watched/add")
async def add_watched(req: WatchedItemRequest):
    # Check duplicate
    existing = watched_col.find_one({"user_id": req.user_id, "show_id": req.show_id})
    if existing:
        return {"success": True}
        
    watched_record = {
        "user_id": req.user_id,
        "show_id": req.show_id,
        "title": req.title,
        "type": req.type,
        "watchedAt": int(time.time())
    }
    watched_col.insert_one(watched_record)
    return {"success": True}


@app.post("/api/watched/remove")
async def remove_watched(req: WatchedItemRequest):
    watched_col.delete_one({"user_id": req.user_id, "show_id": req.show_id})
    return {"success": True}


# --- Catalog Admin (Dynamic Schema support!) ---

@app.post("/api/catalog/item")
async def create_catalog_item(body: dict):
    """
    Dynamically insert a new movie/show.
    Accepts arbitrary JSON fields.
    """
    if "show_id" not in body or not body["show_id"]:
        raise HTTPException(status_code=400, detail="Missing required field 'show_id'.")
    if "title" not in body or not body["title"]:
        raise HTTPException(status_code=400, detail="Missing required field 'title'.")
        
    show_id = body["show_id"]
    existing = titles_col.find_one({"show_id": show_id})
    if existing:
        raise HTTPException(status_code=400, detail=f"Item with show_id '{show_id}' already exists.")

    engine.add_or_update_title(show_id, body)
    return {"success": True, "message": f"Successfully created title '{body['title']}'."}


@app.put("/api/catalog/item/{show_id}")
async def update_catalog_item(show_id: str, body: dict):
    """
    Dynamically update fields or create new fields for a specific show_id.
    Accepts arbitrary JSON fields.
    """
    existing = titles_col.find_one({"show_id": show_id})
    if not existing:
        raise HTTPException(status_code=404, detail=f"Content with show_id '{show_id}' not found.")

    engine.add_or_update_title(show_id, body)
    return {"success": True, "message": f"Successfully updated title '{show_id}'."}


@app.delete("/api/catalog/item/{show_id}")
async def delete_catalog_item(show_id: str):
    """Delete a movie/show from the database and rebuild models."""
    existing = titles_col.find_one({"show_id": show_id})
    if not existing:
        raise HTTPException(status_code=404, detail=f"Content with show_id '{show_id}' not found.")

    engine.delete_title(show_id)
    return {"success": True, "message": f"Successfully deleted title '{show_id}'."}


# --- Personalized Recommendations ---

@app.get("/api/recommendations/personalized")
async def get_personalized_recommendations(
    user_id: str,
    limit: int = Query(20, ge=1, le=50),
    exclude_genres: Optional[List[str]] = Query(None),
    content_type: Optional[str] = Query(None)
):
    """
    Generate personalized recommendations based on the user's combined watchlist & watched history.
    """
    watchlist_titles = []
    try:
        lists = list(watchlists_col.find({"user_id": user_id}))
        for wl in lists:
            for item in wl.get("items", []):
                t = item.get("title")
                if t:
                    watchlist_titles.append(t)
    except Exception as e:
        print(f">> [WARN] Failed to fetch watchlist for personalized recs: {e}")
        
    watched_titles = []
    try:
        history = list(watched_col.find({"user_id": user_id}))
        for item in history:
            t = item.get("title")
            if t:
                watched_titles.append(t)
    except Exception as e:
        print(f">> [WARN] Failed to fetch watched history for personalized recs: {e}")
        
    combined_titles = list(set(watchlist_titles + watched_titles))
    
    if not combined_titles:
        return {
            "recommendation_type": "personalized",
            "count": 0,
            "results": [],
            "message": "Add items to your watchlist or mark them as watched to see personalized recommendations!"
        }
        
    # Get multi-select recommendations
    results = engine.recommend_multi_select(
        titles=combined_titles,
        n=limit,
        content_type=content_type,
        exclude_genres=exclude_genres,
        user_id=user_id
    )
    
    return {
        "recommendation_type": "personalized",
        "count": len(results),
        "results": results
    }


# --- Ratings & Reviews ---

@app.get("/api/reviews/{show_id}")
async def get_reviews(show_id: str):
    """Retrieve all ratings/reviews and calculate the average rating for a show."""
    try:
        reviews = list(reviews_col.find({"show_id": show_id}, {"_id": 0}))
        
        if not reviews:
            return {
                "show_id": show_id,
                "average_rating": 0.0,
                "review_count": 0,
                "reviews": []
            }
            
        total_rating = sum(r.get("rating", 0) for r in reviews)
        avg_rating = round(total_rating / len(reviews), 1)
        
        return {
            "show_id": show_id,
            "average_rating": avg_rating,
            "review_count": len(reviews),
            "reviews": reviews
        }
    except Exception as e:
        print(f">> [ERROR] Failed to fetch reviews for '{show_id}': {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve reviews.")


@app.post("/api/reviews/submit")
async def submit_review(req: ReviewSubmitRequest):
    """Submit or update a rating and review for a show."""
    try:
        review_record = {
            "user_id": req.user_id,
            "displayName": req.displayName,
            "show_id": req.show_id,
            "rating": req.rating,
            "review_text": req.review_text,
            "timestamp": int(time.time())
        }
        
        # Upsert the review (one review per user per title)
        reviews_col.update_one(
            {"user_id": req.user_id, "show_id": req.show_id},
            {"$set": review_record},
            upsert=True
        )
        return {"success": True, "message": "Review submitted successfully!"}
    except Exception as e:
        print(f">> [ERROR] Failed to submit review: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit review.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

