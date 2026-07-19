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

from fastapi import FastAPI, HTTPException, Query
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


class CastRecommendRequest(BaseModel):
    """Request body for cast-based recommendations."""
    title: str = Field(..., description="Title to find cast-similar content for")
    n_recommendations: int = Field(10, ge=1, le=50)
    content_type: Optional[str] = None


class MultiRecommendRequest(BaseModel):
    """Request body for multi-select profile recommendations."""
    titles: List[str] = Field(..., min_length=1, max_length=10, description="List of liked titles")
    n_recommendations: int = Field(10, ge=1, le=50)
    content_type: Optional[str] = None


class GenreRecommendRequest(BaseModel):
    """Request body for genre + mood filtered recommendations."""
    genre: Optional[str] = Field(None, description="Genre to filter by")
    mood: Optional[str] = Field(None, description="Mood: feel-good, intense, dark, thought-provoking")
    content_type: Optional[str] = None
    n_recommendations: int = Field(20, ge=1, le=50)


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
        content_type=req.content_type
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
        content_type=req.content_type
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
        content_type=req.content_type
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
        n=req.n_recommendations
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
from database import users_col, watchlists_col, watched_col, titles_col

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
    name: str
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
        "uid": user["uid"],
        "email": user["email"],
        "displayName": user["displayName"],
        "provider": user["provider"],
        "avatar": user["avatar"],
        "gradient": user["gradient"]
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
        "uid": user["uid"],
        "email": user["email"],
        "displayName": user["displayName"],
        "provider": user["provider"],
        "avatar": user["avatar"],
        "gradient": user["gradient"],
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

