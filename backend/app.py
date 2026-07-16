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
# Run (development)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
