#!/usr/bin/env python
"""
migrate_to_cloudinary.py - Concurrent TMDB poster migration to Cloudinary CDN.
Supports:
1. Standard migration: Upload already-resolved TMDB poster links in MongoDB to Cloudinary.
2. Complete catalog generation (--fetch-missing): Searches TMDB for titles lacking metadata,
   downloads/uploads their posters to Cloudinary, and saves full info back to MongoDB.
Uses multi-threading to speed up network requests by 10x-20x.
"""
import os
import argparse
import sys
import re
import urllib.request
import urllib.parse
import json
import ssl
from dotenv import load_dotenv
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

CLOUDINARY_URL = os.getenv("CLOUDINARY_URL", "").strip()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/").strip()
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

if not CLOUDINARY_URL:
    print(">> [ERROR] CLOUDINARY_URL environment variable is not set in backend/.env!")
    print("   Please register on Cloudinary and add your credentials before running this migration.")
    sys.exit(1)

# Import and configure Cloudinary SDK
try:
    import cloudinary
    import cloudinary.uploader
except ImportError:
    print(">> [ERROR] The 'cloudinary' package is not installed in the virtual environment!")
    print("   Please run: pip install cloudinary")
    sys.exit(1)

def clean_query(title: str) -> str:
    cleaned = title.replace("&", " ")
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned

def fetch_tmdb_data(title, media_type, release_year):
    if not TMDB_API_KEY:
        return None

    cleaned_title = clean_query(title)
    encoded_query = urllib.parse.quote(cleaned_title)
    tmdb_type = "movie" if media_type == "Movie" else "tv"
    
    # 1. Search title
    search_url = f"http://api.themoviedb.org/3/search/{tmdb_type}?api_key={TMDB_API_KEY}&query={encoded_query}"
    if release_year:
        year_key = "primary_release_year" if tmdb_type == "movie" else "first_air_date_year"
        search_url += f"&{year_key}={release_year}"
        
    try:
        req = urllib.request.Request(search_url, headers={"User-Agent": "SmartRec/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            search_data = json.loads(response.read().decode())
        results = search_data.get("results", [])
        
        if not results and release_year:
            search_url_fallback = f"http://api.themoviedb.org/3/search/{tmdb_type}?api_key={TMDB_API_KEY}&query={encoded_query}"
            req_fb = urllib.request.Request(search_url_fallback, headers={"User-Agent": "SmartRec/1.0"})
            with urllib.request.urlopen(req_fb, timeout=5) as response:
                search_data = json.loads(response.read().decode())
                results = search_data.get("results", [])
                
        if not results:
            return None
            
        best_match = results[0]
        tmdb_id = best_match["id"]
        
        # 2. Get details + videos
        details_url = f"http://api.themoviedb.org/3/{tmdb_type}/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=videos"
        req_details = urllib.request.Request(details_url, headers={"User-Agent": "SmartRec/1.0"})
        with urllib.request.urlopen(req_details, timeout=5) as response:
            details = json.loads(response.read().decode())
            
        poster_path = details.get("poster_path")
        backdrop_path = details.get("backdrop_path")
        vote_average = details.get("vote_average")
        
        videos = details.get("videos", {}).get("results", [])
        trailer_key = None
        for vid in videos:
            if vid.get("site") == "YouTube" and vid.get("type") in ["Trailer", "Teaser"]:
                trailer_key = vid.get("key")
                if vid.get("type") == "Trailer":
                    break
        trailer_url = f"https://www.youtube.com/embed/{trailer_key}" if trailer_key else None
        
        return {
            "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
            "backdrop_url": f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else None,
            "trailer_url": trailer_url,
            "vote_average": vote_average
        }
    except Exception as e:
        return {"error": str(e)}

def process_single_title(doc, titles_col, fetch_missing):
    show_id = doc.get("show_id")
    title = doc.get("title")
    media_type = doc.get("type")
    release_year = doc.get("release_year")

    tmdb_poster_url = None
    backdrop_path = None
    trailer_url = None
    vote_average = None

    if fetch_missing:
        res = fetch_tmdb_data(title, media_type, release_year)
        if res:
            if "error" in res:
                print(f"[{show_id}] [FAIL] TMDB search error for '{title}': {res['error']}")
                return "error"
            tmdb_poster_url = res.get("poster_url")
            backdrop_path = res.get("backdrop_url")
            trailer_url = res.get("trailer_url")
            vote_average = res.get("vote_average")
        else:
            # Negative cache
            negative_cache = {
                "status": "not_found",
                "poster_path": None,
                "backdrop_path": None,
                "trailer_url": None,
                "vote_average": None
            }
            titles_col.update_one({"show_id": show_id}, {"$set": {"tmdb_data": negative_cache}})
            print(f"[{show_id}] [NOT_FOUND] '{title}' not matched on TMDB.")
            return "not_found"
    else:
        tmdb_data = doc.get("tmdb_data", {})
        tmdb_poster_url = tmdb_data.get("poster_path")
        backdrop_path = tmdb_data.get("backdrop_path")
        trailer_url = tmdb_data.get("trailer_url")
        vote_average = tmdb_data.get("vote_average")

    if not tmdb_poster_url:
        if fetch_missing:
            empty_cache = {
                "status": "ok",
                "poster_path": None,
                "backdrop_path": backdrop_path,
                "trailer_url": trailer_url,
                "vote_average": round(vote_average, 1) if vote_average else None
            }
            titles_col.update_one({"show_id": show_id}, {"$set": {"tmdb_data": empty_cache}})
        print(f"[{show_id}] [SKIPPED] No poster image for '{title}'.")
        return "skipped"

    try:
        # Upload poster to Cloudinary
        upload_result = cloudinary.uploader.upload(
            tmdb_poster_url,
            folder="netflix_posters",
            public_id=show_id,
            overwrite=True
        )

        cloudinary_url = upload_result.get("secure_url")
        if not cloudinary_url:
            raise Exception("No secure_url returned from Cloudinary.")

        final_tmdb_data = {
            "status": "ok",
            "poster_path": cloudinary_url,
            "backdrop_path": backdrop_path,
            "trailer_url": trailer_url,
            "vote_average": round(vote_average, 1) if vote_average else None
        }
        
        titles_col.update_one(
            {"show_id": show_id},
            {"$set": {"tmdb_data": final_tmdb_data}}
        )
        print(f"[{show_id}] [OK] Migrated '{title}' -> {cloudinary_url}")
        return "success"
    except Exception as err:
        print(f"[{show_id}] [FAIL] Cloudinary upload failed for '{title}': {err}")
        return "error"

def run_migration(limit=None, fetch_missing=False, max_workers=10):
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client["smartrec"]
        titles_col = db["titles"]
    except Exception as e:
        print(f">> [ERROR] Could not connect to MongoDB Atlas: {e}")
        sys.exit(1)

    print(">> Connected to MongoDB Atlas successfully.")

    if fetch_missing:
        query = {
            "$or": [
                {"tmdb_data": {"$exists": False}},
                {"tmdb_data": None},
                {"tmdb_data.status": {"$in": ["error", "no_key", None]}}
            ]
        }
        print(">> Mode: Fetch missing metadata from TMDB + Upload to Cloudinary.")
    else:
        query = {
            "tmdb_data": {"$exists": True},
            "tmdb_data.status": "ok",
            "tmdb_data.poster_path": {"$regex": "^https://image.tmdb.org/"}
        }
        print(">> Mode: Migrate existing hotlinked TMDB posters to Cloudinary.")

    total_docs = titles_col.count_documents(query)
    print(f">> Found {total_docs} titles matching criteria.")

    if total_docs == 0:
        print(">> No titles need processing. All caught up!")
        return

    # Load matching documents into memory
    print(">> Loading catalog items from database...")
    docs_cursor = titles_col.find(query)
    if limit:
        docs_cursor = docs_cursor.limit(limit)
    docs = list(docs_cursor)
    
    total_to_process = len(docs)
    print(f">> Starting concurrent execution with {max_workers} threads for {total_to_process} titles...")

    success_count = 0
    error_count = 0
    skipped_count = 0
    not_found_count = 0

    print("-" * 70)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_title, doc, titles_col, fetch_missing): doc 
            for doc in docs
        }
        
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                if res == "success":
                    success_count += 1
                elif res == "skipped":
                    skipped_count += 1
                elif res == "not_found":
                    not_found_count += 1
                elif res == "error":
                    error_count += 1
            except Exception as exc:
                print(f">> Unhandled thread exception: {exc}")
                error_count += 1

            if idx % 20 == 0 or idx == total_to_process:
                print(f">> PROGRESS: {idx}/{total_to_process} items processed (Successes: {success_count}, Errors: {error_count})")

    print("-" * 70)
    print("Migration Complete Summary:")
    print(f"   Successes (Uploaded & Saved): {success_count}")
    print(f"   Skipped / No Poster URL:     {skipped_count}")
    print(f"   TMDB Matches Not Found:      {not_found_count}")
    print(f"   Failures (Errors):            {error_count}")
    print("-" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate posters to Cloudinary.")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit the number of titles to process in this run."
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Query TMDB search for titles that do not have posters cached in MongoDB yet."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Number of concurrent worker threads (default: 15)."
    )
    args = parser.parse_args()

    run_migration(limit=args.limit, fetch_missing=args.fetch_missing, max_workers=args.workers)
