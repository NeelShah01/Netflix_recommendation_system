"""
recommender.py — ML Recommendation Engine
Smart Content Recommender

This module provides multiple recommendation strategies:
1. Content-Based (TF-IDF + Cosine Similarity on metadata soup)
2. Cast-Based (CountVectorizer similarity on cast members)
3. Genre + Mood filtering (VADER sentiment + genre matching)
4. Multi-Select Profile (average feature vector from multiple liked titles)
5. Cluster-Based Exploration (K-Means thematic clusters)

All recommendations include explanations for transparency.
"""

import os
import pickle
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


class RecommendationEngine:
    """Core recommendation engine that loads pre-computed models and serves predictions."""

    def __init__(self):
        self.df = None
        self.tfidf_matrix = None    # Sparse matrix — used for on-the-fly cosine similarity
        self.cast_matrix = None     # Sparse cast matrix — used for on-the-fly cast similarity
        self.tfidf_vectorizer = None
        self.pca_data = None
        self.title_to_idx = {}
        self.id_to_idx = {}
        self._loaded = False

    def load_models(self):
        """
        Load recommendation models and data.
        1. Load PCA/cluster data (keeps K-means labels persistent).
        2. Load local pickle files first so the server is ready instantly.
        3. If MongoDB is available, trigger a background thread to seed the database
           (if empty) and refresh/rebuild similarity models from MongoDB.
        """
        print("Loading recommendation models...")

        # Import database availability and titles collection
        from database import db_available, titles_col

        # Load PCA/cluster data (keeps K-means labels persistent)
        pca_path = os.path.join(MODELS_DIR, 'pca_data.pkl')
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                self.pca_data = pickle.load(f)
        else:
            self.pca_data = {'cluster_labels': {}, 'coords': []}

        # Step 1: Load local fallback files FIRST so the server starts instantly
        try:
            self.df = pd.read_pickle(os.path.join(MODELS_DIR, 'processed_data.pkl'))
            with open(os.path.join(MODELS_DIR, 'tfidf_matrix.pkl'), 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            cast_matrix_path = os.path.join(MODELS_DIR, 'cast_matrix.pkl')
            if os.path.exists(cast_matrix_path):
                with open(cast_matrix_path, 'rb') as f:
                    self.cast_matrix = pickle.load(f)
            
            # Build lookup indices
            self.title_to_idx = {
                title.lower(): idx for idx, title in enumerate(self.df['title'])
            }
            self.id_to_idx = {
                show_id: idx for idx, show_id in enumerate(self.df['show_id'])
            }
            self._loaded = True
            print(f"   [OK] Preloaded {len(self.df)} titles from local files. Server is starting immediately!")
        except Exception as err:
            print(f"   [CRITICAL] Failed to preload local files: {err}")

        # Step 2: If MongoDB is available, sync and rebuild in a background thread
        if db_available:
            import threading
            def bg_sync():
                try:
                    # 1. Seeding check
                    if titles_col.count_documents({}) == 0:
                        print("   [INFO] MongoDB 'titles' collection is empty. Seeding from local processed_data.pkl...")
                        pickle_path = os.path.join(MODELS_DIR, 'processed_data.pkl')
                        if os.path.exists(pickle_path):
                            df_seed = pd.read_pickle(pickle_path)
                            if 'date_added' in df_seed.columns:
                                df_seed['date_added'] = df_seed['date_added'].apply(
                                    lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
                                )
                            records = df_seed.fillna('').to_dict(orient='records')
                            for r in records:
                                if isinstance(r.get('genres_list'), np.ndarray):
                                    r['genres_list'] = r['genres_list'].tolist()
                                if isinstance(r.get('cast_list'), np.ndarray):
                                    r['cast_list'] = r['cast_list'].tolist()
                                if isinstance(r.get('country_list'), np.ndarray):
                                    r['country_list'] = r['country_list'].tolist()
                            titles_col.insert_many(records)
                            print(f"   [OK] Successfully seeded {len(records)} titles into MongoDB!")
                        else:
                            print("   [ERROR] processed_data.pkl not found. Cannot seed MongoDB catalog.")
                    
                    # 2. Rebuild similarity models from MongoDB
                    self.rebuild_models()
                    print("   [OK] Background refresh of models from MongoDB completed successfully.")
                except Exception as e:
                    print(f"   [WARN] Background MongoDB model sync failed: {e}. Keeping local/previous models.")
            
            # Start background sync thread as daemon (won't block server shutdown)
            thread = threading.Thread(target=bg_sync, name="MongoDBSyncThread", daemon=True)
            thread.start()

    def rebuild_models(self):
        """
        Loads all catalog documents from MongoDB, generates metadata soups,
        fits TF-IDF and Cast vectorizers, and updates lookup indexes.
        """
        from database import db_available, titles_col
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

        if not db_available:
            print("   [INFO] Rebuild ignored: MongoDB is not available in fail-safe mode.")
            return

        print("   Rebuilding similarity models from MongoDB...")
        
        # Load from MongoDB
        cursor = titles_col.find({}, {"_id": 0})
        df_new = pd.DataFrame(list(cursor))

        if len(df_new) == 0:
            print("   [WARN] Database contains 0 titles. Cannot fit vectorizers.")
            return

        # Ensure necessary lists exist
        df_new['director'] = df_new['director'].fillna('')
        df_new['cast'] = df_new['cast'].fillna('')
        df_new['country'] = df_new['country'].fillna('')
        df_new['rating'] = df_new['rating'].fillna('')
        df_new['description'] = df_new['description'].fillna('')
        df_new['listed_in'] = df_new['listed_in'].fillna('')

        if 'genres_list' not in df_new.columns:
            df_new['genres_list'] = df_new['listed_in'].apply(
                lambda x: [g.strip() for g in x.split(',') if g.strip()] if x else []
            )
        if 'cast_list' not in df_new.columns:
            df_new['cast_list'] = df_new['cast'].apply(
                lambda x: [c.strip() for c in x.split(',') if c.strip()] if x else []
            )
        if 'country_list' not in df_new.columns:
            df_new['country_list'] = df_new['country'].apply(
                lambda x: [c.strip() for c in x.split(',') if c.strip()] if x else []
            )
        if 'primary_country' not in df_new.columns:
            df_new['primary_country'] = df_new['country_list'].apply(
                lambda x: x[0] if x else 'Unknown'
            )
        if 'mood' not in df_new.columns:
            df_new['mood'] = 'thought-provoking'

        # Generate metadata soup dynamically (similar to preprocessing.py)
        def _build_soup(row):
            director = row['director'].lower().replace(' ', '').replace(',', ' ') if row['director'] else ''
            cast_members = row['cast_list'][:5] if row['cast_list'] else []
            cast = ' '.join([c.lower().replace(' ', '') for c in cast_members])
            genres = ' '.join([g.lower().replace(' ', '').replace('&', 'and') for g in row['genres_list']])
            country = row['primary_country'].lower().replace(' ', '') if row['primary_country'] != 'Unknown' else ''
            description = row['description'].lower() if row['description'] else ''
            description = re.sub(r'[^a-z0-9\s]', '', description)
            rating = row['rating'].lower().replace('-', '') if row['rating'] else ''

            soup_parts = [
                genres, genres, genres,   # 3x weight
                director, director,       # 2x weight
                cast, cast,               # 2x weight
                country,
                rating,
                description
            ]
            return ' '.join(part for part in soup_parts if part)

        df_new['soup'] = df_new.apply(_build_soup, axis=1)

        # Fit TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_new['soup'])

        # Fit Cast Count Vectorizer
        cast_vectorizer = CountVectorizer(stop_words='english')
        cast_soup = df_new['cast_list'].apply(lambda x: ' '.join([c.replace(' ', '').lower() for c in x]))
        cast_matrix = cast_vectorizer.fit_transform(cast_soup)

        # Fit separate Genre Vectorizer
        genre_vectorizer = TfidfVectorizer(stop_words='english')
        genre_soup = df_new['genres_list'].apply(lambda x: ' '.join([g.replace(' ', '').lower() for g in x]) if x else '')
        genre_matrix = genre_vectorizer.fit_transform(genre_soup)

        # Fit separate Director + Cast Count Vectorizer
        dir_cast_vectorizer = CountVectorizer(stop_words='english')
        dir_cast_soup = df_new.apply(
            lambda r: ' '.join(
                ([r['director'].replace(' ', '').lower()] if r['director'] else []) +
                ([c.replace(' ', '').lower() for c in r['cast_list']] if r['cast_list'] else [])
            ),
            axis=1
        )
        dir_cast_matrix = dir_cast_vectorizer.fit_transform(dir_cast_soup)

        # Fit separate Description TF-IDF Vectorizer
        desc_vectorizer = TfidfVectorizer(stop_words='english')
        desc_matrix = desc_vectorizer.fit_transform(df_new['description'].fillna(''))

        # Build lookups
        title_to_idx = {
            row['title'].lower(): idx for idx, row in df_new.iterrows()
        }
        id_to_idx = {
            row['show_id']: idx for idx, row in df_new.iterrows()
        }

        # Atomically assign to instance variables so that concurrent requests see a consistent state
        self.df = df_new
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.cast_vectorizer = cast_vectorizer
        self.cast_matrix = cast_matrix
        self.genre_vectorizer = genre_vectorizer
        self.genre_matrix = genre_matrix
        self.dir_cast_vectorizer = dir_cast_vectorizer
        self.dir_cast_matrix = dir_cast_matrix
        self.desc_vectorizer = desc_vectorizer
        self.desc_matrix = desc_matrix
        self.title_to_idx = title_to_idx
        self.id_to_idx = id_to_idx

        # Clear cached autocomplete search results
        if hasattr(self, '_titles_index_cache'):
            delattr(self, '_titles_index_cache')

        print(f"   [OK] Loaded {len(self.df)} titles from MongoDB")
        print(f"   [OK] TF-IDF matrix: {self.tfidf_matrix.shape}")
        if self.cast_matrix is not None:
            print(f"   [OK] Cast matrix: {self.cast_matrix.shape}")

    def add_or_update_title(self, show_id, title_data):
        """Insert a new title or update an existing one in MongoDB, then rebuild vector models."""
        from database import titles_col
        
        # Clean and construct the record
        record = {**title_data}
        record['show_id'] = show_id
        
        # Ensure array fields are formatted as python lists
        if 'listed_in' in record and 'genres_list' not in record:
            record['genres_list'] = [g.strip() for g in record['listed_in'].split(',') if g.strip()]
        if 'cast' in record and 'cast_list' not in record:
            record['cast_list'] = [c.strip() for c in record['cast'].split(',') if c.strip()]
        if 'country' in record and 'country_list' not in record:
            record['country_list'] = [c.strip() for c in record['country'].split(',') if c.strip()]
            record['primary_country'] = record['country_list'][0] if record['country_list'] else 'Unknown'

        # Upsert in MongoDB
        titles_col.update_one({'show_id': show_id}, {'$set': record}, upsert=True)

        # Re-initialize models
        self.rebuild_models()

    def delete_title(self, show_id):
        """Remove a title from MongoDB, then rebuild vector models."""
        from database import titles_col
        titles_col.delete_one({'show_id': show_id})
        self.rebuild_models()

    def _ensure_loaded(self):
        """Ensure models are loaded before making predictions."""
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")


    # ──────────────────────────────────────────
    # Title Lookup & Search
    # ──────────────────────────────────────────

    def search_titles(self, query, limit=10):
        """Search for titles matching a query string (case-insensitive fuzzy match)."""
        self._ensure_loaded()
        query_lower = query.lower().strip()

        if not query_lower:
            return []

        prefix_matches = []
        substring_matches = []
        fuzzy_matches = []

        for _, row in self.df.iterrows():
            title_lower = row['title'].lower()
            if title_lower.startswith(query_lower):
                prefix_matches.append(self._format_title(row))
            elif query_lower in title_lower:
                substring_matches.append(self._format_title(row))
            else:
                score = self._title_similarity(query_lower, title_lower)
                if score >= 0.35:
                    fuzzy_matches.append((score, self._format_title(row)))

        fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
        results = prefix_matches + substring_matches + [item for _, item in fuzzy_matches]
        return results[:limit]

    def get_title_details(self, show_id):
        """Get full details for a specific title by show_id."""
        self._ensure_loaded()
        if show_id not in self.id_to_idx:
            return None
        idx = self.id_to_idx[show_id]
        row = self.df.iloc[idx]
        return self._format_title_full(row)

    def get_all_genres(self):
        """Get all unique genres sorted alphabetically."""
        self._ensure_loaded()
        all_genres = set()
        for genres in self.df['genres_list']:
            all_genres.update(genres)
        return sorted(list(all_genres))

    def get_titles_index(self):
        """
        Return a compact index of all titles for client-side instant search.
        Result is cached after first call so subsequent calls are free (O(1)).
        Only includes the minimal fields needed for autocomplete rendering.
        """
        self._ensure_loaded()
        if hasattr(self, '_titles_index_cache'):
            return self._titles_index_cache

        cols = ['show_id', 'title', 'type', 'release_year', 'rating', 'listed_in']
        available = [c for c in cols if c in self.df.columns]
        records = (
            self.df[available]
            .fillna('')
            .astype({'release_year': str})
            .to_dict(orient='records')
        )
        self._titles_index_cache = records
        return records

    # ──────────────────────────────────────────
    # Recommendation Strategies
    # ──────────────────────────────────────────

    def recommend_by_title(self, title, n=10, content_type=None, exclude_genres=None, user_id=None):
        """
        Content-based recommendation: Find titles most similar to the given title.
        Computes cosine similarity on-the-fly from the sparse TF-IDF matrix
        (one row vs. all rows) — avoids loading a giant NxN matrix into memory.
        """
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        self._ensure_loaded()
        idx = self._find_title_index(title)
        if idx is None:
            return []

        source_row = self.df.iloc[idx]

        # Compute similarity of this one title against all others (efficient sparse op)
        if hasattr(self, 'genre_matrix') and hasattr(self, 'dir_cast_matrix') and hasattr(self, 'desc_matrix'):
            q_genre = self.genre_matrix[idx]
            q_dir_cast = self.dir_cast_matrix[idx]
            q_desc = self.desc_matrix[idx]
            
            genre_scores = cos_sim(q_genre, self.genre_matrix).flatten()
            dir_cast_scores = cos_sim(q_dir_cast, self.dir_cast_matrix).flatten()
            desc_scores = cos_sim(q_desc, self.desc_matrix).flatten()
            
            scores = (0.50 * genre_scores) + (0.25 * dir_cast_scores) + (0.25 * desc_scores)
        else:
            query_vec = self.tfidf_matrix[idx]  # shape: (1, n_features)
            scores = cos_sim(query_vec, self.tfidf_matrix).flatten()  # shape: (n_titles,)

        sim_scores = list(enumerate(scores))

        # Retrieve user ratings genre boosts
        boosts = self.get_user_genre_boosts(user_id)

        # Exclude and boost
        boosted_scores = []
        for i, score in sim_scores:
            if i == idx:
                continue
            rec_row = self.df.iloc[i]
            
            # 1. Content Type Filter
            if content_type and rec_row['type'].lower() != content_type.lower():
                continue
                
            # 2. Exclude Genres Filter
            if exclude_genres:
                rec_genres_lower = {g.lower().strip() for g in rec_row.get("genres_list", [])}
                exclude_genres_lower = {g.lower().strip() for g in exclude_genres}
                if rec_genres_lower.intersection(exclude_genres_lower):
                    continue
                    
            # 3. Apply Genre Rating Boost
            boost = 1.0
            if boosts:
                for g in rec_row.get("genres_list", []):
                    g_lower = g.lower().strip()
                    if g_lower in boosts:
                        boost = max(boost, boosts[g_lower])
                        
            boosted_scores.append((i, float(score) * boost))

        # Sort by boosted score
        boosted_scores = sorted(boosted_scores, key=lambda x: x[1], reverse=True)
        top_scores = boosted_scores[:n]

        results = []
        for i, score in top_scores:
            rec_row = self.df.iloc[i]
            explanation = self._generate_explanation(source_row, rec_row, score)
            result = self._format_title(rec_row)
            result['similarity_score'] = round(float(score), 4)
            result['explanation'] = explanation
            result['match_breakdown'] = self._get_match_breakdown([source_row], rec_row)
            results.append(result)

        return results

    def recommend_by_cast(self, title, n=10, content_type=None, exclude_genres=None, user_id=None):
        """
        Cast-based recommendation: Find titles with the most similar cast members.
        Computes cast similarity on-the-fly from the sparse cast matrix
        (CountVectorizer) — avoids loading a giant NxN matrix into memory.
        """
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        self._ensure_loaded()

        if self.cast_matrix is None:
            return []  # cast_matrix.pkl not available

        idx = self._find_title_index(title)
        if idx is None:
            return []

        source_row = self.df.iloc[idx]
        source_cast = set(source_row['cast_list'][:10])

        # Compute cast similarity on-the-fly for this one title
        query_vec = self.cast_matrix[idx]  # shape: (1, n_cast_features)
        scores = cos_sim(query_vec, self.cast_matrix).flatten()  # shape: (n_titles,)

        sim_scores = list(enumerate(scores))

        # Retrieve user ratings genre boosts
        boosts = self.get_user_genre_boosts(user_id)

        # Exclude and boost
        boosted_scores = []
        for i, score in sim_scores:
            if i == idx:
                continue
            rec_row = self.df.iloc[i]
            
            # 1. Content Type Filter
            if content_type and rec_row['type'].lower() != content_type.lower():
                continue
                
            # 2. Exclude Genres Filter
            if exclude_genres:
                rec_genres_lower = {g.lower().strip() for g in rec_row.get("genres_list", [])}
                exclude_genres_lower = {g.lower().strip() for g in exclude_genres}
                if rec_genres_lower.intersection(exclude_genres_lower):
                    continue
                    
            # 3. Apply Genre Rating Boost
            boost = 1.0
            if boosts:
                for g in rec_row.get("genres_list", []):
                    g_lower = g.lower().strip()
                    if g_lower in boosts:
                        boost = max(boost, boosts[g_lower])
                        
            # Only include results with non-zero cast similarity
            if score > 0:
                boosted_scores.append((i, float(score) * boost))

        # Sort by boosted score
        boosted_scores = sorted(boosted_scores, key=lambda x: x[1], reverse=True)
        top_scores = boosted_scores[:n]

        results = []
        for i, score in top_scores:
            rec_row = self.df.iloc[i]
            # Build cast-specific explanation
            rec_cast = set(rec_row['cast_list'][:10])
            shared_cast = source_cast.intersection(rec_cast)
            if shared_cast:
                cast_names = ', '.join(list(shared_cast)[:3])
                explanation = f"Shares cast members: {cast_names}"
            else:
                explanation = "Similar cast profile"

            result = self._format_title(rec_row)
            result['similarity_score'] = round(float(score), 4)
            result['explanation'] = explanation
            result['match_breakdown'] = self._get_match_breakdown([source_row], rec_row)
            results.append(result)

        return results

    def recommend_by_genre_mood(self, genre=None, mood=None, content_type=None, n=20, exclude_genres=None, user_id=None):
        """
        Filter and recommend by genre and/or mood (sentiment).
        Mood values: 'feel-good', 'intense', 'dark', 'thought-provoking'
        """
        self._ensure_loaded()
        filtered = self.df.copy()

        # Filter by content type
        if content_type:
            filtered = filtered[filtered['type'].str.lower() == content_type.lower()]

        # Filter by genre
        if genre:
            genre_lower = genre.lower()
            filtered = filtered[
                filtered['genres_list'].apply(
                    lambda genres: any(g.lower() == genre_lower for g in genres)
                )
            ]

        # Filter by exclude_genres
        if exclude_genres:
            exclude_genres_lower = {g.lower().strip() for g in exclude_genres}
            filtered = filtered[
                filtered['genres_list'].apply(
                    lambda genres: not any(g.lower().strip() in exclude_genres_lower for g in genres)
                )
            ]

        # Filter by mood
        if mood:
            filtered = filtered[filtered['mood'] == mood.lower()]

        # Sort by release year (recent first) and sentiment score
        filtered = filtered.sort_values(
            by=['release_year', 'sentiment_compound'],
            ascending=[False, False]
        )

        results = []
        for _, row in filtered.head(n).iterrows():
            result = self._format_title(row)
            result['similarity_score'] = 1.0
            parts = []
            if genre:
                parts.append(f"Genre: {genre}")
            if mood:
                mood_labels = {
                    'feel-good': 'Feel-Good 😊',
                    'intense': 'Intense 🔥',
                    'dark': 'Dark 🌑',
                    'thought-provoking': 'Thought-Provoking 🧠'
                }
                parts.append(f"Mood: {mood_labels.get(mood, mood)}")
            result['explanation'] = ' | '.join(parts) if parts else 'Matches your filters'
            result['match_breakdown'] = {
                "genres": [genre] if genre else [],
                "cast": [],
                "director": None,
                "country": None,
                "mood": mood
            }
            results.append(result)

        return results

    def recommend_multi_select(self, titles, n=10, content_type=None, exclude_genres=None, user_id=None):
        """
        Multi-select profile recommendation: Given multiple liked titles,
        compute an average feature vector and find nearest neighbors.
        """
        self._ensure_loaded()
        indices = []
        for title in titles:
            idx = self._find_title_index(title)
            if idx is not None:
                indices.append(idx)

        if not indices:
            return []

        # Compute candidate recommendations for each liked item individually using round-robin interleaving.
        # This guarantees representation of all user interests (e.g. movies vs TV shows) and
        # prevents rare items (with high IDF weights) from completely hijacking the taste profile.
        from sklearn.metrics.pairwise import cosine_similarity as cs
        
        # Retrieve user ratings genre boosts
        boosts = self.get_user_genre_boosts(user_id)

        candidates_by_item = []
        for src_idx in indices:
            # Compute similarities for this specific liked item against all items
            if hasattr(self, 'genre_matrix') and hasattr(self, 'dir_cast_matrix') and hasattr(self, 'desc_matrix'):
                q_genre = self.genre_matrix[src_idx]
                q_dir_cast = self.dir_cast_matrix[src_idx]
                q_desc = self.desc_matrix[src_idx]
                
                genre_scores = cs(q_genre, self.genre_matrix).flatten()
                dir_cast_scores = cs(q_dir_cast, self.dir_cast_matrix).flatten()
                desc_scores = cs(q_desc, self.desc_matrix).flatten()
                
                scores = (0.50 * genre_scores) + (0.25 * dir_cast_scores) + (0.25 * desc_scores)
            else:
                query_vec = self.tfidf_matrix[src_idx]
                scores = cs(query_vec, self.tfidf_matrix).flatten()

            # Filter and boost
            item_candidates = []
            for i, score in enumerate(scores):
                if i in indices:
                    continue
                rec_row = self.df.iloc[i]
                
                # 1. Content Type Filter
                if content_type and rec_row['type'].lower() != content_type.lower():
                    continue
                    
                # 2. Exclude Genres Filter
                if exclude_genres:
                    rec_genres_lower = {g.lower().strip() for g in rec_row.get("genres_list", [])}
                    exclude_genres_lower = {g.lower().strip() for g in exclude_genres}
                    if rec_genres_lower.intersection(exclude_genres_lower):
                        continue
                        
                # 3. Apply Genre Rating Boost
                boost = 1.0
                if boosts:
                    for g in rec_row.get("genres_list", []):
                        g_lower = g.lower().strip()
                        if g_lower in boosts:
                            boost = max(boost, boosts[g_lower])
                            
                item_candidates.append((float(score) * boost, i, src_idx))
            
            # Sort this item's candidates by score descending
            item_candidates = sorted(item_candidates, key=lambda x: x[0], reverse=True)
            candidates_by_item.append(item_candidates)

        # Interleave recommendations across liked items (Round-Robin)
        selected_candidates = []
        selected_indices = set()
        
        for step in range(n):
            if len(selected_candidates) >= n:
                break
            
            any_added = False
            for item_candidates in candidates_by_item:
                if step < len(item_candidates):
                    score, cand_idx, src_idx = item_candidates[step]
                    if cand_idx not in selected_indices:
                        selected_candidates.append((score, cand_idx, src_idx))
                        selected_indices.add(cand_idx)
                        any_added = True
                        if len(selected_candidates) >= n:
                            break
            
            if not any_added:
                break

        # Sort final interleaved results by score descending
        selected_candidates = sorted(selected_candidates, key=lambda x: x[0], reverse=True)

        # Format output
        liked_rows = [self.df.iloc[src_idx] for src_idx in indices]
        results = []
        for score, cand_idx, src_idx in selected_candidates:
            rec_row = self.df.iloc[cand_idx]
            liked_title = self.df.iloc[src_idx]['title']
            
            result = self._format_title(rec_row)
            result['similarity_score'] = round(float(score), 4)
            result['explanation'] = f"Matches your taste profile (closest to: {liked_title})"
            result['match_breakdown'] = self._get_match_breakdown(liked_rows, rec_row)
            results.append(result)

        return results

    # ──────────────────────────────────────────
    # Clusters & Stats
    # ──────────────────────────────────────────

    def get_cluster_data(self):
        """Get PCA-reduced cluster data for visualization."""
        self._ensure_loaded()
        return self.pca_data

    def get_cluster_titles(self, cluster_id, n=20):
        """Get titles belonging to a specific cluster."""
        self._ensure_loaded()
        cluster_df = self.df[self.df['cluster'] == cluster_id]
        cluster_df = cluster_df.sort_values('release_year', ascending=False)

        results = []
        for _, row in cluster_df.head(n).iterrows():
            result = self._format_title(row)
            label = self.pca_data['cluster_labels'].get(cluster_id, f'Cluster {cluster_id}')
            result['similarity_score'] = 1.0
            result['explanation'] = f"Part of theme: {label}"
            results.append(result)

        return results

    def get_stats(self):
        """Get dataset statistics for the dashboard."""
        self._ensure_loaded()

        # Genre distribution
        all_genres = [g for gl in self.df['genres_list'] for g in gl]
        genre_counts = pd.Series(all_genres).value_counts().head(15).to_dict()

        # Year distribution
        year_counts = self.df['release_year'].value_counts().sort_index().tail(20).to_dict()

        # Type distribution
        type_counts = self.df['type'].value_counts().to_dict()

        # Rating distribution
        rating_counts = self.df['rating'].value_counts().head(10).to_dict()

        # Mood distribution
        mood_counts = self.df['mood'].value_counts().to_dict()

        # Country distribution
        country_counts = self.df['primary_country'].value_counts().head(10).to_dict()

        # Top directors
        directors = self.df[self.df['director'] != '']['director'].value_counts().head(10).to_dict()

        return {
            'total_titles': len(self.df),
            'total_movies': len(self.df[self.df['type'] == 'Movie']),
            'total_shows': len(self.df[self.df['type'] == 'TV Show']),
            'total_genres': len(set(all_genres)),
            'total_countries': len(country_counts),
            'genre_distribution': genre_counts,
            'year_distribution': {str(k): v for k, v in year_counts.items()},
            'type_distribution': type_counts,
            'rating_distribution': rating_counts,
            'mood_distribution': mood_counts,
            'country_distribution': country_counts,
            'top_directors': directors
        }

    def get_trending(self, n=20, content_type=None):
        """Get trending/recent content (most recently added)."""
        self._ensure_loaded()
        filtered = self.df.copy()

        if content_type:
            filtered = filtered[filtered['type'].str.lower() == content_type.lower()]

        # Sort by date_added (most recent first), then release_year
        filtered = filtered.sort_values(
            by=['date_added', 'release_year'],
            ascending=[False, False],
            na_position='last'
        )

        results = []
        for _, row in filtered.head(n).iterrows():
            result = self._format_title(row)
            result['similarity_score'] = 1.0
            result['explanation'] = 'Recently added to Netflix'
            results.append(result)

        return results

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _find_title_index(self, title):
        """Find the dataframe index for a given title (case-insensitive)."""
        title_lower = title.lower().strip()

        if not title_lower:
            return None

        if title_lower in self.title_to_idx:
            return self.title_to_idx[title_lower]

        matches = []
        for stored_title, idx in self.title_to_idx.items():
            if title_lower in stored_title:
                matches.append((len(stored_title), idx))
            else:
                score = self._title_similarity(title_lower, stored_title)
                if score >= 0.35:
                    matches.append((score, idx))

        if not matches:
            return None

        if isinstance(matches[0][0], float):
            best = max(matches, key=lambda x: x[0])
            return best[1]

        return min(matches, key=lambda x: x[0])[1]

    def _title_similarity(self, query, title):
        """Return a lightweight similarity score for title matching."""
        if not query or not title:
            return 0.0
        if query == title:
            return 1.0
        if query in title or title in query:
            return 0.9

        query_tokens = set(re.findall(r'[a-z0-9]+', query))
        title_tokens = set(re.findall(r'[a-z0-9]+', title))
        token_overlap = 0.0
        if query_tokens and title_tokens:
            token_overlap = len(query_tokens & title_tokens) / len(query_tokens | title_tokens)

        return max(token_overlap, SequenceMatcher(None, query, title).ratio())

    def _format_title(self, row):
        """Format a dataframe row into an API-friendly dict (summary)."""
        date_str = ''
        if pd.notna(row.get('date_added')):
            try:
                date_str = row['date_added'].strftime('%B %d, %Y')
            except Exception:
                date_str = str(row['date_added'])

        return {
            'show_id': row['show_id'],
            'type': row['type'],
            'title': row['title'],
            'director': row['director'] if row['director'] else 'Not Available',
            'cast': ', '.join(row['cast_list'][:5]) if row['cast_list'] else 'Not Available',
            'country': row['primary_country'],
            'date_added': date_str,
            'release_year': int(row['release_year']) if pd.notna(row['release_year']) else None,
            'rating': row['rating'] if row['rating'] else 'Not Rated',
            'duration': row['duration'],
            'listed_in': row['listed_in'],
            'description': row['description'],
            'mood': row.get('mood', 'thought-provoking'),
            'cluster': int(row.get('cluster', 0))
        }

    def _format_title_full(self, row):
        """Format a dataframe row with full details (for detail view)."""
        result = self._format_title(row)
        result['full_cast'] = ', '.join(row['cast_list']) if row['cast_list'] else 'Not Available'
        result['all_countries'] = ', '.join(row['country_list']) if row['country_list'] else 'Not Available'
        result['genres'] = row['genres_list']
        result['sentiment_score'] = round(float(row.get('sentiment_compound', 0)), 3)
        result['cluster_id'] = int(row.get('cluster', 0))
        return result

    def _generate_explanation(self, source_row, rec_row, score):
        """
        Generate a human-readable explanation for why a title was recommended.
        Analyzes feature overlaps between source and recommended titles.
        """
        reasons = []

        # Check shared genres
        source_genres = set(source_row['genres_list'])
        rec_genres = set(rec_row['genres_list'])
        shared_genres = source_genres.intersection(rec_genres)
        if shared_genres:
            genres_str = ', '.join(list(shared_genres)[:3])
            reasons.append(f"same genres ({genres_str})")

        # Check shared director
        if (source_row['director'] and rec_row['director'] and
                source_row['director'] == rec_row['director']):
            reasons.append(f"directed by {rec_row['director']}")

        # Check shared cast members
        source_cast = set(source_row['cast_list'][:10])
        rec_cast = set(rec_row['cast_list'][:10])
        shared_cast = source_cast.intersection(rec_cast)
        if shared_cast:
            cast_str = ', '.join(list(shared_cast)[:2])
            reasons.append(f"features {cast_str}")

        # Check same country
        if (source_row['primary_country'] != 'Unknown' and
                source_row['primary_country'] == rec_row['primary_country']):
            reasons.append(f"from {rec_row['primary_country']}")

        # Check similar mood
        if source_row.get('mood') == rec_row.get('mood'):
            mood_labels = {
                'feel-good': 'feel-good',
                'intense': 'intense',
                'dark': 'dark',
                'thought-provoking': 'thought-provoking'
            }
            mood = mood_labels.get(rec_row.get('mood', ''), '')
            if mood:
                reasons.append(f"{mood} tone")

        if reasons:
            return f"Recommended because: {', '.join(reasons)}"
        else:
            return f"Similar content profile (score: {round(float(score), 2)})"

    def get_user_genre_boosts(self, user_id):
        """
        Query MongoDB reviews collection to find user's highly rated titles (4 or 5 stars).
        Collect the genres from those titles, and return a dictionary of genre weights (boost factors).
        """
        from database import db_available, reviews_col
        if not db_available or reviews_col is None or not user_id:
            return {}
        
        try:
            # Get reviews with rating >= 4 for this user
            user_ratings = list(reviews_col.find({"user_id": user_id, "rating": {"$gte": 4}}))
            if not user_ratings:
                return {}
            
            boosted_genres = {}
            for r in user_ratings:
                show_id = r.get("show_id")
                if show_id in self.id_to_idx:
                    idx = self.id_to_idx[show_id]
                    row = self.df.iloc[idx]
                    genres = row.get("genres_list", [])
                    weight = 1.15 if r.get("rating") == 5 else 1.08
                    for genre in genres:
                        genre_lower = genre.lower().strip()
                        boosted_genres[genre_lower] = max(boosted_genres.get(genre_lower, 1.0), weight)
            return boosted_genres
        except Exception as e:
            print(f">> [WARN] Failed to fetch rating boosts: {e}")
            return {}

    def _get_match_breakdown(self, source_rows, rec_row):
        """
        Compute overlapping attributes between source titles and the recommended title.
        """
        profile_genres = set()
        profile_cast = set()
        profile_directors = set()
        profile_countries = set()
        profile_moods = set()
        
        for row in source_rows:
            profile_genres.update(row.get('genres_list', []))
            profile_cast.update(row.get('cast_list', [])[:10])
            if row.get('director') and row.get('director') not in ['Unknown', 'Not Available']:
                profile_directors.add(row.get('director'))
            if row.get('primary_country') and row.get('primary_country') not in ['Unknown', 'Not Available']:
                profile_countries.add(row.get('primary_country'))
            if row.get('mood'):
                profile_moods.add(row.get('mood'))
                
        rec_genres = set(rec_row.get('genres_list', []))
        rec_cast = set(rec_row.get('cast_list', [])[:10])
        rec_director = rec_row.get('director')
        rec_country = rec_row.get('primary_country')
        rec_mood = rec_row.get('mood')
        
        shared_genres = list(profile_genres.intersection(rec_genres))
        shared_cast = list(profile_cast.intersection(rec_cast))
        shared_director = rec_director if rec_director in profile_directors else None
        shared_country = rec_country if rec_country in profile_countries else None
        shared_mood = rec_mood if rec_mood in profile_moods else None
        
        return {
            "genres": [g for g in shared_genres if g][:2],
            "cast": [c for c in shared_cast if c][:2],
            "director": shared_director if shared_director not in ['Unknown', 'Not Available', None] else None,
            "country": shared_country if shared_country not in ['Unknown', 'Not Available', None] else None,
            "mood": shared_mood
        }
