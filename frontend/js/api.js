/**
 * api.js Smart Content Recommender API Client
 * Handles all communication with the FastAPI backend at localhost:8000
 */

const API = (() => {
  const BASE_URL = (typeof window !== 'undefined' && window.location && window.location.origin) ? window.location.origin : 'http://127.0.0.1:8002';
  const cache = new Map();
  const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

  /* Helpers */

  function cacheKey(url, body) {
    return body ? `${url}::${JSON.stringify(body)}` : url;
  }

  function getCached(key) {
    const entry = cache.get(key);
    if (!entry) return null;
    if (Date.now() - entry.ts > CACHE_TTL) { cache.delete(key); return null; }
    return entry.data;
  }

  function setCache(key, data) {
    cache.set(key, { data, ts: Date.now() });
    // Evict oldest if cache grows too large
    if (cache.size > 200) {
      const oldest = cache.keys().next().value;
      cache.delete(oldest);
    }
  }

  async function request(endpoint, options = {}) {
    const url = `${BASE_URL}${endpoint}`;
    const key = cacheKey(url, options.body);

    // Use cache for GET requests
    if (!options.method || options.method === 'GET') {
      const cached = getCached(key);
      if (cached) return cached;
    }

    const config = {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    };

    try {
      const res = await fetch(url, config);
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();

      // Cache successful responses
      setCache(key, data);
      return data;
    } catch (err) {
      if (err.name === 'AbortError') throw err;
      console.error(`[API] ${endpoint}:`, err);
      throw err;
    }
  }

  /* Debounce Utility */

  function debounce(fn, ms = 300) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      return new Promise((resolve) => {
        timer = setTimeout(() => resolve(fn(...args)), ms);
      });
    };
  }

  /* Public API */

  /** Search / autocomplete titles */
  async function searchTitles(query, signal) {
    if (!query || query.trim().length < 2) return [];
    const params = new URLSearchParams({ q: query.trim() });
    return request(`/api/titles?${params}`, { signal });
  }

  const debouncedSearch = debounce(searchTitles, 280);

  /** Get a single title by show_id */
  async function getTitle(showId) {
    return request(`/api/title/${encodeURIComponent(showId)}`);
  }

  /** List all genres */
  async function getGenres() {
    return request('/api/genres');
  }

  /** Get dataset statistics */
  async function getStats() {
    return request('/api/stats');
  }

  /** Get trending / recent content */
  async function getTrending() {
    return request('/api/trending');
  }

  /** Content-based recommendations for a single title */
  async function getRecommendations(title, n = 12) {
    return request('/api/recommend', {
      method: 'POST',
      body: JSON.stringify({ title, n_recommendations: n }),
    });
  }

  /** Blended recommendations from multiple titles */
  async function getMultiRecommendations(titles, n = 15) {
    return request('/api/recommend/multi', {
      method: 'POST',
      body: JSON.stringify({ titles, n_recommendations: n }),
    });
  }

  /** Genre + mood based recommendations */
  async function getGenreRecommendations(genre, mood = null, contentType = null, n = 15) {
    const body = { genre, n_recommendations: n };
    if (mood) body.mood = mood;
    if (contentType) body.content_type = contentType;
    return request('/api/recommend/genre', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  /** Cast-based recommendations */
  async function getCastRecommendations(title, n = 12) {
    return request('/api/recommend/cast', {
      method: 'POST',
      body: JSON.stringify({ title, n_recommendations: n }),
    });
  }

  /* Expose */

  return {
    searchTitles,
    debouncedSearch,
    getTitle,
    getGenres,
    getStats,
    getTrending,
    getRecommendations,
    getMultiRecommendations,
    getGenreRecommendations,
    getCastRecommendations,
    debounce,
  };
})();


