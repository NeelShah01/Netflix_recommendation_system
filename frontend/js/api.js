/**
 * api.js — Smart Content Recommender API Client
 * Handles all communication with the FastAPI backend.
 *
 * KEY OPTIMIZATION: Client-Side Instant Search
 * On startup, the full title index (~8,000 titles, compact fields only) is
 * fetched ONCE from /api/titles/index and stored in memory. All autocomplete
 * queries are then resolved locally in <5ms with zero network latency.
 * The backend /api/titles endpoint is kept as a fallback.
 */

const API = (() => {
  const BASE_URL = (typeof window !== 'undefined' && window.location && window.location.origin && !window.location.origin.startsWith('file') && !window.location.origin.includes('5500'))
    ? window.location.origin
    : 'http://127.0.0.1:8000';

  const cache = new Map();
  const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

  /* ═══════════════════ HTTP CACHE ═══════════════════ */

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
    if (cache.size > 200) {
      const oldest = cache.keys().next().value;
      cache.delete(oldest);
    }
  }

  async function request(endpoint, options = {}) {
    const url = `${BASE_URL}${endpoint}`;
    const key = cacheKey(url, options.body);

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
      setCache(key, data);
      return data;
    } catch (err) {
      if (err.name === 'AbortError') throw err;
      console.error(`[API] ${endpoint}:`, err);
      throw err;
    }
  }

  /* ═══════════════════ CLIENT-SIDE SEARCH INDEX ═══════════════════ */

  // The full title list loaded once on startup
  let _titleIndex = null;        // Array<{show_id, title, type, release_year, rating, listed_in}>
  let _titleIndexLoading = null; // Promise while loading

  /**
   * Load the compact title index from the backend once.
   * Subsequent calls return the cached result immediately.
   */
  async function loadTitleIndex() {
    if (_titleIndex) return _titleIndex;
    if (_titleIndexLoading) return _titleIndexLoading;

    _titleIndexLoading = request('/api/titles/index').then(data => {
      _titleIndex = Array.isArray(data) ? data : [];
      // Pre-lowercase title for fast matching
      _titleIndex.forEach(item => {
        item._titleLower = (item.title || '').toLowerCase();
      });
      console.log(`[Search] Loaded ${_titleIndex.length} titles into client-side index`);
      _titleIndexLoading = null;
      return _titleIndex;
    }).catch(err => {
      console.warn('[Search] Could not load title index, falling back to server search:', err);
      _titleIndex = [];           // empty but non-null, prevents re-fetch loops
      _titleIndexLoading = null;
      return [];
    });

    return _titleIndexLoading;
  }

  /**
   * Instant client-side search — no network request.
   * Priority: prefix match → substring match → word-start match
   * Returns up to `limit` results sorted by match quality.
   */
  function searchLocal(query, limit = 8) {
    if (!_titleIndex || !query) return null; // null = index not ready yet
    const q = query.toLowerCase().trim();
    if (q.length < 2) return [];

    const prefix = [];
    const wordStart = [];
    const substring = [];

    for (const item of _titleIndex) {
      const t = item._titleLower;
      if (t.startsWith(q)) {
        prefix.push(item);
      } else if (t.includes(` ${q}`) || t.includes(`:${q}`)) {
        wordStart.push(item);
      } else if (t.includes(q)) {
        substring.push(item);
      }
      if (prefix.length + wordStart.length + substring.length >= limit * 3) break;
    }

    return [...prefix, ...wordStart, ...substring].slice(0, limit);
  }

  /**
   * Primary search function — instant client-side when index is ready,
   * falls back to backend API if not yet loaded.
   */
  async function searchTitles(query, signal) {
    if (!query || query.trim().length < 2) return [];

    // Try instant local search first
    const localResults = searchLocal(query, 8);
    if (localResults !== null) return localResults; // index is ready, return immediately

    // Index not yet loaded — use server as fallback (only happens in first ~500ms)
    const params = new URLSearchParams({ q: query.trim() });
    return request(`/api/titles?${params}`, { signal });
  }

  /* ═══════════════════ DEBOUNCE ═══════════════════ */

  function debounce(fn, ms = 300) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      return new Promise((resolve) => {
        timer = setTimeout(() => resolve(fn(...args)), ms);
      });
    };
  }

  // Reduced debounce to 120ms — fast enough to feel instant, prevents excess calls
  const debouncedSearch = debounce(searchTitles, 120);

  /* ═══════════════════ PUBLIC API METHODS ═══════════════════ */

  async function getTitle(showId) {
    return request(`/api/title/${encodeURIComponent(showId)}`);
  }

  async function getGenres() {
    return request('/api/genres');
  }

  async function getStats() {
    return request('/api/stats');
  }

  async function getTrending() {
    return request('/api/trending');
  }

  async function getRecommendations(title, n = 12) {
    return request('/api/recommend', {
      method: 'POST',
      body: JSON.stringify({ title, n_recommendations: n }),
    });
  }

  async function getMultiRecommendations(titles, n = 15) {
    return request('/api/recommend/multi', {
      method: 'POST',
      body: JSON.stringify({ titles, n_recommendations: n }),
    });
  }

  async function getGenreRecommendations(genre, mood = null, contentType = null, n = 15) {
    const body = { genre, n_recommendations: n };
    if (mood) body.mood = mood;
    if (contentType) body.content_type = contentType;
    return request('/api/recommend/genre', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  async function getCastRecommendations(title, n = 12) {
    return request('/api/recommend/cast', {
      method: 'POST',
      body: JSON.stringify({ title, n_recommendations: n }),
    });
  }

  // --- Auth APIs ---
  async function authRegister(email, password, displayName) {
    return request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, displayName }),
    });
  }

  async function authLogin(email, password) {
    return request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  async function authGoogle(googlePayload) {
    return request('/api/auth/google', {
      method: 'POST',
      body: JSON.stringify({
        email: googlePayload.email,
        name: googlePayload.name,
        picture: googlePayload.picture,
        sub: googlePayload.sub,
      }),
    });
  }

  // --- Watchlist APIs ---
  async function getWatchlists(userId) {
    return request(`/api/watchlist?user_id=${encodeURIComponent(userId)}`);
  }

  async function createWatchlist(userId, name) {
    return request('/api/watchlist/create', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, name }),
    });
  }

  async function deleteWatchlist(userId, name) {
    return request(`/api/watchlist?user_id=${encodeURIComponent(userId)}&name=${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
  }

  async function addWatchlistItem(userId, listName, showId, title, type) {
    return request('/api/watchlist/item/add', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, list_name: listName, show_id: showId, title, type }),
    });
  }

  async function removeWatchlistItem(userId, listName, showId, title, type) {
    return request('/api/watchlist/item/remove', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, list_name: listName, show_id: showId, title, type }),
    });
  }

  // --- Watched History APIs ---
  async function getWatched(userId) {
    return request(`/api/watched?user_id=${encodeURIComponent(userId)}`);
  }

  async function addWatched(userId, showId, title, type) {
    return request('/api/watched/add', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, show_id: showId, title, type }),
    });
  }

  async function removeWatched(userId, showId, title, type) {
    return request('/api/watched/remove', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, show_id: showId, title, type }),
    });
  }

  // --- Catalog Editing APIs (Dynamic schemas) ---
  async function addCatalogItem(itemData) {
    return request('/api/catalog/item', {
      method: 'POST',
      body: JSON.stringify(itemData),
    });
  }

  async function updateCatalogItem(showId, itemData) {
    return request(`/api/catalog/item/${encodeURIComponent(showId)}`, {
      method: 'PUT',
      body: JSON.stringify(itemData),
    });
  }

  async function deleteCatalogItem(showId) {
    return request(`/api/catalog/item/${encodeURIComponent(showId)}`, {
      method: 'DELETE',
    });
  }

  /* ═══════════════════ EXPOSE ═══════════════════ */

  return {
    searchTitles,
    debouncedSearch,
    loadTitleIndex,
    getTitle,
    getGenres,
    getStats,
    getTrending,
    getRecommendations,
    getMultiRecommendations,
    getGenreRecommendations,
    getCastRecommendations,
    // New MongoDB linked methods
    authRegister,
    authLogin,
    authGoogle,
    getWatchlists,
    createWatchlist,
    deleteWatchlist,
    addWatchlistItem,
    removeWatchlistItem,
    getWatched,
    addWatched,
    removeWatched,
    addCatalogItem,
    updateCatalogItem,
    deleteCatalogItem,
    debounce,
  };
})();

