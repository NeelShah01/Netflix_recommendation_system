/**
 * watchlist.js — Smart Content Recommender Watchlist & Watched Manager
 * Integrates with FastAPI + MongoDB backend.
 * 
 * HYBRID CACHING STRATEGY:
 * Loads all user-specific data from MongoDB at login/startup into an in-memory cache.
 * All queries (e.g., isInAnyList, isWatched) run synchronously against the cache.
 * Mutations (add, create, delete, remove) are sent to MongoDB and sync back to cache.
 */

const WatchlistManager = (() => {
  'use strict';

  /* ═══════════════════ CACHE STATE ═══════════════════ */
  const cache = {
    lists: {},        // { listName: Array<{show_id, title, type}> }
    watched: [],      // Array<{show_id, title, type, watchedAt}>
    loading: false
  };

  /* ═══════════════════ SYNC FROM BACKEND ═══════════════════ */

  /**
   * Load watchlists and watched history from MongoDB for the logged-in user.
   * Called automatically by app.js on startup or login.
   */
  async function loadFromServer() {
    const user = Auth.getUser();
    if (!user) {
      cache.lists = {};
      cache.watched = [];
      return;
    }

    cache.loading = true;
    try {
      const [wlData, watchedData] = await Promise.all([
        API.getWatchlists(user.uid),
        API.getWatched(user.uid)
      ]);

      cache.lists = wlData || {};
      cache.watched = watchedData || [];
      console.log('[WatchlistManager] Synced data from MongoDB', cache);
    } catch (err) {
      console.error('[WatchlistManager] Failed to sync data from backend:', err);
    } finally {
      cache.loading = false;
    }
  }

  /* ═══════════════════════════════════════════════════
     WATCHLIST MANAGER
     ═══════════════════════════════════════════════════ */

  /** Get all watchlists formatted as [{ id, name, items: [] }]. */
  function getLists() {
    return Object.entries(cache.lists).map(([name, items]) => ({
      id: name, // use name as ID for simplicity
      name: name,
      items: items || []
    }));
  }

  /** Create a new watchlist. */
  async function createList(name) {
    const user = Auth.getUser();
    if (!user) return null;
    const trimmed = name ? name.trim() : '';
    if (!trimmed) return null;

    // Check duplicate
    if (cache.lists[trimmed]) {
      return { error: 'A list with that name already exists.' };
    }

    try {
      await API.createWatchlist(user.uid, trimmed);
      // Update cache
      cache.lists[trimmed] = [];
      document.dispatchEvent(new CustomEvent('watchlistUpdated'));
      return { name: trimmed, items: [] };
    } catch (err) {
      return { error: err.message || 'Failed to create watchlist.' };
    }
  }

  /** Delete a watchlist. */
  async function deleteList(name) {
    const user = Auth.getUser();
    if (!user) return false;

    try {
      await API.deleteWatchlist(user.uid, name);
      delete cache.lists[name];
      document.dispatchEvent(new CustomEvent('watchlistUpdated'));
      return true;
    } catch (err) {
      console.error('[Watchlist] Delete list error:', err);
      return false;
    }
  }

  /** Add an item to a watchlist. */
  async function addToList(listName, item) {
    const user = Auth.getUser();
    if (!user) return { error: 'Not logged in.' };

    const items = cache.lists[listName] || [];
    const isDuplicate = items.some(i => i.show_id === item.show_id);
    if (isDuplicate) return { duplicate: true };

    try {
      await API.addWatchlistItem(user.uid, listName, item.show_id, item.title, item.type);
      // Update cache
      if (!cache.lists[listName]) cache.lists[listName] = [];
      cache.lists[listName].push({
        show_id: item.show_id,
        title: item.title,
        type: item.type
      });
      document.dispatchEvent(new CustomEvent('watchlistUpdated'));
      return { added: true };
    } catch (err) {
      if (err.message.includes('Already')) return { duplicate: true };
      return { error: err.message || 'Failed to add item to watchlist.' };
    }
  }

  /** Remove an item from a watchlist. */
  async function removeFromList(listName, showId) {
    const user = Auth.getUser();
    if (!user) return false;

    try {
      await API.removeWatchlistItem(user.uid, listName, showId, '', '');
      // Update cache
      if (cache.lists[listName]) {
        cache.lists[listName] = cache.lists[listName].filter(i => i.show_id !== showId);
      }
      document.dispatchEvent(new CustomEvent('watchlistUpdated'));
      return true;
    } catch (err) {
      console.error('[Watchlist] Remove item error:', err);
      return false;
    }
  }

  /** Check if an item is in ANY watchlist. (Synchronous lookup) */
  function isInAnyList(item) {
    return Object.values(cache.lists).some(items =>
      items.some(i => i.show_id === item.show_id)
    );
  }

  /** Check if item is in a specific list. (Synchronous lookup) */
  function isInList(listName, item) {
    const items = cache.lists[listName];
    if (!items) return false;
    return items.some(i => i.show_id === item.show_id);
  }

  /* ═══════════════════════════════════════════════════
     WATCHED MANAGER
     ═══════════════════════════════════════════════════ */

  /** Get all watched items for current user. */
  function getWatched() {
    return cache.watched;
  }

  /** Get watched items filtered by type ('Movie' | 'TV Show'). */
  function getWatchedByType(type) {
    return cache.watched.filter(i => i.type === type);
  }

  /** Mark an item as watched. */
  async function markWatched(item) {
    const user = Auth.getUser();
    if (!user) return { error: 'Not logged in.' };

    const alreadyWatched = cache.watched.some(i => i.show_id === item.show_id);
    if (alreadyWatched) return { duplicate: true };

    try {
      await API.addWatched(user.uid, item.show_id, item.title, item.type);
      // Update cache
      cache.watched.push({
        show_id: item.show_id,
        title: item.title,
        type: item.type,
        watchedAt: Date.now()
      });
      document.dispatchEvent(new CustomEvent('watchlistUpdated'));
      return { added: true };
    } catch (err) {
      return { error: err.message || 'Failed to mark watched.' };
    }
  }

  /** Unmark an item as watched. */
  async function unmarkWatched(showId) {
    const user = Auth.getUser();
    if (!user) return false;

    try {
      // Find item details from cache first
      const item = cache.watched.find(i => i.show_id === showId);
      if (!item) return false;

      await API.removeWatched(user.uid, showId, item.title, item.type);
      // Update cache
      cache.watched = cache.watched.filter(i => i.show_id !== showId);
      document.dispatchEvent(new CustomEvent('watchlistUpdated'));
      return true;
    } catch (err) {
      console.error('[Watchlist] Unmark watched error:', err);
      return false;
    }
  }

  /** Check if an item is already watched. (Synchronous lookup) */
  function isWatched(item) {
    return cache.watched.some(i => i.show_id === item.show_id);
  }

  /* Expose */
  return {
    loadFromServer,
    // Watchlists
    getLists,
    createList,
    deleteList,
    addToList,
    removeFromList,
    isInAnyList,
    isInList,
    // Watched
    getWatched,
    getWatchedByType,
    markWatched,
    unmarkWatched,
    isWatched,
  };
})();
