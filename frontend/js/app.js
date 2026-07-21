/**
 * app.js — Smart Content Recommender Application Core
 * Orchestrates all UI interactions, data fetching, and rendering
 */

(() => {
  'use strict';

  /* ═══════════════════ STATE ═══════════════════ */
  const state = {
    contentType: '',           // '' | 'Movie' | 'TV Show'
    activeGenre: null,
    activeMood: null,
    multiSelectMode: false,
    multiSelectTitles: [],     // string[]
    currentModalItem: null,
    searchAbortController: null,
    genres: [],
    stats: null,
    sidebarView: 'profile',   // 'profile' | 'watchlists' | 'watched-movies' | 'watched-tv'
    excludeGenres: [],
  };

  /* ═══════════════════ DOM REFS ═══════════════════ */
  const $ = (id) => document.getElementById(id);
  const $$ = (sel) => document.querySelectorAll(sel);

  /* ═══════════════════ INIT ═══════════════════ */
  document.addEventListener('DOMContentLoaded', () => {
    loadStoredExclusions();
    initParticles();
    initNavScroll();
    initSearch();
    initContentTypeToggle();
    initMoodSelector();
    initMultiSelect();
    initPreferencesPanel();
    initModal();
    initTrailerModal();
    initBackToTop();
    initFooterLinks();
    loadInitialData();
    initStaggeredReveal();
    initAuth();
    initSidebar();
    initWatchlistPicker();
  });

  /* ═══════════════════ PARTICLE SYSTEM ═══════════════════ */
  function initParticles() {
    const canvas = $('particleCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let particles = [];
    let animId;
    let width, height;

    function resize() {
      width = canvas.width = canvas.offsetWidth;
      height = canvas.height = canvas.offsetHeight;
    }

    function createParticle() {
      return {
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        r: Math.random() * 2 + 0.5,
        alpha: Math.random() * 0.5 + 0.1,
        color: ['#7b2ff7', '#f72f8e', '#ff6b6b', '#4fc3f7', '#ba68c8'][Math.floor(Math.random() * 5)],
      };
    }

    function init() {
      resize();
      const count = Math.min(Math.floor((width * height) / 12000), 120);
      particles = Array.from({ length: count }, createParticle);
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;

        // Wrap around edges
        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.fill();
      }

      // Draw connections
      ctx.globalAlpha = 1;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 100) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = particles[i].color;
            ctx.globalAlpha = 0.06 * (1 - dist / 100);
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      ctx.globalAlpha = 1;
      animId = requestAnimationFrame(draw);
    }

    window.addEventListener('resize', () => {
      cancelAnimationFrame(animId);
      init();
      draw();
    });

    init();
    draw();
  }

  /* ═══════════════════ NAV SCROLL ═══════════════════ */
  function initNavScroll() {
    const nav = $('mainNav');
    const scrollHint = $('heroScrollHint');

    window.addEventListener('scroll', () => {
      const scrolled = window.scrollY > 60;
      nav.classList.toggle('scrolled', scrolled);
      if (scrollHint) {
        scrollHint.style.opacity = window.scrollY > 200 ? '0' : '1';
      }
    }, { passive: true });
  }

  /* ═══════════════════ SEARCH ═══════════════════ */
  function initSearch() {
    const input = $('searchInput');
    const dropdown = $('autocompleteDropdown');
    const clearBtn = $('searchClear');

    if (!input) return;

    input.addEventListener('input', async () => {
      const query = input.value.trim();
      clearBtn.classList.toggle('visible', query.length > 0);

      if (query.length < 2) {
        closeDropdown(dropdown);
        return;
      }

      // Abort previous search
      if (state.searchAbortController) state.searchAbortController.abort();
      state.searchAbortController = new AbortController();

      try {
        const results = await API.debouncedSearch(query, state.searchAbortController.signal);
        renderAutocomplete(results, dropdown, (item) => {
          input.value = '';
          clearBtn.classList.remove('visible');
          closeDropdown(dropdown);
          showDetailModal(item);
        });
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Search error:', err);
        }
      }
    });

    input.addEventListener('focus', () => {
      if (input.value.trim().length >= 2 && dropdown.children.length > 0) {
        dropdown.classList.add('open');
      }
    });

    clearBtn.addEventListener('click', () => {
      input.value = '';
      clearBtn.classList.remove('visible');
      closeDropdown(dropdown);
      input.focus();
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#searchContainer')) closeDropdown(dropdown);
    });

    // Keyboard navigation
    input.addEventListener('keydown', (e) => {
      handleAutocompleteKeyboard(e, dropdown, input);
    });
  }

  function renderAutocomplete(results, dropdown, onClick) {
    dropdown.innerHTML = '';
    if (!results || results.length === 0) {
      dropdown.innerHTML = '<div class="autocomplete-empty">No results found</div>';
      dropdown.classList.add('open');
      return;
    }

    const filtered = state.contentType
      ? results.filter(r => r.type === state.contentType)
      : results;

    if (filtered.length === 0) {
      dropdown.innerHTML = '<div class="autocomplete-empty">No results for this content type</div>';
      dropdown.classList.add('open');
      return;
    }

    filtered.slice(0, 8).forEach((item) => {
      dropdown.appendChild(Components.autocompleteItem(item, onClick));
    });
    dropdown.classList.add('open');
  }

  function handleAutocompleteKeyboard(e, dropdown, input) {
    const items = dropdown.querySelectorAll('.autocomplete-item');
    if (!items.length) return;

    const active = dropdown.querySelector('.autocomplete-item.active');
    let idx = Array.from(items).indexOf(active);

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (active) active.classList.remove('active');
      idx = (idx + 1) % items.length;
      items[idx].classList.add('active');
      items[idx].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (active) active.classList.remove('active');
      idx = idx <= 0 ? items.length - 1 : idx - 1;
      items[idx].classList.add('active');
      items[idx].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter' && active) {
      e.preventDefault();
      active.click();
    } else if (e.key === 'Escape') {
      closeDropdown(dropdown);
      input.blur();
    }
  }

  function closeDropdown(dropdown) {
    dropdown.classList.remove('open');
  }

  /* ═══════════════════ CONTENT TYPE TOGGLE ═══════════════════ */
  function initContentTypeToggle() {
    const btns = $$('.type-btn');
    const indicator = document.querySelector('.toggle-indicator');

    btns.forEach((btn) => {
      btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.contentType = btn.dataset.type;

        // Move indicator
        if (indicator) {
          indicator.style.width = `${btn.offsetWidth}px`;
          indicator.style.left = `${btn.offsetLeft}px`;
        }

        // Re-trigger all recommendations
        refreshRecommendations();
      });
    });

    // Init indicator position
    const activeBtn = document.querySelector('.type-btn.active');
    if (indicator && activeBtn) {
      requestAnimationFrame(() => {
        indicator.style.width = `${activeBtn.offsetWidth}px`;
        indicator.style.left = `${activeBtn.offsetLeft}px`;
      });
    }
  }

  /* ═══════════════════ MOOD SELECTOR ═══════════════════ */
  function initMoodSelector() {
    const cards = $$('.mood-card');
    cards.forEach((card) => {
      card.addEventListener('click', () => {
        const wasActive = card.classList.contains('active');
        cards.forEach(c => c.classList.remove('active'));

        if (wasActive) {
          state.activeMood = null;
        } else {
          card.classList.add('active');
          state.activeMood = card.dataset.mood;
        }

        // If a genre is selected, reload with mood
        if (state.activeGenre) {
          loadGenreRecommendations(state.activeGenre, state.activeMood);
        } else if (state.activeMood) {
          // Pick a default genre or load mood-only
          loadGenreRecommendations('Dramas', state.activeMood);
        }
      });
    });
  }

  /* ═══════════════════ MULTI-SELECT ═══════════════════ */
  function initMultiSelect() {
    const panel = $('multiSelectPanel');
    const searchInput = $('multiSelectSearch');
    const dropdown = $('multiSelectDropdown');
    const chipsContainer = $('multiSelectChips');
    const submitBtn = $('multiSelectSubmit');
    const cancelBtn = $('multiSelectCancel');
    const countBadge = $('multiSelectCount');

    // Activate multi-select mode
    const activate = () => {
      state.multiSelectMode = true;
      state.multiSelectTitles = [];
      panel.classList.remove('hidden');
      panel.classList.add('visible');
      chipsContainer.innerHTML = '';
      updateMultiCount();
      searchInput.focus();
      document.body.style.overflow = '';
    };

    $('heroMultiSelectBtn')?.addEventListener('click', activate);
    $('footerMultiPick')?.addEventListener('click', (e) => {
      e.preventDefault();
      activate();
      panel.scrollIntoView({ behavior: 'smooth' });
    });

    // Cancel
    cancelBtn?.addEventListener('click', () => {
      state.multiSelectMode = false;
      state.multiSelectTitles = [];
      panel.classList.remove('visible');
      setTimeout(() => panel.classList.add('hidden'), 400);
    });

    // Search within multi-select
    searchInput?.addEventListener('input', async () => {
      const query = searchInput.value.trim();
      if (query.length < 2) { closeDropdown(dropdown); return; }

      try {
        const results = await API.debouncedSearch(query);
        renderAutocomplete(results, dropdown, (item) => {
          addMultiTitle(item.title);
          searchInput.value = '';
          closeDropdown(dropdown);
        });
      } catch (err) {
        console.error('Multi-search error:', err);
      }
    });

    searchInput?.addEventListener('keydown', (e) => {
      handleAutocompleteKeyboard(e, dropdown, searchInput);
    });

    document.addEventListener('click', (e) => {
      if (!e.target.closest('.multi-select-search-wrap')) closeDropdown(dropdown);
    });

    function addMultiTitle(title) {
      if (state.multiSelectTitles.includes(title)) {
        Components.toast('Title already added', 'warning');
        return;
      }
      if (state.multiSelectTitles.length >= 10) {
        Components.toast('Maximum 10 titles allowed', 'warning');
        return;
      }
      state.multiSelectTitles.push(title);
      const chip = Components.multiChip(title, removeMultiTitle);
      chipsContainer.appendChild(chip);
      updateMultiCount();
    }

    function removeMultiTitle(title) {
      state.multiSelectTitles = state.multiSelectTitles.filter(t => t !== title);
      updateMultiCount();
    }

    function updateMultiCount() {
      const count = state.multiSelectTitles.length;
      if (countBadge) countBadge.textContent = count;
      if (submitBtn) submitBtn.disabled = count < 2;
    }

    // Submit multi-select
    submitBtn?.addEventListener('click', async () => {
      if (state.multiSelectTitles.length < 2) return;
      const container = $('carouselContainer');

      // Show loading
      container.innerHTML = '';
      container.appendChild(Components.skeletonCards(8));

      try {
        const results = await API.getMultiRecommendations(
          state.multiSelectTitles,
          15,
          state.excludeGenres,
          Auth.isLoggedIn() ? Auth.getUser().uid : null,
          state.contentType || null
        );
        const recs = normalizeRecommendationResults(results);
        container.innerHTML = '';

        const titles = state.multiSelectTitles.join(', ');
        container.appendChild(
          Components.carousel(`Because you liked: ${titles}`, recs, {
            id: 'multiRecCarousel',
            subtitle: `${recs.length} blended recommendations`,
          })
        );

        // Close panel
        state.multiSelectMode = false;
        panel.classList.remove('visible');
        setTimeout(() => {
          panel.classList.add('hidden');
          // Scroll to results after the panel is fully hidden and document layout has settled
          container.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 500);

        Components.toast(`Found ${recs.length} recommendations!`, 'success');
      } catch (err) {
        container.innerHTML = '';
        Components.toast('Failed to get recommendations. Please try again.', 'error');
      }
    });
  }

  /* ═══════════════════ MODAL ═══════════════════ */
  function initModal() {
    const modal = $('detailModal');
    const closeBtn = $('modalClose');
    const recBtn = $('modalRecommendBtn');
    const castBtn = $('modalCastRecBtn');

    // Close modal
    closeBtn?.addEventListener('click', closeModal);
    modal?.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modal.classList.contains('open')) closeModal();
    });

    // Open detail event
    document.addEventListener('openDetail', async (e) => {
      const item = e.detail;
      await showDetailModal(item);
    });

    // "Find Similar" button
    recBtn?.addEventListener('click', async () => {
      if (!state.currentModalItem) return;
      const title = state.currentModalItem.title;
      const similarDiv = $('modalSimilar');

      similarDiv.innerHTML = `
        <h3 class="modal-similar-title">Similar to "${Components.escapeHtml(title)}"</h3>
        <div class="carousel-track">${Array.from({ length: 6 }, () => `
          <div class="content-card skeleton-card">
            <div class="skeleton skeleton-poster"></div>
            <div class="card-info"><div class="skeleton skeleton-text"></div></div>
          </div>
        `).join('')}</div>
      `;

      try {
        const results = await API.getRecommendations(
          title,
          12,
          state.excludeGenres,
          Auth.isLoggedIn() ? Auth.getUser().uid : null,
          state.contentType || null
        );
        const recs = normalizeRecommendationResults(results);
        similarDiv.innerHTML = '';

        const section = Components.carousel(`Similar to "${title}"`, recs, {
          subtitle: `${recs.length} matches`,
        });
        similarDiv.appendChild(section);
      } catch (err) {
        similarDiv.innerHTML = `<p class="modal-error">Could not load recommendations.</p>`;
      }
    });

    // Cast-based
    castBtn?.addEventListener('click', async () => {
      if (!state.currentModalItem) return;
      const title = state.currentModalItem.title;
      const similarDiv = $('modalSimilar');

      similarDiv.innerHTML = `
        <h3 class="modal-similar-title">Same cast as "${Components.escapeHtml(title)}"</h3>
        <div class="carousel-track">${Array.from({ length: 6 }, () => `
          <div class="content-card skeleton-card">
            <div class="skeleton skeleton-poster"></div>
            <div class="card-info"><div class="skeleton skeleton-text"></div></div>
          </div>
        `).join('')}</div>
      `;

      try {
        const results = await API.getCastRecommendations(
          title,
          12,
          state.excludeGenres,
          Auth.isLoggedIn() ? Auth.getUser().uid : null,
          state.contentType || null
        );
        const recs = normalizeRecommendationResults(results);
        similarDiv.innerHTML = '';

        const section = Components.carousel(`Same cast as "${title}"`, recs, {
          subtitle: `${recs.length} titles`,
        });
        similarDiv.appendChild(section);
      } catch (err) {
        similarDiv.innerHTML = `<p class="modal-error">Could not load cast recommendations.</p>`;
      }
    });

    // Star rating input clicks
    const starContainer = $('starRatingInput');
    if (starContainer) {
      const stars = starContainer.querySelectorAll('.star-icon');
      stars.forEach(star => {
        star.addEventListener('click', (e) => {
          const val = parseInt(e.target.dataset.value);
          starContainer.dataset.rating = val;
          stars.forEach((s, idx) => {
            s.classList.toggle('active', idx < val);
          });
        });
      });
    }

    // Submit review form
    const reviewForm = $('modalReviewForm');
    reviewForm?.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!state.currentModalItem) return;
      const user = Auth.getUser();
      if (!user) {
        Components.toast("Please sign in to leave a review.", "warning");
        return;
      }
      
      const rating = parseInt($('starRatingInput').dataset.rating || '0');
      if (rating === 0) {
        Components.toast("Please select a star rating first.", "warning");
        return;
      }
      
      const reviewText = $('modalReviewText').value.trim();
      const showId = state.currentModalItem.show_id;
      const displayName = user.displayName || user.email || 'User';
      
      const submitBtn = $('submitReviewBtn');
      if (submitBtn) submitBtn.disabled = true;
      
      try {
        await API.submitReview(user.uid, displayName, showId, rating, reviewText);
        Components.toast("Review submitted successfully!", "success");
        await loadModalReviews(showId);
      } catch (err) {
        Components.toast("Failed to submit review.", "error");
      } finally {
        if (submitBtn) submitBtn.disabled = false;
      }
    });

    // Review login link click
    const reviewLoginLink = $('reviewLoginLink');
    reviewLoginLink?.addEventListener('click', (e) => {
      e.preventDefault();
      closeModal();
      openLoginModal();
    });
  }

  async function showDetailModal(item) {
    const modal = $('detailModal');

    // If we only have a show_id, fetch full details
    let fullItem = item;
    if (item.show_id && !item.description) {
      try {
        fullItem = await API.getTitle(item.show_id);
      } catch (err) {
        fullItem = item; // Fallback to partial data
      }
    }

    state.currentModalItem = fullItem;
    Components.populateModal(fullItem);
    updateModalUserActions(fullItem);
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';

    // Fetch and load community reviews
    if (fullItem.show_id) {
      loadModalReviews(fullItem.show_id);
    }

    // Fetch rich media (posters, ratings, trailers) from TMDB dynamically
    if (fullItem.show_id) {
      API.getMedia(fullItem.show_id).then(media => {
        if (!media || state.currentModalItem?.show_id !== fullItem.show_id) return;
        
        // 1. Show TMDB Poster if resolved
        const posterSide = $('modalPosterSide');
        const posterImg = $('modalPosterImg');
        if (media.poster_path && posterSide && posterImg) {
          posterImg.src = media.poster_path;
          posterSide.classList.remove('hidden');
        }

        // 2. Add TMDB score badge
        const tmdbRating = $('modalTmdbRating');
        if (media.vote_average != null && tmdbRating) {
          tmdbRating.textContent = `⭐ TMDB: ${media.vote_average}`;
          tmdbRating.classList.remove('hidden');
        }

        // 3. Render backdrop image in the modal header
        const hero = document.querySelector('.modal-hero');
        if (media.backdrop_path && hero) {
          hero.style.background = `linear-gradient(to bottom, rgba(10, 10, 15, 0.4), rgba(10, 10, 15, 0.9)), url(${media.backdrop_path}) center/cover no-repeat`;
        }

        // 4. Wire trailer play button if available
        const trailerBtn = $('modalTrailerBtn');
        if (media.trailer_url && trailerBtn) {
          // Replace button with clean clone to clear previous listeners
          const newTrailerBtn = trailerBtn.cloneNode(true);
          newTrailerBtn.classList.remove('hidden');
          trailerBtn.parentNode.replaceChild(newTrailerBtn, trailerBtn);

          newTrailerBtn.addEventListener('click', () => {
            const watchUrl = media.trailer_url.replace('/embed/', '/watch?v=');
            window.open(watchUrl, '_blank');
          });
        }
      }).catch(err => {
        console.warn('Failed to load TMDB details:', err);
      });
    }
  }

  function updateModalUserActions(item) {
    const actionsEl = $('modalUserActions');
    if (!actionsEl) return;
    if (!Auth.isLoggedIn()) {
      actionsEl.innerHTML = '';
      return;
    }
    const isWatched = WatchlistManager.isWatched(item);
    const inList = WatchlistManager.isInAnyList(item);
    actionsEl.innerHTML = `
      <button id="modalWatchlistBtn" class="btn btn-glass ${inList ? 'active-action' : ''}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M12 5v14M5 12h14"/></svg>
        ${inList ? 'In Watchlist' : 'Add to Watchlist'}
      </button>
      <button id="modalWatchedBtn" class="btn btn-glass ${isWatched ? 'active-action watched-active' : ''}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M20 6L9 17l-5-5"/></svg>
        ${isWatched ? 'Watched ✓' : 'Mark as Watched'}
      </button>
    `;
    $('modalWatchlistBtn')?.addEventListener('click', (e) => {
      Components.watchlistPickerDropdown(item, e.currentTarget, () => updateModalUserActions(item));
      // Refresh after picker closes
      setTimeout(() => updateModalUserActions(item), 500);
    });
    $('modalWatchedBtn')?.addEventListener('click', async () => {
      const alreadyWatched = WatchlistManager.isWatched(item);
      if (alreadyWatched) {
        await WatchlistManager.unmarkWatched(item.show_id || item.title);
        Components.toast(`"${item.title}" removed from watched`, 'info');
      } else {
        await WatchlistManager.markWatched(item);
        Components.toast(`"${item.title}" marked as watched! ✓`, 'success');
      }
      updateModalUserActions(item);
    });
  }

  function closeModal() {
    const modal = $('detailModal');
    modal.classList.remove('open');
    document.body.style.overflow = '';
    state.currentModalItem = null;

    // Clear and hide trailer video player
    const player = $('trailerPlayer');
    if (player) player.src = '';
    $('trailerModal')?.classList.add('hidden');
  }

  /* ═══════════════════ BACK TO TOP ═══════════════════ */
  function initBackToTop() {
    const btn = $('backToTop');
    if (!btn) return;

    window.addEventListener('scroll', () => {
      btn.classList.toggle('visible', window.scrollY > 600);
    }, { passive: true });

    btn.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  /* ═══════════════════ FOOTER LINKS ═══════════════════ */
  function initFooterLinks() {
    $('heroExploreBtn')?.addEventListener('click', () => {
      $('discoverySection')?.scrollIntoView({ behavior: 'smooth' });
    });

    $('heroScrollHint')?.addEventListener('click', () => {
      $('discoverySection')?.scrollIntoView({ behavior: 'smooth' });
    });
  }

  /* ═══════════════════ STAGGERED REVEAL ═══════════════════ */
  function initStaggeredReveal() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    $$('.section').forEach((el) => observer.observe(el));
  }

  /* ═══════════════════ LOAD DATA ═══════════════════ */
  async function loadInitialData() {
    // Load all data concurrently — title index loads alongside other data
    const [genresPromise, statsPromise, trendingPromise] = [
      API.getGenres().catch(() => []),
      API.getStats().catch(() => null),
      API.getTrending().catch(() => []),
    ];

    // Kick off title index load in parallel (no await — runs in background)
    API.loadTitleIndex().catch(() => {});

    // Sync watchlist / watched history from MongoDB if already logged in
    if (Auth.isLoggedIn()) {
      WatchlistManager.loadFromServer().catch(() => {});
    }

    const [genres, stats, trending] = await Promise.all([genresPromise, statsPromise, trendingPromise]);
    const trendingItems = normalizeRecommendationResults(trending);

    // Genres
    if (genres && genres.length) {
      state.genres = genres;
      renderGenreChips(genres);
    }

    // Stats
    if (stats) {
      state.stats = stats;
      renderHeroStats(stats);
      renderStats(stats);
    }

    // Trending
    if (trendingItems.length) {
      renderTrending(trendingItems);
    }
  }

  /* ═══════════════════ RENDER GENRE CHIPS ═══════════════════ */
  function normalizeRecommendationResults(payload) {
    if (!payload) return [];
    if (Array.isArray(payload)) return payload;
    if (Array.isArray(payload.results)) return payload.results;
    if (Array.isArray(payload.recommendations)) return payload.recommendations;
    return [];
  }

  function renderGenreChips(genres) {
    const container = $('genreChips');
    if (!container) return;
    container.innerHTML = '';

    const genreList = Array.isArray(genres) ? genres : [];

    genreList.forEach((genre) => {
      const genreName = typeof genre === 'string' ? genre : genre.name || genre.genre || String(genre);
      const chip = Components.genreChip(genreName);
      chip.addEventListener('click', () => {
        // Toggle active state
        const wasActive = chip.classList.contains('active');
        $$('.genre-chip').forEach(c => c.classList.remove('active'));

        if (wasActive) {
          state.activeGenre = null;
          // Clear genre recommendations
          $('carouselContainer').innerHTML = '';
        } else {
          chip.classList.add('active');
          state.activeGenre = genreName;
          loadGenreRecommendations(genreName, state.activeMood);
        }
      });
      container.appendChild(chip);
    });
  }

  /* ═══════════════════ LOAD GENRE RECOMMENDATIONS ═══════════════════ */
  async function loadGenreRecommendations(genre, mood) {
    const container = $('carouselContainer');

    // Show skeleton
    container.innerHTML = '';
    container.appendChild(Components.skeletonCards(8));

    try {
      const results = await API.getGenreRecommendations(
        genre,
        mood,
        state.contentType || null,
        15,
        state.excludeGenres,
        Auth.isLoggedIn() ? Auth.getUser().uid : null
      );
      const recs = normalizeRecommendationResults(results);

      container.innerHTML = '';
      const moodLabel = mood ? ` • ${mood.replace('-', ' ')}` : '';
      container.appendChild(
        Components.carousel(`${genre}${moodLabel}`, recs, {
          id: 'genreRecCarousel',
          subtitle: `${recs.length} recommendations`,
        })
      );

      container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (err) {
      container.innerHTML = '';
      Components.toast(`Failed to load ${genre} recommendations`, 'error');
    }
  }

  /* ═══════════════════ RENDER TRENDING ═══════════════════ */
  function renderTrending(items) {
    const container = $('trendingCarousel');
    if (!container) return;
    container.innerHTML = '';

    container.appendChild(
      Components.carousel('Trending & Recent', items, {
        id: 'trendingRow',
        subtitle: 'Recently added to the catalog',
      })
    );
  }

  /* ═══════════════════ RENDER HERO STATS ═══════════════════ */
  function renderHeroStats(stats) {
    const titles = stats.total_titles || stats.total || '8000+';
    const genres = stats.total_genres || stats.genres_count || '40+';
    const countryCount = stats.total_countries || stats.countries_count || (stats.country_distribution ? Object.keys(stats.country_distribution).length : 0);
    const countries = countryCount || '100+';

    $('heroStatTitles').textContent = typeof titles === 'number' ? titles.toLocaleString() : titles;
    $('heroStatGenres').textContent = typeof genres === 'number' ? genres.toLocaleString() : genres;
    $('heroStatCountries').textContent = typeof countries === 'number' ? countries.toLocaleString() : countries;
  }

  /* ═══════════════════ RENDER STATS SECTION ═══════════════════ */
  function renderStats(stats) {
    const grid = $('statsGrid');
    const barsContainer = $('genreBars');
    if (!grid) return;
    grid.innerHTML = '';

    // Stat cards
    const countryCount = stats.total_countries || stats.countries_count || (stats.country_distribution ? Object.keys(stats.country_distribution).length : 0);
    const cards = [
      { icon: '🎬', value: stats.total_movies || stats.movies || 0, label: 'Movies', cls: 'stat-purple' },
      { icon: '📺', value: stats.total_shows || stats.total_tv_shows || stats.tv_shows || 0, label: 'TV Shows', cls: 'stat-pink' },
      { icon: '🌍', value: countryCount, label: 'Countries', cls: 'stat-coral' },
      { icon: '🎭', value: stats.total_genres || stats.genres_count || 0, label: 'Genres', cls: 'stat-blue' },
    ];

    cards.forEach(({ icon, value, label, cls }) => {
      grid.appendChild(Components.statCard(icon, value, label, cls));
    });

    // Genre distribution bars
    if (barsContainer && stats.genre_distribution) {
      barsContainer.innerHTML = '';
      const distribution = stats.genre_distribution;
      const entries = Array.isArray(distribution)
        ? distribution
        : Object.entries(distribution).map(([genre, count]) => ({ genre, count }));

      const sorted = entries.sort((a, b) => (b.count || b.value || 0) - (a.count || a.value || 0)).slice(0, 15);
      const maxCount = sorted.length > 0 ? (sorted[0].count || sorted[0].value || 1) : 1;

      sorted.forEach((item) => {
        const genre = item.genre || item.name || item[0] || '';
        const count = item.count || item.value || item[1] || 0;
        barsContainer.appendChild(Components.genreBar(genre, count, maxCount));
      });
    }

    // Initialize animations
    requestAnimationFrame(() => {
      Components.initCounterAnimations();
      Components.initGenreBarAnimations();
    });
  }

  /* ═══════════════════ CONTENT TYPE TOGGLE INDICATOR ═══════════════════ */
  // Recalculate on resize
  window.addEventListener('resize', () => {
    const activeBtn = document.querySelector('.type-btn.active');
    const indicator = document.querySelector('.toggle-indicator');
    if (indicator && activeBtn) {
      indicator.style.width = `${activeBtn.offsetWidth}px`;
      indicator.style.left = `${activeBtn.offsetLeft}px`;
    }
  });

  /* ═══════════════════ AUTH ═══════════════════ */
  function initAuth() {
    // Update navbar and load personal recommendations whenever auth state changes
    Auth.onChange((user) => {
      updateNavProfile(user);
      handlePersonalRecommendations(user);
    });
    updateNavProfile(Auth.getUser());
    handlePersonalRecommendations(Auth.getUser());

    // Listen for watchlist mutations to reload recommendations
    document.addEventListener('watchlistUpdated', () => {
      const user = Auth.getUser();
      if (user) {
        handlePersonalRecommendations(user);
      }
    });

    // Login Modal Open
    $('navProfileBtn')?.addEventListener('click', () => {
      if (Auth.isLoggedIn()) {
        openSidebar();
      } else {
        openLoginModal();
      }
    });

    window.handleGoogleSignIn = async function(response) {
      try {
        const payload = parseJwt(response.credential);
        const result = await Auth.loginWithGoogle(payload);
        if (result.success) {
          closeLoginModal();
          await WatchlistManager.loadFromServer();
          Components.toast(`Welcome, ${result.user.displayName}! 👋`, 'success');
        } else {
          Components.toast(result.error || 'Google sign-in failed.', 'error');
        }
      } catch (e) {
        console.error('[Google Sign-In Callback Error]:', e);
        Components.toast('Google sign-in failed. Please try again.', 'error');
      }
    };

    // Login form
    initLoginForm();
  }

  function parseJwt(token) {
    try {
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const padded = base64.padEnd(base64.length + (4 - base64.length % 4) % 4, '=');
      return JSON.parse(atob(padded));
    } catch (e) {
      console.error('[parseJwt] failed:', e);
      throw new Error('Invalid token format');
    }
  }

  function updateNavProfile(user) {
    const btn = $('navProfileBtn');
    const avatarEl = $('navAvatarDisplay');
    if (!btn || !avatarEl) return;
    if (user) {
      if (user.picture) {
        avatarEl.innerHTML = `<img src="${user.picture}" alt="Profile" style="width:36px;height:36px;border-radius:50%;object-fit:cover;" />`;
      } else {
        avatarEl.innerHTML = `<span class="nav-avatar-letter" style="background:${user.gradient}">${user.avatar}</span>`;
      }
      btn.title = user.displayName;
    } else {
      avatarEl.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="22" height="22"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 010 8z"/></svg>`;
      btn.title = 'Sign In';
    }
  }

  /* ─── Login Modal ─── */
  function openLoginModal() {
    const modal = $('loginModal');
    if (!modal) return;
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
    $('loginEmailInput')?.focus();
    // Reset to login tab
    switchAuthTab('login');
  }

  function closeLoginModal() {
    const modal = $('loginModal');
    if (!modal) return;
    modal.classList.remove('open');
    document.body.style.overflow = '';
    clearLoginErrors();
  }

  function switchAuthTab(tab) {
    const loginTab = $('loginTabBtn');
    const registerTab = $('registerTabBtn');
    const nameWrap = $('loginNameWrap');
    const submitBtn = $('loginSubmitBtn');
    const formTitle = $('loginFormTitle');

    if (tab === 'login') {
      loginTab?.classList.add('active');
      registerTab?.classList.remove('active');
      if (nameWrap) nameWrap.style.display = 'none';
      if (submitBtn) submitBtn.textContent = 'Sign In';
      if (formTitle) formTitle.textContent = 'Welcome back';
    } else {
      registerTab?.classList.add('active');
      loginTab?.classList.remove('active');
      if (nameWrap) nameWrap.style.display = 'block';
      if (submitBtn) submitBtn.textContent = 'Create Account';
      if (formTitle) formTitle.textContent = 'Create account';
    }
    clearLoginErrors();
  }

  function clearLoginErrors() {
    $$('.login-field-error').forEach(el => el.textContent = '');
    $$('.login-input.error').forEach(el => el.classList.remove('error'));
  }

  function showFieldError(fieldId, errorId, message) {
    const field = $(fieldId);
    const errorEl = $(errorId);
    if (field) field.classList.add('error');
    if (errorEl) errorEl.textContent = message;
  }

  function initLoginForm() {
    $('loginModal')?.addEventListener('click', (e) => {
      if (e.target === $('loginModal')) closeLoginModal();
    });
    $('loginCloseBtn')?.addEventListener('click', closeLoginModal);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && $('loginModal')?.classList.contains('open')) closeLoginModal();
    });

    $('loginTabBtn')?.addEventListener('click', () => switchAuthTab('login'));
    $('registerTabBtn')?.addEventListener('click', () => switchAuthTab('register'));

    // Password show/hide
    $('loginPasswordToggle')?.addEventListener('click', () => {
      const input = $('loginPasswordInput');
      if (!input) return;
      const isText = input.type === 'text';
      input.type = isText ? 'password' : 'text';
      $('loginPasswordToggle').innerHTML = isText
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24M1 1l22 22"/></svg>`;
    });

    // Real-time validation
    $('loginEmailInput')?.addEventListener('input', () => {
      const val = $('loginEmailInput').value;
      if (val.length > 3) {
        const check = Auth.validateEmail(val);
        if (!check.valid) {
          showFieldError('loginEmailInput', 'loginEmailError', check.error);
        } else {
          $('loginEmailInput').classList.remove('error');
          $('loginEmailError').textContent = '';
        }
      }
    });

    $('loginPasswordInput')?.addEventListener('input', () => {
      const val = $('loginPasswordInput').value;
      if (val.length > 0) {
        const check = Auth.validatePassword(val);
        const activeTab = document.querySelector('.auth-tab.active')?.dataset.tab;
        if (!check.valid && activeTab === 'register') {
          showFieldError('loginPasswordInput', 'loginPasswordError', check.error);
        } else {
          $('loginPasswordInput').classList.remove('error');
          $('loginPasswordError').textContent = '';
        }
      }
    });

    // Submit
    $('loginSubmitBtn')?.addEventListener('click', async () => {
      clearLoginErrors();
      const isRegister = $('registerTabBtn')?.classList.contains('active');
      const email = $('loginEmailInput')?.value.trim() || '';
      const password = $('loginPasswordInput')?.value || '';
      const displayName = isRegister ? ($('loginNameInput')?.value.trim() || '') : null;

      // Show loading state on button
      const submitBtn = $('loginSubmitBtn');
      const originalText = submitBtn.textContent;
      submitBtn.textContent = isRegister ? 'Creating account...' : 'Signing in...';
      submitBtn.disabled = true;

      try {
        // Route to the correct async function
        const result = isRegister
          ? await Auth.registerWithEmail(email, password, displayName)
          : await Auth.loginWithEmail(email, password);

        if (!result.success) {
          if (result.field === 'email')    showFieldError('loginEmailInput',    'loginEmailError',    result.error);
          else if (result.field === 'password') showFieldError('loginPasswordInput', 'loginPasswordError', result.error);
          else if (result.field === 'name')     showFieldError('loginNameInput',      'loginNameError',      result.error);
          return;
        }
        closeLoginModal();
        await WatchlistManager.loadFromServer();
        const greeting = isRegister ? `Account created! Welcome, ${result.user.displayName}! 🎉` : `Welcome back, ${result.user.displayName}! 👋`;
        Components.toast(greeting, 'success');
      } catch (err) {
        console.error('[Auth] Submit error:', err);
        Components.toast('Something went wrong. Please try again.', 'error');
      } finally {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
      }
    });

    // Enter key on inputs
    [$('loginEmailInput'), $('loginPasswordInput'), $('loginNameInput')].forEach(el => {
      el?.addEventListener('keydown', (e) => { if (e.key === 'Enter') $('loginSubmitBtn')?.click(); });
    });
  }

  /* ═══════════════════ WATCHLIST PICKER ═══════════════════ */
  function initWatchlistPicker() {
    document.addEventListener('openWatchlistPicker', (e) => {
      const { item, anchor } = e.detail;
      Components.watchlistPickerDropdown(item, anchor, () => renderSidebarContent());
    });
  }

  /* ═══════════════════ PROFILE SIDEBAR ═══════════════════ */
  function openSidebar() {
    const sidebar = $('profileSidebar');
    const backdrop = $('sidebarBackdrop');
    if (!sidebar) return;
    sidebar.classList.add('open');
    backdrop?.classList.add('open');
    renderSidebarContent();
  }

  function closeSidebar() {
    const sidebar = $('profileSidebar');
    const backdrop = $('sidebarBackdrop');
    sidebar?.classList.remove('open');
    backdrop?.classList.remove('open');
  }

  function initSidebar() {
    $('sidebarBackdrop')?.addEventListener('click', closeSidebar);
    $('sidebarCloseBtn')?.addEventListener('click', closeSidebar);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && $('profileSidebar')?.classList.contains('open')) closeSidebar();
    });

    // Nav items
    $$('.sidebar-nav-item').forEach(btn => {
      btn.addEventListener('click', () => {
        $$('.sidebar-nav-item').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.sidebarView = btn.dataset.view;
        renderSidebarContent();
      });
    });

    // Logout
    $('sidebarLogoutBtn')?.addEventListener('click', () => {
      Auth.logout();
      closeSidebar();
      WatchlistManager.loadFromServer(); // clear watchlist cache
      Components.toast('You have been signed out.', 'info');
    });

    // Create list button inside sidebar
    $('sidebarCreateListBtn')?.addEventListener('click', async () => {
      const name = prompt('Enter a name for your new watchlist:');
      if (!name) return;
      const result = await WatchlistManager.createList(name.trim());
      if (result && result.error) { Components.toast(result.error, 'warning'); return; }
      if (result && result.name) { Components.toast(`List "${result.name}" created!`, 'success'); }
      renderSidebarContent();
    });
  }

  function renderSidebarContent() {
    const user = Auth.getUser();
    if (!user) return;

    // Update profile header
    const nameEl = $('sidebarUserName');
    const emailEl = $('sidebarUserEmail');
    const avatarEl = $('sidebarAvatar');
    if (nameEl) nameEl.textContent = user.displayName;
    if (emailEl) emailEl.textContent = user.email;
    if (avatarEl) {
      if (user.picture) {
        avatarEl.innerHTML = `<img src="${user.picture}" alt="Avatar" style="width:100%;height:100%;border-radius:50%;object-fit:cover;" />`;
      } else {
        avatarEl.style.background = user.gradient;
        avatarEl.textContent = user.avatar;
      }
    }

    const contentEl = $('sidebarContent');
    if (!contentEl) return;
    contentEl.innerHTML = '';

    const view = state.sidebarView || 'profile';

    if (view === 'profile') {
      renderSidebarProfile(contentEl, user);
    } else if (view === 'watchlists') {
      renderSidebarWatchlists(contentEl);
    } else if (view === 'watched-movies') {
      renderSidebarWatched(contentEl, 'Movie');
    } else if (view === 'watched-tv') {
      renderSidebarWatched(contentEl, 'TV Show');
    }
  }

  function renderSidebarProfile(container, user) {
    container.innerHTML = `
      <div class="sidebar-section">
        <h3 class="sidebar-section-title">Account Info</h3>
        <div class="sidebar-info-row"><span class="sidebar-info-label">Name</span><span class="sidebar-info-value">${Components.escapeHtml(user.displayName)}</span></div>
        <div class="sidebar-info-row"><span class="sidebar-info-label">Email</span><span class="sidebar-info-value">${Components.escapeHtml(user.email)}</span></div>
        <div class="sidebar-info-row"><span class="sidebar-info-label">Signed in via</span><span class="sidebar-info-value" style="text-transform:capitalize">${user.provider}</span></div>
      </div>
      <div class="sidebar-section">
        <h3 class="sidebar-section-title">Quick Stats</h3>
        <div class="sidebar-stats-grid">
          <div class="sidebar-stat"><span class="sidebar-stat-number">${WatchlistManager.getLists().length}</span><span class="sidebar-stat-label">Watchlists</span></div>
          <div class="sidebar-stat"><span class="sidebar-stat-number">${WatchlistManager.getLists().reduce((a, l) => a + l.items.length, 0)}</span><span class="sidebar-stat-label">Saved Items</span></div>
          <div class="sidebar-stat"><span class="sidebar-stat-number">${WatchlistManager.getWatchedByType('Movie').length}</span><span class="sidebar-stat-label">Movies Watched</span></div>
          <div class="sidebar-stat"><span class="sidebar-stat-number">${WatchlistManager.getWatchedByType('TV Show').length}</span><span class="sidebar-stat-label">Shows Watched</span></div>
        </div>
      </div>
    `;
  }

  function renderSidebarWatchlists(container) {
    const lists = WatchlistManager.getLists();
    const headerEl = document.createElement('div');
    headerEl.className = 'sidebar-section-header';
    headerEl.innerHTML = `
      <h3 class="sidebar-section-title">My Watchlists</h3>
      <button class="sidebar-create-list-btn" id="sidebarCreateListInline">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="14" height="14"><path d="M12 5v14M5 12h14"/></svg>
        New List
      </button>
    `;
    container.appendChild(headerEl);

    $('sidebarCreateListInline')?.addEventListener('click', () => {
      const nameInput = document.createElement('div');
      nameInput.className = 'sidebar-new-list-input-wrap';
      nameInput.innerHTML = `
        <input type="text" class="sidebar-new-list-input" placeholder="List name..." maxlength="50" />
        <button class="sidebar-new-list-save">Save</button>
        <button class="sidebar-new-list-cancel">×</button>
      `;
      container.insertBefore(nameInput, headerEl.nextSibling);
      const inp = nameInput.querySelector('.sidebar-new-list-input');
      inp.focus();
      nameInput.querySelector('.sidebar-new-list-save').addEventListener('click', async () => {
        const name = inp.value.trim();
        if (!name) return;
        const result = await WatchlistManager.createList(name);
        if (result && result.error) { Components.toast(result.error, 'warning'); return; }
        if (result && result.name) Components.toast(`List "${result.name}" created!`, 'success');
        renderSidebarContent();
      });
      nameInput.querySelector('.sidebar-new-list-cancel').addEventListener('click', () => nameInput.remove());
      inp.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') nameInput.querySelector('.sidebar-new-list-save').click();
        if (e.key === 'Escape') nameInput.remove();
      });
    });

    if (lists.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'sidebar-empty';
      empty.innerHTML = `<span>📱</span><p>No watchlists yet!</p><p>Click "+ New List" to create one.</p>`;
      container.appendChild(empty);
      return;
    }

    lists.forEach(list => {
      const section = document.createElement('div');
      section.className = 'sidebar-list-section';
      section.innerHTML = `
        <div class="sidebar-list-header">
          <button class="sidebar-list-toggle" aria-expanded="false">
            <svg class="sidebar-list-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="14" height="14"><path d="M9 18l6-6-6-6"/></svg>
            <span class="sidebar-list-name">${Components.escapeHtml(list.name)}</span>
            <span class="sidebar-list-count">${list.items.length}</span>
          </button>
          <button class="sidebar-list-delete" title="Delete list">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6M10 11v6M14 11v6M9 6V4h6v2"/></svg>
          </button>
        </div>
        <div class="sidebar-list-items collapsed"></div>
      `;

      const toggleBtn = section.querySelector('.sidebar-list-toggle');
      const itemsEl = section.querySelector('.sidebar-list-items');
      const arrow = section.querySelector('.sidebar-list-arrow');

      toggleBtn.addEventListener('click', () => {
        const isOpen = !itemsEl.classList.contains('collapsed');
        if (isOpen) {
          itemsEl.classList.add('collapsed');
          arrow.style.transform = '';
          toggleBtn.setAttribute('aria-expanded', 'false');
        } else {
          itemsEl.classList.remove('collapsed');
          arrow.style.transform = 'rotate(90deg)';
          toggleBtn.setAttribute('aria-expanded', 'true');
          // Populate items
          itemsEl.innerHTML = '';
          if (list.items.length === 0) {
            itemsEl.innerHTML = '<p class="sidebar-list-empty">This list is empty.</p>';
          } else {
            list.items.forEach(it => {
              itemsEl.appendChild(Components.watchlistItemRow(it, list.id, () => renderSidebarContent()));
            });
          }
        }
      });

      section.querySelector('.sidebar-list-delete').addEventListener('click', async () => {
        if (!confirm(`Delete watchlist "${list.name}"?`)) return;
        await WatchlistManager.deleteList(list.id);
        Components.toast(`"${list.name}" deleted`, 'info');
        renderSidebarContent();
      });

      container.appendChild(section);
    });
  }

  function renderSidebarWatched(container, type) {
    const items = WatchlistManager.getWatchedByType(type);
    const label = type === 'Movie' ? 'Movies' : 'TV Shows';

    const header = document.createElement('div');
    header.className = 'sidebar-section';
    header.innerHTML = `<h3 class="sidebar-section-title">Watched ${label} <span class="sidebar-count-badge">${items.length}</span></h3>`;
    container.appendChild(header);

    if (items.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'sidebar-empty';
      empty.innerHTML = `<span>${type === 'Movie' ? '🎬' : '📺'}</span><p>No ${label.toLowerCase()} watched yet!</p><p>Mark items as watched using the ✓ button on any card.</p>`;
      container.appendChild(empty);
      return;
    }

    const grid = document.createElement('div');
    grid.className = 'sidebar-watched-list';
    items.forEach(item => {
      grid.appendChild(Components.watchedItemCard(item, () => renderSidebarContent()));
    });
    container.appendChild(grid);
  }



  function playTrailer(url) {
    const trailerModal = $('trailerModal');
    const player = $('trailerPlayer');
    if (!trailerModal || !player) return;
    
    const autoplayUrl = url.includes('?') ? `${url}&autoplay=1` : `${url}?autoplay=1`;
    player.src = autoplayUrl;
    trailerModal.classList.remove('hidden');
  }

  function initTrailerModal() {
    const trailerModal = $('trailerModal');
    const closeBtn = $('trailerModalClose');
    const player = $('trailerPlayer');
    
    if (!trailerModal || !closeBtn || !player) return;
    
    const closeTrailer = () => {
      player.src = '';
      trailerModal.classList.add('hidden');
    };
    
    closeBtn.addEventListener('click', closeTrailer);
    trailerModal.addEventListener('click', (e) => {
      if (e.target === trailerModal) closeTrailer();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !trailerModal.classList.contains('hidden')) {
        closeTrailer();
        e.stopPropagation();
      }
    });
  }

  /* ═══════════════════ RECOMMENDATIONS EXCLUSIONS & PERSONALIZATION ═══════════════════ */

  function loadStoredExclusions() {
    try {
      const stored = localStorage.getItem('smartrec_exclusions');
      if (stored) {
        state.excludeGenres = JSON.parse(stored);
      }
    } catch (e) {
      console.warn("Failed to load stored exclusions:", e);
    }
  }

  function initPreferencesPanel() {
    const panel = $('preferencesPanel');
    const toggleBtn = $('navPreferencesBtn');
    const closeBtn = $('closePreferencesBtn');
    const clearBtn = $('clearPreferencesBtn');
    const chipsContainer = $('excludeGenresChips');

    if (!panel || !toggleBtn || !closeBtn || !clearBtn || !chipsContainer) return;

    toggleBtn.addEventListener('click', () => {
      const isHidden = panel.classList.contains('hidden');
      if (isHidden) {
        panel.classList.remove('hidden');
        renderPreferencesChips();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } else {
        panel.classList.add('hidden');
      }
    });

    closeBtn.addEventListener('click', () => {
      panel.classList.add('hidden');
      applyExclusions();
    });

    clearBtn.addEventListener('click', () => {
      state.excludeGenres = [];
      const checkboxes = chipsContainer.querySelectorAll('input[type="checkbox"]');
      checkboxes.forEach(cb => cb.checked = false);
      chipsContainer.querySelectorAll('.pref-chip-label').forEach(label => label.classList.remove('active'));
      applyExclusions();
    });

    function renderPreferencesChips() {
      if (!state.genres || state.genres.length === 0) {
        API.getGenres().then(genresList => {
          state.genres = genresList;
          populateChips();
        });
      } else {
        populateChips();
      }
    }

    function populateChips() {
      chipsContainer.innerHTML = '';
      state.genres.forEach(genre => {
        const isExcluded = state.excludeGenres.includes(genre);
        const label = document.createElement('label');
        label.className = `pref-chip-label ${isExcluded ? 'active' : ''}`;
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = genre;
        checkbox.checked = isExcluded;
        
        checkbox.addEventListener('change', () => {
          label.classList.toggle('active', checkbox.checked);
        });

        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(genre));
        chipsContainer.appendChild(label);
      });
    }

    function applyExclusions() {
      const selected = [];
      const checkedInputs = chipsContainer.querySelectorAll('input[type="checkbox"]:checked');
      checkedInputs.forEach(input => selected.push(input.value));
      state.excludeGenres = selected;
      
      localStorage.setItem('smartrec_exclusions', JSON.stringify(selected));
      refreshRecommendations();
    }
  }

  function refreshRecommendations() {
    // 1. Refresh Trending Carousel
    API.getTrending(state.contentType || null).then(trending => {
      const trendingItems = normalizeRecommendationResults(trending);
      renderTrending(trendingItems);
    }).catch(err => console.warn("Failed to refresh trending:", err));

    // 2. Refresh Genre/Mood Carousel
    if (state.activeGenre || state.activeMood) {
      loadGenreRecommendations(state.activeGenre, state.activeMood);
    }

    // 3. Refresh Personalized Carousel
    const user = Auth.getUser();
    if (user) {
      handlePersonalRecommendations(user);
    }
  }

  async function handlePersonalRecommendations(user) {
    const section = $('forYouSection');
    const container = $('forYouCarousel');
    if (!section || !container) return;

    if (!user) {
      section.classList.add('hidden');
      container.innerHTML = '';
      return;
    }

    section.classList.remove('hidden');
    container.innerHTML = `
      <div class="carousel-header">
        <h2 class="carousel-title"><span class="title-accent">✦</span> Recommended For You</h2>
      </div>
      <div class="carousel-track-wrapper">
        <div class="carousel-track">${Array.from({ length: 6 }, () => `
          <div class="content-card skeleton-card">
            <div class="skeleton skeleton-poster"></div>
            <div class="card-info"><div class="skeleton skeleton-text"></div></div>
          </div>
        `).join('')}</div>
      </div>
    `;

    try {
      const res = await API.getPersonalRecommendations(user.uid, 20, state.excludeGenres, state.contentType || null);
      
      const currentUser = Auth.getUser();
      if (!currentUser || currentUser.uid !== user.uid) return;

      if (res && res.count > 0) {
        container.innerHTML = '';
        const recCarousel = Components.carousel("Recommended For You 🌟", res.results, {
          id: "forYouCarouselRow",
          subtitle: "Based on your watch history & list"
        });
        container.appendChild(recCarousel);
      } else {
        container.innerHTML = `
          <div class="carousel-header">
            <h2 class="carousel-title"><span class="title-accent">✦</span> Recommended For You</h2>
          </div>
          <div class="personalized-fallback-card">
            <div class="fallback-icon">🌟</div>
            <h3>Your Personalized Feed is Ready!</h3>
            <p>Add titles to your watchlist or mark them as watched, and we'll curate recommendations tailored to your taste right here.</p>
          </div>
        `;
      }
    } catch (err) {
      console.warn("Failed to load personalized recommendations:", err);
      container.innerHTML = `<p class="carousel-error">Could not load personalized recommendations.</p>`;
    }
  }

  /* ═══════════════════ RATINGS & REVIEWS ═══════════════════ */

  async function loadModalReviews(showId) {
    const list = $('modalReviewsList');
    const badge = $('modalAverageRatingBadge');
    const formContainer = $('modalReviewFormContainer');
    const authTip = $('modalReviewAuthTip');

    if (!list || !badge) return;

    // Toggle Form visibility based on login state
    if (Auth.isLoggedIn()) {
      formContainer?.classList.remove('hidden');
      authTip?.classList.add('hidden');
    } else {
      formContainer?.classList.add('hidden');
      authTip?.classList.remove('hidden');
    }

    try {
      const res = await API.getReviews(showId);
      if (state.currentModalItem?.show_id !== showId) return;

      if (res.review_count > 0) {
        badge.textContent = `⭐ ${res.average_rating} (${res.review_count} review${res.review_count > 1 ? 's' : ''})`;
        badge.classList.remove('hidden');
        list.innerHTML = '';
        res.reviews.forEach(r => {
          list.appendChild(Components.reviewCard(r));
        });
      } else {
        badge.classList.add('hidden');
        list.innerHTML = '<div class="no-reviews-placeholder">Be the first to rate and review this title!</div>';
      }
    } catch (err) {
      console.warn("Failed to load reviews:", err);
      list.innerHTML = '<div class="modal-error">Could not load reviews.</div>';
    }
  }

})();
