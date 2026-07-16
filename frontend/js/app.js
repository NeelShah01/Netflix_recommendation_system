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
  };

  /* ═══════════════════ DOM REFS ═══════════════════ */
  const $ = (id) => document.getElementById(id);
  const $$ = (sel) => document.querySelectorAll(sel);

  /* ═══════════════════ INIT ═══════════════════ */
  document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initNavScroll();
    initSearch();
    initContentTypeToggle();
    initMoodSelector();
    initMultiSelect();
    initModal();
    initBackToTop();
    initFooterLinks();
    loadInitialData();
    initStaggeredReveal();
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

        // Re-trigger active genre/mood if any
        if (state.activeGenre) {
          loadGenreRecommendations(state.activeGenre, state.activeMood);
        }
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
        const results = await API.getMultiRecommendations(state.multiSelectTitles);
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
        setTimeout(() => panel.classList.add('hidden'), 400);

        // Scroll to results
        container.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
        const results = await API.getRecommendations(title);
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
        const results = await API.getCastRecommendations(title);
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
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    const modal = $('detailModal');
    modal.classList.remove('open');
    document.body.style.overflow = '';
    state.currentModalItem = null;
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
    // Load all data concurrently
    const [genresPromise, statsPromise, trendingPromise] = [
      API.getGenres().catch(() => []),
      API.getStats().catch(() => null),
      API.getTrending().catch(() => []),
    ];

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
      const results = await API.getGenreRecommendations(genre, mood, state.contentType || null);
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

})();
