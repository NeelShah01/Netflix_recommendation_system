/**
 * components.js — Reusable UI Component Library
 * Pure vanilla JS component factory functions
 */

const Components = (() => {
  /* ═══════════════════════════════════════════════════
     CONTENT CARD
     ═══════════════════════════════════════════════════ */
  function contentCard(item, options = {}) {
    const card = document.createElement('div');
    card.className = `content-card ${options.className || ''}`.trim();
    card.dataset.showId = item.show_id || '';
    card.dataset.title = item.title || '';

    const typeIcon = item.type === 'Movie' ? '🎬' : '📺';
    const year = item.release_year || '—';
    const rating = item.rating || '—';
    const similarity = item.similarity_score != null
      ? `<div class="card-similarity"><div class="similarity-fill" style="width:${Math.round(item.similarity_score * 100)}%"></div><span>${Math.round(item.similarity_score * 100)}% match</span></div>`
      : '';
    const genres = (item.listed_in || '').split(',').slice(0, 2).map(g => g.trim()).filter(Boolean);
    const genreChips = genres.map(g => `<span class="card-genre-chip">${g}</span>`).join('');
    const duration = item.duration || '';

    // Generate a deterministic color based on title
    const hue = hashString(item.title || 'Untitled') % 360;

    // Determine watched/watchlist state
    const loggedIn = (typeof Auth !== 'undefined') && Auth.isLoggedIn();
    const watched = loggedIn && (typeof WatchlistManager !== 'undefined') && WatchlistManager.isWatched(item);
    const inList = loggedIn && (typeof WatchlistManager !== 'undefined') && WatchlistManager.isInAnyList(item);

    const actionsHtml = loggedIn ? `
      <div class="card-actions" role="group" aria-label="Card actions">
        <button class="card-action-btn card-watchlist-btn ${inList ? 'in-list' : ''}" title="Add to Watchlist" data-action="watchlist">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="16" height="16"><path d="M12 5v14M5 12h14"/></svg>
        </button>
        <button class="card-action-btn card-watched-btn ${watched ? 'is-watched' : ''}" title="Mark as Watched" data-action="watched">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="16" height="16"><path d="M20 6L9 17l-5-5"/></svg>
        </button>
      </div>` : '';

    card.innerHTML = `
      <div class="card-poster" style="background: linear-gradient(135deg, hsl(${hue}, 70%, 25%) 0%, hsl(${(hue + 40) % 360}, 60%, 15%) 100%);">
        <span class="card-poster-icon">${typeIcon}</span>
        <div class="card-poster-overlay">
          <span class="card-play-btn">▶</span>
        </div>
        ${similarity}
        ${actionsHtml}
      </div>
      <div class="card-info">
        <h3 class="card-title" title="${escapeHtml(item.title || '')}">${escapeHtml(item.title || 'Untitled')}</h3>
        <div class="card-meta">
          <span class="card-year">${year}</span>
          <span class="card-dot">•</span>
          <span class="card-rating">${rating}</span>
          ${duration ? `<span class="card-dot">•</span><span class="card-duration">${duration}</span>` : ''}
        </div>
        <div class="card-genres">${genreChips}</div>
      </div>
    `;

    // Card action button handlers
    if (loggedIn) {
      const watchlistBtn = card.querySelector('[data-action="watchlist"]');
      const watchedBtn = card.querySelector('[data-action="watched"]');

      watchlistBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        document.dispatchEvent(new CustomEvent('openWatchlistPicker', { detail: { item, anchor: watchlistBtn } }));
      });

      watchedBtn?.addEventListener('click', async (e) => {
        e.stopPropagation();
        const alreadyWatched = WatchlistManager.isWatched(item);
        if (alreadyWatched) {
          await WatchlistManager.unmarkWatched(item.show_id || item.title);
          watchedBtn.classList.remove('is-watched');
          toast(`"${item.title}" removed from watched`, 'info');
        } else {
          const result = await WatchlistManager.markWatched(item);
          if (result.added) {
            watchedBtn.classList.add('is-watched');
            toast(`"${item.title}" marked as watched! ✓`, 'success');
          }
        }
      });
    }

    // Click handler
    card.addEventListener('click', (e) => {
      if (e.target.closest('.card-actions')) return;
      if (options.onSelect) {
        options.onSelect(item);
      } else if (options.multiSelect) {
        card.classList.toggle('selected');
        options.multiSelect(item, card.classList.contains('selected'));
      } else {
        document.dispatchEvent(new CustomEvent('openDetail', { detail: item }));
      }
    });

    return card;
  }

  /* ═══════════════════════════════════════════════════
     CAROUSEL ROW
     ═══════════════════════════════════════════════════ */
  function carousel(title, items, options = {}) {
    const section = document.createElement('div');
    section.className = 'carousel-section';
    if (options.id) section.id = options.id;

    const hasItems = items && items.length > 0;

    section.innerHTML = `
      <div class="carousel-header">
        <h2 class="carousel-title">
          <span class="title-accent">✦</span> ${escapeHtml(title)}
          ${options.subtitle ? `<span class="carousel-subtitle">${escapeHtml(options.subtitle)}</span>` : ''}
        </h2>
        ${hasItems ? `
        <div class="carousel-nav">
          <button class="carousel-arrow carousel-prev" aria-label="Previous">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="20" height="20"><path d="M15 18l-6-6 6-6"/></svg>
          </button>
          <button class="carousel-arrow carousel-next" aria-label="Next">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="20" height="20"><path d="M9 18l6-6-6-6"/></svg>
          </button>
        </div>` : ''}
      </div>
      <div class="carousel-track-wrapper">
        <div class="carousel-track"></div>
      </div>
    `;

    const track = section.querySelector('.carousel-track');

    if (hasItems) {
      items.forEach((item, idx) => {
        const card = contentCard(item, options);
        card.style.animationDelay = `${idx * 60}ms`;
        card.classList.add('animate-fade-up');
        track.appendChild(card);
      });

      // Arrow navigation
      const prevBtn = section.querySelector('.carousel-prev');
      const nextBtn = section.querySelector('.carousel-next');
      const wrapper = section.querySelector('.carousel-track-wrapper');

      if (prevBtn && nextBtn) {
        const scrollAmount = () => wrapper.clientWidth * 0.8;
        nextBtn.addEventListener('click', () => {
          wrapper.scrollBy({ left: scrollAmount(), behavior: 'smooth' });
        });
        prevBtn.addEventListener('click', () => {
          wrapper.scrollBy({ left: -scrollAmount(), behavior: 'smooth' });
        });

        // Hide/show arrows based on scroll position
        const updateArrows = () => {
          const { scrollLeft, scrollWidth, clientWidth } = wrapper;
          prevBtn.classList.toggle('hidden-arrow', scrollLeft <= 10);
          nextBtn.classList.toggle('hidden-arrow', scrollLeft + clientWidth >= scrollWidth - 10);
        };
        wrapper.addEventListener('scroll', updateArrows, { passive: true });
        // Initial check after render
        requestAnimationFrame(updateArrows);
      }
    } else {
      track.innerHTML = `<div class="carousel-empty">No results found. Try a different search or filter.</div>`;
    }

    return section;
  }

  /* ═══════════════════════════════════════════════════
     SKELETON LOADERS
     ═══════════════════════════════════════════════════ */
  function skeletonCards(count = 6) {
    const wrapper = document.createElement('div');
    wrapper.className = 'carousel-section';
    wrapper.innerHTML = `
      <div class="carousel-header">
        <div class="skeleton skeleton-title"></div>
      </div>
      <div class="carousel-track-wrapper">
        <div class="carousel-track">
          ${Array.from({ length: count }, () => `
            <div class="content-card skeleton-card">
              <div class="skeleton skeleton-poster"></div>
              <div class="card-info">
                <div class="skeleton skeleton-text" style="width:80%"></div>
                <div class="skeleton skeleton-text" style="width:50%"></div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
    return wrapper;
  }

  function skeletonStatsGrid() {
    return `
      <div class="stats-grid">
        ${Array.from({ length: 4 }, () => `
          <div class="stat-card skeleton-stat">
            <div class="skeleton skeleton-circle"></div>
            <div class="skeleton skeleton-text" style="width:60%"></div>
            <div class="skeleton skeleton-text-sm" style="width:40%"></div>
          </div>
        `).join('')}
      </div>
    `;
  }

  /* ═══════════════════════════════════════════════════
     GENRE CHIP
     ═══════════════════════════════════════════════════ */
  function genreChip(genre, isActive = false) {
    const chip = document.createElement('button');
    chip.className = `genre-chip ${isActive ? 'active' : ''}`;
    chip.textContent = genre;
    chip.dataset.genre = genre;
    return chip;
  }

  /* ═══════════════════════════════════════════════════
     STAT CARD
     ═══════════════════════════════════════════════════ */
  function statCard(icon, value, label, colorClass = '') {
    const card = document.createElement('div');
    card.className = `stat-card ${colorClass}`;
    card.innerHTML = `
      <div class="stat-icon">${icon}</div>
      <div class="stat-value">${animateNumber(value)}</div>
      <div class="stat-label">${label}</div>
    `;
    return card;
  }

  function animateNumber(value) {
    const num = typeof value === 'number' ? value : parseInt(value, 10);
    if (isNaN(num)) return escapeHtml(String(value));
    return `<span class="counter" data-target="${num}">0</span>`;
  }

  /* ═══════════════════════════════════════════════════
     GENRE DISTRIBUTION BAR
     ═══════════════════════════════════════════════════ */
  function genreBar(genre, count, maxCount) {
    const pct = Math.round((count / maxCount) * 100);
    const bar = document.createElement('div');
    bar.className = 'genre-bar-row';
    bar.innerHTML = `
      <span class="genre-bar-label">${escapeHtml(genre)}</span>
      <div class="genre-bar-track">
        <div class="genre-bar-fill" style="--target-width:${pct}%" data-count="${count}"></div>
      </div>
      <span class="genre-bar-count">${count}</span>
    `;
    return bar;
  }

  /* ═══════════════════════════════════════════════════
     AUTOCOMPLETE ITEM
     ═══════════════════════════════════════════════════ */
  function autocompleteItem(item, onClick) {
    const div = document.createElement('div');
    div.className = 'autocomplete-item';
    div.dataset.showId = item.show_id || '';
    const typeIcon = item.type === 'Movie' ? '🎬' : '📺';
    div.innerHTML = `
      <span class="ac-icon">${typeIcon}</span>
      <div class="ac-info">
        <span class="ac-title">${escapeHtml(item.title || '')}</span>
        <span class="ac-meta">${item.release_year || ''} ${item.rating ? '• ' + item.rating : ''}</span>
      </div>
    `;
    div.addEventListener('click', () => onClick(item));
    return div;
  }

  /* ═══════════════════════════════════════════════════
     MULTI-SELECT CHIP
     ═══════════════════════════════════════════════════ */
  function multiChip(title, onRemove) {
    const chip = document.createElement('span');
    chip.className = 'multi-chip';
    chip.innerHTML = `
      <span class="multi-chip-text">${escapeHtml(title)}</span>
      <button class="multi-chip-remove" aria-label="Remove">&times;</button>
    `;
    chip.querySelector('.multi-chip-remove').addEventListener('click', (e) => {
      e.stopPropagation();
      chip.classList.add('removing');
      setTimeout(() => { chip.remove(); onRemove(title); }, 250);
    });
    return chip;
  }

  /* ═══════════════════════════════════════════════════
     TOAST NOTIFICATION
     ═══════════════════════════════════════════════════ */
  function toast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
    const t = document.createElement('div');
    t.className = `toast toast-${type}`;
    t.innerHTML = `
      <span class="toast-icon">${icons[type] || icons.info}</span>
      <span class="toast-message">${escapeHtml(message)}</span>
      <button class="toast-close">&times;</button>
    `;

    t.querySelector('.toast-close').addEventListener('click', () => dismissToast(t));
    container.appendChild(t);

    // Trigger entrance animation
    requestAnimationFrame(() => t.classList.add('toast-visible'));

    // Auto-dismiss
    setTimeout(() => dismissToast(t), duration);
  }

  function dismissToast(t) {
    t.classList.remove('toast-visible');
    t.classList.add('toast-exit');
    setTimeout(() => t.remove(), 400);
  }

  /* ═══════════════════════════════════════════════════
     MODAL POPULATION
     ═══════════════════════════════════════════════════ */
  function populateModal(item) {
    const $ = (id) => document.getElementById(id);

    $('modalTitle').textContent = item.title || 'Untitled';
    $('modalType').textContent = item.type || '';
    $('modalRating').textContent = item.rating || '';
    $('modalYear').textContent = item.release_year || '';
    $('modalDuration').textContent = item.duration || '';
    $('modalDescription').textContent = item.description || 'No description available.';
    $('modalDirector').textContent = item.director || 'Unknown';
    $('modalCast').textContent = item.cast || 'Not available';
    $('modalCountry').textContent = item.country || 'Unknown';
    $('modalGenres').textContent = item.listed_in || '';

    // Hide empty fields
    $('modalDirectorWrap').style.display = item.director ? '' : 'none';
    $('modalCastWrap').style.display = item.cast ? '' : 'none';
    $('modalCountryWrap').style.display = item.country ? '' : 'none';
    $('modalGenresWrap').style.display = item.listed_in ? '' : 'none';

    // Type badge color
    $('modalType').className = `modal-badge type-badge ${item.type === 'Movie' ? 'badge-movie' : 'badge-tv'}`;

    // Clear previous similar
    $('modalSimilar').innerHTML = '';

    // Set hero gradient based on title
    const hue = hashString(item.title || '') % 360;
    const hero = document.querySelector('.modal-hero');
    if (hero) {
      hero.style.background = `linear-gradient(135deg, hsl(${hue}, 70%, 20%) 0%, hsl(${(hue + 50) % 360}, 60%, 10%) 100%)`;
    }
  }

  /* ═══════════════════════════════════════════════════
     UTILITIES
     ═══════════════════════════════════════════════════ */
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + ((hash << 5) - hash);
      hash |= 0;
    }
    return Math.abs(hash);
  }

  /* ═══════════════════════════════════════════════════
     COUNTER ANIMATION (Intersection Observer)
     ═══════════════════════════════════════════════════ */
  function initCounterAnimations() {
    const counters = document.querySelectorAll('.counter:not(.counted)');
    if (!counters.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const el = entry.target;
          el.classList.add('counted');
          const target = parseInt(el.dataset.target, 10);
          animateCounter(el, target);
          observer.unobserve(el);
        }
      });
    }, { threshold: 0.3 });

    counters.forEach((c) => observer.observe(c));
  }

  function animateCounter(el, target) {
    const duration = 1500;
    const start = performance.now();

    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      el.textContent = Math.round(eased * target).toLocaleString();
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ═══════════════════════════════════════════════════
     ANIMATE GENRE BARS
     ═══════════════════════════════════════════════════ */
  function initGenreBarAnimations() {
    const bars = document.querySelectorAll('.genre-bar-fill:not(.animated)');
    if (!bars.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const el = entry.target;
          el.classList.add('animated');
          requestAnimationFrame(() => {
            el.style.width = el.style.getPropertyValue('--target-width');
          });
          observer.unobserve(el);
        }
      });
    }, { threshold: 0.1 });

    bars.forEach((b) => observer.observe(b));
  }

  /* ═══════════════════════════════════════════════════
     WATCHLIST PICKER DROPDOWN
     ═══════════════════════════════════════════════════ */
  function watchlistPickerDropdown(item, anchor, onListCreated) {
    // Remove any existing picker
    document.querySelectorAll('.watchlist-picker').forEach(el => el.remove());

    if (!Auth.isLoggedIn()) return;
    const lists = WatchlistManager.getLists();

    const picker = document.createElement('div');
    picker.className = 'watchlist-picker';
    picker.setAttribute('role', 'menu');

    const header = document.createElement('div');
    header.className = 'watchlist-picker-header';
    header.textContent = 'Add to list';
    picker.appendChild(header);

    if (lists.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'watchlist-picker-empty';
      empty.textContent = 'No lists yet. Create one below!';
      picker.appendChild(empty);
    }

    lists.forEach(list => {
      const inList = WatchlistManager.isInList(list.id, item);
      const row = document.createElement('button');
      row.className = `watchlist-picker-row${inList ? ' in-list' : ''}`;
      row.setAttribute('role', 'menuitem');
      row.innerHTML = `
        <span class="picker-list-name">${escapeHtml(list.name)}</span>
        <span class="picker-check">${inList ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="14" height="14"><path d="M20 6L9 17l-5-5"/></svg>` : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M12 5v14M5 12h14"/></svg>`}</span>
      `;
      row.addEventListener('click', async () => {
        if (inList) {
          toast(`Already in "${list.name}"`, 'warning');
          return;
        }
        const result = await WatchlistManager.addToList(list.id, item);
        if (result.duplicate) {
          toast(`Already in "${list.name}"`, 'warning');
        } else if (result.added) {
          toast(`Added to "${list.name}"!`, 'success');
          // Update + button on any matching cards
          document.querySelectorAll(`[data-title="${CSS.escape(item.title)}"] .card-watchlist-btn`).forEach(btn => btn.classList.add('in-list'));
        }
        picker.remove();
      });
      picker.appendChild(row);
    });

    // Divider
    const divider = document.createElement('div');
    divider.className = 'watchlist-picker-divider';
    picker.appendChild(divider);

    // New list row
    const newRow = document.createElement('div');
    newRow.className = 'watchlist-picker-new';
    newRow.innerHTML = `
      <input type="text" class="picker-new-input" placeholder="New list name..." maxlength="50" />
      <button class="picker-new-btn">Create</button>
    `;
    const newInput = newRow.querySelector('.picker-new-input');
    const newBtn = newRow.querySelector('.picker-new-btn');
    newBtn.addEventListener('click', async () => {
      const name = newInput.value.trim();
      if (!name) { newInput.focus(); return; }
      const result = await WatchlistManager.createList(name);
      if (result && result.error) {
        toast(result.error, 'warning');
        return;
      }
      if (result && result.name) {
        await WatchlistManager.addToList(result.name, item);
        toast(`Created "${name}" and added item!`, 'success');
        if (onListCreated) onListCreated();
      }
      picker.remove();
    });
    newInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') newBtn.click();
      e.stopPropagation();
    });
    picker.appendChild(newRow);

    // Position near anchor
    document.body.appendChild(picker);
    const rect = anchor.getBoundingClientRect();
    const pickerRect = picker.getBoundingClientRect();
    let top = rect.bottom + window.scrollY + 6;
    let left = rect.left + window.scrollX;
    if (left + pickerRect.width > window.innerWidth - 12) {
      left = window.innerWidth - pickerRect.width - 12;
    }
    picker.style.top = `${top}px`;
    picker.style.left = `${left}px`;
    picker.style.opacity = '0';
    requestAnimationFrame(() => { picker.style.opacity = '1'; });

    // Close on outside click
    setTimeout(() => {
      function outsideClick(e) {
        if (!picker.contains(e.target) && e.target !== anchor) {
          picker.remove();
          document.removeEventListener('click', outsideClick);
        }
      }
      document.addEventListener('click', outsideClick);
    }, 0);

    return picker;
  }

  /* ═══════════════════════════════════════════════════
     WATCHED ITEM CARD (for sidebar views)
     ═══════════════════════════════════════════════════ */
  function watchedItemCard(item, onUnmark) {
    const hue = hashString(item.title || '') % 360;
    const typeIcon = item.type === 'Movie' ? '🎬' : '📺';
    const card = document.createElement('div');
    card.className = 'watched-item-card';
    card.innerHTML = `
      <div class="watched-item-poster" style="background: linear-gradient(135deg, hsl(${hue}, 70%, 25%), hsl(${(hue+40)%360}, 60%, 15%));">
        <span>${typeIcon}</span>
      </div>
      <div class="watched-item-info">
        <p class="watched-item-title">${escapeHtml(item.title)}</p>
        <p class="watched-item-meta">${item.release_year || ''} ${item.rating ? '• ' + item.rating : ''}</p>
      </div>
      <button class="watched-item-remove" title="Remove from watched">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="14" height="14"><path d="M18 6L6 18M6 6l12 12"/></svg>
      </button>
    `;
    card.querySelector('.watched-item-remove').addEventListener('click', async () => {
      await WatchlistManager.unmarkWatched(item.show_id || item.title);
      card.classList.add('removing');
      setTimeout(() => { card.remove(); if (onUnmark) onUnmark(); }, 300);
      // Update cards on the main page
      document.querySelectorAll(`[data-title="${CSS.escape(item.title)}"] .card-watched-btn`).forEach(btn => btn.classList.remove('is-watched'));
      toast(`"${item.title}" removed from watched`, 'info');
    });
    return card;
  }

  /* ═══════════════════════════════════════════════════
     WATCHLIST ITEM ROW (for sidebar views)
     ═══════════════════════════════════════════════════ */
  function watchlistItemRow(item, listId, onRemove) {
    const hue = hashString(item.title || '') % 360;
    const typeIcon = item.type === 'Movie' ? '🎬' : '📺';
    const row = document.createElement('div');
    row.className = 'watchlist-item-row';
    row.innerHTML = `
      <div class="wl-item-poster" style="background: linear-gradient(135deg, hsl(${hue},70%,25%), hsl(${(hue+40)%360},60%,15%))"><span>${typeIcon}</span></div>
      <div class="wl-item-info">
        <p class="wl-item-title">${escapeHtml(item.title)}</p>
        <p class="wl-item-meta">${item.release_year || ''} ${item.rating ? '• ' + item.rating : ''}</p>
      </div>
      <button class="wl-item-remove" title="Remove">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="14" height="14"><path d="M18 6L6 18M6 6l12 12"/></svg>
      </button>
    `;
    row.querySelector('.wl-item-remove').addEventListener('click', async () => {
      await WatchlistManager.removeFromList(listId, item.show_id || item.title);
      row.classList.add('removing');
      setTimeout(() => { row.remove(); if (onRemove) onRemove(); }, 300);
      document.querySelectorAll(`[data-title="${CSS.escape(item.title)}"] .card-watchlist-btn`).forEach(btn => {
        if (!WatchlistManager.isInAnyList(item)) btn.classList.remove('in-list');
      });
      toast(`"${item.title}" removed from list`, 'info');
    });
    return row;
  }

  /* ───── Expose ───── */

  return {
    contentCard,
    carousel,
    skeletonCards,
    skeletonStatsGrid,
    genreChip,
    statCard,
    genreBar,
    autocompleteItem,
    multiChip,
    toast,
    populateModal,
    escapeHtml,
    hashString,
    initCounterAnimations,
    initGenreBarAnimations,
    watchlistPickerDropdown,
    watchedItemCard,
    watchlistItemRow,
  };
})();
