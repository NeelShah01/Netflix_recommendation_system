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

    card.innerHTML = `
      <div class="card-poster" style="background: linear-gradient(135deg, hsl(${hue}, 70%, 25%) 0%, hsl(${(hue + 40) % 360}, 60%, 15%) 100%);">
        <span class="card-poster-icon">${typeIcon}</span>
        <div class="card-poster-overlay">
          <span class="card-play-btn">▶</span>
        </div>
        ${similarity}
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

    // Click handler
    card.addEventListener('click', () => {
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
  };
})();
