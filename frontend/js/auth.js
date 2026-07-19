/**
 * auth.js — Smart Content Recommender Authentication Module
 * Communicates with the FastAPI backend to store and authenticate user profiles
 * in the MongoDB database.
 * 
 * Session state is kept locally under `smartrec_user` so logins persist across refreshes.
 */

const Auth = (() => {
  'use strict';

  /* ═══════════════════ STORAGE KEYS ═══════════════════ */
  const SESSION_KEY = 'smartrec_user'; // current logged-in user session

  /* ═══════════════════ VALIDATION REGEXES (client-side checks) ═══════════════════ */
  const EMAIL_REGEX = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$/;

  /* ═══════════════════ SESSION STORE ═══════════════════ */
  const UserStore = {
    save(user) {
      try {
        localStorage.setItem(SESSION_KEY, JSON.stringify(user));
      } catch (e) {
        console.error('[Auth] Failed to save session:', e);
      }
    },
    load() {
      try {
        const raw = localStorage.getItem(SESSION_KEY);
        return raw ? JSON.parse(raw) : null;
      } catch { return null; }
    },
    clear() {
      localStorage.removeItem(SESSION_KEY);
    },
  };

  /* ═══════════════════ STATE ═══════════════════ */
  let currentUser = UserStore.load();
  const listeners = [];

  function notifyListeners() {
    listeners.forEach(fn => fn(currentUser));
  }

  /* ═══════════════════ VALIDATION ═══════════════════ */
  function validateEmail(email) {
    if (!email || !email.trim()) return { valid: false, error: 'Email is required.' };
    if (!EMAIL_REGEX.test(email.trim())) return { valid: false, error: 'Enter a valid email address.' };
    return { valid: true, error: null };
  }

  function validatePassword(password) {
    if (!password) return { valid: false, error: 'Password is required.' };
    if (password.length < 8) return { valid: false, error: 'Password must be at least 8 characters.' };
    if (!/[A-Z]/.test(password)) return { valid: false, error: 'Password must contain at least one uppercase letter.' };
    if (!/[a-z]/.test(password)) return { valid: false, error: 'Password must contain at least one lowercase letter.' };
    if (!/\d/.test(password)) return { valid: false, error: 'Password must contain at least one number.' };
    if (!/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?`~]/.test(password)) {
      return { valid: false, error: 'Password must contain at least one special character (!@#$%^&* etc.).' };
    }
    return { valid: true, error: null };
  }

  function validateDisplayName(name) {
    if (!name || !name.trim()) return { valid: false, error: 'Display name is required.' };
    if (name.trim().length < 2) return { valid: false, error: 'Display name must be at least 2 characters.' };
    if (name.trim().length > 50) return { valid: false, error: 'Display name must be under 50 characters.' };
    return { valid: true, error: null };
  }

  /* ═══════════════════ AUTH ACTIONS (Backend delegated) ═══════════════════ */

  /**
   * SIGN IN — calls FastAPI /api/auth/login endpoint
   */
  async function loginWithEmail(email, password) {
    const emailCheck = validateEmail(email);
    if (!emailCheck.valid) return { success: false, error: emailCheck.error, field: 'email' };

    const passCheck = validatePassword(password);
    if (!passCheck.valid) return { success: false, error: passCheck.error, field: 'password' };

    try {
      const res = await API.authLogin(email, password);
      if (res.success && res.user) {
        currentUser = res.user;
        UserStore.save(currentUser);
        notifyListeners();
        return { success: true, user: currentUser };
      }
      return { success: false, error: res.detail || 'Login failed.' };
    } catch (err) {
      return { success: false, error: err.message || 'Login connection error.', field: 'email' };
    }
  }

  /**
   * REGISTER — calls FastAPI /api/auth/register endpoint
   */
  async function registerWithEmail(email, password, displayName) {
    const emailCheck = validateEmail(email);
    if (!emailCheck.valid) return { success: false, error: emailCheck.error, field: 'email' };

    const passCheck = validatePassword(password);
    if (!passCheck.valid) return { success: false, error: passCheck.error, field: 'password' };

    const nameCheck = validateDisplayName(displayName);
    if (!nameCheck.valid) return { success: false, error: nameCheck.error, field: 'name' };

    try {
      const res = await API.authRegister(email, password, displayName);
      if (res.success && res.user) {
        currentUser = res.user;
        UserStore.save(currentUser);
        notifyListeners();
        return { success: true, user: currentUser };
      }
      return { success: false, error: res.detail || 'Registration failed.' };
    } catch (err) {
      return { success: false, error: err.message || 'Registration connection error.', field: 'email' };
    }
  }

  /**
   * GOOGLE SIGN-IN — delegates mapping/persisting Google auth details to MongoDB
   */
  async function loginWithGoogle(googleUser) {
    try {
      const res = await API.authGoogle(googleUser);
      if (res.success && res.user) {
        currentUser = res.user;
        UserStore.save(currentUser);
        notifyListeners();
        return { success: true, user: currentUser };
      }
      return { success: false, error: 'Google authentication failed.' };
    } catch (err) {
      console.error('[Auth] Google login error:', err);
      return { success: false, error: 'Google login connection error.' };
    }
  }

  function logout() {
    currentUser = null;
    UserStore.clear();
    notifyListeners();
  }

  function getUser()    { return currentUser; }
  function isLoggedIn() { return currentUser !== null; }

  function onChange(fn) {
    listeners.push(fn);
    return () => {
      const idx = listeners.indexOf(fn);
      if (idx !== -1) listeners.splice(idx, 1);
    };
  }

  /* Expose */
  return {
    loginWithEmail,
    registerWithEmail,
    loginWithGoogle,
    logout,
    getUser,
    isLoggedIn,
    onChange,
    validateEmail,
    validatePassword,
    validateDisplayName,
  };
})();
