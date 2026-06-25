/**
 * useAppStore.js — Zustand global store for the Movie Recommender frontend.
 *
 * Replaces the scattered useState calls in App.jsx with a single,
 * debuggable, predictable state atom. Components subscribe to only
 * the slices they need — no unnecessary re-renders.
 *
 * Slices:
 *  - Session  : userId, token, hasOnboarded
 *  - Data     : recommendations, popularMovies, feed, stats
 *  - Controls : weightCol, novelty, diversity
 *  - UI       : searchQuery, searchResults, sessionRatings, isLoading, errorMsg
 *  - Modal    : isLoginModalOpen
 */

import { create } from 'zustand';
import api from '../api';

const useAppStore = create((set, get) => ({

  // ── Session ─────────────────────────────────────────────────────────────────
  userId: localStorage.getItem('userId') || '',
  token: localStorage.getItem('token') || '',
  hasOnboarded: localStorage.getItem('hasOnboarded') === 'true',

  // ── Data ────────────────────────────────────────────────────────────────────
  recommendations: [],
  popularMovies: [],
  feed: [],
  stats: null,

  // ── Algorithm Controls ──────────────────────────────────────────────────────
  weightCol: 0.5,
  novelty: 0.0,
  diversity: 0.2,

  // ── UI State ────────────────────────────────────────────────────────────────
  searchQuery: '',
  searchResults: [],
  sessionRatings: {},
  isLoading: false,
  errorMsg: '',
  isLoginModalOpen: false,

  // ── Actions: Session ────────────────────────────────────────────────────────

  setLogin(username, token) {
    localStorage.setItem('userId', username);
    localStorage.setItem('token', token);
    set({ userId: username, token, sessionRatings: {} });
  },

  resetSession() {
    localStorage.removeItem('userId');
    localStorage.removeItem('token');
    localStorage.removeItem('hasOnboarded');
    set({
      userId: '', token: '', hasOnboarded: false,
      sessionRatings: {}, recommendations: [],
      searchResults: [], searchQuery: '',
    });
  },

  completeOnboarding(returnedUserId) {
    const { userId } = get();
    // If not logged in yet, adopt the guest ID from onboarding
    if (!userId) {
      localStorage.setItem('userId', returnedUserId);
      set({ userId: returnedUserId });
    }
    localStorage.setItem('hasOnboarded', 'true');
    set({ hasOnboarded: true, sessionRatings: {} });
  },

  // ── Actions: Data fetching ──────────────────────────────────────────────────

  async fetchStats() {
    try {
      const data = await api.getStats();
      set({ stats: data });
    } catch (e) {
      console.error('Failed to load stats', e);
    }
  },

  async fetchPopular() {
    try {
      const data = await api.getPopularMovies(5);
      // /popular now returns { page, limit, total, results } — handle both shapes
      set({ popularMovies: Array.isArray(data) ? data : (data.results || []) });
    } catch (e) {
      console.error('Failed to load popular movies', e);
    }
  },

  async fetchFeed() {
    try {
      const data = await api.getFeed(8);
      // /feed now returns { page, limit, total, results } — handle both shapes
      set({ feed: Array.isArray(data) ? data : (data.results || []) });
    } catch (e) {
      console.error('Failed to load activity feed', e);
    }
  },

  async fetchRecommendations() {
    const { userId, weightCol, novelty, diversity } = get();
    if (!userId) return;
    set({ isLoading: true, errorMsg: '' });
    try {
      const data = await api.getRecommendations(userId, { weightCol, novelty, diversity });
      set({ recommendations: data });
    } catch (e) {
      set({ errorMsg: e.status === 401 ? 'Session expired — please sign in again.' : 'Cannot connect to API server.' });
    } finally {
      set({ isLoading: false });
    }
  },

  async handleSearch(query) {
    set({ searchQuery: query });
    if (!query.trim()) { set({ searchResults: [] }); return; }
    try {
      const data = await api.searchMovies(query, 5);
      set({ searchResults: data });
    } catch (e) {
      console.error('Search failed', e);
    }
  },

  // ── Actions: Interactions ───────────────────────────────────────────────────

  async rateMovie(movieId, rating) {
    const { userId } = get();
    if (!userId) return;
    try {
      await api.submitRating(userId, movieId, rating);
      set((s) => ({ sessionRatings: { ...s.sessionRatings, [movieId]: rating } }));
      get().fetchRecommendations();
      get().fetchStats();
      get().fetchFeed();
    } catch (e) {
      console.error('Rating submission failed', e);
    }
  },

  async deleteMovie(movieId) {
    try {
      await api.deleteMovie(movieId);
      set((s) => ({
        searchResults: s.searchResults.filter((m) => m.movieId !== movieId),
        popularMovies: s.popularMovies.filter((m) => m.movieId !== movieId),
        recommendations: s.recommendations.filter((m) => m.movieId !== movieId),
      }));
      get().fetchStats();
    } catch (e) {
      console.error('Archiving movie failed', e);
    }
  },

  async loadUserRatings(userId) {
    try {
      const data = await api.getUserRatings(userId);
      const hasRatings = Object.keys(data).length > 0;
      localStorage.setItem('hasOnboarded', hasRatings ? 'true' : 'false');
      set({ sessionRatings: data, hasOnboarded: hasRatings });
    } catch (e) {
      console.error('Failed to load user ratings', e);
    }
  },

  // ── Actions: Controls ───────────────────────────────────────────────────────
  setWeightCol: (v) => set({ weightCol: v }),
  setNovelty: (v) => set({ novelty: v }),
  setDiversity: (v) => set({ diversity: v }),

  // ── Actions: Modal ──────────────────────────────────────────────────────────
  openLoginModal: () => set({ isLoginModalOpen: true }),
  closeLoginModal: () => set({ isLoginModalOpen: false }),
}));

export default useAppStore;
