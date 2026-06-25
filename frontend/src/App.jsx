import React, { useEffect } from 'react';
import SidebarSettings from './components/SidebarSettings';
import LiveFeed from './components/LiveFeed';
import Onboarding from './components/Onboarding';
import MovieShelf from './components/MovieShelf';
import VisualCharts from './components/VisualCharts';
import LoginModal from './components/LoginModal';
import useAppStore from './store/useAppStore';

export default function App() {
  // Pull everything from the Zustand store
  const {
    userId, hasOnboarded,
    recommendations, popularMovies, feed, stats,
    weightCol, novelty, diversity,
    searchQuery, searchResults, sessionRatings,
    isLoading, errorMsg, isLoginModalOpen,
    // Actions
    fetchStats, fetchPopular, fetchFeed, fetchRecommendations,
    handleSearch, rateMovie, deleteMovie, loadUserRatings,
    setWeightCol, setNovelty, setDiversity,
    setLogin, resetSession, completeOnboarding,
    openLoginModal, closeLoginModal,
  } = useAppStore();

  // 1. Initial page load — fetch stats, popular, feed
  useEffect(() => {
    fetchStats();
    fetchPopular();
    fetchFeed();

    // Poll feed + stats every 5 seconds for real-time multiplayer updates
    const interval = setInterval(() => {
      fetchFeed();
      fetchStats();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // 2. Fetch recommendations when userId or algorithm params change
  useEffect(() => {
    if (userId) fetchRecommendations();
  }, [userId, weightCol, novelty, diversity]);

  // 2b. Load user ratings history when session starts or changes
  useEffect(() => {
    if (userId) loadUserRatings(userId);
  }, [userId]);

  // 3. Debounced search
  useEffect(() => {
    const timer = setTimeout(() => handleSearch(searchQuery), 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Login handler — also loads the user's existing ratings
  const handleLogin = async (username, token) => {
    setLogin(username, token);
    await loadUserRatings(username);
  };

  const refreshCatalog = () => {
    fetchPopular();
    fetchStats();
    if (userId) fetchRecommendations();
  };

  return (
    <div className="app-container">
      {/* Heavenly background glowing blobs */}
      <div className="glow-orb glow-orb-1"></div>
      <div className="glow-orb glow-orb-2"></div>

      {/* Left Sidebar: Algorithm Controls */}
      <SidebarSettings
        weightCol={weightCol}
        setWeightCol={setWeightCol}
        novelty={novelty}
        setNovelty={setNovelty}
        diversity={diversity}
        setDiversity={setDiversity}
        userId={userId}
        onReset={resetSession}
        onAddMovieSuccess={refreshCatalog}
      />

      {/* Main Content: Center Column */}
      <main className="main-content">
        <div>
          <h1>🎬 Personal Recommendations Portal</h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Real-time SVD matrix factorizer &amp; content keyword TF-IDF ensembled engine.
          </p>
        </div>

        {!hasOnboarded ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', marginTop: '20px' }}>
            {!userId && (
              <p style={{ color: 'var(--text-secondary)', textAlign: 'center', maxWidth: '500px', fontSize: '0.85rem' }}>
                Browse as a guest by filling out preferences below, or use the <strong>Session Profile</strong> on the right to sign in.
              </p>
            )}
            {userId && (
              <p style={{ color: 'var(--text-secondary)', textAlign: 'center', maxWidth: '500px', fontSize: '0.85rem' }}>
                Welcome, <strong>{userId}</strong>! Choose your interests to bootstrap your SVD latent vector.
              </p>
            )}
            <Onboarding onSubmit={completeOnboarding} currentUserId={userId} />
          </div>
        ) : (
          <>
            {/* Stats Banner */}
            {stats && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px' }}>
                {[
                  { label: 'RMSE (Accuracy)', value: stats.metrics.rmse.toFixed(4), color: 'var(--accent-emerald)' },
                  { label: 'NDCG@10 Rank', value: `${(stats.metrics.ndcg_10 * 100).toFixed(2)}%`, color: '#ffffff' },
                  { label: 'MAP@10 Precision', value: `${(stats.metrics.map_10 * 100).toFixed(2)}%`, color: '#ffffff' },
                  { label: 'Model Users (SVD)', value: stats.users_count, color: '#ffffff' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '14px' }}>
                    <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>{label}</span>
                    <span style={{ fontSize: '1.4rem', fontWeight: 'bold', color, marginTop: '4px' }}>{value}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Search */}
            <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <h3 style={{ fontSize: '0.85rem' }}>🔍 Search Catalog &amp; Rate</h3>
              <input
                type="text"
                placeholder="Search movies to rate and instantly influence SVD vectors..."
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                style={{
                  width: '100%', padding: '12px', borderRadius: '6px',
                  border: '1px solid var(--border-color)', background: 'rgba(0,0,0,0.2)',
                  color: 'white', outline: 'none', fontFamily: 'var(--font-family)'
                }}
              />
              {searchResults.length > 0 && (
                <MovieShelf title="Search Results" movies={searchResults} onRate={rateMovie} onDelete={deleteMovie} sessionRatings={sessionRatings} />
              )}
            </div>

            {/* Recommendations */}
            {isLoading ? (
              <div className="glass-card" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                Re-indexing user latent matrix coordinates...
              </div>
            ) : errorMsg ? (
              <div className="glass-card" style={{ padding: '24px', textAlign: 'center', color: 'var(--accent-rose)' }}>
                {errorMsg}
              </div>
            ) : (
              <MovieShelf title="💡 Recommended For You" movies={recommendations} onRate={rateMovie} onDelete={deleteMovie} sessionRatings={sessionRatings} />
            )}

            {/* Popular */}
            <MovieShelf title="🔥 Popular Right Now" movies={popularMovies} onRate={rateMovie} onDelete={deleteMovie} sessionRatings={sessionRatings} />

            {/* Charts */}
            {stats && <VisualCharts stats={stats} />}
          </>
        )}
      </main>

      {/* Right Sidebar: Session Profile + Live Feed */}
      <aside className="right-sidebar">
        <div className="glass-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '20px' }}>
          <h4 style={{ fontSize: '0.85rem', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>👤 Session Profile</h4>
          {userId ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <p style={{ fontSize: '0.8rem', wordBreak: 'break-all' }}>
                ID: <code style={{ color: 'var(--accent-emerald)', fontWeight: 'bold' }}>{userId}</code>
              </p>
              <button className="btn-primary" style={{ padding: '6px', fontSize: '0.8rem', background: '#25283c' }} onClick={resetSession}>
                Sign Out 🚪
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Browsing as Guest</p>
              <button className="btn-primary" style={{ padding: '6px', fontSize: '0.8rem' }} onClick={openLoginModal}>
                Sign In / Register 🔑
              </button>
            </div>
          )}
        </div>
        <LiveFeed feed={feed} />
      </aside>

      {/* Auth Modal */}
      <LoginModal isOpen={isLoginModalOpen} onClose={closeLoginModal} onLogin={handleLogin} />
    </div>
  );
}
