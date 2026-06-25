import React, { useState } from 'react';

export default function MovieShelf({ title, movies, onRate, onDelete, sessionRatings }) {
  const [openExplanations, setOpenExplanations] = useState({});

  const toggleExplanation = (movieId) => {
    setOpenExplanations((prev) => ({
      ...prev,
      [movieId]: !prev[movieId]
    }));
  };

  const getPrimaryGenre = (genresStr) => {
    const list = genresStr.split('|');
    return list.length > 0 ? list[0] : 'General';
  };

  const extractYear = (titleStr) => {
    const match = titleStr.match(/\((\d{4})\)/);
    return match ? match[1] : 'N/A';
  };

  const cleanTitle = (titleStr) => {
    return titleStr.replace(/\s*\(\d{4}\)/, '');
  };

  return (
    <div className="shelf-container">
      <h2 style={{ fontSize: '1.3rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
        {title}
      </h2>

      {movies.length === 0 ? (
        <div className="glass-card" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-secondary)' }}>
          No movies matching your criteria in this shelf.
        </div>
      ) : (
        <div className="shelf-cards">
          {movies.map((movie) => {
            const primaryGenre = getPrimaryGenre(movie.genres);
            const year = extractYear(movie.title);
            const titleClean = cleanTitle(movie.title);
            const userRating = sessionRatings?.[movie.movieId];
            const isLiked = userRating === 5.0;
            const isDisliked = userRating === 1.0;

            // Generate deterministic gradient background based on movieId
            const gradients = [
              'linear-gradient(135deg, #13122c 0%, #2a0e35 100%)',
              'linear-gradient(135deg, #0a1128 0%, #1c325c 100%)',
              'linear-gradient(135deg, #240a34 0%, #511257 100%)',
              'linear-gradient(135deg, #01161e 0%, #124559 100%)',
              'linear-gradient(135deg, #3c0919 0%, #1e112c 100%)'
            ];
            const bgGrad = gradients[movie.movieId % gradients.length];

            return (
              <div
                key={movie.movieId}
                className="glass-card movie-card"
                style={{
                  background: bgGrad,
                  border: isLiked ? '1px solid var(--accent-emerald)' : isDisliked ? '1px solid var(--accent-rose)' : '1px solid var(--border-color)',
                  paddingTop: movie.poster_url ? '12px' : '18px'
                }}
              >
                {/* Poster Media */}
                {movie.poster_url ? (
                  <div style={{
                    width: '100%',
                    height: '190px',
                    borderRadius: '8px',
                    overflow: 'hidden',
                    marginBottom: '12px',
                    position: 'relative',
                    boxShadow: '0 8px 20px rgba(0, 0, 0, 0.4)',
                    background: 'rgba(0, 0, 0, 0.2)'
                  }}>
                    <img
                      src={movie.poster_url}
                      alt={movie.title}
                      className="movie-poster-img"
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        transition: 'transform 0.4s cubic-bezier(0.16, 1, 0.3, 1)'
                      }}
                    />
                  </div>
                ) : null}

                {/* Header info */}
                <div className="movie-card-header">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className="movie-genre">{primaryGenre}</span>
                    {onDelete && (
                      <button
                        title="Archive movie"
                        style={{ background: 'transparent', padding: '2px 4px', fontSize: '0.8rem', color: 'rgba(255,255,255,0.4)', borderRadius: '4px' }}
                        onClick={() => onDelete(movie.movieId)}
                      >
                        🗑️
                      </button>
                    )}
                  </div>
                  <div className="movie-title" title={movie.title}>{titleClean}</div>
                </div>

                {/* Score tags */}
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '10px' }}>
                  {movie.final_score !== undefined && (
                    <span style={{ fontSize: '0.75rem', padding: '2px 6px', background: 'rgba(99, 102, 241, 0.25)', border: '1px solid rgba(99, 102, 241, 0.4)', borderRadius: '4px' }}>
                      Score: <strong>{movie.final_score.toFixed(2)}</strong>
                    </span>
                  )}
                  {movie.popularity_hits !== undefined && (
                    <span style={{ fontSize: '0.75rem', padding: '2px 6px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '4px', color: 'var(--text-secondary)' }}>
                      Hits: {movie.popularity_hits}
                    </span>
                  )}
                </div>

                {/* Explanations expander */}
                {movie.explanation && (
                  <div className="xai-expander">
                    <div className="xai-title" onClick={() => toggleExplanation(movie.movieId)}>
                      🔎 Why Recommended? {openExplanations[movie.movieId] ? '▲' : '▼'}
                    </div>

                    {openExplanations[movie.movieId] && (
                      <div className="xai-content">
                        {movie.explanation.collaborative ? (
                          <div>
                            💡 Latent fit matching <strong>{movie.explanation.collaborative.liked_movie_title}</strong> (similarity: <code>{movie.explanation.collaborative.latent_similarity.toFixed(2)}</code>).
                          </div>
                        ) : null}
                        {movie.explanation.content ? (
                          <div>
                            🏷️ Shared keywords: <code>{movie.explanation.content.overlapping_words.join(', ')}</code>.
                          </div>
                        ) : null}
                        {!movie.explanation.collaborative && !movie.explanation.content ? (
                          <div>Recommended based on global catalog popularity.</div>
                        ) : null}
                      </div>
                    )}
                  </div>
                )}

                {/* Footer metadata & buttons */}
                <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div className="movie-meta">
                    <span>ID: {movie.movieId}</span>
                    <span style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '4px' }}>{year}</span>
                  </div>

                  <div className="card-actions">
                    <button
                      className="btn-icon like"
                      onClick={() => onRate(movie.movieId, 5.0)}
                      style={{
                        background: isLiked ? 'var(--accent-emerald)' : '',
                        borderColor: isLiked ? 'var(--accent-emerald)' : '',
                        color: isLiked ? 'white' : ''
                      }}
                      disabled={isLiked || isDisliked}
                    >
                      Like 👍
                    </button>
                    <button
                      className="btn-icon dislike"
                      onClick={() => onRate(movie.movieId, 1.0)}
                      style={{
                        background: isDisliked ? 'var(--accent-rose)' : '',
                        borderColor: isDisliked ? 'var(--accent-rose)' : '',
                        color: isDisliked ? 'white' : ''
                      }}
                      disabled={isLiked || isDisliked}
                    >
                      Dislike 👎
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
