"""
Main FastAPI application — startup, middleware, and router mounting.
All business logic lives in backend/app/routers/*.py
"""
import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

# ── Load environment variables from backend/.env ──────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_env_path)

from backend.app.database import engine, SessionLocal, Base
from backend.app.models_db import DBMovie, DBRating
from backend.src.models import HybridRecommender
from backend.app.dependencies import limiter
from backend.app.routers import auth_router, movies_router, recommendations_router, feed_router, admin_router

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="🎬 Production Recommendation Engine API",
    description=(
        "FastAPI Backend with Real-Time Online SVD learning, "
        "SQLite persistence, and Hybrid TF-IDF + Collaborative filtering."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — read allowed origins from .env ─────────────────────────────────────
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-ID", "X-Request-ID"],
)

# ── Mount versioned routers (/api/v1/...) ─────────────────────────────────────
app.include_router(auth_router, prefix="/api/v1")
app.include_router(movies_router, prefix="/api/v1")
app.include_router(recommendations_router, prefix="/api/v1")
app.include_router(feed_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")

# ── Backward-compatible legacy /api/ aliases ──────────────────────────────────
# Expose same handlers under /api/... to prevent breaking existing clients.
app.include_router(auth_router, prefix="/api")
app.include_router(movies_router, prefix="/api")
app.include_router(recommendations_router, prefix="/api")
app.include_router(feed_router, prefix="/api")
app.include_router(admin_router, prefix="/api")


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


# ── Startup: load models and sync database ────────────────────────────────────
@app.on_event("startup")
def startup_event():
    if os.environ.get("TESTING") == "True":
        return
    Base.metadata.create_all(bind=engine)

    col_path = os.path.join(MODELS_DIR, "collaborative_model.pkl")
    content_path = os.path.join(MODELS_DIR, "content_model.pkl")
    cache_path = os.path.join(MODELS_DIR, "data_cache.pkl")

    if not (os.path.exists(col_path) and os.path.exists(content_path) and os.path.exists(cache_path)):
        raise RuntimeError("Cached models not found in models/ directory. Run batch training first.")

    print("[Startup] Loading batch-trained models and dataset cache...")
    with open(col_path, "rb") as f:
        app.state.col_model = pickle.load(f)
    with open(content_path, "rb") as f:
        app.state.content_model = pickle.load(f)
    with open(cache_path, "rb") as f:
        app.state.cache = pickle.load(f)

    db = SessionLocal()
    try:
        # Seed movies table from cache if empty
        if db.query(DBMovie).count() == 0:
            print("[Startup] Seeding database movies table from cached data...")
            movies_to_insert = []
            for _, row in app.state.cache["movies"].iterrows():
                title_str = str(row.get("title", ""))
                genres_str = str(row.get("genres", ""))
                metadata_text = row.get("metadata_text", f"{title_str} {genres_str.replace('|', ' ')}")
                links_df = app.state.cache["links"]
                tmdb_row = links_df[links_df["movieId"] == row["movieId"]]
                tmdb_id = (
                    int(tmdb_row["tmdbId"].values[0])
                    if not tmdb_row.empty and not pd.isna(tmdb_row["tmdbId"].values[0])
                    else None
                )
                movies_to_insert.append(
                    DBMovie(
                        movieId=int(row["movieId"]),
                        title=title_str,
                        genres=genres_str,
                        metadata_text=str(metadata_text),
                        tmdbId=tmdb_id,
                        is_active=True,
                    )
                )
            db.bulk_save_objects(movies_to_insert)
            db.commit()
            print(f"[Startup] Seeded {len(movies_to_insert)} movies.")

        # Load catalog into memory
        all_movies = db.query(DBMovie).all()
        app.state.movies_df = pd.DataFrame([{
            "movieId": m.movieId,
            "title": m.title,
            "genres": m.genres,
            "metadata_text": m.metadata_text,
            "tmdbId": m.tmdbId,
            "poster_path": m.poster_path,
            "is_active": m.is_active,
        } for m in all_movies])

        # Fit ContentModel with full current catalog
        app.state.content_model.fit(app.state.movies_df)

        # Sync CollaborativeModel movie mappings
        for m in all_movies:
            app.state.col_model.register_new_movie(m.movieId)

        # Replay live ratings from DB to update SVD in-memory
        db_ratings = db.query(DBRating).order_by(DBRating.timestamp.asc()).all()
        print(f"[Startup] Replaying {len(db_ratings)} live interactions...")
        for r in db_ratings:
            app.state.col_model.update_rating_online(r.userId, r.movieId, r.rating)

        # Build composite ratings DataFrame (base dataset + live DB ratings)
        base_ratings = app.state.cache["ratings"].copy()
        base_ratings["userId"] = base_ratings["userId"].astype(str)

        if db_ratings:
            live_df = pd.DataFrame([{
                "userId": str(r.userId),
                "movieId": int(r.movieId),
                "rating": float(r.rating),
                "timestamp": int(r.timestamp),
            } for r in db_ratings])
            app.state.ratings_df = pd.concat([base_ratings, live_df], ignore_index=True)
        else:
            app.state.ratings_df = base_ratings

        app.state.hybrid_recommender = HybridRecommender(app.state.col_model, app.state.content_model)
        print("[Startup] Backend successfully synchronized and ready.")

        # Start background daemon thread for 24-hour model retraining
        import threading
        import time
        def schedule_retrain_loop():
            # Retrain every 24 hours (86400 seconds)
            interval = 86400
            while True:
                time.sleep(interval)
                print("[Scheduler] Starting automatic 24-hour model retraining...")
                try:
                    from backend.src.retrain import retrain_model_pipeline
                    retrain_model_pipeline(app=app)
                    print("[Scheduler] Automatic model retraining completed successfully.")
                except Exception as e:
                    print(f"[Scheduler] Automatic retraining failed: {e}")

        retrain_thread = threading.Thread(target=schedule_retrain_loop, daemon=True)
        retrain_thread.start()
        print("[Startup] Background retrain scheduler thread started.")

    finally:
        db.close()
