"""Movies router — popular, search, add, delete."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
import pandas as pd

from backend.app.database import get_db
from backend.app.models_db import DBMovie
from backend.app.schemas import MovieCreate
from backend.app.dependencies import get_current_user_optional
from backend.app.cache import cache_dec, invalidate_all_caches

from backend.app.tmdb import enrich_movie_poster

router = APIRouter(prefix="/movies", tags=["Movies"])


@router.get("/popular")
@cache_dec("popular_movies", maxsize=64, ttl=300)
def get_popular_movies(request: Request, limit: int = 10, page: int = 1, db: Session = Depends(get_db)):
    """Returns globally popular movies ranked by rating count.

    Args:
        limit: Number of results per page (max 50).
        page:  Page number (1-indexed).
    """
    limit = min(limit, 50)
    offset = (page - 1) * limit

    from backend.app.main import app
    popularity = app.state.ratings_df.groupby("movieId").size().reset_index(name="hits")
    merged = popularity.merge(app.state.movies_df, on="movieId")
    active = merged[merged["is_active"] == True]
    popular = active.sort_values(by="hits", ascending=False)

    total = len(popular)
    page_slice = popular.iloc[offset: offset + limit]
    records = page_slice.to_dict(orient="records")
    for r in records:
        enrich_movie_poster(r, db)

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "results": records,
    }


@router.get("/search")
def search_movies(request: Request, query: str, limit: int = 10, db: Session = Depends(get_db)):
    """Searches active movies by title substring."""
    if not query:
        return []

    from backend.app.main import app
    active = app.state.movies_df[app.state.movies_df["is_active"] == True]
    matches = active[active["title"].str.contains(query, case=False, na=False)].head(limit)
    records = matches.to_dict(orient="records")
    for r in records:
        enrich_movie_poster(r, db)
    return records


@router.post("")
def add_movie(
    request: Request,
    movie: MovieCreate,
    db: Session = Depends(get_db),
    current_user: Optional[str] = Depends(get_current_user_optional),
):
    """Admin: Dynamically add a new movie to catalog and index in SVD + TF-IDF."""
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: A valid auth token is required to modify catalog.",
        )

    from backend.app.main import app
    max_id = db.query(DBMovie).order_by(DBMovie.movieId.desc()).first()
    new_movie_id = (max_id.movieId + 1) if max_id else 1
    metadata_text = f"{movie.title} {movie.genres.replace('|', ' ')}"

    db_movie = DBMovie(
        movieId=new_movie_id,
        title=movie.title,
        genres=movie.genres,
        metadata_text=metadata_text,
        tmdbId=None,
        is_active=True,
    )
    db.add(db_movie)
    db.commit()
    db.refresh(db_movie)

    app.state.col_model.register_new_movie(new_movie_id)
    app.state.content_model.register_new_movie(new_movie_id, movie.title, movie.genres, metadata_text)

    new_row = pd.DataFrame([{
        "movieId": new_movie_id,
        "title": movie.title,
        "genres": movie.genres,
        "metadata_text": metadata_text,
        "tmdbId": None,
        "is_active": True,
    }])
    app.state.movies_df = pd.concat([app.state.movies_df, new_row], ignore_index=True)

    # Invalidate cache since movie count changed
    invalidate_all_caches()

    return db_movie


@router.delete("/{movieId}")
def delete_movie(
    request: Request,
    movieId: int,
    db: Session = Depends(get_db),
    current_user: Optional[str] = Depends(get_current_user_optional),
):
    """Admin: Soft-delete (archive) a movie so it no longer appears in recommendations."""
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: A valid auth token is required to modify catalog.",
        )

    from backend.app.main import app
    db_movie = db.query(DBMovie).filter(DBMovie.movieId == movieId).first()
    if not db_movie:
        raise HTTPException(status_code=404, detail="Movie not found.")

    db_movie.is_active = False
    db.commit()
    app.state.movies_df.loc[app.state.movies_df["movieId"] == movieId, "is_active"] = False

    # Invalidate cache since active movie count and list changed
    invalidate_all_caches()

    return {"message": f"Movie {movieId} archived successfully."}
