"""Feed router — live activity feed and system stats."""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models_db import DBRating, DBMovie

router = APIRouter(prefix="", tags=["Feed"])


@router.get("/stats")
def get_stats(request: Request, db: Session = Depends(get_db)):
    """Returns SVD model diagnostics, rating counts, and evaluation metrics."""
    from backend.app.main import app

    db_ratings_count = db.query(DBRating).count()
    db_movies_count = db.query(DBMovie).filter(DBMovie.is_active == True).count()
    total_ratings = len(app.state.ratings_df)
    total_users = len(app.state.col_model.user_mapper)

    return {
        "total_ratings": total_ratings,
        "live_ratings": db_ratings_count,
        "users_count": total_users,
        "movies_count": db_movies_count,
        "svd_components": app.state.col_model.n_factors,
        "metrics": {
            "rmse": float(app.state.cache["metrics"]["rmse"]),
            "ndcg_10": float(app.state.cache["metrics"]["ndcg_10"]),
            "map_10": float(app.state.cache["metrics"]["map_10"]),
        },
    }


@router.get("/feed")
def get_global_feed(request: Request, db: Session = Depends(get_db), limit: int = 10, page: int = 1):
    """Returns paginated live rating activity from all users (network stream).

    Args:
        limit: Entries per page (max 50).
        page:  Page number (1-indexed).
    """
    limit = min(limit, 50)
    offset = (page - 1) * limit

    results = (
        db.query(DBRating, DBMovie)
        .join(DBMovie, DBRating.movieId == DBMovie.movieId)
        .order_by(DBRating.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    total = db.query(DBRating).count()
    feed = [
        {
            "id": r.id,
            "userId": r.userId,
            "movieId": r.movieId,
            "title": m.title,
            "rating": r.rating,
            "timestamp": r.timestamp,
        }
        for r, m in results
    ]

    return {"page": page, "limit": limit, "total": total, "results": feed}
