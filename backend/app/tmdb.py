import os
import math
import requests
from typing import Optional
from sqlalchemy.orm import Session
from backend.app.models_db import DBMovie

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
TMDB_ACCESS_TOKEN = os.environ.get("TMDB_ACCESS_TOKEN", "").strip()


def is_valid_poster_path(val) -> bool:
    if not val:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    if not isinstance(val, str):
        return False
    return True


def is_valid_tmdb_id(val) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    try:
        float_val = float(val)
        return float_val > 0
    except (ValueError, TypeError):
        return False


def enrich_movie_poster(movie_dict: dict, db: Session) -> dict:
    """
    Enriches a movie dictionary with a poster_url key.
    If poster_path is not cached in the DB, it queries the TMDB API,
    saves the fetched path to the database and active in-memory dataframe,
    and constructs the complete image asset URL.
    """
    from backend.app.main import app

    tmdb_id = movie_dict.get("tmdbId")
    movie_id = movie_dict.get("movieId")
    poster_path = movie_dict.get("poster_path")

    # If already have poster_path, construct poster_url
    if is_valid_poster_path(poster_path):
        movie_dict["poster_path"] = poster_path
        movie_dict["poster_url"] = f"https://image.tmdb.org/t/p/w342{poster_path}"
        return movie_dict

    # Check in-memory dataframe first in case it was updated by another thread
    if app is not None and hasattr(app, "state") and hasattr(app.state, "movies_df") and movie_id is not None:
        matches = app.state.movies_df[app.state.movies_df["movieId"] == movie_id]
        if not matches.empty:
            df_poster = matches.iloc[0].get("poster_path")
            if is_valid_poster_path(df_poster):
                movie_dict["poster_path"] = df_poster
                movie_dict["poster_url"] = f"https://image.tmdb.org/t/p/w342{df_poster}"
                return movie_dict

    # Try to fetch from TMDB if API key or access token is provided and we have a valid tmdbId
    if is_valid_tmdb_id(tmdb_id) and (TMDB_API_KEY or TMDB_ACCESS_TOKEN):
        try:
            headers = {}
            if TMDB_ACCESS_TOKEN:
                headers["Authorization"] = f"Bearer {TMDB_ACCESS_TOKEN}"
                url = f"https://api.tmdb.org/3/movie/{int(float(tmdb_id))}"
            else:
                url = f"https://api.tmdb.org/3/movie/{int(float(tmdb_id))}?api_key={TMDB_API_KEY}"
            
            response = requests.get(url, headers=headers, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                fetched_path = data.get("poster_path")
                if is_valid_poster_path(fetched_path):
                    # Update database DBMovie row
                    db_movie = db.query(DBMovie).filter(DBMovie.movieId == movie_id).first()
                    if db_movie:
                        db_movie.poster_path = fetched_path
                        db.commit()

                    # Update in-memory state movies_df
                    if app is not None and hasattr(app, "state") and hasattr(app.state, "movies_df"):
                        app.state.movies_df.loc[app.state.movies_df["movieId"] == movie_id, "poster_path"] = fetched_path

                    movie_dict["poster_path"] = fetched_path
                    movie_dict["poster_url"] = f"https://image.tmdb.org/t/p/w342{fetched_path}"
                    return movie_dict
        except Exception as e:
            print(f"[TMDB] Error fetching poster for tmdbId {tmdb_id}: {e}")

    movie_dict["poster_url"] = None
    return movie_dict
