import os
os.environ["TESTING"] = "True"

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from backend.app.dependencies import limiter
limiter.enabled = False

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from unittest.mock import MagicMock

from backend.app.database import Base, get_db
from backend.app.models_db import DBMovie, DBRating, DBUser
from backend.app.main import app


# ── In-memory SQLite engine (isolated per test session) ───────────────────────
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Stub ML Models ─────────────────────────────────────────────────────────────
def make_stub_col_model():
    """Returns a minimal stub CollaborativeModel with required interface."""
    m = MagicMock()
    m.n_factors = 20
    m.user_mapper = {"User 10": 0, "User 50": 1}
    m.update_rating_online = MagicMock()
    m.register_new_movie = MagicMock()
    # predict returns a middling score regardless of input
    m.predict = MagicMock(return_value=3.5)
    return m


def make_stub_content_model():
    """Returns a minimal stub ContentModel with required interface."""
    m = MagicMock()
    m.vectorizer = MagicMock()
    m.vectorizer.transform = MagicMock(return_value=MagicMock(toarray=lambda: np.zeros((1, 10))))
    m.tfidf_matrix = MagicMock()
    m.tfidf_matrix.toarray = MagicMock(return_value=np.zeros((5, 10)))
    m.movie_idx_to_id = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    m.fit = MagicMock()
    m.register_new_movie = MagicMock()
    return m


def make_stub_hybrid(col, content):
    m = MagicMock()
    # Returns a 2-row DataFrame simulating recommendations
    m.get_recommendations = MagicMock(return_value=pd.DataFrame([
        {"movieId": 1, "title": "Test Movie A", "genres": "Action", "score": 0.85, "hits": 10},
        {"movieId": 2, "title": "Test Movie B", "genres": "Drama",  "score": 0.72, "hits": 5},
    ]))
    m.explain_recommendation = MagicMock(return_value={"reason": "stub explanation"})
    return m


# ── Seed movies and ratings ────────────────────────────────────────────────────
SEED_MOVIES = [
    {"movieId": 1, "title": "Test Movie A", "genres": "Action",  "metadata_text": "action hero", "is_active": True},
    {"movieId": 2, "title": "Test Movie B", "genres": "Drama",   "metadata_text": "drama sad",   "is_active": True},
    {"movieId": 3, "title": "Archived Film","genres": "Horror",  "metadata_text": "horror dark",  "is_active": False},
]


@pytest.fixture(scope="function")
def db():
    """Fresh in-memory DB for each test function."""
    Base.metadata.create_all(bind=test_engine)
    session = TestSessionLocal()
    # Seed movies
    for m in SEED_MOVIES:
        session.add(DBMovie(**m))
    session.commit()
    yield session
    session.close()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db):
    """TestClient with DB override and stubbed ML app.state."""
    app.dependency_overrides[get_db] = override_get_db

    # Inject stub app state
    col = make_stub_col_model()
    content = make_stub_content_model()
    hybrid = make_stub_hybrid(col, content)

    app.state.col_model = col
    app.state.content_model = content
    app.state.hybrid_recommender = hybrid
    app.state.cache = {
        "metrics": {"rmse": 0.85, "ndcg_10": 0.42, "map_10": 0.31},
    }
    app.state.movies_df = pd.DataFrame(SEED_MOVIES)
    app.state.ratings_df = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
