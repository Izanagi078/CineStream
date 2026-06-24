"""
test_ml_math.py — Unit tests for ML recommendation math.
Covers: CollaborativeModel SVD fitting/prediction/online updates, ContentModel TF-IDF vectorization.
"""
import numpy as np
import pandas as pd
import pytest
from backend.src.models import CollaborativeModel, ContentModel

def test_collaborative_model_fit_and_update():
    # 1. Create a dummy ratings dataset
    ratings_data = pd.DataFrame([
        {"userId": "1", "movieId": 101, "rating": 5.0, "timestamp": 1000000},
        {"userId": "1", "movieId": 102, "rating": 3.0, "timestamp": 1000000},
        {"userId": "2", "movieId": 101, "rating": 4.0, "timestamp": 1000000},
        {"userId": "2", "movieId": 103, "rating": 2.0, "timestamp": 1000000},
        {"userId": "3", "movieId": 102, "rating": 4.5, "timestamp": 1000000},
        {"userId": "3", "movieId": 103, "rating": 5.0, "timestamp": 1000000},
    ])

    model = CollaborativeModel(n_factors=2)
    model.fit(ratings_data, apply_decay=False)

    assert model.P is not None
    assert model.Q is not None
    assert model.P.shape[0] == 3  # 3 users
    assert model.Q.shape[0] == 3  # 3 movies

    # 2. Predict rating
    pred = model.predict_rating("1", 101)
    assert 0.5 <= pred <= 5.0

    # 3. Test online SVD update
    u_idx = model.user_mapper["1"]
    m_idx = model.movie_mapper[103]

    p_before = model.P[u_idx].copy()
    q_before = model.Q[m_idx].copy()

    # Perform online SGD update
    model.update_rating_online("1", 103, 5.0)

    p_after = model.P[u_idx]
    q_after = model.Q[m_idx]

    # Latent vectors must shift
    assert not np.array_equal(p_before, p_after)
    assert not np.array_equal(q_before, q_after)

def test_content_model_tfidf():
    # 1. Create dummy movie dataset
    movies_data = pd.DataFrame([
        {"movieId": 101, "title": "Space Battles", "genres": "Sci-Fi|Action", "metadata_text": "Space Battles Sci-Fi Action"},
        {"movieId": 102, "title": "Love Story", "genres": "Romance|Drama", "metadata_text": "Love Story Romance Drama"},
        {"movieId": 103, "title": "Funny Movie", "genres": "Comedy", "metadata_text": "Funny Movie Comedy"},
    ])

    model = ContentModel()
    model.fit(movies_data)

    assert model.tfidf_matrix is not None
    assert model.tfidf_matrix.shape[0] == 3
    # Check vocabulary size is positive
    assert len(model.vectorizer.vocabulary_) > 0

    # 2. Add a new movie dynamically
    new_idx = model.register_new_movie(104, "Dark Space", "Sci-Fi|Horror")
    assert new_idx == 3
    assert model.tfidf_matrix.shape[0] == 4
    assert 104 in model.movie_id_to_idx
