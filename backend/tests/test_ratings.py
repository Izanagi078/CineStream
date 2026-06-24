"""
test_ratings.py — Tests for /api/ratings and /api/users/{userId}/ratings.
Covers: guest rating, registered user rating (auth required), unauthorized, user ratings fetch.
"""
import pytest


def register_and_login(client, username="rater", password="raterpass"):
    client.post("/api/auth/register", json={"username": username, "password": password})
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    return res.json()["token"]


class TestSubmitRating:
    def test_guest_can_rate(self, client):
        res = client.post("/api/ratings", json={
            "userId": "guest_abc123",
            "movieId": 1,
            "rating": 5.0
        })
        assert res.status_code == 200
        assert "message" in res.json()

    def test_registered_user_rating_requires_token(self, client):
        res = client.post("/api/ratings", json={
            "userId": "some_registered_user",
            "movieId": 1,
            "rating": 4.0
        })
        assert res.status_code == 401

    def test_registered_user_rating_with_valid_token(self, client):
        token = register_and_login(client, "rater1", "pass1234")
        res = client.post(
            "/api/ratings",
            json={"userId": "rater1", "movieId": 1, "rating": 4.0},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert res.status_code == 200

    def test_cannot_rate_as_different_user(self, client):
        """Token for userX cannot submit a rating on behalf of userY."""
        token = register_and_login(client, "userX", "passX1234")
        res = client.post(
            "/api/ratings",
            json={"userId": "userY", "movieId": 1, "rating": 3.0},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert res.status_code == 401

    def test_rating_triggers_svd_update(self, client):
        """After a successful rating, SVD update_rating_online should have been called."""
        client.post("/api/ratings", json={
            "userId": "guest_trigger",
            "movieId": 1,
            "rating": 5.0
        })
        # col_model is a MagicMock — verify it was called
        from backend.app.main import app
        assert app.state.col_model.update_rating_online.called

    def test_dislike_rating(self, client):
        """Rating of 1.0 (dislike) should also be accepted."""
        res = client.post("/api/ratings", json={
            "userId": "guest_dislike",
            "movieId": 2,
            "rating": 1.0
        })
        assert res.status_code == 200


class TestGetUserRatings:
    def test_returns_dict(self, client):
        res = client.get("/api/users/guest_abc/ratings")
        assert res.status_code == 200
        assert isinstance(res.json(), dict)

    def test_empty_for_unknown_user(self, client):
        res = client.get("/api/users/unknown_ghost_user/ratings")
        assert res.status_code == 200
        assert res.json() == {}

    def test_ratings_appear_after_submission(self, client):
        client.post("/api/ratings", json={
            "userId": "guest_history",
            "movieId": 1,
            "rating": 5.0
        })
        res = client.get("/api/users/guest_history/ratings")
        assert res.status_code == 200
        data = res.json()
        assert 1 in data or "1" in data  # movieId key (int or str)
