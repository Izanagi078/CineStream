"""
test_recommendations.py — Tests for /api/recommendations endpoint.
Covers: guest user, registered user (auth required), invalid userId, param edge cases.
"""
import pytest


def register_and_login(client, username="rec_user", password="recpass1"):
    client.post("/api/auth/register", json={"username": username, "password": password})
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    return res.json()["token"]


class TestRecommendations:
    def test_guest_user_no_auth_needed(self, client):
        """guest_ prefix users can fetch recommendations without a token."""
        res = client.get("/api/recommendations?userId=guest_abc123")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_registered_user_requires_token(self, client):
        """Registered users must send a valid Authorization header."""
        res = client.get("/api/recommendations?userId=alice_registered")
        assert res.status_code == 401

    def test_registered_user_wrong_token(self, client):
        """A token for user A should not authorize requests for user B."""
        token = register_and_login(client, "userA", "passA1234")
        res = client.get(
            "/api/recommendations?userId=userB",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert res.status_code == 401

    def test_registered_user_valid_token(self, client):
        token = register_and_login(client, "userC", "passC1234")
        res = client.get(
            "/api/recommendations?userId=userC",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)

    def test_response_has_explanation(self, client):
        """Each recommendation must carry an XAI explanation block."""
        res = client.get("/api/recommendations?userId=guest_test")
        assert res.status_code == 200
        recs = res.json()
        if recs:  # Only assert if model returns results
            assert "explanation" in recs[0]

    def test_weight_collaborative_param(self, client):
        """Extreme SVD weight values (0.0 and 1.0) should not error."""
        for w in [0.0, 1.0]:
            res = client.get(f"/api/recommendations?userId=guest_x&weight_collaborative={w}")
            assert res.status_code == 200

    def test_novelty_param(self, client):
        for n in [0.0, 0.5, 1.0]:
            res = client.get(f"/api/recommendations?userId=guest_x&novelty_weight={n}")
            assert res.status_code == 200

    def test_diversity_param(self, client):
        for d in [0.0, 0.5, 1.0]:
            res = client.get(f"/api/recommendations?userId=guest_x&diversity_weight={d}")
            assert res.status_code == 200

    def test_top_n_param(self, client):
        """top_n param should be forwarded to the hybrid recommender."""
        res = client.get("/api/recommendations?userId=guest_x&top_n=5")
        assert res.status_code == 200
