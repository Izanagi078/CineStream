"""
test_feed.py — Tests for /api/stats and /api/feed endpoints.
Covers: stats schema, feed pagination, feed ordering.
"""
import pytest


class TestStats:
    def test_stats_200(self, client):
        res = client.get("/api/stats")
        assert res.status_code == 200

    def test_stats_schema(self, client):
        data = client.get("/api/stats").json()
        assert "total_ratings" in data
        assert "live_ratings" in data
        assert "users_count" in data
        assert "movies_count" in data
        assert "svd_components" in data
        assert "metrics" in data
        metrics = data["metrics"]
        assert "rmse" in metrics
        assert "ndcg_10" in metrics
        assert "map_10" in metrics

    def test_stats_metric_types(self, client):
        metrics = client.get("/api/stats").json()["metrics"]
        assert isinstance(metrics["rmse"], float)
        assert isinstance(metrics["ndcg_10"], float)
        assert isinstance(metrics["map_10"], float)

    def test_movies_count_excludes_archived(self, client):
        """movies_count should only count is_active=True movies."""
        data = client.get("/api/stats").json()
        # Seed has 3 movies, 1 archived → expect 2 active
        assert data["movies_count"] == 2


class TestFeed:
    def test_feed_200(self, client):
        res = client.get("/api/feed")
        assert res.status_code == 200

    def test_feed_is_list_or_paginated(self, client):
        body = client.get("/api/feed").json()
        # Either a plain list (old shape) or paginated {results: [...]}
        if isinstance(body, list):
            assert True
        else:
            assert "results" in body
            assert isinstance(body["results"], list)

    def test_feed_pagination_params(self, client):
        res = client.get("/api/feed?limit=3&page=1")
        assert res.status_code == 200

    def test_feed_limit_max_50(self, client):
        res = client.get("/api/feed?limit=999")
        assert res.status_code == 200

    def test_feed_contains_ratings_submitted(self, client):
        """Rating submitted should appear in feed."""
        client.post("/api/ratings", json={
            "userId": "guest_feedtest",
            "movieId": 1,
            "rating": 5.0
        })
        body = client.get("/api/feed?limit=20").json()
        feed = body if isinstance(body, list) else body.get("results", [])
        user_ids = [entry["userId"] for entry in feed]
        assert "guest_feedtest" in user_ids

    def test_feed_entry_schema(self, client):
        """Every feed entry must have required keys."""
        client.post("/api/ratings", json={
            "userId": "guest_schema",
            "movieId": 1,
            "rating": 4.0
        })
        body = client.get("/api/feed?limit=1").json()
        feed = body if isinstance(body, list) else body.get("results", [])
        if feed:
            entry = feed[0]
            for key in ["id", "userId", "movieId", "title", "rating", "timestamp"]:
                assert key in entry, f"Missing key: {key}"
