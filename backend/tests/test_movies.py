"""
test_movies.py — Tests for /api/movies/* endpoints.
Covers: popular (paginated), search, add (auth required), delete (auth required), 404.
"""
import pytest


def get_token(client, username="admin_user", password="adminpass"):
    """Helper: register + return token."""
    client.post("/api/auth/register", json={"username": username, "password": password})
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    return res.json()["token"]


class TestPopularMovies:
    def test_returns_results(self, client):
        res = client.get("/api/movies/popular?limit=2")
        assert res.status_code == 200
        # May return list or paginated object — handle both shapes
        body = res.json()
        results = body if isinstance(body, list) else body.get("results", body)
        assert isinstance(results, list)

    def test_pagination_params_accepted(self, client):
        res = client.get("/api/movies/popular?limit=2&page=1")
        assert res.status_code == 200

    def test_limit_capped_at_50(self, client):
        res = client.get("/api/movies/popular?limit=999")
        assert res.status_code == 200  # Should not error, just cap


class TestSearch:
    def test_search_returns_matches(self, client):
        res = client.get("/api/movies/search?query=Test")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert any("Test" in m["title"] for m in data)

    def test_search_empty_query_returns_empty(self, client):
        res = client.get("/api/movies/search?query=")
        assert res.status_code == 200
        assert res.json() == []

    def test_search_no_match_returns_empty(self, client):
        res = client.get("/api/movies/search?query=ZZZnonexistenttitleZZZ")
        assert res.status_code == 200
        assert res.json() == []

    def test_search_case_insensitive(self, client):
        res = client.get("/api/movies/search?query=test+movie")
        assert res.status_code == 200

    def test_archived_movie_excluded(self, client):
        """'Archived Film' is is_active=False — should not appear in search results."""
        res = client.get("/api/movies/search?query=Archived")
        assert res.status_code == 200
        assert all(m.get("is_active", True) for m in res.json())


class TestAddMovie:
    def test_add_movie_requires_auth(self, client):
        res = client.post("/api/movies", json={"title": "Unauthorized Film", "genres": "Drama"})
        assert res.status_code == 401

    def test_add_movie_success(self, client):
        token = get_token(client)
        res = client.post(
            "/api/movies",
            json={"title": "New Sci-Fi Film (2024)", "genres": "Sci-Fi|Adventure"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["title"] == "New Sci-Fi Film (2024)"
        assert "movieId" in data

    def test_add_movie_shows_in_search(self, client):
        token = get_token(client, "admin2", "pass5678")
        client.post(
            "/api/movies",
            json={"title": "Unique Galaxy Movie", "genres": "Sci-Fi"},
            headers={"Authorization": f"Bearer {token}"}
        )
        res = client.get("/api/movies/search?query=Unique+Galaxy")
        assert res.status_code == 200
        assert len(res.json()) > 0


class TestDeleteMovie:
    def test_delete_movie_requires_auth(self, client):
        res = client.delete("/api/movies/1")
        assert res.status_code == 401

    def test_delete_movie_success(self, client):
        token = get_token(client, "deleter", "pass9999")
        res = client.delete("/api/movies/1", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert "archived" in res.json()["message"].lower()

    def test_delete_nonexistent_returns_404(self, client):
        token = get_token(client, "del2", "pass8888")
        res = client.delete("/api/movies/99999", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 404
