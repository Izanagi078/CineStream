"""
test_auth.py — Tests for /api/auth/* endpoints.
Covers: register, login, demo, duplicate user, bad password, expired token.
"""
import pytest


class TestRegister:
    def test_register_success(self, client):
        res = client.post("/api/auth/register", json={"username": "alice", "password": "pass123"})
        assert res.status_code == 200
        data = res.json()
        assert "token" in data
        assert data["username"] == "alice"

    def test_register_duplicate_user(self, client):
        client.post("/api/auth/register", json={"username": "bob", "password": "pass123"})
        res = client.post("/api/auth/register", json={"username": "bob", "password": "different"})
        assert res.status_code == 409
        assert "taken" in res.json()["detail"].lower()

    def test_register_empty_username(self, client):
        res = client.post("/api/auth/register", json={"username": "  ", "password": "pass123"})
        assert res.status_code == 422

    def test_register_short_password(self, client):
        res = client.post("/api/auth/register", json={"username": "carol", "password": "ab"})
        assert res.status_code == 422
        assert "4 characters" in res.json()["detail"].lower()

    def test_register_returns_bearer_token(self, client):
        res = client.post("/api/auth/register", json={"username": "dave", "password": "securepass"})
        token = res.json()["token"]
        # Token should have exactly 2 dot-separated parts (header.signature)
        assert len(token.split(".")) == 2


class TestLogin:
    def test_login_success(self, client):
        client.post("/api/auth/register", json={"username": "eve", "password": "mypassword"})
        res = client.post("/api/auth/login", json={"username": "eve", "password": "mypassword"})
        assert res.status_code == 200
        assert "token" in res.json()

    def test_login_wrong_password(self, client):
        client.post("/api/auth/register", json={"username": "frank", "password": "correct"})
        res = client.post("/api/auth/login", json={"username": "frank", "password": "wrong"})
        assert res.status_code == 401  # Must be 401, not 400
        assert "WWW-Authenticate" in res.headers

    def test_login_nonexistent_user(self, client):
        res = client.post("/api/auth/login", json={"username": "ghost", "password": "pass"})
        assert res.status_code == 401

    def test_login_token_is_usable(self, client):
        """Token obtained from login should authorize recommendations."""
        client.post("/api/auth/register", json={"username": "grace", "password": "pass1234"})
        login_res = client.post("/api/auth/login", json={"username": "grace", "password": "pass1234"})
        token = login_res.json()["token"]

        res = client.get(
            "/api/recommendations?userId=grace",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert res.status_code == 200


class TestDemo:
    def test_demo_valid_user(self, client):
        res = client.post("/api/auth/demo", json={"username": "User 10", "password": ""})
        assert res.status_code == 200
        assert res.json()["username"] == "User 10"

    def test_demo_invalid_identifier(self, client):
        res = client.post("/api/auth/demo", json={"username": "hacker", "password": ""})
        assert res.status_code == 403

    def test_demo_guest_prefix_allowed(self, client):
        res = client.post("/api/auth/demo", json={"username": "guest_abc123", "password": ""})
        assert res.status_code == 200
