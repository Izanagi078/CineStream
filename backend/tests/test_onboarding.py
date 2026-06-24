"""
test_onboarding.py — Tests for the /api/onboarding endpoint.
Covers: guest onboarding, registered user onboarding, validation of empty inputs, and SVD seeding.
"""
import pytest
from backend.app.models_db import DBRating

class TestOnboarding:
    def test_onboarding_guest_success(self, client, db):
        payload = {
            "genres": ["Action", "Sci-Fi"],
            "keywords": "space hero lasers",
            "userId": None
        }
        res = client.post("/api/onboarding", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "userId" in data
        assert data["userId"].startswith("guest_")
        assert "matched_movies" in data
        assert len(data["matched_movies"]) > 0

        # Verify ratings were seeded in SQLite
        seeded_ratings = db.query(DBRating).filter(DBRating.userId == data["userId"]).all()
        assert len(seeded_ratings) > 0
        for r in seeded_ratings:
            assert r.rating == 5.0

    def test_onboarding_registered_user(self, client, db):
        payload = {
            "genres": ["Drama"],
            "keywords": "sad crying family",
            "userId": "registered_user_123"
        }
        res = client.post("/api/onboarding", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["userId"] == "registered_user_123"
        
        # Verify ratings were seeded in SQLite for this specific user
        seeded_ratings = db.query(DBRating).filter(DBRating.userId == "registered_user_123").all()
        assert len(seeded_ratings) > 0

    def test_onboarding_empty_input_validation(self, client):
        payload = {
            "genres": [],
            "keywords": "",
            "userId": None
        }
        res = client.post("/api/onboarding", json=payload)
        assert res.status_code == 400
        assert "at least one genre or keyword" in res.json()["detail"].lower()
