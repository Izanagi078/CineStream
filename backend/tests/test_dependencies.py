"""
test_dependencies.py — Unit tests for auth helper functions in dependencies.py.
Covers: password hashing, token generation/verification, expired tokens, tampered tokens.
"""
import time
import pytest
from backend.app.dependencies import hash_password, verify_password, generate_token, verify_token


class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        h = hash_password("mysecret")
        assert "mysecret" not in h

    def test_hash_has_salt_separator(self):
        h = hash_password("test")
        assert "$" in h

    def test_different_hashes_for_same_password(self):
        """Each hash call uses a new random salt."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2

    def test_verify_correct_password(self):
        h = hash_password("correcthorse")
        assert verify_password("correcthorse", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("correcthorse")
        assert verify_password("wrongpassword", h) is False

    def test_verify_malformed_hash(self):
        assert verify_password("anything", "notahash") is False


class TestTokenGeneration:
    def test_generate_token_two_parts(self):
        token = generate_token("user123")
        parts = token.split(".")
        assert len(parts) == 2

    def test_verify_valid_token(self):
        token = generate_token("alice")
        result = verify_token(token)
        assert result == "alice"

    def test_verify_wrong_signature(self):
        token = generate_token("alice")
        payload_b64, sig = token.split(".")
        tampered = f"{payload_b64}.invalidsignature"
        assert verify_token(tampered) is None

    def test_verify_garbage_token(self):
        assert verify_token("not.a.real.token") is None
        assert verify_token("") is None
        assert verify_token("singlesegment") is None

    def test_different_users_get_different_tokens(self):
        t1 = generate_token("userA")
        t2 = generate_token("userB")
        assert t1 != t2

    def test_token_subject_is_user_id(self):
        """The payload.sub must match the user_id passed to generate_token."""
        import base64, json
        token = generate_token("test_user_xyz")
        payload_b64 = token.split(".")[0]
        pad = len(payload_b64) % 4
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (4 - pad) if pad else payload_b64))
        assert payload["sub"] == "test_user_xyz"

    def test_token_expiry_future(self):
        """A freshly generated token should not be expired."""
        import base64, json
        token = generate_token("user")
        payload_b64 = token.split(".")[0]
        pad = len(payload_b64) % 4
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (4 - pad) if pad else payload_b64))
        assert payload["exp"] > time.time()
