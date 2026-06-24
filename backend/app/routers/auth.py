"""Auth router — register, login, demo (rate-limited)."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models_db import DBUser
from backend.app.schemas import UserAuth, TokenResponse
from backend.app.dependencies import hash_password, verify_password, generate_token, limiter

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=TokenResponse)
@limiter.limit("5/minute")
def register(request: Request, auth: UserAuth, db: Session = Depends(get_db)):
    """Register a new user account. Rate limited to 5 attempts per minute."""
    username = auth.username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="Username cannot be empty.")
    if len(auth.password) < 4:
        raise HTTPException(status_code=422, detail="Password must be at least 4 characters long.")

    existing = db.query(DBUser).filter(DBUser.username == username).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken.")

    hashed = hash_password(auth.password)
    new_user = DBUser(username=username, hashed_password=hashed)
    db.add(new_user)
    db.commit()

    token = generate_token(username)
    return {"token": token, "username": username}


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request, auth: UserAuth, db: Session = Depends(get_db)):
    """Authenticate with username + password. Rate limited to 5 attempts per minute."""
    username = auth.username.strip()
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if not user or not verify_password(auth.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = generate_token(username)
    return {"token": token, "username": username}


@router.post("/demo", response_model=TokenResponse)
@limiter.limit("10/minute")
def login_demo(request: Request, auth: UserAuth):
    """Passwordless token for pre-trained demo profiles."""
    username = auth.username.strip()
    if not username.startswith("User ") and not username.startswith("guest_"):
        raise HTTPException(status_code=403, detail="Not a valid demo profile identifier.")

    token = generate_token(username)
    return {"token": token, "username": username}
