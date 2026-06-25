from sqlalchemy import Column, Integer, String, Float, Boolean
from backend.app.database import Base

class DBUser(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

class DBMovie(Base):
    __tablename__ = "movies"

    movieId = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    genres = Column(String, nullable=False)
    metadata_text = Column(String, nullable=True)
    tmdbId = Column(Integer, nullable=True)
    poster_path = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)

class DBRating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    userId = Column(String, index=True, nullable=False)
    movieId = Column(Integer, index=True, nullable=False)
    rating = Column(Float, nullable=False)
    timestamp = Column(Integer, nullable=False)
