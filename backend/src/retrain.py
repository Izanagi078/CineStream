"""
retrain.py — Model retraining pipeline.
Combines base MovieLens dataset with live SQLite ratings, fits new SVD and TF-IDF models,
evaluates accuracy and ranking metrics, serializes them to disk, and performs an atomic in-memory swap
on the running FastAPI application.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.src.data_loader import load_recommender_data
from backend.src.models import CollaborativeModel, ContentModel, HybridRecommender
from backend.src.evaluation import calculate_accuracy_metrics, calculate_map_k, calculate_ndcg_k, calculate_precision_recall_k
from backend.app.database import SessionLocal
from backend.app.models_db import DBRating, DBMovie

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

def retrain_model_pipeline(app=None):
    """
    Loads base datasets, merges live SQLite DB interactions, retrains the SVD and TF-IDF models,
    pickles them to disk, and atomically updates the running FastAPI app state if app is provided.
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    print("[Retrain] Loading base dataset from CSVs...")
    ratings, movies, links = load_recommender_data()
    
    # 1. Fetch live ratings from SQLite DB
    print("[Retrain] Fetching live ratings from SQLite database...")
    db = SessionLocal()
    try:
        db_ratings = db.query(DBRating).all()
        db_movies = db.query(DBMovie).all()
    finally:
        db.close()
        
    print(f"[Retrain] Found {len(db_ratings)} live ratings and {len(db_movies)} catalog movies.")

    # 2. Merge live database ratings
    if db_ratings:
        live_ratings_df = pd.DataFrame([{
            "userId": str(r.userId),
            "movieId": int(r.movieId),
            "rating": float(r.rating),
            "timestamp": int(r.timestamp),
        } for r in db_ratings])
        # Concatenate ratings (ensure userId is string to match mapped profiles)
        ratings["userId"] = ratings["userId"].astype(str)
        combined_ratings = pd.concat([ratings, live_ratings_df], ignore_index=True)
    else:
        combined_ratings = ratings.copy()
        combined_ratings["userId"] = combined_ratings["userId"].astype(str)

    # 3. Merge custom/active movies from database
    if db_movies:
        # Ensure base movies has poster_path column for concatenation
        if "poster_path" not in movies.columns:
            movies = movies.copy()
            movies["poster_path"] = None

        db_movies_df = pd.DataFrame([{
            "movieId": int(m.movieId),
            "title": str(m.title),
            "genres": str(m.genres),
            "metadata_text": str(m.metadata_text or f"{m.title} {m.genres.replace('|', ' ')}"),
            "tmdbId": m.tmdbId,
            "poster_path": m.poster_path,
            "is_active": m.is_active,
        } for m in db_movies])
        
        # Overlay active state and append new movies
        # First, drop any movies that exist in the database from the base df to prevent duplicates
        base_filtered = movies[~movies["movieId"].isin(db_movies_df["movieId"])]
        combined_movies = pd.concat([base_filtered, db_movies_df], ignore_index=True)
    else:
        combined_movies = movies.copy()
        if "poster_path" not in combined_movies.columns:
            combined_movies["poster_path"] = None

    # Filter out archived movies from training catalog
    training_movies = combined_movies[combined_movies["is_active"] == True].copy()

    # ── 1. Model Validation (80/20 Train-Test Split) ──────────────────────────
    print("[Retrain] Running model validation (80/20 train-test split)...")
    train_ratings, test_ratings = train_test_split(combined_ratings, test_size=0.2, random_state=42)

    col_val = CollaborativeModel(n_factors=50)
    col_val.fit(train_ratings, apply_decay=True)

    content_val = ContentModel()
    content_val.fit(training_movies)

    # A. Accuracy Metrics (RMSE / MAE)
    test_user_movie_pairs = test_ratings[['userId', 'movieId', 'rating']].values
    y_true = []
    y_pred = []
    for uid, mid, rating in test_user_movie_pairs:
        pred = col_val.predict_rating(uid, int(mid))
        y_true.append(rating)
        y_pred.append(pred)

    accuracy_results = calculate_accuracy_metrics(np.array(y_true), np.array(y_pred))
    print(f"[Retrain] Accuracy - RMSE: {accuracy_results['rmse']:.4f}, MAE: {accuracy_results['mae']:.4f}")

    # B. Ranking Metrics (MAP@10 / NDCG@10)
    test_ratings_dict = {}
    for uid, g in test_ratings.groupby('userId'):
        test_ratings_dict[uid] = list(zip(g['movieId'].astype(int), g['rating'].astype(float)))

    test_users_with_likes = [
        uid for uid, ratings_list in test_ratings_dict.items()
        if any(r >= 4.0 for _, r in ratings_list)
    ]
    
    np.random.seed(42)
    sampled_users = np.random.choice(test_users_with_likes, size=min(100, len(test_users_with_likes)), replace=False)

    recommender_val = HybridRecommender(col_val, content_val)
    recommendations_dict = {}
    for uid in sampled_users:
        recs_df = recommender_val.get_recommendations(
            user_id=uid,
            movies_df=training_movies,
            ratings_df=train_ratings,
            top_n=10,
            weight_collaborative=0.5,
            diversity_weight=0.0,
            novelty_weight=0.0
        )
        recommendations_dict[uid] = recs_df['movieId'].tolist()

    map_10 = calculate_map_k(recommendations_dict, test_ratings_dict, k=10)
    ndcg_10 = calculate_ndcg_k(recommendations_dict, test_ratings_dict, k=10)
    print(f"[Retrain] Ranking - MAP@10: {map_10 * 100:.2f}%, NDCG@10: {ndcg_10 * 100:.2f}%")

    # ── 2. Fit Final Models on 100% of Combined Data ──────────────────────────
    print("[Retrain] Fitting final models on full combined dataset...")
    final_col = CollaborativeModel(n_factors=50)
    final_col.fit(combined_ratings, apply_decay=True)

    final_content = ContentModel()
    final_content.fit(combined_movies)

    # ── 3. Save Serialized Models to Disk ──────────────────────────────────────
    col_path = os.path.join(MODELS_DIR, 'collaborative_model.pkl')
    with open(col_path, 'wb') as f:
        pickle.dump(final_col, f)

    content_path = os.path.join(MODELS_DIR, 'content_model.pkl')
    with open(content_path, 'wb') as f:
        pickle.dump(final_content, f)

    data_cache = {
        'ratings': combined_ratings,
        'movies': combined_movies,
        'links': links,
        'metrics': {
            'rmse': accuracy_results['rmse'],
            'mae': accuracy_results['mae'],
            'map_10': map_10,
            'ndcg_10': ndcg_10
        }
    }
    cache_path = os.path.join(MODELS_DIR, 'data_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(data_cache, f)
    
    print("[Retrain] Retrained models and data cache serialized successfully to disk.")

    # ── 4. Atomic In-Memory Swap ──────────────────────────────────────────────
    if app is not None:
        print("[Retrain] Performing atomic in-memory swap on active server state...")
        
        # Instantiate new hybrid recommender
        new_hybrid = HybridRecommender(final_col, final_content)
        
        # Atomic swap
        app.state.col_model = final_col
        app.state.content_model = final_content
        app.state.movies_df = combined_movies
        app.state.ratings_df = combined_ratings
        app.state.hybrid_recommender = new_hybrid
        app.state.cache = data_cache
        
        print("[Retrain] Server models updated successfully.")

    return data_cache

if __name__ == "__main__":
    retrain_model_pipeline()
