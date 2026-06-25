import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

class CollaborativeModel:
    """
    Collaborative Filtering using Matrix Factorization via scikit-learn's TruncatedSVD.
    Refactored to explicitly expose User (P) and Item (Q) latent factor matrices
    for online/incremental Stochastic Gradient Descent (SGD) updates.
    """
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.svd = None
        self.P = None  # Latent user matrix (N x K)
        self.Q = None  # Latent movie matrix (M x K)
        self.user_means = None  # pandas.Series of user mean ratings
        self.user_mapper = {}  # userId -> row_idx
        self.movie_mapper = {}  # movieId -> col_idx
        self.inv_movie_mapper = {}  # col_idx -> movieId
        self.global_mean = 3.5

    def fit(self, ratings_df: pd.DataFrame, apply_decay: bool = True, decay_lambda: float = 0.05):
        """
        Fits TruncatedSVD on the user-movie rating matrix.
        Extracts P and Q matrices for later online tuning.
        """
        self.global_mean = ratings_df['rating'].mean()
        
        # 1. Pivot ratings table (userId x movieId)
        pivot = ratings_df.pivot(index='userId', columns='movieId', values='rating')
        self.user_means = pivot.mean(axis=1).to_dict()
        
        # Centered pivot
        pivot_centered = pivot.sub(pivot.mean(axis=1), axis=0)
        
        # 2. Apply Temporal Decay to centered ratings
        if apply_decay and 'timestamp' in ratings_df.columns:
            max_time = ratings_df['timestamp'].max()
            time_diff_years = (max_time - ratings_df['timestamp']) / (60 * 60 * 24 * 365)
            decay_weights = np.exp(-decay_lambda * time_diff_years)
            
            ratings_weighted = ratings_df.copy()
            ratings_weighted['weight'] = decay_weights
            ratings_weighted['weighted_centered'] = (ratings_weighted['rating'] - ratings_weighted['userId'].map(self.user_means)) * ratings_weighted['weight']
            
            pivot_imputed = ratings_weighted.pivot(index='userId', columns='movieId', values='weighted_centered')
        else:
            pivot_imputed = pivot_centered
            
        # Fill missing centered values with 0.0 (imputation at user mean)
        pivot_imputed = pivot_imputed.fillna(0.0)
        
        # 3. Fit TruncatedSVD
        n_components = min(self.n_factors, pivot_imputed.shape[0] - 1, pivot_imputed.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Extract Latent Factors
        # P matrix (N x K Component Weights)
        self.P = self.svd.fit_transform(pivot_imputed.values)
        # Q matrix (M x K Components, transposed so shape is Movies x Components)
        self.Q = self.svd.components_.T
        
        # Store index mappers
        self.user_ids = pivot.index.tolist()
        self.movie_ids = pivot.columns.tolist()
        
        self.user_mapper = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.movie_mapper = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.inv_movie_mapper = {idx: mid for idx, mid in enumerate(self.movie_ids)}
        print(f"[CollaborativeModel] Trained SVD. P shape: {self.P.shape}, Q shape: {self.Q.shape}")

    def predict_rating(self, user_id, movie_id) -> float:
        """
        Predicts rating dynamically using the dot product of user vector P_u and movie vector Q_i.
        """
        # Clean user_id input
        u_str = str(user_id)
        u_int = int(user_id) if str(user_id).isdigit() else None
        
        # Check user mapping in both forms
        u_idx = None
        if user_id in self.user_mapper:
            u_idx = self.user_mapper[user_id]
        elif u_str in self.user_mapper:
            u_idx = self.user_mapper[u_str]
        elif u_int in self.user_mapper:
            u_idx = self.user_mapper[u_int]
            
        m_idx = self.movie_mapper.get(movie_id)
        
        user_known = u_idx is not None
        movie_known = m_idx is not None
        
        if user_known and movie_known:
            pred = np.dot(self.P[u_idx], self.Q[m_idx])
            mean_val = self.user_means.get(user_id, self.user_means.get(u_str, self.user_means.get(u_int, self.global_mean)))
            return float(np.clip(pred + mean_val, 0.5, 5.0))
        elif user_known:
            mean_val = self.user_means.get(user_id, self.user_means.get(u_str, self.user_means.get(u_int, self.global_mean)))
            return float(mean_val)
        elif movie_known:
            return float(self.global_mean)
        else:
            return float(self.global_mean)

    def register_new_user(self, user_id):
        """
        Registers a new user by appending an empty row to matrix P.
        """
        if user_id in self.user_mapper:
            return self.user_mapper[user_id]
            
        # Allocate new index
        new_idx = len(self.user_mapper)
        self.user_mapper[user_id] = new_idx
        
        # Initialize a new latent vector row (small random variance)
        new_vec = np.random.normal(scale=0.01, size=(1, self.P.shape[1]))
        self.P = np.vstack([self.P, new_vec])
        
        # Initialize user mean to default global average
        self.user_means[user_id] = float(self.global_mean)
        return new_idx

    def register_new_movie(self, movie_id: int):
        """
        Registers a new movie by appending a new coordinate row to matrix Q.
        """
        if movie_id in self.movie_mapper:
            return self.movie_mapper[movie_id]
            
        new_idx = len(self.movie_mapper)
        self.movie_mapper[movie_id] = new_idx
        self.inv_movie_mapper[new_idx] = movie_id
        
        # Append a new item row to Q (initialize to small random values)
        new_vec = np.random.normal(scale=0.01, size=(1, self.Q.shape[1]))
        self.Q = np.vstack([self.Q, new_vec])
        return new_idx

    def update_rating_online(self, user_id, movie_id: int, rating: float, lr_p=0.1, lr_q=0.005, reg=0.02, epochs=5):
        """
        Performs real-time Stochastic Gradient Descent (SGD) update steps
        to adjust the latent User vector (P_u) and movie vector (Q_i) in-memory.
        Runs for multiple epochs to accelerate convergence for dynamic profiling.
        """
        u_idx = self.register_new_user(user_id)
        m_idx = self.register_new_movie(movie_id)
        
        # 1. Retrieve the user's running mean rating (baseline bias parameter)
        user_key = user_id if user_id in self.user_means else str(user_id)
        if user_key not in self.user_means:
            self.user_means[user_key] = float(self.global_mean)
            
        # 2. Run multiple update epochs to boost responsiveness
        for _ in range(epochs):
            pred_centered = np.dot(self.P[u_idx], self.Q[m_idx])
            pred_rating = pred_centered + self.user_means[user_key]
            error = rating - pred_rating
            
            # Incrementally adjust user rating bias/mean with error correction
            self.user_means[user_key] = float(np.clip(self.user_means[user_key] + 0.1 * error, 0.5, 5.0))
            
            # Perform SVD Latent Vector adjustments using SGD equations
            p_temp = self.P[u_idx].copy()
            q_temp = self.Q[m_idx].copy()
            
            self.P[u_idx] += lr_p * (error * q_temp - reg * p_temp)
            self.Q[m_idx] += lr_q * (error * p_temp - reg * q_temp)


class ContentModel:
    """
    Content-Based Filtering using TF-IDF on Movie metadata (Title + Genres + Tags).
    Refactored to support dynamic document and keyword index additions.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
        self.tfidf_matrix = None
        self.movies_df = None
        self.movie_id_to_idx = {}
        self.movie_idx_to_id = {}

    def fit(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.copy()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['metadata_text'])
        
        self.movie_id_to_idx = {row['movieId']: idx for idx, row in self.movies_df.iterrows()}
        self.movie_idx_to_id = {idx: row['movieId'] for idx, row in self.movies_df.iterrows()}
        print(f"[ContentModel] Fitted TF-IDF matrix of shape {self.tfidf_matrix.shape}")

    def register_new_movie(self, movie_id: int, title: str, genres: str, metadata_text: str = ""):
        """
        Transforms and appends a new movie description into the sparse TF-IDF space.
        """
        if movie_id in self.movie_id_to_idx:
            return self.movie_id_to_idx[movie_id]
            
        if not metadata_text:
            metadata_text = f"{title} {genres.replace('|', ' ')}"
            
        new_row = pd.DataFrame([{
            'movieId': movie_id, 
            'title': title, 
            'genres': genres, 
            'metadata_text': metadata_text
        }])
        self.movies_df = pd.concat([self.movies_df, new_row], ignore_index=True)
        
        # Transform text to vector
        new_vec = self.vectorizer.transform([metadata_text])
        self.tfidf_matrix = sp.vstack([self.tfidf_matrix, new_vec])
        
        # Map indices
        new_idx = self.tfidf_matrix.shape[0] - 1
        self.movie_id_to_idx[movie_id] = new_idx
        self.movie_idx_to_id[new_idx] = movie_id
        return new_idx


class HybridRecommender:
    """
    Hybrid Recommender coordinating collaborative and content modules.
    Exposes explainability structures and settings hooks.
    """
    def __init__(self, col_model: CollaborativeModel, content_model: ContentModel):
        self.col_model = col_model
        self.content_model = content_model

    def get_recommendations(self, 
                             user_id, 
                             movies_df: pd.DataFrame, 
                             ratings_df: pd.DataFrame, 
                             top_n: int = 10, 
                             weight_collaborative: float = 0.5, 
                             diversity_weight: float = 0.0, 
                             novelty_weight: float = 0.0, 
                             session_ratings: dict = None) -> pd.DataFrame:
        session_ratings = session_ratings or {}
        
        # Exclude movies this user has already rated in ratings_df or active session
        rated_movie_ids = set()
        
        # Clean user check
        u_str = str(user_id)
        u_int = int(user_id) if str(user_id).isdigit() else None
        
        user_key = None
        if user_id in self.col_model.user_mapper:
            user_key = user_id
        elif u_str in self.col_model.user_mapper:
            user_key = u_str
        elif u_int in self.col_model.user_mapper:
            user_key = u_int
            
        if user_key is not None:
            rated_movie_ids = set(ratings_df[ratings_df['userId'] == user_key]['movieId'].tolist())
            
        session_rated_ids = set(session_ratings.keys())
        excluded_ids = rated_movie_ids.union(session_rated_ids)
        
        # Filter candidate movies (only active movies)
        active_movies = movies_df[movies_df['is_active'] == True] if 'is_active' in movies_df.columns else movies_df
        candidates_df = active_movies[~active_movies['movieId'].isin(excluded_ids)].copy()
        if len(candidates_df) == 0:
            return pd.DataFrame()
            
        # 1. Build User Content Interest Vector
        n_features = self.content_model.tfidf_matrix.shape[1]
        u_text_vector = np.zeros((1, n_features))
        has_text_profile = False
        
        # Aggregate historical high ratings with exponential time decay (6 hours halflife)
        if user_key is not None:
            import time
            user_ratings = ratings_df[ratings_df['userId'] == user_key]
            if not user_ratings.empty:
                max_time = user_ratings['timestamp'].max() if 'timestamp' in user_ratings.columns else time.time()
                for _, r_row in user_ratings.iterrows():
                    mid = r_row['movieId']
                    if mid in self.content_model.movie_id_to_idx:
                        m_idx = self.content_model.movie_id_to_idx[mid]
                        weight = r_row['rating'] - 3.0
                        if weight > 0:
                            ts = r_row.get('timestamp', max_time)
                            age = max_time - ts
                            time_weight = np.exp(-age / 21600.0)  # 6 hours halflife
                            time_weight = max(time_weight, 0.1)  # keep older preferences at min 10% weight
                            
                            u_text_vector += weight * time_weight * self.content_model.tfidf_matrix[m_idx].toarray()
                            has_text_profile = True
                        
        # Aggregate live session ratings
        for mid, rating in session_ratings.items():
            if mid in self.content_model.movie_id_to_idx:
                m_idx = self.content_model.movie_id_to_idx[mid]
                weight = rating - 3.0
                if weight > 0:
                    u_text_vector += weight * 1.0 * self.content_model.tfidf_matrix[m_idx].toarray()
                    has_text_profile = True
                    
        # 2. Score Candidates
        candidate_ids = candidates_df['movieId'].values
        candidate_indices = [self.content_model.movie_id_to_idx[mid] for mid in candidate_ids if mid in self.content_model.movie_id_to_idx]
        
        # Collaborative SVD Predictions (normalized scaling)
        col_predictions = np.array([self.col_model.predict_rating(user_id, mid) for mid in candidate_ids])
        col_predictions_norm = (col_predictions - 0.5) / 4.5
        
        # Content NLP Cosine Scores
        if has_text_profile:
            u_vec_norm = np.linalg.norm(u_text_vector)
            if u_vec_norm > 0:
                u_text_vector = u_text_vector / u_vec_norm
            candidate_tfidf = self.content_model.tfidf_matrix[candidate_indices].toarray()
            content_scores = np.dot(candidate_tfidf, u_text_vector.T).flatten()
        else:
            content_scores = np.zeros(len(candidate_ids))
            
        # Dynamic blending weight allocation (force content if true new profile)
        is_new_user = user_key is None
        active_weight = 0.0 if (is_new_user and not session_ratings) else weight_collaborative
        
        hybrid_scores = active_weight * col_predictions_norm + (1.0 - active_weight) * content_scores
        
        # 3. Apply Novelty scaling
        popularity = ratings_df.groupby('movieId').size()
        pop_min = popularity.min() if not popularity.empty else 1
        pop_max = popularity.max() if not popularity.empty else 100
        
        candidate_popularity = np.array([popularity.get(mid, 0) for mid in candidate_ids])
        candidate_popularity_norm = (candidate_popularity - pop_min) / (pop_max - pop_min + 1e-8)
        
        adjusted_scores = hybrid_scores * (1.0 - novelty_weight * candidate_popularity_norm)
        
        candidates_df['base_score'] = hybrid_scores
        candidates_df['col_prediction'] = col_predictions
        candidates_df['content_similarity'] = content_scores
        candidates_df['final_score'] = adjusted_scores
        candidates_df['popularity_hits'] = candidate_popularity
        
        # 4. Apply Greedy Genre Diversification
        if diversity_weight > 0.0:
            candidates_df = candidates_df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
            selected_indices = []
            selected_genres = set()
            
            while len(selected_indices) < min(top_n * 2, len(candidates_df)):
                best_idx = None
                best_score = -999.0
                
                search_limit = min(50, len(candidates_df))
                for idx in range(search_limit):
                    if idx in selected_indices:
                        continue
                    
                    row = candidates_df.iloc[idx]
                    movie_genres = set(row['genres'].split('|'))
                    
                    if selected_genres:
                        overlap = len(selected_genres.intersection(movie_genres))
                        penalty = (overlap / len(movie_genres)) * diversity_weight
                    else:
                        penalty = 0.0
                        
                    score = row['final_score'] * (1.0 - penalty)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        
                if best_idx is None:
                    break
                    
                selected_indices.append(best_idx)
                row_selected = candidates_df.iloc[best_idx]
                selected_genres.update(row_selected['genres'].split('|'))
                
            reordered_df = candidates_df.iloc[selected_indices].copy()
            remaining_indices = [i for i in range(len(candidates_df)) if i not in selected_indices]
            reordered_df = pd.concat([reordered_df, candidates_df.iloc[remaining_indices]])
            final_recs_df = reordered_df.head(top_n)
        else:
            final_recs_df = candidates_df.sort_values(by='final_score', ascending=False).head(top_n)
            
        return final_recs_df.reset_index(drop=True)

    def explain_recommendation(self, user_id, movie_id: int, ratings_df: pd.DataFrame, session_ratings: dict = None) -> dict:
        session_ratings = session_ratings or {}
        
        if movie_id not in self.content_model.movie_id_to_idx:
            return {'collaborative': None, 'content': None}
            
        movie_idx = self.content_model.movie_id_to_idx[movie_id]
        movie_row = self.content_model.movies_df.iloc[movie_idx]
        movie_genres = set(movie_row['genres'].split('|'))
        
        col_explanation = None
        user_likes = []
        
        # Handle user check
        u_str = str(user_id)
        u_int = int(user_id) if str(user_id).isdigit() else None
        
        user_key = None
        if user_id in self.col_model.user_mapper:
            user_key = user_id
        elif u_str in self.col_model.user_mapper:
            user_key = u_str
        elif u_int in self.col_model.user_mapper:
            user_key = u_int
            
        if user_key is not None:
            hist_likes = ratings_df[(ratings_df['userId'] == user_key) & (ratings_df['rating'] >= 4.0)]['movieId'].tolist()
            user_likes.extend(hist_likes)
            
        for mid, rating in session_ratings.items():
            if rating >= 4.0:
                user_likes.append(mid)
                
        user_likes = list(set(user_likes))
        
        if user_likes and movie_id in self.col_model.movie_mapper:
            m_svd_idx = self.col_model.movie_mapper[movie_id]
            m_latent = self.col_model.Q[m_svd_idx]
            
            best_similarity = -1.0
            best_movie_id = None
            
            for liked_id in user_likes:
                if liked_id in self.col_model.movie_mapper:
                    liked_svd_idx = self.col_model.movie_mapper[liked_id]
                    liked_latent = self.col_model.Q[liked_svd_idx]
                    
                    denom = (np.linalg.norm(m_latent) * np.linalg.norm(liked_latent)) + 1e-8
                    sim = np.dot(m_latent, liked_latent) / denom
                    if sim > best_similarity:
                        best_similarity = sim
                        best_movie_id = liked_id
                        
            if best_movie_id:
                liked_title = self.content_model.movies_df[self.content_model.movies_df['movieId'] == best_movie_id]['title'].values[0]
                col_explanation = {
                    'liked_movie_title': liked_title,
                    'latent_similarity': float(best_similarity)
                }
                
        content_explanation = None
        liked_indices = [self.content_model.movie_id_to_idx[mid] for mid in user_likes if mid in self.content_model.movie_id_to_idx]
        if liked_indices:
            user_tfidf = self.content_model.tfidf_matrix[liked_indices].mean(axis=0)
            movie_tfidf = self.content_model.tfidf_matrix[movie_idx].toarray().flatten()
            
            overlap = np.multiply(user_tfidf.A1, movie_tfidf)
            top_term_indices = np.argsort(overlap)[::-1][:3]
            
            feature_names = self.content_model.vectorizer.get_feature_names_out()
            overlapping_words = [feature_names[i] for i in top_term_indices if overlap[i] > 0]
            
            if overlapping_words:
                content_explanation = {
                    'overlapping_words': overlapping_words,
                    'matching_genres': list(movie_genres)
                }
                
        return {
            'collaborative': col_explanation,
            'content': content_explanation
        }
