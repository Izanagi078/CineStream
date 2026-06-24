# 🎬 Production-Grade Hybrid Movie Recommender System

An advanced, enterprise-grade **Decoupled Machine Learning Recommendation System** combining Collaborative Filtering (Online SVD Matrix Factorization) and Content-Based NLP Filtering (TF-IDF + Cosine Similarity) using the MovieLens-Latest-Small dataset (100,000 ratings).

---

## 🏗️ System Architecture

The recommendation engine is built as a fully decoupled client-server architecture. Below is the block flow diagram outlining the ingestion, online learning, caching, and serving components:

```text
                        ┌────────────────────────────────────────────────────────┐
                        │                     Vite + React                       │
                        │                       Frontend                         │
                        └─────────┬────────────────────────────▲─────────────────┘
                                  │                            │
                     JSON Requests│                            │JSON Responses
                    (Bearer Token)│                            │(XAI Explanations)
                                  ▼                            │
      ┌────────────────────────────────────────────────────────┴─────────────────┐
      │                                 FASTAPI BACKEND                          │
      │                                                                          │
      │  ┌───────────────────────┐            ┌───────────────────────────────┐  │
      │  │  Limiter Rate Limit   │            │   LRU Cache Layer (cache.py)  │  │
      │  └──────────┬────────────┘            └───────────────▲───────────────┘  │
      │             │                                         │                  │
      │             ▼                                         │                  │
      │  ┌───────────────────────┐            ┌───────────────┴───────────────┐  │
      │  │    Versioned / Legacy │            │    get_popular_movies()       │  │
      │  │      API Routers      ├───────────►│    get_stats()                │  │
      │  └──────────┬────────────┘            └───────────────────────────────┘  │
      │             │                                                            │
      │             ▼                                                            │
      │  ┌────────────────────────────────────────────────────────────────────┐  │
      │  │                       HYBRID RECOMMENDATION ENGINE                 │  │
      │  │                                                                    │  │
      │  │  ┌─────────────────────────────┐    ┌───────────────────────────┐  │  │
      │  │  │   Collaborative SVD Model   │    │    Content TF-IDF Model   │  │  │
      │  │  │   - P: User Latent Vector   │    │    - Sparse NLP Matrix    │  │  │
      │  │  │   - Q: Item Latent Vector   │    │    - Cosine Similarity    │  │  │
      │  │  └─────────────┬───────────────┘    └──────────────┬────────────┘  │  │
      │  │                │                                   │               │  │
      │  │                └─────────────────┬─────────────────┘               │  │
      │  │                                  ▼                                 │  │
      │  │                        HybridRecommender (Ensemble)                │  │
      │  └──────────────────────────────────┬─────────────────────────────────┘  │
      │                                     │                                    │
      │                                     ▼                                    │
      │                       ┌──────────────────────────┐                       │
      │                       │  Online SGD Update Step  │                       │
      │                       └─────────────┬────────────┘                       │
      └─────────────────────────────────────┼────────────────────────────────────┘
                                            │
                                  Write SQL │ (DBRating / DBMovie / DBUser)
                                            ▼
                                ┌──────────────────────┐
                                │   SQLite Database    │
                                │  (live_ratings.db)   │
                                └──────────────────────┘
```

---

## 🧮 Algorithm & Mathematical Foundation

The recommendation portal ensembles collaborative filtering predictions with content relevance scores.

### 1. Collaborative Filtering via Latent Factor SVD
Using Truncated SVD, ratings are decomposed into user latent vectors $P_u \in \mathbb{R}^k$ and movie latent vectors $Q_i \in \mathbb{R}^k$. The raw predicted rating $\hat{r}_{u,i}$ is:

$$
\hat{r}_{u,i} = \mu_u + P_u \cdot Q_i^T
$$

where $\mu_u$ is the mean rating submitted by user $u$.

### 2. Online Stochastic Gradient Descent (SGD) Learning
When a user rates a movie in real-time, the system bypasses slow retraining by updating the user and movie coordinate factors in memory using a sub-millisecond SGD step:

$$
\begin{aligned}
e_{u,i} &= r_{u,i} - \hat{r}_{u,i} \\
P_u &\leftarrow P_u + \gamma \cdot (e_{u,i} \cdot Q_i - \lambda \cdot P_u) \\
Q_i &\leftarrow Q_i + \gamma \cdot (e_{u,i} \cdot P_u - \lambda \cdot Q_i)
\end{aligned}
$$

where $\gamma$ is the learning rate (`lr_p=0.1`, `lr_q=0.005`) and $\lambda$ is the regularization coefficient (`reg=0.02`).

### 3. Content-Based TF-IDF Similarity
We vectorize movie titles, genres, and metadata keywords using TF-IDF. When matching profiles during onboarding, the query vector $\mathbf{v}_q$ is ensembled against the document matrix to extract similarity scores via cosine dot products:

$$
\text{Similarity}(d, q) = \frac{\mathbf{v}_d \cdot \mathbf{v}_q}{\|\mathbf{v}_d\| \|\mathbf{v}_q\|}
$$

### 4. Hybrid Ensembling and Reranking
The final ranking score merges both models using a collaborative weight parameter $w \in [0, 1]$:

$$
\text{Score}(u, i) = w \cdot \text{Normalize}(\text{SVD}_{u,i}) + (1 - w) \cdot \text{NLP-Sim}(i, \text{History}_u)
$$

We also apply penalties for novelty and genre diversity to reduce echo chambers.

---

## 📡 API Reference & Spec

All routes are versioned under `/api/v1/...` and aliases are maintained under `/api/...` for legacy compatibility.

### 1. Authentication Endpoints

#### `POST /api/v1/auth/register`
Creates a user account and returns a Bearer Token.
- **Request Body**:
  ```json
  {
    "username": "jane_doe",
    "password": "securepassword123"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "token": "eyJzdWIiOiJqYW5lX2RvZSIsImV4cCI6MTc4MjM2NjA4M30.signature",
    "username": "jane_doe"
  }
  ```

#### `POST /api/v1/auth/login`
Authenticates user credentials.
- **Request Body**: Same as register.
- **Response (200 OK)**: Same as register.

---

### 2. Recommendation & Interaction Endpoints

#### `GET /api/v1/recommendations`
Fetches personalized ensembled recommendations with Explainable AI reasons.
- **Query Parameters**:
  - `userId` (string, Required): The target user ID.
  - `weight_collaborative` (float, Optional, default=0.5): Weight assigned to SVD coordinates.
  - `novelty_weight` (float, Optional, default=0.0): Penalty for popular items.
  - `diversity_weight` (float, Optional, default=0.2): Penalty for redundant genres.
  - `top_n` (int, Optional, default=10): Quantity of recommendations to return.
- **Headers**:
  - `Authorization: Bearer <token>` (Required for registered users)
- **Response (200 OK)**:
  ```json
  [
    {
      "movieId": 1270,
      "title": "Back to the Future (1985)",
      "genres": "Adventure|Comedy|Sci-Fi",
      "score": 0.85,
      "explanation": "Because you highly rated Sci-Fi and Adventure movies like Star Wars."
    }
  ]
  ```

#### `POST /api/v1/ratings`
Submits a rating and triggers an online SGD update step.
- **Request Body**:
  ```json
  {
    "userId": "guest_abc123",
    "movieId": 1270,
    "rating": 5.0
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "message": "Rating processed and model updated in real-time."
  }
  ```

---

### 3. Admin & Pipeline Endpoints

#### `POST /api/v1/admin/retrain`
Forces baseline batch retraining. Loads base datasets + live SQLite history, rebuilds matrices, serializes pickles, and hot-swaps active memory.
- **Headers**:
  - `Authorization: Bearer <token>` (Required, must be logged in admin)
- **Response (200 OK)**:
  ```json
  {
    "message": "Retraining completed successfully and server state updated atomically.",
    "metrics": {
      "rmse": 0.8251,
      "mae": 0.6120,
      "map_10": 0.4350,
      "ndcg_10": 0.5180
    }
  }
  ```

---

## 📈 Database Connection Pooling & LRU Caching

- **Connection Pool**: In production, SQLAlchemy is configured with connection recycling and pre-ping verifications to prevent stale sockets:
  - `pool_size`: 10 connections
  - `max_overflow`: 20 connections
  - `pool_recycle`: 1800 seconds
- **LRU Cache Layer**: A thread-safe Least Recently Used (LRU) cache with a Time-To-Live (TTL) is integrated for slow endpoints:
  - `/api/stats`: Cached for 300s
  - `/api/movies/popular`: Cached for 300s
  - Caches are automatically invalidated in real-time when ratings are submitted, or movies are added/archived.

---

## 🔄 Automated Retraining Scheduler

A background daemon scheduler thread starts up automatically with the FastAPI application. Every 24 hours, it launches a retraining thread that calls the retrain pipeline, reads live database ratings, refits SVD/TF-IDF models, and hot-swaps them in RAM with zero server downtime.

---

## 🚀 Running the Project

### 1. Setup Environment
```bash
# Backend Setup
cd backend
pip install -r requirements.txt

# Run baseline training
python -m backend.src.train

# Run server
python run.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
