# 🎬 CineStream AI — Hybrid Movie Recommender System

An advanced, enterprise-grade **Decoupled Machine Learning Recommendation System** combining Collaborative Filtering (Online SVD Matrix Factorization) and Content-Based NLP Filtering (TF-IDF + Cosine Similarity) using the MovieLens-Latest-Small dataset (100,000 ratings).

---

## 🌐 Live Deployments
*   **Web Application (Frontend)**: [https://machine-learning-recommendation-sys.vercel.app](https://machine-learning-recommendation-sys.vercel.app)
*   **API Service (Backend)**: [https://cinestream-backend-3gh8.onrender.com](https://cinestream-backend-3gh8.onrender.com)

---

## 🏗️ System Architecture

The recommendation engine is built as a fully decoupled client-server architecture. Below is the block flow diagram outlining the ingestion, online learning, caching, TMDB integrations, and storage components:

```text
                                        ┌──────────────────────────────┐
                                        │          TMDB API            │
                                        │       (api.tmdb.org)         │
                                        └──────────────▲───────────────┘
                                                       │
                                                       │ HTTPS GET (poster_path)
                                                       │
                         ┌─────────────────────────────┴──────────────────────────┐
                         │                     Vite + React                       │
                         │                       Frontend                         │
                         └─────────┬────────────────────────────▲─────────────────┘
                                   │                            │
                      JSON Requests│                            │JSON Responses
                     (Bearer Token)│                            │(XAI + Posters)
                                   ▼                            │
       ┌────────────────────────────────────────────────────────┴─────────────────┐
       │                                 FASTAPI BACKEND                          │
       │                                                                          │
       │  ┌───────────────────────┐            ┌───────────────────────────────┐  │
       │  │  Limiter Rate Limit   │            │     Redis Cache Namespace     │  │
       │  └──────────┬────────────┘            │     (In-Memory Fallback)      │  │
       │             │                         └───────────────▲───────────────┘  │
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
       │                       │   Online SGD (5 Epochs)  │                       │
       │                       └─────────────┬────────────┘                       │
       └─────────────────────────────────────┼────────────────────────────────────┘
                                             │
                                   Write SQL │ (DBRating / DBMovie / DBUser)
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
         ┌──────────────────────┐                          ┌──────────────────────┐
         │     PostgreSQL       │◄───[Synced / Fallback]───►│        SQLite        │
         │    (Neon Cloud)      │                          │  (live_ratings.db)   │
         └──────────────────────┘                          └──────────────────────┘
```

---

## 🧮 Algorithm & Mathematical Foundation

The recommendation portal ensembles collaborative filtering predictions with content relevance scores.

### 1. Collaborative Filtering via Latent Factor SVD
Using Truncated SVD, ratings are decomposed into user latent vectors $P_u \in \mathbb{R}^k$ and movie latent vectors $Q_i \in \mathbb{R}^k$ (where $k = 50$). The predicted rating $\hat{r}_{u,i}$ is:

$$
\hat{r}_{u,i} = \mu_u + P_u \cdot Q_i^T
$$

where $\mu_u$ is the user's average rating.

### 2. Bayesian User Bias Shrinkage
To prevent cold-start user average ratings from flatlining or skewing predictions (e.g. if a user submits a single $5.0$ rating, SVD residual prediction gradients would collapse to zero), the user bias $\mu_u$ is regularized toward the global mean average $\mu_{\text{global}}$ based on sample evidence:

$$
\mu_u = \frac{\sum_{j=1}^{N_u} r_{u,j} + K \cdot \mu_{\text{global}}}{N_u + K}
$$

where $N_u$ is the number of ratings user $u$ has submitted, and $K = 5.0$ represents the shrinkage smoothing factor.

### 3. Online Stochastic Gradient Descent (SGD) Learning
When a user rates a movie in real-time, the system bypasses slow retraining by updating the user and movie coordinate factors in memory using 5 epochs of online SGD steps:

$$
\begin{aligned}
e_{u,i} &= r_{u,i} - \hat{r}_{u,i} \\
P_u &\leftarrow P_u + \gamma_p \cdot (e_{u,i} \cdot Q_i - \lambda \cdot P_u) \\
Q_i &\leftarrow Q_i + \gamma_q \cdot (e_{u,i} \cdot P_u - \lambda \cdot Q_i)
\end{aligned}
$$

where $\gamma_p$ and $\gamma_q$ are learning rates (`lr_p=0.1`, `lr_q=0.005`) and $\lambda$ is the regularization coefficient (`reg=0.02`).

### 4. Content-Based TF-IDF Similarity
We vectorize movie titles, genres, and metadata keywords using TF-IDF. When matching profiles during onboarding, the query vector $\mathbf{v}_q$ is ensembled against the document matrix to extract similarity scores via cosine dot products:

$$
\text{Similarity}(d, q) = \frac{\mathbf{v}_d \cdot \mathbf{v}_q}{\|\mathbf{v}_d\| \|\mathbf{v}_q\|}
$$

### 5. Hybrid Ensembling, Novelty and Diversified Reranking
The final ranking score merges both models using a collaborative weight parameter $w \in [0, 1]$:

$$
\text{Score}_{\text{hybrid}}(u, i) = w \cdot \text{Normalize}(\text{SVD}_{u,i}) + (1 - w) \cdot \text{NLP-Sim}(i, \text{History}_u)
$$

To prevent redundancy, recommendations are diversified using a greedy Maximal Marginal Relevance selection step. A candidate movie $c$ receives a penalty proportional to the number of overlapping genres already added to the list:

$$
\text{penalty}(c) = \left( \frac{\text{Shared Genres}(c, \text{List})}{\text{Total Genres}(c)} \right) \cdot \beta
$$

where $\beta$ is the diversity weight. If a shared genre matches the user's onboarding favorites, the overlap count receives a **75% discount penalty** (multiplied by $0.25$), allowing onboarding favorites to appear multiple times while encouraging general diversity.

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

## 📈 Database & Distributed Cache Configurations

- **Primary Database (Neon PostgreSQL)**: The system connects to a remote cloud PostgreSQL database by default using connection pooling configurations. If the connection details are omitted or fail, it gracefully falls back to a local disk-backed **SQLite** (`live_ratings.db`).
  - `pool_size`: 10 connections (configurable via `DB_POOL_SIZE`)
  - `max_overflow`: 20 connections (configurable via `DB_MAX_OVERFLOW`)
  - `pool_recycle`: 1800 seconds (configurable via `DB_POOL_RECYCLE`)

- **Distributed Cache (Redis)**: When `REDIS_URL` is set, the caching namespaces (`popular_movies`, etc.) serialize and store payloads on a centralized Redis cache.
  - If Redis is unavailable or unconfigured, the system automatically falls back to an **in-memory thread-safe LRU Cache** (`SimpleLRUCache`).
  - Cache namespaces are automatically invalidated when custom ratings are submitted or catalog movies are added/deleted.

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
