"""
cache.py — Distributed cache layer with Redis support and selective invalidation,
falling back to an in-memory LRU cache if Redis is not configured or fails.
Provides thread-safe decorators for FastAPI routers.
"""
import time
import functools
import threading
import hashlib
import json
import os
from collections import OrderedDict

# Attempt to configure Redis connection if REDIS_URL is present
REDIS_URL = os.environ.get("REDIS_URL")
_redis_client = None

if REDIS_URL:
    try:
        import redis
        # Set short socket timeout to fail fast if Redis server is down/unreachable
        _redis_client = redis.from_url(REDIS_URL, socket_timeout=2.0, decode_responses=True)
        _redis_client.ping()
        print(f"[Cache] Connected to Redis at {REDIS_URL}")
    except Exception as e:
        print(f"[Cache] Failed to connect to Redis at {REDIS_URL}. Falling back to local in-memory cache. Error: {e}")
        _redis_client = None

class SimpleLRUCache:
    def __init__(self, maxsize=128, ttl=300):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            val, expiry = self.cache[key]
            if time.time() > expiry:
                del self.cache[key]
                return None
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return val

    def set(self, key, value):
        with self.lock:
            expiry = time.time() + self.ttl
            if key in self.cache:
                del self.cache[key]
            elif len(self.cache) >= self.maxsize:
                # Evict oldest (least recently used)
                self.cache.popitem(last=False)
            self.cache[key] = (value, expiry)

    def clear(self):
        with self.lock:
            self.cache.clear()

class RedisLRUCache:
    def __init__(self, name, ttl=300):
        self.name = name
        self.ttl = ttl

    def get(self, key):
        if not _redis_client:
            return None
        try:
            val_str = _redis_client.get(f"recommender:cache:{self.name}:{key}")
            if val_str:
                return json.loads(val_str)
        except Exception as e:
            print(f"[Cache] Redis GET error for {self.name}: {e}")
        return None

    def set(self, key, value):
        if not _redis_client:
            return
        try:
            val_str = json.dumps(value)
            _redis_client.setex(f"recommender:cache:{self.name}:{key}", self.ttl, val_str)
        except Exception as e:
            print(f"[Cache] Redis SET error for {self.name}: {e}")

    def clear(self):
        if not _redis_client:
            return
        try:
            pattern = f"recommender:cache:{self.name}:*"
            keys = _redis_client.keys(pattern)
            if keys:
                _redis_client.delete(*keys)
        except Exception as e:
            print(f"[Cache] Redis CLEAR error for {self.name}: {e}")

# Global cache registry for invalidating by namespace
_registries = {}

def cache_dec(name, maxsize=128, ttl=300):
    """
    Thread-safe Cache decorator. If Redis is configured, uses Redis.
    Otherwise, falls back to SimpleLRUCache in-memory.
    Ignores non-hashable arguments like request or db sessions.
    """
    if name not in _registries:
        if _redis_client:
            _registries[name] = RedisLRUCache(name, ttl=ttl)
        else:
            _registries[name] = SimpleLRUCache(maxsize=maxsize, ttl=ttl)
    
    cache = _registries[name]
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            hashable_args = []
            for arg in args:
                if hasattr(arg, "scope") or hasattr(arg, "bind"):
                    continue
                hashable_args.append(arg)
                
            hashable_kwargs = {}
            for k, v in kwargs.items():
                if k in ("request", "db"):
                    continue
                hashable_kwargs[k] = v
                
            # Deterministic string representation for key hashing
            key_repr = f"{hashable_args}:{sorted(hashable_kwargs.items())}"
            key_hash = hashlib.md5(key_repr.encode("utf-8")).hexdigest()
            
            cached_val = cache.get(key_hash)
            if cached_val is not None:
                return cached_val
            
            val = func(*args, **kwargs)
            cache.set(key_hash, val)
            return val
        return wrapper
    return decorator

def invalidate_cache(name):
    """Clears all entries in a specific cache namespace."""
    if name in _registries:
        _registries[name].clear()

def invalidate_all_caches():
    """Clears all entries across all cache namespaces."""
    if _redis_client:
        try:
            keys = _redis_client.keys("recommender:cache:*")
            if keys:
                _redis_client.delete(*keys)
        except Exception as e:
            print(f"[Cache] Redis clear all error: {e}")
    for cache in _registries.values():
        if isinstance(cache, SimpleLRUCache):
            cache.clear()
