import os
from redis import Redis
from rq import Queue

_redis_conn = None
_queue = None

def _build_redis():
    """Create a Redis connection (supports REDIS_URL)."""
    url =  "rediss://default:ASdJAAIncDJkZWI3ZWVmZjRiMWI0MWM5OWU4ZDE0YWQ0OTgyYmI1NXAyMTAwNTc@eternal-goshawk-10057.upstash.io:6379"#os.getenv("REDIS_URL")
    if url:
        return Redis.from_url(url, decode_responses=False)
    return Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD') or None,
        db=int(os.getenv('REDIS_DB', 0)),
        socket_timeout=5,
        socket_connect_timeout=5,
        health_check_interval=30,
        decode_responses=False
    )

def get_redis():
    """Return a singleton Redis connection (lazy)."""
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = _build_redis()
    return _redis_conn

def get_queue():
    """Return the RQ queue (lazy)."""
    global _queue
    if _queue is None:
        _queue = Queue(
            name=os.getenv('RQ_QUEUE_NAME', 'financial_analysis'),
            connection=get_redis(),
            default_timeout=600  # 10 minutes
        )
    return _queue

# Backwards compatibility for existing imports
redis_conn = get_redis()
task_queue = get_queue()
