from pathlib import Path
from dotenv import load_dotenv
from rq import SimpleWorker
from redis_config import get_queue, get_redis

if __name__ == "__main__":
    # Load ../.env so worker matches app environment
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    # Sanity check Redis connectivity
    get_redis().ping()
    print("[Worker] Connected to Redis. Subscribing to queue (SimpleWorker, Windows)...")
    SimpleWorker([get_queue()]).work()