"""
Small in-memory rate limiter (sliding window) plus a helper to find the caller's IP.

In memory means: counters reset when the server restarts, and each server process keeps its own
counters. That is fine for one server process (what the Docker setup runs). If you ever run
several copies, move this to Redis.
"""
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class SlidingWindowLimiter:
    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def retry_after(self, bucket: str, key: str, limit: int, window: int) -> int:
        """0 if the caller may go ahead, otherwise how many seconds to wait."""
        now = time.monotonic()
        with self._lock:
            dq = self._hits[(bucket, key)]
            while dq and dq[0] <= now - window:
                dq.popleft()
            if not dq:
                self._hits.pop((bucket, key), None)
                return 0
            return int(dq[0] + window - now) + 1 if len(dq) >= limit else 0

    def add(self, bucket: str, key: str) -> None:
        with self._lock:
            self._hits[(bucket, key)].append(time.monotonic())

    def reset(self, bucket: str, key: str) -> None:
        with self._lock:
            self._hits.pop((bucket, key), None)

    def enforce(self, bucket: str, key: str, limit: int, window: int) -> None:
        """Count this call, and raise 429 if the caller is over the limit."""
        wait = self.retry_after(bucket, key, limit, window)
        if wait:
            raise HTTPException(
                status_code=429,
                detail=f"Too many attempts. Try again in {wait} seconds.",
                headers={"Retry-After": str(wait)},
            )
        self.add(bucket, key)


limiter = SlidingWindowLimiter()


def client_ip(request: Request) -> str:
    # uvicorn runs with --proxy-headers behind Caddy, so this is the real visitor address
    return request.client.host if request.client else "unknown"
