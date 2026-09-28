import threading
import time

import config


_LOCK = threading.Lock()
_SHARED = None


class TokenBucket:
    """Thread-safe token bucket. rate_per_min tokens are added each minute."""

    def __init__(self, rate_per_min=180):
        self.capacity = float(rate_per_min)
        self.tokens = float(rate_per_min)
        self.rate_per_sec = float(rate_per_min) / 60.0
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.updated
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
            self.updated = now

    def acquire(self, tokens=1, block=True):
        need = float(tokens)
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= need:
                    self.tokens -= need
                    return True
                if not block:
                    return False
                deficit = need - self.tokens
                wait = deficit / self.rate_per_sec if self.rate_per_sec > 0 else 0.05
            time.sleep(max(wait, 0.01))


def get_rest_limiter():
    global _SHARED
    with _LOCK:
        if _SHARED is None:
            rate = int(getattr(config, 'ALPACA_REST_PER_MIN', 180))
            _SHARED = TokenBucket(rate_per_min=rate)
        return _SHARED


def reset_rest_limiter(rate_per_min=None):
    global _SHARED
    with _LOCK:
        if rate_per_min is None:
            rate_per_min = int(getattr(config, 'ALPACA_REST_PER_MIN', 180))
        _SHARED = TokenBucket(rate_per_min=rate_per_min)
        return _SHARED
