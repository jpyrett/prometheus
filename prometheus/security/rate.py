"""The per-account rate limit (S9): spam and SLA protection, in memory per process.

    limiter = RateLimiter(max_per_minute=6, max_per_hour=60)       # `clock` is injectable: a test passes its own time, never sleeps
    limiter.allow("12")        # True for the first six requests of a minute, False for the seventh

A request that is allowed counts; one that is refused does not (a user who stops is free again a minute after the last request that ran). An account id is whatever the caller names the requester (`"cli"` when none).
"""
import threading
import time
from collections import defaultdict, deque

MINUTE, HOUR = 60.0, 3600.0
CLI = "cli"


class RateLimiter:
    def __init__(self, max_per_minute: int, max_per_hour: int, clock=time.monotonic):
        if max_per_minute < 1 or max_per_hour < 1:
            raise ValueError("the limits must be at least 1")
        self.max_per_minute, self.max_per_hour, self.clock = max_per_minute, max_per_hour, clock
        self._seen: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def key(account_id) -> str:
        return CLI if account_id is None or str(account_id).strip() == "" else str(account_id).strip()

    def allow(self, account_id=None) -> bool:
        """Count this request for the account and say whether it may run."""
        key, now = self.key(account_id), self.clock()
        with self._lock:
            seen = self._seen[key]
            while seen and now - seen[0] >= HOUR:
                seen.popleft()
            in_minute = sum(1 for moment in seen if now - moment < MINUTE)
            if in_minute >= self.max_per_minute or len(seen) >= self.max_per_hour:
                return False
            seen.append(now)
            return True
