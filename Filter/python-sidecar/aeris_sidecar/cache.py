"""Thread-safe bounded LRU idempotency cache for gRPC advisory responses."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Generic, TypeVar

V = TypeVar("V")

DEFAULT_CACHE_CAPACITY: int = 4096


class ThreadSafeLruCache(Generic[V]):
    """Thread-safe bounded LRU cache keyed by RPC ``request_id``.

    Invariant:
        ``0 <= len(self) <= capacity`` under concurrent access.
    """

    def __init__(self, capacity: int = DEFAULT_CACHE_CAPACITY) -> None:
        """Initializes an empty bounded LRU cache.

        Args:
            capacity: Maximum number of cached entries (must be ``>= 1``).

        Raises:
            ValueError: If ``capacity < 1``.
        """
        if capacity < 1:
            raise ValueError(f"cache capacity must be >= 1, got {capacity}")
        self._capacity: int = capacity
        self._lock: threading.Lock = threading.Lock()
        self._store: OrderedDict[str, V] = OrderedDict()

    def get(self, key: str) -> V | None:
        """Retrieves cached value and promotes it as most recently used.

        Args:
            key: Non-empty RPC ``request_id`` key.

        Returns:
            The cached value if present, otherwise ``None``.
        """
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self._store.move_to_end(key)
            return value

    def put(self, key: str, value: V) -> None:
        """Inserts or updates ``key``, evicting the least recently used item if full.

        Args:
            key: RPC ``request_id`` key.
            value: Response payload to store.
        """
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        """Returns the current number of cached entries."""
        with self._lock:
            return len(self._store)
