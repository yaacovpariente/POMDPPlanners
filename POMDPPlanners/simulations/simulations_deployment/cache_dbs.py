from pathlib import Path
from typing import Any

from diskcache import Cache

from POMDPPlanners.core.simulation import DataBaseInterface
from POMDPPlanners.utils.logger import get_logger


class DiskCacheDB(DataBaseInterface):
    """A disk-based cache database implementation using diskcache."""

    def __init__(
        self,
        cache_dir: str = "./cache",
        size_limit: int = int(2e9),  # 2GB default size limit
        eviction_policy: str = "least-recently-used",
        debug: bool = False,
    ):
        """Initialize the disk cache database.

        Args:
            cache_dir: Directory to store cache files
            size_limit: Maximum size of cache in bytes
            eviction_policy: Cache eviction policy ('least-recently-used' or 'least-frequently-used')
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache = Cache(
            directory=str(self.cache_dir),
            size_limit=size_limit,
            eviction_policy=eviction_policy,
        )

        self.logger = get_logger(name=f"disk_cache_db", debug=debug, output_dir=Path(cache_dir))

    def get(self, key: str) -> Any:
        """Retrieve a value from the cache.

        Args:
            key: Cache key to retrieve

        Returns:
            The cached value, or None if not found
        """
        try:
            data = self.cache.get(key)
            if data is None:
                return None
            return data
        except Exception as e:
            self.logger.error("Error retrieving from cache: %s", e)
            return None

    def is_key_in_cache(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key to check

        Returns:
            bool: True if key exists, False otherwise
        """
        return key in self.cache

    def set(self, key: str, value: Any):
        """Store a value in the cache.

        Args:
            key: Cache key to store
            value: Value to store
        """
        try:
            self.cache.set(key, value)
        except Exception as e:
            self.logger.error("Error storing in cache: %s", e)

    def clear(self):
        """Clear all entries from the cache."""
        self.cache.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cache.close()

    def close(self):
        """Close the underlying cache and release resources."""
        self.cache.close()
