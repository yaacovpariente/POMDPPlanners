from typing import Any
import json
import numpy as np
from diskcache import Cache
from pathlib import Path

from POMDPPlanners.core.simulation import DataBaseInterface, History


class DiskCacheDB(DataBaseInterface):
    """A disk-based cache database implementation using diskcache."""
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        size_limit: int = 2e9,  # 2GB default size limit
        eviction_policy: str = "least-recently-used"
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
            eviction_policy=eviction_policy
        )
    
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
            return History.from_dict(json.loads(data))
        except Exception as e:
            print(f"Error retrieving from cache: {e}")
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
            value: Value to store (must be a History object)
        """
        if not isinstance(value, History):
            raise TypeError("Cache can only store History objects")
            
        try:
            # Convert History to JSON-serializable dict
            data = json.dumps(value.to_dict())
            self.cache.set(key, data)
        except Exception as e:
            print(f"Error storing in cache: {e}")
    
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
