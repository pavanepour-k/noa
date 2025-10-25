"""
Hierarchical Caching System for Japanese Tokenizer

This module implements a three-tier caching system:
- L1: LRU in-memory cache (10,000 entries)
- L2: Hot kanji data (top 2,000 frequent kanji)
- L3: Full database with lazy loading
"""

import time
from collections import OrderedDict
from typing import Dict, Optional, Any, Tuple
from threading import Lock

# Import logging utilities
try:
    from ..utils.logger import get_logger, log_cache_stats, log_performance
except ImportError:
    # Fallback for direct execution
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def log_cache_stats(logger, stats):
        logger.info(f"Cache Stats: {stats}")
    def log_performance(logger, operation, duration, details=None):
        logger.info(f"Performance: {operation} took {duration:.4f}s")


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = Lock()
        self.access_count = 0
        self.hit_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU update."""
        with self.lock:
            self.access_count += 1
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hit_count += 1
                return value
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache with LRU eviction."""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = (self.hit_count / self.access_count) if self.access_count > 0 else 0.0
            return {
                "hit_rate": hit_rate,
                "size": len(self.cache),
                "capacity": self.capacity,
                "access_count": self.access_count,
                "hit_count": self.hit_count
            }


class HierarchicalCacheSystem:
    """
    Three-tier hierarchical caching system for kanji data.
    
    L1: LRU in-memory cache (fastest, limited size)
    L2: Hot kanji data (frequent kanji, moderate size)
    L3: Full database (complete data, slowest access)
    """
    
    def __init__(self, l1_capacity: int = 10000, l2_capacity: int = 2000):
        self.l1_cache = LRUCache(l1_capacity)  # Fastest access
        self.l2_cache = {}  # Hot data cache
        self.l2_capacity = l2_capacity
        self.l3_database = None  # Full database (lazy loaded)
        
        # Thread safety locks
        self._l2_lock = Lock()
        self._patterns_lock = Lock()
        self._prefetch_lock = Lock()
        self._metrics_lock = Lock()
        
        # Access pattern tracking with size limits
        self.access_patterns = {}
        self.max_access_patterns = 50000  # Size limit
        self.prefetch_queue = []
        self.max_prefetch_queue = 1000  # Size limit
        self.performance_metrics = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "total_accesses": 0
        }
        
        # Initialize logger
        self.logger = get_logger('cache_system')
        self.logger.info(f"Initialized HierarchicalCacheSystem with L1={l1_capacity}, L2={l2_capacity}")
    
    def set_l3_database(self, database):
        """Set the L3 database reference."""
        self.l3_database = database
    
    def get_kanji_info(self, kanji: str) -> Optional[Dict]:
        """
        Get kanji information through hierarchical cache lookup.
        
        Returns:
            Kanji information dict or None if not found
        """
        start_time = time.time()
        
        # L1 Cache lookup (fastest)
        result = self.l1_cache.get(kanji)
        if result is not None:
            with self._metrics_lock:
                self.performance_metrics["l1_hits"] += 1
            self._update_access_pattern(kanji, "l1_hit", time.time() - start_time)
            self.logger.debug(f"L1 cache hit for kanji: {kanji}")
            return result
        
        # L2 Cache lookup (hot data)
        with self._l2_lock:
            if kanji in self.l2_cache:
                with self._metrics_lock:
                    self.performance_metrics["l2_hits"] += 1
                # Promote to L1 cache
                self.l1_cache.put(kanji, self.l2_cache[kanji])
                self._update_access_pattern(kanji, "l2_hit", time.time() - start_time)
                self.logger.debug(f"L2 cache hit for kanji: {kanji}")
                return self.l2_cache[kanji]
        
        # L3 Database lookup (slowest)
        if self.l3_database:
            result = self.l3_database.get_kanji_info(kanji)
            if result is not None:
                with self._metrics_lock:
                    self.performance_metrics["l3_hits"] += 1
                # Add to L2 cache if it's a frequent kanji
                if self._is_frequent_kanji(kanji):
                    self._add_to_l2_cache(kanji, result)
                # Add to L1 cache
                self.l1_cache.put(kanji, result)
                self._update_access_pattern(kanji, "l3_hit", time.time() - start_time)
                self.logger.debug(f"L3 database hit for kanji: {kanji}")
                return result
        
        # Cache miss
        with self._metrics_lock:
            self.performance_metrics["misses"] += 1
        self._update_access_pattern(kanji, "miss", time.time() - start_time)
        self.logger.warning(f"Cache miss for kanji: {kanji}")
        return None
    
    def _is_frequent_kanji(self, kanji: str) -> bool:
        """Check if kanji is frequent enough for L2 cache."""
        if not self.l3_database:
            return False
        return self.l3_database.is_frequent_kanji(kanji, self.l2_capacity)
    
    def _add_to_l2_cache(self, kanji: str, data: Dict) -> None:
        """Add kanji data to L2 cache with size management."""
        with self._l2_lock:
            if len(self.l2_cache) >= self.l2_capacity:
                # Remove least recently accessed kanji from L2
                with self._patterns_lock:
                    if self.access_patterns:
                        oldest_kanji = min(self.access_patterns.keys(), 
                                         key=lambda k: self.access_patterns[k].get("last_access", 0))
                        if oldest_kanji in self.l2_cache:
                            del self.l2_cache[oldest_kanji]
            
            self.l2_cache[kanji] = data
    
    def _update_access_pattern(self, kanji: str, cache_level: str, access_time: float) -> None:
        """Update access pattern tracking for optimization."""
        with self._patterns_lock:
            # Clean up old patterns if we exceed the limit
            if len(self.access_patterns) >= self.max_access_patterns:
                self._cleanup_old_patterns()
            
            if kanji not in self.access_patterns:
                self.access_patterns[kanji] = {
                    "access_count": 0,
                    "cache_levels": [],
                    "access_times": [],
                    "last_access": time.time()
                }
            
            pattern = self.access_patterns[kanji]
            pattern["access_count"] += 1
            pattern["cache_levels"].append(cache_level)
            pattern["access_times"].append(access_time)
            pattern["last_access"] = time.time()
        
        with self._metrics_lock:
            self.performance_metrics["total_accesses"] += 1
    
    def preload_frequent_kanji(self, kanji_list: list) -> None:
        """Preload frequent kanji into L2 cache."""
        for kanji in kanji_list:
            if self.l3_database:
                data = self.l3_database.get_kanji_info(kanji)
                if data:
                    self.l2_cache[kanji] = data
    
    def prefetch_related_kanji(self, kanji: str, max_related: int = 5) -> None:
        """Prefetch kanji that are likely to be accessed together."""
        if not self.l3_database:
            return
        
        # Get similar kanji
        similar_kanji = self.l3_database.find_similar_kanji(kanji, max_related)
        
        with self._prefetch_lock:
            for related_kanji, similarity in similar_kanji:
                if (related_kanji not in self.l1_cache.cache and 
                    related_kanji not in self.l2_cache and
                    len(self.prefetch_queue) < self.max_prefetch_queue):
                    # Add to prefetch queue
                    self.prefetch_queue.append((related_kanji, similarity))
        
        # Process prefetch queue (limit to avoid overwhelming)
        with self._l2_lock:
            l2_size = len(self.l2_cache)
        
        with self._prefetch_lock:
            while (self.prefetch_queue and 
                   l2_size < self.l2_capacity * 0.8):
                prefetch_kanji, _ = self.prefetch_queue.pop(0)
                if self.l3_database:
                    data = self.l3_database.get_kanji_info(prefetch_kanji)
                    if data:
                        self._add_to_l2_cache(prefetch_kanji, data)
                        l2_size += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = self.l1_cache.get_stats()
        
        with self._metrics_lock:
            total_hits = (self.performance_metrics["l1_hits"] + 
                         self.performance_metrics["l2_hits"] + 
                         self.performance_metrics["l3_hits"])
            
            overall_hit_rate = (total_hits / self.performance_metrics["total_accesses"] 
                               if self.performance_metrics["total_accesses"] > 0 else 0.0)
        
        stats = {
            "l1_cache": l1_stats,
            "l2_cache": {
                "size": len(self.l2_cache),
                "capacity": self.l2_capacity
            },
            "performance": {
                "l1_hits": self.performance_metrics["l1_hits"],
                "l2_hits": self.performance_metrics["l2_hits"],
                "l3_hits": self.performance_metrics["l3_hits"],
                "misses": self.performance_metrics["misses"],
                "overall_hit_rate": overall_hit_rate
            },
            "access_patterns": {
                "unique_kanji": len(self.access_patterns),
                "prefetch_queue_size": len(self.prefetch_queue)
            }
        }
        
        # Log cache statistics periodically
        if self.performance_metrics["total_accesses"] > 0 and self.performance_metrics["total_accesses"] % 100 == 0:
            log_cache_stats(self.logger, stats)
        
        return stats
    
    def optimize_cache_sizes(self) -> None:
        """Optimize cache sizes based on access patterns with incremental updates."""
        self.logger.info("Starting cache optimization")
        start_time = time.time()
        
        with self._patterns_lock:
            # Analyze access patterns to determine optimal sizes
            frequent_kanji = []
            for kanji, pattern in self.access_patterns.items():
                if pattern["access_count"] > 2:  # Accessed more than twice
                    frequent_kanji.append((kanji, pattern["access_count"]))
            
            # Sort by frequency
            frequent_kanji.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Found {len(frequent_kanji)} frequent kanji for optimization")
        
        # Incremental L2 cache update (avoid full rebuild)
        with self._l2_lock:
            old_l2_size = len(self.l2_cache)
            
            # Only update if we have significant changes
            if len(frequent_kanji) > 0:
                # Keep existing high-frequency items, add new ones
                current_frequent = set(self.l2_cache.keys())
                new_frequent = {kanji for kanji, _ in frequent_kanji[:self.l2_capacity]}
                
                # Remove items that are no longer frequent
                to_remove = current_frequent - new_frequent
                for kanji in to_remove:
                    if kanji in self.l2_cache:
                        del self.l2_cache[kanji]
                
                # Add new frequent items
                for kanji, _ in frequent_kanji[:self.l2_capacity]:
                    if kanji not in self.l2_cache and len(self.l2_cache) < self.l2_capacity:
                        if self.l3_database:
                            data = self.l3_database.get_kanji_info(kanji)
                            if data:
                                self.l2_cache[kanji] = data
        
        optimization_time = time.time() - start_time
        self.logger.info(f"Cache optimization complete: L2 size {old_l2_size} -> {len(self.l2_cache)} in {optimization_time:.4f}s")
    
    def clear_cache(self) -> None:
        """Clear all cache levels."""
        with self._l2_lock:
            self.l2_cache.clear()
        
        with self._patterns_lock:
            self.access_patterns.clear()
        
        with self._prefetch_lock:
            self.prefetch_queue.clear()
        
        with self._metrics_lock:
            self.performance_metrics = {
                "l1_hits": 0,
                "l2_hits": 0,
                "l3_hits": 0,
                "misses": 0,
                "total_accesses": 0
            }
        
        # L1 cache is already thread-safe
        self.l1_cache.cache.clear()
    
    def _cleanup_old_patterns(self) -> None:
        """Clean up old access patterns to manage memory."""
        if len(self.access_patterns) <= self.max_access_patterns:
            return
        
        # Remove least recently accessed patterns
        sorted_patterns = sorted(
            self.access_patterns.items(),
            key=lambda x: x[1].get("last_access", 0)
        )
        
        # Remove oldest 10% of patterns
        remove_count = len(sorted_patterns) // 10
        for kanji, _ in sorted_patterns[:remove_count]:
            del self.access_patterns[kanji]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for optimization."""
        with self._metrics_lock:
            total_accesses = self.performance_metrics["total_accesses"]
            if total_accesses == 0:
                return {"error": "No accesses recorded"}
            
            l1_hit_rate = self.performance_metrics["l1_hits"] / total_accesses
            l2_hit_rate = self.performance_metrics["l2_hits"] / total_accesses
            l3_hit_rate = self.performance_metrics["l3_hits"] / total_accesses
            miss_rate = self.performance_metrics["misses"] / total_accesses
            
            return {
                "access_counts": self.performance_metrics.copy(),
                "hit_rates": {
                    "l1_hit_rate": l1_hit_rate,
                    "l2_hit_rate": l2_hit_rate,
                    "l3_hit_rate": l3_hit_rate,
                    "miss_rate": miss_rate,
                    "overall_hit_rate": 1.0 - miss_rate
                },
                "cache_sizes": {
                    "l1_size": len(self.l1_cache.cache),
                    "l2_size": len(self.l2_cache),
                    "access_patterns_size": len(self.access_patterns),
                    "prefetch_queue_size": len(self.prefetch_queue)
                },
                "performance_issues": self._identify_performance_issues()
            }
    
    def _identify_performance_issues(self) -> List[str]:
        """Identify potential performance issues based on metrics."""
        issues = []
        
        with self._metrics_lock:
            total_accesses = self.performance_metrics["total_accesses"]
            if total_accesses == 0:
                return issues
            
            l1_hit_rate = self.performance_metrics["l1_hits"] / total_accesses
            l2_hit_rate = self.performance_metrics["l2_hits"] / total_accesses
            miss_rate = self.performance_metrics["misses"] / total_accesses
            
            # Check for performance issues
            if l1_hit_rate < 0.3:
                issues.append("Low L1 cache hit rate - consider increasing L1 capacity")
            
            if l2_hit_rate < 0.2:
                issues.append("Low L2 cache hit rate - consider optimizing L2 content")
            
            if miss_rate > 0.3:
                issues.append("High miss rate - consider expanding database coverage")
            
            if len(self.access_patterns) > self.max_access_patterns * 0.9:
                issues.append("Access patterns near capacity - cleanup needed")
            
            if len(self.prefetch_queue) > self.max_prefetch_queue * 0.8:
                issues.append("Prefetch queue near capacity - processing needed")
        
        return issues
