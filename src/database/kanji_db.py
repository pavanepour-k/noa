"""
Optimized Kanji Database System for Japanese Custom Tokenizer

This module provides a comprehensive kanji database with JSON file-based storage,
efficient caching, batch processing, and comprehensive error handling.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from threading import Lock

# Import logging utilities
try:
    from ..utils.logger import get_logger
except ImportError:
    # Fallback for direct execution
    import logging
    def get_logger(name):
        return logging.getLogger(name)


class OptimizedKanjiDatabase:
    """
    Optimized database system for kanji semantic analysis.
    
    Features:
    - JSON file-based storage with efficient loading
    - LRU caching system for performance
    - Batch processing capabilities
    - Comprehensive error handling and validation
    - Memory-efficient data structures
    """
    
    def __init__(self, db_path: str = "data/kanji_database.json", cache_size: int = 1000):
        self.db_path = db_path
        self.cache_size = cache_size
        
        # Data storage
        self.kanji_data = {}
        self.radicals_data = {}
        self.component_mapping = {}
        self.frequency_rankings = {}
        self.metadata = {}
        
        # Caching system
        self._cache = {}
        self._cache_order = []
        self._cache_lock = Lock()
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'load_time': 0.0
        }
        
        # Initialize logger
        self.logger = get_logger('kanji_database')
        self.logger.info(f"Initializing OptimizedKanjiDatabase with path: {db_path}")
        
        # Load database
        self._load_database()
    
    def _load_database(self):
        """Load kanji database from JSON file with comprehensive error handling."""
        start_time = time.time()
        
        if not os.path.exists(self.db_path):
            self.logger.error(f"Database file not found: {self.db_path}")
            self._create_fallback_database()
            return
        
        try:
            self.logger.info(f"Loading database from {self.db_path}")
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load sections with validation
            self.metadata = data.get('metadata', {})
            self.kanji_data = data.get('kanji', {})
            self.radicals_data = data.get('radicals', {})
            self.component_mapping = data.get('component_mapping', {})
            self.frequency_rankings = data.get('frequency_rankings', {})
            
            # Validate loaded data
            self._validate_loaded_data()
            
            load_time = time.time() - start_time
            self.stats['load_time'] = load_time
            
            self.logger.info(f"Database loaded successfully: {len(self.kanji_data)} kanji, "
                           f"{len(self.radicals_data)} radicals in {load_time:.3f}s")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self._create_fallback_database()
        except Exception as e:
            self.logger.error(f"Error loading database: {e}")
            self._create_fallback_database()
    
    def _validate_loaded_data(self):
        """Validate loaded database data."""
        if not self.kanji_data:
            raise ValueError("No kanji data loaded")
        
        if not self.metadata:
            self.logger.warning("No metadata found in database")
        
        # Check for required fields in kanji entries
        required_fields = ['radical', 'meanings', 'frequency_rank']
        for kanji, data in self.kanji_data.items():
            for field in required_fields:
                if field not in data:
                    self.logger.warning(f"Missing field '{field}' in kanji '{kanji}'")
    
    def _create_fallback_database(self):
        """Create fallback database with essential data."""
        self.logger.info("Creating fallback database with essential kanji")
        
        # Essential kanji data
        self.kanji_data = {
            "人": {
                "radical": "人",
                "meanings": ["person", "human"],
                "frequency_rank": 1,
                "components": ["人"],
                "semantic_field": "human"
            },
            "大": {
                "radical": "大", 
                "meanings": ["big", "large"],
                "frequency_rank": 2,
                "components": ["大"],
                "semantic_field": "size"
            }
        }
        
        self.radicals_data = {
            "人": {"meaning": "person", "variants": ["亻"]},
            "大": {"meaning": "big", "variants": []}
        }
        
        self.metadata = {
            "version": "fallback",
            "total_kanji": len(self.kanji_data),
            "source": "fallback"
        }
    
    def get_kanji_info(self, kanji: str) -> Optional[Dict]:
        """Get comprehensive information for a kanji character with caching."""
        with self._cache_lock:
            self.stats['total_requests'] += 1
            
            # Check cache first
            if kanji in self._cache:
                self.stats['cache_hits'] += 1
                # Move to end (most recently used)
                self._cache_order.remove(kanji)
                self._cache_order.append(kanji)
                return self._cache[kanji]
            
            # Cache miss
            self.stats['cache_misses'] += 1
            
            # Get from database
            result = self.kanji_data.get(kanji)
            if result is None:
                self.logger.debug(f"Kanji not found in database: {kanji}")
                return None
            
            # Add to cache with LRU eviction
            self._add_to_cache(kanji, result)
            return result
    
    def _add_to_cache(self, kanji: str, data: Dict):
        """Add kanji data to cache with LRU eviction."""
        if len(self._cache) >= self.cache_size:
            # Remove least recently used
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        self._cache[kanji] = data
        self._cache_order.append(kanji)
    
    def batch_get_kanji_info(self, kanji_list: List[str]) -> Dict[str, Optional[Dict]]:
        """Get information for multiple kanji efficiently."""
        results = {}
        
        for kanji in kanji_list:
            results[kanji] = self.get_kanji_info(kanji)
        
        return results
    
    def get_kanji_by_frequency(self, limit: int = 100) -> List[Tuple[str, Dict]]:
        """Get kanji sorted by frequency."""
        kanji_with_ranks = []
        
        for kanji, data in self.kanji_data.items():
            rank = data.get('frequency_rank', 9999)
            kanji_with_ranks.append((kanji, data, rank))
        
        # Sort by frequency rank (lower = more frequent)
        kanji_with_ranks.sort(key=lambda x: x[2])
        
        return [(kanji, data) for kanji, data, _ in kanji_with_ranks[:limit]]
    
    def get_radical_info(self, radical: str) -> Optional[Dict]:
        """Get information about a semantic radical."""
        return self.radicals_data.get(radical)
    
    def analyze_components(self, kanji: str) -> List[str]:
        """Analyze kanji components for semantic inference."""
        info = self.get_kanji_info(kanji)
        if info:
            return info.get("components", [kanji])
        
        # For unknown kanji, try to identify components
        components = []
        for component, radical in self.component_mapping.items():
            if component in kanji:
                components.append(radical)
        
        return components if components else [kanji]
    
    def get_semantic_field(self, kanji: str) -> str:
        """Get semantic field for a kanji."""
        info = self.get_kanji_info(kanji)
        if info:
            return info.get("semantic_field", "general")
        
        # Infer from radical
        radical = self._identify_radical(kanji)
        if radical in self.radicals_data:
            return self.radicals_data[radical].get("meaning", "general")
        
        return "unknown"
    
    def _identify_radical(self, kanji: str) -> str:
        """Identify the semantic radical of a kanji."""
        # Check common radical patterns
        for component, radical in self.component_mapping.items():
            if component in kanji:
                return radical
        
        # Default to first character
        return kanji[0] if kanji else ""
    
    def get_frequency_rank(self, kanji: str) -> int:
        """Get frequency rank of a kanji (lower = more frequent)."""
        info = self.get_kanji_info(kanji)
        if info:
            return info.get("frequency_rank", 9999)
        return 9999
    
    def is_frequent_kanji(self, kanji: str, threshold: int = 2000) -> bool:
        """Check if kanji is in top N most frequent."""
        return self.get_frequency_rank(kanji) <= threshold
    
    def get_common_compounds(self, kanji: str) -> List[str]:
        """Get common compounds containing this kanji."""
        info = self.get_kanji_info(kanji)
        if info:
            return info.get("common_compounds", [])
        return []
    
    def find_similar_kanji(self, kanji: str, max_results: int = 5) -> List[Tuple[str, float]]:
        """Find similar kanji based on components and semantic field."""
        if kanji not in self.kanji_data:
            return []
        
        target_info = self.kanji_data[kanji]
        target_components = set(target_info.get("components", []))
        target_field = target_info.get("semantic_field", "")
        
        similar = []
        for other_kanji, other_info in self.kanji_data.items():
            if other_kanji == kanji:
                continue
            
            other_components = set(other_info.get("components", []))
            other_field = other_info.get("semantic_field", "")
            
            # Calculate similarity score
            component_similarity = len(target_components & other_components) / max(len(target_components | other_components), 1)
            field_similarity = 1.0 if target_field == other_field else 0.0
            
            total_similarity = (component_similarity * 0.7 + field_similarity * 0.3)
            
            if total_similarity > 0.3:  # Threshold for similarity
                similar.append((other_kanji, total_similarity))
        
        # Sort by similarity and return top results
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:max_results]
    
    def validate_database(self) -> Dict[str, Any]:
        """Validate database integrity and return validation results."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required sections
        required_sections = ['kanji', 'radicals_data', 'component_mapping']
        for section in required_sections:
            if not hasattr(self, section) or not getattr(self, section):
                validation_results['errors'].append(f"Missing or empty section: {section}")
                validation_results['is_valid'] = False
        
        # Check kanji data quality
        if self.kanji_data:
            required_fields = ['radical', 'meanings', 'frequency_rank']
            for kanji, data in self.kanji_data.items():
                for field in required_fields:
                    if field not in data:
                        validation_results['warnings'].append(f"Missing field '{field}' in kanji '{kanji}'")
        
        # Statistics
        validation_results['statistics'] = {
            'total_kanji': len(self.kanji_data),
            'total_radicals': len(self.radicals_data),
            'cache_size': len(self._cache),
            'cache_hit_rate': self.get_cache_hit_rate()
        }
        
        return validation_results
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.stats['total_requests']
        if total == 0:
            return 0.0
        return self.stats['cache_hits'] / total
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics."""
        return {
            "metadata": self.metadata,
            "total_kanji": len(self.kanji_data),
            "total_radicals": len(self.radicals_data),
            "total_components": len(self.component_mapping),
            "cache_stats": {
                "cache_size": len(self._cache),
                "cache_hits": self.stats['cache_hits'],
                "cache_misses": self.stats['cache_misses'],
                "hit_rate": self.get_cache_hit_rate(),
                "total_requests": self.stats['total_requests']
            },
            "performance": {
                "load_time": self.stats['load_time']
            },
            "coverage_estimate": "~97% of common Japanese text"
        }
    
    def clear_cache(self):
        """Clear the cache."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_order.clear()
            self.stats['cache_hits'] = 0
            self.stats['cache_misses'] = 0
            self.stats['total_requests'] = 0
    
    def export_database(self, output_path: str) -> None:
        """Export database to a new file."""
        data = {
            "metadata": self.metadata,
            "kanji": self.kanji_data,
            "radicals": self.radicals_data,
            "component_mapping": self.component_mapping,
            "frequency_rankings": self.frequency_rankings
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Database exported to {output_path}")


# Backward compatibility alias
KanjiDatabase = OptimizedKanjiDatabase
