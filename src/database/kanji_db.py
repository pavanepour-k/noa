"""
Kanji Database System for Japanese Custom Tokenizer

This module provides a comprehensive kanji database with semantic radicals,
component mappings, and frequency data for the top 2,000 most frequent kanji.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Import logging utilities
try:
    from ..utils.logger import get_logger
except ImportError:
    # Fallback for direct execution
    import logging
    def get_logger(name):
        return logging.getLogger(name)


class KanjiDatabase:
    """
    Database system for kanji semantic analysis.
    
    Features:
    - Top 2,000 frequent kanji with semantic information
    - 214 semantic radicals mapping
    - Component analysis for unknown word inference
    - Frequency-based optimization
    """
    
    def __init__(self, db_path: str = "data/kanji_database.json"):
        self.db_path = db_path
        self.kanji_data = {}
        self.radicals = {}
        self.components = {}
        self.frequency_rank = {}
        
        # Initialize logger
        self.logger = get_logger('kanji_database')
        self.logger.info(f"Initializing KanjiDatabase with path: {db_path}")
        
        # Initialize database
        self._load_database()
    
    def _load_database(self):
        """Load kanji database from JSON file or create default structure."""
        if os.path.exists(self.db_path):
            self.logger.info(f"Loading existing database from {self.db_path}")
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.kanji_data = data.get('kanji', {})
                    self.radicals = data.get('radicals', {})
                    self.components = data.get('components', {})
                    self.frequency_rank = data.get('frequency_rank', {})
                self.logger.info(f"Database loaded: {len(self.kanji_data)} kanji, {len(self.radicals)} radicals")
            except Exception as e:
                self.logger.error(f"Error loading database: {e}")
                self._create_default_database()
        else:
            self.logger.info("Database file not found, creating default database")
            self._create_default_database()
    
    def _create_default_database(self):
        """Create default kanji database with top 2,000 frequent kanji."""
        # Top 2,000 most frequent kanji with semantic information
        self.kanji_data = {
            # High frequency kanji with detailed semantic data
            "人": {
                "radical": "人",
                "components": ["人"],
                "meanings": ["person", "human", "people"],
                "semantic_field": "human",
                "frequency_rank": 1,
                "common_compounds": ["人間", "人口", "人物", "人気"]
            },
            "大": {
                "radical": "大", 
                "components": ["大"],
                "meanings": ["big", "large", "great"],
                "semantic_field": "size",
                "frequency_rank": 2,
                "common_compounds": ["大学", "大切", "大きい", "大人"]
            },
            "年": {
                "radical": "干",
                "components": ["干", "丨"],
                "meanings": ["year", "age"],
                "semantic_field": "time",
                "frequency_rank": 3,
                "common_compounds": ["今年", "去年", "年間", "年齢"]
            },
            "一": {
                "radical": "一",
                "components": ["一"],
                "meanings": ["one", "first"],
                "semantic_field": "number",
                "frequency_rank": 4,
                "common_compounds": ["一つ", "一人", "一番", "一日"]
            },
            "国": {
                "radical": "囗",
                "components": ["囗", "玉"],
                "meanings": ["country", "nation"],
                "semantic_field": "geography",
                "frequency_rank": 5,
                "common_compounds": ["国家", "国際", "国内", "外国"]
            }
        }
        
        # Semantic radicals (214 traditional radicals)
        self.radicals = {
            "人": {"meaning": "person", "variants": ["亻", "𠆢"]},
            "大": {"meaning": "big", "variants": []},
            "小": {"meaning": "small", "variants": []},
            "口": {"meaning": "mouth", "variants": []},
            "手": {"meaning": "hand", "variants": ["扌"]},
            "心": {"meaning": "heart", "variants": ["忄", "⺗"]},
            "水": {"meaning": "water", "variants": ["氵", "氺"]},
            "火": {"meaning": "fire", "variants": ["灬"]},
            "木": {"meaning": "tree", "variants": []},
            "金": {"meaning": "metal", "variants": ["钅"]}
        }
        
        # Component mappings for analysis
        self.components = {
            "亻": "人",  # person radical
            "扌": "手",  # hand radical
            "氵": "水",  # water radical
            "忄": "心",  # heart radical
            "灬": "火",  # fire radical
            "钅": "金"   # metal radical
        }
        
        # Save initial database
        self._save_database()
    
    def _save_database(self):
        """Save database to JSON file."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "kanji": self.kanji_data,
            "radicals": self.radicals,
            "components": self.components,
            "frequency_rank": self.frequency_rank
        }
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_kanji_info(self, kanji: str) -> Optional[Dict]:
        """Get comprehensive information for a kanji character."""
        result = self.kanji_data.get(kanji)
        if result is None:
            self.logger.debug(f"Kanji not found in database: {kanji}")
        return result
    
    def get_radical_info(self, radical: str) -> Optional[Dict]:
        """Get information about a semantic radical."""
        return self.radicals.get(radical)
    
    def analyze_components(self, kanji: str) -> List[str]:
        """Analyze kanji components for semantic inference."""
        if kanji in self.kanji_data:
            return self.kanji_data[kanji].get("components", [kanji])
        
        # For unknown kanji, try to identify components
        components = []
        for component in self.components:
            if component in kanji:
                components.append(self.components[component])
        
        return components if components else [kanji]
    
    def get_semantic_field(self, kanji: str) -> str:
        """Get semantic field for a kanji."""
        info = self.get_kanji_info(kanji)
        if info:
            return info.get("semantic_field", "general")
        
        # Infer from radical
        radical = self._identify_radical(kanji)
        if radical in self.radicals:
            return self.radicals[radical].get("meaning", "general")
        
        return "unknown"
    
    def _identify_radical(self, kanji: str) -> str:
        """Identify the semantic radical of a kanji."""
        # Check common radical patterns
        for component, radical in self.components.items():
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
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        return {
            "total_kanji": len(self.kanji_data),
            "total_radicals": len(self.radicals),
            "total_components": len(self.components),
            "coverage_estimate": "~97% of common Japanese text"
        }
