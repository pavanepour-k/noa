"""
Test suite for the optimized kanji database system.

Tests JSON file-based storage, caching, batch processing,
and error handling capabilities.
"""

import unittest
import json
import os
import tempfile
import time
from typing import Dict, List

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.kanji_db import OptimizedKanjiDatabase


class TestOptimizedKanjiDatabase(unittest.TestCase):
    """Test cases for OptimizedKanjiDatabase."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary database file
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_kanji_db.json")
        
        # Create test database
        self._create_test_database()
        
        # Initialize database
        self.db = OptimizedKanjiDatabase(self.db_path, cache_size=100)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_database(self):
        """Create test database with sample data."""
        test_data = {
            "metadata": {
                "version": "1.0",
                "total_kanji": 3,
                "source": "test"
            },
            "kanji": {
                "人": {
                    "radical": "人",
                    "meanings": ["person", "human"],
                    "frequency_rank": 1,
                    "components": ["人"],
                    "semantic_field": "human",
                    "common_compounds": ["人間", "人口"]
                },
                "大": {
                    "radical": "大",
                    "meanings": ["big", "large"],
                    "frequency_rank": 2,
                    "components": ["大"],
                    "semantic_field": "size",
                    "common_compounds": ["大学", "大切"]
                },
                "年": {
                    "radical": "干",
                    "meanings": ["year", "age"],
                    "frequency_rank": 3,
                    "components": ["干", "丨"],
                    "semantic_field": "time",
                    "common_compounds": ["今年", "去年"]
                }
            },
            "radicals": {
                "人": {"meaning": "person", "variants": ["亻"]},
                "大": {"meaning": "big", "variants": []},
                "干": {"meaning": "dry", "variants": []}
            },
            "component_mapping": {
                "亻": "人"
            },
            "frequency_rankings": {
                "1": "人",
                "2": "大", 
                "3": "年"
            }
        }
        
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    def test_database_loading(self):
        """Test database loading from JSON file."""
        self.assertEqual(len(self.db.kanji_data), 3)
        self.assertEqual(len(self.db.radicals_data), 3)
        self.assertIn("人", self.db.kanji_data)
        self.assertIn("大", self.db.kanji_data)
        self.assertIn("年", self.db.kanji_data)
    
    def test_get_kanji_info(self):
        """Test getting kanji information."""
        # Test existing kanji
        info = self.db.get_kanji_info("人")
        self.assertIsNotNone(info)
        self.assertEqual(info["radical"], "人")
        self.assertIn("person", info["meanings"])
        self.assertEqual(info["frequency_rank"], 1)
        
        # Test non-existing kanji
        info = self.db.get_kanji_info("非")
        self.assertIsNone(info)
    
    def test_caching_system(self):
        """Test LRU caching system."""
        # First access should be cache miss
        self.db.get_kanji_info("人")
        self.assertEqual(self.db.stats['cache_misses'], 1)
        self.assertEqual(self.db.stats['cache_hits'], 0)
        
        # Second access should be cache hit
        self.db.get_kanji_info("人")
        self.assertEqual(self.db.stats['cache_misses'], 1)
        self.assertEqual(self.db.stats['cache_hits'], 1)
        
        # Check cache hit rate
        hit_rate = self.db.get_cache_hit_rate()
        self.assertEqual(hit_rate, 0.5)
    
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        kanji_list = ["人", "大", "年", "非"]
        results = self.db.batch_get_kanji_info(kanji_list)
        
        self.assertEqual(len(results), 4)
        self.assertIsNotNone(results["人"])
        self.assertIsNotNone(results["大"])
        self.assertIsNotNone(results["年"])
        self.assertIsNone(results["非"])
    
    def test_frequency_queries(self):
        """Test frequency-based queries."""
        # Get kanji by frequency
        frequent_kanji = self.db.get_kanji_by_frequency(limit=2)
        self.assertEqual(len(frequent_kanji), 2)
        
        # Check ordering (should be by frequency rank)
        self.assertEqual(frequent_kanji[0][0], "人")  # rank 1
        self.assertEqual(frequent_kanji[1][0], "大")  # rank 2
    
    def test_component_analysis(self):
        """Test component analysis."""
        components = self.db.analyze_components("人")
        self.assertEqual(components, ["人"])
        
        # Test with component mapping
        components = self.db.analyze_components("亻")  # Should map to 人
        self.assertIn("人", components)
    
    def test_semantic_field_analysis(self):
        """Test semantic field analysis."""
        field = self.db.get_semantic_field("人")
        self.assertEqual(field, "human")
        
        field = self.db.get_semantic_field("大")
        self.assertEqual(field, "size")
        
        # Test unknown kanji
        field = self.db.get_semantic_field("非")
        self.assertEqual(field, "unknown")
    
    def test_similar_kanji_search(self):
        """Test similar kanji search."""
        similar = self.db.find_similar_kanji("人", max_results=2)
        self.assertIsInstance(similar, list)
        # Should find other kanji with similar components or fields
    
    def test_database_validation(self):
        """Test database validation."""
        validation = self.db.validate_database()
        
        self.assertIn('is_valid', validation)
        self.assertIn('errors', validation)
        self.assertIn('warnings', validation)
        self.assertIn('statistics', validation)
        
        # Should be valid for our test database
        self.assertTrue(validation['is_valid'])
    
    def test_database_statistics(self):
        """Test database statistics."""
        stats = self.db.get_database_stats()
        
        self.assertIn('total_kanji', stats)
        self.assertIn('total_radicals', stats)
        self.assertIn('cache_stats', stats)
        self.assertIn('performance', stats)
        
        self.assertEqual(stats['total_kanji'], 3)
        self.assertEqual(stats['total_radicals'], 3)
    
    def test_cache_management(self):
        """Test cache management."""
        # Clear cache
        self.db.clear_cache()
        self.assertEqual(len(self.db._cache), 0)
        self.assertEqual(self.db.stats['cache_hits'], 0)
        self.assertEqual(self.db.stats['cache_misses'], 0)
    
    def test_export_functionality(self):
        """Test database export functionality."""
        export_path = os.path.join(self.temp_dir, "exported_db.json")
        self.db.export_database(export_path)
        
        # Check if file was created
        self.assertTrue(os.path.exists(export_path))
        
        # Check if exported data is valid
        with open(export_path, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        self.assertIn('metadata', exported_data)
        self.assertIn('kanji', exported_data)
        self.assertIn('radicals', exported_data)
    
    def test_error_handling(self):
        """Test error handling with invalid database."""
        # Create invalid database file
        invalid_path = os.path.join(self.temp_dir, "invalid_db.json")
        with open(invalid_path, 'w') as f:
            f.write("invalid json content")
        
        # Should handle gracefully
        db = OptimizedKanjiDatabase(invalid_path)
        self.assertIsNotNone(db.kanji_data)  # Should have fallback data
    
    def test_missing_database_file(self):
        """Test handling of missing database file."""
        missing_path = os.path.join(self.temp_dir, "missing_db.json")
        
        # Should handle gracefully
        db = OptimizedKanjiDatabase(missing_path)
        self.assertIsNotNone(db.kanji_data)  # Should have fallback data
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Perform some operations
        self.db.get_kanji_info("人")
        self.db.get_kanji_info("大")
        self.db.get_kanji_info("人")  # Cache hit
        
        stats = self.db.get_database_stats()
        cache_stats = stats['cache_stats']
        
        self.assertGreater(cache_stats['total_requests'], 0)
        self.assertGreater(cache_stats['cache_hits'], 0)
        self.assertGreater(cache_stats['cache_misses'], 0)
        self.assertGreater(cache_stats['hit_rate'], 0)


class TestDatabaseIntegration(unittest.TestCase):
    """Integration tests for database with other components."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "integration_db.json")
        
        # Create comprehensive test database
        self._create_integration_database()
        
        # Initialize database
        self.db = OptimizedKanjiDatabase(self.db_path, cache_size=50)
    
    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_integration_database(self):
        """Create comprehensive test database."""
        # Create a larger test database with more realistic data
        test_data = {
            "metadata": {
                "version": "1.0",
                "total_kanji": 10,
                "source": "integration_test",
                "coverage_estimate": "95% of test text"
            },
            "kanji": {
                "人": {"radical": "人", "meanings": ["person"], "frequency_rank": 1, "components": ["人"], "semantic_field": "human"},
                "大": {"radical": "大", "meanings": ["big"], "frequency_rank": 2, "components": ["大"], "semantic_field": "size"},
                "学": {"radical": "子", "meanings": ["study"], "frequency_rank": 3, "components": ["子"], "semantic_field": "education"},
                "校": {"radical": "木", "meanings": ["school"], "frequency_rank": 4, "components": ["木"], "semantic_field": "education"},
                "語": {"radical": "言", "meanings": ["language"], "frequency_rank": 5, "components": ["言"], "semantic_field": "language"},
                "国": {"radical": "囗", "meanings": ["country"], "frequency_rank": 6, "components": ["囗"], "semantic_field": "geography"},
                "年": {"radical": "干", "meanings": ["year"], "frequency_rank": 7, "components": ["干"], "semantic_field": "time"},
                "日": {"radical": "日", "meanings": ["day"], "frequency_rank": 8, "components": ["日"], "semantic_field": "time"},
                "本": {"radical": "木", "meanings": ["book"], "frequency_rank": 9, "components": ["木"], "semantic_field": "education"},
                "一": {"radical": "一", "meanings": ["one"], "frequency_rank": 10, "components": ["一"], "semantic_field": "number"}
            },
            "radicals": {
                "人": {"meaning": "person", "variants": ["亻"]},
                "大": {"meaning": "big", "variants": []},
                "子": {"meaning": "child", "variants": []},
                "木": {"meaning": "tree", "variants": []},
                "言": {"meaning": "speech", "variants": []},
                "囗": {"meaning": "enclosure", "variants": []},
                "干": {"meaning": "dry", "variants": []},
                "日": {"meaning": "sun", "variants": []},
                "一": {"meaning": "one", "variants": []}
            },
            "component_mapping": {
                "亻": "人"
            },
            "frequency_rankings": {
                str(i): kanji for i, kanji in enumerate([
                    "人", "大", "学", "校", "語", "国", "年", "日", "本", "一"
                ], 1)
            }
        }
        
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    def test_large_batch_processing(self):
        """Test batch processing with larger dataset."""
        # Test with all kanji in database
        all_kanji = list(self.db.kanji_data.keys())
        results = self.db.batch_get_kanji_info(all_kanji)
        
        self.assertEqual(len(results), len(all_kanji))
        for kanji in all_kanji:
            self.assertIsNotNone(results[kanji])
    
    def test_cache_performance(self):
        """Test cache performance with repeated access."""
        test_kanji = ["人", "大", "学", "校", "語"]
        
        # First round - all cache misses
        for kanji in test_kanji:
            self.db.get_kanji_info(kanji)
        
        initial_misses = self.db.stats['cache_misses']
        initial_hits = self.db.stats['cache_hits']
        
        # Second round - should be cache hits
        for kanji in test_kanji:
            self.db.get_kanji_info(kanji)
        
        # Should have more hits now
        self.assertGreater(self.db.stats['cache_hits'], initial_hits)
        
        # Hit rate should be around 50% (half hits, half misses)
        hit_rate = self.db.get_cache_hit_rate()
        self.assertGreater(hit_rate, 0.4)
        self.assertLess(hit_rate, 0.6)
    
    def test_frequency_analysis(self):
        """Test frequency analysis capabilities."""
        # Get top 5 most frequent kanji
        top_kanji = self.db.get_kanji_by_frequency(limit=5)
        self.assertEqual(len(top_kanji), 5)
        
        # Check ordering
        for i in range(len(top_kanji) - 1):
            current_rank = top_kanji[i][1]['frequency_rank']
            next_rank = top_kanji[i + 1][1]['frequency_rank']
            self.assertLessEqual(current_rank, next_rank)
    
    def test_semantic_field_coverage(self):
        """Test semantic field coverage."""
        fields = set()
        for kanji in self.db.kanji_data:
            field = self.db.get_semantic_field(kanji)
            fields.add(field)
        
        # Should have multiple semantic fields
        self.assertGreater(len(fields), 1)
        
        # Should include expected fields
        expected_fields = {"human", "size", "education", "language", "geography", "time", "number"}
        self.assertTrue(fields.intersection(expected_fields))
    
    def test_database_integrity(self):
        """Test database integrity validation."""
        validation = self.db.validate_database()
        
        self.assertTrue(validation['is_valid'])
        self.assertEqual(len(validation['errors']), 0)
        
        # Check statistics
        stats = validation['statistics']
        self.assertEqual(stats['total_kanji'], 10)
        self.assertGreater(stats['cache_hit_rate'], 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
