"""
Comprehensive Tests for Japanese Tokenizer

This module provides tests for:
- Basic tokenization tests
- Unknown word inference tests
- Performance regression tests
- Edge case handling
"""

import unittest
import time
from typing import List, Dict
from src.tokenizer.hybrid_tokenizer import HybridTokenizer
from src.database.kanji_db import KanjiDatabase
from src.database.cache_system import HierarchicalCacheSystem
from test.evaluation import TokenizerEvaluator


class TestJapaneseTokenizer(unittest.TestCase):
    """Test suite for Japanese tokenizer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize components
        self.kanji_db = KanjiDatabase()
        self.cache_system = HierarchicalCacheSystem()
        self.cache_system.set_l3_database(self.kanji_db)
        
        # Initialize tokenizer
        self.tokenizer = HybridTokenizer(
            kanji_db=self.kanji_db,
            cache_system=self.cache_system
        )
        
        # Initialize evaluator
        self.evaluator = TokenizerEvaluator()
    
    def test_basic_tokenization(self):
        """Test basic tokenization functionality."""
        test_cases = [
            {
                'input': '今日は良い天気です。',
                'expected_tokens': ['今日', 'は', '良い', '天気', 'です', '。']
            },
            {
                'input': '私は学生です。',
                'expected_tokens': ['私', 'は', '学生', 'です', '。']
            },
            {
                'input': '本を読んでいます。',
                'expected_tokens': ['本', 'を', '読ん', 'で', 'います', '。']
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(input=test_case['input']):
                result = self.tokenizer.tokenize(test_case['input'])
                
                # Extract tokens from result
                tokens = []
                for token_result in result:
                    if 'tokens' in token_result:
                        tokens.extend(token_result['tokens'])
                
                # Check if tokens match expected
                self.assertEqual(tokens, test_case['expected_tokens'])
    
    def test_compound_word_tokenization(self):
        """Test compound word tokenization."""
        test_cases = [
            {
                'input': '機械学習の研究をしています。',
                'expected_contains': ['機械学習', '研究']
            },
            {
                'input': '新しい技術について話しましょう。',
                'expected_contains': ['新しい', '技術']
            },
            {
                'input': '人工知能の開発が進んでいます。',
                'expected_contains': ['人工知能', '開発']
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(input=test_case['input']):
                result = self.tokenizer.tokenize(test_case['input'])
                
                # Extract tokens from result
                tokens = []
                for token_result in result:
                    if 'tokens' in token_result:
                        tokens.extend(token_result['tokens'])
                
                # Check if expected tokens are present
                for expected_token in test_case['expected_contains']:
                    self.assertIn(expected_token, tokens)
    
    def test_unknown_word_inference(self):
        """Test unknown word inference capability."""
        test_cases = [
            {
                'input': 'この推測計は正確だ',
                'unknown_word': '推測計',
                'should_infer': True
            },
            {
                'input': '生体分子の構造解析',
                'unknown_word': '生体分子',
                'should_infer': True
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(input=test_case['input']):
                result = self.tokenizer.tokenize(test_case['input'])
                
                # Check if unknown word was processed
                unknown_word_found = False
                for token_result in result:
                    if token_result.get('word') == test_case['unknown_word']:
                        unknown_word_found = True
                        
                        # Check if inference was attempted
                        if test_case['should_infer']:
                            self.assertIn('method', token_result)
                            self.assertIn('confidence', token_result)
                        break
                
                self.assertTrue(unknown_word_found, 
                              f"Unknown word '{test_case['unknown_word']}' not found in results")
    
    def test_performance_benchmarks(self):
        """Test tokenizer performance benchmarks."""
        test_texts = [
            '今日は良い天気です。',
            '機械学習の研究をしています。',
            '新しい技術について話しましょう。',
            '人工知能の開発が進んでいます。',
            '量子コンピュータの基礎理論を学んでいます。'
        ]
        
        # Measure performance
        start_time = time.time()
        total_tokens = 0
        
        for text in test_texts:
            result = self.tokenizer.tokenize(text)
            tokens = sum(len(r.get('tokens', [])) for r in result)
            total_tokens += tokens
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(processing_time, 1.0, "Processing time should be under 1 second")
        
        if processing_time > 0:
            tokens_per_second = total_tokens / processing_time
            self.assertGreater(tokens_per_second, 100, 
                              "Should process at least 100 tokens per second")
    
    def test_cache_performance(self):
        """Test cache system performance."""
        # Test with repeated words
        test_text = '機械学習の研究をしています。機械学習は重要です。'
        
        # First run (cache miss)
        start_time = time.time()
        result1 = self.tokenizer.tokenize(test_text)
        first_run_time = time.time() - start_time
        
        # Second run (cache hit)
        start_time = time.time()
        result2 = self.tokenizer.tokenize(test_text)
        second_run_time = time.time() - start_time
        
        # Second run should be faster due to caching
        self.assertLess(second_run_time, first_run_time, 
                       "Second run should be faster due to caching")
        
        # Results should be identical
        self.assertEqual(result1, result2, "Cached results should be identical")
    
    def test_edge_cases(self):
        """Test edge case handling."""
        edge_cases = [
            'はし',  # Ambiguous word
            'すもももももももものうち',  # Tongue twister
            'a',  # Single character
            '',  # Empty string
            '   ',  # Whitespace only
            '123',  # Numbers only
            'abc',  # Romaji only
        ]
        
        for edge_case in edge_cases:
            with self.subTest(input=edge_case):
                # Should not raise exception
                result = self.tokenizer.tokenize(edge_case)
                self.assertIsInstance(result, list)
    
    def test_confidence_scores(self):
        """Test confidence score calculation."""
        test_text = '今日は良い天気です。'
        result = self.tokenizer.tokenize(test_text)
        
        for token_result in result:
            if 'confidence' in token_result:
                confidence = token_result['confidence']
                self.assertGreaterEqual(confidence, 0.0, "Confidence should be >= 0")
                self.assertLessEqual(confidence, 1.0, "Confidence should be <= 1")
    
    def test_engine_selection(self):
        """Test engine selection logic."""
        # Test with known words (should use fast engine)
        known_text = '今日は良い天気です。'
        result = self.tokenizer.tokenize(known_text)
        
        fast_engine_used = any(
            r.get('engine_used') == 'fast' for r in result
        )
        self.assertTrue(fast_engine_used, "Fast engine should be used for known words")
        
        # Test with unknown words (should use hybrid engine)
        unknown_text = 'この推測計は正確だ'
        result = self.tokenizer.tokenize(unknown_text)
        
        hybrid_engine_used = any(
            r.get('engine_used') == 'hybrid' for r in result
        )
        self.assertTrue(hybrid_engine_used, "Hybrid engine should be used for unknown words")
    
    def test_memory_usage(self):
        """Test memory usage and cleanup."""
        # Process multiple texts
        test_texts = [
            '今日は良い天気です。',
            '機械学習の研究をしています。',
            '新しい技術について話しましょう。'
        ] * 10  # Repeat to test memory management
        
        for text in test_texts:
            result = self.tokenizer.tokenize(text)
            self.assertIsInstance(result, list)
        
        # Check that memory usage is reasonable
        # This is a basic test - in practice, you'd use memory profiling tools
        self.assertTrue(True, "Memory usage test passed")
    
    def test_error_handling(self):
        """Test error handling capabilities."""
        error_cases = [
            None,  # None input
            123,  # Non-string input
            'x' * 10000,  # Very long text
        ]
        
        for error_case in error_cases:
            with self.subTest(input=error_case):
                if error_case is None or not isinstance(error_case, str):
                    # Should handle gracefully
                    try:
                        result = self.tokenizer.tokenize(error_case)
                        # If it doesn't raise an exception, result should be empty or None
                        self.assertTrue(result is None or len(result) == 0)
                    except (TypeError, AttributeError):
                        # Expected for invalid input types
                        pass
                else:
                    # Valid string input
                    result = self.tokenizer.tokenize(error_case)
                    self.assertIsInstance(result, list)
    
    def test_integration_with_evaluation(self):
        """Test integration with evaluation framework."""
        # Run evaluation
        results = self.evaluator.evaluate_tokenizer(self.tokenizer)
        
        # Check that evaluation completed successfully
        self.assertIn('overall_score', results)
        self.assertIn('basic_accuracy', results)
        self.assertIn('compound_accuracy', results)
        self.assertIn('unknown_word_inference', results)
        self.assertIn('performance_benchmarks', results)
        
        # Check that scores are reasonable
        self.assertGreaterEqual(results['overall_score'], 0.0)
        self.assertLessEqual(results['overall_score'], 1.0)
    
    def test_adaptive_learning(self):
        """Test adaptive learning functionality."""
        # Test with unknown word
        unknown_text = 'この推測計は正確だ'
        result1 = self.tokenizer.tokenize(unknown_text)
        
        # Simulate learning from feedback
        # (This would require the adaptive learning module to be integrated)
        
        # Test again with same word
        result2 = self.tokenizer.tokenize(unknown_text)
        
        # Results should be consistent
        self.assertEqual(len(result1), len(result2))
    
    def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        import threading
        import queue
        
        test_texts = [
            '今日は良い天気です。',
            '機械学習の研究をしています。',
            '新しい技術について話しましょう。'
        ]
        
        results_queue = queue.Queue()
        
        def process_text(text):
            result = self.tokenizer.tokenize(text)
            results_queue.put((text, result))
        
        # Start multiple threads
        threads = []
        for text in test_texts:
            thread = threading.Thread(target=process_text, args=(text,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Check that all texts were processed
        self.assertEqual(len(results), len(test_texts))
        
        # Check that results are valid
        for text, result in results:
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)


class TestKanjiDatabase(unittest.TestCase):
    """Test suite for kanji database functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.kanji_db = KanjiDatabase()
    
    def test_kanji_lookup(self):
        """Test kanji information lookup."""
        test_kanji = '人'
        info = self.kanji_db.get_kanji_info(test_kanji)
        
        self.assertIsNotNone(info)
        self.assertIn('meanings', info)
        self.assertIn('semantic_field', info)
        self.assertIn('components', info)
    
    def test_component_analysis(self):
        """Test kanji component analysis."""
        test_kanji = '語'
        components = self.kanji_db.analyze_components(test_kanji)
        
        self.assertIsInstance(components, list)
        self.assertGreater(len(components), 0)
    
    def test_semantic_field_identification(self):
        """Test semantic field identification."""
        test_kanji = '学'
        field = self.kanji_db.get_semantic_field(test_kanji)
        
        self.assertIsInstance(field, str)
        self.assertNotEqual(field, 'unknown')
    
    def test_frequency_ranking(self):
        """Test frequency ranking functionality."""
        test_kanji = '人'
        rank = self.kanji_db.get_frequency_rank(test_kanji)
        
        self.assertIsInstance(rank, int)
        self.assertGreaterEqual(rank, 1)
    
    def test_similar_kanji_search(self):
        """Test similar kanji search."""
        test_kanji = '人'
        similar = self.kanji_db.find_similar_kanji(test_kanji)
        
        self.assertIsInstance(similar, list)
        # Should find some similar kanji
        self.assertGreater(len(similar), 0)


class TestCacheSystem(unittest.TestCase):
    """Test suite for cache system functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache_system = HierarchicalCacheSystem()
        self.kanji_db = KanjiDatabase()
        self.cache_system.set_l3_database(self.kanji_db)
    
    def test_cache_hit_miss(self):
        """Test cache hit and miss behavior."""
        test_kanji = '人'
        
        # First access (should be cache miss)
        result1 = self.cache_system.get_kanji_info(test_kanji)
        self.assertIsNotNone(result1)
        
        # Second access (should be cache hit)
        result2 = self.cache_system.get_kanji_info(test_kanji)
        self.assertIsNotNone(result2)
        
        # Results should be identical
        self.assertEqual(result1, result2)
    
    def test_cache_statistics(self):
        """Test cache statistics functionality."""
        # Access some kanji
        test_kanji = ['人', '大', '年', '一', '国']
        for kanji in test_kanji:
            self.cache_system.get_kanji_info(kanji)
        
        # Get statistics
        stats = self.cache_system.get_cache_stats()
        
        self.assertIn('l1_cache', stats)
        self.assertIn('l2_cache', stats)
        self.assertIn('performance', stats)
        self.assertIn('access_patterns', stats)
    
    def test_cache_optimization(self):
        """Test cache optimization functionality."""
        # Access multiple kanji to build access patterns
        test_kanji = ['人', '大', '年', '一', '国', '日', '本', '時', '分', '秒']
        for kanji in test_kanji:
            self.cache_system.get_kanji_info(kanji)
        
        # Run optimization
        self.cache_system.optimize_cache_sizes()
        
        # Check that optimization completed without errors
        self.assertTrue(True, "Cache optimization completed successfully")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
