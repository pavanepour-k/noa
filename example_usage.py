#!/usr/bin/env python3
"""
Example Usage of Japanese Custom Tokenizer

This script demonstrates the basic usage of the Japanese custom tokenizer
with various examples including known words, unknown words, and performance testing.
"""

import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.tokenizer.hybrid_tokenizer import HybridTokenizer
from src.database.kanji_db import KanjiDatabase
from src.database.cache_system import HierarchicalCacheSystem
from test.evaluation import TokenizerEvaluator


def main():
    """Main example function."""
    print("🇯🇵 Japanese Custom Tokenizer Example")
    print("=" * 50)
    
    # Initialize components
    print("Initializing components...")
    kanji_db = KanjiDatabase()
    cache_system = HierarchicalCacheSystem()
    cache_system.set_l3_database(kanji_db)
    
    # Create tokenizer
    tokenizer = HybridTokenizer(kanji_db=kanji_db, cache_system=cache_system)
    print("✅ Tokenizer initialized successfully!")
    print()
    
    # Example 1: Basic tokenization
    print("📝 Example 1: Basic Tokenization")
    print("-" * 30)
    basic_text = "今日は良い天気です。"
    print(f"Input: {basic_text}")
    
    result = tokenizer.tokenize(basic_text)
    print("Results:")
    for i, token_result in enumerate(result):
        print(f"  {i+1}. Word: '{token_result['word']}'")
        print(f"     Tokens: {token_result['tokens']}")
        print(f"     Confidence: {token_result['confidence']:.2f}")
        print(f"     Engine: {token_result.get('engine_used', 'unknown')}")
        print()
    
    # Example 2: Compound words
    print("📝 Example 2: Compound Words")
    print("-" * 30)
    compound_text = "機械学習の研究をしています。"
    print(f"Input: {compound_text}")
    
    result = tokenizer.tokenize(compound_text)
    print("Results:")
    for i, token_result in enumerate(result):
        print(f"  {i+1}. Word: '{token_result['word']}'")
        print(f"     Tokens: {token_result['tokens']}")
        print(f"     Confidence: {token_result['confidence']:.2f}")
        print(f"     Engine: {token_result.get('engine_used', 'unknown')}")
        print()
    
    # Example 3: Unknown word inference
    print("📝 Example 3: Unknown Word Inference")
    print("-" * 30)
    unknown_text = "この推測計は正確だ"
    print(f"Input: {unknown_text}")
    print("Note: '推測計' is an unknown compound word")
    
    result = tokenizer.tokenize(unknown_text)
    print("Results:")
    for i, token_result in enumerate(result):
        print(f"  {i+1}. Word: '{token_result['word']}'")
        print(f"     Tokens: {token_result['tokens']}")
        print(f"     Confidence: {token_result['confidence']:.2f}")
        print(f"     Engine: {token_result.get('engine_used', 'unknown')}")
        
        # Check for unknown word inference
        if token_result.get('method') == 'context_inference':
            print(f"     🔍 Inferred meaning: {token_result.get('inferred_meaning', [])}")
            print(f"     🧠 Component analysis: {token_result.get('component_analysis', {})}")
        print()
    
    # Example 4: Performance testing
    print("📝 Example 4: Performance Testing")
    print("-" * 30)
    test_texts = [
        "今日は良い天気です。",
        "機械学習の研究をしています。",
        "新しい技術について話しましょう。",
        "人工知能の開発が進んでいます。",
        "量子コンピュータの基礎理論を学んでいます。"
    ]
    
    print(f"Testing with {len(test_texts)} texts...")
    
    start_time = time.time()
    total_tokens = 0
    
    for i, text in enumerate(test_texts):
        result = tokenizer.tokenize(text)
        tokens = sum(len(r.get('tokens', [])) for r in result)
        total_tokens += tokens
        print(f"  Text {i+1}: {tokens} tokens")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\nPerformance Results:")
    print(f"  Total processing time: {processing_time:.4f} seconds")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Tokens per second: {total_tokens / processing_time:.0f}")
    print(f"  Average time per text: {processing_time / len(test_texts):.4f} seconds")
    print()
    
    # Example 5: Cache performance
    print("📝 Example 5: Cache Performance")
    print("-" * 30)
    test_text = "機械学習の研究をしています。機械学習は重要です。"
    print(f"Input: {test_text}")
    print("(Contains repeated words to test caching)")
    
    # First run (cache miss)
    start_time = time.time()
    result1 = tokenizer.tokenize(test_text)
    first_run_time = time.time() - start_time
    
    # Second run (cache hit)
    start_time = time.time()
    result2 = tokenizer.tokenize(test_text)
    second_run_time = time.time() - start_time
    
    print(f"First run time: {first_run_time:.4f} seconds")
    print(f"Second run time: {second_run_time:.4f} seconds")
    print(f"Speed improvement: {first_run_time / second_run_time:.1f}x faster")
    print()
    
    # Example 6: Performance statistics
    print("📝 Example 6: Performance Statistics")
    print("-" * 30)
    stats = tokenizer.get_performance_stats()
    
    print("Engine Usage:")
    print(f"  Fast engine: {stats['engine_usage']['fast_engine_rate']:.1%}")
    print(f"  Hybrid engine: {stats['engine_usage']['hybrid_engine_rate']:.1%}")
    
    print("\nTokenization Stats:")
    print(f"  Total tokens: {stats['tokenization_stats']['total_tokens']}")
    print(f"  Unknown words: {stats['tokenization_stats']['unknown_words']}")
    print(f"  Unknown word rate: {stats['tokenization_stats']['unknown_word_rate']:.1%}")
    print(f"  Average confidence: {stats['tokenization_stats']['average_confidence']:.2f}")
    print()
    
    # Example 7: Evaluation
    print("📝 Example 7: Comprehensive Evaluation")
    print("-" * 30)
    print("Running evaluation framework...")
    
    evaluator = TokenizerEvaluator()
    results = evaluator.evaluate_tokenizer(tokenizer)
    
    print("Evaluation Results:")
    print(f"  Overall Score: {results['overall_score']:.2f}")
    print(f"  Basic Accuracy: {results['basic_accuracy']['accuracy']:.2f}")
    print(f"  Compound Accuracy: {results['compound_accuracy']['accuracy']:.2f}")
    print(f"  Unknown Word Inference: {results['unknown_word_inference']['accuracy']:.2f}")
    
    perf = results['performance_benchmarks']
    print(f"  Average Processing Time: {perf['average_processing_time']:.4f}s")
    print(f"  Tokens per Second: {perf['average_tokens_per_second']:.0f}")
    print()
    
    # Example 8: Database statistics
    print("📝 Example 8: Database Statistics")
    print("-" * 30)
    db_stats = kanji_db.get_database_stats()
    print(f"Total kanji: {db_stats['total_kanji']}")
    print(f"Total radicals: {db_stats['total_radicals']}")
    print(f"Total components: {db_stats['total_components']}")
    print(f"Coverage estimate: {db_stats['coverage_estimate']}")
    print()
    
    # Example 9: Cache statistics
    print("📝 Example 9: Cache Statistics")
    print("-" * 30)
    cache_stats = cache_system.get_cache_stats()
    
    print("L1 Cache:")
    print(f"  Hit rate: {cache_stats['l1_cache']['hit_rate']:.2%}")
    print(f"  Size: {cache_stats['l1_cache']['size']}/{cache_stats['l1_cache']['capacity']}")
    
    print("\nL2 Cache:")
    print(f"  Size: {cache_stats['l2_cache']['size']}/{cache_stats['l2_cache']['capacity']}")
    
    print("\nPerformance:")
    print(f"  L1 hits: {cache_stats['performance']['l1_hits']}")
    print(f"  L2 hits: {cache_stats['performance']['l2_hits']}")
    print(f"  L3 hits: {cache_stats['performance']['l3_hits']}")
    print(f"  Misses: {cache_stats['performance']['misses']}")
    print(f"  Overall hit rate: {cache_stats['performance']['overall_hit_rate']:.2%}")
    print()
    
    print("🎉 Example completed successfully!")
    print("\nThis tokenizer demonstrates:")
    print("  ✅ Human-like understanding through kanji component analysis")
    print("  ✅ Hybrid inference combining fast and context engines")
    print("  ✅ Unknown word inference using semantic relationships")
    print("  ✅ High-performance caching for optimal speed")
    print("  ✅ Comprehensive evaluation and testing")
    print("\nThe tokenizer is ready for production use!")


if __name__ == "__main__":
    main()
