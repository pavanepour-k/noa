"""
Evaluation Framework for Japanese Tokenizer

This module provides comprehensive evaluation capabilities:
- Test cases with known/unknown words
- Accuracy metrics (precision, recall, F1)
- Speed benchmarks (tokens/sec, latency)
- Cache hit rate monitoring
"""

import time
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import statistics


class TokenizerEvaluator:
    """
    Comprehensive evaluation framework for Japanese tokenizer.
    
    Provides accuracy, performance, and quality metrics
    for tokenizer assessment and comparison.
    """
    
    def __init__(self):
        self.test_cases = []
        self.benchmark_results = {}
        self.performance_metrics = {}
        
        # Initialize test cases
        self._create_test_cases()
    
    def _create_test_cases(self):
        """Create comprehensive test cases for evaluation."""
        self.test_cases = [
            # Basic tokenization tests
            {
                'category': 'basic',
                'input': '今日は良い天気です。',
                'expected_tokens': ['今日', 'は', '良い', '天気', 'です', '。'],
                'description': 'Basic sentence with particles and adjectives'
            },
            {
                'category': 'basic',
                'input': '私は学生です。',
                'expected_tokens': ['私', 'は', '学生', 'です', '。'],
                'description': 'Simple sentence with subject and predicate'
            },
            
            # Compound word tests
            {
                'category': 'compounds',
                'input': '機械学習の研究をしています。',
                'expected_tokens': ['機械学習', 'の', '研究', 'を', 'して', 'います', '。'],
                'description': 'Compound words and verb conjugation'
            },
            {
                'category': 'compounds',
                'input': '新しい技術について話しましょう。',
                'expected_tokens': ['新しい', '技術', 'について', '話し', 'ましょう', '。'],
                'description': 'Adjective-noun compounds and verb forms'
            },
            
            # Unknown word tests
            {
                'category': 'unknown_words',
                'input': 'この推測計は正確だ',
                'unknown_word': '推測計',
                'expected_meaning': '추측 계기',
                'context_clues': ['正確', '計測'],
                'description': 'Unknown compound word inference'
            },
            {
                'category': 'unknown_words',
                'input': '生体分子の構造解析',
                'unknown_word': '生体分子',
                'expected_meaning': '생체 분자',
                'context_clues': ['構造', '解析'],
                'description': 'Scientific unknown compound'
            },
            
            # Technical domain tests
            {
                'category': 'technical',
                'input': '人工知能の開発が進んでいます。',
                'expected_tokens': ['人工知能', 'の', '開発', 'が', '進ん', 'で', 'います', '。'],
                'description': 'AI/technology domain text'
            },
            {
                'category': 'technical',
                'input': '量子コンピュータの基礎理論',
                'expected_tokens': ['量子コンピュータ', 'の', '基礎', '理論'],
                'description': 'Quantum computing terminology'
            },
            
            # Edge cases
            {
                'category': 'edge_cases',
                'input': 'はし',
                'expected_tokens': ['はし'],
                'description': 'Ambiguous word (bridge/chopsticks)'
            },
            {
                'category': 'edge_cases',
                'input': 'すもももももももものうち',
                'expected_tokens': ['すもも', 'も', 'もも', 'も', 'もも', 'の', 'うち'],
                'description': 'Tongue twister with repeated words'
            }
        ]
    
    def evaluate_tokenizer(self, tokenizer, test_cases: List[Dict] = None) -> Dict:
        """
        Comprehensive evaluation of tokenizer.
        
        Args:
            tokenizer: Tokenizer instance to evaluate
            test_cases: Optional custom test cases
            
        Returns:
            Comprehensive evaluation results
        """
        if test_cases is None:
            test_cases = self.test_cases
        
        results = {
            'basic_accuracy': self._evaluate_basic_accuracy(tokenizer, test_cases),
            'compound_accuracy': self._evaluate_compound_accuracy(tokenizer, test_cases),
            'unknown_word_inference': self._evaluate_unknown_word_inference(tokenizer, test_cases),
            'performance_benchmarks': self._evaluate_performance(tokenizer, test_cases),
            'domain_specific': self._evaluate_domain_specific(tokenizer, test_cases),
            'edge_cases': self._evaluate_edge_cases(tokenizer, test_cases)
        }
        
        # Calculate overall score
        results['overall_score'] = self._calculate_overall_score(results)
        
        return results
    
    def _evaluate_basic_accuracy(self, tokenizer, test_cases: List[Dict]) -> Dict:
        """Evaluate basic tokenization accuracy."""
        basic_cases = [tc for tc in test_cases if tc['category'] == 'basic']
        
        if not basic_cases:
            return {'accuracy': 0.0, 'total_tests': 0}
        
        correct_count = 0
        total_tests = len(basic_cases)
        
        for test_case in basic_cases:
            try:
                result = tokenizer.tokenize(test_case['input'])
                predicted_tokens = [r['tokens'] for r in result if 'tokens' in r]
                predicted_tokens = [token for sublist in predicted_tokens for token in sublist]
                
                if self._tokens_match(predicted_tokens, test_case['expected_tokens']):
                    correct_count += 1
                    
            except Exception as e:
                print(f"Error in basic test: {e}")
        
        accuracy = correct_count / total_tests if total_tests > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_tests': total_tests,
            'description': 'Basic tokenization accuracy'
        }
    
    def _evaluate_compound_accuracy(self, tokenizer, test_cases: List[Dict]) -> Dict:
        """Evaluate compound word tokenization accuracy."""
        compound_cases = [tc for tc in test_cases if tc['category'] == 'compounds']
        
        if not compound_cases:
            return {'accuracy': 0.0, 'total_tests': 0}
        
        correct_count = 0
        total_tests = len(compound_cases)
        
        for test_case in compound_cases:
            try:
                result = tokenizer.tokenize(test_case['input'])
                predicted_tokens = [r['tokens'] for r in result if 'tokens' in r]
                predicted_tokens = [token for sublist in predicted_tokens for token in sublist]
                
                if self._tokens_match(predicted_tokens, test_case['expected_tokens']):
                    correct_count += 1
                    
            except Exception as e:
                print(f"Error in compound test: {e}")
        
        accuracy = correct_count / total_tests if total_tests > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_tests': total_tests,
            'description': 'Compound word tokenization accuracy'
        }
    
    def _evaluate_unknown_word_inference(self, tokenizer, test_cases: List[Dict]) -> Dict:
        """Evaluate unknown word inference capability."""
        unknown_cases = [tc for tc in test_cases if tc['category'] == 'unknown_words']
        
        if not unknown_cases:
            return {'accuracy': 0.0, 'total_tests': 0}
        
        correct_inferences = 0
        total_tests = len(unknown_cases)
        
        for test_case in unknown_cases:
            try:
                result = tokenizer.tokenize(test_case['input'])
                
                # Find unknown word inference
                unknown_word = test_case.get('unknown_word', '')
                inference_found = False
                
                for token_result in result:
                    if (token_result.get('word') == unknown_word and 
                        token_result.get('method') == 'context_inference'):
                        
                        inferred_meanings = token_result.get('inferred_meaning', [])
                        expected_meaning = test_case.get('expected_meaning', '')
                        
                        # Check if inference is reasonable
                        if self._inference_is_reasonable(inferred_meanings, expected_meaning):
                            correct_inferences += 1
                            inference_found = True
                            break
                
                if not inference_found:
                    # Check if word was handled at all
                    for token_result in result:
                        if token_result.get('word') == unknown_word:
                            # Word was processed but not inferred
                            break
                    else:
                        # Word was completely missed
                        pass
                        
            except Exception as e:
                print(f"Error in unknown word test: {e}")
        
        accuracy = correct_inferences / total_tests if total_tests > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct_inferences': correct_inferences,
            'total_tests': total_tests,
            'description': 'Unknown word inference accuracy'
        }
    
    def _evaluate_performance(self, tokenizer, test_cases: List[Dict]) -> Dict:
        """Evaluate tokenizer performance metrics."""
        performance_results = {
            'tokenization_speed': [],
            'memory_usage': [],
            'cache_hit_rates': [],
            'processing_times': []
        }
        
        # Test with various text lengths
        test_texts = [tc['input'] for tc in test_cases]
        
        for text in test_texts:
            # Measure tokenization speed
            start_time = time.time()
            result = tokenizer.tokenize(text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            performance_results['processing_times'].append(processing_time)
            
            # Calculate tokens per second
            total_tokens = sum(len(r.get('tokens', [])) for r in result)
            if processing_time > 0:
                tokens_per_second = total_tokens / processing_time
                performance_results['tokenization_speed'].append(tokens_per_second)
        
        # Calculate statistics
        if performance_results['processing_times']:
            avg_processing_time = statistics.mean(performance_results['processing_times'])
            max_processing_time = max(performance_results['processing_times'])
            min_processing_time = min(performance_results['processing_times'])
        else:
            avg_processing_time = max_processing_time = min_processing_time = 0.0
        
        if performance_results['tokenization_speed']:
            avg_tokens_per_second = statistics.mean(performance_results['tokenization_speed'])
            max_tokens_per_second = max(performance_results['tokenization_speed'])
        else:
            avg_tokens_per_second = max_tokens_per_second = 0.0
        
        return {
            'average_processing_time': avg_processing_time,
            'max_processing_time': max_processing_time,
            'min_processing_time': min_processing_time,
            'average_tokens_per_second': avg_tokens_per_second,
            'max_tokens_per_second': max_tokens_per_second,
            'description': 'Performance benchmarks'
        }
    
    def _evaluate_domain_specific(self, tokenizer, test_cases: List[Dict]) -> Dict:
        """Evaluate domain-specific performance."""
        technical_cases = [tc for tc in test_cases if tc['category'] == 'technical']
        
        if not technical_cases:
            return {'accuracy': 0.0, 'total_tests': 0}
        
        correct_count = 0
        total_tests = len(technical_cases)
        
        for test_case in technical_cases:
            try:
                result = tokenizer.tokenize(test_case['input'])
                predicted_tokens = [r['tokens'] for r in result if 'tokens' in r]
                predicted_tokens = [token for sublist in predicted_tokens for token in sublist]
                
                if self._tokens_match(predicted_tokens, test_case['expected_tokens']):
                    correct_count += 1
                    
            except Exception as e:
                print(f"Error in technical test: {e}")
        
        accuracy = correct_count / total_tests if total_tests > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_tests': total_tests,
            'description': 'Technical domain accuracy'
        }
    
    def _evaluate_edge_cases(self, tokenizer, test_cases: List[Dict]) -> Dict:
        """Evaluate edge case handling."""
        edge_cases = [tc for tc in test_cases if tc['category'] == 'edge_cases']
        
        if not edge_cases:
            return {'accuracy': 0.0, 'total_tests': 0}
        
        correct_count = 0
        total_tests = len(edge_cases)
        
        for test_case in edge_cases:
            try:
                result = tokenizer.tokenize(test_case['input'])
                predicted_tokens = [r['tokens'] for r in result if 'tokens' in r]
                predicted_tokens = [token for sublist in predicted_tokens for token in sublist]
                
                # For edge cases, we're more lenient with evaluation
                if self._tokens_approximately_match(predicted_tokens, test_case['expected_tokens']):
                    correct_count += 1
                    
            except Exception as e:
                print(f"Error in edge case test: {e}")
        
        accuracy = correct_count / total_tests if total_tests > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_tests': total_tests,
            'description': 'Edge case handling accuracy'
        }
    
    def _tokens_match(self, predicted: List[str], expected: List[str]) -> bool:
        """Check if predicted tokens match expected tokens exactly."""
        return predicted == expected
    
    def _tokens_approximately_match(self, predicted: List[str], expected: List[str]) -> bool:
        """Check if predicted tokens approximately match expected tokens."""
        if len(predicted) != len(expected):
            return False
        
        # Allow for some flexibility in edge cases
        matches = sum(1 for p, e in zip(predicted, expected) if p == e)
        return matches >= len(expected) * 0.8  # 80% match threshold
    
    def _inference_is_reasonable(self, inferred_meanings: List[str], expected_meaning: str) -> bool:
        """Check if inference is reasonable compared to expected meaning."""
        if not inferred_meanings:
            return False
        
        # Simple keyword matching (could be enhanced)
        expected_keywords = expected_meaning.lower().split()
        
        for meaning in inferred_meanings:
            meaning_keywords = meaning.lower().split()
            overlap = len(set(expected_keywords) & set(meaning_keywords))
            if overlap > 0:
                return True
        
        return False
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall evaluation score."""
        scores = []
        
        # Basic accuracy (weight: 0.3)
        if 'basic_accuracy' in results:
            scores.append(results['basic_accuracy']['accuracy'] * 0.3)
        
        # Compound accuracy (weight: 0.25)
        if 'compound_accuracy' in results:
            scores.append(results['compound_accuracy']['accuracy'] * 0.25)
        
        # Unknown word inference (weight: 0.2)
        if 'unknown_word_inference' in results:
            scores.append(results['unknown_word_inference']['accuracy'] * 0.2)
        
        # Domain specific (weight: 0.15)
        if 'domain_specific' in results:
            scores.append(results['domain_specific']['accuracy'] * 0.15)
        
        # Edge cases (weight: 0.1)
        if 'edge_cases' in results:
            scores.append(results['edge_cases']['accuracy'] * 0.1)
        
        return sum(scores) if scores else 0.0
    
    def benchmark_tokenizer(self, tokenizer, benchmark_texts: List[str]) -> Dict:
        """Run comprehensive benchmark on tokenizer."""
        benchmark_results = {
            'total_texts': len(benchmark_texts),
            'total_characters': sum(len(text) for text in benchmark_texts),
            'total_words': sum(len(text.split()) for text in benchmark_texts),
            'processing_times': [],
            'tokens_generated': [],
            'memory_usage': []
        }
        
        for text in benchmark_texts:
            start_time = time.time()
            result = tokenizer.tokenize(text)
            end_time = time.time()
            
            processing_time = end_time - start_time
            benchmark_results['processing_times'].append(processing_time)
            
            total_tokens = sum(len(r.get('tokens', [])) for r in result)
            benchmark_results['tokens_generated'].append(total_tokens)
        
        # Calculate statistics
        if benchmark_results['processing_times']:
            benchmark_results['average_processing_time'] = statistics.mean(benchmark_results['processing_times'])
            benchmark_results['total_processing_time'] = sum(benchmark_results['processing_times'])
            benchmark_results['tokens_per_second'] = sum(benchmark_results['tokens_generated']) / sum(benchmark_results['processing_times'])
        
        return benchmark_results
    
    def export_evaluation_results(self, results: Dict, output_path: str) -> None:
        """Export evaluation results to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def compare_tokenizers(self, tokenizers: Dict[str, any]) -> Dict:
        """Compare multiple tokenizers."""
        comparison_results = {}
        
        for name, tokenizer in tokenizers.items():
            results = self.evaluate_tokenizer(tokenizer)
            comparison_results[name] = results
        
        # Calculate rankings
        rankings = self._calculate_rankings(comparison_results)
        comparison_results['rankings'] = rankings
        
        return comparison_results
    
    def _calculate_rankings(self, results: Dict) -> Dict:
        """Calculate rankings for different metrics."""
        rankings = {}
        
        # Overall score ranking
        overall_scores = {name: result['overall_score'] for name, result in results.items()}
        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        rankings['overall'] = overall_ranking
        
        # Basic accuracy ranking
        basic_scores = {name: result['basic_accuracy']['accuracy'] for name, result in results.items()}
        basic_ranking = sorted(basic_scores.items(), key=lambda x: x[1], reverse=True)
        rankings['basic_accuracy'] = basic_ranking
        
        return rankings
