"""
Hybrid Inference System for Japanese Tokenizer

This module integrates fast rule-based engine and context semantic engine:
- 0.7 confidence threshold for engine selection
- Weighted scoring system (fast: 0.6, context: 0.4)
- Multi-evidence integration for final decision
- Unknown word inference with hybrid approach
"""

import time
from typing import List, Dict, Tuple, Optional
from .fast_engine import FastRuleBasedEngine
from .context_engine import ContextSemanticEngine
from .base_tokenizer import BaseTokenizer

# Import logging utilities
try:
    from ..utils.logger import get_logger, log_performance, log_tokenization_result, log_engine_selection
except ImportError:
    # Fallback for direct execution
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def log_performance(logger, operation, duration, details=None):
        logger.info(f"Performance: {operation} took {duration:.4f}s")
    def log_tokenization_result(logger, text, result_count, processing_time, unknown_words=0):
        logger.info(f"Tokenization: {result_count} tokens ({processing_time:.4f}s)")
    def log_engine_selection(logger, word, engine, confidence, reason):
        logger.debug(f"Engine Selection: {word} -> {engine}")


class HybridTokenizer:
    """
    Hybrid tokenizer combining fast rule-based and context semantic engines.
    
    Uses intelligent engine selection based on confidence thresholds
    and weighted scoring for optimal accuracy and performance.
    """
    
    def __init__(self, kanji_db=None, cache_system=None):
        self.kanji_db = kanji_db
        self.cache_system = cache_system
        
        # Initialize engines
        self.fast_engine = FastRuleBasedEngine(kanji_db)
        self.context_engine = ContextSemanticEngine(kanji_db)
        self.base_tokenizer = BaseTokenizer()
        
        # Configuration
        self.confidence_threshold = 0.7
        self.fast_engine_weight = 0.6
        self.context_engine_weight = 0.4
        
        # Performance tracking
        self.performance_metrics = {
            'fast_engine_usage': 0,
            'context_engine_usage': 0,
            'hybrid_usage': 0,
            'total_tokens': 0,
            'unknown_words': 0,
            'average_confidence': 0.0
        }
        
        # Initialize logger
        self.logger = get_logger('hybrid_tokenizer')
        self.logger.info("Initialized HybridTokenizer")
    
    def tokenize(self, text: str) -> List[Dict]:
        """
        Main tokenization method using hybrid approach.
        
        Args:
            text: Input Japanese text
            
        Returns:
            List of tokenization results with confidence scores
        """
        start_time = time.time()
        self.logger.info(f"Starting tokenization of text: {text[:50]}...")
        
        # Check if text is large enough for batch processing
        if len(text) > 1000:  # Large text threshold
            return self._tokenize_large_text(text, start_time)
        
        # Preprocess text
        processed_text = self.base_tokenizer.preprocess_text(text)
        
        # Basic segmentation
        words = self.base_tokenizer.basic_segmentation(processed_text)
        self.logger.debug(f"Segmented into {len(words)} words")
        
        # Tokenize each word using hybrid approach
        results = []
        unknown_count = 0
        for i, word in enumerate(words):
            token_result = self._tokenize_word_hybrid(word, words, i)
            results.append(token_result)
            if token_result.get('method') == 'context_inference':
                unknown_count += 1
        
        processing_time = time.time() - start_time
        
        # Log tokenization results
        log_tokenization_result(self.logger, text, len(results), processing_time, unknown_count)
        
        # Update performance metrics
        self._update_performance_metrics(results, processing_time)
        
        return results
    
    def _tokenize_large_text(self, text: str, start_time: float) -> List[Dict]:
        """Optimized tokenization for large texts using batch processing."""
        self.logger.info(f"Using batch processing for large text ({len(text)} chars)")
        
        # Split into sentences for batch processing
        sentences = self.base_tokenizer.segment_sentences(text)
        all_results = []
        total_unknown_count = 0
        
        # Process sentences in batches
        batch_size = 10  # Process 10 sentences at a time
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_results = self._process_sentence_batch(batch_sentences)
            all_results.extend(batch_results)
            
            # Count unknown words in batch
            batch_unknown = sum(1 for r in batch_results if r.get('method') == 'context_inference')
            total_unknown_count += batch_unknown
            
            # Log progress for large batches
            if len(sentences) > 50:
                progress = min(100, (i + batch_size) * 100 // len(sentences))
                self.logger.debug(f"Batch processing progress: {progress}%")
        
        processing_time = time.time() - start_time
        
        # Log tokenization results
        log_tokenization_result(self.logger, text, len(all_results), processing_time, total_unknown_count)
        
        # Update performance metrics
        self._update_performance_metrics(all_results, processing_time)
        
        return all_results
    
    def _process_sentence_batch(self, sentences: List[str]) -> List[Dict]:
        """Process a batch of sentences efficiently."""
        batch_results = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Preprocess and segment
            processed_text = self.base_tokenizer.preprocess_text(sentence)
            words = self.base_tokenizer.basic_segmentation(processed_text)
            
            # Tokenize words in sentence
            for i, word in enumerate(words):
                token_result = self._tokenize_word_hybrid(word, words, i)
                batch_results.append(token_result)
        
        return batch_results
    
    def _tokenize_word_hybrid(self, word: str, context_words: List[str], 
                             word_index: int) -> Dict:
        """
        Tokenize word using hybrid approach.
        
        Args:
            word: Word to tokenize
            context_words: All words in text for context
            word_index: Index of word in context
            
        Returns:
            Tokenization result with hybrid analysis
        """
        # Early termination for very short words
        if len(word) <= 1:
            return self._create_simple_result(word, 'character_level', 0.9)
        
        # Get results from both engines
        fast_result = self.fast_engine.tokenize_word(word)
        
        # Early termination for high confidence fast results
        if fast_result['confidence'] >= self.confidence_threshold:
            # Use fast engine result
            result = self._enhance_fast_result(fast_result, {})
            result['engine_used'] = 'fast'
            self.performance_metrics['fast_engine_usage'] += 1
            log_engine_selection(self.logger, word, 'fast', fast_result['confidence'], 'high confidence')
            return result
        
        # Early termination for very low confidence (likely unknown word)
        if fast_result['confidence'] < 0.2:
            return self._create_unknown_word_result(word, context_words, word_index)
        
        # Use hybrid approach for medium confidence
        context_result = self.context_engine.analyze_context_window(
            context_words, word_index
        )
        
        result = self._combine_engines(fast_result, context_result, word, context_words)
        result['engine_used'] = 'hybrid'
        self.performance_metrics['hybrid_usage'] += 1
        log_engine_selection(self.logger, word, 'hybrid', result['confidence'], 'medium confidence')
        
        # Add context information
        result['context_analysis'] = context_result
        result['processing_time'] = time.time()
        
        return result
    
    def _create_simple_result(self, word: str, method: str, confidence: float) -> Dict:
        """Create a simple result for basic cases."""
        return {
            'word': word,
            'tokens': [word],
            'confidence': confidence,
            'method': method,
            'engine_used': 'simple',
            'processing_time': time.time()
        }
    
    def _create_unknown_word_result(self, word: str, context_words: List[str], word_index: int) -> Dict:
        """Create result for unknown words with minimal processing."""
        return {
            'word': word,
            'tokens': [word],
            'confidence': 0.3,  # Low confidence for unknown words
            'method': 'unknown_word',
            'engine_used': 'minimal',
            'processing_time': time.time(),
            'note': 'Unknown word - minimal processing applied'
        }
    
    def _enhance_fast_result(self, fast_result: Dict, context_result: Dict) -> Dict:
        """Enhance fast engine result with context information."""
        enhanced_result = fast_result.copy()
        
        # Add semantic field information
        enhanced_result['semantic_field'] = context_result.get('dominant_field', 'unknown')
        enhanced_result['context_confidence'] = context_result.get('field_scores', {})
        
        # Boost confidence if context supports the result
        if context_result.get('dominant_field') != 'unknown':
            enhanced_result['confidence'] = min(enhanced_result['confidence'] + 0.1, 1.0)
        
        return enhanced_result
    
    def _combine_engines(self, fast_result: Dict, context_result: Dict, 
                       word: str, context_words: List[str]) -> Dict:
        """Combine results from both engines using weighted scoring."""
        # Calculate weighted confidence
        fast_confidence = fast_result['confidence'] * self.fast_engine_weight
        context_confidence = self._calculate_context_confidence(context_result) * self.context_engine_weight
        combined_confidence = fast_confidence + context_confidence
        
        # Determine final tokens
        if fast_result['confidence'] > 0.3:
            # Use fast engine tokens as base
            final_tokens = fast_result['tokens']
            base_method = fast_result['method']
        else:
            # Use character-level segmentation
            final_tokens = list(word)
            base_method = 'character_level'
        
        # Check for unknown word inference
        if self._is_unknown_word(word, fast_result):
            inference_result = self._infer_unknown_word(word, context_words)
            final_tokens = inference_result.get('tokens', final_tokens)
            combined_confidence = max(combined_confidence, inference_result.get('confidence', 0.0))
        
        return {
            'word': word,
            'tokens': final_tokens,
            'confidence': combined_confidence,
            'method': f'hybrid_{base_method}',
            'fast_confidence': fast_result['confidence'],
            'context_confidence': context_confidence,
            'semantic_field': context_result.get('dominant_field', 'unknown'),
            'analysis': {
                'fast_analysis': fast_result,
                'context_analysis': context_result,
                'hybrid_combination': True
            }
        }
    
    def _calculate_context_confidence(self, context_result: Dict) -> float:
        """Calculate confidence based on context analysis."""
        confidence = 0.0
        
        # Field identification confidence
        field_scores = context_result.get('field_scores', {})
        if field_scores:
            max_field_score = max(field_scores.values())
            confidence += max_field_score * 0.4
        
        # Semantic relations confidence
        relations = context_result.get('semantic_relations', [])
        if relations:
            avg_relation_strength = sum(r['strength'] for r in relations) / len(relations)
            confidence += avg_relation_strength * 0.3
        
        # Context window quality
        context_words = context_result.get('context_words', [])
        meaningful_words = [w for w in context_words if len(w) > 1]
        if meaningful_words:
            confidence += min(len(meaningful_words) / 5, 1.0) * 0.3
        
        return min(confidence, 1.0)
    
    def _is_unknown_word(self, word: str, fast_result: Dict) -> bool:
        """Check if word is unknown based on fast engine result."""
        return (fast_result['confidence'] < 0.5 and 
                fast_result['method'] in ['character_level', 'unknown'])
    
    def _infer_unknown_word(self, word: str, context_words: List[str]) -> Dict:
        """Infer unknown word using context semantic engine."""
        # Get context around the word
        word_index = context_words.index(word) if word in context_words else 0
        context_window = self._get_context_window(context_words, word_index)
        
        # Use context engine for inference
        inference_result = self.context_engine.infer_unknown_word_meaning(
            word, context_window
        )
        
        # Convert inference to tokenization result
        return {
            'tokens': [word],  # Keep as single token for unknown words
            'confidence': inference_result['confidence'],
            'method': 'context_inference',
            'inferred_meaning': inference_result['probable_meanings'],
            'semantic_field': inference_result['context_field'],
            'component_analysis': inference_result['component_analysis']
        }
    
    def _get_context_window(self, words: List[str], word_index: int, 
                           window_size: int = 5) -> List[str]:
        """Get context window around word."""
        start = max(0, word_index - window_size)
        end = min(len(words), word_index + window_size + 1)
        return words[start:end]
    
    def _update_performance_metrics(self, results: List[Dict], processing_time: float):
        """Update performance metrics with incremental updates."""
        # Incremental counters (avoid full recalculation)
        self.performance_metrics['total_tokens'] += len(results)
        
        # Count unknown words incrementally
        unknown_count = sum(1 for r in results if r.get('method') == 'context_inference')
        self.performance_metrics['unknown_words'] += unknown_count
        
        # Incremental confidence calculation
        if results:
            total_confidence = sum(r['confidence'] for r in results)
            # Update running average
            current_avg = self.performance_metrics['average_confidence']
            total_tokens = self.performance_metrics['total_tokens']
            
            if total_tokens > 0:
                # Weighted average update
                self.performance_metrics['average_confidence'] = (
                    (current_avg * (total_tokens - len(results)) + total_confidence) / total_tokens
                )
        
        # Track processing time per token
        if len(results) > 0:
            time_per_token = processing_time / len(results)
            if 'time_per_token' not in self.performance_metrics:
                self.performance_metrics['time_per_token'] = []
            
            self.performance_metrics['time_per_token'].append(time_per_token)
            
            # Keep only last 100 measurements to avoid memory growth
            if len(self.performance_metrics['time_per_token']) > 100:
                self.performance_metrics['time_per_token'] = self.performance_metrics['time_per_token'][-100:]
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        total_usage = (self.performance_metrics['fast_engine_usage'] + 
                       self.performance_metrics['hybrid_usage'])
        
        if total_usage > 0:
            fast_usage_rate = self.performance_metrics['fast_engine_usage'] / total_usage
            hybrid_usage_rate = self.performance_metrics['hybrid_usage'] / total_usage
        else:
            fast_usage_rate = hybrid_usage_rate = 0.0
        
        return {
            'engine_usage': {
                'fast_engine_rate': fast_usage_rate,
                'hybrid_engine_rate': hybrid_usage_rate,
                'fast_engine_count': self.performance_metrics['fast_engine_usage'],
                'hybrid_count': self.performance_metrics['hybrid_usage']
            },
            'tokenization_stats': {
                'total_tokens': self.performance_metrics['total_tokens'],
                'unknown_words': self.performance_metrics['unknown_words'],
                'unknown_word_rate': (self.performance_metrics['unknown_words'] / 
                                    max(self.performance_metrics['total_tokens'], 1)),
                'average_confidence': self.performance_metrics['average_confidence']
            },
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'fast_engine_weight': self.fast_engine_weight,
                'context_engine_weight': self.context_engine_weight
            }
        }
    
    def adjust_confidence_threshold(self, new_threshold: float):
        """Adjust confidence threshold for engine selection."""
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
    
    def adjust_engine_weights(self, fast_weight: float, context_weight: float):
        """Adjust engine weights for hybrid scoring."""
        total_weight = fast_weight + context_weight
        if total_weight > 0:
            self.fast_engine_weight = fast_weight / total_weight
            self.context_engine_weight = context_weight / total_weight
    
    def get_tokenization_summary(self, results: List[Dict]) -> Dict:
        """Get summary of tokenization results."""
        if not results:
            return {}
        
        # Count by method
        method_counts = {}
        for result in results:
            method = result.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Confidence distribution
        confidences = [r['confidence'] for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        # Unknown words
        unknown_words = [r for r in results if r.get('method') == 'context_inference']
        
        return {
            'total_tokens': len(results),
            'method_distribution': method_counts,
            'confidence_stats': {
                'average': avg_confidence,
                'minimum': min_confidence,
                'maximum': max_confidence
            },
            'unknown_words': {
                'count': len(unknown_words),
                'rate': len(unknown_words) / len(results),
                'words': [r['word'] for r in unknown_words]
            }
        }
