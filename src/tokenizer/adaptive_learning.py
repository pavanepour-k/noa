"""
Adaptive Learning Module for Japanese Tokenizer

This module provides pattern generalization and incremental learning:
- Pattern generalization from inferred words
- Incremental vocabulary expansion
- User feedback processing (optional)
- Learning memory for pattern recognition
"""

import json
import time
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# Import logging utilities
try:
    from ..utils.logger import get_logger, log_learning_event
except ImportError:
    # Fallback for direct execution
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def log_learning_event(logger, event_type, word, confidence, pattern_count):
        logger.info(f"Learning: {event_type} for '{word}'")


class AdaptiveLearningModule:
    """
    Adaptive learning module for continuous improvement of tokenization.
    
    Learns from new patterns and user feedback to improve accuracy
    and expand vocabulary over time.
    """
    
    def __init__(self, learning_memory_path: str = "data/learning_memory.json"):
        self.learning_memory_path = learning_memory_path
        self.learning_memory = []
        self.pattern_database = {}
        self.generalization_rules = {}
        self.feedback_history = []
        
        # Learning parameters
        self.min_pattern_frequency = 3
        self.confidence_threshold = 0.7
        self.pattern_decay_factor = 0.95
        self.max_memory_size = 10000
        
        # Initialize logger
        self.logger = get_logger('adaptive_learning')
        self.logger.info(f"Initialized AdaptiveLearningModule with path: {learning_memory_path}")
        
        # Load existing learning data
        self._load_learning_memory()
    
    def _load_learning_memory(self):
        """Load existing learning memory from file."""
        try:
            with open(self.learning_memory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.learning_memory = data.get('learning_memory', [])
                self.pattern_database = data.get('pattern_database', {})
                self.generalization_rules = data.get('generalization_rules', {})
                self.feedback_history = data.get('feedback_history', [])
        except FileNotFoundError:
            # Initialize empty learning memory
            self.learning_memory = []
            self.pattern_database = {}
            self.generalization_rules = {}
            self.feedback_history = []
    
    def _save_learning_memory(self):
        """Save learning memory to file."""
        import os
        os.makedirs(os.path.dirname(self.learning_memory_path), exist_ok=True)
        
        data = {
            'learning_memory': self.learning_memory,
            'pattern_database': self.pattern_database,
            'generalization_rules': self.generalization_rules,
            'feedback_history': self.feedback_history,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.learning_memory_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def learn_from_inference(self, word: str, inference_result: Dict, 
                           context: List[str] = None) -> None:
        """
        Learn from successful inference results.
        
        Args:
            word: Word that was inferred
            inference_result: Result from inference engine
            context: Context words used for inference
        """
        if inference_result['confidence'] < self.confidence_threshold:
            self.logger.debug(f"Low confidence inference for '{word}', skipping learning")
            return
        
        # Create learning pattern
        pattern = {
            'word': word,
            'inferred_meaning': inference_result.get('probable_meanings', []),
            'semantic_field': inference_result.get('semantic_field', 'unknown'),
            'component_analysis': inference_result.get('component_analysis', {}),
            'context': context or [],
            'confidence': inference_result['confidence'],
            'learned_at': datetime.now().isoformat(),
            'pattern_type': 'inference_success'
        }
        
        # Add to learning memory
        self.learning_memory.append(pattern)
        
        # Log learning event
        log_learning_event(self.logger, "inference_success", word, 
                          inference_result['confidence'], len(self.learning_memory))
        
        # Update pattern database
        self._update_pattern_database(pattern)
        
        # Check for generalization opportunities
        self._check_generalization_opportunities(pattern)
        
        # Clean up old patterns if memory is full
        self._cleanup_old_patterns()
    
    def learn_from_feedback(self, word: str, correct_meaning: str, 
                          user_confidence: float = 1.0) -> None:
        """
        Learn from user feedback.
        
        Args:
            word: Word that was corrected
            correct_meaning: Correct meaning provided by user
            user_confidence: User's confidence in the correction
        """
        feedback_entry = {
            'word': word,
            'correct_meaning': correct_meaning,
            'user_confidence': user_confidence,
            'feedback_at': datetime.now().isoformat(),
            'pattern_type': 'user_feedback'
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Log learning event
        log_learning_event(self.logger, "user_feedback", word, 
                          user_confidence, len(self.learning_memory))
        
        # Update learning memory with correction
        correction_pattern = {
            'word': word,
            'inferred_meaning': [correct_meaning],
            'semantic_field': 'user_corrected',
            'component_analysis': {},
            'context': [],
            'confidence': user_confidence,
            'learned_at': datetime.now().isoformat(),
            'pattern_type': 'user_correction'
        }
        
        self.learning_memory.append(correction_pattern)
        
        # Update pattern database with correction
        self._update_pattern_database(correction_pattern)
    
    def _update_pattern_database(self, pattern: Dict):
        """Update pattern database with new learning pattern."""
        word = pattern['word']
        
        if word not in self.pattern_database:
            self.pattern_database[word] = {
                'patterns': [],
                'frequency': 0,
                'last_seen': pattern['learned_at'],
                'confidence_sum': 0.0
            }
        
        # Add pattern
        self.pattern_database[word]['patterns'].append(pattern)
        self.pattern_database[word]['frequency'] += 1
        self.pattern_database[word]['last_seen'] = pattern['learned_at']
        self.pattern_database[word]['confidence_sum'] += pattern['confidence']
        
        # Update component patterns
        self._update_component_patterns(pattern)
    
    def _update_component_patterns(self, pattern: Dict):
        """Update component-based patterns."""
        component_analysis = pattern.get('component_analysis', {})
        components = component_analysis.get('components', [])
        
        if len(components) >= 2:
            # Create component combination pattern
            component_key = ''.join(sorted(components))
            
            if component_key not in self.generalization_rules:
                self.generalization_rules[component_key] = {
                    'component_combinations': [],
                    'meanings': [],
                    'frequency': 0,
                    'confidence_sum': 0.0
                }
            
            rule = self.generalization_rules[component_key]
            rule['component_combinations'].append(components)
            rule['meanings'].extend(pattern.get('inferred_meaning', []))
            rule['frequency'] += 1
            rule['confidence_sum'] += pattern['confidence']
    
    def _check_generalization_opportunities(self, pattern: Dict):
        """Check for opportunities to create generalization rules."""
        word = pattern['word']
        component_analysis = pattern.get('component_analysis', {})
        components = component_analysis.get('components', [])
        
        if len(components) < 2:
            return
        
        # Look for similar patterns
        similar_patterns = self._find_similar_patterns(components)
        
        if len(similar_patterns) >= self.min_pattern_frequency:
            # Create generalization rule
            self._create_generalization_rule(components, similar_patterns)
    
    def _find_similar_patterns(self, components: List[str]) -> List[Dict]:
        """Find patterns with similar component structures."""
        similar = []
        
        for word, data in self.pattern_database.items():
            for pattern in data['patterns']:
                pattern_components = pattern.get('component_analysis', {}).get('components', [])
                
                # Check component overlap
                overlap = len(set(components) & set(pattern_components))
                if overlap >= len(components) * 0.5:  # 50% overlap
                    similar.append(pattern)
        
        return similar
    
    def _create_generalization_rule(self, components: List[str], similar_patterns: List[Dict]):
        """Create generalization rule from similar patterns."""
        rule_id = f"rule_{len(self.generalization_rules)}"
        
        # Extract common meanings
        all_meanings = []
        for pattern in similar_patterns:
            all_meanings.extend(pattern.get('inferred_meaning', []))
        
        # Find most common meanings
        meaning_counts = Counter(all_meanings)
        common_meanings = [meaning for meaning, count in meaning_counts.most_common(3)]
        
        # Calculate average confidence
        avg_confidence = sum(p['confidence'] for p in similar_patterns) / len(similar_patterns)
        
        rule = {
            'components': components,
            'common_meanings': common_meanings,
            'confidence': avg_confidence,
            'pattern_count': len(similar_patterns),
            'created_at': datetime.now().isoformat()
        }
        
        self.generalization_rules[rule_id] = rule
    
    def apply_learned_patterns(self, word: str) -> Optional[Dict]:
        """
        Apply learned patterns to improve tokenization.
        
        Args:
            word: Word to analyze
            
        Returns:
            Learned pattern result or None
        """
        # Check direct word patterns
        if word in self.pattern_database:
            data = self.pattern_database[word]
            if data['frequency'] >= self.min_pattern_frequency:
                avg_confidence = data['confidence_sum'] / data['frequency']
                if avg_confidence >= self.confidence_threshold:
                    return {
                        'word': word,
                        'learned_meanings': [p['inferred_meaning'] for p in data['patterns']],
                        'confidence': avg_confidence,
                        'frequency': data['frequency'],
                        'source': 'learned_pattern'
                    }
        
        # Check component-based patterns
        component_result = self._apply_component_patterns(word)
        if component_result:
            return component_result
        
        return None
    
    def _apply_component_patterns(self, word: str) -> Optional[Dict]:
        """Apply component-based learned patterns."""
        # Extract components (simplified)
        components = list(word)  # This would be more sophisticated in practice
        
        # Look for matching component combinations
        for rule_id, rule in self.generalization_rules.items():
            rule_components = rule['components']
            
            # Check for component overlap
            overlap = len(set(components) & set(rule_components))
            if overlap >= len(rule_components) * 0.6:  # 60% overlap
                return {
                    'word': word,
                    'learned_meanings': rule['common_meanings'],
                    'confidence': rule['confidence'],
                    'frequency': rule['pattern_count'],
                    'source': 'component_generalization',
                    'rule_id': rule_id
                }
        
        return None
    
    def _cleanup_old_patterns(self):
        """Clean up old patterns to manage memory size."""
        if len(self.learning_memory) <= self.max_memory_size:
            return
        
        # Sort by recency and confidence
        self.learning_memory.sort(
            key=lambda x: (x['confidence'], x['learned_at']), 
            reverse=True
        )
        
        # Keep top patterns
        self.learning_memory = self.learning_memory[:self.max_memory_size]
        
        # Update pattern database
        self._rebuild_pattern_database()
    
    def _rebuild_pattern_database(self):
        """Rebuild pattern database from current learning memory."""
        self.pattern_database = {}
        
        for pattern in self.learning_memory:
            word = pattern['word']
            if word not in self.pattern_database:
                self.pattern_database[word] = {
                    'patterns': [],
                    'frequency': 0,
                    'last_seen': pattern['learned_at'],
                    'confidence_sum': 0.0
                }
            
            self.pattern_database[word]['patterns'].append(pattern)
            self.pattern_database[word]['frequency'] += 1
            self.pattern_database[word]['confidence_sum'] += pattern['confidence']
    
    def get_learning_stats(self) -> Dict:
        """Get learning module statistics."""
        total_patterns = len(self.learning_memory)
        unique_words = len(self.pattern_database)
        generalization_rules = len(self.generalization_rules)
        feedback_entries = len(self.feedback_history)
        
        # Calculate learning effectiveness
        high_confidence_patterns = sum(
            1 for p in self.learning_memory 
            if p['confidence'] >= self.confidence_threshold
        )
        
        effectiveness = (high_confidence_patterns / total_patterns) if total_patterns > 0 else 0.0
        
        return {
            'total_patterns': total_patterns,
            'unique_words_learned': unique_words,
            'generalization_rules': generalization_rules,
            'feedback_entries': feedback_entries,
            'learning_effectiveness': effectiveness,
            'high_confidence_patterns': high_confidence_patterns,
            'memory_usage': len(self.learning_memory) / self.max_memory_size
        }
    
    def export_learned_patterns(self, output_path: str) -> None:
        """Export learned patterns to file."""
        export_data = {
            'learning_memory': self.learning_memory,
            'pattern_database': self.pattern_database,
            'generalization_rules': self.generalization_rules,
            'export_timestamp': datetime.now().isoformat(),
            'stats': self.get_learning_stats()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    def import_learned_patterns(self, input_path: str) -> None:
        """Import learned patterns from file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            self.learning_memory = data.get('learning_memory', [])
            self.pattern_database = data.get('pattern_database', {})
            self.generalization_rules = data.get('generalization_rules', {})
            
            # Save to current memory
            self._save_learning_memory()
    
    def reset_learning_memory(self) -> None:
        """Reset all learning memory."""
        self.learning_memory = []
        self.pattern_database = {}
        self.generalization_rules = {}
        self.feedback_history = []
        self._save_learning_memory()
    
    def get_pattern_suggestions(self, word: str) -> List[Dict]:
        """Get pattern suggestions for a word."""
        suggestions = []
        
        # Direct word suggestions
        if word in self.pattern_database:
            data = self.pattern_database[word]
            suggestions.append({
                'type': 'direct_match',
                'word': word,
                'meanings': [p['inferred_meaning'] for p in data['patterns']],
                'confidence': data['confidence_sum'] / data['frequency'],
                'frequency': data['frequency']
            })
        
        # Component-based suggestions
        component_suggestions = self._get_component_suggestions(word)
        suggestions.extend(component_suggestions)
        
        return suggestions
    
    def _get_component_suggestions(self, word: str) -> List[Dict]:
        """Get suggestions based on component patterns."""
        suggestions = []
        components = list(word)  # Simplified component extraction
        
        for rule_id, rule in self.generalization_rules.items():
            rule_components = rule['components']
            overlap = len(set(components) & set(rule_components))
            
            if overlap > 0:
                suggestions.append({
                    'type': 'component_match',
                    'word': word,
                    'meanings': rule['common_meanings'],
                    'confidence': rule['confidence'],
                    'overlap_ratio': overlap / len(rule_components),
                    'rule_id': rule_id
                })
        
        return suggestions
