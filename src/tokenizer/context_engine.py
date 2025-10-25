"""
Context Semantic Engine for Japanese Tokenizer

This module provides context-aware tokenization using:
- Semantic field identification (technology, nature, emotion, academic)
- Context window analysis (5 tokens before/after)
- Kanji component meaning extraction
- Contextual meaning inference for unknown words
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import re


class ContextSemanticEngine:
    """
    Context-aware semantic analysis engine for Japanese text.
    
    Analyzes semantic fields and context to improve tokenization
    and unknown word inference.
    """
    
    def __init__(self, kanji_db=None):
        self.kanji_db = kanji_db
        self.context_window = 5
        
        # Caching for semantic field analysis
        self._semantic_field_cache = {}
        self._max_cache_size = 1000
        
        # Semantic field definitions
        self.semantic_fields = {
            'technology': {
                'keywords': ['コンピュータ', 'ソフトウェア', 'データ', 'システム', 'ネットワーク', 
                           'プログラミング', 'アルゴリズム', '機械学習', '人工知能'],
                'kanji_patterns': ['電', '機', '算', '計', '制', '制御', '情報', '通信'],
                'weight': 1.0
            },
            'nature': {
                'keywords': ['自然', '環境', '生物', '植物', '動物', '気候', '天気', '季節',
                           '山', '川', '海', '空', '太陽', '月', '星'],
                'kanji_patterns': ['自', '然', '生', '物', '植', '動', '気', '天'],
                'weight': 1.0
            },
            'emotion': {
                'keywords': ['感情', '愛情', '悲しみ', '喜び', '怒り', '驚き', '恐れ', '希望',
                           '幸せ', '悲しい', '嬉しい', '楽しい', '苦しい'],
                'kanji_patterns': ['心', '情', '愛', '悲', '喜', '怒', '驚', '恐'],
                'weight': 1.0
            },
            'academic': {
                'keywords': ['研究', '理論', '分析', '実験', '論文', '学問', '知識', '教育',
                           '大学', '教授', '学生', '学習', '勉強'],
                'kanji_patterns': ['学', '研', '究', '理', '論', '分', '析', '実', '験'],
                'weight': 1.0
            },
            'business': {
                'keywords': ['会社', '企業', '経営', '市場', '経済', '投資', '利益', '売上',
                           '顧客', 'サービス', '商品', '価格'],
                'kanji_patterns': ['会', '社', '企', '業', '経', '営', '市', '場'],
                'weight': 1.0
            }
        }
        
        # Context analysis patterns
        self.context_patterns = {
            'noun_modifier': re.compile(r'[\u4e00-\u9faf]+の[\u4e00-\u9faf]+'),
            'verb_object': re.compile(r'[\u3040-\u309f]+[\u4e00-\u9faf]+'),
            'adjective_noun': re.compile(r'[\u3040-\u309f]+[\u4e00-\u9faf]+'),
            'compound_kanji': re.compile(r'[\u4e00-\u9faf]{2,}')
        }
    
    def analyze_semantic_field(self, text: str) -> Dict[str, float]:
        """
        Analyze semantic field of text with caching.
        
        Returns:
            Dictionary mapping field names to confidence scores
        """
        # Check cache first
        cache_key = text[:50]  # Use first 50 chars as cache key
        if cache_key in self._semantic_field_cache:
            return self._semantic_field_cache[cache_key]
        
        field_scores = {}
        
        for field_name, field_info in self.semantic_fields.items():
            score = self._calculate_field_score(text, field_info)
            field_scores[field_name] = score
        
        # Normalize scores
        total_score = sum(field_scores.values())
        if total_score > 0:
            field_scores = {k: v/total_score for k, v in field_scores.items()}
        
        # Cache the result
        self._cache_semantic_field_result(cache_key, field_scores)
        
        return field_scores
    
    def _cache_semantic_field_result(self, cache_key: str, result: Dict[str, float]) -> None:
        """Cache semantic field analysis result."""
        # Manage cache size
        if len(self._semantic_field_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._semantic_field_cache))
            del self._semantic_field_cache[oldest_key]
        
        self._semantic_field_cache[cache_key] = result
    
    def _calculate_field_score(self, text: str, field_info: Dict) -> float:
        """Calculate semantic field score for text."""
        score = 0.0
        
        # Keyword matching
        keyword_matches = sum(1 for keyword in field_info['keywords'] 
                             if keyword in text)
        score += keyword_matches * 0.3
        
        # Kanji pattern matching
        kanji_matches = sum(1 for pattern in field_info['kanji_patterns']
                           if pattern in text)
        score += kanji_matches * 0.2
        
        # Context pattern analysis
        context_matches = 0
        for pattern in self.context_patterns.values():
            matches = pattern.findall(text)
            context_matches += len(matches)
        
        score += context_matches * 0.1
        
        return score * field_info['weight']
    
    def analyze_context_window(self, words: List[str], target_index: int) -> Dict:
        """
        Analyze context window around target word.
        
        Args:
            words: List of words in text
            target_index: Index of target word
            
        Returns:
            Context analysis results
        """
        start_idx = max(0, target_index - self.context_window)
        end_idx = min(len(words), target_index + self.context_window + 1)
        
        context_words = words[start_idx:end_idx]
        target_word = words[target_index] if target_index < len(words) else ""
        
        # Analyze semantic relationships
        semantic_relations = self._analyze_semantic_relations(target_word, context_words)
        
        # Identify dominant semantic field
        context_text = " ".join(context_words)
        field_scores = self.analyze_semantic_field(context_text)
        dominant_field = max(field_scores.items(), key=lambda x: x[1])[0] if field_scores else "general"
        
        return {
            'context_words': context_words,
            'target_word': target_word,
            'semantic_relations': semantic_relations,
            'dominant_field': dominant_field,
            'field_scores': field_scores,
            'context_span': (start_idx, end_idx)
        }
    
    def _analyze_semantic_relations(self, target_word: str, context_words: List[str]) -> List[Dict]:
        """Analyze semantic relations between target word and context."""
        relations = []
        
        for context_word in context_words:
            if context_word == target_word:
                continue
            
            relation = self._calculate_semantic_relation(target_word, context_word)
            if relation['strength'] > 0.3:
                relations.append(relation)
        
        return relations
    
    def _calculate_semantic_relation(self, word1: str, word2: str) -> Dict:
        """Calculate semantic relation between two words."""
        relation = {
            'word1': word1,
            'word2': word2,
            'relation_type': 'unknown',
            'strength': 0.0
        }
        
        # Check for kanji component overlap
        if self.kanji_db:
            components1 = self.kanji_db.analyze_components(word1)
            components2 = self.kanji_db.analyze_components(word2)
            
            overlap = len(set(components1) & set(components2))
            if overlap > 0:
                relation['relation_type'] = 'component_overlap'
                relation['strength'] = overlap / max(len(components1), len(components2))
        
        # Check for semantic field similarity
        field1 = self._get_word_semantic_field(word1)
        field2 = self._get_word_semantic_field(word2)
        
        if field1 == field2 and field1 != 'unknown':
            relation['relation_type'] = 'semantic_field_match'
            relation['strength'] = max(relation['strength'], 0.8)
        
        return relation
    
    def _get_word_semantic_field(self, word: str) -> str:
        """Get semantic field for a word."""
        if not self.kanji_db:
            return 'unknown'
        
        return self.kanji_db.get_semantic_field(word)
    
    def infer_unknown_word_meaning(self, unknown_word: str, context: List[str]) -> Dict:
        """
        Infer meaning of unknown word based on context.
        
        Args:
            unknown_word: Word to analyze
            context: Surrounding context words
            
        Returns:
            Inferred meaning with confidence
        """
        # Analyze kanji components
        component_analysis = self._analyze_kanji_components(unknown_word)
        
        # Analyze context semantic field
        context_text = " ".join(context)
        field_scores = self.analyze_semantic_field(context_text)
        dominant_field = max(field_scores.items(), key=lambda x: x[1])[0] if field_scores else "general"
        
        # Generate probable meanings
        probable_meanings = self._generate_probable_meanings(
            unknown_word, component_analysis, dominant_field, context
        )
        
        # Calculate confidence
        confidence = self._calculate_inference_confidence(
            unknown_word, component_analysis, context, probable_meanings
        )
        
        return {
            'word': unknown_word,
            'component_analysis': component_analysis,
            'context_field': dominant_field,
            'probable_meanings': probable_meanings,
            'confidence': confidence,
            'context_clues': self._extract_context_clues(context)
        }
    
    def _analyze_kanji_components(self, word: str) -> Dict:
        """Analyze kanji components for semantic inference."""
        if not self.kanji_db:
            return {'components': [], 'meanings': [], 'semantic_field': 'unknown'}
        
        components = self.kanji_db.analyze_components(word)
        meanings = []
        semantic_fields = []
        
        for component in components:
            if self.kanji_db:
                info = self.kanji_db.get_kanji_info(component)
                if info:
                    meanings.extend(info.get('meanings', []))
                    semantic_fields.append(info.get('semantic_field', 'unknown'))
        
        return {
            'components': components,
            'meanings': meanings,
            'semantic_fields': semantic_fields,
            'dominant_field': max(set(semantic_fields), key=semantic_fields.count) if semantic_fields else 'unknown'
        }
    
    def _generate_probable_meanings(self, word: str, component_analysis: Dict, 
                                  dominant_field: str, context: List[str]) -> List[str]:
        """Generate probable meanings for unknown word."""
        meanings = []
        
        # Component-based meanings
        if component_analysis['meanings']:
            meanings.extend(component_analysis['meanings'][:3])  # Top 3 meanings
        
        # Context-based inference
        context_meanings = self._infer_from_context(word, context, dominant_field)
        meanings.extend(context_meanings)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_meanings = []
        for meaning in meanings:
            if meaning not in seen:
                seen.add(meaning)
                unique_meanings.append(meaning)
        
        return unique_meanings[:5]  # Return top 5 meanings
    
    def _infer_from_context(self, word: str, context: List[str], field: str) -> List[str]:
        """Infer meaning from context and semantic field."""
        meanings = []
        
        # Field-specific inference
        if field in self.semantic_fields:
            field_keywords = self.semantic_fields[field]['keywords']
            for keyword in field_keywords:
                if any(char in word for char in keyword):
                    meanings.append(f"related to {field}")
        
        # Context word similarity
        for context_word in context:
            if self._words_are_semantically_related(word, context_word):
                meanings.append(f"similar to {context_word}")
        
        return meanings
    
    def _words_are_semantically_related(self, word1: str, word2: str) -> bool:
        """Check if two words are semantically related."""
        if not self.kanji_db:
            return False
        
        # Check component overlap
        components1 = self.kanji_db.analyze_components(word1)
        components2 = self.kanji_db.analyze_components(word2)
        
        overlap = len(set(components1) & set(components2))
        return overlap > 0
    
    def _calculate_inference_confidence(self, word: str, component_analysis: Dict,
                                      context: List[str], meanings: List[str]) -> float:
        """Calculate confidence in inference result."""
        confidence = 0.0
        
        # Component analysis confidence
        if component_analysis['components']:
            confidence += 0.3
        
        # Context strength
        context_strength = len([w for w in context if self._is_meaningful_word(w)]) / max(len(context), 1)
        confidence += context_strength * 0.3
        
        # Meaning diversity
        if len(meanings) > 1:
            confidence += 0.2
        
        # Field consistency
        if component_analysis['dominant_field'] != 'unknown':
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _is_meaningful_word(self, word: str) -> bool:
        """Check if word is meaningful (not particles, etc.)."""
        particles = {'は', 'が', 'を', 'に', 'で', 'と', 'から', 'まで', 'より', 'へ', 'の', 'も'}
        return word not in particles and len(word) > 1
    
    def _extract_context_clues(self, context: List[str]) -> List[str]:
        """Extract context clues for meaning inference."""
        clues = []
        
        for word in context:
            if self._is_meaningful_word(word):
                # Check if word provides semantic clues
                if any(char in word for char in '学研理論分析実験'):
                    clues.append("academic context")
                elif any(char in word for char in '感情愛情悲喜怒'):
                    clues.append("emotional context")
                elif any(char in word for char in '自然環境生物'):
                    clues.append("natural context")
        
        return clues
    
    def get_engine_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'semantic_fields_count': len(self.semantic_fields),
            'context_window_size': self.context_window,
            'context_patterns_count': len(self.context_patterns),
            'analysis_capabilities': [
                'semantic field identification',
                'context window analysis',
                'unknown word inference',
                'component-based meaning extraction'
            ]
        }
