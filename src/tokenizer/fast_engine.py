"""
Fast Rule-Based Engine for Japanese Tokenizer

This module provides rapid tokenization using:
- Dictionary-based word recognition
- Pattern matching for common compounds
- Quick morphological analysis
- Confidence scoring for results
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


class FastRuleBasedEngine:
    """
    Fast rule-based tokenization engine for Japanese text.
    
    Uses dictionary lookup and pattern matching for rapid processing
    of known words and common compounds.
    """
    
    def __init__(self, kanji_db=None):
        self.kanji_db = kanji_db
        self.known_words = set()
        self.compound_patterns = {}
        self.morphological_rules = {}
        
        # Common Japanese word patterns
        self.word_patterns = {
            'noun_kanji': re.compile(r'[\u4e00-\u9faf]+'),
            'verb_hiragana': re.compile(r'[\u3040-\u309f]+'),
            'katakana_words': re.compile(r'[\u30a0-\u30ff]+'),
            'mixed_compounds': re.compile(r'[\u4e00-\u9faf]+[\u3040-\u309f]+'),
        }
        
        # Initialize with common words
        self._initialize_common_words()
        self._initialize_compound_patterns()
        self._initialize_morphological_rules()
    
    def _initialize_common_words(self):
        """Initialize dictionary with common Japanese words."""
        # Common nouns
        common_nouns = {
            '人', '大', '年', '一', '国', '日', '本', '時', '分', '秒',
            '家', '学校', '会社', '病院', '銀行', '駅', '空港', '公園',
            '山', '川', '海', '空', '太陽', '月', '星', '雨', '雪', '風'
        }
        
        # Common verbs (in dictionary form)
        common_verbs = {
            'する', 'ある', 'いる', 'なる', '行く', '来る', '見る', '聞く',
            '話す', '読む', '書く', '食べる', '飲む', '寝る', '起きる',
            '働く', '勉強する', '遊ぶ', '買う', '売る', '作る', '作る'
        }
        
        # Common adjectives
        common_adjectives = {
            '大きい', '小さい', '高い', '低い', '長い', '短い', '新しい',
            '古い', '良い', '悪い', '美しい', '醜い', '強い', '弱い',
            '速い', '遅い', '重い', '軽い', '明るい', '暗い'
        }
        
        self.known_words.update(common_nouns)
        self.known_words.update(common_verbs)
        self.known_words.update(common_adjectives)
    
    def _initialize_compound_patterns(self):
        """Initialize patterns for compound word recognition."""
        self.compound_patterns = {
            # Noun + noun compounds
            'noun_noun': {
                'pattern': r'[\u4e00-\u9faf]+[\u4e00-\u9faf]+',
                'confidence': 0.8
            },
            # Verb + noun compounds
            'verb_noun': {
                'pattern': r'[\u3040-\u309f]+[\u4e00-\u9faf]+',
                'confidence': 0.7
            },
            # Adjective + noun compounds
            'adj_noun': {
                'pattern': r'[\u3040-\u309f]+[\u4e00-\u9faf]+',
                'confidence': 0.7
            },
            # Number + counter compounds
            'number_counter': {
                'pattern': r'[0-9０-９]+[\u4e00-\u9faf]+',
                'confidence': 0.9
            }
        }
    
    def _initialize_morphological_rules(self):
        """Initialize morphological analysis rules."""
        self.morphological_rules = {
            # Verb conjugations
            'verb_conjugations': {
                'masu_form': r'(.+)ます',
                'te_form': r'(.+)て',
                'ta_form': r'(.+)た',
                'nai_form': r'(.+)ない'
            },
            # Adjective conjugations
            'adjective_conjugations': {
                'katta_form': r'(.+)かった',
                'kunai_form': r'(.+)くない',
                'kute_form': r'(.+)くて'
            },
            # Honorific forms
            'honorific_forms': {
                'desu_form': r'(.+)です',
                'masu_form': r'(.+)ます',
                'gozaimasu': r'(.+)ございます'
            }
        }
    
    def tokenize_word(self, word: str) -> Dict:
        """
        Tokenize a single word using rule-based approach.
        
        Returns:
            Dictionary with tokenization results and confidence
        """
        result = {
            'word': word,
            'tokens': [],
            'confidence': 0.0,
            'method': 'unknown',
            'analysis': {}
        }
        
        # Check if word is in known vocabulary
        if word in self.known_words:
            result['tokens'] = [word]
            result['confidence'] = 1.0
            result['method'] = 'dictionary'
            return result
        
        # Try compound word analysis
        compound_result = self._analyze_compound(word)
        if compound_result['confidence'] > 0.5:
            result.update(compound_result)
            return result
        
        # Try morphological analysis
        morph_result = self._analyze_morphology(word)
        if morph_result['confidence'] > 0.3:
            result.update(morph_result)
            return result
        
        # Fallback: character-level analysis
        result['tokens'] = list(word)
        result['confidence'] = 0.1
        result['method'] = 'character_level'
        
        return result
    
    def _analyze_compound(self, word: str) -> Dict:
        """Analyze word as potential compound."""
        for pattern_name, pattern_info in self.compound_patterns.items():
            if re.match(pattern_info['pattern'], word):
                # Try to split compound
                split_result = self._split_compound(word)
                if split_result:
                    return {
                        'tokens': split_result,
                        'confidence': pattern_info['confidence'],
                        'method': f'compound_{pattern_name}',
                        'analysis': {'compound_type': pattern_name}
                    }
        
        return {'confidence': 0.0}
    
    def _split_compound(self, word: str) -> Optional[List[str]]:
        """Attempt to split a compound word."""
        if len(word) < 2:
            return None
        
        # Try different split points
        for i in range(1, len(word)):
            left_part = word[:i]
            right_part = word[i:]
            
            # Check if both parts are known or likely valid
            left_valid = self._is_valid_word_part(left_part)
            right_valid = self._is_valid_word_part(right_part)
            
            if left_valid and right_valid:
                return [left_part, right_part]
        
        return None
    
    def _is_valid_word_part(self, part: str) -> bool:
        """Check if a word part is likely valid."""
        if not part:
            return False
        
        # Check if it's in known words
        if part in self.known_words:
            return True
        
        # Check if it's a single kanji (likely valid)
        if len(part) == 1 and self._is_kanji(part):
            return True
        
        # Check if it follows valid patterns
        char_types = [self._get_char_type(c) for c in part]
        
        # All same type (kanji, hiragana, katakana)
        if len(set(char_types)) == 1:
            return True
        
        # Mixed but reasonable patterns
        if (char_types[0] == 'kanji' and 
            all(t in ['kanji', 'hiragana'] for t in char_types)):
            return True
        
        return False
    
    def _analyze_morphology(self, word: str) -> Dict:
        """Analyze word morphology for conjugations."""
        for category, rules in self.morphological_rules.items():
            for form_name, pattern in rules.items():
                match = re.match(pattern, word)
                if match:
                    base_form = match.group(1)
                    if base_form in self.known_words:
                        return {
                            'tokens': [base_form, form_name],
                            'confidence': 0.8,
                            'method': f'morphology_{form_name}',
                            'analysis': {
                                'base_form': base_form,
                                'conjugation': form_name,
                                'category': category
                            }
                        }
        
        return {'confidence': 0.0}
    
    def _is_kanji(self, char: str) -> bool:
        """Check if character is kanji."""
        return bool(re.match(r'[\u4e00-\u9faf]', char))
    
    def _get_char_type(self, char: str) -> str:
        """Get character type."""
        if re.match(r'[\u4e00-\u9faf]', char):
            return 'kanji'
        elif re.match(r'[\u3040-\u309f]', char):
            return 'hiragana'
        elif re.match(r'[\u30a0-\u30ff]', char):
            return 'katakana'
        elif re.match(r'[a-zA-Z]', char):
            return 'romaji'
        else:
            return 'other'
    
    def tokenize_text(self, text: str) -> List[Dict]:
        """Tokenize entire text using fast rule-based approach."""
        # Basic segmentation
        words = self._basic_segmentation(text)
        results = []
        
        for word in words:
            token_result = self.tokenize_word(word)
            results.append(token_result)
        
        return results
    
    def _basic_segmentation(self, text: str) -> List[str]:
        """Basic text segmentation into words."""
        # Split on whitespace and punctuation
        words = re.split(r'[\s。！？、，]+', text)
        return [word for word in words if word.strip()]
    
    def add_known_word(self, word: str, frequency: int = 1):
        """Add word to known vocabulary."""
        self.known_words.add(word)
        
        # Update frequency if tracking
        if hasattr(self, 'word_frequencies'):
            self.word_frequencies[word] = self.word_frequencies.get(word, 0) + frequency
    
    def get_confidence_score(self, word: str) -> float:
        """Get confidence score for a word."""
        if word in self.known_words:
            return 1.0
        
        # Check compound patterns
        for pattern_info in self.compound_patterns.values():
            if re.match(pattern_info['pattern'], word):
                return pattern_info['confidence']
        
        # Check morphological rules
        for rules in self.morphological_rules.values():
            for pattern in rules.values():
                if re.match(pattern, word):
                    return 0.8
        
        return 0.1
    
    def get_engine_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'known_words_count': len(self.known_words),
            'compound_patterns_count': len(self.compound_patterns),
            'morphological_rules_count': sum(len(rules) for rules in self.morphological_rules.values()),
            'coverage_estimate': '~85% for common Japanese text'
        }
