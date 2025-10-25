"""
Utility Functions for Japanese Text Processing

This module provides utility functions for text processing,
character analysis, and helper functions used throughout the tokenizer.
"""

import re
import unicodedata
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter


class JapaneseTextUtils:
    """Utility class for Japanese text processing."""
    
    # Japanese character ranges
    KANJI_RANGE = re.compile(r'[\u4e00-\u9faf]')
    HIRAGANA_RANGE = re.compile(r'[\u3040-\u309f]')
    KATAKANA_RANGE = re.compile(r'[\u30a0-\u30ff]')
    ROMAJI_RANGE = re.compile(r'[a-zA-Z]')
    NUMBER_RANGE = re.compile(r'[0-9０-９]')
    
    # Common Japanese particles
    PARTICLES = {
        'は', 'が', 'を', 'に', 'で', 'と', 'から', 'まで', 'より', 'へ',
        'の', 'も', 'や', 'か', 'ね', 'よ', 'わ', 'さ', 'な', 'だ',
        'です', 'である', 'だろ', 'だろう', 'ですよ', 'ですね'
    }
    
    # Sentence endings
    SENTENCE_ENDINGS = {'。', '！', '？', '．', '！', '？', '…', '—', '―'}
    
    @staticmethod
    def is_kanji(char: str) -> bool:
        """Check if character is kanji."""
        return bool(JapaneseTextUtils.KANJI_RANGE.match(char))
    
    @staticmethod
    def is_hiragana(char: str) -> bool:
        """Check if character is hiragana."""
        return bool(JapaneseTextUtils.HIRAGANA_RANGE.match(char))
    
    @staticmethod
    def is_katakana(char: str) -> bool:
        """Check if character is katakana."""
        return bool(JapaneseTextUtils.KATAKANA_RANGE.match(char))
    
    @staticmethod
    def is_romaji(char: str) -> bool:
        """Check if character is romaji."""
        return bool(JapaneseTextUtils.ROMAJI_RANGE.match(char))
    
    @staticmethod
    def is_number(char: str) -> bool:
        """Check if character is number."""
        return bool(JapaneseTextUtils.NUMBER_RANGE.match(char))
    
    @staticmethod
    def get_character_type(char: str) -> str:
        """Get character type."""
        if JapaneseTextUtils.is_kanji(char):
            return 'kanji'
        elif JapaneseTextUtils.is_hiragana(char):
            return 'hiragana'
        elif JapaneseTextUtils.is_katakana(char):
            return 'katakana'
        elif JapaneseTextUtils.is_romaji(char):
            return 'romaji'
        elif JapaneseTextUtils.is_number(char):
            return 'number'
        else:
            return 'other'
    
    @staticmethod
    def analyze_character_composition(text: str) -> Dict[str, int]:
        """Analyze character composition of text."""
        composition = {
            'kanji': 0,
            'hiragana': 0,
            'katakana': 0,
            'romaji': 0,
            'number': 0,
            'other': 0,
            'total': len(text)
        }
        
        for char in text:
            char_type = JapaneseTextUtils.get_character_type(char)
            composition[char_type] += 1
        
        return composition
    
    @staticmethod
    def is_particle(word: str) -> bool:
        """Check if word is a Japanese particle."""
        return word in JapaneseTextUtils.PARTICLES
    
    @staticmethod
    def is_sentence_ending(char: str) -> bool:
        """Check if character is sentence ending."""
        return char in JapaneseTextUtils.SENTENCE_ENDINGS
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode text."""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace full-width characters with half-width
        text = text.replace('（', '(').replace('）', ')')
        text = text.replace('「', '"').replace('」', '"')
        text = text.replace('『', '"').replace('』', '"')
        
        return text
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Clean and normalize whitespace."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text
    
    @staticmethod
    def extract_kanji_compounds(text: str) -> List[Dict]:
        """Extract potential kanji compounds from text."""
        compounds = []
        current_compound = ""
        start_pos = 0
        
        for i, char in enumerate(text):
            if JapaneseTextUtils.is_kanji(char):
                if not current_compound:
                    start_pos = i
                current_compound += char
            else:
                if current_compound:
                    compounds.append({
                        'compound': current_compound,
                        'start': start_pos,
                        'end': i,
                        'length': len(current_compound)
                    })
                    current_compound = ""
        
        # Add final compound if exists
        if current_compound:
            compounds.append({
                'compound': current_compound,
                'start': start_pos,
                'end': len(text),
                'length': len(current_compound)
            })
        
        return compounds
    
    @staticmethod
    def segment_sentences(text: str) -> List[str]:
        """Segment text into sentences."""
        # Japanese sentence endings
        sentence_endings = r'[。！？]'
        
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    @staticmethod
    def calculate_text_quality(text: str) -> Dict[str, float]:
        """Calculate text quality metrics."""
        if not text:
            return {'quality_score': 0.0, 'issues': []}
        
        issues = []
        quality_score = 1.0
        
        # Length check
        if len(text) < 50:
            issues.append('too_short')
            quality_score -= 0.3
        
        # Character composition analysis
        composition = JapaneseTextUtils.analyze_character_composition(text)
        
        # Kanji ratio
        kanji_ratio = composition['kanji'] / composition['total']
        if kanji_ratio < 0.1:
            issues.append('low_kanji_ratio')
            quality_score -= 0.2
        
        # Romaji ratio
        romaji_ratio = composition['romaji'] / composition['total']
        if romaji_ratio > 0.3:
            issues.append('high_romaji_ratio')
            quality_score -= 0.2
        
        # Hiragana ratio
        hiragana_ratio = composition['hiragana'] / composition['total']
        if hiragana_ratio < 0.05:
            issues.append('low_hiragana_ratio')
            quality_score -= 0.1
        
        # Check for repetitive patterns
        if JapaneseTextUtils._has_repetitive_patterns(text):
            issues.append('repetitive_patterns')
            quality_score -= 0.2
        
        # Check for meaningful content
        if not JapaneseTextUtils._has_meaningful_content(text):
            issues.append('low_meaningful_content')
            quality_score -= 0.3
        
        return {
            'quality_score': max(quality_score, 0.0),
            'issues': issues,
            'composition': composition
        }
    
    @staticmethod
    def _has_repetitive_patterns(text: str) -> bool:
        """Check for repetitive patterns."""
        # Check for excessive repetition of single characters
        char_counts = Counter(text)
        max_char_count = max(char_counts.values())
        if max_char_count > len(text) * 0.3:  # 30% threshold
            return True
        
        # Check for repetitive word patterns
        words = text.split()
        if len(words) > 10:
            word_counts = Counter(words)
            max_word_count = max(word_counts.values())
            if max_word_count > len(words) * 0.2:  # 20% threshold
                return True
        
        return False
    
    @staticmethod
    def _has_meaningful_content(text: str) -> bool:
        """Check if text has meaningful content."""
        # Check for minimum word count
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for sentence structure
        sentences = re.split(r'[。！？]', text)
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 10]
        if len(meaningful_sentences) < 2:
            return False
        
        # Check for variety in characters
        unique_chars = len(set(text))
        if unique_chars < 20:  # Too few unique characters
            return False
        
        return True
    
    @staticmethod
    def find_word_boundaries(text: str) -> List[int]:
        """Find potential word boundaries in text."""
        boundaries = []
        
        for i, char in enumerate(text):
            # Check for particles
            for particle in JapaneseTextUtils.PARTICLES:
                if text[i:i+len(particle)] == particle:
                    boundaries.append(i)
                    boundaries.append(i + len(particle))
            
            # Check for character type changes
            if i > 0:
                current_type = JapaneseTextUtils.get_character_type(char)
                prev_type = JapaneseTextUtils.get_character_type(text[i-1])
                
                # Kanji to hiragana/katakana transitions
                if (current_type == 'kanji' and 
                    prev_type in ['hiragana', 'katakana']):
                    boundaries.append(i)
                
                if (prev_type == 'kanji' and 
                    current_type in ['hiragana', 'katakana']):
                    boundaries.append(i)
        
        return sorted(set(boundaries))
    
    @staticmethod
    def extract_context_words(text: str, target_word: str, window_size: int = 5) -> List[str]:
        """Extract context words around target word."""
        words = text.split()
        
        try:
            target_index = words.index(target_word)
        except ValueError:
            return []
        
        start = max(0, target_index - window_size)
        end = min(len(words), target_index + window_size + 1)
        
        context_words = words[start:end]
        # Remove target word from context
        context_words = [w for w in context_words if w != target_word]
        
        return context_words
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def get_text_statistics(text: str) -> Dict:
        """Get comprehensive text statistics."""
        composition = JapaneseTextUtils.analyze_character_composition(text)
        sentences = JapaneseTextUtils.segment_sentences(text)
        words = text.split()
        compounds = JapaneseTextUtils.extract_kanji_compounds(text)
        
        return {
            'total_characters': composition['total'],
            'character_composition': composition,
            'sentence_count': len(sentences),
            'word_count': len(words),
            'kanji_compound_count': len(compounds),
            'average_sentence_length': len(words) / len(sentences) if sentences else 0,
            'kanji_density': composition['kanji'] / composition['total'] if composition['total'] > 0 else 0
        }
