"""
Base Tokenizer for Japanese Text Processing

This module provides the foundation for Japanese text tokenization with:
- Text segmentation (sentence splitting)
- Character type detection (kanji/hiragana/katakana/romaji)
- Particle recognition for boundary detection
- Token boundary inference using semantic coherence
"""

import re
from typing import List, Dict, Tuple, Optional
from enum import Enum


class CharacterType(Enum):
    """Character type enumeration for Japanese text."""
    KANJI = "kanji"
    HIRAGANA = "hiragana"
    KATAKANA = "katakana"
    ROMAJI = "romaji"
    PUNCTUATION = "punctuation"
    NUMBER = "number"
    SYMBOL = "symbol"
    UNKNOWN = "unknown"


class BaseTokenizer:
    """
    Base tokenizer for Japanese text processing.
    
    Provides fundamental text analysis capabilities including:
    - Character type detection
    - Sentence segmentation
    - Particle recognition
    - Basic boundary inference
    """
    
    def __init__(self):
        # Japanese character ranges
        self.kanji_range = re.compile(r'[\u4e00-\u9faf]')
        self.hiragana_range = re.compile(r'[\u3040-\u309f]')
        self.katakana_range = re.compile(r'[\u30a0-\u30ff]')
        self.romaji_range = re.compile(r'[a-zA-Z]')
        self.number_range = re.compile(r'[0-9０-９]')
        
        # Japanese particles for boundary detection
        self.particles = {
            'は', 'が', 'を', 'に', 'で', 'と', 'から', 'まで', 'より', 'へ',
            'の', 'も', 'や', 'か', 'ね', 'よ', 'わ', 'さ', 'な', 'だ',
            'です', 'である', 'だろ', 'だろう', 'ですよ', 'ですね'
        }
        
        # Sentence ending patterns
        self.sentence_endings = {
            '。', '！', '？', '．', '！', '？', '…', '—', '―'
        }
        
        # Common word boundaries
        self.boundary_patterns = [
            r'[。！？]',  # Sentence endings
            r'[、，]',    # Comma-like punctuation
            r'\s+',      # Whitespace
            r'[「」『』（）]',  # Brackets
        ]
    
    def detect_character_type(self, char: str) -> CharacterType:
        """Detect the type of a character."""
        if self.kanji_range.match(char):
            return CharacterType.KANJI
        elif self.hiragana_range.match(char):
            return CharacterType.HIRAGANA
        elif self.katakana_range.match(char):
            return CharacterType.KATAKANA
        elif self.romaji_range.match(char):
            return CharacterType.ROMAJI
        elif self.number_range.match(char):
            return CharacterType.NUMBER
        elif char in '。！？、，（）「」『』':
            return CharacterType.PUNCTUATION
        elif char in '！？…—―':
            return CharacterType.SYMBOL
        else:
            return CharacterType.UNKNOWN
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences."""
        # Split on sentence endings
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in self.sentence_endings:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add remaining text if any
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def is_particle(self, word: str) -> bool:
        """Check if a word is a Japanese particle."""
        return word in self.particles
    
    def find_particle_boundaries(self, text: str) -> List[int]:
        """Find potential word boundaries based on particles."""
        boundaries = []
        
        for i, char in enumerate(text):
            # Check for particles at current position
            for particle in self.particles:
                if text[i:i+len(particle)] == particle:
                    boundaries.append(i)
                    boundaries.append(i + len(particle))
        
        return sorted(set(boundaries))
    
    def analyze_character_sequence(self, text: str) -> List[Dict]:
        """Analyze character sequence and types."""
        analysis = []
        
        for i, char in enumerate(text):
            char_type = self.detect_character_type(char)
            
            # Determine if this is a potential boundary
            is_boundary = self._is_potential_boundary(text, i)
            
            analysis.append({
                'char': char,
                'position': i,
                'type': char_type,
                'is_boundary': is_boundary
            })
        
        return analysis
    
    def _is_potential_boundary(self, text: str, position: int) -> bool:
        """Determine if position is a potential word boundary."""
        if position == 0 or position >= len(text):
            return True
        
        current_char = text[position]
        prev_char = text[position - 1]
        
        # Check for punctuation boundaries
        if current_char in '。！？、，':
            return True
        
        # Check for particle boundaries
        for particle in self.particles:
            if text[position:position+len(particle)] == particle:
                return True
        
        # Check for character type changes
        current_type = self.detect_character_type(current_char)
        prev_type = self.detect_character_type(prev_char)
        
        # Kanji to hiragana/katakana transitions often indicate boundaries
        if (current_type == CharacterType.KANJI and 
            prev_type in [CharacterType.HIRAGANA, CharacterType.KATAKANA]):
            return True
        
        if (prev_type == CharacterType.KANJI and 
            current_type in [CharacterType.HIRAGANA, CharacterType.KATAKANA]):
            return True
        
        return False
    
    def basic_segmentation(self, text: str) -> List[str]:
        """Basic text segmentation into potential words."""
        words = []
        current_word = ""
        
        for i, char in enumerate(text):
            current_word += char
            
            # Check if this is a boundary
            if self._is_potential_boundary(text, i + 1):
                if current_word.strip():
                    words.append(current_word.strip())
                current_word = ""
        
        # Add remaining word
        if current_word.strip():
            words.append(current_word.strip())
        
        return words
    
    def extract_kanji_compounds(self, text: str) -> List[Dict]:
        """Extract potential kanji compounds from text."""
        compounds = []
        current_compound = ""
        start_pos = 0
        
        for i, char in enumerate(text):
            char_type = self.detect_character_type(char)
            
            if char_type == CharacterType.KANJI:
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
    
    def identify_unknown_words(self, words: List[str], known_vocabulary: set) -> List[str]:
        """Identify potentially unknown words."""
        unknown_words = []
        
        for word in words:
            # Check if word contains kanji and is not in known vocabulary
            has_kanji = any(self.detect_character_type(char) == CharacterType.KANJI 
                          for char in word)
            
            if has_kanji and word not in known_vocabulary:
                unknown_words.append(word)
        
        return unknown_words
    
    def get_text_statistics(self, text: str) -> Dict:
        """Get comprehensive text statistics."""
        char_analysis = self.analyze_character_sequence(text)
        
        char_counts = {}
        for analysis in char_analysis:
            char_type = analysis['type']
            char_counts[char_type.value] = char_counts.get(char_type.value, 0) + 1
        
        sentences = self.segment_sentences(text)
        words = self.basic_segmentation(text)
        compounds = self.extract_kanji_compounds(text)
        
        return {
            'total_characters': len(text),
            'character_distribution': char_counts,
            'sentence_count': len(sentences),
            'word_count': len(words),
            'kanji_compound_count': len(compounds),
            'average_sentence_length': len(words) / len(sentences) if sentences else 0,
            'kanji_density': char_counts.get('kanji', 0) / len(text) if text else 0
        }
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[。！？]{2,}', '。', text)
        
        # Normalize brackets
        text = text.replace('（', '(').replace('）', ')')
        text = text.replace('「', '"').replace('」', '"')
        
        return text.strip()
