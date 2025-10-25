"""
Preprocessing Pipeline for Japanese Text

This module provides text cleaning and normalization:
- Unicode normalization
- HTML/markup removal
- Deduplication
- Quality filtering
"""

import re
import unicodedata
from typing import List, Dict, Optional, Set
from collections import Counter
import logging


class JapaneseTextPreprocessor:
    """
    Preprocessing pipeline for Japanese text data.
    
    Handles cleaning, normalization, and quality filtering
    of Japanese text for tokenizer training.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Japanese-specific patterns
        self.japanese_patterns = {
            'hiragana': re.compile(r'[\u3040-\u309f]+'),
            'katakana': re.compile(r'[\u30a0-\u30ff]+'),
            'kanji': re.compile(r'[\u4e00-\u9faf]+'),
            'romaji': re.compile(r'[a-zA-Z]+'),
            'numbers': re.compile(r'[0-9０-９]+'),
            'punctuation': re.compile(r'[。！？、，；：""''（）【】「」『』〈〉《》]'),
            'whitespace': re.compile(r'\s+'),
            'line_breaks': re.compile(r'[\n\r]+')
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_length': 50,
            'max_length': 10000,
            'min_kanji_ratio': 0.1,
            'max_romaji_ratio': 0.3,
            'min_hiragana_ratio': 0.05
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Unicode normalization
        text = self._normalize_unicode(text)
        
        # Remove HTML/markup
        text = self._remove_html_markup(text)
        
        # Clean whitespace
        text = self._clean_whitespace(text)
        
        # Remove excessive punctuation
        text = self._clean_punctuation(text)
        
        # Normalize Japanese characters
        text = self._normalize_japanese_characters(text)
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace full-width characters with half-width
        text = text.replace('（', '(').replace('）', ')')
        text = text.replace('「', '"').replace('」', '"')
        text = text.replace('『', '"').replace('』', '"')
        
        return text
    
    def _remove_html_markup(self, text: str) -> str:
        """Remove HTML tags and markup."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove common markup patterns
        markup_patterns = [
            r'\[.*?\]',  # Square brackets
            r'\{.*?\}',  # Curly braces
            r'\(.*?\)',  # Parentheses (but keep Japanese ones)
            r'【.*?】',   # Japanese brackets
        ]
        
        for pattern in markup_patterns:
            text = re.sub(pattern, '', text)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean excessive punctuation."""
        # Remove excessive punctuation
        text = re.sub(r'[。！？]{2,}', '。', text)
        text = re.sub(r'[、，]{2,}', '、', text)
        text = re.sub(r'[！？]{2,}', '！', text)
        
        # Remove excessive dots
        text = re.sub(r'\.{3,}', '...', text)
        
        return text
    
    def _normalize_japanese_characters(self, text: str) -> str:
        """Normalize Japanese character variants."""
        # Normalize katakana variants
        text = text.replace('ヰ', 'イ').replace('ヱ', 'エ')
        
        # Normalize hiragana variants
        text = text.replace('ゐ', 'い').replace('ゑ', 'え')
        
        # Normalize punctuation
        text = text.replace('〜', '～')
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of preprocessed texts
        """
        preprocessed = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                self.logger.info(f"Preprocessing text {i}/{len(texts)}")
            
            processed = self.preprocess_text(text)
            if processed:
                preprocessed.append(processed)
        
        return preprocessed
    
    def filter_by_quality(self, texts: List[str]) -> List[str]:
        """
        Filter texts by quality criteria.
        
        Args:
            texts: List of texts to filter
            
        Returns:
            List of high-quality texts
        """
        filtered = []
        
        for text in texts:
            if self._is_high_quality(text):
                filtered.append(text)
        
        self.logger.info(f"Filtered {len(filtered)}/{len(texts)} texts by quality")
        return filtered
    
    def _is_high_quality(self, text: str) -> bool:
        """Check if text meets quality criteria."""
        # Length check
        if len(text) < self.quality_thresholds['min_length']:
            return False
        
        if len(text) > self.quality_thresholds['max_length']:
            return False
        
        # Character composition analysis
        char_stats = self._analyze_character_composition(text)
        
        # Kanji ratio check
        if char_stats['kanji_ratio'] < self.quality_thresholds['min_kanji_ratio']:
            return False
        
        # Romaji ratio check
        if char_stats['romaji_ratio'] > self.quality_thresholds['max_romaji_ratio']:
            return False
        
        # Hiragana ratio check
        if char_stats['hiragana_ratio'] < self.quality_thresholds['min_hiragana_ratio']:
            return False
        
        # Check for repetitive patterns
        if self._has_repetitive_patterns(text):
            return False
        
        # Check for meaningful content
        if not self._has_meaningful_content(text):
            return False
        
        return True
    
    def _analyze_character_composition(self, text: str) -> Dict[str, float]:
        """Analyze character composition of text."""
        total_chars = len(text)
        if total_chars == 0:
            return {
                'kanji_ratio': 0.0,
                'hiragana_ratio': 0.0,
                'katakana_ratio': 0.0,
                'romaji_ratio': 0.0,
                'number_ratio': 0.0,
                'punctuation_ratio': 0.0
            }
        
        kanji_count = len(self.japanese_patterns['kanji'].findall(text))
        hiragana_count = len(self.japanese_patterns['hiragana'].findall(text))
        katakana_count = len(self.japanese_patterns['katakana'].findall(text))
        romaji_count = len(self.japanese_patterns['romaji'].findall(text))
        number_count = len(self.japanese_patterns['numbers'].findall(text))
        punctuation_count = len(self.japanese_patterns['punctuation'].findall(text))
        
        return {
            'kanji_ratio': kanji_count / total_chars,
            'hiragana_ratio': hiragana_count / total_chars,
            'katakana_ratio': katakana_count / total_chars,
            'romaji_ratio': romaji_count / total_chars,
            'number_ratio': number_count / total_chars,
            'punctuation_ratio': punctuation_count / total_chars
        }
    
    def _has_repetitive_patterns(self, text: str) -> bool:
        """Check for repetitive patterns that indicate low quality."""
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
    
    def _has_meaningful_content(self, text: str) -> bool:
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
    
    def deduplicate_texts(self, texts: List[str]) -> List[str]:
        """
        Remove duplicate texts.
        
        Args:
            texts: List of texts to deduplicate
            
        Returns:
            List of unique texts
        """
        # Simple deduplication based on exact match
        unique_texts = list(set(texts))
        
        # More sophisticated deduplication based on similarity
        final_texts = []
        for text in unique_texts:
            if not self._is_similar_to_existing(text, final_texts):
                final_texts.append(text)
        
        self.logger.info(f"Deduplicated {len(texts)} -> {len(final_texts)} texts")
        return final_texts
    
    def _is_similar_to_existing(self, text: str, existing_texts: List[str]) -> bool:
        """Check if text is similar to existing texts."""
        # Simple similarity check based on common words
        text_words = set(text.split())
        
        for existing_text in existing_texts:
            existing_words = set(existing_text.split())
            
            # Calculate Jaccard similarity
            intersection = len(text_words & existing_words)
            union = len(text_words | existing_words)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.8:  # 80% similarity threshold
                    return True
        
        return False
    
    def segment_into_sentences(self, text: str) -> List[str]:
        """Segment text into sentences."""
        # Japanese sentence endings
        sentence_endings = r'[。！？]'
        
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def get_text_statistics(self, texts: List[str]) -> Dict:
        """Get comprehensive statistics about texts."""
        if not texts:
            return {}
        
        total_texts = len(texts)
        total_chars = sum(len(text) for text in texts)
        total_words = sum(len(text.split()) for text in texts)
        
        # Character composition analysis
        all_text = ' '.join(texts)
        char_stats = self._analyze_character_composition(all_text)
        
        # Length statistics
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        return {
            'total_texts': total_texts,
            'total_characters': total_chars,
            'total_words': total_words,
            'average_length': total_chars / total_texts,
            'average_words': total_words / total_texts,
            'character_composition': char_stats,
            'length_distribution': {
                'min_length': min(lengths),
                'max_length': max(lengths),
                'median_length': sorted(lengths)[len(lengths) // 2]
            },
            'word_distribution': {
                'min_words': min(word_counts),
                'max_words': max(word_counts),
                'median_words': sorted(word_counts)[len(word_counts) // 2]
            }
        }
    
    def export_processed_data(self, texts: List[str], output_path: str) -> None:
        """Export processed texts to file."""
        import json
        
        data = {
            'texts': texts,
            'count': len(texts),
            'total_characters': sum(len(text) for text in texts),
            'total_words': sum(len(text.split()) for text in texts),
            'processed_at': time.time()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Exported {len(texts)} processed texts to {output_path}")
    
    def create_training_corpus(self, texts: List[str], output_path: str) -> None:
        """Create training corpus from processed texts."""
        # Preprocess texts
        processed_texts = self.preprocess_batch(texts)
        
        # Filter by quality
        quality_texts = self.filter_by_quality(processed_texts)
        
        # Deduplicate
        unique_texts = self.deduplicate_texts(quality_texts)
        
        # Export final corpus
        self.export_processed_data(unique_texts, output_path)
        
        # Log statistics
        stats = self.get_text_statistics(unique_texts)
        self.logger.info(f"Created training corpus with {stats['total_texts']} texts")
        self.logger.info(f"Total characters: {stats['total_characters']:,}")
        self.logger.info(f"Total words: {stats['total_words']:,}")
