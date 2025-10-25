"""
Kanji Inference Engine for Unknown Word Analysis

This module provides kanji component-based inference for unknown words:
- Semantic component analysis for unknown kanji compounds
- Probable meaning generation from components
- Context-based meaning filtering
- Confidence calculation based on evidence strength
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import re


class KanjiInferenceEngine:
    """
    Kanji component-based inference engine for unknown word analysis.
    
    Analyzes kanji components to infer meaning of unknown compounds
    using semantic radicals and component relationships.
    """
    
    def __init__(self, kanji_db=None):
        self.kanji_db = kanji_db
        
        # Component semantic mappings
        self.component_semantics = {
            '人': {'meanings': ['person', 'human', 'people'], 'semantic_field': 'human'},
            '大': {'meanings': ['big', 'large', 'great'], 'semantic_field': 'size'},
            '小': {'meanings': ['small', 'little'], 'semantic_field': 'size'},
            '口': {'meanings': ['mouth', 'opening'], 'semantic_field': 'body'},
            '手': {'meanings': ['hand', 'action'], 'semantic_field': 'body'},
            '心': {'meanings': ['heart', 'mind', 'emotion'], 'semantic_field': 'emotion'},
            '水': {'meanings': ['water', 'liquid'], 'semantic_field': 'nature'},
            '火': {'meanings': ['fire', 'heat'], 'semantic_field': 'nature'},
            '木': {'meanings': ['tree', 'wood'], 'semantic_field': 'nature'},
            '金': {'meanings': ['metal', 'gold'], 'semantic_field': 'material'},
            '土': {'meanings': ['earth', 'soil'], 'semantic_field': 'nature'},
            '山': {'meanings': ['mountain', 'hill'], 'semantic_field': 'geography'},
            '川': {'meanings': ['river', 'stream'], 'semantic_field': 'geography'},
            '日': {'meanings': ['sun', 'day'], 'semantic_field': 'time'},
            '月': {'meanings': ['moon', 'month'], 'semantic_field': 'time'},
            '年': {'meanings': ['year', 'age'], 'semantic_field': 'time'},
            '学': {'meanings': ['study', 'learning'], 'semantic_field': 'education'},
            '文': {'meanings': ['writing', 'text'], 'semantic_field': 'language'},
            '言': {'meanings': ['speech', 'word'], 'semantic_field': 'language'},
            '目': {'meanings': ['eye', 'vision'], 'semantic_field': 'body'},
            '耳': {'meanings': ['ear', 'hearing'], 'semantic_field': 'body'},
            '足': {'meanings': ['foot', 'leg'], 'semantic_field': 'body'},
            '車': {'meanings': ['car', 'vehicle'], 'semantic_field': 'transport'},
            '電': {'meanings': ['electricity', 'electric'], 'semantic_field': 'technology'},
            '機': {'meanings': ['machine', 'mechanism'], 'semantic_field': 'technology'},
            '計': {'meanings': ['measure', 'calculate'], 'semantic_field': 'technology'},
            '算': {'meanings': ['calculate', 'compute'], 'semantic_field': 'technology'},
            '制': {'meanings': ['control', 'system'], 'semantic_field': 'technology'},
            '情': {'meanings': ['emotion', 'feeling'], 'semantic_field': 'emotion'},
            '報': {'meanings': ['report', 'news'], 'semantic_field': 'communication'},
            '通': {'meanings': ['communication', 'passage'], 'semantic_field': 'communication'},
            '信': {'meanings': ['trust', 'belief'], 'semantic_field': 'emotion'},
            '息': {'meanings': ['breath', 'rest'], 'semantic_field': 'body'},
            '命': {'meanings': ['life', 'fate'], 'semantic_field': 'existence'},
            '生': {'meanings': ['life', 'birth'], 'semantic_field': 'existence'},
            '死': {'meanings': ['death', 'die'], 'semantic_field': 'existence'},
            '愛': {'meanings': ['love', 'affection'], 'semantic_field': 'emotion'},
            '悲': {'meanings': ['sad', 'sorrow'], 'semantic_field': 'emotion'},
            '喜': {'meanings': ['joy', 'happiness'], 'semantic_field': 'emotion'},
            '怒': {'meanings': ['anger', 'rage'], 'semantic_field': 'emotion'},
            '驚': {'meanings': ['surprise', 'amazement'], 'semantic_field': 'emotion'},
            '恐': {'meanings': ['fear', 'dread'], 'semantic_field': 'emotion'},
            '希': {'meanings': ['hope', 'wish'], 'semantic_field': 'emotion'},
            '望': {'meanings': ['hope', 'desire'], 'semantic_field': 'emotion'},
            '夢': {'meanings': ['dream', 'vision'], 'semantic_field': 'emotion'},
            '想': {'meanings': ['thought', 'idea'], 'semantic_field': 'mind'},
            '思': {'meanings': ['think', 'consider'], 'semantic_field': 'mind'},
            '考': {'meanings': ['think', 'consider'], 'semantic_field': 'mind'},
            '知': {'meanings': ['know', 'knowledge'], 'semantic_field': 'mind'},
            '識': {'meanings': ['knowledge', 'awareness'], 'semantic_field': 'mind'},
            '理': {'meanings': ['reason', 'logic'], 'semantic_field': 'mind'},
            '論': {'meanings': ['theory', 'discussion'], 'semantic_field': 'mind'},
            '分': {'meanings': ['divide', 'understand'], 'semantic_field': 'mind'},
            '析': {'meanings': ['analyze', 'break down'], 'semantic_field': 'mind'},
            '研': {'meanings': ['research', 'study'], 'semantic_field': 'education'},
            '究': {'meanings': ['research', 'investigate'], 'semantic_field': 'education'},
            '実': {'meanings': ['reality', 'truth'], 'semantic_field': 'existence'},
            '験': {'meanings': ['test', 'experiment'], 'semantic_field': 'education'},
            '試': {'meanings': ['try', 'attempt'], 'semantic_field': 'action'},
            '験': {'meanings': ['test', 'trial'], 'semantic_field': 'action'},
            '験': {'meanings': ['test', 'examination'], 'semantic_field': 'education'}
        }
        
        # Semantic field relationships
        self.semantic_relationships = {
            'human': ['body', 'emotion', 'mind'],
            'nature': ['geography', 'time', 'existence'],
            'technology': ['education', 'mind', 'action'],
            'emotion': ['mind', 'human', 'existence'],
            'education': ['mind', 'technology', 'action'],
            'body': ['human', 'emotion', 'existence'],
            'time': ['nature', 'existence'],
            'geography': ['nature', 'existence'],
            'mind': ['education', 'technology', 'emotion'],
            'action': ['technology', 'education', 'human']
        }
    
    def infer_unknown_word(self, word: str, context: List[str] = None) -> Dict:
        """
        Infer meaning of unknown word using kanji component analysis.
        
        Args:
            word: Unknown word to analyze
            context: Optional context words for additional clues
            
        Returns:
            Inference result with probable meanings and confidence
        """
        # Analyze kanji components
        component_analysis = self._analyze_kanji_components(word)
        
        # Generate probable meanings
        probable_meanings = self._generate_meanings_from_components(component_analysis)
        
        # Apply context filtering if available
        if context:
            probable_meanings = self._filter_by_context(probable_meanings, context)
        
        # Calculate confidence
        confidence = self._calculate_inference_confidence(
            word, component_analysis, probable_meanings, context
        )
        
        # Determine semantic field
        semantic_field = self._determine_semantic_field(component_analysis, context)
        
        return {
            'word': word,
            'component_analysis': component_analysis,
            'probable_meanings': probable_meanings,
            'confidence': confidence,
            'semantic_field': semantic_field,
            'context_clues': self._extract_context_clues(context) if context else [],
            'inference_method': 'kanji_component_analysis'
        }
    
    def _analyze_kanji_components(self, word: str) -> Dict:
        """Analyze kanji components for semantic inference."""
        components = []
        component_meanings = []
        semantic_fields = []
        
        for char in word:
            if char in self.component_semantics:
                components.append(char)
                component_info = self.component_semantics[char]
                component_meanings.append(component_info['meanings'])
                semantic_fields.append(component_info['semantic_field'])
            elif self.kanji_db:
                # Try database lookup
                kanji_info = self.kanji_db.get_kanji_info(char)
                if kanji_info:
                    components.append(char)
                    component_meanings.append(kanji_info.get('meanings', []))
                    semantic_fields.append(kanji_info.get('semantic_field', 'unknown'))
        
        return {
            'components': components,
            'component_meanings': component_meanings,
            'semantic_fields': semantic_fields,
            'component_count': len(components),
            'total_chars': len(word)
        }
    
    def _generate_meanings_from_components(self, component_analysis: Dict) -> List[str]:
        """Generate probable meanings from component analysis."""
        meanings = []
        
        components = component_analysis['components']
        component_meanings = component_analysis['component_meanings']
        semantic_fields = component_analysis['semantic_fields']
        
        if not components:
            return ['unknown meaning']
        
        # Direct component meaning combinations
        if len(components) == 1:
            # Single kanji - use primary meaning
            meanings.extend(component_meanings[0][:2])
        elif len(components) == 2:
            # Two kanji compound - combine meanings
            meanings.extend(self._combine_two_kanji_meanings(
                component_meanings[0], component_meanings[1]
            ))
        else:
            # Multiple kanji - use most significant components
            primary_meanings = self._extract_primary_meanings(component_meanings)
            meanings.extend(primary_meanings)
        
        # Semantic field-based meanings
        field_meanings = self._generate_field_based_meanings(semantic_fields)
        meanings.extend(field_meanings)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_meanings = []
        for meaning in meanings:
            if meaning not in seen:
                seen.add(meaning)
                unique_meanings.append(meaning)
        
        return unique_meanings[:5]  # Return top 5 meanings
    
    def _combine_two_kanji_meanings(self, meanings1: List[str], meanings2: List[str]) -> List[str]:
        """Combine meanings from two kanji components."""
        combined = []
        
        # Try different combinations
        for m1 in meanings1[:2]:  # Top 2 meanings from first kanji
            for m2 in meanings2[:2]:  # Top 2 meanings from second kanji
                # Simple concatenation
                combined.append(f"{m1} {m2}")
                
                # More natural combinations
                if m1 in ['big', 'large'] and m2 in ['person', 'people']:
                    combined.append('important person')
                elif m1 in ['new'] and m2 in ['year']:
                    combined.append('new year')
                elif m1 in ['good'] and m2 in ['thing']:
                    combined.append('good thing')
        
        return combined
    
    def _extract_primary_meanings(self, component_meanings: List[List[str]]) -> List[str]:
        """Extract primary meanings from multiple components."""
        primary = []
        
        # Get most common semantic fields
        all_fields = []
        for meanings in component_meanings:
            # This is simplified - in practice, you'd map meanings to fields
            all_fields.extend(meanings[:1])  # Take first meaning from each
        
        # Count frequency and take most common
        field_counts = Counter(all_fields)
        most_common = field_counts.most_common(3)
        
        for field, count in most_common:
            primary.append(field)
        
        return primary
    
    def _generate_field_based_meanings(self, semantic_fields: List[str]) -> List[str]:
        """Generate meanings based on semantic fields."""
        field_meanings = []
        
        # Count field frequency
        field_counts = Counter(semantic_fields)
        dominant_field = field_counts.most_common(1)[0][0] if field_counts else 'unknown'
        
        # Generate field-specific meanings
        if dominant_field == 'technology':
            field_meanings.extend(['technical device', 'system', 'machine'])
        elif dominant_field == 'emotion':
            field_meanings.extend(['emotional state', 'feeling', 'mood'])
        elif dominant_field == 'nature':
            field_meanings.extend(['natural phenomenon', 'environmental', 'natural'])
        elif dominant_field == 'education':
            field_meanings.extend(['educational concept', 'learning', 'study'])
        elif dominant_field == 'human':
            field_meanings.extend(['human-related', 'personal', 'social'])
        
        return field_meanings
    
    def _filter_by_context(self, meanings: List[str], context: List[str]) -> List[str]:
        """Filter meanings based on context clues."""
        if not context:
            return meanings
        
        # Analyze context semantic field
        context_field = self._analyze_context_semantic_field(context)
        
        # Filter meanings that match context
        filtered_meanings = []
        for meaning in meanings:
            if self._meaning_matches_context(meaning, context_field):
                filtered_meanings.append(meaning)
        
        # If no matches, return original meanings
        return filtered_meanings if filtered_meanings else meanings
    
    def _analyze_context_semantic_field(self, context: List[str]) -> str:
        """Analyze semantic field of context."""
        field_scores = defaultdict(int)
        
        for word in context:
            for char in word:
                if char in self.component_semantics:
                    field = self.component_semantics[char]['semantic_field']
                    field_scores[field] += 1
        
        if field_scores:
            return max(field_scores.items(), key=lambda x: x[1])[0]
        return 'unknown'
    
    def _meaning_matches_context(self, meaning: str, context_field: str) -> bool:
        """Check if meaning matches context semantic field."""
        # Simple keyword matching - could be enhanced
        field_keywords = {
            'technology': ['system', 'device', 'machine', 'technical'],
            'emotion': ['feeling', 'emotional', 'mood', 'state'],
            'nature': ['natural', 'environmental', 'phenomenon'],
            'education': ['learning', 'study', 'educational', 'concept'],
            'human': ['person', 'human', 'personal', 'social']
        }
        
        if context_field in field_keywords:
            return any(keyword in meaning.lower() for keyword in field_keywords[context_field])
        
        return True  # Default to include if no specific match
    
    def _calculate_inference_confidence(self, word: str, component_analysis: Dict,
                                      meanings: List[str], context: List[str]) -> float:
        """Calculate confidence in inference result."""
        confidence = 0.0
        
        # Component coverage
        component_ratio = component_analysis['component_count'] / component_analysis['total_chars']
        confidence += component_ratio * 0.4
        
        # Meaning diversity
        if len(meanings) > 1:
            confidence += 0.2
        
        # Context support
        if context and len(context) > 2:
            confidence += 0.2
        
        # Semantic field consistency
        semantic_fields = component_analysis['semantic_fields']
        if semantic_fields:
            field_consistency = len(set(semantic_fields)) / len(semantic_fields)
            confidence += field_consistency * 0.2
        
        return min(confidence, 1.0)
    
    def _determine_semantic_field(self, component_analysis: Dict, context: List[str]) -> str:
        """Determine semantic field of the word."""
        semantic_fields = component_analysis['semantic_fields']
        
        if semantic_fields:
            # Use most common field from components
            field_counts = Counter(semantic_fields)
            return field_counts.most_common(1)[0][0]
        
        # Fallback to context analysis
        if context:
            return self._analyze_context_semantic_field(context)
        
        return 'unknown'
    
    def _extract_context_clues(self, context: List[str]) -> List[str]:
        """Extract context clues for meaning inference."""
        clues = []
        
        if not context:
            return clues
        
        # Look for semantic indicators
        for word in context:
            for char in word:
                if char in self.component_semantics:
                    field = self.component_semantics[char]['semantic_field']
                    if field not in clues:
                        clues.append(field)
        
        return clues
    
    def get_engine_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'component_semantics_count': len(self.component_semantics),
            'semantic_relationships_count': len(self.semantic_relationships),
            'analysis_capabilities': [
                'kanji component analysis',
                'semantic field identification',
                'context-based meaning filtering',
                'confidence calculation'
            ]
        }
