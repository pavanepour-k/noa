"""
Database Validation Script

Validates the integrity and quality of the kanji database.
Checks data completeness, consistency, and performance.
"""

import json
import os
import sys
import time
from typing import Dict, List, Any, Tuple
from collections import Counter

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from database.kanji_db import OptimizedKanjiDatabase
    from utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)


class DatabaseValidator:
    """Validates kanji database integrity and quality."""
    
    def __init__(self, db_path: str = "data/kanji_database.json"):
        self.db_path = db_path
        self.logger = get_logger('database_validator')
        self.db = None
        self.validation_results = {
            'overall_score': 0.0,
            'checks_passed': 0,
            'checks_failed': 0,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
    
    def validate_database(self) -> Dict[str, Any]:
        """Run comprehensive database validation."""
        self.logger.info("Starting comprehensive database validation")
        
        try:
            # Load database
            self.db = OptimizedKanjiDatabase(self.db_path)
            
            # Run validation checks
            self._check_metadata()
            self._check_kanji_data_quality()
            self._check_radicals_data()
            self._check_component_mapping()
            self._check_frequency_rankings()
            self._check_data_consistency()
            self._check_performance()
            self._check_coverage()
            
            # Calculate overall score
            self._calculate_overall_score()
            
            self.logger.info(f"Validation complete: {self.validation_results['overall_score']:.1f}/100")
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.validation_results['errors'].append(f"Validation failed: {e}")
        
        return self.validation_results
    
    def _check_metadata(self):
        """Check metadata completeness and validity."""
        self.logger.info("Checking metadata")
        
        metadata = self.db.metadata
        required_fields = ['version', 'total_kanji', 'source']
        
        for field in required_fields:
            if field not in metadata:
                self.validation_results['errors'].append(f"Missing metadata field: {field}")
                self.validation_results['checks_failed'] += 1
            else:
                self.validation_results['checks_passed'] += 1
        
        # Check version format
        if 'version' in metadata:
            version = metadata['version']
            if not isinstance(version, str) or not version:
                self.validation_results['warnings'].append("Invalid version format")
        
        # Check total_kanji consistency
        if 'total_kanji' in metadata:
            declared_total = metadata['total_kanji']
            actual_total = len(self.db.kanji_data)
            if declared_total != actual_total:
                self.validation_results['warnings'].append(
                    f"Metadata total_kanji ({declared_total}) doesn't match actual count ({actual_total})"
                )
    
    def _check_kanji_data_quality(self):
        """Check kanji data quality and completeness."""
        self.logger.info("Checking kanji data quality")
        
        required_fields = ['radical', 'meanings', 'frequency_rank', 'components', 'semantic_field']
        optional_fields = ['readings', 'common_compounds', 'strokes', 'grade', 'jlpt_level']
        
        quality_issues = 0
        total_kanji = len(self.db.kanji_data)
        
        for kanji, data in self.db.kanji_data.items():
            # Check required fields
            for field in required_fields:
                if field not in data:
                    self.validation_results['warnings'].append(f"Missing required field '{field}' in kanji '{kanji}'")
                    quality_issues += 1
                elif not data[field]:  # Check if field is empty
                    self.validation_results['warnings'].append(f"Empty field '{field}' in kanji '{kanji}'")
                    quality_issues += 1
            
            # Check data types
            if 'meanings' in data and not isinstance(data['meanings'], list):
                self.validation_results['errors'].append(f"Invalid meanings type in kanji '{kanji}'")
                quality_issues += 1
            
            if 'frequency_rank' in data and not isinstance(data['frequency_rank'], int):
                self.validation_results['errors'].append(f"Invalid frequency_rank type in kanji '{kanji}'")
                quality_issues += 1
        
        # Calculate quality score
        quality_score = max(0, 100 - (quality_issues / total_kanji * 100))
        if quality_score >= 90:
            self.validation_results['checks_passed'] += 1
        else:
            self.validation_results['checks_failed'] += 1
            self.validation_results['warnings'].append(f"Kanji data quality score: {quality_score:.1f}/100")
    
    def _check_radicals_data(self):
        """Check radicals data completeness."""
        self.logger.info("Checking radicals data")
        
        radicals = self.db.radicals_data
        if not radicals:
            self.validation_results['errors'].append("No radicals data found")
            self.validation_results['checks_failed'] += 1
            return
        
        # Check for required fields
        required_fields = ['meaning']
        for radical, data in radicals.items():
            for field in required_fields:
                if field not in data:
                    self.validation_results['warnings'].append(f"Missing field '{field}' in radical '{radical}'")
        
        # Check if radicals are referenced in kanji data
        used_radicals = set()
        for kanji_data in self.db.kanji_data.values():
            if 'radical' in kanji_data:
                used_radicals.add(kanji_data['radical'])
        
        unused_radicals = set(radicals.keys()) - used_radicals
        if unused_radicals:
            self.validation_results['warnings'].append(f"Unused radicals: {list(unused_radicals)[:5]}")
        
        self.validation_results['checks_passed'] += 1
    
    def _check_component_mapping(self):
        """Check component mapping consistency."""
        self.logger.info("Checking component mapping")
        
        mapping = self.db.component_mapping
        if not mapping:
            self.validation_results['warnings'].append("No component mapping found")
            return
        
        # Check if mapped radicals exist
        for component, radical in mapping.items():
            if radical not in self.db.radicals_data:
                self.validation_results['warnings'].append(f"Component '{component}' maps to unknown radical '{radical}'")
        
        self.validation_results['checks_passed'] += 1
    
    def _check_frequency_rankings(self):
        """Check frequency rankings consistency."""
        self.logger.info("Checking frequency rankings")
        
        rankings = self.db.frequency_rankings
        if not rankings:
            self.validation_results['warnings'].append("No frequency rankings found")
            return
        
        # Check if rankings match kanji data
        ranking_kanji = set(rankings.values())
        kanji_data_kanji = set(self.db.kanji_data.keys())
        
        if ranking_kanji != kanji_data_kanji:
            missing_in_rankings = kanji_data_kanji - ranking_kanji
            missing_in_data = ranking_kanji - kanji_data_kanji
            
            if missing_in_rankings:
                self.validation_results['warnings'].append(f"Kanji missing from rankings: {list(missing_in_rankings)[:5]}")
            if missing_in_data:
                self.validation_results['warnings'].append(f"Rankings reference missing kanji: {list(missing_in_data)[:5]}")
        
        # Check ranking order
        ranks = []
        for rank_str, kanji in rankings.items():
            try:
                rank = int(rank_str)
                ranks.append((rank, kanji))
            except ValueError:
                self.validation_results['warnings'].append(f"Invalid rank format: {rank_str}")
        
        ranks.sort()
        for i in range(len(ranks) - 1):
            if ranks[i][0] >= ranks[i + 1][0]:
                self.validation_results['warnings'].append("Frequency rankings not properly ordered")
                break
        
        self.validation_results['checks_passed'] += 1
    
    def _check_data_consistency(self):
        """Check data consistency across sections."""
        self.logger.info("Checking data consistency")
        
        # Check semantic field consistency
        semantic_fields = set()
        for kanji_data in self.db.kanji_data.values():
            if 'semantic_field' in kanji_data:
                semantic_fields.add(kanji_data['semantic_field'])
        
        if len(semantic_fields) < 3:
            self.validation_results['warnings'].append(f"Limited semantic field diversity: {len(semantic_fields)} fields")
        
        # Check frequency rank distribution
        ranks = [data.get('frequency_rank', 9999) for data in self.db.kanji_data.values()]
        if ranks:
            min_rank = min(ranks)
            max_rank = max(ranks)
            if max_rank - min_rank < 10:
                self.validation_results['warnings'].append("Limited frequency rank distribution")
        
        self.validation_results['checks_passed'] += 1
    
    def _check_performance(self):
        """Check database performance."""
        self.logger.info("Checking database performance")
        
        # Test cache performance
        test_kanji = list(self.db.kanji_data.keys())[:10]
        
        # First access (cache miss)
        start_time = time.time()
        for kanji in test_kanji:
            self.db.get_kanji_info(kanji)
        first_access_time = time.time() - start_time
        
        # Second access (cache hit)
        start_time = time.time()
        for kanji in test_kanji:
            self.db.get_kanji_info(kanji)
        second_access_time = time.time() - start_time
        
        # Check cache hit rate
        hit_rate = self.db.get_cache_hit_rate()
        if hit_rate < 0.5:
            self.validation_results['warnings'].append(f"Low cache hit rate: {hit_rate:.2f}")
        
        # Check performance improvement
        if second_access_time > 0 and first_access_time > second_access_time:
            speedup = first_access_time / second_access_time
            if speedup < 2:
                self.validation_results['warnings'].append(f"Limited cache performance improvement: {speedup:.1f}x")
        
        self.validation_results['checks_passed'] += 1
    
    def _check_coverage(self):
        """Check database coverage and completeness."""
        self.logger.info("Checking database coverage")
        
        total_kanji = len(self.db.kanji_data)
        
        # Check if we have enough kanji
        if total_kanji < 100:
            self.validation_results['warnings'].append(f"Low kanji count: {total_kanji}")
        elif total_kanji >= 1000:
            self.validation_results['checks_passed'] += 1
        
        # Check semantic field coverage
        semantic_fields = set()
        for kanji_data in self.db.kanji_data.values():
            if 'semantic_field' in kanji_data:
                semantic_fields.add(kanji_data['semantic_field'])
        
        if len(semantic_fields) >= 5:
            self.validation_results['checks_passed'] += 1
        else:
            self.validation_results['warnings'].append(f"Limited semantic field coverage: {len(semantic_fields)} fields")
        
        # Check frequency distribution
        ranks = [data.get('frequency_rank', 9999) for data in self.db.kanji_data.values()]
        if ranks:
            rank_distribution = Counter(ranks)
            if len(rank_distribution) >= 10:
                self.validation_results['checks_passed'] += 1
            else:
                self.validation_results['warnings'].append("Limited frequency rank distribution")
    
    def _calculate_overall_score(self):
        """Calculate overall validation score."""
        total_checks = self.validation_results['checks_passed'] + self.validation_results['checks_failed']
        
        if total_checks == 0:
            self.validation_results['overall_score'] = 0.0
        else:
            base_score = (self.validation_results['checks_passed'] / total_checks) * 100
            
            # Deduct points for errors and warnings
            error_penalty = len(self.validation_results['errors']) * 5
            warning_penalty = len(self.validation_results['warnings']) * 2
            
            self.validation_results['overall_score'] = max(0, base_score - error_penalty - warning_penalty)
        
        # Generate recommendations
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generate improvement recommendations."""
        recommendations = []
        
        if self.validation_results['overall_score'] < 70:
            recommendations.append("Database needs significant improvements")
        
        if len(self.validation_results['errors']) > 0:
            recommendations.append("Fix critical errors before deployment")
        
        if len(self.validation_results['warnings']) > 5:
            recommendations.append("Address warnings to improve data quality")
        
        # Check specific issues
        if any("Missing required field" in warning for warning in self.validation_results['warnings']):
            recommendations.append("Complete missing required fields in kanji data")
        
        if any("Low cache hit rate" in warning for warning in self.validation_results['warnings']):
            recommendations.append("Optimize caching strategy")
        
        if any("Limited semantic field coverage" in warning for warning in self.validation_results['warnings']):
            recommendations.append("Add more diverse semantic fields")
        
        self.validation_results['recommendations'] = recommendations
    
    def print_validation_report(self):
        """Print detailed validation report."""
        print("\n" + "="*60)
        print("KANJI DATABASE VALIDATION REPORT")
        print("="*60)
        
        print(f"\nOverall Score: {self.validation_results['overall_score']:.1f}/100")
        print(f"Checks Passed: {self.validation_results['checks_passed']}")
        print(f"Checks Failed: {self.validation_results['checks_failed']}")
        
        if self.validation_results['errors']:
            print(f"\n❌ ERRORS ({len(self.validation_results['errors']}):")
            for error in self.validation_results['errors']:
                print(f"  • {error}")
        
        if self.validation_results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results['warnings']:
                print(f"  • {warning}")
        
        if self.validation_results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in self.validation_results['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "="*60)


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate kanji database")
    parser.add_argument("--db-path", default="data/kanji_database.json", 
                       help="Path to kanji database file")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Run validation
    validator = DatabaseValidator(args.db_path)
    results = validator.validate_database()
    
    # Print report
    validator.print_validation_report()
    
    # Exit with appropriate code
    if results['overall_score'] >= 80:
        print("\n✅ Database validation PASSED")
        sys.exit(0)
    elif results['overall_score'] >= 60:
        print("\n⚠️  Database validation PASSED with warnings")
        sys.exit(0)
    else:
        print("\n❌ Database validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
