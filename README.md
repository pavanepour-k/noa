# Japanese Custom Tokenizer

A novel Japanese tokenizer implementing human-like understanding through kanji semantic analysis, hybrid inference engines, and adaptive learning.

## Features

### 🧠 Human-like Understanding
- **Kanji Component Analysis**: Infers unknown words using semantic radicals and component relationships
- **Context Semantic Engine**: Analyzes semantic fields and context windows for better tokenization
- **Hybrid Inference**: Combines fast rule-based and context-aware engines with intelligent selection

### ⚡ Performance Optimized
- **Hierarchical Caching**: Three-tier caching system (L1 LRU, L2 hot data, L3 full DB)
- **Fast Rule-based Engine**: Dictionary lookup and pattern matching for known words
- **Adaptive Learning**: Continuous improvement through pattern generalization

### 🔧 Advanced Capabilities
- **Unknown Word Inference**: Component-based meaning inference for unseen compounds
- **Semantic Field Analysis**: Technology, nature, emotion, academic domain recognition
- **Quality Filtering**: Comprehensive text preprocessing and quality assessment

## Architecture

```
src/
├── tokenizer/
│   ├── hybrid_tokenizer.py      # Main hybrid inference system
│   ├── base_tokenizer.py        # Basic text segmentation
│   ├── fast_engine.py           # Fast rule-based engine
│   ├── context_engine.py        # Context semantic engine
│   ├── kanji_inference.py       # Unknown word inference
│   └── adaptive_learning.py      # Pattern learning module
├── database/
│   ├── kanji_db.py              # Kanji database (2,000 characters)
│   └── cache_system.py         # Hierarchical caching system
└── utils/
    └── text_utils.py            # Utility functions

tools/
└── data_collection/
    ├── corpus_collector.py      # Data collection tools
    └── preprocessor.py          # Text preprocessing pipeline

test/
├── evaluation.py                # Comprehensive evaluation framework
└── test_tokenizer.py           # Unit and integration tests
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd noa-1

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.tokenizer.hybrid_tokenizer import HybridTokenizer
from src.database.kanji_db import KanjiDatabase
from src.database.cache_system import HierarchicalCacheSystem

# Initialize components
kanji_db = KanjiDatabase()
cache_system = HierarchicalCacheSystem()
cache_system.set_l3_database(kanji_db)

# Create tokenizer
tokenizer = HybridTokenizer(kanji_db=kanji_db, cache_system=cache_system)

# Tokenize text
text = "機械学習の研究をしています。"
result = tokenizer.tokenize(text)

# Print results
for token_result in result:
    print(f"Word: {token_result['word']}")
    print(f"Tokens: {token_result['tokens']}")
    print(f"Confidence: {token_result['confidence']:.2f}")
    print(f"Engine: {token_result.get('engine_used', 'unknown')}")
    print("---")
```

### Advanced Usage

```python
# Unknown word inference
unknown_text = "この推測計は正確だ"
result = tokenizer.tokenize(unknown_text)

# Check for unknown word inference
for token_result in result:
    if token_result.get('method') == 'context_inference':
        print(f"Unknown word: {token_result['word']}")
        print(f"Inferred meaning: {token_result.get('inferred_meaning', [])}")
        print(f"Confidence: {token_result['confidence']:.2f}")

# Performance monitoring
stats = tokenizer.get_performance_stats()
print(f"Fast engine usage: {stats['engine_usage']['fast_engine_rate']:.2%}")
print(f"Hybrid engine usage: {stats['engine_usage']['hybrid_engine_rate']:.2%}")
print(f"Average confidence: {stats['tokenization_stats']['average_confidence']:.2f}")
```

## Data Collection

### Collect Japanese Corpus

```python
from tools.data_collection.corpus_collector import JapaneseCorpusCollector
from tools.data_collection.preprocessor import JapaneseTextPreprocessor

# Initialize collector
collector = JapaneseCorpusCollector(output_dir="data/corpus")

# Collect Wikipedia articles
wikipedia_articles = collector.collect_wikipedia_articles(max_articles=1000)

# Collect news articles
news_articles = collector.collect_news_articles()

# Collect blog posts
blog_posts = collector.collect_blog_posts(max_posts=500)

# Preprocess collected data
preprocessor = JapaneseTextPreprocessor()
all_texts = [article['content'] for article in wikipedia_articles + news_articles + blog_posts]
processed_texts = preprocessor.preprocess_batch(all_texts)
quality_texts = preprocessor.filter_by_quality(processed_texts)
final_corpus = preprocessor.deduplicate_texts(quality_texts)

# Create training corpus
preprocessor.create_training_corpus(final_corpus, "data/training_corpus.json")
```

## Evaluation

### Run Comprehensive Evaluation

```python
from test.evaluation import TokenizerEvaluator

# Initialize evaluator
evaluator = TokenizerEvaluator()

# Evaluate tokenizer
results = evaluator.evaluate_tokenizer(tokenizer)

# Print results
print(f"Overall Score: {results['overall_score']:.2f}")
print(f"Basic Accuracy: {results['basic_accuracy']['accuracy']:.2f}")
print(f"Compound Accuracy: {results['compound_accuracy']['accuracy']:.2f}")
print(f"Unknown Word Inference: {results['unknown_word_inference']['accuracy']:.2f}")

# Performance benchmarks
perf = results['performance_benchmarks']
print(f"Average Processing Time: {perf['average_processing_time']:.4f}s")
print(f"Tokens per Second: {perf['average_tokens_per_second']:.0f}")
```

### Run Tests

```bash
# Run all tests
python -m pytest test/ -v

# Run specific test categories
python -m pytest test/test_tokenizer.py::TestJapaneseTokenizer::test_basic_tokenization -v

# Run with coverage
python -m pytest test/ --cov=src --cov-report=html
```

## Configuration

### Performance Tuning

```python
# Adjust confidence threshold
tokenizer.adjust_confidence_threshold(0.8)  # Higher threshold for more conservative inference

# Adjust engine weights
tokenizer.adjust_engine_weights(fast_weight=0.7, context_weight=0.3)  # Favor fast engine

# Cache optimization
cache_system.optimize_cache_sizes()
```

### Quality Thresholds

```python
# Adjust preprocessing quality thresholds
preprocessor.quality_thresholds['min_length'] = 100
preprocessor.quality_thresholds['min_kanji_ratio'] = 0.2
preprocessor.quality_thresholds['max_romaji_ratio'] = 0.2
```

## Performance Targets

- **Sentence Processing**: ≤ 50ms per sentence
- **Basic Accuracy**: 90%+
- **Unknown Word Inference**: 80%+ accuracy
- **Cache Hit Rate**: > 85%
- **Memory Usage**: < 1GB for 2,000 kanji database

## Database Coverage

- **Top 2,000 Kanji**: Covers ~97% of common Japanese text
- **214 Semantic Radicals**: Complete radical system
- **Component Mappings**: Detailed component relationships
- **Frequency Data**: Optimized for common usage patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on innovative approaches described in the strategy documents
- Implements human-like language understanding principles
- Optimized for Japanese language characteristics
- Designed for production use with comprehensive evaluation

---

**Note**: This tokenizer is designed specifically for Japanese text and implements novel approaches to tokenization that differ from traditional methods like BPE, SentencePiece, or WordPiece.