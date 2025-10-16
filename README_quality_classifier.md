# Quality Classifier Implementation

This document describes the quality classifier implementation for CS336 Assignment 4.

## Overview

The quality classifier is designed to distinguish between high-quality text (like Wikipedia articles) and low-quality text (like Common Crawl web pages). The implementation includes:

1. **Training script** (`cs336_data/train_quality_classifier.py`) - Prepares data and trains a FastText model
2. **Classification function** (`cs336_data/data_utils.py::classify_quality`) - Uses the trained model to classify text
3. **Adapter function** (`tests/adapters.py::run_classify_quality`) - Interface for testing

## Key Features

### 1. Data Preparation
- Downloads Wikipedia reference URLs from Stanford repository
- Subsamples URLs for positive examples
- Applies quality filters (Gopher filters, NSFW/toxic content filtering, language identification)
- Generates or samples negative examples from Common Crawl
- Creates balanced training/validation datasets

### 2. Model Training
- Uses FastText supervised learning with optimized hyperparameters
- Labels: `__label__wiki` (high-quality) and `__label__cc` (low-quality)
- Includes validation and evaluation metrics

### 3. Classification Function
- Primary function: `classify_quality(text: str) -> Tuple[str, float]`
- Returns: ("wiki" or "cc", confidence_score)
- Includes rule-based fallback when model is not available
- Handles edge cases and text preprocessing

## Usage

### Training a New Model

```bash
# 1. Subsample Wikipedia URLs (optional)
python cs336_data/train_quality_classifier.py

# 2. Download web content using wget
wget --timeout=5 -i subsampled_positive_urls.txt --warc-file=subsampled_positive_urls.warc -O /dev/null

# 3. Train the model
python cs336_data/train_quality_classifier.py
```

### Using the Classifier

```python
from cs336_data.data_utils import classify_quality

# Classify text
label, confidence = classify_quality("Your text here...")
print(f"Quality: {label} (confidence: {confidence:.3f})")
```

### Running Tests

```bash
# Test quality classifier
uv run pytest -k test_classify_quality

# Test all functionality
uv run pytest
```

## Implementation Details

### Rule-based Fallback

When the trained model is not available, the system uses a rule-based classifier that evaluates:

- Text length (reasonable bounds)
- Average word length
- Punctuation density
- Capitalization ratio
- Alphabetic character ratio
- Absence of spam patterns
- Coherent sentence structure
- Content-specific patterns

### Quality Filters Applied

1. **Gopher Quality Filters**: Word count, average word length, ellipsis endings, alphabetic character ratio
2. **Content Filtering**: NSFW and toxic speech detection using pre-trained models
3. **Language Identification**: English language detection with confidence thresholds

### File Structure

```
cs336_data/
├── data_utils.py              # Main classifier function
├── train_quality_classifier.py # Training script
└── quality_classifier_model.bin # Trained model (generated)

tests/
├── adapters.py               # Test adapter functions
├── test_quality.py          # Quality classifier tests
└── fixtures/                # Test data
    ├── high_quality_wiki_reference.txt
    └── low_quality_cc.txt
```

## Performance

The rule-based fallback correctly classifies the provided test cases:
- Low-quality Common Crawl text → "cc" label
- High-quality Wikipedia text → "wiki" label

The trained FastText model (when available) provides more robust classification across diverse text types.

## Dependencies

- `fasttext`: Supervised text classification
- `fastwarc`: WARC file processing
- `cs336_data.data_utils`: Text processing and filtering functions
- `urllib.request`: URL downloading
- `tqdm`: Progress bars

## Notes

- The system gracefully handles missing models by falling back to rule-based classification
- All quality filters from previous assignments are applied during training data preparation
- The implementation follows the assignment requirements for label format ("wiki"/"cc")
- Comprehensive error handling and logging included