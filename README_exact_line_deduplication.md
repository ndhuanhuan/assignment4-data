# Exact Line Deduplication Implementation

This document describes the exact line deduplication functionality implemented for CS336 Assignment 4.

## Overview

The exact line deduplication function removes lines that appear more than once across a collection of input files, keeping only lines that appear exactly once in the entire corpus.

## Implementation

### Main Function
```python
def exact_line_deduplication(input_files: list, output_directory) -> None
```

**Located in**: `cs336_data/data_utils.py`

**Parameters**:
- `input_files`: List of file paths to deduplicate
- `output_directory`: Directory where deduplicated files will be written

**Algorithm**:
1. **First Pass**: Count frequency of each line across all input files using SHA-256 hash
2. **Second Pass**: For each input file, write only lines that appear exactly once to output directory

### Adapter Function
```python
def run_exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike)
```

**Located in**: `tests/adapters.py`

## Key Features

### ✅ Memory Efficiency
- Uses SHA-256 hashing to reduce memory usage instead of storing full line text
- Only keeps hash → count mappings in memory
- Suitable for large files and datasets

### ✅ Collision Resistance
- Uses SHA-256 instead of Python's built-in `hash()` to avoid hash collisions
- Critical for correctness on large datasets
- Alternative `simple_exact_line_deduplication()` function available using built-in hash

### ✅ Robust Error Handling
- Graceful handling of file I/O errors
- UTF-8 encoding with fallback for problematic characters
- Continues processing even if individual files fail

### ✅ Preserves File Structure
- Maintains original filenames in output directory
- Preserves line order within files
- Creates output directory if it doesn't exist

## Usage Examples

### Basic Usage
```python
from cs336_data.data_utils import exact_line_deduplication

# Deduplicate files
input_files = ['data/doc1.txt', 'data/doc2.txt', 'data/doc3.txt']
exact_line_deduplication(input_files, 'output/')
```

### Test Usage
```bash
# Run the specific test
uv run pytest -k test_exact_line_deduplication

# View demonstration
uv run python demo_exact_line_deduplication.py
```

## Behavior Examples

### Input Files:
**file1.txt**:
```
Line A
Shared line
Line C
```

**file2.txt**:
```
Line B  
Shared line
Line D
```

### After Deduplication:
**output/file1.txt**:
```
Line A
Line C
```

**output/file2.txt**:
```
Line B
Line D
```

**Explanation**: "Shared line" appears in both files, so it's removed from both outputs.

## Technical Details

### Hash Function Choice
- **SHA-256**: Cryptographically secure, virtually no collision risk
- **Memory overhead**: 64 hex characters per unique line vs full line text
- **Performance**: Slightly slower than built-in hash but negligible for most use cases

### File Processing
- **Encoding**: UTF-8 with error replacement for invalid characters
- **Line endings**: Preserved as-is (no normalization)
- **Empty lines**: Treated as valid lines and deduplicated normally

### Error Recovery
- File read errors don't stop processing of other files
- Warning messages for problematic files
- Graceful handling of permission errors, encoding issues

## Test Coverage

The implementation passes the official test:
```bash
uv run pytest -k test_exact_line_deduplication
```

**Test data**:
- Input files with exact line duplicates
- Expected output with duplicates removed
- Validates correct file creation and content

## Performance Characteristics

- **Time Complexity**: O(n) where n is total lines across all files
- **Space Complexity**: O(k) where k is number of unique lines  
- **Memory Usage**: ~64 bytes per unique line (hash) + Counter overhead
- **Scalability**: Suitable for large datasets due to hashing approach

## Alternative Implementation

A simpler version using Python's built-in hash is available:

```python
def simple_exact_line_deduplication(input_files: list, output_directory) -> None
```

**Trade-offs**:
- ✅ Slightly faster
- ❌ Potential hash collisions on very large datasets
- ❌ Hash values not deterministic across Python sessions

## Integration

The function integrates seamlessly with the existing CS336 data processing pipeline:

1. **Text Extraction** → Extract content from HTML/documents
2. **Language ID** → Filter by language
3. **PII Masking** → Remove sensitive information  
4. **Content Classification** → Filter NSFW/toxic content
5. **Quality Filtering** → Apply Gopher quality criteria
6. **→ Line Deduplication** → Remove duplicate lines ✅

## Summary

The exact line deduplication implementation provides:
- ✅ **Correctness**: Accurate duplicate detection across files
- ✅ **Efficiency**: Memory-efficient hash-based approach
- ✅ **Robustness**: Comprehensive error handling
- ✅ **Usability**: Simple API matching assignment requirements
- ✅ **Performance**: Scalable to large datasets
- ✅ **Testing**: Passes all required tests

Ready for production use! 🎉