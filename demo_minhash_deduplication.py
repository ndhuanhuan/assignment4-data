#!/usr/bin/env python3
"""
MinHash Deduplication Demonstration

This script demonstrates the minhash deduplication functionality.
"""

from cs336_data.minhash_dedup import minhash_dedupe
import tempfile
import os

def demonstrate_minhash_deduplication():
    """Demonstrate minhash deduplication with example files."""
    
    print("=" * 60)
    print("MinHash Deduplication Demonstration")
    print("=" * 60)
    print()
    
    # Create temporary directory for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        
        # Create example files with exact and fuzzy duplicates
        files_content = {
            'document1.txt': """
            The quick brown fox jumps over the lazy dog.
            This is a sample document for testing deduplication.
            Machine learning algorithms are becoming more sophisticated.
            Natural language processing has many applications.
            """.strip(),
            
            'document2.txt': """
            THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG!
            This is a sample document for testing deduplication...
            Machine learning algorithms are becoming more sophisticated!!
            Natural language processing has many applications???
            """.strip(),  # Fuzzy duplicate of document1
            
            'document3.txt': """
            Deep learning is a subset of machine learning.
            Neural networks have multiple layers of abstraction.
            Artificial intelligence is transforming many industries.
            Data science combines statistics with computer science.
            """.strip(),
            
            'document4.txt': """
            The quick brown fox jumps over the lazy dog.
            This is a sample document for testing deduplication.
            Machine learning algorithms are becoming more sophisticated.
            Natural language processing has many applications.
            """.strip(),  # Exact duplicate of document1
            
            'document5.txt': """
            Computer vision allows machines to interpret visual information.
            Reinforcement learning uses rewards and penalties.
            Unsupervised learning finds patterns in unlabeled data.
            Big data requires scalable processing frameworks.
            """.strip()
        }
        
        # Write input files
        input_files = []
        print("Created input files:")
        for filename, content in files_content.items():
            filepath = os.path.join(input_dir, filename)
            input_files.append(filepath)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f"\n{filename}:")
            for i, line in enumerate(content.split('\n'), 1):
                print(f"  {i}: {line.strip()}")
        
        print("\n" + "=" * 60)
        print("RUNNING MINHASH DEDUPLICATION")
        print("=" * 60)
        
        # Run minhash deduplication
        minhash_dedupe(
            input_files=input_files,
            num_hashes=100,
            num_bands=10, 
            ngrams=5,
            jaccard_threshold=0.8,
            output_directory=output_dir,
            progress=True
        )
        
        print("\n" + "=" * 60)
        print("RESULTS AFTER DEDUPLICATION")
        print("=" * 60)
        
        # Show results
        output_files = os.listdir(output_dir)
        output_files.sort()
        
        print(f"\nFiles kept: {len(output_files)} out of {len(input_files)} original files")
        print(f"Files removed: {len(input_files) - len(output_files)}")
        print(f"\nRemaining files: {output_files}")
        
        for filename in output_files:
            output_file = os.path.join(output_dir, filename)
            
            with open(output_file, 'r') as f:
                content = f.read()
            
            print(f"\n{filename}:")
            for i, line in enumerate(content.split('\n'), 1):
                if line.strip():
                    print(f"  {i}: {line.strip()}")


def show_minhash_info():
    """Show information about the minhash deduplication algorithm."""
    
    print("\n" + "=" * 60)
    print("MINHASH DEDUPLICATION ALGORITHM")
    print("=" * 60)
    print()
    print("How it works:")
    print("1. Text normalization:")
    print("   - Convert to lowercase")
    print("   - Remove punctuation") 
    print("   - Normalize whitespace")
    print("   - Remove accents (NFD unicode normalization)")
    print()
    print("2. N-gram extraction:")
    print("   - Create overlapping word n-grams")
    print("   - Example: 'the quick brown fox' -> ['the quick', 'quick brown', 'brown fox'] (2-grams)")
    print()
    print("3. MinHash signatures:")
    print("   - Generate multiple hash values for each n-gram set")
    print("   - Keep minimum hash for each permutation")
    print("   - Creates compact document fingerprint")
    print()
    print("4. LSH (Locality Sensitive Hashing):")
    print("   - Split signatures into bands")
    print("   - Documents with matching bands are candidate duplicates")
    print("   - Reduces comparison complexity from O(n²) to O(n)")
    print()
    print("5. Jaccard similarity verification:")
    print("   - Compute exact n-gram overlap for candidates")
    print("   - Jaccard = |intersection| / |union|")
    print("   - Remove documents above threshold")
    print()
    print("Parameters:")
    print("- num_hashes: More hashes = better accuracy, more computation")
    print("- num_bands: More bands = higher recall, more false positives")
    print("- ngrams: Larger n-grams = more specific matching")
    print("- jaccard_threshold: Higher threshold = stricter matching")


if __name__ == "__main__":
    demonstrate_minhash_deduplication()
    show_minhash_info()