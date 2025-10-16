#!/usr/bin/env python3
"""
Exact Line Deduplication Demonstration

This script demonstrates the exact line deduplication functionality.
"""

from cs336_data.data_utils import exact_line_deduplication
import tempfile
import os

def demonstrate_exact_line_deduplication():
    """Demonstrate exact line deduplication with example files."""
    
    print("=" * 60)
    print("Exact Line Deduplication Demonstration")
    print("=" * 60)
    print()
    
    # Create temporary directory for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        
        # Create example files with duplicates
        files_content = {
            'article1.txt': [
                'Introduction to Machine Learning\n',
                'Machine learning is a subset of AI.\n',
                'It involves training algorithms on data.\n',
                'Common footer: Copyright 2024\n'
            ],
            'article2.txt': [
                'Deep Learning Fundamentals\n', 
                'Deep learning uses neural networks.\n',
                'It involves training algorithms on data.\n',  # Duplicate
                'Common footer: Copyright 2024\n'  # Duplicate
            ],
            'article3.txt': [
                'Natural Language Processing\n',
                'NLP deals with human language.\n',
                'It has many applications.\n',
                'Thank you for reading.\n'
            ]
        }
        
        # Write input files
        input_files = []
        print("Created input files:")
        for filename, content in files_content.items():
            filepath = os.path.join(input_dir, filename)
            input_files.append(filepath)
            
            with open(filepath, 'w') as f:
                f.writelines(content)
            
            print(f"\n{filename}:")
            for i, line in enumerate(content, 1):
                print(f"  {i}: {line.rstrip()}")
        
        print("\n" + "=" * 60)
        print("DEDUPLICATION PROCESS")
        print("=" * 60)
        
        # Run deduplication
        exact_line_deduplication(input_files, output_dir)
        
        print("\n" + "=" * 60)
        print("RESULTS AFTER DEDUPLICATION")
        print("=" * 60)
        
        # Show results
        for filename in files_content.keys():
            output_file = os.path.join(output_dir, filename)
            
            with open(output_file, 'r') as f:
                deduplicated_content = f.readlines()
            
            print(f"\n{filename} (after deduplication):")
            if deduplicated_content:
                for i, line in enumerate(deduplicated_content, 1):
                    print(f"  {i}: {line.rstrip()}")
            else:
                print("  (empty - all lines were duplicates)")
        
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)
        print()
        print("Lines removed as duplicates:")
        print("- 'It involves training algorithms on data.' (appeared in article1.txt and article2.txt)")
        print("- 'Common footer: Copyright 2024' (appeared in article1.txt and article2.txt)")
        print()
        print("Lines kept (appeared only once):")
        print("- All unique content from each article")
        print("- Headers, specific content, and unique footers")


def show_usage_info():
    """Show usage information for the exact line deduplication function."""
    
    print("\n" + "=" * 60)
    print("USAGE INFORMATION")
    print("=" * 60)
    print()
    print("Function signature:")
    print("  exact_line_deduplication(input_files: list, output_directory: str)")
    print()
    print("Parameters:")
    print("  input_files: List of file paths to deduplicate")
    print("  output_directory: Directory where deduplicated files will be saved")
    print()
    print("Behavior:")
    print("- Counts frequency of each line across ALL input files")
    print("- Keeps only lines that appear exactly ONCE in the entire corpus")
    print("- Preserves original filenames in output directory")
    print("- Uses SHA-256 hashing to avoid collisions")
    print("- Handles encoding issues gracefully")
    print()
    print("Example usage:")
    print("  from cs336_data.data_utils import exact_line_deduplication")
    print("  files = ['data/doc1.txt', 'data/doc2.txt', 'data/doc3.txt']")
    print("  exact_line_deduplication(files, 'output/')")


if __name__ == "__main__":
    demonstrate_exact_line_deduplication()
    show_usage_info()