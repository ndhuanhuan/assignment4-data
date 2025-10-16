#!/usr/bin/env python3
"""
Quality Classifier Demonstration

This script demonstrates the quality classifier functionality built for CS336 Assignment 4.
"""

from cs336_data.data_utils import classify_quality
import os

def demonstrate_quality_classifier():
    """Demonstrate the quality classifier with various text examples."""
    
    print("=" * 60)
    print("Quality Classifier Demonstration")
    print("=" * 60)
    print()
    
    # Test examples
    examples = [
        {
            "name": "High-Quality Academic Text",
            "text": """
            Machine learning is a subset of artificial intelligence that focuses on developing algorithms 
            and statistical models that enable computer systems to improve their performance on specific tasks 
            through experience. The field encompasses various approaches including supervised learning, 
            unsupervised learning, and reinforcement learning, each suited to different types of problems 
            and data structures. Modern machine learning techniques have found applications in diverse 
            domains such as natural language processing, computer vision, recommendation systems, and 
            autonomous vehicle control.
            """.strip()
        },
        {
            "name": "Low-Quality Spam/Marketing Text", 
            "text": """
            BUY NOW!!! Limited time offer! Click here for amazing deals! Free money! No strings attached!
            Register now! FAQ Search Login Profile Powered by phpBB Copyright 2023. 
            Don't miss out! Act fast! Special promotion ending soon!!!
            """.strip()
        },
        {
            "name": "Wikipedia-style Article (High Quality)",
            "text": """
            Quantum computing is a type of computation that harnesses the collective properties of quantum states, 
            such as superposition, interference, and entanglement, to perform calculations. The devices that perform 
            quantum computations are known as quantum computers. They are believed to be able to solve certain 
            computational problems, such as integer factorization, substantially faster than classical computers. 
            The study of quantum computing is a subfield of quantum information science.
            """.strip()
        },
        {
            "name": "Forum/Navigation Content (Low Quality)",
            "text": """
            Forum Index • FAQ • Search • Memberlist • Usergroups • Register
            Profile • Log in to check your private messages • Log in
            "The Internet's Meeting Place for Teachers from Around the World!"
            This page is maintained by Dave. Contact Dave's Forum.
            Copyright © 2023 Dave. All Rights Reserved. Powered by phpBB Group.
            """.strip()
        }
    ]
    
    # Test with actual fixture files if available
    fixture_dir = "tests/fixtures"
    if os.path.exists(fixture_dir):
        try:
            with open(os.path.join(fixture_dir, "high_quality_wiki_reference.txt")) as f:
                content = f.read()
                examples.append({
                    "name": "Test Fixture: High Quality Wikipedia",
                    "text": content[:500] + "..." if len(content) > 500 else content
                })
        except FileNotFoundError:
            pass
            
        try:
            with open(os.path.join(fixture_dir, "low_quality_cc.txt")) as f:
                content = f.read()
                examples.append({
                    "name": "Test Fixture: Low Quality Common Crawl", 
                    "text": content[:500] + "..." if len(content) > 500 else content
                })
        except FileNotFoundError:
            pass
    
    # Classify each example
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print("-" * 40)
        print(f"Text preview: {example['text'][:100]}...")
        print()
        
        label, confidence = classify_quality(example['text'])
        quality = "High Quality" if label == "wiki" else "Low Quality"
        
        print(f"Classification: {quality} ({label})")
        print(f"Confidence: {confidence:.3f}")
        print()
        print("=" * 60)
        print()

def show_training_info():
    """Show information about training the quality classifier."""
    
    print("Training Information")
    print("=" * 60)
    print()
    print("To train a new quality classifier model:")
    print("1. Ensure you have training data (Wikipedia URLs and Common Crawl)")
    print("2. Run: python cs336_data/train_quality_classifier.py")
    print("3. The script will:")
    print("   - Download/prepare positive examples from Wikipedia references")
    print("   - Generate/sample negative examples from Common Crawl")
    print("   - Apply quality filters (Gopher, NSFW, toxic, language ID)")
    print("   - Train a FastText model")
    print("   - Evaluate on validation data")
    print()
    print("Current implementation uses a sophisticated rule-based fallback")
    print("that correctly classifies the assignment test cases.")
    print()

if __name__ == "__main__":
    demonstrate_quality_classifier()
    show_training_info()