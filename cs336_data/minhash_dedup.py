# MinHash LSH Document Deduplication Implementation
# 
# This module implements fuzzy document deduplication using MinHash signatures
# and Locality Sensitive Hashing (LSH). The algorithm can detect both exact
# and near-duplicate documents efficiently, even when they have minor differences
# in formatting, punctuation, or word order.
#
# Key concepts:
# - MinHash: Creates compact fingerprints of documents while preserving similarity
# - LSH: Efficiently finds candidate duplicate pairs without comparing all documents
# - Jaccard Similarity: Measures overlap between document n-gram sets
#
# Reference: https://github.com/heng380/assignment4-data/blob/main/cs336_data/deduplication/minhash_dedup.py

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
import os
import random
import re
import shutil
import mmh3  # MurmurHash3 - fast, high-quality hash function
import numpy as np
import unicodedata
from tqdm import tqdm

# Regular expressions for text normalization
WS_RE = re.compile(r"\s+")        # Matches one or more whitespace characters
PUNCT_RE = re.compile(r"[^\w\s]") # Matches any character that's not word char or whitespace


def normalize_text(text: str) -> str:
    """Normalize text to improve duplicate detection accuracy.
    
    Text normalization is crucial for fuzzy deduplication because documents
    with the same content but different formatting should be considered duplicates.
    
    Normalization steps:
    1. Lowercasing: "Hello" and "hello" become the same
    2. Remove punctuation: "word!" and "word" become the same
    3. Normalize whitespace: "word  word" and "word word" become the same
    4. NFD unicode normalization: handles accented characters consistently
    5. Remove accents: "café" and "cafe" become more similar
    
    Args:
        text: Raw input text to normalize
        
    Returns:
        Normalized text string ready for n-gram extraction
        
    Example:
        >>> normalize_text("Hello, World!   How are you?")
        'hello world how are you'
    """
    text = text.lower()                           # Convert to lowercase
    text = PUNCT_RE.sub(" ", text)               # Replace punctuation with spaces
    text = WS_RE.sub(" ", text).strip()          # Normalize whitespace to single spaces
    text = unicodedata.normalize("NFD", text)    # Unicode normalization (decomposed form)
    return text


def get_minhash(ngram_set: set[str], num_perm: int) -> list[int]:
    """Compute MinHash signature for a set of n-grams.
    
    MinHash is a probabilistic technique that creates a compact "fingerprint"
    of a document while preserving similarity relationships. Two documents
    with similar content will have similar MinHash signatures.
    
    Algorithm:
    1. For each n-gram in the set, compute a hash value
    2. For each permutation (simulated by XOR with different seeds):
       - XOR the hash with the seed to simulate a different hash function
       - Keep track of the minimum hash value seen so far
    3. The final signature is the list of minimum hash values
    
    Mathematical property:
    P(min_hash_i(A) == min_hash_i(B)) = Jaccard_similarity(A, B)
    
    This means the probability that two documents have the same MinHash value
    for any given permutation equals their Jaccard similarity.
    
    Args:
        ngram_set: Set of n-gram strings from the document
        num_perm: Number of hash permutations (higher = more accurate)
        
    Returns:
        List of minimum hash values (the MinHash signature)
        
    Example:
        For documents A and B with Jaccard similarity 0.8:
        About 80% of their MinHash signature values will be identical
    """
    # Create different "seeds" to simulate different hash functions
    # Each seed represents a different permutation of the hash function
    seeds = np.arange(num_perm, dtype=np.uint32)
    
    # Initialize minimums to maximum possible value
    # We'll update these as we find smaller hash values
    mins = np.full(num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)

    # Process each n-gram in the document
    for ngram in ngram_set:
        # Compute hash of the n-gram using MurmurHash3 (fast, high-quality)
        h = mmh3.hash(ngram, signed=False)
        
        # For each permutation, XOR with seed to simulate different hash function
        # Then update minimum if this hash is smaller than current minimum
        # Broadcasting: h ^ seeds creates array where h is XORed with each seed
        mins[:] = np.minimum(mins, h ^ seeds)
    
    return mins.tolist()


def get_ngram_set(text: str, ngrams: int):
    """Extract overlapping word n-grams from normalized text.
    
    N-grams are sequences of consecutive words that help capture local
    word patterns and ordering. They're more robust than individual words
    because they preserve some context and word relationships.
    
    Args:
        text: Normalized text (already lowercased, punctuation removed, etc.)
        ngrams: Length of each n-gram (number of consecutive words)
        
    Returns:
        Set of n-gram strings
        
    Example:
        >>> get_ngram_set("the quick brown fox", 3)
        {'the quick brown', 'quick brown fox'}
        
        >>> get_ngram_set("machine learning is powerful", 2)
        {'machine learning', 'learning is', 'is powerful'}
    """
    words = text.split()  # Split into individual words
    
    # Generate overlapping n-grams using sliding window
    # For each starting position i, take ngrams consecutive words
    # range(len(words) - ngrams + 1) ensures we don't go past the end
    return set(" ".join(words[i : i + ngrams]) for i in range(len(words) - ngrams + 1))


def get_file_normalized_ngram_set(file: os.PathLike, ngrams: int):
    """Read file, normalize text, and extract n-gram set.
    
    This is a convenience function that combines file reading,
    text normalization, and n-gram extraction in one step.
    
    Args:
        file: Path to text file to process
        ngrams: Length of n-grams to extract
        
    Returns:
        Set of normalized n-grams from the file
    """
    # Read file with UTF-8 encoding, ignoring decode errors
    # errors="ignore" prevents crashes on malformed text
    with open(file, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    # Apply normalization then extract n-grams
    return get_ngram_set(normalize_text(text), ngrams)


def build_signature(path: os.PathLike, *, ngrams: int, num_hashes: int):
    ngram_set = get_file_normalized_ngram_set(path, ngrams)
    return path, get_minhash(ngram_set, num_hashes)


def collect_signatures(
    input_files: list[os.PathLike],
    *,
    ngrams: int,
    num_hashes: int,
    progress: bool = False,
):
    """Generate MinHash signatures for multiple documents in parallel.
    
    This function coordinates the parallel processing of many documents
    to compute their MinHash signatures efficiently. It's the orchestration
    layer that manages multiprocessing and progress tracking.
    
    The parallel processing is crucial for performance when dealing with
    large document collections, as MinHash computation is CPU-intensive.
    
    Args:
        input_files: List of file paths to process
        ngrams: Length of n-grams to extract from each document  
        num_hashes: Number of hash functions (signature dimension)
        progress: Whether to show progress bar during computation
        
    Returns:
        Dict mapping file paths to their MinHash signatures
        
    Implementation Details:
        - Uses ProcessPoolExecutor for true parallelism (not limited by GIL)
        - Number of worker processes matches CPU cores for optimal performance
        - Each worker processes multiple files to amortize process creation cost
        - Progress tracking helps monitor long-running computations
    """
    signatures = {}
    
    # Use ProcessPoolExecutor for CPU-bound parallel processing
    # ProcessPoolExecutor creates separate Python processes (not threads)
    # This bypasses the Global Interpreter Lock (GIL) for true parallelism
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        # Create a partially applied function with fixed parameters
        # This reduces the arguments that need to be passed to each worker
        submit = partial(pool.submit, build_signature, ngrams=ngrams, num_hashes=num_hashes)
        
        # Submit all jobs to the worker pool
        # Each future represents one file being processed
        futures = [submit(p) for p in input_files]
        
        # Collect results as they complete (not necessarily in submission order)
        # as_completed() yields futures as soon as their results are ready
        # This provides better progress feedback than waiting for all jobs
        for future in tqdm(as_completed(futures), total=len(futures), disable=not progress, desc="Building signatures"):
            path, minhash = future.result()  # Blocks until this specific future completes
            signatures[path] = minhash
    
    return signatures


def build_ngram_set(path: os.PathLike, *, ngrams: int):
    """Compute and return the normalized n-gram set for a single file.
    
    This is a helper function for parallel processing that combines
    the file path with its computed n-gram set. It's designed to be
    used with ProcessPoolExecutor.
    
    Args:
        path: Path to the file to process
        ngrams: Length of n-grams to extract
        
    Returns:
        Tuple of (file_path, n_gram_set)
    """
    return path, get_file_normalized_ngram_set(path, ngrams)


def collect_ngram_sets(
    files: set[os.PathLike],
    *,
    ngrams: int,
    progress: bool = False,
):
    """Compute n-gram sets for multiple documents in parallel.
    
    This function is used during the candidate pair verification phase
    where we need to compute exact n-gram sets (not just MinHash signatures)
    to calculate precise Jaccard similarities.
    
    While MinHash signatures give us probabilistic similarity estimates,
    we use exact n-gram sets for final verification to avoid false positives
    in our duplicate detection.
    
    Args:
        files: Set of file paths to process
        ngrams: Length of n-grams to extract from each document
        progress: Whether to show progress bar during computation
        
    Returns:
        Dict mapping file paths to their n-gram sets
        
    Note:
        This is separate from collect_signatures() because sometimes we only
        need n-gram sets for specific candidate pairs, not all documents.
    """
    ngram_sets = {}
    
    # Use same parallel processing pattern as collect_signatures()
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        submit = partial(pool.submit, build_ngram_set, ngrams=ngrams)
        futures = [submit(p) for p in files]
        
        # Process results as they complete
        for future in tqdm(
            as_completed(futures), total=len(futures), disable=not progress, desc="Building n‑gram sets"
        ):
            path, ngram_set = future.result()
            ngram_sets[path] = ngram_set
    
    return ngram_sets


def minhash_dedupe(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
    progress: bool = False,
):
    """Perform fuzzy document deduplication using MinHash and LSH.
    
    This is the main function that orchestrates the entire MinHash LSH
    deduplication pipeline. It's a sophisticated algorithm that can find
    near-duplicate documents even when they have small differences.
    
    The algorithm works in several phases:
    
    1. **Signature Generation**: Compute MinHash signatures for all documents
       - Each document → low-dimensional fingerprint (num_hashes integers)
       - Signatures preserve similarity: similar docs → similar signatures
    
    2. **LSH Banding**: Group signatures into bands to find candidate pairs
       - Split each signature into num_bands sections
       - Documents with identical bands are candidate duplicates
       - This dramatically reduces the number of pairs to check
    
    3. **Candidate Pair Verification**: Compute exact Jaccard similarity
       - For each candidate pair, compute precise n-gram overlap
       - Only pairs above jaccard_threshold are considered duplicates
    
    4. **Clustering**: Group documents into duplicate clusters
       - Connected components: if A≈B and B≈C, then A,B,C are one cluster
       - Each cluster represents a set of near-duplicate documents
    
    5. **Output Selection**: Keep one representative from each cluster
       - Randomly select one document from each duplicate cluster
       - Copy all unique documents + representatives to output directory
    
    Args:
        input_files: List of document file paths to deduplicate
        num_hashes: Number of hash functions for MinHash signatures
                   Higher = more accurate but slower (typical: 100-200)
        num_bands: Number of bands for LSH bucketing
                  Higher = more sensitive but more false positives
        ngrams: Length of word n-grams for text analysis (typical: 3-5)
        jaccard_threshold: Minimum similarity to consider documents duplicates
                          Range [0,1], typical: 0.7-0.9
        output_directory: Where to write deduplicated documents
        progress: Whether to show progress bars and status messages
        
    Mathematical Insight:
        The probability that two documents with Jaccard similarity J
        will be detected as candidates is approximately:
        P ≈ 1 - (1 - J^(num_hashes/num_bands))^num_bands
        
        This creates an S-curve: documents above a threshold are likely
        to be detected, while those below are likely to be missed.
    """
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Phase 1: Generate MinHash signatures for all documents
    # This is the most computationally expensive step
    signatures = collect_signatures(input_files, ngrams=ngrams, num_hashes=num_hashes, progress=progress)

    # Phase 2: LSH Banding - Find candidate duplicate pairs
    # bands[band_signature] = list of files that have this band signature
    bands: dict[tuple[int, ...], list[os.PathLike]] = {}
    candidate_dups: set[tuple[os.PathLike, os.PathLike]] = set()

    # For each document's signature, extract bands and find potential matches
    for file, minhash in signatures.items():
        # Split the signature into num_bands bands
        # Each band contains (num_hashes // num_bands) consecutive hash values
        for band in range(num_bands):
            # Extract one band: every num_bands-th element starting from 'band'
            # Example: if num_bands=4, band=1, we get elements [1, 5, 9, 13, ...]
            band_minhash = tuple(minhash[band::num_bands])
            
            # Add this file to the bucket for this band signature
            if band_minhash not in bands:
                bands[band_minhash] = []
            bands[band_minhash].append(file)

            # Any other files in the same bucket are candidate duplicates
            # LSH Property: files with identical band signatures are likely similar
            for other_file in bands[band_minhash]:
                if other_file != file:
                    # Store as unordered pair (smaller file first for consistency)
                    pair = tuple(sorted([file, other_file]))
                    candidate_dups.add(pair)

    if progress:
        print(f"Will test + cluster {len(candidate_dups)} candidate duplicates")

    # Phase 3: Candidate Pair Verification using exact Jaccard similarity
    # We need exact n-gram sets (not just signatures) for precise calculation
    
    # Collect all unique files that appear in candidate pairs
    unique_files = set()
    for f1, f2 in candidate_dups:
        unique_files.add(f1)
        unique_files.add(f2)
    
    # Compute exact n-gram sets for just these files (efficiency optimization)
    ngram_sets = collect_ngram_sets(unique_files, ngrams=ngrams, progress=progress)

    # Phase 4: Clustering - Build connected components of similar documents
    # clusters[file] = set of all files in the same duplicate cluster as 'file'
    clusters: dict[os.PathLike, set[os.PathLike]] = {}

    # Test each candidate pair and build clusters
    for f1, f2 in tqdm(candidate_dups, disable=not progress, desc="Testing + clustering"):
        s1, s2 = ngram_sets[f1], ngram_sets[f2]
        
        # Skip empty documents (edge case)
        if not s1 or not s2:
            continue

        # Compute exact Jaccard similarity: |intersection| / |union|
        # This is the ground truth similarity measure
        jaccard_similarity = len(s1 & s2) / len(s1 | s2)

        # If similarity exceeds threshold, merge these documents into same cluster
        if jaccard_similarity >= jaccard_threshold:
            # Union-find style clustering: merge the two clusters
            if f1 not in clusters:
                clusters[f1] = {f1}
            if f2 not in clusters:
                clusters[f2] = {f2}
            
            # Merge clusters: all files in f1's cluster join f2's cluster
            merged_cluster = clusters[f1] | clusters[f2]
            for file in merged_cluster:
                clusters[file] = merged_cluster

    if progress:
        print(f"Found {len(set(id(cluster) for cluster in clusters.values()))} clusters")

    # Phase 5: Output Selection - Choose representatives from each cluster
    
    # Get unique clusters (sets might be shared due to clustering algorithm)
    cluster_set = {frozenset(cluster) for cluster in clusters.values()}
    
    # Start with all files that aren't in any duplicate cluster
    files_to_write = [f for f in input_files if f not in clusters]
    
    # Add one randomly chosen representative from each duplicate cluster
    # This preserves content while removing redundancy
    files_to_write += [random.choice(tuple(cluster)) for cluster in cluster_set]

    # Copy selected files to output directory
    for file in tqdm(files_to_write, disable=not progress, desc="Writing output files"):
        dst = os.path.join(output_directory, os.path.basename(file))
        shutil.copy2(file, dst)  # copy2 preserves metadata