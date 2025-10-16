from fastwarc.warc import ArchiveIterator, WarcRecordType
import random
import tqdm
import fasttext
import os
import urllib.request
import gzip

from cs336_data import data_utils


def download_wiki_urls():
    """Download Wikipedia URLs if not already present."""
    url_file = "enwiki-20240420-extracted_urls.txt.gz"
    cluster_path = "/data/wiki/enwiki-20240420-extracted_urls.txt.gz"
    download_url = "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz"
    
    # Try cluster path first
    if os.path.exists(cluster_path):
        return cluster_path
    
    # Check if already downloaded locally
    if os.path.exists(url_file):
        return url_file
    
    # Download from Stanford
    print(f"Downloading Wikipedia URLs from {download_url}...")
    urllib.request.urlretrieve(download_url, url_file)
    return url_file


def subsample_urls(max_urls: int = 10000):
    """Subsample URLs from Wikipedia references."""
    url_file = download_wiki_urls()
    
    print(f"Reading URLs from {url_file}...")
    urls = []
    
    with gzip.open(url_file, 'rt', encoding='utf-8') as f:
        for line in f:
            url = line.strip()
            if url:
                urls.append(url)
    
    print(f"Total URLs available: {len(urls)}")
    
    # Subsample URLs
    sampled_urls = random.sample(urls, min(len(urls), max_urls))
    
    # Save subsampled URLs
    with open("subsampled_positive_urls.txt", "w") as f:
        for url in sampled_urls:
            f.write(f"{url}\n")
    
    print(f"Saved {len(sampled_urls)} URLs to subsampled_positive_urls.txt")
    return sampled_urls


def load_and_filter_warc(warc_file: str):
    """Load WARC file and apply quality filters."""
    if not os.path.exists(warc_file):
        print(f"WARC file {warc_file} not found. Please create it first using wget.")
        return []
    
    texts = []
    num_filtered = 0
    
    print(f"Processing WARC file: {warc_file}")
    
    with open(warc_file, "rb") as f:
        warc = ArchiveIterator(f)
        for record in tqdm.tqdm(warc):
            if record.record_type == WarcRecordType.response:
                try:
                    # Extract text from HTML
                    html_bytes = record.reader.read()
                    text = data_utils.extract_text(html_bytes)
                    
                    if not text or len(text.strip()) < 100:  # Skip very short texts
                        num_filtered += 1
                        continue
                    
                    # Apply Gopher quality filters
                    if not data_utils.gopher_quality_filter(text):
                        num_filtered += 1
                        continue
                    
                    # Filter NSFW content
                    nsfw_label, nsfw_score = data_utils.classify_nsfw(text)
                    if nsfw_label == "nsfw" and nsfw_score > 0.9:
                        num_filtered += 1
                        continue
                    
                    # Filter toxic content
                    toxic_label, toxic_score = data_utils.classify_toxic_speech(text)
                    if toxic_label == "toxic" and toxic_score > 0.9:
                        num_filtered += 1
                        continue
                    
                    # Check language (keep only English)
                    lang, lang_score = data_utils.identify_language(text)
                    if lang != "en" or lang_score < 0.8:
                        num_filtered += 1
                        continue
                    
                    texts.append(text)
                    
                except Exception as e:
                    print(f"Error processing record: {e}")
                    num_filtered += 1
                    continue
    
    print(f"Total filtered records: {num_filtered}")
    print(f"Total remaining high-quality records: {len(texts)}")
    return texts


def sample_common_crawl(warc_file: str, max_records: int = 50000):
    """Sample texts from Common Crawl WARC file for negative examples."""
    if not os.path.exists(warc_file):
        print(f"Common Crawl WARC file {warc_file} not found.")
        print("Using synthetic low-quality examples instead.")
        return generate_synthetic_low_quality(max_records)
    
    texts = []
    with open(warc_file, "rb") as f:
        warc = ArchiveIterator(f)
        for record in tqdm.tqdm(warc):
            if record.record_type == WarcRecordType.response and len(texts) < max_records:
                try:
                    html_bytes = record.reader.read()
                    text = data_utils.extract_text(html_bytes)
                    if text and len(text.strip()) > 50:
                        texts.append(text)
                except Exception:
                    continue
    
    print(f"Sampled {len(texts)} texts from Common Crawl")
    return texts[:max_records]


def generate_synthetic_low_quality(count: int = 50000):
    """Generate synthetic low-quality text examples."""
    print(f"Generating {count} synthetic low-quality examples...")
    
    templates = [
        # Very short content
        "Buy now! Limited time offer!",
        "Click here for more info.",
        "Error 404: Page not found.",
        
        # Repetitive content
        "Lorem ipsum dolor sit amet. " * 20,
        "This is a test. " * 30,
        "Hello world! " * 25,
        
        # Symbol-heavy content
        "!@#$%^&*()_+ " * 40,
        "123456789 " * 50,
        ".................. " * 30,
        
        # Poor grammar/structure
        "very bad grammar here no punctuation or structure at all just words",
        "teh qiuck brwon fox jmps ovr teh lazy dag",  # Typos
        "EVERYTHING IS IN CAPS AND VERY ANNOYING TO READ",
        
        # Spam-like content
        "URGENT: You have won $1,000,000! Click now!",
        "Free money! No strings attached! Act now!",
        "Single moms in your area want to meet you!",
    ]
    
    synthetic_texts = []
    for i in range(count):
        # Randomly combine and modify templates
        base_template = random.choice(templates)
        repeat_count = random.randint(1, 10)
        text = base_template * repeat_count
        
        # Add some random variations
        if random.random() < 0.3:
            text += "\n" + "..." * random.randint(5, 20)
        
        synthetic_texts.append(text)
    
    return synthetic_texts


def prepare_training_data():
    """Prepare training data for the quality classifier."""
    print("Preparing training data for quality classifier...")
    
    # Load high-quality examples (Wikipedia references)
    warc_file = "subsampled_positive_urls.warc"
    if os.path.exists(warc_file):
        positives = load_and_filter_warc(warc_file)
    else:
        print(f"WARC file {warc_file} not found.")
        print("Please run: wget --timeout=5 -i subsampled_positive_urls.txt --warc-file=subsampled_positive_urls.warc -O /dev/null")
        print("Using synthetic high-quality examples for now...")
        positives = generate_synthetic_high_quality(5000)
    
    # Load low-quality examples
    cc_warc_file = "example.warc.gz"  # Common Crawl sample
    negatives = sample_common_crawl(cc_warc_file, max_records=len(positives))
    
    # Balance the dataset
    min_size = min(len(positives), len(negatives))
    positives = positives[:min_size]
    negatives = negatives[:min_size]
    
    print(f"Final dataset: {len(positives)} high-quality, {len(negatives)} low-quality")
    
    # Format for FastText
    positive_labeled = [f"__label__wiki {text.replace(chr(10), ' ')}" for text in positives]
    negative_labeled = [f"__label__cc {text.replace(chr(10), ' ')}" for text in negatives]
    
    # Combine and shuffle
    all_texts = positive_labeled + negative_labeled
    random.shuffle(all_texts)
    
    # Split train/validation
    train_size = int(len(all_texts) * 0.8)
    train_texts = all_texts[:train_size]
    valid_texts = all_texts[train_size:]
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(valid_texts)}")
    
    # Save training data
    with open("quality_classifier_data.train", "w", encoding='utf-8') as f:
        for text in train_texts:
            f.write(f"{text}\n")
    
    with open("quality_classifier_data.valid", "w", encoding='utf-8') as f:
        for text in valid_texts:
            f.write(f"{text}\n")
    
    print("Training data saved successfully!")
    return train_texts, valid_texts


def generate_synthetic_high_quality(count: int = 5000):
    """Generate synthetic high-quality text examples."""
    print(f"Generating {count} synthetic high-quality examples...")
    
    templates = [
        """Machine learning is a subset of artificial intelligence that focuses on developing algorithms 
        and statistical models that enable computer systems to improve their performance on specific tasks 
        through experience. The field encompasses various approaches including supervised learning, 
        unsupervised learning, and reinforcement learning, each suited to different types of problems 
        and data structures.""",
        
        """Natural language processing represents a significant branch of computer science that deals with 
        the interaction between computers and human language. Modern NLP systems utilize sophisticated 
        algorithms to understand, interpret, and generate human language in a valuable way, enabling 
        applications such as machine translation, sentiment analysis, and conversational AI systems.""",
        
        """The history of computer science spans several decades, beginning with early mechanical calculators 
        and evolving into the sophisticated digital systems we use today. Key milestones include the 
        development of programming languages, the creation of the internet, and the advancement of 
        algorithms that power modern computing applications across various industries."""
    ]
    
    high_quality_texts = []
    for i in range(count):
        base_text = random.choice(templates)
        # Add some variation
        variations = [
            " Furthermore, recent developments have shown promising results.",
            " Research in this area continues to advance rapidly.",
            " These concepts form the foundation for many applications.",
            " The implications of this work extend across multiple disciplines."
        ]
        text = base_text + random.choice(variations)
        high_quality_texts.append(text)
    
    return high_quality_texts


def train_model():
    """Train the quality classifier model."""
    print("Training quality classifier...")
    
    # Check if training data exists
    if not os.path.exists("quality_classifier_data.train"):
        print("Training data not found. Preparing data first...")
        prepare_training_data()
    
    # Train FastText model
    model = fasttext.train_supervised(
        "quality_classifier_data.train",
        lr=0.1,
        epoch=25,
        wordNgrams=2,
        dim=100
    )
    
    # Save model
    model.save_model("quality_classifier_model.bin")
    print("Model saved as quality_classifier_model.bin")
    
    # Test model
    if os.path.exists("quality_classifier_data.valid"):
        print("Evaluating model...")
        results = model.test("quality_classifier_data.valid")
        print(f"Validation results: {results}")
    
    # Test with sample predictions
    print("\nSample predictions:")
    test_cases = [
        "Machine learning algorithms have revolutionized data analysis.",
        "Buy now! Limited time offer! Click here!!!",
        "The economic implications of artificial intelligence are profound.",
        "Free money! No strings attached!"
    ]
    
    for text in test_cases:
        prediction = model.predict(text)
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {prediction}")
        print()
    
    return model


if __name__ == "__main__":
    # Step 1: Subsample URLs (optional, if you want fresh URLs)
    # subsample_urls(max_urls=5000)
    
    # Step 2: Prepare training data
    prepare_training_data()
    
    # Step 3: Train model
    train_model()