from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import re
from typing import Tuple


def extract_text(html_bytes: bytes) -> str:
    """Extract plain text from HTML bytes.
    
    Args:
        html_bytes: Raw HTML content as bytes
        
    Returns:
        Extracted plain text content
    """
    # Detect the encoding of the HTML content
    encoding = detect_encoding(html_bytes)
    
    # Decode using detected encoding with error handling
    html_str = html_bytes.decode(encoding, errors='replace')
    
    # Extract plain text from HTML
    return extract_plain_text(html_str)


def identify_language(text: str) -> Tuple[str, float]:
    """Identify the main language present in a text string.
    
    Args:
        text: Unicode string to analyze
        
    Returns:
        A tuple containing:
        - language_code: String identifier of the predicted language (e.g., "en", "zh")
        - confidence: Float between 0 and 1 representing confidence in the prediction
    """
    # Suppress FastText warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Load FastText language identification model
        try:
            model = fasttext.load_model('lid.176.bin')
        except Exception:
            # If the model file doesn't exist, try to download it
            import urllib.request
            import os
            
            model_path = 'lid.176.bin'
            if not os.path.exists(model_path):
                print("Downloading FastText language identification model...")
                url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
                urllib.request.urlretrieve(url, model_path)
            model = fasttext.load_model(model_path)
    
    # Clean the text: remove extra whitespace and newlines
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    
    # FastText expects text without newlines for language identification
    cleaned_text = cleaned_text.replace('\n', ' ')
    
    # Handle empty or very short text
    if len(cleaned_text.strip()) == 0:
        return "en", 0.0  # Default to English with low confidence
    
    # Predict language
    predictions = model.predict(cleaned_text, k=1)
    
    # Extract language code and confidence
    language_label = predictions[0][0]  # e.g., '__label__en'
    confidence = float(predictions[1][0])
    
    # Remove the '__label__' prefix that FastText adds
    language_code = language_label.replace('__label__', '')
    
    # Map common language codes to expected format
    language_mapping = {
        'zh-cn': 'zh',  # Chinese Simplified
        'zh-tw': 'zh',  # Chinese Traditional  
        'en': 'en',     # English
    }
    
    # Apply mapping if available, otherwise use original code
    mapped_language = language_mapping.get(language_code, language_code)
    
    return mapped_language, confidence