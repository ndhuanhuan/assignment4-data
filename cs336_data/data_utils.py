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


def mask_emails(text: str) -> Tuple[str, int]:
    """Mask email addresses in text with |||EMAIL_ADDRESS|||.
    
    Args:
        text: Input string that may contain email addresses
        
    Returns:
        A tuple containing:
        - masked_text: String with email addresses replaced by |||EMAIL_ADDRESS|||
        - count: Number of email addresses that were masked
    """
    # Regular expression for email addresses
    # This pattern matches most common email formats including:
    # - Standard emails: user@domain.com
    # - Emails with dots, hyphens, underscores: user.name@sub-domain.co.uk
    # - Emails with plus signs: user+tag@domain.com
    # - Numeric domains: user@192.168.1.1
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    
    # Find all email addresses
    emails = re.findall(email_pattern, text)
    
    # Replace each email with the mask string
    masked_text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)
    
    # Return the masked text and count of emails found
    return masked_text, len(emails)


def mask_phone_numbers(text: str) -> Tuple[str, int]:
    """Mask phone numbers in text with |||PHONE_NUMBER|||.
    
    Args:
        text: Input string that may contain phone numbers
        
    Returns:
        A tuple containing:
        - masked_text: String with phone numbers replaced by |||PHONE_NUMBER|||
        - count: Number of phone numbers that were masked
    """
    # Regular expression patterns for common US phone number formats
    # This pattern matches:
    # - 10 consecutive digits: 2831823829
    # - (XXX)-XXX-XXXX: (283)-182-3829
    # - (XXX) XXX XXXX: (283) 182 3829
    # - XXX-XXX-XXXX: 283-182-3829
    # - XXX.XXX.XXXX: 283.182.3829
    # - XXX XXX XXXX: 283 182 3829
    # - 1-XXX-XXX-XXXX: 1-283-182-3829 (with country code)
    
    phone_patterns = [
        # 10 consecutive digits (word boundaries to avoid matching longer numbers)
        r'\b\d{10}\b',
        
        # (XXX)-XXX-XXXX format
        r'\(\d{3}\)-\d{3}-\d{4}',
        
        # (XXX) XXX XXXX format
        r'\(\d{3}\)\s+\d{3}\s+\d{4}',
        
        # XXX-XXX-XXXX format
        r'\b\d{3}-\d{3}-\d{4}\b',
        
        # XXX.XXX.XXXX format
        r'\b\d{3}\.\d{3}\.\d{4}\b',
        
        # XXX XXX XXXX format (with word boundaries)
        r'\b\d{3}\s+\d{3}\s+\d{4}\b',
        
        # 1-XXX-XXX-XXXX format (with country code)
        r'\b1-\d{3}-\d{3}-\d{4}\b',
        
        # +1-XXX-XXX-XXXX format (international)
        r'\+1-\d{3}-\d{3}-\d{4}\b',
        
        # (XXX) XXX-XXXX format (mixed parentheses and dash)
        r'\(\d{3}\)\s*\d{3}-\d{4}',
    ]
    
    # Combine all patterns with OR
    combined_pattern = '|'.join(f'({pattern})' for pattern in phone_patterns)
    
    # Find all phone numbers
    matches = re.findall(combined_pattern, text)
    # Count non-empty matches (since we use groups, some will be empty strings)
    count = sum(1 for match_groups in matches if any(group for group in match_groups))
    
    # Replace all phone numbers with the mask string
    masked_text = re.sub(combined_pattern, '|||PHONE_NUMBER|||', text)
    
    return masked_text, count


def mask_ip_addresses(text: str) -> Tuple[str, int]:
    """Mask IPv4 addresses in text with |||IP_ADDRESS|||.
    
    Args:
        text: Input string that may contain IPv4 addresses
        
    Returns:
        A tuple containing:
        - masked_text: String with IPv4 addresses replaced by |||IP_ADDRESS|||
        - count: Number of IPv4 addresses that were masked
    """
    # Regular expression for IPv4 addresses
    # This pattern matches valid IPv4 addresses where each octet is 0-255
    # We use word boundaries to ensure we match complete IP addresses
    # Pattern explanation:
    # - \b: word boundary at start
    # - (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?): matches 0-255
    #   - 25[0-5]: 250-255
    #   - 2[0-4][0-9]: 200-249  
    #   - [01]?[0-9][0-9]?: 0-199 (including single digits)
    # - \.: literal dot
    # - {3}: repeat the octet+dot pattern 3 times
    # - (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?): final octet (0-255)
    # - \b: word boundary at end
    
    ipv4_pattern = r'\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    
    # Find all IPv4 addresses
    ip_addresses = re.findall(ipv4_pattern, text)
    
    # Replace each IP address with the mask string
    masked_text = re.sub(ipv4_pattern, '|||IP_ADDRESS|||', text)
    
    # Return the masked text and count of IP addresses found
    return masked_text, len(ip_addresses)