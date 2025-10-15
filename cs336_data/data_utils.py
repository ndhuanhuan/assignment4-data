from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

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