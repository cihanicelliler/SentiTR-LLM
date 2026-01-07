import re
import emoji
from bs4 import BeautifulSoup
import unicodedata

def normalize_turkish_chars(text: str) -> str:
    """
    Normalizes Turkish characters and handles case folding correctly.
    Specifically handles the dotted 'i' and dotless 'I' issue in Turkish.
    """
    # Custom case folding for Turkish
    charmap = {
        "I": "ı",
        "İ": "i",
    }
    for search, replace in charmap.items():
        text = text.replace(search, replace)
        
    return text.lower()

def remove_urls(text: str) -> str:
    """Removes URLs from text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_html_tags(text: str) -> str:
    """Removes HTML tags from text."""
    return BeautifulSoup(text, "html.parser").get_text()

def demojize_text(text: str, lang: str = 'en') -> str:
    """
    Converts emojis to text description. 
    Turkish support in `emoji` library might be limited, defaulting to English aliases 
    which might actually add noise if not translated, so for now we might want to just 
    remove them or keep them as is depending on the paper's specific method.
    The paper mentions 'handling emojis', often meaning replacing them via `emoji` lib.
    """
    return emoji.demojize(text) # e.g. 👍 -> :thumbs_up:

def clean_text(text: str) -> str:
    """
    Applies the full preprocessing pipeline:
    1. Remove HTML
    2. Remove URLs
    3. Normalize Turkish characters and lowercase
    4. Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
        
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = normalize_turkish_chars(text)
    text = demojize_text(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
