"""
Regex-based sentence and word tokenizers for Hindi (and general Indic) text.
Handles: URLs, email addresses, dates, decimal/integer numbers, and punctuation as separate tokens. No NLTK/spaCy or other NLP tokenization libraries are used.
"""

import re

# ---------------------------------------------------------------------------
# Word-level patterns (matched in priority order inside word_tokenize)
# ---------------------------------------------------------------------------

# HTTP/HTTPS and www. URLs
_URL = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)

# Email addresses
_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)

# Dates: YYYY-MM-DD / YYYY/MM/DD  or  DD-MM-YYYY / DD/MM/YYYY / DD.MM.YYYY
_DATE = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
    r"|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b",
)

# Decimal numbers must be matched before plain integers
_DECIMAL = re.compile(r"\b\d+\.\d+\b")

# Standalone integers
_INTEGER = re.compile(r"\b\d+\b")

# Single punctuation character (Hindi danda U+0964 included)
_PUNCT = re.compile(r"[^\w\s\u0900-\u097F]", re.UNICODE)

# Hindi script words (consonants, vowels, vowel signs, virama — full block)
_HINDI_WORD = re.compile(r"[\u0900-\u097F]+")

# Latin-alphabet words (e.g. Rs, email local parts already handled separately)
_LATIN_WORD = re.compile(r"[a-zA-Z]+")

# Ordered list used by word_tokenize
_WORD_PATTERNS = (
    _URL,
    _EMAIL,
    _DATE,
    _DECIMAL,
    _INTEGER,
    _HINDI_WORD,
    _LATIN_WORD,
    _PUNCT,
)


def word_tokenize(text: str) -> list[str]:
    """
    Tokenize a single sentence (or line) into words and special tokens.
    Returns a list of token strings. Punctuation, URLs, emails, dates, and numbers (including decimals) each become their own token.
    """
    tokens: list[str] = []
    pos = 0
    length = len(text)

    while pos < length:
        # Skip whitespace between tokens
        if text[pos].isspace():
            pos += 1
            continue

        matched = False
        for pattern in _WORD_PATTERNS:
            m = pattern.match(text, pos)
            if m:
                tokens.append(m.group())
                pos = m.end()
                matched = True
                break

        if not matched:
            # Fallback: consume one non-whitespace character
            tokens.append(text[pos])
            pos += 1

    return tokens


# ---------------------------------------------------------------------------
# Sentence tokenizer
# ---------------------------------------------------------------------------

# Protect URLs, emails, and decimal numbers from being split at their dots
_PLACEHOLDER = "\x00"

def _protect(text: str) -> tuple[str, list[str]]:
    """Replace special spans with placeholders so dots inside them are safe."""
    saved: list[str] = []

    def _store(match: re.Match) -> str:
        saved.append(match.group())
        return _PLACEHOLDER * len(match.group())

    for pattern in (_URL, _EMAIL, _DECIMAL):
        text = pattern.sub(_store, text)
    return text, saved


def _restore(text: str, saved: list[str]) -> str:
    """Put protected spans back after sentence splitting."""
    for span in saved:
        text = text.replace(_PLACEHOLDER * len(span), span, 1)
    return text


# Split after sentence-ending punctuation (. ! ? Hindi danda)
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?।\u0964\u0965])\s+",
)


def sentence_tokenize(paragraph: str) -> list[str]:
    """
    Split a paragraph into sentence strings.
    Sentence boundaries are . ! ? and Hindi danda (।). Decimal points, URL dots, and email dots are not treated as boundaries.
    """
    paragraph = paragraph.strip()
    if not paragraph:
        return []

    protected, saved = _protect(paragraph)
    parts = _SENTENCE_BOUNDARY.split(protected)

    sentences: list[str] = []
    for part in parts:
        part = _restore(part.strip(), saved)
        if part:
            sentences.append(part)

    return sentences


def tokenize_paragraph(paragraph: str) -> list[list[str]]:
    """
    Tokenize a paragraph into sentences, then each sentence into words.
    Returns a list of sentences, where each sentence is a list of word tokens.
    """
    return [word_tokenize(s) for s in sentence_tokenize(paragraph)]
