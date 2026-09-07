"""Phase 10 tests: multilingual support."""

from __future__ import annotations

import pytest

from app.languages import (
    detect_language,
    normalize_multilingual_currency,
    normalize_multilingual_value,
)


def test_english_detection():
    result = detect_language("I am a 27 year old entrepreneur from Delhi.")
    assert result.language_code == "en"
    assert result.language_name == "English"
    assert result.original_text == "I am a 27 year old entrepreneur from Delhi."


def test_hindi_detection():
    result = detect_language("मैं दिल्ली से हूँ और मेरी आय 3.5 लाख है")
    assert result.language_code == "hi"


def test_bengali_detection():
    result = detect_language("আমি দিল্লি থেকে এসেছি")
    assert result.language_code == "bn"


def test_marathi_detection():
    result = detect_language("मी महाराष्ट्रातून आहे")
    assert result.language_code == "mr"


def test_telugu_detection():
    result = detect_language("నేను ఇంట్రప్రెన్యూర్")
    assert result.language_code == "te"


def test_tamil_detection():
    result = detect_language("நான் தொழிலதிபர்")
    assert result.language_code == "ta"


def test_gujarati_detection():
    result = detect_language("હું ઉદ્યોગસાહસિક છું")
    assert result.language_code == "gu"


def test_kannada_detection():
    result = detect_language("ನಾನು ಉದ್ಯಮಿ")
    assert result.language_code == "kn"


def test_malayalam_detection():
    result = detect_language("ഞാൻ സംരംഭകനാണ്")
    assert result.language_code == "ml"


def test_punjabi_detection():
    result = detect_language("ਮੈਂ ਉਦਯੋਗਪਤੀ ਹਾਂ")
    assert result.language_code == "pa"


def test_odia_detection():
    result = detect_language("ମୁଁ ଉଦ୍ୟୋଗୀ")
    assert result.language_code == "or"


def test_empty_text_rejected():
    with pytest.raises(ValueError, match="empty"):
        detect_language("")
