"""Multilingual support foundation.

Language handling is a provider abstraction.  A deterministic language
identification foundation supports English, Hindi, Bengali, Marathi,
Telugu, Tamil, Gujarati, Kannada, Malayalam, Punjabi, and Odia.

We never fake translation quality.  If a real translation provider is
not configured, safe fallback behavior is used.  User input retains the
original text, detected language, and a normalized/internal
representation where available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "mr": "Marathi",
    "te": "Telugu",
    "ta": "Tamil",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
}

# Unicode script ranges for deterministic language identification.
_DEVANAGARI = range(0x0900, 0x0980)  # Hindi, Marathi
_BENGALI = range(0x0980, 0x0A00)  # Bengali
_GURMUKHI = range(0x0A00, 0x0A80)  # Punjabi
_GUJARATI = range(0x0A80, 0x0B00)  # Gujarati
_ORIYA = range(0x0B00, 0x0B80)  # Odia
_TAMIL = range(0x0B80, 0x0C00)  # Tamil
_TELUGU = range(0x0C00, 0x0C80)  # Telugu
_KANNADA = range(0x0C80, 0x0D00)  # Kannada
_MALAYALAM = range(0x0D00, 0x0D80)  # Malayalam

# Common Hindi/Marathi words for disambiguation between Devanagari scripts.
_HINDI_MARKERS = {
    "है", "में", "का", "की", "के", "और", "नहीं", "मैं", "आप", "यह", "वह",
    "लाख", "करोड़", "महिला", "पुरुष", "दिल्ली", "अनुसूचित", "जाति",
}
_MARATHI_MARKERS = {
    "आहे", "आहेत", "मध्ये", "चा", "ची", "चे", "आणि", "नाही", "मी", "तू",
    "लाख", "कोटी", "स्त्री", "पुरुष", "दिल्ली", "अनुसूचित", "जात",
}

# Common vocabulary for deterministic value normalization.
GENDER_MAP = {
    "महिला": "female",
    "स्त्री": "female",
    "औरत": "female",
    "पुरुष": "male",
    "आदमी": "male",
    "মহিলা": "female",
    "পুরুষ": "male",
    "महिला": "female",
    "पुरुष": "male",
    "மகளிர்": "female",
    "ஆண்": "male",
    "మహిళ": "female",
    "పురుషుడు": "male",
    "મહિલા": "female",
    "પુરુષ": "male",
    "ಮಹಿಳೆ": "female",
    "ಪುರುಷ": "male",
    "സ്ത്രീ": "female",
    "പുരുഷൻ": "male",
    "ਔਰਤ": "female",
    "ਆਦਮੀ": "male",
    "ମହିଳା": "female",
    "ପୁରୁଷ": "male",
}

CATEGORY_MAP = {
    "अनुसूचित जाति": "SC",
    "अनुसूचित जनजाति": "ST",
    "अन्य पिछड़ा वर्ग": "OBC",
    "सामान्य": "General",
    "आर्थिक रूप से कमजोर वर्ग": "EWS",
    "তফসিলি জাতি": "SC",
    "তফসিলি উপজাতি": "ST",
    "অন্যান্য অনগ্রসর শ্রেণি": "OBC",
    "সাধারণ": "General",
    "अनुसूचित जाती": "SC",
    "अनुसूचित जमाती": "ST",
    "इतर मागास प्रवर्ग": "OBC",
    "सामान्य": "General",
    "షెడ్యూల్డ్ కులం": "SC",
    "షెడ్యూల్డ్ తెగ": "ST",
    "వెనుకబడిన తరగతి": "OBC",
    "సాధారణ": "General",
    "பட்டியல் சாதி": "SC",
    "பட்டியல் பழங்குடியினர்": "ST",
    "இதர பிற்படுத்தப்பட்ட வகுப்பு": "OBC",
    "பொது": "General",
    "અનુસૂચિત જાતિ": "SC",
    "અનુસૂચિત જનજાતિ": "ST",
    "અન્ય પછાત વર્ગ": "OBC",
    "સામાન્ય": "General",
    "ಪರಿಶಿಷ್ಟ ಜಾತಿ": "SC",
    "ಪರಿಶಿಷ್ಟ ಪಂಗಡ": "ST",
    "ಹಿಂದುಳಿದ ವರ್ಗ": "OBC",
    "ಸಾಮಾನ್ಯ": "General",
    "പട്ടിക ജാതി": "SC",
    "പട്ടിക വർഗ്ഗം": "ST",
    "പിന്നോക്ക വിഭാഗം": "OBC",
    "സാധാരണ": "General",
    "ਅਨੁਸੂਚਿਤ ਜਾਤੀ": "SC",
    "ਅਨੁਸੂਚਿਤ ਜਨਜਾਤੀ": "ST",
    "ਪੱਛੜਾ ਵਰਗ": "OBC",
    "ਆਮ": "General",
    "ଅନୁସୂଚିତ ଜାତି": "SC",
    "ଅନୁସୂଚିତ ଜନଜାତି": "ST",
    "ପଛୁଆ ବର୍ଗ": "OBC",
    "ସାଧାରଣ": "General",
}

STATE_MAP = {
    "दिल्ली": "Delhi",
    "महाराष्ट्र": "Maharashtra",
    "तमिलनाडु": "Tamil Nadu",
    "कर्नाटक": "Karnataka",
    "गुजरात": "Gujarat",
    "पश्चिम बंगाल": "West Bengal",
    "उत्तर प्रदेश": "Uttar Pradesh",
    "तेलंगाना": "Telangana",
    "आंध्र प्रदेश": "Andhra Pradesh",
    "দিল্লি": "Delhi",
    "মহারাষ্ট্র": "Maharashtra",
    "তামিলনাড়ু": "Tamil Nadu",
    "কর্ণাটক": "Karnataka",
    "গুজরাট": "Gujarat",
    "পশ্চিমবঙ্গ": "West Bengal",
    "উত্তরপ্রদেশ": "Uttar Pradesh",
    "तेलंगाणा": "Telangana",
    "आंध्र प्रदेश": "Andhra Pradesh",
    "दिल्ली": "Delhi",
    "महाराष्ट्र": "Maharashtra",
    "तमिळनाडू": "Tamil Nadu",
    "कर्नाटक": "Karnataka",
    "गुजरात": "Gujarat",
    "पश्चिम बंगाल": "West Bengal",
    "उत्तर प्रदेश": "Uttar Pradesh",
    "तेलंगाना": "Telangana",
    "ఆంధ్ర ప్రదేశ్": "Andhra Pradesh",
    "తెలంగాణ": "Telangana",
    "తమిళనాడు": "Tamil Nadu",
    "కర్ణాటక": "Karnataka",
    "గుజరాత్": "Gujarat",
    "పశ్చిమ బెంగాల్": "West Bengal",
    "ఉత్తర ప్రదేశ్": "Uttar Pradesh",
    "டெல்லி": "Delhi",
    "மகாராஷ்டிரா": "Maharashtra",
    "தமிழ்நாடு": "Tamil Nadu",
    "கர்நாடகா": "Karnataka",
    "குஜராத்": "Gujarat",
    "மேற்கு வங்காளம்": "West Bengal",
    "உத்தரப் பிரதேசம்": "Uttar Pradesh",
    "દિલ્હી": "Delhi",
    "મહારાષ્ટ્ર": "Maharashtra",
    "તમિલનાડુ": "Tamil Nadu",
    "કર્ણાટક": "Karnataka",
    "ગુજરાત": "Gujarat",
    "પશ્ચિમ બંગાળ": "West Bengal",
    "ઉત્તર પ્રદેશ": "Uttar Pradesh",
    "ದೆಹಲಿ": "Delhi",
    "ಮಹಾರಾಷ್ಟ್ರ": "Maharashtra",
    "ತಮಿಳುನಾಡು": "Tamil Nadu",
    "ಕರ್ನಾಟಕ": "Karnataka",
    "ಗುಜರಾತ್": "Gujarat",
    "ಪಶ್ಚಿಮ ಬಂಗಾಳ": "West Bengal",
    "ಉತ್ತರ ಪ್ರದೇಶ": "Uttar Pradesh",
    "ഡൽഹി": "Delhi",
    "മഹാരാഷ്ട്ര": "Maharashtra",
    "തമിഴ്നാട്": "Tamil Nadu",
    "കർണാടക": "Karnataka",
    "ഗുജറാത്": "Gujarat",
    "പശ്ചിമ ബംഗാൾ": "West Bengal",
    "ഉത്തർപ്രദേശ്": "Uttar Pradesh",
    "ਦਿੱਲੀ": "Delhi",
    "ਮਹਾਰਾਸ਼ਟਰ": "Maharashtra",
    "ਤਮਿਲਨਾਡੂ": "Tamil Nadu",
    "ਕਰਨਾਟਕ": "Karnataka",
    "ਗੁਜਰਾਤ": "Gujarat",
    "ਪੱਛਮੀ ਬੰਗਾਲ": "West Bengal",
    "ਉੱਤਰ ਪ੍ਰਦੇਸ਼": "Uttar Pradesh",
    "ଦିଲ୍ଲୀ": "Delhi",
    "ମହାରାଷ୍ଟ୍ର": "Maharashtra",
    "ତାମିଲନାଡୁ": "Tamil Nadu",
    "କର୍ଣ୍ଣାଟକ": "Karnataka",
    "ଗୁଜରାଟ": "Gujarat",
    "ପଶ୍ଚିମ ବଙ୍ଗଳ": "West Bengal",
    "ଉତ୍ତର ପ୍ରଦେଶ": "Uttar Pradesh",
}

# Currency words in Indian languages.
CURRENCY_WORDS = {
    "लाख": 100000,
    "लाखों": 100000,
    "करोड़": 10000000,
    "करोड़ों": 10000000,
    "লাখ": 100000,
    "কোটি": 10000000,
    "लाख": 100000,
    "कोटी": 10000000,
    "లక్ష": 100000,
    "కోటి": 10000000,
    "லட்சம்": 100000,
    "கோடி": 10000000,
    "લાખ": 100000,
    "કરોડ": 10000000,
    "ಲಕ್ಷ": 100000,
    "ಕೋಟಿ": 10000000,
    "ലക്ഷം": 100000,
    "കോടി": 10000000,
    "ਲੱਖ": 100000,
    "ਕਰੋੜ": 10000000,
    "ଲକ୍ଷ": 100000,
    "କୋଟି": 10000000,
}


@dataclass(frozen=True)
class LanguageResult:
    language_code: str
    language_name: str
    confidence: Literal["high", "medium", "low"]
    original_text: str
    normalized_text: str


class LanguageDetector(Protocol):
    def detect(self, text: str) -> LanguageResult:
        ...


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        code = ord(char)
        if code in _DEVANAGARI:
            counts["devanagari"] = counts.get("devanagari", 0) + 1
        elif code in _BENGALI:
            counts["bengali"] = counts.get("bengali", 0) + 1
        elif code in _GURMUKHI:
            counts["gurmukhi"] = counts.get("gurmukhi", 0) + 1
        elif code in _GUJARATI:
            counts["gujarati"] = counts.get("gujarati", 0) + 1
        elif code in _ORIYA:
            counts["oriya"] = counts.get("oriya", 0) + 1
        elif code in _TAMIL:
            counts["tamil"] = counts.get("tamil", 0) + 1
        elif code in _TELUGU:
            counts["telugu"] = counts.get("telugu", 0) + 1
        elif code in _KANNADA:
            counts["kannada"] = counts.get("kannada", 0) + 1
        elif code in _MALAYALAM:
            counts["malayalam"] = counts.get("malayalam", 0) + 1
    return counts


def _disambiguate_devanagari(text: str) -> str:
    """Distinguish Hindi from Marathi using common marker words."""
    hindi_hits = sum(1 for word in _HINDI_MARKERS if word in text)
    marathi_hits = sum(1 for word in _MARATHI_MARKERS if word in text)
    if marathi_hits > hindi_hits:
        return "mr"
    return "hi"


def detect_language(text: str) -> LanguageResult:
    """Deterministically detect the language of a text.

    Uses Unicode script ranges.  English is detected when the text is
    predominantly Latin script.  Devanagari is disambiguated between
    Hindi and Marathi using common marker words.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")

    counts = _script_counts(text)
    if not counts:
        # No non-Latin script characters: treat as English.
        return LanguageResult(
            language_code="en",
            language_name="English",
            confidence="high",
            original_text=text,
            normalized_text=text.strip(),
        )

    dominant = max(counts, key=counts.get)
    script_to_language = {
        "bengali": "bn",
        "gurmukhi": "pa",
        "gujarati": "gu",
        "oriya": "or",
        "tamil": "ta",
        "telugu": "te",
        "kannada": "kn",
        "malayalam": "ml",
    }
    if dominant == "devanagari":
        code = _disambiguate_devanagari(text)
    else:
        code = script_to_language[dominant]

    return LanguageResult(
        language_code=code,
        language_name=SUPPORTED_LANGUAGES[code],
        confidence="high",
        original_text=text,
        normalized_text=text.strip(),
    )


def normalize_multilingual_value(text: str) -> str | None:
    """Normalize a multilingual value to a canonical English value.

    Returns None when the value is ambiguous or unknown.  Never invents
    translations for ambiguous text.
    """
    if not text:
        return None
    stripped = text.strip()
    if stripped in GENDER_MAP:
        return GENDER_MAP[stripped]
    if stripped in CATEGORY_MAP:
        return CATEGORY_MAP[stripped]
    if stripped in STATE_MAP:
        return STATE_MAP[stripped]
    return None


def normalize_multilingual_currency(text: str) -> float | None:
    """Normalize a multilingual currency expression to a numeric value.

    Supports expressions like ``3.5 लाख`` or ``5 कोटी``.  Returns None
    when the expression is ambiguous or unsupported.
    """
    if not text:
        return None
    stripped = text.strip()
    # Match a number followed by a currency word.
    for word, multiplier in CURRENCY_WORDS.items():
        pattern = re.compile(rf"(\d+(?:\.\d+)?)\s*{re.escape(word)}")
        match = pattern.search(stripped)
        if match:
            try:
                return float(match.group(1)) * multiplier
            except ValueError:
                return None
    return None