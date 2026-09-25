from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


# Characters commonly used in Urdu but not standard Arabic.
URDU_SPECIFIC_CHARACTERS = set(
    "پ چ ژ گ ڑ ڈ ں ٹ ٹھ ڈھ ڑھ ھ ے"
)


def detect_language(text):
    """
    Detect the main language of the supplied study material.

    Returns:
        english
        arabic
        urdu
        mixed
        unknown
    """

    if not text or not text.strip():
        return "unknown"

    arabic_script = 0
    latin_characters = 0
    urdu_specific = 0

    for character in text:

        # Arabic-script characters
        if "\u0600" <= character <= "\u06FF":
            arabic_script += 1

            if character in URDU_SPECIFIC_CHARACTERS:
                urdu_specific += 1

        # English / Latin characters
        elif "A" <= character <= "Z" or "a" <= character <= "z":
            latin_characters += 1

    total_letters = arabic_script + latin_characters

    if total_letters == 0:
        return "unknown"

    arabic_ratio = arabic_script / total_letters
    english_ratio = latin_characters / total_letters

    # --------------------------------------------------
    # Mixed Arabic/Urdu + English
    # --------------------------------------------------
    if arabic_ratio > 0.20 and english_ratio > 0.20:
        return "mixed"

    # --------------------------------------------------
    # Mostly Arabic-script material
    # --------------------------------------------------
    if arabic_ratio > 0.50:

        # Strong Urdu-specific characters
        if urdu_specific > 0:
            return "urdu"

        # Use langdetect as an additional check
        try:
            detected = detect(text)

            if detected == "ur":
                return "urdu"

            if detected == "ar":
                return "arabic"

        except Exception:
            pass

        # If there are no Urdu-specific characters,
        # treat Arabic-script material as Arabic.
        return "arabic"

    # --------------------------------------------------
    # Mostly English material
    # --------------------------------------------------
    if english_ratio > 0.50:

        try:
            detected = detect(text)

            if detected == "en":
                return "english"

        except Exception:
            pass

        return "english"

    return "unknown"