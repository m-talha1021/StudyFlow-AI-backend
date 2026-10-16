import os
import io
import re
import json
import urllib.request
import urllib.error

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from google import genai
from openai import OpenAI
from google.genai import types

from document_parser import extract_text
from image_processor import extract_image_text
from language_detector import detect_language

from rag import (
    chunk_text,
    create_embeddings,
    retrieve_chunks,
    build_context
)

from pdf_generator import generate_pdf


# ============================================================
# ENVIRONMENT
# ============================================================

# ============================================================
# API KEYS
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Please add it to your environment variables."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# OPENAI CLIENT
# ============================================================

openai_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY
    )


# ============================================================
# GEMINI MODELS
# ============================================================

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash"
]


# ============================================================
# OPENAI MODELS
# ============================================================

OPENAI_MODELS = [
    "gpt-6-luna",
    "gpt-5.6-luna",
    "gpt-5.4-mini",
    "gpt-5.1",
    "gpt-4.1-mini",
]

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# CORS / VERCEL FRONTEND
# ============================================================

# Production frontend plus Vercel preview deployments.
# You can override these with FRONTEND_ORIGINS in Vercel, using
# comma-separated origins, for example:
# https://study-flow-ai-lac.vercel.app,http://localhost:5173
FRONTEND_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "https://study-flow-ai-lac.vercel.app,http://localhost:5173"
    ).split(",")
    if origin.strip()
}

# The project's own Vercel production, preview and branch URLs all share
# the "study-flow-" prefix. This keeps preview deployments working without
# opening CORS to every unrelated vercel.app app.
VERCEL_PREVIEW_ORIGIN_PATTERN = re.compile(
    r"^https://study-flow-[a-z0-9-]+\.vercel\.app$",
    re.IGNORECASE
)

# Local development servers run on a variety of ports.
LOCALHOST_ORIGIN_PATTERN = re.compile(
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    re.IGNORECASE
)


def is_allowed_origin(origin):
    if not origin:
        return False

    normalized = origin.rstrip("/")

    if normalized in FRONTEND_ORIGINS:
        return True

    if LOCALHOST_ORIGIN_PATTERN.fullmatch(normalized):
        return True

    return bool(VERCEL_PREVIEW_ORIGIN_PATTERN.fullmatch(normalized))


CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                *FRONTEND_ORIGINS,
                LOCALHOST_ORIGIN_PATTERN,
                VERCEL_PREVIEW_ORIGIN_PATTERN
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Disposition"],
        }
    },
    supports_credentials=False,
)


@app.before_request
def handle_cors_preflight():
    # Explicitly answer browser preflight requests.
    if request.method == "OPTIONS":
        return ("", 204)


@app.after_request
def add_cors_headers(response):
    # Also add CORS headers to error responses so the browser can
    # display the real backend error instead of hiding it as CORS.
    origin = request.headers.get("Origin")

    if is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin.rstrip("/")
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization"
        )
        response.headers["Access-Control-Expose-Headers"] = (
            "Content-Disposition"
        )

    return response


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "pdf",
    "docx",
    "pptx",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "heic",
    "heif"
}


IMAGE_MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif"
}


MAX_IMAGE_SIZE = 15 * 1024 * 1024


# ============================================================
# TEMPORARY MATERIAL STORE
# ============================================================

# Active AI oral-exam sessions.
oral_exam_sessions = {}

material_store = {
    "text": "",
    "language": "unknown",
    "chunks": [],
    "embeddings": []
}


# ============================================================
# LANGUAGE DETECTION HELPERS
# ============================================================

def count_script_characters(text):
    """
    Count Arabic-family and Latin characters.

    Arabic-family characters include Arabic, Urdu,
    Persian and related Unicode blocks.
    """

    arabic_count = len(
        re.findall(
            r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
            r"\uFB50-\uFDFF\uFE70-\uFEFF]",
            text
        )
    )

    latin_count = len(
        re.findall(
            r"[A-Za-z]",
            text
        )
    )

    return arabic_count, latin_count


def normalize_detected_language(language, text):
    """
    Make language detection more reliable.
    """

    detected = (
        language or ""
    ).lower().strip()

    arabic_count, latin_count = (
        count_script_characters(text)
    )

    total_script = (
        arabic_count +
        latin_count
    )


    # --------------------------------------------------------
    # Strong Arabic-script material
    # --------------------------------------------------------

    if total_script > 0:

        arabic_ratio = (
            arabic_count /
            total_script
        )

        if arabic_ratio >= 0.85:

            # Urdu-specific characters
            urdu_specific = len(
                re.findall(
                    r"[ٹڈڑںھہﮯےژچگپڤ]",
                    text
                )
            )

            if urdu_specific >= 2:
                return "urdu"

            return "arabic"


    # --------------------------------------------------------
    # Respect existing detector
    # --------------------------------------------------------

    if detected in {
        "arabic",
        "urdu",
        "english",
        "mixed"
    }:

        return detected


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if arabic_count > latin_count:
        return "arabic"

    if latin_count > 0:
        return "english"

    return "unknown"


# ============================================================
# CLEAN GEMINI MARKDOWN
# ============================================================

def clean_generated_result(text):

    """
    Remove Markdown formatting while preserving
    Arabic and Urdu Unicode characters.
    """

    if not text:
        return ""


    # Bold

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
        flags=re.DOTALL
    )


    # Italic

    text = re.sub(
        r"(?<!\*)(\*)(?!\s)(.*?)(?<!\s)\*(?!\*)",
        r"\2",
        text
    )


    # Underscore bold

    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text,
        flags=re.DOTALL
    )


    # Headings

    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE
    )


    # Bullet points

    text = re.sub(
        r"^\s*\*\s+",
        "• ",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"^\s*-\s+",
        "• ",
        text,
        flags=re.MULTILINE
    )


    # Numbered lists

    text = re.sub(
        r"^\s*(\d+)\.\s+",
        r"\1. ",
        text,
        flags=re.MULTILINE
    )


    # Inline code

    text = text.replace(
        "`",
        ""
    )


    # Horizontal lines

    text = re.sub(
        r"^\s*([-*_]){3,}\s*$",
        "",
        text,
        flags=re.MULTILINE
    )


    # Trailing spaces

    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text
    )


    # Excessive blank lines

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text.strip()


# ============================================================
# AI TEXT GENERATION WITH AUTOMATIC FALLBACK
# ============================================================

def generate_ai_response(
    prompt,
    system_instruction=None,
    temperature=0.2
):
    """
    Generate a text response using Gemini first.

    If every configured Gemini model fails, automatically try
    the configured OpenAI models in order.

    Returns a dictionary containing the generated text, provider
    and model that actually produced the response.
    """

    last_error = None

    # --------------------------------------------------------
    # 1. TRY GEMINI MODELS
    # --------------------------------------------------------

    for model_name in GEMINI_MODELS:

        try:

            print(
                f"[AI] Trying Gemini model: {model_name}"
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature
                )
            )

            if response and response.text:

                print(
                    f"[AI] Gemini succeeded: {model_name}"
                )

                return {
                    "text": response.text,
                    "provider": "Gemini",
                    "model": model_name
                }

        except Exception as error:

            last_error = error

            print(
                f"[AI] Gemini failed: {model_name}"
            )
            print(error)

            continue

    # --------------------------------------------------------
    # 2. GEMINI FAILED → TRY OPENAI MODELS
    # --------------------------------------------------------

    if openai_client is not None:

        for model_name in OPENAI_MODELS:

            try:

                print(
                    f"[AI] Trying OpenAI model: {model_name}"
                )

                request_args = {
                    "model": model_name,
                    "input": prompt
                }

                if system_instruction:
                    request_args["instructions"] = system_instruction

                response = openai_client.responses.create(
                    **request_args
                )

                if response and response.output_text:

                    print(
                        f"[AI] OpenAI succeeded: {model_name}"
                    )

                    return {
                        "text": response.output_text,
                        "provider": "OpenAI",
                        "model": model_name
                    }

            except Exception as error:

                last_error = error

                print(
                    f"[AI] OpenAI failed: {model_name}"
                )
                print(error)

                continue

    else:

        print(
            "[AI] OPENAI_API_KEY is not configured; "
            "skipping OpenAI fallback."
        )

    # --------------------------------------------------------
    # 3. ALL TEXT MODELS FAILED
    # --------------------------------------------------------

    if last_error:
        raise last_error

    raise RuntimeError(
        "All configured AI models are currently unavailable."
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "success": True,
        "message":
            "StudyFlow AI backend is running!"
    })


# ============================================================
# API TEST
# ============================================================

@app.route("/api/test")
def test_api():

    return jsonify({
        "success": True,
        "message":
            "Hello from the StudyFlow AI backend!"
    })


# ============================================================
# AUDIO / VIDEO TRANSCRIPTION
# ============================================================

TRANSCRIPTION_ALLOWED_EXTENSIONS = {
    "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg", "flac"
}
# OpenAI transcription models, used as a fallback when Gemini is unavailable.
TRANSCRIPTION_MODELS = ["gpt-4o-mini-transcribe", "whisper-1"]
AUDIO_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "mpga": "audio/mpeg",
    "mp4": "video/mp4",
    "mpeg": "video/mpeg",
    "m4a": "audio/mp4",
    "wav": "audio/wav",
    "webm": "audio/webm",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}
MAX_TRANSCRIPTION_UPLOAD_SIZE = 4 * 1024 * 1024  # Per-request cap: Vercel rejects bodies over 4.5 MB. The frontend chunks larger lectures into small segments.

# A verbatim speech-to-text engine must never translate or romanize.
TRANSCRIPTION_SYSTEM_INSTRUCTION = (
    "You are a verbatim speech-to-text engine. You always transcribe speech "
    "in the exact language that was spoken and you always write it in that "
    "language's native script. You never translate, romanize, summarize, "
    "paraphrase or comment. You ignore all non-speech sounds."
)

# Native-script hints keep languages such as Urdu out of Latin transliteration.
NATIVE_SCRIPT_HINTS = {
    "urdu": "the Urdu (Arabic) script, for example اردو",
    "arabic": "the Arabic script, for example العربية",
    "persian": "the Persian (Arabic) script, for example فارسی",
    "pashto": "the Pashto (Arabic) script",
    "sindhi": "the Sindhi (Arabic) script",
    "punjabi": "the Shahmukhi (Arabic) or Gurmukhi script",
    "dari": "the Dari (Arabic) script",
    "hindi": "the Devanagari script, for example हिन्दी",
    "bengali": "the Bengali script",
    "russian": "the Cyrillic script",
    "ukrainian": "the Cyrillic script",
    "hebrew": "the Hebrew script",
    "greek": "the Greek script",
    "thai": "the Thai script",
    "tamil": "the Tamil script",
    "telugu": "the Telugu script",
    "chinese": "Chinese characters",
    "japanese": "Japanese script",
    "korean": "Hangul",
}


def build_transcription_prompt(language_hint=None):
    """
    Build a strict transcription instruction.

    The prompt forces the original spoken language, forbids translation and
    romanization, and tells the model to discard non-speech sounds such as
    background music so the transcript never mentions them.
    """
    lines = [
        "Transcribe the spoken words in this audio verbatim.",
        "Rules:",
        "- Output ONLY the transcript text.",
        "- Keep the EXACT original spoken language. Never translate into English or any other language.",
        "- Never romanize or transliterate: write each word using the native script of the spoken language.",
        "- Do not add headings, notes, translations, timestamps, or commentary.",
        "- Ignore background music, humming, instrumental audio and every other non-speech sound.",
        "  Never write labels such as [Music], (music), notes like the musical symbols, "
        "[Applause], [Laughter] or [Silence], and never describe sounds.",
        "- If there is no speech in the segment, return an empty response.",
    ]

    if language_hint:
        script = NATIVE_SCRIPT_HINTS.get(language_hint)
        if script:
            lines.insert(
                1,
                f"The speech is most likely in {language_hint}; if so, write it in {script}. Do not translate.",
            )
        else:
            lines.insert(
                1,
                f"The speech is most likely in {language_hint}; if so, write it in {language_hint}. Do not translate.",
            )

    return "\n".join(lines)


# Spoken-language detection is used to lock the transcript language. Without
# it a model can "helpfully" translate a non-English lecture into English.
AUDIO_LANGUAGE_PROMPT = (
    "Identify the language being spoken in this audio. "
    "Reply with ONLY the English name of that language in lowercase, "
    "for example: urdu, arabic, english, hindi, punjabi, persian, pashto, "
    "sindhi, spanish, french, bengali, turkish. "
    "If the audio contains no speech at all, reply with: none"
)

NO_SPEECH_TOKENS = {"none", "no", "silence", "music", "unknown", "noise", "instrumental"}

# Common spoken-language names, used to parse the model's detection reply.
KNOWN_SPOKEN_LANGUAGES = set(NATIVE_SCRIPT_HINTS) | {
    "english", "spanish", "french", "german", "italian", "portuguese",
    "dutch", "turkish", "indonesian", "malay", "swahili", "vietnamese",
    "polish", "romanian", "czech", "swedish", "norwegian", "danish",
    "finnish", "kannada", "malayalam", "marathi", "gujarati", "odia",
}

# Words that frequently appear in a detection sentence but are not languages.
LANGUAGE_REPLY_STOPWORDS = {
    "the", "language", "is", "this", "audio", "spoken", "in", "a", "an",
    "detected", "it", "appears", "to", "be", "speech", "contains", "no",
    "i", "cannot", "determine", "sounds", "like", "with", "and", "only",
    "clip", "recording", "voice", "speaker", "says", "of", "at",
}

# Markers such as [Music], (music), musical-note symbols or standalone sound
# descriptions are removed so the transcript never mentions music or noise.
NON_SPEECH_PATTERNS = [
    re.compile(
        r"\[\s*(?:music|musique|instrumental|instrumental music|background music|"
        r"applause|clapping|cheering|laughter|laughs|chuckles|sighs|coughing|cough|"
        r"sneezes|sniffing|silence|inaudible|indistinct|noise|noises|background|"
        r"beep|beeps|intro|outro|sound|sounds|sound effect|sound effects|audio|"
        r"birds|wind|rain|footsteps|door|phone)[^\]]*\]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\(\s*(?:music|musique|instrumental|background music|applause|clapping|"
        r"cheering|laughter|laughs|silence|inaudible|noise|background|sound effect|"
        r"sound effects)[^\)]*\)",
        re.IGNORECASE,
    ),
    re.compile(r"[♪♫♬♩🎵🎶]+"),
    re.compile(
        r"^\s*(?:music|musique|instrumental|applause|laughter|silence|inaudible|"
        r"background music|sound effect)\s*[.!]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def clean_transcript_text(text):
    """Remove non-speech markers (especially music) and tidy whitespace."""
    if not text:
        return ""

    cleaned = text
    for pattern in NON_SPEECH_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    cleaned = re.sub(r"[ \t\u00A0]{2,}", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)

    return cleaned.strip()


# Languages that must be written in a non-Latin script. When the model
# romanizes these (for example writing Urdu in Latin letters) the transcript
# is repaired by rewriting it into the native script.
NATIVE_SCRIPT_LANGUAGES = set(NATIVE_SCRIPT_HINTS)


def is_latin_dominant(text):
    """True when a transcript is written mostly with Latin letters."""
    arabic_count, latin_count = count_script_characters(text)
    other_script_count = len(
        re.findall(
            r"[\u0900-\u097F\u0980-\u09FF\u0400-\u04FF\u0590-\u05FF"
            r"\u0370-\u03FF\u0E00-\u0E7F\u0B80-\u0BFF\u0C00-\u0C7F"
            r"\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]",
            text
        )
    )
    total_letters = arabic_count + latin_count + other_script_count
    if total_letters == 0:
        return False
    return latin_count > 0 and latin_count >= total_letters * 0.6


def rewrite_transcript_to_native_script(text, language):
    """
    Repair a romanized/translated transcript by rewriting it into the native
    script of the spoken language. Falls back to the original text on failure.
    """
    if not text or language not in NATIVE_SCRIPT_LANGUAGES:
        return text

    if not is_latin_dominant(text):
        return text

    script = NATIVE_SCRIPT_HINTS.get(language, language)
    system_instruction = (
        "You are a verbatim transcriber. You convert text into the native "
        "script of its language without changing a single word, adding any "
        "word, translating, or adding labels."
    )
    prompt = (
        f"The text below is spoken {language} written with Latin letters "
        "(transliterated / Roman script).\n\n"
        f"Rewrite it EXACTLY into {script}, keeping every word and its order.\n"
        "Rules: Do not translate. Do not add or remove any word. "
        "Do not add any label, note or sound description. "
        "Return only the rewritten text.\n\n"
        f"{text}"
    )

    try:
        result = generate_ai_response(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0
        )
        rewritten = clean_transcript_text((result.get("text") or "").strip())
        if rewritten and not is_latin_dominant(rewritten):
            return rewritten
    except Exception as error:
        print("[Transcription] Native-script repair failed:", error)

    return text


def resolve_audio_mime(filename, fallback=None):
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return AUDIO_MIME_TYPES.get(extension, fallback or "application/octet-stream")


def detect_audio_language(file_bytes, mime_type):
    """
    Ask the AI to identify the spoken language before transcribing.

    Returns a lowercase language name, or an empty string when the language
    cannot be determined (for example when the clip holds only music).
    """
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    AUDIO_LANGUAGE_PROMPT,
                ],
                config=types.GenerateContentConfig(temperature=0),
            )
            raw = (getattr(response, "text", "") or "").strip().lower()
            tokens = re.findall(r"[a-z]+", raw)
            if not tokens:
                continue

            # "no speech" / "none" / "music only" mean there is nothing to transcribe.
            if tokens == ["no"] or "none" in tokens or "silence" in tokens:
                return ""
            if "no" in tokens and "speech" in tokens:
                return ""

            name = ""
            for token in reversed(tokens):
                if token in KNOWN_SPOKEN_LANGUAGES:
                    name = token
                    break
            if not name:
                candidates = [
                    token for token in tokens
                    if token not in LANGUAGE_REPLY_STOPWORDS
                    and token not in NO_SPEECH_TOKENS
                ]
                if len(candidates) == 1:
                    name = candidates[0]

            if not name or name in NO_SPEECH_TOKENS:
                return ""
            return name
        except Exception as error:
            print(f"[Transcription] Language detection failed ({model_name}):", error)

    return ""


def transcribe_with_gemini(file_bytes, mime_type, language_hint=None):
    """Transcribe audio using the Gemini client, which is always configured."""
    last_error = None
    prompt_text = build_transcription_prompt(language_hint)

    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    prompt_text,
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    system_instruction=TRANSCRIPTION_SYSTEM_INSTRUCTION,
                ),
            )
            candidate = (getattr(response, "text", "") or "").strip()
            if candidate:
                return candidate
        except Exception as error:
            last_error = error
            print(f"[Transcription] Gemini model {model_name} failed:", error)

    if last_error:
        raise last_error
    return ""


def transcribe_with_openai(file_bytes, filename, mime_type, language_hint=None):
    """Transcribe audio using OpenAI as a secondary provider."""
    last_error = None
    prompt_text = build_transcription_prompt(language_hint)

    for model_name in TRANSCRIPTION_MODELS:
        try:
            transcription = openai_client.audio.transcriptions.create(
                model=model_name,
                file=(filename, io.BytesIO(file_bytes), mime_type),
                prompt=prompt_text,
            )
            candidate = (getattr(transcription, "text", "") or "").strip()
            if candidate:
                return candidate
        except Exception as error:
            last_error = error
            print(f"[Transcription] OpenAI model {model_name} failed:", error)

    if last_error:
        raise last_error
    return ""


@app.route("/api/transcribe", methods=["POST", "OPTIONS"])
def transcribe_lecture():
    """Transcribe speech, then reuse the existing text language detector."""
    if request.method == "OPTIONS":
        return ("", 204)

    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"success": False, "error": "Choose an audio or video file first."}), 400

    filename = os.path.basename(uploaded_file.filename)
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in TRANSCRIPTION_ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": "Unsupported format. Use MP3, MP4, MPEG, M4A, WAV, WebM, OGG or FLAC."
        }), 400

    try:
        file_bytes = uploaded_file.read(MAX_TRANSCRIPTION_UPLOAD_SIZE + 1)
        if not file_bytes:
            return jsonify({"success": False, "error": "The selected file is empty."}), 400
        if len(file_bytes) > MAX_TRANSCRIPTION_UPLOAD_SIZE:
            return jsonify({
                "success": False,
                "error": "This upload segment is too large. Uploads are split automatically; please try again."
            }), 413

        mime_type = resolve_audio_mime(filename, uploaded_file.mimetype)
        transcript_text = ""
        provider = None
        errors = []

        # The client sends the language it already detected on the first
        # segment so later segments stay locked to the same language.
        language_hint = (request.form.get("language") or "").strip().lower()
        language_hint = re.sub(r"[^a-z ]", "", language_hint).strip()
        if language_hint in NO_SPEECH_TOKENS or language_hint == "mixed":
            language_hint = ""

        # Detect the spoken language up front so a non-English lecture is
        # never silently translated into English.
        audio_language = language_hint
        if not audio_language:
            try:
                audio_language = detect_audio_language(file_bytes, mime_type)
            except Exception as error:
                print("[Transcription] Audio language detection error:", error)
                audio_language = ""

        hint = audio_language or None

        # Gemini is the app's primary AI provider and GEMINI_API_KEY is
        # required to boot, so try it first. Fall back to OpenAI, which is
        # optional, if Gemini cannot handle the clip.
        try:
            transcript_text = transcribe_with_gemini(file_bytes, mime_type, hint)
            if transcript_text:
                provider = "Gemini"
        except Exception as error:
            errors.append(f"Gemini: {error}")

        if not transcript_text and openai_client is not None:
            try:
                transcript_text = transcribe_with_openai(file_bytes, filename, mime_type, hint)
                if transcript_text:
                    provider = "OpenAI"
            except Exception as error:
                errors.append(f"OpenAI: {error}")

        if not transcript_text:
            if errors:
                return jsonify({
                    "success": False,
                    "error": "Transcription failed on the server. Check the Vercel function logs for details.",
                    "details": " | ".join(errors)[:400]
                }), 500

            return jsonify({
                "success": False,
                "error": "No speech could be transcribed from this file."
            }), 422

        # Drop non-speech markers such as [Music] so music is never reported.
        transcript_text = clean_transcript_text(transcript_text)

        if not transcript_text:
            return jsonify({
                "success": False,
                "error": "No speech could be transcribed from this file."
            }), 422

        # A model sometimes romanizes or translates non-Latin speech (for
        # example writing Urdu in Latin letters). Rewrite it into the native
        # script of the language that was actually spoken.
        if audio_language in NATIVE_SCRIPT_LANGUAGES:
            transcript_text = rewrite_transcript_to_native_script(
                transcript_text, audio_language
            )

        # Reuse the same language detector and normalizer as study materials.
        detected_language = detect_language(transcript_text)
        language = normalize_detected_language(detected_language, transcript_text)

        # If the spoken language uses a non-Latin script (Urdu, Arabic, Hindi,
        # ...) but the detector could not read the script, trust the audio
        # detection so a translated or romanized transcript is still labelled
        # in its true language.
        if audio_language and audio_language in NATIVE_SCRIPT_HINTS and language in {"english", "unknown"}:
            language = audio_language
        elif language == "unknown" and audio_language:
            language = audio_language

        return jsonify({
            "success": True,
            "filename": filename,
            "provider": provider,
            "language": language,
            "transcript": transcript_text,
            "characters": len(transcript_text),
            "translated": False
        })
    except Exception as error:
        print("Transcription error:", error)
        return jsonify({
            "success": False,
            "error": "Could not transcribe this file.",
            "details": str(error)[:400]
        }), 500


# ============================================================
# AUTH / CORS DIAGNOSTIC
# ============================================================

@app.route("/api/auth/status", methods=["GET", "OPTIONS"])
def auth_status():
    """
    Authentication is handled by Firebase on the frontend.
    This endpoint is intentionally diagnostic only; it does not
    create or store passwords.
    """

    if request.method == "OPTIONS":
        return ("", 204)

    return jsonify({
        "success": True,
        "authentication": "Firebase Authentication",
        "backend_auth_required": False,
        "message": "Login and signup are handled by Firebase on the frontend."
    })


# ============================================================
# PROCESS STUDY MATERIAL
# ============================================================

@app.route(
    "/api/material",
    methods=["POST"]
)
def process_material():

    try:

        text = ""
        filenames = []
        source = None

        # ====================================================
        # MULTIPLE FILE UPLOAD
        # ====================================================

        uploaded_files = request.files.getlist("files")

        # Backward compatibility with the previous single-file
        # frontend, which used the field name "file".
        if not uploaded_files and "file" in request.files:
            uploaded_files = [request.files["file"]]

        if uploaded_files:

            if len(uploaded_files) > 10:
                return jsonify({
                    "success": False,
                    "error": "Please upload a maximum of 10 files at a time."
                }), 400

            extracted_parts = []

            for file in uploaded_files:

                if not file or file.filename == "":
                    continue

                filename = file.filename
                filenames.append(filename)

                extension = filename.rsplit(
                    ".",
                    1
                )[-1].lower()

                if extension not in ALLOWED_EXTENSIONS:
                    return jsonify({
                        "success": False,
                        "error": (
                            f"Unsupported file: {filename}. "
                            "Supported files are PDF, DOCX, PPTX, PNG, JPG, "
                            "JPEG, WEBP, HEIC and HEIF."
                        )
                    }), 400

                file_bytes = file.read()

                if not file_bytes:
                    return jsonify({
                        "success": False,
                        "error": f"The uploaded file is empty: {filename}."
                    }), 400

                # ------------------------------------------------
                # IMAGE
                # ------------------------------------------------

                if extension in IMAGE_MIME_TYPES:

                    if len(file_bytes) > MAX_IMAGE_SIZE:
                        return jsonify({
                            "success": False,
                            "error": (
                                f"Image '{filename}' is too large. "
                                "Please upload an image smaller than 15 MB."
                            )
                        }), 400

                    mime_type = IMAGE_MIME_TYPES[extension]

                    print(f"Processing image: {filename}")

                    image_text = ""
                    last_image_error = None

                    for model_name in GEMINI_MODELS:
                        try:
                            print(
                                f"Trying image model: {model_name} "
                                f"for {filename}"
                            )

                            image_text = extract_image_text(
                                image_bytes=file_bytes,
                                mime_type=mime_type,
                                client=client,
                                model_name=model_name
                            )

                            if image_text:
                                print(
                                    "Image processing succeeded with: "
                                    f"{model_name} for {filename}"
                                )
                                break

                        except Exception as error:
                            last_image_error = error
                            print(
                                f"{model_name} failed for image "
                                f"{filename}:"
                            )
                            print(error)

                    if not image_text:
                        if last_image_error:
                            raise last_image_error
                        raise RuntimeError(
                            f"Could not extract readable study content "
                            f"from image: {filename}."
                        )

                    extracted_parts.append(
                        f"\n\n===== {filename} =====\n\n{image_text.strip()}"
                    )

                # ------------------------------------------------
                # PDF / DOCX / PPTX
                # ------------------------------------------------

                else:
                    extracted_text = extract_text(
                        file_bytes,
                        filename
                    )

                    if extracted_text and extracted_text.strip():
                        extracted_parts.append(
                            f"\n\n===== {filename} =====\n\n{extracted_text.strip()}"
                        )

            if not extracted_parts:
                return jsonify({
                    "success": False,
                    "error": "No readable files were selected."
                }), 400

            text = "\n".join(extracted_parts).strip()
            source = "files" if len(filenames) > 1 else "file"

        # ====================================================
        # PASTED TEXT
        # ====================================================

        else:

            data = request.get_json(
                silent=True
            )

            if data:
                text = data.get(
                    "text",
                    ""
                ).strip()
                source = "text"

        # ====================================================
        # CHECK MATERIAL
        # ====================================================

        if not text or not text.strip():
            return jsonify({
                "success": False,
                "error": "Please provide study material."
            }), 400

        text = text.strip()

        # ====================================================
        # DETECT LANGUAGE
        # ====================================================

        detected_language = detect_language(text)

        language = normalize_detected_language(
            detected_language,
            text
        )

        print(f"Detected language: {language}")

        # ====================================================
        # CREATE CHUNKS
        # ====================================================

        chunks = chunk_text(text)

        if not chunks:
            return jsonify({
                "success": False,
                "error": "Could not create text chunks."
            }), 400

        # ====================================================
        # CREATE EMBEDDINGS
        # ====================================================

        embeddings = create_embeddings(chunks)

        if not embeddings:
            return jsonify({
                "success": False,
                "error": "Could not create study material embeddings."
            }), 500

        # ====================================================
        # SAVE MATERIAL
        # ====================================================

        material_store["text"] = text
        material_store["language"] = language
        material_store["chunks"] = chunks
        material_store["embeddings"] = embeddings

        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({
            "success": True,
            "source": source,
            "filename": filenames[0] if filenames else None,
            "filenames": filenames,
            "file_count": len(filenames),
            "characters": len(text),
            "language": language,
            "chunks": len(chunks)
        })

    except Exception as error:

        print(
            "Material processing error:",
            error
        )

        return jsonify({
            "success": False,
            "error": (
                "Could not process the study material. "
                f"{str(error)}"
            )
        }), 500


# ============================================================
# GENERATE STUDY RESULT
# ============================================================

@app.route(
    "/api/generate",
    methods=["POST"]
)
def generate_result():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({
                "success": False,
                "error":
                    "Invalid request."
            }), 400


        # ====================================================
        # MODE
        # ====================================================

        mode = data.get(
            "mode",
            "summary"
        )


        allowed_modes = {
            "summary",
            "explain",
            "quiz"
        }


        if mode not in allowed_modes:

            return jsonify({
                "success": False,
                "error":
                    "Invalid generation mode."
            }), 400


        # ====================================================
        # CHECK MATERIAL
        # ====================================================

        if not material_store["chunks"]:

            return jsonify({
                "success": False,
                "error": (
                    "Please upload or paste study "
                    "material first."
                )
            }), 400


        # ====================================================
        # RAG QUERY
        # ====================================================

        if mode == "summary":

            query = (
                "Find the most important concepts, facts, "
                "definitions, ideas and key points from "
                "this study material."
            )


        elif mode == "explain":

            query = (
                "Find the concepts, definitions, processes "
                "and difficult ideas that should be explained "
                "clearly to a student."
            )


        else:

            query = (
                "Find important concepts, facts, definitions, "
                "processes and ideas that can be used to "
                "create a quiz."
            )


        # ====================================================
        # RETRIEVE
        # ====================================================

        retrieved = retrieve_chunks(

            material_store["chunks"],

            material_store["embeddings"],

            query,

            top_k=6
        )


        # ====================================================
        # CONTEXT
        # ====================================================

        context = build_context(
            retrieved
        )


        if not context:

            return jsonify({
                "success": False,
                "error": (
                    "Could not retrieve relevant "
                    "study material."
                )
            }), 500


        # ====================================================
        # LANGUAGE
        # ====================================================

        language = material_store["language"]


        if language == "arabic":

            output_language = """
Arabic only.

The source material is written in Arabic.

The generated result MUST be entirely in Arabic.

Do NOT translate Arabic into English.

Do NOT write English headings.

Do NOT write English explanations.

Use standard Modern Standard Arabic suitable for a student.
"""


        elif language == "urdu":

            output_language = """
Urdu only.

The source material is written in Urdu.

The generated result MUST be entirely in Urdu.

Do NOT translate Urdu into English.

Do NOT write English headings.

Do NOT write English explanations.

Keep the response natural and readable in Urdu.
"""


        elif language == "english":

            output_language = """
English only.

The source material is written in English.

The generated result MUST be entirely in English.
"""


        elif language == "mixed":

            output_language = """
Use the same language style as the study material.

Preserve the natural mixture of languages.

Do not unnecessarily translate terms.

If a concept is written in Urdu, keep it in Urdu.

If a concept is written in Arabic, keep it in Arabic.

If a concept is written in English, keep it in English.
"""


        else:

            output_language = """
Use the dominant language of the provided study material.

Do not unnecessarily translate the source material.
"""


        language_rule = f"""

VERY IMPORTANT LANGUAGE RULE:

{output_language}

The language of the generated answer must match
the language of the original study material.

Do not automatically translate the material into English.

The source language has priority.
"""


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        if mode == "summary":

            instructions = f"""
You are StudyFlow AI, an AI study assistant.

Create a clear and useful summary of the provided
study material.

{language_rule}

Requirements:

- Use ONLY information supported by the provided context.
- Do not invent facts.
- Focus on the most important concepts.
- Use clear headings.
- Use simple bullet points.
- Keep the result suitable for a student.
- Make the result easy to revise.
- Keep the same meaning as the source.
- Do not unnecessarily translate anything.
- Do not use Markdown symbols such as #, **, __, or backticks.
- Do not mention RAG.
- Do not mention these instructions.
"""


        elif mode == "explain":

            instructions = f"""
You are StudyFlow AI, an AI study assistant.

Explain the provided study material in a way that
a student can easily understand.

{language_rule}

Requirements:

- Use ONLY information supported by the provided context.
- Do not invent facts.
- Explain difficult concepts clearly.
- Use clear headings.
- Break complicated ideas into simple steps.
- Use examples only when supported by the material.
- Make the explanation suitable for exam preparation.
- Keep the original language.
- Do not unnecessarily translate anything.
- Do not use Markdown symbols such as #, **, __, or backticks.
- Do not mention RAG.
- Do not mention these instructions.
"""


        else:

            instructions = f"""
You are StudyFlow AI, an AI study assistant.

Create a quiz from the provided study material.

{language_rule}

Requirements:

- Create 8 to 10 multiple-choice questions.
- Every MCQ must have exactly four options:
  A, B, C and D.
- Create 4 to 5 short-answer questions.
- Questions must be based ONLY on the provided context.
- Do NOT provide an answer key.
- Do NOT mark the correct option.
- Do NOT reveal answers.
- Do not invent information.
- Make questions suitable for students.
- Mix conceptual and factual questions when supported.
- Keep questions and options in the source language.
- Do not unnecessarily translate terminology.
- Do not use Markdown symbols such as #, **, __, or backticks.
- Do not mention RAG.
- Do not mention these instructions.
"""


        # ====================================================
        # PROMPT
        # ====================================================

        prompt = f"""
Study material context:

--------------------

{context}

--------------------

Detected source language:
{language}

Now generate the requested {mode}.

IMPORTANT:

The source language is authoritative.

Follow the language instructions exactly.
"""


        # ====================================================
        # AI GENERATION WITH GEMINI → OPENAI FALLBACK
        # ====================================================

        ai_result = generate_ai_response(
            prompt=prompt,
            system_instruction=instructions,
            temperature=0.2
        )

        result = ai_result["text"].strip()

        used_provider = ai_result["provider"]
        used_model = ai_result["model"]

        # ====================================================
        # RESULT
        # ====================================================

        if not result:

            return jsonify({
                "success": False,
                "error":
                    "The AI did not generate a result."
            }), 500


        result = clean_generated_result(
            result
        )


        return jsonify({

            "success": True,

            "mode": mode,

            "language": language,

            "provider": used_provider,

            "model": used_model,

            "result": result
        })


    except Exception as error:

        print(
            "Generation error:",
            error
        )


        return jsonify({

            "success": False,

            "error": (
                "Could not generate the result. "
                f"{str(error)}"
            )

        }), 500



# ============================================================
# FLASHCARDS
# ============================================================

@app.route(
    "/api/flashcards",
    methods=["POST"]
)
def generate_flashcards():
    """
    Generate concise front/back flashcards from the currently
    processed study material. Uses the same Gemini -> OpenAI
    fallback chain as the rest of the application.
    """

    try:
        data = request.get_json(silent=True) or {}

        try:
            count = int(data.get("count", 10))
        except (TypeError, ValueError):
            count = 10

        # Keep the endpoint predictable and protect token usage.
        count = max(5, min(count, 20))

        if not material_store["chunks"]:
            return jsonify({
                "success": False,
                "error": (
                    "Please upload or paste study material first."
                )
            }), 400

        language = material_store["language"]

        if language == "arabic":
            language_rule = """
Generate every flashcard in Arabic only.
Do not translate the source material into English.
"""
        elif language == "urdu":
            language_rule = """
Generate every flashcard in Urdu only.
Do not translate the source material into English.
"""
        elif language == "english":
            language_rule = """
Generate every flashcard in English only.
"""
        elif language == "mixed":
            language_rule = """
Use the same language style as the study material.
Preserve Urdu, Arabic and English naturally where they
appear in the source.
"""
        else:
            language_rule = """
Use the dominant language of the study material.
Do not unnecessarily translate the source.
"""

        query = (
            "Find the most important concepts, definitions, "
            "facts, processes, relationships and exam-relevant "
            "ideas that can be converted into flashcards."
        )

        retrieved = retrieve_chunks(
            material_store["chunks"],
            material_store["embeddings"],
            query,
            top_k=min(10, max(6, count))
        )

        context = build_context(retrieved)

        if not context:
            return jsonify({
                "success": False,
                "error": (
                    "Could not retrieve relevant study material "
                    "for flashcards."
                )
            }), 500

        instructions = f"""
You are StudyFlow AI, an AI study assistant.

Create exactly {count} flashcards from the provided study
material.

{language_rule}

IMPORTANT:
- Use ONLY information supported by the provided context.
- Do not invent facts.
- Do not use outside knowledge.
- Each flashcard must test one important concept.
- The front must be a concise question or prompt.
- The back must be a concise, accurate answer.
- Keep each front and back easy to read.
- Avoid duplicate cards.
- Return ONLY valid JSON.
- Do not use Markdown code fences.
- Do not add commentary before or after the JSON.

Return exactly this structure:

[
  {{
    "front": "Question or prompt",
    "back": "Short answer"
  }}
]
"""

        prompt = f"""
Study material context:

--------------------

{context}

--------------------

Detected source language:
{language}

Generate exactly {count} flashcards.
Return ONLY the JSON array requested above.
"""

        ai_result = generate_ai_response(
            prompt=prompt,
            system_instruction=instructions,
            temperature=0.2
        )

        raw = ai_result["text"].strip()

        # Remove accidental Markdown fences.
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE
        )
        raw = re.sub(
            r"\s*```$",
            "",
            raw
        ).strip()

        # Extract the JSON array if a model added a small amount
        # of surrounding text.
        start = raw.find("[")
        end = raw.rfind("]")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "The AI returned an invalid flashcard format."
            )

        json_text = raw[start:end + 1]

        import json

        cards = json.loads(json_text)

        if not isinstance(cards, list):
            raise ValueError(
                "The AI did not return a flashcard list."
            )

        cleaned_cards = []

        for card in cards:
            if not isinstance(card, dict):
                continue

            front = str(card.get("front", "")).strip()
            back = str(card.get("back", "")).strip()

            if not front or not back:
                continue

            cleaned_cards.append({
                "front": front,
                "back": back
            })

            if len(cleaned_cards) >= count:
                break

        if len(cleaned_cards) < count:
            raise ValueError(
                "The AI did not generate enough valid flashcards."
            )

        return jsonify({
            "success": True,
            "cards": cleaned_cards,
            "count": len(cleaned_cards),
            "language": language,
            "provider": ai_result["provider"],
            "model": ai_result["model"]
        })

    except Exception as error:
        print("Flashcards error:", error)

        return jsonify({
            "success": False,
            "error": (
                "Could not generate flashcards. "
                f"{str(error)}"
            )
        }), 500


# ============================================================
# TEST CONCEPTS
# TIMED MULTIPLE-CHOICE EXAM FROM STUDY MATERIAL
# ============================================================

@app.route(
    "/api/test-concepts",
    methods=["POST"]
)
def generate_test_concepts():
    """
    Generate a timed 30-question MCQ exam from the current study material.
    The frontend handles the timer and scoring so the correct answers are
    never exposed to the browser until the test is finished.
    """

    try:
        data = request.get_json(silent=True) or {}

        try:
            count = int(data.get("count", 30))
        except (TypeError, ValueError):
            count = 30

        count = max(25, min(count, 30))

        if not material_store["chunks"]:
            return jsonify({
                "success": False,
                "error": "Please upload or paste study material first."
            }), 400

        language = material_store.get("language", "english")

        language_rule = {
            "arabic": "Generate all questions and options in Arabic.",
            "urdu": "Generate all questions and options in Urdu.",
            "english": "Generate all questions and options in English.",
            "mixed": "Use the dominant language of the study material and preserve important terminology.",
        }.get(
            language,
            "Use the dominant language of the study material."
        )

        query = (
            "Find important concepts, definitions, facts, processes, "
            "relationships and exam-relevant details across the study "
            "material that can be tested with multiple-choice questions."
        )

        retrieved = retrieve_chunks(
            material_store["chunks"],
            material_store["embeddings"],
            query,
            top_k=min(18, max(10, count // 2))
        )

        context = build_context(retrieved)

        if not context:
            return jsonify({
                "success": False,
                "error": "Could not retrieve study material for the test."
            }), 500

        instructions = f"""
You are StudyFlow AI's exam generator.

Create exactly {count} high-quality multiple-choice questions from ONLY
THE PROVIDED STUDY MATERIAL.

{language_rule}

Rules:
- Every question must be answerable from the provided material.
- Do not use outside knowledge.
- Each question must have exactly four options.
- Exactly one option must be correct.
- Avoid duplicate questions.
- Cover different concepts from the material where possible.
- Mix conceptual, factual, application and comparison questions when supported.
- Do not reveal the answer in the question text.
- Return ONLY valid JSON.
- Do not use Markdown code fences.
- Do not add explanations before or after the JSON.

Return exactly this structure:
[
  {{
    "question": "Question text",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Option A"
  }}
]
"""

        prompt = f"""
Study material context:

--------------------
{context}
--------------------

Detected source language: {language}

Generate exactly {count} exam questions now.
Return ONLY the requested JSON array.
"""

        ai_result = generate_ai_response(
            prompt=prompt,
            system_instruction=instructions,
            temperature=0.25
        )

        raw = ai_result["text"].strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()

        start = raw.find("[")
        end = raw.rfind("]")

        if start == -1 or end <= start:
            raise ValueError("The AI returned an invalid test format.")

        import json
        questions = json.loads(raw[start:end + 1])

        if not isinstance(questions, list):
            raise ValueError("The AI did not return a question list.")

        cleaned_questions = []
        seen = set()

        for item in questions:
            if not isinstance(item, dict):
                continue

            question = str(item.get("question", "")).strip()
            options = item.get("options", [])
            answer = str(item.get("answer", "")).strip()

            if not question or not isinstance(options, list) or len(options) != 4:
                continue

            options = [str(option).strip() for option in options]

            if any(not option for option in options):
                continue

            if len(set(options)) != 4 or answer not in options:
                continue

            key = question.casefold()
            if key in seen:
                continue

            seen.add(key)
            cleaned_questions.append({
                "question": question,
                "options": options,
                "answer": answer
            })

            if len(cleaned_questions) >= count:
                break

        if len(cleaned_questions) < count:
            raise ValueError(
                f"The AI generated only {len(cleaned_questions)} valid questions; {count} are required."
            )

        # One minute per question gives a predictable exam duration.
        duration_seconds = count * 60

        return jsonify({
            "success": True,
            "questions": cleaned_questions,
            "count": len(cleaned_questions),
            "duration_seconds": duration_seconds,
            "language": language,
            "provider": ai_result["provider"],
            "model": ai_result["model"]
        })

    except Exception as error:
        print("Test concepts error:", error)

        return jsonify({
            "success": False,
            "error": (
                "Could not generate the Test concepts exam. "
                f"{str(error)}"
            )
        }), 500


# ============================================================
# AI CHATBOT
# ASK QUESTIONS FROM UPLOADED MATERIAL
# ============================================================

@app.route(
    "/api/chat",
    methods=["GET", "POST", "OPTIONS"],
    strict_slashes=False
)
def chat_with_material():

    # GET is intentionally supported as a deployment diagnostic.
    # Open /api/chat in the browser after deployment; a JSON response
    # proves that the Vercel deployment contains this route.
    if request.method == "GET":
        return jsonify({
            "success": True,
            "message": "StudyFlow AI chatbot endpoint is running.",
            "endpoint": "/api/chat",
            "method": "POST"
        })

    if request.method == "OPTIONS":
        return ("", 204)

    try:

        data = request.get_json(
            silent=True
        ) or {}


        question = str(
            data.get("question", "")
        ).strip()


        if not question:

            return jsonify({
                "success": False,
                "error":
                    "Please enter a question."
            }), 400


        # ====================================================
        # CHECK MATERIAL
        # ====================================================

        if not material_store["chunks"]:

            return jsonify({
                "success": False,
                "error": (
                    "Please upload or paste study "
                    "material before using the chatbot."
                )
            }), 400


        # ====================================================
        # CHAT HISTORY
        # ====================================================

        history = data.get(
            "history",
            []
        )


        if not isinstance(history, list):

            history = []


        # Keep only the most recent messages

        history = history[-8:]


        # ====================================================
        # RETRIEVE RELEVANT CHUNKS
        # ====================================================

        retrieved = retrieve_chunks(

            material_store["chunks"],

            material_store["embeddings"],

            question,

            top_k=6
        )


        context = build_context(
            retrieved
        )


        if not context:

            return jsonify({
                "success": False,
                "error": (
                    "I could not retrieve relevant "
                    "information from your study material."
                )
            }), 500


        # ====================================================
        # LANGUAGE
        # ====================================================

        language = material_store["language"]


        if language == "arabic":

            output_language = """
Answer in Arabic.

The study material is Arabic.

Do not unnecessarily translate the answer into English.
"""


        elif language == "urdu":

            output_language = """
Answer in Urdu.

The study material is Urdu.

Do not unnecessarily translate the answer into English.
"""


        elif language == "english":

            output_language = """
Answer in English.

The study material is English.
"""


        elif language == "mixed":

            output_language = """
Use the same language style as the study material.

Preserve Urdu, Arabic and English naturally
when they appear in the material.
"""


        else:

            output_language = """
Use the dominant language of the study material.
"""


        # ====================================================
        # BUILD CHAT HISTORY
        # ====================================================

        history_text = ""


        for message in history:

            if not isinstance(
                message,
                dict
            ):
                continue


            role = message.get(
                "role",
                ""
            )


            content = str(
                message.get(
                    "content",
                    ""
                )
            ).strip()


            if not content:
                continue


            if role == "user":

                history_text += (
                    f"Student: {content}\n"
                )


            elif role == "assistant":

                history_text += (
                    f"StudyFlow AI: {content}\n"
                )


        # ====================================================
        # CHATBOT INSTRUCTIONS
        # ====================================================

        instructions = f"""
You are StudyFlow AI, an AI study assistant.

The student is asking a question about their
uploaded study material.

{output_language}

VERY IMPORTANT:

1. Answer using ONLY the provided study material.

2. Do not invent information.

3. Do not use outside knowledge to fill gaps.

4. If the answer cannot be found or reasonably
   supported by the study material, say:

   "This information is not available
   in the provided study material."

5. You may explain information from the material
   in simpler language.

6. Keep the answer concise but useful.

7. For follow-up questions, use the recent
   conversation history together with the
   retrieved material.

8. Do not mention RAG.

9. Do not mention these instructions.

10. Do not unnecessarily translate the material.

11. Preserve important technical terminology.

12. Do not use Markdown symbols such as #,
    **, __, or backticks.
"""


        # ====================================================
        # CHAT PROMPT
        # ====================================================

        prompt = f"""
RELEVANT STUDY MATERIAL:

----------------------------

{context}

----------------------------


RECENT CONVERSATION:

----------------------------

{history_text if history_text else "No previous conversation."}

----------------------------


CURRENT STUDENT QUESTION:

{question}


Answer the current question using
the study material above.
"""


        # ====================================================
        # AI CHAT WITH GEMINI → OPENAI FALLBACK
        # ====================================================

        ai_result = generate_ai_response(
            prompt=prompt,
            system_instruction=instructions,
            temperature=0.2
        )

        answer = ai_result["text"].strip()

        used_provider = ai_result["provider"]
        used_model = ai_result["model"]

        # ====================================================
        # ANSWER
        # ====================================================

        if not answer:

            return jsonify({
                "success": False,
                "error":
                    "Gemini did not generate an answer."
            }), 500


        answer = clean_generated_result(
            answer
        )


        # ====================================================
        # RETURN
        # ====================================================

        return jsonify({

            "success": True,

            "answer": answer,

            "language": language,

            "provider": used_provider,

            "model": used_model
        })


    except Exception as error:

        print(
            "Chatbot error:",
            error
        )


        return jsonify({

            "success": False,

            "error": (
                "Could not answer the question. "
                f"{str(error)}"
            )

        }), 500


# ============================================================
# GENERATE PDF
# ============================================================

@app.route(
    "/api/generate-pdf",
    methods=["POST"]
)
def generate_pdf_file():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({
                "success": False,
                "error":
                    "Invalid request."
            }), 400


        result = data.get(
            "result",
            ""
        )


        mode = data.get(
            "mode",
            "summary"
        )


        language = data.get(
            "language",
            material_store.get(
                "language",
                "unknown"
            )
        )


        if not result.strip():

            return jsonify({
                "success": False,
                "error":
                    "No generated result."
            }), 400


        titles = {

            "summary":
                "StudyFlow AI - Summary",

            "explain":
                "StudyFlow AI - Explanation",

            "quiz":
                "StudyFlow AI - Quiz"
        }


        title = titles.get(
            mode,
            "StudyFlow AI"
        )


        pdf_buffer = generate_pdf(
            title,
            result
        )


        return send_file(

            pdf_buffer,

            mimetype="application/pdf",

            as_attachment=True,

            download_name=(
                f"studyflow_{mode}.pdf"
            )
        )


    except Exception as error:

        print(
            "PDF error:",
            error
        )


        return jsonify({

            "success": False,

            "error": (
                "Could not generate PDF. "
                f"{str(error)}"
            )

        }), 500


# ============================================================
# AI ORAL EXAM
# ============================================================
@app.route("/api/oral-exam", methods=["POST"])
def oral_exam():
    """Voice-only oral exam. The hidden reference answer stays server-side."""
    import json
    import uuid

    def parse_json(raw):
        """Parse a JSON object returned by an AI provider safely."""
        if isinstance(raw, dict):
            parsed = raw
        else:
            if not isinstance(raw, str):
                raise ValueError("The AI returned an unexpected response type.")

            cleaned = raw.strip()
            cleaned = re.sub(
                r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE
            )
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            if first_brace < 0 or last_brace <= first_brace:
                raise ValueError("The AI returned an invalid oral-exam format.")

            parsed = json.loads(cleaned[first_brace:last_brace + 1])

        if not isinstance(parsed, dict):
            raise ValueError("The AI response must be a JSON object.")
        return parsed

    def oral_context(query):
        """Retrieve relevant chunks before formatting context for the model."""
        chunks = material_store.get("chunks") or []
        if not chunks:
            raise ValueError("Please upload or paste study material first.")

        embeddings = material_store.get("embeddings") or []
        retrieved = retrieve_chunks(
            chunks,
            embeddings,
            query,
            top_k=min(10, len(chunks))
        )
        return build_context(retrieved)

    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Invalid request body. Expected a JSON object."
            }), 400

        action = str(data.get("action", "start")).strip().lower()

        if not material_store.get("chunks"):
            return jsonify({
                "success": False,
                "error": "Please upload or paste study material first."
            }), 400

        language = material_store.get("language", "english")
        total_questions = 10
        language_rule = {
            "urdu": "Ask and evaluate in Urdu only.",
            "arabic": "Ask and evaluate in Arabic only.",
            "english": "Ask and evaluate in English only.",
            "mixed": "Use the same natural language style as the study material."
        }.get(language, "Use the dominant language of the study material.")

        if action == "start":
            context = oral_context(
                "Find important concepts, definitions, facts, processes, and "
                "exam-relevant ideas for short oral questions."
            )
            instructions = f"""You are StudyFlow AI conducting a short oral exam.
{language_rule}
Create ONE short, clear oral question from the provided study material. Use ONLY the material. Do not use outside knowledge. Return ONLY JSON: {{"question":"short question","reference_answer":"short correct answer"}}"""

            ai_result = generate_ai_response(
                prompt=(
                    f"STUDY MATERIAL:\n{context}\n\n"
                    "Generate the first oral-exam question."
                ),
                system_instruction=instructions,
                temperature=0.2
            )
            ai_text = ai_result.get("text") if isinstance(ai_result, dict) else ai_result
            item = parse_json(ai_text)
            question = str(item.get("question", "")).strip()
            reference_answer = str(item.get("reference_answer", "")).strip()

            if not question or not reference_answer:
                raise ValueError("The AI did not return a valid oral-exam question.")

            session_id = str(uuid.uuid4())
            oral_exam_sessions[session_id] = {
                "language": language,
                "question_number": 1,
                "total_questions": total_questions,
                "score": 0,
                "question": question,
                "reference_answer": reference_answer
            }
            return jsonify({
                "success": True,
                "session_id": session_id,
                "question": question,
                "question_number": 1,
                "total_questions": total_questions,
                "language": language
            })

        if action != "evaluate":
            return jsonify({
                "success": False,
                "error": "Invalid oral-exam action."
            }), 400

        session_id = str(data.get("session_id", "")).strip()
        answer = str(data.get("answer", "")).strip()
        session = oral_exam_sessions.get(session_id)

        if not session:
            return jsonify({
                "success": False,
                "error": "The oral-exam session has expired. Please start a new test."
            }), 400

        if not answer:
            return jsonify({
                "success": False,
                "error": "No spoken answer was detected."
            }), 400

        # IMPORTANT: pass retrieved chunks to build_context, matching the
        # successful question-generation path. Do not pass raw stored chunks.
        context = oral_context(
            f"Find information relevant to this oral-exam question: {session['question']}"
        )
        evaluation_instructions = f"""You are evaluating a student's spoken answer in an oral exam.
{language_rule}
Use ONLY the study material. Compare the student's answer with the reference answer and accept different wording when meaning is correct. Return ONLY JSON: {{"correct":true,"feedback":"Very short feedback starting with Correct or Incorrect.","score":1}}. score must be 1 or 0. Do not punish minor transcription or grammar differences when meaning is correct."""
        evaluation_prompt = (
            f"STUDY MATERIAL:\n{context}\n\n"
            f"QUESTION:\n{session['question']}\n\n"
            f"REFERENCE ANSWER:\n{session['reference_answer']}\n\n"
            f"STUDENT SPOKEN ANSWER:\n{answer}"
        )
        evaluation_result = generate_ai_response(
            prompt=evaluation_prompt,
            system_instruction=evaluation_instructions,
            temperature=0.1
        )
        evaluation_text = (
            evaluation_result.get("text")
            if isinstance(evaluation_result, dict)
            else evaluation_result
        )
        evaluation = parse_json(evaluation_text)
        correct = bool(evaluation.get("correct", False))
        session["score"] += 1 if correct else 0
        feedback = str(
            evaluation.get("feedback", "Correct." if correct else "Incorrect.")
        ).strip()
        current_number = session["question_number"]

        if current_number >= session["total_questions"]:
            score = session["score"]
            total = session["total_questions"]
            oral_exam_sessions.pop(session_id, None)
            return jsonify({
                "success": True,
                "completed": True,
                "correct": correct,
                "feedback": feedback,
                "score": score,
                "total_questions": total,
                "final_message": f"Oral test complete. You scored {score} out of {total}."
            })

        next_context = oral_context(
            "Find another important concept or fact for a short oral question. "
            f"Avoid repeating this question: {session['question']}"
        )
        next_instructions = f"""You are StudyFlow AI conducting an oral exam.
{language_rule}
Generate ONE short oral question from the study material, different from the previous question. Use ONLY the material. Return ONLY JSON: {{"question":"short question","reference_answer":"short correct answer"}}"""
        next_prompt = (
            f"STUDY MATERIAL:\n{next_context}\n\n"
            f"PREVIOUS QUESTION:\n{session['question']}\n\n"
            "Generate the next oral question."
        )
        next_result = generate_ai_response(
            prompt=next_prompt,
            system_instruction=next_instructions,
            temperature=0.2
        )
        next_text = next_result.get("text") if isinstance(next_result, dict) else next_result
        next_item = parse_json(next_text)
        next_question = str(next_item.get("question", "")).strip()
        next_reference_answer = str(next_item.get("reference_answer", "")).strip()

        if not next_question or not next_reference_answer:
            raise ValueError("The AI did not return a valid next oral question.")

        next_number = current_number + 1
        session.update(
            question_number=next_number,
            question=next_question,
            reference_answer=next_reference_answer
        )
        return jsonify({
            "success": True,
            "completed": False,
            "correct": correct,
            "feedback": feedback,
            "score": session["score"],
            "question_number": next_number,
            "total_questions": session["total_questions"],
            "next_question": next_question,
            "language": session["language"]
        })

    except Exception as error:
        print("Oral exam error:", repr(error))
        return jsonify({
            "success": False,
            "error": f"Could not process the oral exam. {str(error)}"
        }), 500


# ============================================================
# ============================================================
# USER FEEDBACK EMAIL (Resend API)
# Configure RESEND_API_KEY, FEEDBACK_TO_EMAIL, RESEND_FROM_EMAIL in Vercel.
# ============================================================

@app.route("/api/feedback", methods=["POST", "OPTIONS"])
def submit_feedback():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    feedback_type = str(data.get("type", "General feedback")).strip()
    message = str(data.get("message", "")).strip()
    sender_name = str(data.get("name", "StudyFlow AI user")).strip()[:120]

    if feedback_type not in {"Suggestion", "Bug report", "General feedback"}:
        feedback_type = "General feedback"
    if len(message) > 3000:
        return jsonify({"success": False, "error": "Feedback must be 3000 characters or fewer."}), 400

    api_key = (os.getenv("RESEND_API_KEY") or "").strip()
    destination = (os.getenv("FEEDBACK_TO_EMAIL") or "").strip()
    sender = (os.getenv("RESEND_FROM_EMAIL") or "StudyFlow AI <onboarding@resend.dev>").strip()

    if not api_key or not destination or not sender:
        missing = [
            name for name, value in (
                ("RESEND_API_KEY", api_key),
                ("FEEDBACK_TO_EMAIL", destination),
                ("RESEND_FROM_EMAIL", sender),
            ) if not value
        ]
        app.logger.error("Feedback email configuration missing variables: %s", ", ".join(missing))
        return jsonify({"success": False, "error": "Feedback email service is not configured. Check the Vercel environment variables."}), 503

    email_text = (
        "New feedback submitted through StudyFlow AI.\n\n"
        f"Type: {feedback_type}\n"
        f"Name: {sender_name or 'StudyFlow AI user'}\n\n"
        f"Feedback:\n{message}\n"
    )
    payload = {
        "from": sender,
        "to": [destination],
        "subject": f"StudyFlow AI feedback: {feedback_type}",
        "text": email_text,
    }

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "StudyFlowAI"
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            response.read()
        return jsonify({"success": True, "message": "Feedback sent successfully."}), 200
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        # Keep the provider's detailed response in server logs, never log the API key.
        app.logger.error("Resend rejected feedback email (HTTP %s): %s", error.code, details[:1000])
        return jsonify({
            "success": False,
            "error": f"Resend rejected the email (HTTP {error.code}). Check the Vercel function logs for details.",
        }), 502
    except Exception:
        app.logger.exception("Feedback email delivery failed before Resend confirmed acceptance.")
        return jsonify({"success": False, "error": "We could not reach the email service. Check the Vercel function logs and try again."}), 502


# ============================================================

# API 404 HANDLER
# ============================================================

@app.errorhandler(404)
def api_not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "API endpoint not found.",
            "path": request.path,
            "hint": "Redeploy the latest app.py on Vercel and verify /api/chat exists."
        }), 404

    return error


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
