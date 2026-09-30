import os
import re

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from dotenv import load_dotenv

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

# The current Vercel preview URL pattern is also accepted. This is kept
# narrow to this project rather than enabling CORS for every Vercel app.
VERCEL_PREVIEW_ORIGIN_PATTERN = re.compile(
    r"^https://study-flow-in597ve4c-muhammad-talhas-projects-[a-z0-9-]+\.vercel\.app$",
    re.IGNORECASE
)


def is_allowed_origin(origin):
    if not origin:
        return False

    normalized = origin.rstrip("/")

    if normalized in FRONTEND_ORIGINS:
        return True

    return bool(VERCEL_PREVIEW_ORIGIN_PATTERN.fullmatch(normalized))


CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [*FRONTEND_ORIGINS, VERCEL_PREVIEW_ORIGIN_PATTERN],
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
