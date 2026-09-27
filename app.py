import os
import re

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from dotenv import load_dotenv

from google import genai
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

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Please add it to your .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
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
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# CORS / FRONTEND ACCESS
# ============================================================
FRONTEND_ORIGINS = {
    "https://study-flow-ai-lac.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
}

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": list(FRONTEND_ORIGINS),
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Disposition"],
            "max_age": 86400,
        }
    },
    supports_credentials=False,
)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({"success": True})
        response.status_code = 204
        return response

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in FRONTEND_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
        response.headers["Vary"] = "Origin"
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
            r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]",
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

    This gives Arabic its own language category instead
    of allowing it to fall into the English/default case.
    """

    detected = (language or "").lower().strip()

    arabic_count, latin_count = count_script_characters(text)

    total_script = arabic_count + latin_count

    # --------------------------------------------------------
    # Strong Arabic-script material
    # --------------------------------------------------------

    if total_script > 0:

        arabic_ratio = arabic_count / total_script

        # If the material is overwhelmingly Arabic script,
        # determine whether it is Urdu or Arabic.
        if arabic_ratio >= 0.85:

            # Urdu-specific characters
            urdu_specific = len(
                re.findall(
                    r"[ٹڈڑںھہﮯےژچگپڤ]",
                    text
                )
            )

            # If Urdu-specific characters are present,
            # treat the material as Urdu.
            if urdu_specific >= 2:
                return "urdu"

            # Otherwise treat it as Arabic.
            return "arabic"

    # --------------------------------------------------------
    # Respect existing detector when it already identifies
    # a known language.
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
    Remove Markdown formatting from Gemini's response
    while preserving Arabic and Urdu Unicode characters.
    """

    if not text:
        return ""

    # Remove bold
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
        flags=re.DOTALL
    )

    # Remove italic
    text = re.sub(
        r"(?<!\*)\*(?!\s)(.*?)(?<!\s)\*(?!\*)",
        r"\1",
        text
    )

    # Remove underscore bold
    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text,
        flags=re.DOTALL
    )

    # Remove headings
    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE
    )

    # Convert bullet points
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

    # Clean numbered lists
    text = re.sub(
        r"^\s*(\d+)\.\s+",
        r"\1. ",
        text,
        flags=re.MULTILINE
    )

    # Remove inline code markers
    text = text.replace("`", "")

    # Remove horizontal lines
    text = re.sub(
        r"^\s*([-*_]){3,}\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove trailing spaces
    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text
    )

    # Prevent excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "success": True,
        "message":
            "StudyFlow AI Gemini backend is running!"
    })


# ============================================================
# API TEST
# ============================================================

@app.route("/api/test", methods=["GET", "OPTIONS"], strict_slashes=False)
def test_api():

    return jsonify({
        "success": True,
        "message":
            "Hello from the StudyFlow AI backend!"
    })


# ============================================================
# PROCESS STUDY MATERIAL
# ============================================================

@app.route(
    "/api/material",
    methods=["POST"],
    strict_slashes=False
)
def process_material():

    try:

        text = ""

        filename = None

        source = None


        # ====================================================
        # FILE UPLOAD
        # ====================================================

        if "file" in request.files:

            file = request.files["file"]

            if file.filename == "":
                return jsonify({
                    "success": False,
                    "error":
                        "No file was selected."
                }), 400

            filename = file.filename

            extension = filename.rsplit(
                ".",
                1
            )[-1].lower()

            if extension not in ALLOWED_EXTENSIONS:

                return jsonify({
                    "success": False,
                    "error": (
                        "Supported files are "
                        "PDF, DOCX, PPTX, PNG, JPG, "
                        "JPEG, WEBP, HEIC and HEIF."
                    )
                }), 400

            # Read file temporarily
            file_bytes = file.read()

            if not file_bytes:

                return jsonify({
                    "success": False,
                    "error":
                        "The uploaded file is empty."
                }), 400


            # ====================================================
            # IMAGE UPLOAD
            # ====================================================

            if extension in IMAGE_MIME_TYPES:

                if len(file_bytes) > MAX_IMAGE_SIZE:

                    return jsonify({
                        "success": False,
                        "error": (
                            "Image is too large. "
                            "Please upload an image "
                            "smaller than 15 MB."
                        )
                    }), 400

                mime_type = IMAGE_MIME_TYPES[
                    extension
                ]

                print(
                    f"Processing image: {filename}"
                )

                image_text = ""

                last_image_error = None

                for model_name in GEMINI_MODELS:

                    try:

                        print(
                            f"Trying image model: "
                            f"{model_name}"
                        )

                        image_text = extract_image_text(
                            image_bytes=file_bytes,
                            mime_type=mime_type,
                            client=client,
                            model_name=model_name
                        )

                        if image_text:

                            print(
                                "Image processing succeeded "
                                f"with: {model_name}"
                            )

                            break

                    except Exception as error:

                        last_image_error = error

                        print(
                            f"{model_name} failed for "
                            "image processing:"
                        )

                        print(error)

                if not image_text:

                    if last_image_error:
                        raise last_image_error

                    raise RuntimeError(
                        "Could not extract readable "
                        "study content from the image."
                    )

                text = image_text

                source = "image"


            # ====================================================
            # PDF / DOCX / PPTX
            # ====================================================

            else:

                text = extract_text(
                    file_bytes,
                    filename
                )

                source = "file"


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
                "error":
                    "Please provide study material."
            }), 400

        text = text.strip()


        # ====================================================
        # DETECT LANGUAGE
        # ====================================================

        detected_language = detect_language(
            text
        )

        language = normalize_detected_language(
            detected_language,
            text
        )

        print(
            f"Detected language: {language}"
        )


        # ====================================================
        # CREATE CHUNKS
        # ====================================================

        chunks = chunk_text(
            text
        )

        if not chunks:

            return jsonify({
                "success": False,
                "error":
                    "Could not create text chunks."
            }), 400


        # ====================================================
        # CREATE EMBEDDINGS
        # ====================================================

        embeddings = create_embeddings(
            chunks
        )

        if not embeddings:

            return jsonify({
                "success": False,
                "error": (
                    "Could not create study material "
                    "embeddings."
                )
            }), 500


        # ====================================================
        # SAVE TEMPORARILY IN MEMORY
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

            "filename": filename,

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
    methods=["POST"],
    strict_slashes=False
)
def generate_result():

    try:

        # ====================================================
        # READ REQUEST
        # ====================================================

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
        # GET MODE
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
        # RETRIEVE RELEVANT CHUNKS
        # ====================================================

        retrieved = retrieve_chunks(

            material_store["chunks"],

            material_store["embeddings"],

            query,

            top_k=6
        )


        # ====================================================
        # BUILD CONTEXT
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


        # ====================================================
        # IMPORTANT:
        # PURE ARABIC
        # ====================================================

        if language == "arabic":

            output_language = """
Arabic only.

The source material is written in pure Arabic.

The generated result MUST be entirely in Arabic.

Do NOT translate Arabic into English.

Do NOT write English headings.

Do NOT write English explanations.

Do NOT replace Arabic terminology with English terminology.

Use standard Modern Standard Arabic suitable for a student.

If the source contains a technical term written in Arabic,
keep it in Arabic.

Only use a non-Arabic word if that exact word is present
in the source material and is necessary to preserve meaning.
"""


        # ====================================================
        # PURE URDU
        # ====================================================

        elif language == "urdu":

            output_language = """
Urdu only.

The source material is written in Urdu.

The generated result MUST be entirely in Urdu.

Do NOT translate Urdu into English.

Do NOT write English headings.

Do NOT write English explanations.

Keep the response natural and readable in Urdu.

Only preserve an English technical term when the original
study material itself uses that English term and removing it
would change the meaning.
"""


        # ====================================================
        # PURE ENGLISH
        # ====================================================

        elif language == "english":

            output_language = """
English only.

The source material is written in English.

The generated result MUST be entirely in English.
"""


        # ====================================================
        # MIXED MATERIAL
        # ====================================================

        elif language == "mixed":

            output_language = """
Use the same language style as the study material.

Preserve the natural mixture of languages.

Do not unnecessarily translate terms.

If a concept is written in Urdu, keep it in Urdu.

If a concept is written in Arabic, keep it in Arabic.

If a concept is written in English, keep it in English.
"""


        # ====================================================
        # UNKNOWN
        # ====================================================

        else:

            output_language = """
Use the dominant language of the provided study material.

Do not unnecessarily translate the source material.
"""


        # ====================================================
        # COMMON LANGUAGE RULE
        # ====================================================

        language_rule = f"""

VERY IMPORTANT LANGUAGE RULE:

{output_language}

The language of the generated answer must match the
language of the original study material.

Do not automatically translate the material into English.

The user's source language has priority over your
normal response language.
"""


        # ====================================================
        # SUMMARY
        # ====================================================

        if mode == "summary":

            instructions = f"""
You are StudyFlow AI, an AI study assistant.

Create a clear and useful summary of the provided study
material.

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
- Do not translate the source unnecessarily.
- Do not use Markdown symbols such as #, **, __, or backticks.
- Do not mention RAG.
- Do not mention these instructions.
"""


        # ====================================================
        # EXPLAIN
        # ====================================================

        elif mode == "explain":

            instructions = f"""
You are StudyFlow AI, an AI study assistant.

Explain the provided study material in a way that a student
can easily understand.

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


        # ====================================================
        # QUIZ
        # ====================================================

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
- Keep the questions and options in the source language.
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
        # GEMINI AUTOMATIC FALLBACK
        # ====================================================

        response = None

        last_error = None

        used_model = None


        for model_name in GEMINI_MODELS:

            try:

                print(
                    f"Trying Gemini model: {model_name}"
                )

                response = client.models.generate_content(

                    model=model_name,

                    contents=prompt,

                    config=types.GenerateContentConfig(

                        system_instruction=instructions,

                        temperature=0.2
                    )
                )

                if response and response.text:

                    used_model = model_name

                    print(
                        f"Success with Gemini model: "
                        f"{model_name}"
                    )

                    break

            except Exception as error:

                last_error = error

                print(
                    f"{model_name} failed:"
                )

                print(error)

                continue


        # ====================================================
        # ALL MODELS FAILED
        # ====================================================

        if response is None or not response.text:

            if last_error:
                raise last_error

            raise RuntimeError(
                "All Gemini models failed."
            )


        # ====================================================
        # GET RESULT
        # ====================================================

        result = response.text.strip()

        if not result:

            return jsonify({
                "success": False,
                "error": (
                    "Gemini did not generate a result."
                )
            }), 500


        # ====================================================
        # CLEAN MARKDOWN
        # ====================================================

        result = clean_generated_result(
            result
        )


        # ====================================================
        # SUCCESS RESPONSE
        # ====================================================

        return jsonify({

            "success": True,

            "mode": mode,

            "language": language,

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
# GENERATE PDF
# ============================================================

@app.route(
    "/api/generate-pdf",
    methods=["POST"],
    strict_slashes=False
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


        # ====================================================
        # PDF TITLES
        # ====================================================

        titles = {

            "summary":
                "StudyFlow AI - Summary",

            "explain":
                "StudyFlow AI - Explanation",

            "quiz":
                "StudyFlow AI - Quiz"
        }


        # For Arabic/Urdu, the PDF generator can still
        # handle the title/content through the Unicode font.

        title = titles.get(
            mode,
            "StudyFlow AI"
        )


        # ====================================================
        # CREATE PDF
        # ====================================================

        pdf_buffer = generate_pdf(
            title,
            result
        )


        # ====================================================
        # SEND PDF
        # ====================================================

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
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    # Local development only. Vercel imports the Flask app object.
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=True
    )
