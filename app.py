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

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5173",
                "http://localhost:3000",
                "https://study-flow-assistant.netlify.app"
            ],
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    }
)


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

@app.route("/api/test")
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
    methods=["POST"]
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


            # Read file

            file_bytes = file.read()


            if not file_bytes:

                return jsonify({
                    "success": False,
                    "error":
                        "The uploaded file is empty."
                }), 400


            # =================================================
            # IMAGE
            # =================================================

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
                                "Image processing "
                                "succeeded with: "
                                f"{model_name}"
                            )

                            break


                    except Exception as error:

                        last_image_error = error

                        print(
                            f"{model_name} failed "
                            "for image processing:"
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


            # =================================================
            # PDF / DOCX / PPTX
            # =================================================

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
                    "Could not create study "
                    "material embeddings."
                )
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
                "Could not process the study "
                "material. "
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
        # GEMINI FALLBACK
        # ====================================================

        response = None

        last_error = None

        used_model = None


        for model_name in GEMINI_MODELS:

            try:

                print(
                    f"Trying Gemini model: "
                    f"{model_name}"
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
                        "Success with Gemini model: "
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


        if response is None or not response.text:

            if last_error:
                raise last_error

            raise RuntimeError(
                "All Gemini models failed."
            )


        # ====================================================
        # RESULT
        # ====================================================

        result = response.text.strip()


        if not result:

            return jsonify({
                "success": False,
                "error":
                    "Gemini did not generate a result."
            }), 500


        result = clean_generated_result(
            result
        )


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
# AI CHATBOT
# ASK QUESTIONS FROM UPLOADED MATERIAL
# ============================================================

@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat_with_material():

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


        question = data.get(
            "question",
            ""
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
        # GEMINI FALLBACK
        # ====================================================

        response = None

        last_error = None

        used_model = None


        for model_name in GEMINI_MODELS:

            try:

                print(
                    f"Trying chatbot model: "
                    f"{model_name}"
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
                        "Chatbot succeeded with: "
                        f"{model_name}"
                    )

                    break


            except Exception as error:

                last_error = error

                print(
                    f"{model_name} failed "
                    "for chatbot:"
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
        # ANSWER
        # ====================================================

        answer = response.text.strip()


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
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
