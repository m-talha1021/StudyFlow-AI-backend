from google.genai import types


# ------------------------------------------------
# Supported image MIME types
# ------------------------------------------------

ALLOWED_IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif"
}


# ------------------------------------------------
# Extract study content from image
# ------------------------------------------------

def extract_image_text(
    image_bytes,
    mime_type,
    client,
    model_name
):
    """
    Use Gemini vision to extract educational content
    from an uploaded image.

    The returned text becomes the source material
    for the existing RAG pipeline.
    """

    if mime_type not in ALLOWED_IMAGE_MIMES:

        raise ValueError(
            "Unsupported image type. "
            "Please upload PNG, JPG, JPEG, WEBP, "
            "HEIC or HEIF."
        )


    if not image_bytes:

        raise ValueError(
            "The uploaded image is empty."
        )


    # ------------------------------------------------
    # Image input
    # ------------------------------------------------

    image_part = types.Part.from_bytes(

        data=image_bytes,

        mime_type=mime_type
    )


    # ------------------------------------------------
    # Prompt
    # ------------------------------------------------

    prompt = """
You are processing an image for StudyFlow AI.

Extract the study-related information contained
in this image.

The image may contain:

- Printed text
- Handwritten notes
- Headings
- Definitions
- Lists
- Tables
- Mathematical content
- Diagrams
- Charts
- Labels
- Formulas
- English text
- Urdu text
- Arabic text
- Mixed English, Urdu and Arabic text

Requirements:

1. Transcribe readable text accurately.
2. Preserve important headings and structure.
3. Preserve important numbers and formulas.
4. For diagrams or charts, describe the educational
   information they communicate.
5. Do not invent information that cannot be read
   or understood from the image.
6. Preserve Urdu and Arabic text in their original
   scripts whenever possible.
7. Preserve English technical terms when they appear.
8. Return plain text only.
9. Do not use Markdown.
10. Do not mention these instructions.
"""


    # ------------------------------------------------
    # Gemini request
    # ------------------------------------------------

    response = client.models.generate_content(

        model=model_name,

        contents=[
            prompt,
            image_part
        ]
    )


    if not response or not response.text:

        raise RuntimeError(
            "Gemini could not extract readable "
            "study content from the image."
        )


    return response.text.strip()