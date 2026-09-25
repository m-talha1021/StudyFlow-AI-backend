from io import BytesIO
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import (
    TA_CENTER,
    TA_RIGHT,
    TA_LEFT
)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import arabic_reshaper
from bidi.algorithm import get_display


# ------------------------------------------------
# Urdu Font
# ------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

URDU_FONT_PATH = os.path.join(
    BASE_DIR,
    "fonts",
    "NotoNaskhArabic-Regular.ttf"
)


# Register Urdu font
pdfmetrics.registerFont(
    TTFont(
        "NotoNaskhArabic",
        URDU_FONT_PATH
    )
)


# ------------------------------------------------
# Detect Urdu
# ------------------------------------------------

def contains_urdu(text):

    return any(
        "\u0600" <= character <= "\u06FF"
        for character in text
    )


# ------------------------------------------------
# Prepare Urdu text
# ------------------------------------------------

def prepare_urdu_text(text):

    reshaped = arabic_reshaper.reshape(
        text
    )

    return get_display(
        reshaped
    )


# ------------------------------------------------
# Generate PDF
# ------------------------------------------------

def generate_pdf(
    title,
    content
):

    buffer = BytesIO()


    # ------------------------------------------------
    # PDF document
    # ------------------------------------------------

    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm
    )


    # ------------------------------------------------
    # Styles
    # ------------------------------------------------

    styles = getSampleStyleSheet()


    # English title
    title_style = ParagraphStyle(

        "CustomTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        spaceAfter=20
    )


    # English body
    body_style = ParagraphStyle(

        "CustomBody",

        parent=styles["BodyText"],

        fontName="Helvetica",

        fontSize=10.5,

        leading=16,

        spaceAfter=10,

        alignment=TA_LEFT
    )


    # Urdu title
    urdu_title_style = ParagraphStyle(

        "UrduTitle",

        parent=styles["Title"],

        fontName="NotoNaskhArabic",

        fontSize=20,

        leading=28,

        alignment=TA_RIGHT,

        spaceAfter=20
    )


    # Urdu body
    urdu_body_style = ParagraphStyle(

        "UrduBody",

        parent=styles["BodyText"],

        fontName="NotoNaskhArabic",

        fontSize=12,

        leading=20,

        spaceAfter=10,

        alignment=TA_RIGHT
    )


    # ------------------------------------------------
    # Story
    # ------------------------------------------------

    story = []


    # ------------------------------------------------
    # Title
    # ------------------------------------------------

    if contains_urdu(title):

        title_text = prepare_urdu_text(
            title
        )

        story.append(

            Paragraph(
                title_text,
                urdu_title_style
            )

        )

    else:

        story.append(

            Paragraph(
                title,
                title_style
            )

        )


    # ------------------------------------------------
    # Content
    # ------------------------------------------------

    lines = content.split("\n")


    for line in lines:

        line = line.strip()


        # Empty line
        if not line:

            story.append(
                Spacer(1, 6)
            )

            continue


        # --------------------------------------------
        # Escape HTML characters
        # --------------------------------------------

        line = (
            line
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


        # --------------------------------------------
        # Urdu line
        # --------------------------------------------

        if contains_urdu(line):

            # Remove HTML escaping before shaping
            clean_line = (
                line
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
            )


            shaped_line = prepare_urdu_text(
                clean_line
            )


            # Escape again after shaping
            shaped_line = (
                shaped_line
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )


            story.append(

                Paragraph(
                    shaped_line,
                    urdu_body_style
                )

            )


        # --------------------------------------------
        # English line
        # --------------------------------------------

        else:

            story.append(

                Paragraph(
                    line,
                    body_style
                )

            )


    # ------------------------------------------------
    # Build PDF
    # ------------------------------------------------

    document.build(
        story
    )


    buffer.seek(0)


    return buffer