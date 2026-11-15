"""Build the 16:9 presentation used for the 3-minute video (python-pptx), with narration in speaker notes.

Slide spec (dict):
  {"type": "title", "title": str, "subtitle": str, "kpis": [(value, label), ...], "notes": str}
  {"type": "chart", "kicker": str, "title": str, "image": figure filename, "bullets": [str], "notes": str}
  {"type": "two", "kicker": str, "title": str, "images": [fig_a, fig_b], "bullets": [str], "notes": str}
  {"type": "text", "kicker": str, "title": str, "rows": [(heading, body)], "notes": str}
Bullets support **bold** segments.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
PRIMARY, ACCENT, GOOD, INK, MUTED, LIGHT, DARK = (RGBColor(0x1F, 0x4E, 0x79), RGBColor(0xE4, 0x57, 0x2E), RGBColor(0x2E, 0x8B, 0x57),
                                                 RGBColor(0x1D, 0x27, 0x33), RGBColor(0x5B, 0x6B, 0x7B), RGBColor(0xEE, 0xF3, 0xF8), RGBColor(0x0F, 0x2A, 0x44))
W, H = Inches(13.333), Inches(7.5)
FONT = "Segoe UI"


def _rich(paragraph, text, size, color=INK, bold_color=None):
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        run = paragraph.add_run()
        is_bold = part.startswith("**") and part.endswith("**")
        run.text = part[2:-2] if is_bold else part
        run.font.size = Pt(size); run.font.name = FONT; run.font.bold = is_bold
        run.font.color.rgb = (bold_color or color) if is_bold else color


def _box(slide, x, y, w, h, fill=None, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    return shp


def _text(slide, x, y, w, h, text, size, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = FONT
    return tb


def _frame(slide, kicker, title, n, total):
    _box(slide, 0, 0, W, Inches(0.09), fill=PRIMARY)
    _text(slide, Inches(0.5), Inches(0.25), Inches(10), Inches(0.35), kicker.upper(), 11, ACCENT, bold=True)
    _text(slide, Inches(0.5), Inches(0.55), Inches(12.3), Inches(0.95), title, 26, DARK, bold=True)
    _text(slide, Inches(0.5), Inches(7.08), Inches(9), Inches(0.3), "VoltRelay Energy  |  Gradient Learnings Data Analytics Hackathon", 9, MUTED)
    _text(slide, Inches(11.8), Inches(7.08), Inches(1.0), Inches(0.3), f"{n}/{total}", 9, MUTED, align=PP_ALIGN.RIGHT)


def _picture(slide, path, x, y, max_w, max_h):
    img = Image.open(path)
    ar = img.width / img.height
    w, h = max_w, int(max_w / ar)
    if h > max_h:
        h, w = max_h, int(max_h * ar)
    left = x + int((max_w - w) / 2)
    top = y + int((max_h - h) / 2)
    slide.shapes.add_picture(str(path), left, top, w, h)


def _bullets(slide, x, y, w, h, bullets, size=15):
    _box(slide, x, y, w, h, fill=LIGHT)
    tb = slide.shapes.add_textbox(x + Inches(0.15), y + Inches(0.15), w - Inches(0.3), h - Inches(0.3))
    tf = tb.text_frame; tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        _rich(p, "▪  " + b, size, INK, PRIMARY)


def build(slides: list[dict], out: Path) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    total = len(slides)
    for n, s in enumerate(slides, start=1):
        sl = prs.slides.add_slide(blank)
        t = s["type"]
        if t == "title":
            _box(sl, 0, 0, W, H, fill=DARK)
            _box(sl, 0, Inches(5.2), W, Inches(2.3), fill=PRIMARY)
            _text(sl, Inches(0.7), Inches(0.6), Inches(12), Inches(0.4), s.get("kicker", "GRADIENT LEARNINGS DATA ANALYTICS HACKATHON"), 12, RGBColor(0xC9, 0xD6, 0xE3), bold=True)
            _text(sl, Inches(0.7), Inches(1.3), Inches(11.8), Inches(2.2), s["title"], 38, RGBColor(0xFF, 0xFF, 0xFF), bold=True)
            _text(sl, Inches(0.7), Inches(3.55), Inches(11.5), Inches(1.0), s.get("subtitle", ""), 17, RGBColor(0xDD, 0xE6, 0xEF))
            kp = s.get("kpis", [])
            if kp:
                cw = Inches(12.0 / len(kp))
                for i, (v, l) in enumerate(kp):
                    x = Inches(0.7) + cw * i
                    _text(sl, x, Inches(5.5), cw - Inches(0.2), Inches(0.8), v, 32, RGBColor(0xFF, 0xFF, 0xFF), bold=True)
                    _text(sl, x, Inches(6.3), cw - Inches(0.2), Inches(0.8), l, 12, RGBColor(0xDD, 0xE6, 0xEF))
        elif t == "chart":
            _frame(sl, s.get("kicker", ""), s["title"], n, total)
            if s.get("bullets"):
                _picture(sl, FIG / s["image"], Inches(0.45), Inches(1.6), Inches(8.3), Inches(5.35))
                _bullets(sl, Inches(8.95), Inches(1.65), Inches(3.95), Inches(5.25), s["bullets"], s.get("size", 14))
            else:
                _picture(sl, FIG / s["image"], Inches(0.45), Inches(1.6), Inches(12.4), Inches(5.35))
        elif t == "two":
            _frame(sl, s.get("kicker", ""), s["title"], n, total)
            imgs = s["images"]
            bh = Inches(1.15) if s.get("bullets") else 0
            ih = Inches(5.3) - bh
            _picture(sl, FIG / imgs[0], Inches(0.45), Inches(1.6), Inches(6.1), ih)
            _picture(sl, FIG / imgs[1], Inches(6.75), Inches(1.6), Inches(6.1), ih)
            if s.get("bullets"):
                tb = sl.shapes.add_textbox(Inches(0.5), Inches(1.6) + ih + Inches(0.05), Inches(12.3), bh)
                tf = tb.text_frame; tf.word_wrap = True
                for i, b in enumerate(s["bullets"]):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    _rich(p, "▪  " + b, 14, INK, PRIMARY)
        elif t == "text":
            _frame(sl, s.get("kicker", ""), s["title"], n, total)
            rows = s["rows"]
            top, avail = Inches(1.65), Inches(5.3)
            rh = int(avail / len(rows))
            for i, (head, body) in enumerate(rows):
                y = top + rh * i
                _box(sl, Inches(0.5), y + Inches(0.06), Inches(0.12), rh - Inches(0.16), fill=s.get("colors", [ACCENT, PRIMARY, GOOD, PRIMARY, ACCENT, GOOD, PRIMARY])[i % 7])
                _text(sl, Inches(0.8), y, Inches(3.6), rh - Inches(0.1), head, 16, DARK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
                tb = sl.shapes.add_textbox(Inches(4.5), y, Inches(8.3), rh - Inches(0.1))
                tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
                _rich(tf.paragraphs[0], body, 14, INK, PRIMARY)
        if s.get("notes"):
            sl.notes_slide.notes_text_frame.text = s["notes"]
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print("wrote", out)
    return out
