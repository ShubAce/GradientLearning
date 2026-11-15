"""Render the analysis report: Markdown (+ small block extensions) -> styled HTML -> PDF via headless Edge.

Block extensions (each block starts with a line ':::name' and ends with a line ':::'):
  :::kpis      lines "VALUE | LABEL | sub-label"            -> KPI tile row
  :::callout   markdown text                                 -> highlighted takeaway box
  :::warn      markdown text                                 -> caution box
  :::fig FILE | CAPTION                                      -> (single line) full-width figure with caption
  :::fig2 FILE_A | CAPTION_A || FILE_B | CAPTION_B           -> (single line) two figures side by side
  :::pagebreak                                               -> (single line) page break
  :::cover     key: value lines (title, subtitle, author, date, event) -> cover page
Usage: python src/build_report.py deliverables/report/report.md
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = r"""
@page { size: A4; margin: 15mm 15mm 16mm 15mm; }
@page :first { margin: 0; }
:root { --primary:#1F4E79; --accent:#E4572E; --good:#2E8B57; --warn:#F2A541; --ink:#1d2733; --muted:#5b6b7b; --line:#d9e2ec; --bg:#f4f7fa; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: var(--ink); font-size: 10.4pt; line-height: 1.5; margin: 0; }
h1 { font-size: 20pt; color: var(--primary); margin: 0 0 6pt 0; padding-bottom: 4pt; border-bottom: 2.5pt solid var(--primary); page-break-before: always; break-before: page; }
h1.first { page-break-before: avoid; break-before: avoid; }
h2 { font-size: 13.5pt; color: var(--primary); margin: 16pt 0 4pt 0; break-after: avoid; page-break-after: avoid; }
h3 { font-size: 11.2pt; color: var(--ink); margin: 12pt 0 3pt 0; break-after: avoid; page-break-after: avoid; }
p { margin: 4pt 0 7pt 0; }
ul, ol { margin: 3pt 0 8pt 0; padding-left: 16pt; }
li { margin: 2pt 0; }
strong { color: #0f2a44; }
code { font-family: Consolas, monospace; font-size: 9pt; background: var(--bg); padding: 0 3pt; border-radius: 3pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0; font-size: 9pt; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th { background: var(--primary); color: #fff; text-align: left; padding: 4pt 6pt; font-weight: 600; }
td { border-bottom: 0.6pt solid var(--line); padding: 3.5pt 6pt; vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }
.kpis { display: flex; gap: 7pt; margin: 8pt 0 10pt 0; break-inside: avoid; }
.kpi { flex: 1; background: var(--bg); border-left: 3.5pt solid var(--primary); border-radius: 4pt; padding: 7pt 8pt; }
.kpi.bad { border-left-color: var(--accent); } .kpi.good { border-left-color: var(--good); } .kpi.warn { border-left-color: var(--warn); }
.kpi .v { font-size: 16pt; font-weight: 700; color: var(--primary); line-height: 1.1; }
.kpi.bad .v { color: var(--accent); } .kpi.good .v { color: var(--good); } .kpi.warn .v { color: #b7791f; }
.kpi .l { font-size: 8.6pt; font-weight: 600; margin-top: 2pt; }
.kpi .s { font-size: 7.8pt; color: var(--muted); }
.callout { background: #eaf2fb; border-left: 4pt solid var(--primary); padding: 7pt 10pt; margin: 8pt 0 10pt 0; border-radius: 3pt; break-inside: avoid; }
.callout p { margin: 2pt 0; }
div.warn:not(.kpi) { background: #fff6e8; border-left: 4pt solid var(--warn); padding: 7pt 10pt; margin: 8pt 0 10pt 0; border-radius: 3pt; break-inside: avoid; font-size: 9.4pt; }
figure { margin: 8pt 0 10pt 0; break-inside: avoid; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 118mm; }
figure.half img { max-height: 92mm; }
figcaption { font-size: 8.6pt; color: var(--muted); margin-top: 3pt; text-align: left; }
figcaption b { color: var(--ink); }
.fig2 { display: flex; gap: 8pt; break-inside: avoid; }
.fig2 figure { flex: 1; margin: 6pt 0; }
.pb { page-break-after: always; break-after: page; height: 0; }
.cover { height: 296mm; width: 210mm; display: flex; flex-direction: column; justify-content: space-between; background: linear-gradient(160deg, #0f2a44 0%, #1F4E79 55%, #2b6aa3 100%); color: #fff; margin: 0; padding: 30mm 22mm 22mm 22mm; break-after: page; page-break-after: always; overflow: hidden; }
.cover .event { font-size: 10pt; letter-spacing: 1.5pt; text-transform: uppercase; opacity: .85; }
.cover .title { font-size: 30pt; font-weight: 700; line-height: 1.15; margin-top: 14mm; }
.cover .subtitle { font-size: 13pt; opacity: .92; margin-top: 6mm; max-width: 150mm; }
.cover .meta { font-size: 10.5pt; opacity: .95; border-top: 1pt solid rgba(255,255,255,.4); padding-top: 5mm; }
.cover .band { display: flex; gap: 6mm; margin-top: 12mm; }
.cover .band > div { background: rgba(255,255,255,.12); border-radius: 4pt; padding: 4mm 5mm; flex: 1; }
.cover .band .v { font-size: 18pt; font-weight: 700; }
.cover .band .l { font-size: 8.8pt; opacity: .9; }
.small { font-size: 8.8pt; color: var(--muted); }
.toc a { color: var(--ink); text-decoration: none; }
"""


def _fig_html(spec: str, half: bool = False) -> str:
    file, _, cap = spec.partition("|")
    file, cap = file.strip(), cap.strip()
    src = (FIG / file) if not Path(file).is_absolute() and not file.startswith("..") else Path(file)
    if not src.exists():
        alt = ROOT / file
        src = alt if alt.exists() else src
    if not src.exists():
        print(f"WARNING missing figure: {file}", file=sys.stderr)
    cap_html = markdown.markdown(cap).removeprefix("<p>").removesuffix("</p>")
    cls = ' class="half"' if half else ""
    return f'<figure{cls}><img src="{src.resolve().as_uri()}"><figcaption>{cap_html}</figcaption></figure>'


LIST_ITEM = re.compile(r"^\s*([-*+]|\d+\.)\s+")


def fix_list_spacing(md_text: str) -> str:
    """Python-Markdown needs a blank line before a list that follows a paragraph line; insert it."""
    lines, out = md_text.splitlines(), []
    for ln in lines:
        if out and LIST_ITEM.match(ln) and out[-1].strip() and not LIST_ITEM.match(out[-1]) \
                and not out[-1].startswith((" ", "\t", "|", ":::")):
            out.append("")
        out.append(ln)
    return "\n".join(out)


def preprocess(md_text: str) -> str:
    md_text = fix_list_spacing(md_text)
    out, lines, i = [], md_text.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^:::(\w+)\s*(.*)$", ln)
        if not m:
            out.append(ln); i += 1; continue
        kind, rest = m.group(1), m.group(2)
        if kind == "fig":
            out.append(_fig_html(rest)); i += 1; continue
        if kind == "fig2":
            a, _, b = rest.partition("||")
            out.append(f'<div class="fig2">{_fig_html(a, True)}{_fig_html(b, True)}</div>'); i += 1; continue
        if kind == "pagebreak":
            out.append('<div class="pb"></div>'); i += 1; continue
        # block kinds: collect until ':::'
        body = []
        i += 1
        while i < len(lines) and lines[i].strip() != ":::":
            body.append(lines[i]); i += 1
        i += 1
        if kind == "kpis":
            tiles = []
            for b in body:
                if not b.strip():
                    continue
                parts = [p.strip() for p in b.split("|")]
                tone = ""
                if parts and parts[0].startswith(("bad:", "good:", "warn:")):
                    tone, parts[0] = parts[0].split(":", 1)
                v, l, s = (parts + ["", "", ""])[:3]
                tiles.append(f'<div class="kpi {tone}"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(l)}</div><div class="s">{html.escape(s)}</div></div>')
            out.append(f'<div class="kpis">{"".join(tiles)}</div>')
        elif kind in ("callout", "warn"):
            inner = markdown.markdown("\n".join(body), extensions=["tables"])
            out.append(f'<div class="{kind}">{inner}</div>')
        elif kind == "cover":
            kv = {}
            for b in body:
                k, _, v = b.partition(":")
                kv[k.strip()] = v.strip()
            band = ""
            if kv.get("band"):
                cells = []
                for item in kv["band"].split(";;"):
                    v, _, l = item.partition("|")
                    cells.append(f'<div><div class="v">{html.escape(v.strip())}</div><div class="l">{html.escape(l.strip())}</div></div>')
                band = f'<div class="band">{"".join(cells)}</div>'
            out.append(
                f'<div class="cover"><div><div class="event">{html.escape(kv.get("event",""))}</div>'
                f'<div class="title">{html.escape(kv.get("title",""))}</div><div class="subtitle">{html.escape(kv.get("subtitle",""))}</div>{band}</div>'
                f'<div class="meta">{html.escape(kv.get("author",""))}<br>{html.escape(kv.get("date",""))}</div></div>')
    return "\n".join(out)


def build(md_path: Path, pdf: bool = True) -> Path:
    md_text = Path(md_path).read_text(encoding="utf-8")
    title = re.search(r"^title:\s*(.+)$", md_text, re.M)
    md_text = re.sub(r"^title:[^\n]*\n", "", md_text, count=1, flags=re.M)
    body = markdown.markdown(preprocess(md_text), extensions=["tables", "sane_lists", "attr_list", "md_in_html"])
    body = body.replace("<h1>", '<h1 class="first">', 1)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title.group(1) if title else "Report")}</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    out_html = Path(md_path).with_suffix(".html")
    out_html.write_text(page, encoding="utf-8")
    print("wrote", out_html)
    if pdf:
        out_pdf = Path(md_path).with_suffix(".pdf")
        cmd = [EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
               f"--print-to-pdf={out_pdf}", out_html.resolve().as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        print(r.stdout.strip()[-300:], r.stderr.strip()[-300:])
        print("wrote", out_pdf, out_pdf.exists())
        return out_pdf
    return out_html


if __name__ == "__main__":
    build(Path(sys.argv[1]))
