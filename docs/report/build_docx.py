"""Build the internship report as a Word document in the RWTH format.

    python docs/report/build_docx.py                  # -> docs/report/internship-report.docx
    python docs/report/build_docx.py --pdf            # also render a PDF with LibreOffice

The Markdown file `internship-report.md` stays the single source. This script:

- replaces the `<!-- cover -->` block with a formatted cover page (title, name, student ID,
  company and address, and a box for the trainer's stamp and signature);
- puts a table of contents after the abstract, then a list of figures and a list of tables;
- starts the attachments on a new page;
- applies the required body format (12 pt, 1.5 line spacing) on A4 with 2.5 cm margins;
- numbers the pages in the footer, leaving the cover unnumbered.

It needs pandoc on the PATH, and LibreOffice for --pdf.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "internship-report.md"
OUT = ROOT / "docs" / "report" / "internship-report.docx"

BODY_FONT = "Times New Roman"
BODY_PT = 12          # required by the guidelines
LINE = 360            # 1.5 line spacing (240 = single)
MARGIN = 1418         # 2.5 cm in twips

PAGE_BREAK = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'
TOC_MARK = "VGTOCMARKER"
TOC = '```{=openxml}\n<w:p><w:r><w:t>' + TOC_MARK + '</w:t></w:r></w:p>\n```\n'


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cover_xml(md_cover: str) -> str:
    """The cover page as WordprocessingML, built from the Markdown cover block."""
    title = re.search(r"^# (.+)$", md_cover, re.M).group(1)
    rows = dict(re.findall(r"^\| ([^|]+?) \| ([^|]+?) \|$", md_cover, re.M))
    rows.pop("", None)
    rows.pop("---", None)
    note = re.search(r"^\*(Confidential.+?)\*$", md_cover, re.M | re.S)

    def para(text, size=24, bold=False, align="center", before=0, after=120, italic=False):
        rpr = f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>' + ("<w:b/>" if bold else "") \
            + ("<w:i/>" if italic else "")
        return (f'<w:p><w:pPr><w:spacing w:before="{before}" '
                f'w:after="{after}" w:line="276" w:lineRule="auto"/><w:jc w:val="{align}"/></w:pPr>'
                f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')

    out = [para("RWTH Aachen University · M.Sc. Robotic Systems", 24, before=600),
           para("Internship Report", 28, bold=True, before=1200, after=240),
           para(title, 36, bold=True, after=1200)]
    # Two-column table of the cover details, no borders.
    cells = []
    for k, v in rows.items():
        if k.startswith("Trainer"):
            continue
        cells.append(
            '<w:tr>'
            f'<w:tc><w:tcPr><w:tcW w:w="3600" w:type="dxa"/></w:tcPr>{para(k, 24, bold=True, align="left", after=80)}</w:tc>'
            f'<w:tc><w:tcPr><w:tcW w:w="5470" w:type="dxa"/></w:tcPr>{para(v, 24, align="left", after=80)}</w:tc>'
            '</w:tr>')
    out.append('<w:tbl><w:tblPr><w:tblW w:w="9070" w:type="dxa"/><w:jc w:val="center"/>'
               '<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/>'
               '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>'
               '</w:tblPr><w:tblGrid><w:gridCol w:w="3600"/><w:gridCol w:w="5470"/></w:tblGrid>'
               + "".join(cells) + '</w:tbl>')
    # Stamp and signature box for the trainer.
    box = ('<w:tbl><w:tblPr><w:tblW w:w="9070" w:type="dxa"/><w:jc w:val="center"/>'
           '<w:tblBorders><w:top w:val="single" w:sz="6"/><w:left w:val="single" w:sz="6"/>'
           '<w:bottom w:val="single" w:sz="6"/><w:right w:val="single" w:sz="6"/>'
           '<w:insideV w:val="single" w:sz="6"/></w:tblBorders></w:tblPr>'
           '<w:tblGrid><w:gridCol w:w="4535"/><w:gridCol w:w="4535"/></w:tblGrid><w:tr>'
           '<w:trPr><w:trHeight w:val="1700" w:hRule="atLeast"/></w:trPr>'
           f'<w:tc><w:tcPr><w:tcW w:w="4535" w:type="dxa"/></w:tcPr>{para("Company stamp", 20, align="left", italic=True)}</w:tc>'
           f'<w:tc><w:tcPr><w:tcW w:w="4535" w:type="dxa"/></w:tcPr>{para("Signature of the trainer, date", 20, align="left", italic=True)}</w:tc>'
           '</w:tr></w:tbl>')
    out.append(para("", 12, after=360))
    out.append(box)
    if note:
        out.append(para(" ".join(note.group(1).split()), 18, align="both", before=480, italic=True))
    out.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    return "".join(out)


def headings(md: str) -> list[tuple[int, str]]:
    """Chapter and section headings that go into the contents, in order."""
    out, code = [], False
    body = md[md.index("## 1. Introduction"):]
    for line in body.splitlines():
        if line.startswith("```"):
            code = not code
        m = None if code else re.match(r"^(#{2,3}) (.+)$", line)
        if m:
            out.append((len(m.group(1)) - 1, m.group(2).strip()))
    return out


def toc_xml(entries: list[tuple[int, str]], pages: list[str]) -> str:
    """The contents as a TOC field whose cached result lists every heading with its page.

    Word rebuilds the field on opening (updateFields); other viewers show the cached entries,
    which a second build pass fills with the page numbers measured from the rendered PDF.
    """
    right = 11906 - 2 * MARGIN
    paras = ['<w:p><w:pPr><w:pStyle w:val="TOCHeading"/><w:pageBreakBefore/></w:pPr><w:r><w:t>Contents</w:t></w:r></w:p>']
    for i, ((level, text), page) in enumerate(zip(entries, pages)):
        start = ('<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve">'
                 ' TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
                 '<w:r><w:fldChar w:fldCharType="separate"/></w:r>') if i == 0 else ""
        end = '<w:r><w:fldChar w:fldCharType="end"/></w:r>' if i == len(entries) - 1 else ""
        paras.append(
            f'<w:p><w:pPr><w:pStyle w:val="TOC{level}"/><w:tabs>'
            f'<w:tab w:val="right" w:leader="dot" w:pos="{right}"/></w:tabs></w:pPr>{start}'
            f'<w:r><w:t xml:space="preserve">{esc(text.replace("`", ""))}</w:t></w:r>'
            f'<w:r><w:tab/></w:r><w:r><w:t>{page}</w:t></w:r>{end}</w:p>')
    return "".join(paras)


def captions(md: str) -> list[tuple[str, str, str]]:
    """Every figure and table caption: (kind, label, first sentence), in document order."""
    out = []
    for para in re.split(r"\n\s*\n", md):
        line = " ".join(para.split())  # a caption may wrap over several source lines
        line = re.sub(r"^(?:<!--.*?-->\s*)+", "", line)  # e.g. under an evaluation marker
        fig = re.match(r"^!\[(Figure [A-Z]?\d+)\.\s*(.+?)\]\(", line)
        tab = re.match(r"^\*\*(Table [A-Z]?\d+)\.\*\*\s*(.+)$", line)
        m = fig or tab
        if m:
            text = re.sub(r"[`*]", "", m.group(2))
            first = re.match(r"(.+?[.:])(\s|$)", text)
            text = (first.group(1) if first else text).rstrip(":.")
            if len(text) > 75:  # one line in the list: cut at the first comma or bracket
                cut = re.search(r",| \(", text[30:])
                text = text[: 30 + cut.start()] if cut else text
            text += "."
            out.append(("figure" if fig else "table", m.group(1), text))
    return out


def lists_xml(caps: list[tuple[str, str, str]], pages: list[str]) -> str:
    """List of figures and list of tables on one new page, then a break to the body."""
    right = 11906 - 2 * MARGIN
    paras = []
    for kind, title in (("figure", "List of figures"), ("table", "List of tables")):
        brk = "<w:pageBreakBefore/>" if kind == "figure" else '<w:spacing w:before="480"/>'
        paras.append(f'<w:p><w:pPr><w:pStyle w:val="TOCHeading"/>{brk}</w:pPr>'
                     f'<w:r><w:t>{title}</w:t></w:r></w:p>')
        for (k, label, text), page in zip(caps, pages, strict=True):
            if k != kind:
                continue
            paras.append(
                f'<w:p><w:pPr><w:pStyle w:val="TOC2"/><w:tabs>'
                f'<w:tab w:val="left" w:pos="1080"/>'
                f'<w:tab w:val="right" w:leader="dot" w:pos="{right}"/></w:tabs>'
                f'<w:ind w:left="1080" w:hanging="1080"/></w:pPr>'
                f'<w:r><w:t xml:space="preserve">{esc(label)}</w:t></w:r><w:r><w:tab/></w:r>'
                f'<w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
                f'<w:r><w:tab/></w:r><w:r><w:t>{page}</w:t></w:r></w:p>')
    paras.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    return "".join(paras)


def pdf_text(pdf: Path) -> list[str]:
    """The rendered PDF's text, one string per page."""
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(
        ["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
    return [subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), str(pdf), "-"],
                           capture_output=True, text=True).stdout for i in range(1, n + 1)]


def pdf_pages(text: list[str], keys: list[str]) -> list[str]:
    """Page on which each key (a heading or caption start) first begins a line, in order."""
    n = len(text)
    pages, cur = [], 1
    for h in keys:
        key = re.sub(r"\s+", " ", h.replace("`", ""))[:28]
        found = "?"
        for i in range(cur, n + 1):
            lines = [re.sub(r"\s+", " ", ln.strip()) for ln in text[i - 1].splitlines()]
            # Skip the contents itself: its entries carry dot leaders.
            if any(ln.startswith(key) and "...." not in ln for ln in lines):
                found, cur = str(i), i
                break
        pages.append(found)
    return pages


def prepare_markdown(text: str) -> tuple[str, str]:
    m = re.search(r"<!-- cover -->(.*?)<!-- /cover -->", text, re.S)
    cover = m.group(1)
    text = text[: m.start()] + text[m.end():]
    # Contents after the abstract, i.e. before the first numbered chapter.
    text = text.replace("## 1. Introduction", TOC + "\n## 1. Introduction", 1)
    # The acknowledgements, like the abstract, get a page of their own.
    if "## Acknowledgements" in text:
        text = text.replace("## Abstract", PAGE_BREAK + "\n## Abstract", 1)
    text = text.replace("<!-- attachments -->", PAGE_BREAK)
    return text, cover


def reference_docx(tmp: Path) -> Path:
    """Pandoc's default reference document, restyled to the RWTH format."""
    ref = tmp / "ref.docx"
    subprocess.run(["pandoc", "-o", str(ref), "--print-default-data-file", "reference.docx"],
                   check=True)
    d = tmp / "ref"
    with zipfile.ZipFile(ref) as z:
        z.extractall(d)
    p = d / "word" / "styles.xml"
    s = p.read_text(encoding="utf-8")
    sz = BODY_PT * 2
    s = re.sub(r"<w:rPrDefault>.*?</w:rPrDefault>",
               f'<w:rPrDefault><w:rPr><w:rFonts w:ascii="{BODY_FONT}" w:hAnsi="{BODY_FONT}" '
               f'w:eastAsia="{BODY_FONT}" w:cs="{BODY_FONT}"/><w:sz w:val="{sz}"/>'
               f'<w:szCs w:val="{sz}"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault>',
               s, count=1, flags=re.S)
    s = re.sub(r"<w:pPrDefault>.*?</w:pPrDefault>",
               f'<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="{LINE}" '
               f'w:lineRule="auto"/></w:pPr></w:pPrDefault>', s, count=1, flags=re.S)

    def restyle(style_id: str, ppr: str = "", rpr: str = "") -> None:
        nonlocal s
        m = re.search(rf'<w:style [^>]*w:styleId="{style_id}".*?</w:style>', s, re.S)
        if not m:
            return
        block = m.group(0)
        block = re.sub(r"<w:pPr>.*?</w:pPr>|<w:pPr/>", "", block, flags=re.S)
        block = re.sub(r"<w:rPr>.*?</w:rPr>|<w:rPr/>", "", block, flags=re.S)
        block = block.replace("</w:style>", f"<w:pPr>{ppr}</w:pPr><w:rPr>{rpr}</w:rPr></w:style>")
        s = s[: m.start()] + block + s[m.end():]

    heading = f'<w:rFonts w:ascii="{BODY_FONT}" w:hAnsi="{BODY_FONT}"/><w:b/><w:color w:val="000000"/>'
    # Contents and lists headings: the chapter-heading look, but outside the outline.
    restyle("TOCHeading", '<w:keepNext/><w:spacing w:before="0" w:after="200"/>',
            heading + '<w:sz w:val="30"/><w:szCs w:val="30"/>')
    restyle("Heading1", '<w:keepNext/><w:spacing w:before="360" w:after="120" w:line="300" w:lineRule="auto"/><w:outlineLvl w:val="0"/>',
            heading + '<w:sz w:val="30"/><w:szCs w:val="30"/>')
    restyle("Heading2", '<w:keepNext/><w:spacing w:before="240" w:after="80" w:line="300" w:lineRule="auto"/><w:outlineLvl w:val="1"/>',
            heading + '<w:sz w:val="26"/><w:szCs w:val="26"/>')
    restyle("Heading3", '<w:keepNext/><w:spacing w:before="200" w:after="60" w:line="300" w:lineRule="auto"/><w:outlineLvl w:val="2"/>',
            heading + '<w:sz w:val="24"/><w:szCs w:val="24"/>')
    for sid in ("BodyText", "FirstParagraph"):
        restyle(sid, f'<w:spacing w:before="0" w:after="120" w:line="{LINE}" w:lineRule="auto"/><w:jc w:val="both"/>')
    # List items outside tables: body size and spacing, a little tighter between items.
    s = s.replace("</w:styles>",
                  '<w:style w:type="paragraph" w:customStyle="1" w:styleId="ListItem">'
                  '<w:name w:val="List Item"/><w:basedOn w:val="BodyText"/>'
                  f'<w:pPr><w:spacing w:before="0" w:after="40" w:line="{LINE}" w:lineRule="auto"/>'
                  '<w:jc w:val="left"/></w:pPr></w:style></w:styles>')
    for lvl, ind in ((1, 0), (2, 360)):
        s = s.replace("</w:styles>",
                      f'<w:style w:type="paragraph" w:styleId="TOC{lvl}"><w:name w:val="toc {lvl}"/>'
                      '<w:basedOn w:val="Normal"/><w:uiPriority w:val="39"/>'
                      f'<w:pPr><w:spacing w:before="{20 if lvl == 1 else 0}" w:after="0" w:line="240" '
                      f'w:lineRule="auto"/><w:ind w:left="{ind}"/></w:pPr>'
                      f'<w:rPr>{"<w:b/>" if lvl == 1 else ""}</w:rPr></w:style></w:styles>')
    # Tables, captions and code: single-spaced and smaller, as is usual.
    restyle("Compact", '<w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/>',
            '<w:sz w:val="20"/><w:szCs w:val="20"/>')
    for sid in ("ImageCaption", "TableCaption", "Caption"):
        restyle(sid, '<w:spacing w:before="60" w:after="240" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>',
                '<w:i/><w:sz w:val="20"/><w:szCs w:val="20"/>')
    restyle("CaptionedFigure", '<w:keepNext/><w:spacing w:before="120" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>')
    restyle("SourceCode", '<w:spacing w:before="60" w:after="60" w:line="240" w:lineRule="auto"/>',
            '<w:sz w:val="18"/><w:szCs w:val="18"/>')
    # Table borders on pandoc's "Table" style.
    s = re.sub(r'(<w:style [^>]*w:styleId="Table".*?<w:tblPr>.*?<w:tblInd [^>]*/>)',
               r'\1<w:tblBorders><w:top w:val="single" w:sz="4" w:color="808080"/>'
               r'<w:bottom w:val="single" w:sz="4" w:color="808080"/>'
               r'<w:insideH w:val="single" w:sz="2" w:color="BFBFBF"/></w:tblBorders>',
               s, count=1, flags=re.S)
    p.write_text(s, encoding="utf-8")
    out = tmp / "ref-rwth.docx"
    shutil.make_archive(str(out.with_suffix("")), "zip", d)
    out.with_suffix(".zip").rename(out)
    return out


FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
          '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
          '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
          '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r>'
          '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')


def postprocess(docx: Path, cover: str, toc: str) -> None:
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(docx) as z:
        z.extractall(tmp)
    doc = tmp / "word" / "document.xml"
    s = doc.read_text(encoding="utf-8")
    s = s.replace("<w:body>", "<w:body>" + cover_xml(cover), 1)
    # A table caption is a bold "Table N." paragraph: keep it on the page of its table.
    s = re.sub(r'<w:pStyle w:val="(BodyText|FirstParagraph)" ?/>(</w:pPr><w:r><w:rPr><w:bCs ?/>'
               r'<w:b ?/></w:rPr><w:t[^>]*>Table [A-Z]?\d+\.</w:t>)',
               r'<w:pStyle w:val="\1"/><w:keepNext/>\2', s)
    # References start a new page. A break on the heading itself, not a separate break
    # paragraph, which would leave a blank page whenever the body ends at a page foot.
    s = re.sub(r'(<w:pStyle w:val="Heading1" ?/>)(</w:pPr><w:r><w:t[^>]*>References</w:t>)',
               r"\1<w:pageBreakBefore/>\2", s, count=1)
    s = re.sub(r"<w:p>(?:(?!<w:p>).)*?" + TOC_MARK + r".*?</w:p>", lambda _: toc, s, count=1, flags=re.S)
    # Pandoc gives tight lists and table cells the same "Compact" style. Tables keep it
    # (small, single-spaced); list items outside tables get the body format.
    parts = re.split(r"(<w:tbl>.*?</w:tbl>)", s, flags=re.S)
    s = "".join(part if part.startswith("<w:tbl>") else
                re.sub(r'<w:pStyle w:val="Compact" ?/>', '<w:pStyle w:val="ListItem"/>', part)
                for part in parts)
    # Captioned images: scale to the text width, keeping the aspect ratio.
    width = 11906 - 2 * MARGIN  # A4 width minus margins, twips
    max_emu = int(width * 635 * 0.92)

    def fit(m: re.Match) -> str:
        cx, cy = int(m.group(1)), int(m.group(2))
        if cx <= max_emu:
            return m.group(0)
        return f'cx="{max_emu}" cy="{int(cy * max_emu / cx)}"'
    s = re.sub(r'cx="(\d+)" cy="(\d+)"', fit, s)
    sect = (f'<w:sectPr><w:footerReference w:type="default" r:id="rIdFooterPage"/>'
            f'<w:pgSz w:w="11906" w:h="16838"/>'
            f'<w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" w:bottom="{MARGIN}" w:left="{MARGIN}" '
            f'w:header="709" w:footer="709" w:gutter="0"/><w:titlePg/></w:sectPr>')
    s = re.sub(r"<w:sectPr\b[^>]*/>|<w:sectPr\b.*?</w:sectPr>", lambda _: sect, s, flags=re.S)
    assert "rIdFooterPage" in s, "no section properties found to attach the page numbers to"
    if 'xmlns:r=' not in s[:2000]:
        s = s.replace("<w:document ", '<w:document xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" ', 1)
    doc.write_text(s, encoding="utf-8")
    (tmp / "word" / "footer-page.xml").write_text(FOOTER, encoding="utf-8")
    rels = tmp / "word" / "_rels" / "document.xml.rels"
    r = rels.read_text(encoding="utf-8")
    r = r.replace("</Relationships>",
                  '<Relationship Id="rIdFooterPage" Type="http://schemas.openxmlformats.org/'
                  'officeDocument/2006/relationships/footer" Target="footer-page.xml"/></Relationships>')
    rels.write_text(r, encoding="utf-8")
    ct = tmp / "[Content_Types].xml"
    c = ct.read_text(encoding="utf-8")
    if 'Extension="png"' not in c:
        c = c.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
    c = c.replace("</Types>", '<Override PartName="/word/footer-page.xml" ContentType="application/'
                  'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
    ct.write_text(c, encoding="utf-8")
    settings = tmp / "word" / "settings.xml"
    st = settings.read_text(encoding="utf-8")
    if "updateFields" not in st:  # Word offers to build the contents on opening
        later = re.search(r"<w:(?:hdrShapeDefaults|footnotePr|endnotePr|compat|docVars|rsids)\b|<m:mathPr\b", st)
        st = st[:later.start()] + '<w:updateFields w:val="true"/>' + st[later.start():]
        settings.write_text(st, encoding="utf-8")
    docx.unlink()
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(tmp).as_posix())
    shutil.rmtree(tmp)


def build(md: str, cover: str, toc: str, out: Path) -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        src = tmp / "report.md"
        src.write_text(md, encoding="utf-8")
        ref = reference_docx(tmp)
        subprocess.run(
            ["pandoc", str(src), "-f", "markdown+pipe_tables+tex_math_dollars+raw_attribute-auto_identifiers",
             "--shift-heading-level-by=-1", "-t", "docx", "--reference-doc", str(ref),
             "--resource-path", str(ROOT), "-o", str(out)],
            check=True)
    postprocess(out, cover, toc)


def render_pdf(docx: Path) -> Path:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(docx.parent),
                    str(docx)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return docx.with_suffix(".pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", action="store_true",
                    help="also render a PDF with LibreOffice, and fill the contents' page numbers")
    ap.add_argument("-o", "--out", type=Path, default=OUT)
    args = ap.parse_args()
    raw = SOURCE.read_text(encoding="utf-8")
    md, cover = prepare_markdown(raw)
    entries = headings(raw)
    caps = captions(raw)
    # Pass 1 lays out the contents and lists with placeholder numbers of the same width,
    # so the page on which each heading or caption falls does not change between passes.
    build(md, cover, toc_xml(entries, ["00"] * len(entries))
          + lists_xml(caps, ["00"] * len(caps)), args.out)
    if args.pdf:
        text = pdf_text(render_pdf(args.out))
        pages = pdf_pages(text, [h for _, h in entries])
        cap_pages = pdf_pages(text, [f"{label}. {cap}".rstrip(".") for _, label, cap in caps])
        build(md, cover, toc_xml(entries, pages) + lists_xml(caps, cap_pages), args.out)
        render_pdf(args.out)
        print("wrote", args.out.with_suffix(".pdf"))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
